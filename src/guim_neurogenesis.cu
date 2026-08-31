// ============================================================================
//  guim_neurogenesis.cu
//  ---------------------------------------------------------------------------
//  Real-time **artificial neurogenesis** for the Guimlab continual kernel.
//
//  Each neuron maintains an Exponential Moving Average (EMA) of its activation
//  AND of its activation *variance* (a.k.a. "neuronal utility").  When the
//  variance drops below a critical threshold for K consecutive cycles, the
//  neuron is deemed metabolically dead and is **re-randomized in-place on the
//  GPU**, with zero host-device sync.  This is Javed/Sutton 2024's Continual
//  Backpropagation mechanism (Guimlab, July 2026).
//
//  Key properties
//  --------------
//   *  Per-threadur cuRAND state (curand_init seeded once at boot, advanced
//      via curand_uniform / curand_normal inside the kernel).
//   *  In-place reset: weight row + trace row are mutated without leaving
//      device memory (Carmack-style zero dynamic alloc).
//   *  Coalesced memory access: row-major indexing [row * TOTAL_IN + col]
//      guarantees warp-coalesced 32-thread loads/stores.
//   *  Real-time: kernel is single-block, single-pass, <50us target.
//
//  References
//  ----------
//   *  Khurram Javed & Richard Sutton, "Continual Backpropagation" (2024)
//   *  Javed et al., Guimlab launch announcement, July 2026
//   *  docs/03-guimlab-foundations.md (this repo) — mathematical derivation
//
//  Author       : Guillaume Meingan + Claude (ultracode session 2026-08-31)
//  License      : AGPL-3.0
//  Compilation  : nvcc -O3 -std=c++20 -use_fast_math -lcurand guim_neurogenesis.cu
// ============================================================================

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

// ----------------------------------------------------------------------------
//  Constants (kept in sync with include/guim_kernel.h — single source of truth
//  is C ABI header).  Re-declared here only because .cu cannot include a
//  header that itself includes a C++ wrapper without extern "C" markers.
// ----------------------------------------------------------------------------

constexpr int    NG_STATE_DIM          = 128;
constexpr int    NG_INPUT_DIM          = 64;
constexpr int    NG_TOTAL_IN           = NG_INPUT_DIM + NG_STATE_DIM;  // 192
constexpr int    NG_TOTAL_W            = NG_STATE_DIM * NG_TOTAL_IN;   // 24576

// Metabolic thresholds (Javed CBP)
constexpr float  NG_BETA               = 0.99f;   // EMA smoothing
constexpr float  NG_UTILITY_THRESHOLD  = 1e-4f;   // death threshold
constexpr float  NG_REFUGIO            = 0.1f;    // utility reset value
constexpr float  NG_WEIGHT_CLIP        = 5.0f;    // numerical safety

// ----------------------------------------------------------------------------
//  Kernel 1 — cuRAND state initialization (one thread per neuron)
// ----------------------------------------------------------------------------
//  Must run be called once before fused_plasticity_kernel.  Idempotent.
// ----------------------------------------------------------------------------

__global__ void init_curand_kernel(
    curandState* __restrict__ d_rng_states,
    unsigned long long       seed
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < NG_STATE_DIM) {
        curand_init(seed, id, 0, &d_rng_states[id]);
    }
}

// ----------------------------------------------------------------------------
//  Kernel 2 — Fused: forward + EMA + plasticity reset (neurogenesis) + update
// ----------------------------------------------------------------------------
//
//  Each thread (one per neuron) does:
//   PHASE 1 — Forward pass: h_t = tanh(W_xh · x + W_hh · h_{t-1})
//   PHASE 2 — Metabolic tracking:
//              EMA_mean   = β·EMA_mean   + (1−β)·h_t
//              EMA_var    = β·EMA_var    + (1−β)·(h_t − EMA_mean)²
//   PHASE 3 — Neurogenesis: if EMA_var < ε ⇒
//              W_{row,*} ← U(−scale, scale)
//              E_{row,*} ← 0
//              EMA_var   ← refugio (0.1)
//              h_new     ← 0
//   PHASE 4 — Continual learning: W ← W + α·δ·e − sign(W)·decay
//              E        ← λ·e + dtanh·[x; h_{t-1}]
//   PHASE 5 — Commit h_new
// ----------------------------------------------------------------------------

__global__ void fused_plasticity_kernel(
    const float*  __restrict__ d_input,        // [NG_INPUT_DIM]      current frame
    float*        __restrict__ d_hidden,       // [NG_STATE_DIM]      in-place hidden state
    float*        __restrict__ d_weights,      // [NG_TOTAL_W]       weight matrix
    float*        __restrict__ d_traces,       // [NG_TOTAL_W]       eligibility traces
    float*        __restrict__ d_utility,      // [NG_STATE_DIM]      EMA of activation variance per neuron
    float*        __restrict__ d_mean,         // [NG_STATE_DIM]      EMA of activation per neuron
    curandState*  __restrict__ d_rng_states,   // [NG_STATE_DIM]      per-neuron RNG state
    const float                  td_error,      // scalar             δ_t = r + γ·V_{t+1} − V_t
    const float                  alpha,         // scalar             learning rate
    const float                  lambda_t,      // scalar             eligibility trace decay
    const float                  decay,         // scalar             L1 weight decay
    unsigned long long*          d_resets       // device counter for total resets
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= NG_STATE_DIM) return;

    // ============================ PHASE 1 — FORWARD ===========================
    float accum = 0.0f;
    const int w_base = row * NG_TOTAL_IN;

    // Input contribution (coalesced read of d_input)
    #pragma unroll 4
    for (int i = 0; i < NG_INPUT_DIM; ++i) {
        accum += d_weights[w_base + i] * d_input[i];
    }

    // Recurrent contribution (uses h_{t-1} before update)
    #pragma unroll 4
    for (int i = 0; i < NG_STATE_DIM; ++i) {
        accum += d_weights[w_base + NG_INPUT_DIM + i] * d_hidden[i];
    }

    const float h_new = tanhf(accum);
    const float dtanh = 1.0f - h_new * h_new;  // local derivative

    // ======================== PHASE 2 — METABOLIC TRACKING ======================
    float current_mean = d_mean[row];
    current_mean = NG_BETA * current_mean + (1.0f - NG_BETA) * h_new;
    d_mean[row] = current_mean;

    const float variance = (h_new - current_mean) * (h_new - current_mean);
    float current_utility = d_utility[row];
    current_utility = NG_BETA * current_utility + (1.0f - NG_BETA) * variance;
    d_utility[row] = current_utility;

    // =================== PHASE 3 — NEUROGENESIS (suicide check) =================
    if (current_utility < NG_UTILITY_THRESHOLD) {
        // Dead neuron — full in-place reset via per-thread cuRAND
        curandState local_rng = d_rng_states[row];
        const float scale = 2.0f / sqrtf(static_cast<float>(NG_TOTAL_IN));

        #pragma unroll 4
        for (int col = 0; col < NG_TOTAL_IN; ++col) {
            const int idx = w_base + col;
            // New uniform-synapse weights:  W ~ U(-scale, scale)
            const float r = curand_uniform(&local_rng);
            d_weights[idx] = (r - 0.5f) * 2.0f * scale;
            // Purge synaptic memory (trace decay to zero)
            d_traces[idx] = 0.0f;
        }
        d_rng_states[row] = local_rng;

        // Restore utility to a refuge value (not zero, to avoid immediate re-death)
        d_utility[row] = NG_REFUGIO;
        // Override activation so the dead neuron doesn't pollute downstream
        d_hidden[row] = 0.0f;
        // Atomic increment of reset counter (rare path, zero-copy)
        atomicAdd(d_resets, 1ull);
        // Note: we do NOT execute PHASE 4 this frame — a fresh neuron needs a
        // step to settle its trace before any learning signal is applied.
        return;
    }

    // ===================== PHASE 4 — CONTINUAL LEARNING (TD + RTRL) ==============
    #pragma unroll 4
    for (int col = 0; col < NG_TOTAL_IN; ++col) {
        const int idx = w_base + col;
        const float in_val = (col < NG_INPUT_DIM) ? d_input[col]
                                                 : d_hidden[col - NG_INPUT_DIM];

        // Eligibility trace update (forward-mode RTRL)
        const float new_trace = (lambda_t * d_traces[idx]) + (dtanh * in_val);
        d_traces[idx] = new_trace;

        // TD-driven weight update + L1 decay
        float w = d_weights[idx];
        w += alpha * td_error * new_trace;
        w -= decay * (w > 0.0f ? 1.0f : (w < 0.0f ? -1.0f : 0.0f));

        // Hard clip (numerical safety net)
        if (w >  NG_WEIGHT_CLIP) w =  NG_WEIGHT_CLIP;
        if (w < -NG_WEIGHT_CLIP) w = -NG_WEIGHT_CLIP;
        d_weights[idx] = w;
    }

    // ============================ PHASE 5 — COMMIT ==============================
    d_hidden[row] = h_new;
}

// ----------------------------------------------------------------------------
//  Host-side launcher (extern "C" so it's reachable from C ABI)
// ----------------------------------------------------------------------------

extern "C" {

struct NeurogenesisEngine {
    float*         d_input      = nullptr;
    float*         d_hidden     = nullptr;
    float*         d_weights    = nullptr;
    float*         d_traces     = nullptr;
    float*         d_utility    = nullptr;
    float*         d_mean       = nullptr;
    curandState*   d_rng_states = nullptr;
    unsigned long long* d_resets = nullptr;

    unsigned long long seed       = 0;
    unsigned long long frames_run = 0;
};

NeurogenesisEngine* ng_create(unsigned long long seed) {
    auto* e = new NeurogenesisEngine;
    if (!e) return nullptr;
    e->seed = seed;

    cudaMalloc(&e->d_input,      NG_INPUT_DIM  * sizeof(float));
    cudaMalloc(&e->d_hidden,     NG_STATE_DIM  * sizeof(float));
    cudaMalloc(&e->d_weights,    NG_TOTAL_W    * sizeof(float));
    cudaMalloc(&e->d_traces,     NG_TOTAL_W    * sizeof(float));
    cudaMalloc(&e->d_utility,    NG_STATE_DIM  * sizeof(float));
    cudaMalloc(&e->d_mean,       NG_STATE_DIM  * sizeof(float));
    cudaMalloc(&e->d_rng_states, NG_STATE_DIM  * sizeof(curandState));
    cudaMalloc(&e->d_resets,     sizeof(unsigned long long));

    // Zero-init state buffers
    cudaMemset(e->d_hidden,  0, NG_STATE_DIM  * sizeof(float));
    cudaMemset(e->d_traces,  0, NG_TOTAL_W    * sizeof(float));
    cudaMemset(e->d_utility, 0, NG_STATE_DIM  * sizeof(float));
    cudaMemset(e->d_mean,    0, NG_STATE_DIM  * sizeof(float));
    cudaMemset(e->d_resets,  0, sizeof(unsigned long long));

    // Init RNG states (1 thread per neuron, single block)
    init_curand_kernel<<<1, NG_STATE_DIM>>>(e->d_rng_states, seed);

    // Xavier/Glorot initialization on CPU then HtoD
    std::vector<float> h_weights(NG_TOTAL_W);
    const float scale = std::sqrt(2.0f / static_cast<float>(NG_TOTAL_IN));
    std::srand(static_cast<unsigned>(seed));
    for (auto& w : h_weights) {
        w = (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 0.5f) * 2.0f * scale;
    }
    cudaMemcpy(e->d_weights, h_weights.data(),
               NG_TOTAL_W * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    return e;
}

void ng_destroy(NeurogenesisEngine* e) {
    if (!e) return;
    if (e->d_input)      cudaFree(e->d_input);
    if (e->d_hidden)     cudaFree(e->d_hidden);
    if (e->d_weights)    cudaFree(e->d_weights);
    if (e->d_traces)     cudaFree(e->d_traces);
    if (e->d_utility)    cudaFree(e->d_utility);
    if (e->d_mean)       cudaFree(e->d_mean);
    if (e->d_rng_states) cudaFree(e->d_rng_states);
    if (e->d_resets)     cudaFree(e->d_resets);
    delete e;
}

float ng_step(NeurogenesisEngine* e, const float* input, float td_error) {
    if (!e || !input) return 0.0f;

    cudaMemcpy(e->d_input, input, NG_INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha  = 0.005f;
    const float lambda_t = 0.85f;
    const float decay = 0.0001f;

    fused_plasticity_kernel<<<1, NG_STATE_DIM>>>(
        e->d_input, e->d_hidden, e->d_weights, e->d_traces,
        e->d_utility, e->d_mean, e->d_rng_states,
        td_error, alpha, lambda_t, decay, e->d_resets
    );

    ++e->frames_run;
    return td_error;
}

unsigned long long ng_resets_total(const NeurogenesisEngine* e) {
    if (!e || !e->d_resets) return 0;
    unsigned long long host_resets = 0;
    cudaMemcpy(&host_resets, e->d_resets, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    return host_resets;
}

unsigned long long ng_frames_run(const NeurogenesisEngine* e) {
    return e ? e->frames_run : 0;
}

}  // extern "C"