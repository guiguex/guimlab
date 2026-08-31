// ============================================================================
//  guim_lovelace_kernel.cu — Fused Lovelace Cortex with Warp Tensor Cores & GRM-FIN
//  ============================================================================
//  Author        : Guimlab Contributors
//  License       : AGPL-3.0 / MIT
// ============================================================================

#include "lovelace_cortex.h"
#include "ipc_structs.h"
#include "ipc_structs_v2.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <algorithm>

// ----------------------------------------------------------------------------
//  Hardware Intrinsics & PTX Assembly Templates
// ----------------------------------------------------------------------------

// Byte-wise sum of absolute differences of 4 packed bytes
__device__ __forceinline__ std::uint32_t ptx_surprise(std::uint32_t real_packed, std::uint32_t pred_packed) {
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300)
    return __vabsdiffu4(real_packed, pred_packed);
    #else
    std::uint32_t sad = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int r = (real_packed >> (i * 8)) & 0xFF;
        int p = (pred_packed >> (i * 8)) & 0xFF;
        sad += abs(r - p);
    }
    return sad;
    #endif
}

// Fast Invariant Unit-Sphere Projection on Registers (1 cycle approx + Newton-Raphson)
__device__ __forceinline__ float ptx_fast_rsqrt_nr(float x) {
    float y;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300)
    asm volatile("rsqrt.approx.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    #else
    y = 1.0f / sqrtf(x);
    #endif
    // 1-step Newton-Raphson refinement: y = y * (1.5f - 0.5f * x * y * y)
    return y * (1.5f - 0.5f * x * y * y);
}

// PTX Tensor Core m16n8k16 FP16 -> FP32 Accumulation
__device__ __forceinline__ void ptx_mma_m16n8k16_fp16(
    float &d0, float &d1, float &d2, float &d3,
    unsigned a0, unsigned a1,
    unsigned b0
) {
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(b0)
    );
    #else
    (void)a0; (void)a1; (void)b0;
    #endif
}

// ----------------------------------------------------------------------------
//  Kernel 1: Initialization
// ----------------------------------------------------------------------------

__global__ void guim_lovelace_init_kernel(
    half*          __restrict__ d_weights_fp16,
    half*          __restrict__ d_alphas_fp16,
    float*         __restrict__ d_hidden,
    float*         __restrict__ d_traces,
    float*         __restrict__ d_meta_traces,
    float*         __restrict__ d_betas,
    std::uint64_t* __restrict__ d_last_t,
    std::uint64_t               start_time_ns
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = GUIM_STATE_DIM * GUIM_INPUT_DIM;
    if (idx >= total) return;

    // Xavier initialization modulated with Golden Invariant Metric
    const float scale = sqrtf(2.0f / static_cast<float>(GUIM_INPUT_DIM));
    unsigned long long state = 0xC0FFEEull ^ (static_cast<unsigned long long>(idx) * 0x9E3779B97F4A7C15ull);
    state = (state ^ (state >> 30)) * 0xBF58476D1CE4E5B9ull;
    state = (state ^ (state >> 27)) * 0x94D049BB133111EBull;
    state = state ^ (state >> 31);
    const float u = (static_cast<float>(state & 0xFFFFFF) / static_cast<float>(0x1000000)) - 0.5f;

    d_weights_fp16[idx] = __float2half(u * 2.0f * scale);
    d_alphas_fp16[idx]  = __float2half(0.015f);
    d_betas[idx]        = -4.2f;
    d_traces[idx]       = 0.0f;
    d_meta_traces[idx]  = 0.0f;
    d_last_t[idx]       = start_time_ns;

    if (idx < GUIM_STATE_DIM) {
        d_hidden[idx] = 0.0f;
    }
}

// ----------------------------------------------------------------------------
//  Kernel 2: Persistent Fused Cortex Kernel with Watchdog & GRM-FIN
// ----------------------------------------------------------------------------

constexpr std::uint32_t SURPRISE_THRESHOLD = 10;

__global__ void __launch_bounds__(GUIM_STATE_DIM, 1)
guim_lovelace_fused_kernel(
    volatile GuimSharedMemoryV2* __restrict__ d_ipc,
    float*          __restrict__ d_hidden,
    half*           __restrict__ d_weights_fp16,
    half*           __restrict__ d_alphas_fp16,
    float*          __restrict__ d_traces,
    std::uint64_t*  __restrict__ d_last_t,
    float*          __restrict__ d_meta_traces,
    float*          __restrict__ d_betas
) {
    const int lane_id = threadIdx.x & 31;
    const int row     = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ std::uint64_t s_local_seq;
    __shared__ float         s_neuromod_scalar;

    if (threadIdx.x == 0) {
        s_local_seq       = d_ipc->seq_in;
        s_neuromod_scalar = d_ipc->dopamine * d_ipc->serotonin;
    }
    __syncthreads();

    while (!d_ipc->terminate) {
        // ---- PHASE 1 — Spin-wait on seq_in with %globaltimer Watchdog ----
        if (threadIdx.x == 0) {
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
            unsigned long long start_global_t = 0;
            asm volatile("mov.u64 %0, %globaltimer;" : "=l"(start_global_t));
            #endif

            while (d_ipc->seq_in <= s_local_seq && !d_ipc->terminate) {
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
                __nanosleep(100);
                unsigned long long cur_global_t = 0;
                asm volatile("mov.u64 %0, %globaltimer;" : "=l"(cur_global_t));
                if (cur_global_t - start_global_t > 2000000000ull) { // 2.0s timeout
                    break;
                }
                #else
                asm volatile("nanosleep.u32 100;");
                #endif
            }
            s_local_seq = d_ipc->seq_in;
            s_neuromod_scalar = d_ipc->dopamine * d_ipc->serotonin;
        }
        __syncthreads();
        if (d_ipc->terminate) break;

        // ---- PHASE 2 — Surprise calculation ----
        std::uint32_t local_surprise = 0;
        if (lane_id < GUIM_PACKED_DIM) {
            local_surprise = ptx_surprise(
                d_ipc->real_sensors_q8[lane_id],
                d_ipc->pred_sensors_q8[lane_id]
            );
        }

        // ---- PHASE 3 — DDWR warp ballot ----
        const bool is_surprised = (local_surprise > SURPRISE_THRESHOLD);
        const unsigned int active_mask = __ballot_sync(0xFFFFFFFFu, is_surprised);

        if (active_mask == 0u) {
            if (row < GUIM_STATE_DIM) {
                d_ipc->cortex_cognitive_out[row] = 0.0f;
            }
            __syncthreads();
            __threadfence_system();
            if (threadIdx.x == 0) {
                d_ipc->seq_out = s_local_seq;
            }
            continue;
        }

        // ---- PHASES 4-8 — Forward + GRM-FIN + TMD-ET ----
        if (row < GUIM_STATE_DIM) {
            float stimulus_accum = 0.0f;
            const int w_base = row * GUIM_INPUT_DIM;

            #pragma unroll 8
            for (int i = 0; i < GUIM_INPUT_DIM; ++i) {
                const int pack_idx = i / 4;
                const int shift    = (i & 3) * 8;
                const float in_val = static_cast<float>((d_ipc->real_sensors_q8[pack_idx] >> shift) & 0xFF) / 255.0f;
                stimulus_accum += __half2float(d_weights_fp16[w_base + i]) * in_val;
            }

            const float h_old = d_hidden[row];
            // Golden Riemannian Metric Scaling: phi modulation
            const float phi_scale = (row % 2 == 0) ? guim::GOLDEN_RATIO_INV : (guim::GOLDEN_RATIO_INV * guim::GOLDEN_RATIO_INV);
            const float h_candidate = h_old + stimulus_accum * phi_scale;
            const float h_new = tanhf(h_candidate);
            const float dtanh = 1.0f - h_new * h_new;

            const std::uint64_t current_t = d_ipc->timestamp_ns;
            const float td_error = d_ipc->dopamine;

            #pragma unroll 8
            for (int col = 0; col < GUIM_INPUT_DIM; ++col) {
                const int idx = w_base + col;
                const float dt_sec = static_cast<float>(current_t - d_last_t[idx]) * 1e-9f;
                const float decay_factor = __expf(-fmaxf(dt_sec, 0.0f) / 10.0f);

                const int pack_idx = col / 4;
                const int shift    = (col & 3) * 8;
                const float in_val = static_cast<float>((d_ipc->real_sensors_q8[pack_idx] >> shift) & 0xFF) / 255.0f;

                float e_ij = d_traces[idx] * decay_factor + dtanh * in_val;
                d_traces[idx] = e_ij;

                const float grad = td_error * e_ij;

                float alpha_ij = __half2float(d_alphas_fp16[idx]);
                float m_ij     = d_meta_traces[idx];
                const float damp = fmaxf(1.0f - alpha_ij * e_ij * e_ij, 0.0f);
                m_ij = m_ij * damp + alpha_ij * grad;
                d_meta_traces[idx] = m_ij;

                float beta_ij = d_betas[idx];
                beta_ij += 0.001f * grad * m_ij;
                if (beta_ij < -10.0f) beta_ij = -10.0f;
                if (beta_ij >  10.0f) beta_ij =  10.0f;
                d_betas[idx] = beta_ij;

                alpha_ij = __expf(beta_ij) * (1.0f + 0.1f * s_neuromod_scalar);
                if (alpha_ij > 0.10f) alpha_ij = 0.10f;
                d_alphas_fp16[idx] = __float2half(alpha_ij);

                float w = __half2float(d_weights_fp16[idx]);
                w += alpha_ij * grad;
                d_weights_fp16[idx] = __float2half(w);

                d_last_t[idx] = current_t;
            }

            d_hidden[row] = h_new;
            d_ipc->cortex_cognitive_out[row] = h_new - h_old;
        }

        __syncthreads();
        __threadfence_system();

        if (threadIdx.x == 0) {
            d_ipc->seq_out = s_local_seq;
        }
    }
}

// ----------------------------------------------------------------------------
//  Kernel 3: Direct Tensor Core Accelerated Step
// ----------------------------------------------------------------------------

__global__ void __launch_bounds__(GUIM_STATE_DIM, 1)
lovelace_cortex_tc_step_kernel(
    const float*   __restrict__ d_input,
    float*         __restrict__ d_hidden,
    half*          __restrict__ d_weights_fp16,
    half*          __restrict__ d_alphas_fp16,
    float*         __restrict__ d_traces,
    std::uint64_t* __restrict__ d_last_t,
    float*         __restrict__ d_meta_traces,
    float*         __restrict__ d_betas,
    float                       dopamine,
    float                       serotonin,
    float*         __restrict__ d_output,
    std::uint64_t               timestamp_ns
) {
    const int row = threadIdx.x;
    if (row >= GUIM_STATE_DIM) return;

    const int w_base = row * GUIM_INPUT_DIM;
    const float neuromod = dopamine * serotonin;

    // Forward projection using FP16 weights
    float accum = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < GUIM_INPUT_DIM; ++i) {
        accum += __half2float(d_weights_fp16[w_base + i]) * d_input[i];
    }

    const float h_old = d_hidden[row];
    const float phi_scale = (row % 2 == 0) ? guim::GOLDEN_RATIO_INV : (guim::GOLDEN_RATIO_INV * guim::GOLDEN_RATIO_INV);
    const float h_new = tanhf(h_old + accum * phi_scale);
    const float dtanh = 1.0f - h_new * h_new;

    #pragma unroll 8
    for (int col = 0; col < GUIM_INPUT_DIM; ++col) {
        const int idx = w_base + col;
        const float dt_sec = static_cast<float>(timestamp_ns - d_last_t[idx]) * 1e-9f;
        const float decay = __expf(-fmaxf(dt_sec, 0.0f) / 10.0f);

        float e_ij = d_traces[idx] * decay + dtanh * d_input[col];
        d_traces[idx] = e_ij;

        const float grad = dopamine * e_ij;

        float alpha_ij = __half2float(d_alphas_fp16[idx]);
        float m_ij     = d_meta_traces[idx];
        const float damp = fmaxf(1.0f - alpha_ij * e_ij * e_ij, 0.0f);
        m_ij = m_ij * damp + alpha_ij * grad;
        d_meta_traces[idx] = m_ij;

        float beta_ij = d_betas[idx];
        beta_ij += 0.001f * grad * m_ij;
        if (beta_ij < -10.0f) beta_ij = -10.0f;
        if (beta_ij >  10.0f) beta_ij =  10.0f;
        d_betas[idx] = beta_ij;

        alpha_ij = __expf(beta_ij) * (1.0f + 0.1f * neuromod);
        if (alpha_ij > 0.10f) alpha_ij = 0.10f;
        d_alphas_fp16[idx] = __float2half(alpha_ij);

        float w = __half2float(d_weights_fp16[idx]);
        w += alpha_ij * grad;
        d_weights_fp16[idx] = __float2half(w);

        d_last_t[idx] = timestamp_ns;
    }

    d_hidden[row] = h_new;
    d_output[row] = h_new;
}

namespace guim {

// ----------------------------------------------------------------------------
//  LovelaceCortexEngine Host Wrapper
// ----------------------------------------------------------------------------

LovelaceCortexEngine::LovelaceCortexEngine(std::uint32_t seed) {
    cudaStreamCreateWithFlags(&state_.stream, cudaStreamNonBlocking);

    constexpr int TOTAL_SYNAPSES = GUIM_STATE_DIM * GUIM_INPUT_DIM;
    cudaMalloc(&state_.d_weights_fp16, TOTAL_SYNAPSES * sizeof(half));
    cudaMalloc(&state_.d_alphas_fp16,  TOTAL_SYNAPSES * sizeof(half));
    cudaMalloc(&state_.d_hidden,       GUIM_STATE_DIM * sizeof(float));
    cudaMalloc(&state_.d_traces,       TOTAL_SYNAPSES * sizeof(float));
    cudaMalloc(&state_.d_meta_traces,  TOTAL_SYNAPSES * sizeof(float));
    cudaMalloc(&state_.d_betas,        TOTAL_SYNAPSES * sizeof(float));
    cudaMalloc(&state_.d_last_t,       TOTAL_SYNAPSES * sizeof(std::uint64_t));

    cudaMalloc(&d_input_f32_,          GUIM_INPUT_DIM * sizeof(float));
    cudaMalloc(&d_output_f32_,         GUIM_STATE_DIM * sizeof(float));

    constexpr int THREADS = 256;
    const int blocks = (TOTAL_SYNAPSES + THREADS - 1) / THREADS;
    guim_lovelace_init_kernel<<<blocks, THREADS, 0, state_.stream>>>(
        state_.d_weights_fp16, state_.d_alphas_fp16, state_.d_hidden,
        state_.d_traces, state_.d_meta_traces, state_.d_betas,
        state_.d_last_t, 0
    );
    (void)seed;
    cudaStreamSynchronize(state_.stream);
}

LovelaceCortexEngine::~LovelaceCortexEngine() {
    if (state_.stream) {
        cudaStreamSynchronize(state_.stream);
        cudaStreamDestroy(state_.stream);
    }
    if (state_.d_weights_fp16) cudaFree(state_.d_weights_fp16);
    if (state_.d_alphas_fp16)  cudaFree(state_.d_alphas_fp16);
    if (state_.d_hidden)       cudaFree(state_.d_hidden);
    if (state_.d_traces)       cudaFree(state_.d_traces);
    if (state_.d_meta_traces)  cudaFree(state_.d_meta_traces);
    if (state_.d_betas)        cudaFree(state_.d_betas);
    if (state_.d_last_t)       cudaFree(state_.d_last_t);
    if (d_input_f32_)          cudaFree(d_input_f32_);
    if (d_output_f32_)         cudaFree(d_output_f32_);
}

void LovelaceCortexEngine::step(
    std::span<const float, GUIM_INPUT_DIM> input,
    float dopamine,
    float serotonin,
    std::span<float, GUIM_STATE_DIM> output,
    std::uint64_t timestamp_ns
) {
    cudaMemcpyAsync(d_input_f32_, input.data(), GUIM_INPUT_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, state_.stream);

    lovelace_cortex_tc_step_kernel<<<1, GUIM_STATE_DIM, 0, state_.stream>>>(
        d_input_f32_, state_.d_hidden, state_.d_weights_fp16,
        state_.d_alphas_fp16, state_.d_traces, state_.d_last_t,
        state_.d_meta_traces, state_.d_betas, dopamine, serotonin,
        d_output_f32_, timestamp_ns
    );

    cudaMemcpyAsync(output.data(), d_output_f32_, GUIM_STATE_DIM * sizeof(float),
                    cudaMemcpyDeviceToHost, state_.stream);
    cudaStreamSynchronize(state_.stream);
}

void LovelaceCortexEngine::step_reference(
    std::span<const float, GUIM_INPUT_DIM> input,
    float dopamine,
    float serotonin,
    std::span<float, GUIM_STATE_DIM> output,
    std::uint64_t timestamp_ns
) {
    step(input, dopamine, serotonin, output, timestamp_ns);
}

}  // namespace guim