// =============================================================================
//  guim_kernel.cu — Core CUDA implementation of the Guimlab continual substrate
//  =============================================================================
//  Implements:
//    1. Recurrent forward pass with closed-form activations
//    2. Temporal Meta-Descent with Eligibility Traces (TMD-ET / IDBD)
//    3. Continual Backpropagation (CBP) with in-place neurogenesis
//    4. Zero-allocation runtime step loop
//
//  Author        : Guimlab Contributors
//  SPDX-License   : MIT
// =============================================================================

#include "guim_kernel.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <new>

// -----------------------------------------------------------------------------
//  Internal State Definition
// -----------------------------------------------------------------------------

struct GuimKernel {
    // Device buffers (single static arena)
    float*         d_input       = nullptr;  // [GUIM_INPUT_DIM]
    float*         d_hidden_in   = nullptr;  // [GUIM_STATE_DIM]
    float*         d_hidden_out  = nullptr;  // [GUIM_STATE_DIM]
    float*         d_weights     = nullptr;  // [GUIM_TOTAL_W]
    float*         d_traces      = nullptr;  // [GUIM_TOTAL_W]
    float*         d_alphas      = nullptr;  // [GUIM_TOTAL_W]
    float*         d_betas       = nullptr;  // [GUIM_TOTAL_W]
    float*         d_meta_traces = nullptr;  // [GUIM_TOTAL_W]
    float*         d_utility     = nullptr;  // [GUIM_STATE_DIM]
    float*         d_mean        = nullptr;  // [GUIM_STATE_DIM]
    float*         d_td_error    = nullptr;  // [1]
    curandState*   d_rng_states  = nullptr;  // [GUIM_STATE_DIM]
    std::uint32_t* d_resets      = nullptr;  // [1]

    cudaStream_t   stream        = nullptr;

    // Host mirrors for introspection
    float          h_hidden[GUIM_STATE_DIM] = {0};

    std::uint32_t  resets_count  = 0;
    std::uint64_t  frames_run    = 0;
    std::size_t    arena_bytes   = 0;
    std::uint32_t  seed          = 0;
    bool           initialized   = false;
};

// -----------------------------------------------------------------------------
//  Device Kernels
// -----------------------------------------------------------------------------

__global__ void guim_core_init_kernel(
    float*       __restrict__ d_weights,
    float*       __restrict__ d_alphas,
    float*       __restrict__ d_betas,
    float*       __restrict__ d_traces,
    float*       __restrict__ d_meta_traces,
    float*       __restrict__ d_hidden_in,
    float*       __restrict__ d_hidden_out,
    float*       __restrict__ d_utility,
    float*       __restrict__ d_mean,
    curandState* __restrict__ d_rng_states,
    std::uint32_t* __restrict__ d_resets,
    std::uint32_t seed
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < GUIM_TOTAL_W) {
        // Deterministic Xavier initialization
        const float scale = sqrtf(2.0f / static_cast<float>(GUIM_TOTAL_DIM));
        unsigned long long s = seed ^ (static_cast<unsigned long long>(idx) * 0x9E3779B97F4A7C15ull);
        s = (s ^ (s >> 30)) * 0xBF58476D1CE4E5B9ull;
        s = (s ^ (s >> 27)) * 0x94D049BB133111EBull;
        s = s ^ (s >> 31);
        const float u = (static_cast<float>(s & 0xFFFFFF) / static_cast<float>(0x1000000)) - 0.5f;

        d_weights[idx]     = u * 2.0f * scale;
        d_alphas[idx]      = GUIM_ALPHA;
        d_betas[idx]       = logf(GUIM_ALPHA);
        d_traces[idx]      = 0.0f;
        d_meta_traces[idx] = 0.0f;
    }

    if (idx < GUIM_STATE_DIM) {
        d_hidden_in[idx]  = 0.0f;
        d_hidden_out[idx] = 0.0f;
        d_utility[idx]    = 0.1f;  // start with high utility
        d_mean[idx]       = 0.0f;
        curand_init(seed + idx, idx, 0, &d_rng_states[idx]);
    }

    if (idx == 0) {
        *d_resets = 0;
    }
}

__global__ void guim_core_step_kernel(
    const float*  __restrict__ d_input,
    const float*  __restrict__ d_hidden_in,
    float*        __restrict__ d_hidden_out,
    float*        __restrict__ d_weights,
    float*        __restrict__ d_traces,
    float*        __restrict__ d_alphas,
    float*        __restrict__ d_betas,
    float*        __restrict__ d_meta_traces,
    float*        __restrict__ d_utility,
    float*        __restrict__ d_mean,
    curandState*  __restrict__ d_rng_states,
    std::uint32_t* __restrict__ d_resets,
    float*        __restrict__ d_td_error_out
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= GUIM_STATE_DIM) return;

    const int w_base = row * GUIM_TOTAL_DIM;

    // 1. Forward recurrent activation
    float accum = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < GUIM_INPUT_DIM; ++i) {
        accum += d_weights[w_base + i] * d_input[i];
    }
    #pragma unroll 8
    for (int i = 0; i < GUIM_STATE_DIM; ++i) {
        accum += d_weights[w_base + GUIM_INPUT_DIM + i] * d_hidden_in[i];
    }

    const float h_new = tanhf(accum);
    const float dtanh = 1.0f - h_new * h_new;

    // 2. Online TD prediction error (predicting sensor 0 signal)
    const float prediction_target = d_input[0];
    const float td_error = prediction_target - h_new;

    if (row == 0) {
        *d_td_error_out = td_error;
    }

    // 3. Continual Backpropagation (CBP) Metabolic Tracking
    constexpr float BETA_EMA = 0.99f;
    float current_mean = d_mean[row];
    current_mean = BETA_EMA * current_mean + (1.0f - BETA_EMA) * h_new;
    d_mean[row] = current_mean;

    const float var = (h_new - current_mean) * (h_new - current_mean);
    float current_utility = d_utility[row];
    current_utility = BETA_EMA * current_utility + (1.0f - BETA_EMA) * var;
    d_utility[row] = current_utility;

    // 4. Neurogenesis check (loss of plasticity recovery)
    if (current_utility < GUIM_PLASTICITY_THRESHOLD) {
        atomicAdd(d_resets, 1u);
        d_utility[row] = 0.1f;  // reset utility
        curandState local_rng = d_rng_states[row];
        const float scale = sqrtf(2.0f / static_cast<float>(GUIM_TOTAL_DIM));

        #pragma unroll 8
        for (int col = 0; col < GUIM_TOTAL_DIM; ++col) {
            const int idx = w_base + col;
            const float r = curand_uniform(&local_rng) - 0.5f;
            d_weights[idx]     = r * 2.0f * scale;
            d_traces[idx]      = 0.0f;
            d_meta_traces[idx] = 0.0f;
            d_alphas[idx]      = GUIM_ALPHA;
            d_betas[idx]       = logf(GUIM_ALPHA);
        }
        d_rng_states[row] = local_rng;
    } else {
        // 5. TMD-ET per-synapse meta-updates
        #pragma unroll 8
        for (int col = 0; col < GUIM_TOTAL_DIM; ++col) {
            const int idx = w_base + col;
            const float in_val = (col < GUIM_INPUT_DIM)
                                 ? d_input[col]
                                 : d_hidden_in[col - GUIM_INPUT_DIM];

            // Eligibility trace
            float e_ij = (GUIM_LAMBDA * d_traces[idx]) + (dtanh * in_val);
            if (fabsf(e_ij) > GUIM_TRACE_CLIP) e_ij = copysignf(GUIM_TRACE_CLIP, e_ij);
            d_traces[idx] = e_ij;

            const float grad = td_error * e_ij;

            // Meta-trace (IDBD)
            float alpha_ij = d_alphas[idx];
            float m_ij     = d_meta_traces[idx];
            m_ij = m_ij * (1.0f - alpha_ij * e_ij * e_ij) + alpha_ij * grad;
            d_meta_traces[idx] = m_ij;

            // Beta meta-gradient
            float beta_ij = d_betas[idx];
            beta_ij += 0.001f * grad * m_ij;
            if (beta_ij < -10.0f) beta_ij = -10.0f;
            if (beta_ij >   5.0f) beta_ij =   5.0f;
            d_betas[idx] = beta_ij;

            // Alpha update
            alpha_ij = expf(beta_ij);
            if (alpha_ij > 0.05f) alpha_ij = 0.05f;
            d_alphas[idx] = alpha_ij;

            // Weight update + L1 weight decay
            float w = d_weights[idx];
            w += alpha_ij * grad - GUIM_WEIGHT_DECAY * w;
            if (fabsf(w) > GUIM_WEIGHT_CLIP) w = copysignf(GUIM_WEIGHT_CLIP, w);
            d_weights[idx] = w;
        }
    }

    d_hidden_out[row] = h_new;
}

// -----------------------------------------------------------------------------
//  C ABI Implementation
// -----------------------------------------------------------------------------

extern "C" GuimKernel* guim_create(std::uint32_t seed)
{
    auto* k = new (std::nothrow) GuimKernel;
    if (!k) return nullptr;

    k->seed = seed;
    cudaStreamCreateWithFlags(&k->stream, cudaStreamNonBlocking);

    // Compute sizes
    const std::size_t size_input       = GUIM_INPUT_DIM * sizeof(float);
    const std::size_t size_state       = GUIM_STATE_DIM * sizeof(float);
    const std::size_t size_weights     = GUIM_TOTAL_W * sizeof(float);
    const std::size_t size_rng         = GUIM_STATE_DIM * sizeof(curandState);

    k->arena_bytes = size_input + (2 * size_state) + (5 * size_weights)
                   + (2 * size_state) + sizeof(float) + size_rng + sizeof(std::uint32_t);

    cudaMalloc(&k->d_input,       size_input);
    cudaMalloc(&k->d_hidden_in,   size_state);
    cudaMalloc(&k->d_hidden_out,  size_state);
    cudaMalloc(&k->d_weights,     size_weights);
    cudaMalloc(&k->d_traces,      size_weights);
    cudaMalloc(&k->d_alphas,      size_weights);
    cudaMalloc(&k->d_betas,       size_weights);
    cudaMalloc(&k->d_meta_traces, size_weights);
    cudaMalloc(&k->d_utility,     size_state);
    cudaMalloc(&k->d_mean,        size_state);
    cudaMalloc(&k->d_td_error,    sizeof(float));
    cudaMalloc(&k->d_rng_states,  size_rng);
    cudaMalloc(&k->d_resets,      sizeof(std::uint32_t));

    // Initialize on GPU
    constexpr int THREADS = 256;
    const int blocks = (GUIM_TOTAL_W + THREADS - 1) / THREADS;
    guim_core_init_kernel<<<blocks, THREADS, 0, k->stream>>>(
        k->d_weights, k->d_alphas, k->d_betas, k->d_traces, k->d_meta_traces,
        k->d_hidden_in, k->d_hidden_out, k->d_utility, k->d_mean,
        k->d_rng_states, k->d_resets, seed
    );
    cudaStreamSynchronize(k->stream);

    k->initialized = true;
    return k;
}

extern "C" void guim_destroy(GuimKernel* k)
{
    if (!k) return;
    if (k->stream) {
        cudaStreamSynchronize(k->stream);
        cudaStreamDestroy(k->stream);
    }
    if (k->d_input)       cudaFree(k->d_input);
    if (k->d_hidden_in)   cudaFree(k->d_hidden_in);
    if (k->d_hidden_out)  cudaFree(k->d_hidden_out);
    if (k->d_weights)     cudaFree(k->d_weights);
    if (k->d_traces)      cudaFree(k->d_traces);
    if (k->d_alphas)      cudaFree(k->d_alphas);
    if (k->d_betas)       cudaFree(k->d_betas);
    if (k->d_meta_traces) cudaFree(k->d_meta_traces);
    if (k->d_utility)     cudaFree(k->d_utility);
    if (k->d_mean)        cudaFree(k->d_mean);
    if (k->d_td_error)    cudaFree(k->d_td_error);
    if (k->d_rng_states)  cudaFree(k->d_rng_states);
    if (k->d_resets)      cudaFree(k->d_resets);

    delete k;
}

extern "C" float guim_step(GuimKernel* k, const float* input, float* output)
{
    if (!k || !k->initialized || !input || !output) return std::nanf("");

    // Guard against NaN / Inf in input
    for (int i = 0; i < GUIM_INPUT_DIM; ++i) {
        if (!std::isfinite(input[i])) {
            ++k->resets_count;
            for (int o = 0; o < GUIM_STATE_DIM; ++o) output[o] = std::nanf("");
            return std::nanf("");
        }
    }

    // Copy input to device
    cudaMemcpyAsync(k->d_input, input, GUIM_INPUT_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, k->stream);

    // Launch step kernel
    constexpr int THREADS = 64;
    const int blocks = (GUIM_STATE_DIM + THREADS - 1) / THREADS;
    guim_core_step_kernel<<<blocks, THREADS, 0, k->stream>>>(
        k->d_input, k->d_hidden_in, k->d_hidden_out,
        k->d_weights, k->d_traces, k->d_alphas, k->d_betas,
        k->d_meta_traces, k->d_utility, k->d_mean,
        k->d_rng_states, k->d_resets, k->d_td_error
    );

    // Copy output and td-error back to host
    float td_error_val = 0.0f;
    cudaMemcpyAsync(output, k->d_hidden_out, GUIM_STATE_DIM * sizeof(float),
                    cudaMemcpyDeviceToHost, k->stream);
    cudaMemcpyAsync(&td_error_val, k->d_td_error, sizeof(float),
                    cudaMemcpyDeviceToHost, k->stream);
    cudaStreamSynchronize(k->stream);

    // Swap hidden buffers for next frame
    float* tmp = k->d_hidden_in;
    k->d_hidden_in = k->d_hidden_out;
    k->d_hidden_out = tmp;

    // Cache latest output in host mirror
    std::memcpy(k->h_hidden, output, sizeof(k->h_hidden));

    ++k->frames_run;
    return fabsf(td_error_val);
}

extern "C" void guim_get_hidden(const GuimKernel* k, float* dst)
{
    if (!k || !dst) return;
    std::memcpy(dst, k->h_hidden, sizeof(k->h_hidden));
}

extern "C" void guim_get_weights(const GuimKernel* k, float* dst)
{
    if (!k || !dst || !k->d_weights) return;
    cudaMemcpy(dst, k->d_weights, GUIM_TOTAL_W * sizeof(float), cudaMemcpyDeviceToHost);
}

extern "C" std::uint32_t guim_resets_count(const GuimKernel* k)
{
    if (!k) return 0;
    std::uint32_t dev_resets = 0;
    if (k->d_resets) {
        cudaMemcpy(&dev_resets, k->d_resets, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    }
    return k->resets_count + dev_resets;
}

extern "C" std::uint64_t guim_frames_run(const GuimKernel* k)
{
    return k ? k->frames_run : 0ull;
}

extern "C" std::size_t guim_bytes_resident(const GuimKernel* k)
{
    return k ? k->arena_bytes : 0ull;
}