// ============================================================================
//  persistent_kernel.cu — Zero-copy IPC persistent kernel (Guimlab substrate)
//  ---------------------------------------------------------------------------
//  The GPU kernel stays resident in VRAM:
//    - Spin-waits on seq_in with hardware nanosleep + %globaltimer WATCHDOG
//    - Recurrent forward pass with shared memory double-buffering (0 race condition)
//    - CF-TT continuous eligibility traces
//    - TMD-ET per-synapse adaptive meta-gradients (IDBD)
//    - Continual Backpropagation (CBP) metabolic variance tracking & reset
//    - Publishes to zero-copy POSIX SHM via __threadfence_system + seq_out
//
//  Author        : Guimlab Contributors
//  License       : AGPL-3.0 / MIT
// ============================================================================

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "ipc_structs.h"

// ----------------------------------------------------------------------------
//  Hyperparameters (static constant memory to avoid TU collision)
// ----------------------------------------------------------------------------

static __constant__ float GUIM_GAMMA                = 0.99f;
static __constant__ float GUIM_LAMBDA               = 0.90f;
static __constant__ float GUIM_MU                   = 0.001f;
static __constant__ float GUIM_ALPHA_MAX            = 0.10f;
static __constant__ float GUIM_BETA_MIN             = -10.0f;
static __constant__ float GUIM_BETA_MAX             =  10.0f;
static __constant__ float GUIM_PLASTICITY_THRESH    = 1e-4f;

__device__ __forceinline__ unsigned long long read_globaltimer_ns() {
    unsigned long long t;
    #if __CUDA_ARCH__ >= 300
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    #else
    t = clock64();
    #endif
    return t;
}

// ----------------------------------------------------------------------------
//  Persistent Kernel (1 Block of 256 threads — 1 thread per state neuron)
// ----------------------------------------------------------------------------

__global__ void __launch_bounds__(GUIM_STATE_DIM, 1)
persistent_guim_kernel(
    volatile GuimSharedMemory* __restrict__ ipc,
    float*        __restrict__ d_hidden,
    float*        __restrict__ d_weights,
    float*        __restrict__ d_traces,
    float*        __restrict__ d_alphas,
    float*        __restrict__ d_betas,
    float*        __restrict__ d_meta_traces,
    float*        __restrict__ d_utility,
    float*        __restrict__ d_mean,
    curandState*  __restrict__ d_rng
) {
    const int row = threadIdx.x;  // Exactly [0 .. GUIM_STATE_DIM-1]

    __shared__ std::uint64_t s_local_seq;
    __shared__ float         s_hidden_prev[GUIM_STATE_DIM];
    __shared__ float         s_input[GUIM_INPUT_DIM];
    __shared__ float         s_td_error;

    // Load initial hidden state into shared memory
    s_hidden_prev[row] = d_hidden[row];

    if (row == 0) {
        s_local_seq = ipc->seq_in;
    }
    __syncthreads();

    // 2.0s Hardware Watchdog timeout against Zombie GPU states
    constexpr unsigned long long WATCHDOG_TIMEOUT_NS = 2000000000ull;

    while (!ipc->terminate) {
        // ---- PHASE 1: Spin-wait on seq_in with hardware nanosleep + watchdog ----
        if (row == 0) {
            const unsigned long long t_start = read_globaltimer_ns();
            while (ipc->seq_in <= s_local_seq && !ipc->terminate) {
                #if __CUDA_ARCH__ >= 700
                __nanosleep(100);
                #else
                asm volatile("nanosleep.u32 100;");
                #endif

                if (read_globaltimer_ns() - t_start > WATCHDOG_TIMEOUT_NS) {
                    ipc->terminate = true;
                    break;
                }
            }
            s_local_seq = ipc->seq_in;
            s_td_error  = ipc->td_error;
        }
        __syncthreads();
        if (ipc->terminate) break;

        // ---- PHASE 2: Parallel load input from zero-copy SHM to shared memory ----
        if (row < GUIM_INPUT_DIM) {
            s_input[row] = ipc->input[row];
        }
        __syncthreads();

        const float td_error = s_td_error;
        const int w_base = row * GUIM_TOTAL_IN;

        // ---- PHASE 3: Forward recurrent pass (reads from shared memory, 0 race) ----
        float accum = 0.0f;

        #pragma unroll 8
        for (int i = 0; i < GUIM_INPUT_DIM; ++i) {
            accum += d_weights[w_base + i] * s_input[i];
        }

        #pragma unroll 8
        for (int i = 0; i < GUIM_STATE_DIM; ++i) {
            accum += d_weights[w_base + GUIM_INPUT_DIM + i] * s_hidden_prev[i];
        }

        const float h_new = tanhf(accum);
        const float dtanh = 1.0f - h_new * h_new;

        // ---- PHASE 4: Continual Backpropagation (CBP) Metabolic Tracking ----
        constexpr float BETA_EMA = 0.99f;
        float current_mean = d_mean[row];
        current_mean = BETA_EMA * current_mean + (1.0f - BETA_EMA) * h_new;
        d_mean[row] = current_mean;

        const float var = (h_new - current_mean) * (h_new - current_mean);
        float current_util = d_utility[row];
        current_util = BETA_EMA * current_util + (1.0f - BETA_EMA) * var;
        d_utility[row] = current_util;

        // ---- PHASE 5: Synaptic Update / Plasticity Reset ----
        if (current_util < GUIM_PLASTICITY_THRESH) {
            // Neurogenesis: re-randomize dead neuron weights without host interruption
            d_utility[row] = 0.1f;
            curandState local_rng = d_rng[row];
            const float scale = sqrtf(2.0f / static_cast<float>(GUIM_TOTAL_IN));

            #pragma unroll 8
            for (int col = 0; col < GUIM_TOTAL_IN; ++col) {
                const int idx = w_base + col;
                const float r = curand_uniform(&local_rng) - 0.5f;
                d_weights[idx]     = r * 2.0f * scale;
                d_traces[idx]      = 0.0f;
                d_meta_traces[idx] = 0.0f;
                d_alphas[idx]      = 0.01f;
                d_betas[idx]       = -4.6f;
            }
            d_rng[row] = local_rng;
        } else {
            // TMD-ET per-synapse meta-updates
            #pragma unroll 8
            for (int col = 0; col < GUIM_TOTAL_IN; ++col) {
                const int idx = w_base + col;
                const float in_val = (col < GUIM_INPUT_DIM)
                                     ? s_input[col]
                                     : s_hidden_prev[col - GUIM_INPUT_DIM];

                // Eligibility trace
                float e_ij = (GUIM_LAMBDA * GUIM_GAMMA * d_traces[idx]) + (dtanh * in_val);
                d_traces[idx] = e_ij;

                const float grad = td_error * e_ij;

                // Meta-trace (IDBD)
                float alpha_ij = d_alphas[idx];
                float m_ij     = d_meta_traces[idx];
                m_ij = m_ij * (1.0f - alpha_ij * e_ij * e_ij) + alpha_ij * grad;
                d_meta_traces[idx] = m_ij;

                // Meta-gradient update on log learning rate beta
                float beta_ij = d_betas[idx];
                beta_ij += GUIM_MU * grad * m_ij;
                if (beta_ij < GUIM_BETA_MIN) beta_ij = GUIM_BETA_MIN;
                if (beta_ij > GUIM_BETA_MAX) beta_ij = GUIM_BETA_MAX;
                d_betas[idx] = beta_ij;

                // Update alpha
                alpha_ij = __expf(beta_ij);
                if (alpha_ij > GUIM_ALPHA_MAX) alpha_ij = GUIM_ALPHA_MAX;
                d_alphas[idx] = alpha_ij;

                // Weight update
                d_weights[idx] += alpha_ij * grad;
            }
        }

        // ---- PHASE 6: Write to SHM output and update recurrent buffer ----
        ipc->output[row] = h_new;
        d_hidden[row]    = h_new;

        __syncthreads();
        s_hidden_prev[row] = h_new;

        // ---- PHASE 7: System fence & sequence publish ----
        __syncthreads();              // Wait for all threads to write ipc->output
        __threadfence_system();       // Flush L2 cache to host RAM over PCIe
        __syncthreads();              // Ensure fence is complete across warp before seq_out update

        if (row == 0) {
            ipc->seq_out = s_local_seq;
        }
    }
}

// ----------------------------------------------------------------------------
//  Initialization Kernel
// ----------------------------------------------------------------------------

__global__ void guim_init_kernel(
    float*       __restrict__ d_weights,
    float*       __restrict__ d_alphas,
    float*       __restrict__ d_betas,
    float*       __restrict__ d_hidden,
    float*       __restrict__ d_traces,
    float*       __restrict__ d_meta_traces,
    float*       __restrict__ d_utility,
    float*       __restrict__ d_mean,
    curandState* __restrict__ d_rng,
    unsigned long long        seed
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < GUIM_TOTAL_W) {
        const float scale = sqrtf(2.0f / static_cast<float>(GUIM_TOTAL_IN));
        unsigned long long state = seed ^ (static_cast<unsigned long long>(idx) * 0x9E3779B97F4A7C15ull);
        state = (state ^ (state >> 30)) * 0xBF58476D1CE4E5B9ull;
        state = (state ^ (state >> 27)) * 0x94D049BB133111EBull;
        state = state ^ (state >> 31);
        const float u = (static_cast<float>(state & 0xFFFFFF) / static_cast<float>(0x1000000)) - 0.5f;

        d_weights[idx]     = u * 2.0f * scale;
        d_alphas[idx]      = 0.01f;
        d_betas[idx]       = -4.6f;
        d_traces[idx]      = 0.0f;
        d_meta_traces[idx] = 0.0f;
    }

    if (idx < GUIM_STATE_DIM) {
        d_hidden[idx]  = 0.0f;
        d_utility[idx] = 0.1f;
        d_mean[idx]    = 0.0f;
        curand_init(seed + idx, idx, 0, &d_rng[idx]);
    }
}