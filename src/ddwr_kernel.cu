// ============================================================================
//  ddwr_kernel.cu — Delta-Driven Warp Routing + Closed-Form Continuous-Time Traces
//  ---------------------------------------------------------------------------
//  Sparse persistent kernel: only synapses whose inputs change consume VRAM
//  bandwidth. Warp ballot (__ballot_sync) skips dormant warps in 1 cycle.
//
//  Author        : Guimlab Contributors
//  License       : AGPL-3.0 / MIT
// ============================================================================

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "ipc_structs.h"
#include "ipc_structs_sparse.h"

// Sparse hyperparameters
static __constant__ float DDWR_THRESHOLD    = 1e-4f;     // activation threshold
static __constant__ float DDWR_TAU          = 10.0f;     // trace time constant (seconds)
static __constant__ float DDWR_GAMMA        = 0.99f;     // TD discount
static __constant__ float DDWR_LAMBDA       = 0.90f;     // trace decay factor
static __constant__ float DDWR_MU           = 0.001f;    // meta-learning rate for beta
static __constant__ float DDWR_ALPHA_MAX    = 0.10f;     // per-synapse alpha cap

// ----------------------------------------------------------------------------
//  Sparse persistent kernel
// ----------------------------------------------------------------------------

__global__ void sparse_persistent_guim_kernel(
    volatile GuimSharedMemorySparse* __restrict__ ipc,
    float*         __restrict__ d_hidden,           // [STATE_DIM]
    float*         __restrict__ d_weights,          // [STATE_DIM * TOTAL_IN]
    float*         __restrict__ d_traces,           // [STATE_DIM * TOTAL_IN]
    float*         __restrict__ d_alphas,           // [STATE_DIM * TOTAL_IN]
    float*         __restrict__ d_betas,            // [STATE_DIM * TOTAL_IN]
    float*         __restrict__ d_meta_traces,      // [STATE_DIM * TOTAL_IN]
    std::uint64_t* __restrict__ d_last_t,           // [STATE_DIM * TOTAL_IN]
    float*         __restrict__ d_input_state       // [INPUT_DIM]
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ std::uint64_t s_local_seq;
    __shared__ float         s_sparse_inputs[GUIM_INPUT_DIM];

    if (threadIdx.x == 0) {
        s_local_seq = ipc->seq_in;
    }
    __syncthreads();

    while (!ipc->terminate) {
        // ---- Spin-wait on seq_in ----
        if (threadIdx.x == 0) {
            while (ipc->seq_in <= s_local_seq && !ipc->terminate) {
                #if __CUDA_ARCH__ >= 700
                __nanosleep(100);
                #else
                asm volatile("nanosleep.u32 100;");
                #endif
            }
            s_local_seq = ipc->seq_in;
        }
        __syncthreads();
        if (ipc->terminate) break;

        // ---- Read delta metadata ----
        const int delta_count = ipc->delta_count;
        const std::uint64_t current_t = ipc->timestamp_ns;
        const float td_error = ipc->td_error;

        // Clear sparse input cache
        if (threadIdx.x < GUIM_INPUT_DIM) {
            s_sparse_inputs[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Populate active delta inputs
        if (threadIdx.x == 0) {
            for (int d = 0; d < delta_count && d < GUIM_SPARSE_MAX_DELTAS; ++d) {
                const int in_idx = ipc->delta_indices[d];
                if (in_idx < GUIM_INPUT_DIM) {
                    s_sparse_inputs[in_idx] = ipc->delta_values[d];
                    d_input_state[in_idx] += ipc->delta_values[d];
                }
            }
        }
        __syncthreads();

        // ---- PHASE 1 — Local stimulus accumulation (delta-only) ----
        float stimulus_accum = 0.0f;
        if (delta_count > 0 && row < GUIM_STATE_DIM) {
            const int w_base = row * (GUIM_INPUT_DIM + GUIM_STATE_DIM);
            for (int d = 0; d < delta_count; ++d) {
                const int in_idx = ipc->delta_indices[d];
                if (in_idx < (GUIM_INPUT_DIM + GUIM_STATE_DIM)) {
                    const float in_val = ipc->delta_values[d];
                    stimulus_accum += d_weights[w_base + in_idx] * in_val;
                }
            }
        }

        // ---- PHASE 2 — Warp ballot (DDWR) ----
        const bool is_active = (row < GUIM_STATE_DIM) && (fabsf(stimulus_accum) > DDWR_THRESHOLD);
        const unsigned int active_mask = __ballot_sync(0xFFFFFFFFu, is_active);

        // If NO thread in warp is active, skip VRAM writes
        if (active_mask == 0u) {
            if (row < GUIM_STATE_DIM) {
                ipc->delta_output[row] = 0.0f;
            }
            __syncthreads();
            __threadfence_system();
            if (threadIdx.x == 0) {
                ipc->seq_out = s_local_seq;
            }
            continue;
        }

        // ---- PHASE 3 — Active neuron: full TMD-ET step with CF-CTT ----
        if (is_active) {
            const float h_old = d_hidden[row];
            const float h_new = tanhf(h_old + stimulus_accum);
            const float dtanh = 1.0f - h_new * h_new;

            const int w_base = row * (GUIM_INPUT_DIM + GUIM_STATE_DIM);
            for (int col = 0; col < GUIM_INPUT_DIM + GUIM_STATE_DIM; ++col) {
                const int idx = w_base + col;

                // CF-CTT: closed-form decay
                const float dt_sec = static_cast<float>(current_t - d_last_t[idx]) * 1e-9f;
                const float decay_factor = __expf(-fmaxf(dt_sec, 0.0f) / DDWR_TAU);

                const float in_val = (col >= GUIM_INPUT_DIM)
                                     ? d_hidden[col - GUIM_INPUT_DIM]
                                     : d_input_state[col];

                // Eligibility trace
                float e_ij = d_traces[idx] * decay_factor + dtanh * in_val;
                d_traces[idx] = e_ij;

                // Gradient
                const float grad = td_error * e_ij;

                // Meta-trace (IDBD)
                float alpha_ij = d_alphas[idx];
                float m_ij     = d_meta_traces[idx];
                m_ij = m_ij * (1.0f - alpha_ij * e_ij * e_ij) + alpha_ij * grad;
                d_meta_traces[idx] = m_ij;

                // Meta-gradient update on log learning rate beta
                float beta_ij = d_betas[idx];
                beta_ij += DDWR_MU * grad * m_ij;
                if (beta_ij < -10.0f) beta_ij = -10.0f;
                if (beta_ij >  10.0f) beta_ij =  10.0f;
                d_betas[idx] = beta_ij;

                alpha_ij = __expf(beta_ij);
                if (alpha_ij > DDWR_ALPHA_MAX) alpha_ij = DDWR_ALPHA_MAX;
                d_alphas[idx] = alpha_ij;

                // Weight update
                d_weights[idx] += alpha_ij * grad;
                d_last_t[idx]   = current_t;
            }

            d_hidden[row] = h_new;
            ipc->delta_output[row] = h_new;
        } else if (row < GUIM_STATE_DIM) {
            ipc->delta_output[row] = d_hidden[row];
        }

        __syncthreads();
        __threadfence_system();
        if (threadIdx.x == 0) {
            ipc->seq_out = s_local_seq;
        }
    }
}