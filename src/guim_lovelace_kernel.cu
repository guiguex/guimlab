// ============================================================================
// guim_lovelace_kernel.cu — Fused meta-cognitive kernel (Cortex Layer)
// ---------------------------------------------------------------------------
// Fused pass: SIMD surprise + neuromodulation + DDWR warp ballot + forward +
// + TMD-ET meta-gradients + CF-TT continuous-time eligibility traces.
//
//  Author        : Guimlab Contributors
//  License       : AGPL-3.0
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdio>

#include "ipc_structs.h"
#include "ipc_structs_v2.h"

// Surprise threshold for INT8 SAD (out of 4 * 255 = 1020 max)
constexpr std::uint32_t SURPRISE_THRESHOLD = 10;

// Byte-wise sum of absolute differences of 4 packed bytes
__device__ __forceinline__
std::uint32_t ptx_surprise(std::uint32_t real_packed, std::uint32_t pred_packed) {
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

// Init kernel for fp16 weights/alphas
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

    // Xavier/Glorot pseudo-random initialization
    const float scale = sqrtf(2.0f / static_cast<float>(GUIM_INPUT_DIM));
    unsigned long long state = 0xC0FFEEull ^ (static_cast<unsigned long long>(idx) * 0x9E3779B97F4A7C15ull);
    state = (state ^ (state >> 30)) * 0xBF58476D1CE4E5B9ull;
    state = (state ^ (state >> 27)) * 0x94D049BB133111EBull;
    state = state ^ (state >> 31);
    const float u = (static_cast<float>(state & 0xFFFFFF) / static_cast<float>(0x1000000)) - 0.5f;

    d_weights_fp16[idx] = __float2half(u * 2.0f * scale);
    d_alphas_fp16[idx]  = __float2half(0.01f);
    d_betas[idx]        = -4.6f;
    d_traces[idx]       = 0.0f;
    d_meta_traces[idx]  = 0.0f;
    d_last_t[idx]       = start_time_ns;

    if (idx < GUIM_STATE_DIM) {
        d_hidden[idx] = 0.0f;
    }
}

// ----------------------------------------------------------------------------
//  Main fused cortex kernel
// ----------------------------------------------------------------------------

__global__ void guim_lovelace_fused_kernel(
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
        // ---- PHASE 1 — Spin-wait on seq_in with backoff ----
        if (threadIdx.x == 0) {
            while (d_ipc->seq_in <= s_local_seq && !d_ipc->terminate) {
                #if __CUDA_ARCH__ >= 700
                __nanosleep(100);
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
            __threadfence_system();
            if (threadIdx.x == 0) {
                d_ipc->seq_out = s_local_seq;
            }
            continue;
        }

        // ---- PHASES 4-8 — Forward + TMD-ET + Neuromodulation ----
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
            const float h_new = tanhf(h_old + stimulus_accum);
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
                m_ij = m_ij * (1.0f - alpha_ij * e_ij * e_ij) + alpha_ij * grad;
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

        __threadfence_system();
        __syncthreads();

        if (threadIdx.x == 0) {
            d_ipc->seq_out = s_local_seq;
        }
    }
}