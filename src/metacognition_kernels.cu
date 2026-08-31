// ============================================================================
//  metacognition_kernels.cu — Meta-cognition implemented in PTX intrinsics
//  ---------------------------------------------------------------------------
//  Surprise metric via inline PTX vabsdiff4 and neuromodulation broadcast.
//
//  Author        : Guimlab Contributors
//  License       : AGPL-3.0
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdio>
#include <cmath>

#include "ipc_structs.h"

// PTX surprise metric: SAD of 4 packed uint8 values in one hardware instruction
__device__ __forceinline__
std::uint32_t ptx_surprise_metric(std::uint32_t real_packed,
                                  std::uint32_t predicted_packed) {
    std::uint32_t sad;
    asm volatile(
        "vabsdiff4.u32.u32.u32.add %0, %1, %2, 0;"
        : "=r"(sad)
        : "r"(real_packed), "r"(predicted_packed)
    );
    return sad;  // sum of |real[i] - predicted[i]| over 4 bytes
}

// ============================================================================
//  Surprise gate kernel — one warp (32 threads) per attention window
// ============================================================================

__global__ void surprise_gate_kernel(
    const float* __restrict__ d_predicted,      // [4 * attention_windows]
    const float* __restrict__ d_real,           // [4 * attention_windows]
    float*       __restrict__ d_td_error_gated, // [attention_windows]
    float*       __restrict__ d_neuromodulators, // [3 * attention_windows] = dopamine, serotonin, noradrenaline
    std::uint32_t attention_windows,
    float        surprise_threshold
) {
    const int wid = blockIdx.x;
    if (wid >= attention_windows) return;
    const int tid = threadIdx.x;

    // First 4 lanes read the 4 channels of the attention window
    float real_f = 0.0f;
    float predicted_f = 0.0f;
    if (tid < 4) {
        const int base = wid * 4;
        real_f      = d_real[base + tid];
        predicted_f = d_predicted[base + tid];
    }

    const std::uint32_t real_byte = (tid < 4)
        ? static_cast<std::uint32_t>(__saturatef(real_f) * 255.0f) : 0u;
    const std::uint32_t pred_byte = (tid < 4)
        ? static_cast<std::uint32_t>(__saturatef(predicted_f) * 255.0f) : 0u;

    // Pack all 4 channels across the warp via __shfl_sync
    std::uint32_t real_packed = __shfl_sync(0xFFFFFFFFu, real_byte, 0);
    real_packed |= __shfl_sync(0xFFFFFFFFu, real_byte, 1) << 8;
    real_packed |= __shfl_sync(0xFFFFFFFFu, real_byte, 2) << 16;
    real_packed |= __shfl_sync(0xFFFFFFFFu, real_byte, 3) << 24;

    std::uint32_t pred_packed = __shfl_sync(0xFFFFFFFFu, pred_byte, 0);
    pred_packed |= __shfl_sync(0xFFFFFFFFu, pred_byte, 1) << 8;
    pred_packed |= __shfl_sync(0xFFFFFFFFu, pred_byte, 2) << 16;
    pred_packed |= __shfl_sync(0xFFFFFFFFu, pred_byte, 3) << 24;

    if (tid == 0) {
        const std::uint32_t sad = ptx_surprise_metric(real_packed, pred_packed);
        const float surprise_norm = static_cast<float>(sad) / (4.0f * 255.0f);

        // Dopamine = surprise (unexpected stimulus / prediction error)
        d_neuromodulators[wid * 3 + 0] = surprise_norm;
        // Serotonin = 1 - surprise (stability when prediction matches reality)
        d_neuromodulators[wid * 3 + 1] = 1.0f - surprise_norm;
        // Noradrenaline = sqrt(surprise) (vigilance / attention boost)
        d_neuromodulators[wid * 3 + 2] = sqrtf(surprise_norm + 1e-8f);

        // Gate TD-error
        const float original_td = d_td_error_gated[wid];
        d_td_error_gated[wid] = (surprise_norm >= surprise_threshold) ? original_td : 0.0f;
    }
}

// ============================================================================
//  Neuromodulation broadcast kernel — vectorized elementwise scaling
// ============================================================================

__global__ void neuromodulation_broadcast_kernel(
    float*       __restrict__ d_alphas,         // [total_synapses]
    const float* __restrict__ d_neuromod,       // [3] = [dopamine, serotonin, noradrenaline]
    float        gain_dopamine,
    float        gain_noradrenaline,
    int          total_synapses
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_synapses) return;

    const float scalar = d_neuromod[0] * gain_dopamine + d_neuromod[2] * gain_noradrenaline;
    const float current_alpha = d_alphas[idx];
    const float new_alpha = current_alpha * (1.0f + scalar);
    d_alphas[idx] = fminf(fmaxf(new_alpha, 0.0f), 0.10f);
}

// ============================================================================
//  Host-side launchers (extern "C")
// ============================================================================

extern "C" {

void launch_surprise_gate(
    const float* d_predicted,
    const float* d_real,
    float*       d_td_error_gated,
    float*       d_neuromodulators,
    std::uint32_t attention_windows,
    float        surprise_threshold,
    cudaStream_t stream
) {
    if (attention_windows == 0) return;
    surprise_gate_kernel<<<attention_windows, 32, 0, stream>>>(
        d_predicted, d_real, d_td_error_gated, d_neuromodulators,
        attention_windows, surprise_threshold
    );
}

void launch_neuromodulation_broadcast(
    float*       d_alphas,
    const float* d_neuromod,
    float        gain_dopamine,
    float        gain_noradrenaline,
    int          total_synapses,
    cudaStream_t stream
) {
    if (total_synapses <= 0) return;
    constexpr int THREADS = 256;
    const int blocks = (total_synapses + THREADS - 1) / THREADS;
    neuromodulation_broadcast_kernel<<<blocks, THREADS, 0, stream>>>(
        d_alphas, d_neuromod, gain_dopamine, gain_noradrenaline, total_synapses
    );
}

}  // extern "C"