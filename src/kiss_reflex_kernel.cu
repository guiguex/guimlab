// ============================================================================
//  kiss_reflex_kernel.cu — Zero-Spill Register-Pinned L0 Reflex Kernel
//  ============================================================================
//  Author        : Guimlab Contributors
//  License       : AGPL-3.0 / MIT
// ============================================================================

#include <cuda_runtime.h>
#include <cstdint>

#include "ipc_structs.h"
#include "ipc_structs_v2.h"

constexpr int REFLEX_SENSORS = GUIM_INPUT_DIM;       // 128
constexpr int REFLEX_NEURONS = GUIM_REFLEX_MOTORS;   // 16

__device__ __forceinline__ unsigned long long read_globaltimer_reflex_ns() {
    unsigned long long t;
    #if __CUDA_ARCH__ >= 300
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    #else
    t = clock64();
    #endif
    return t;
}

__global__ void __launch_bounds__(32, 1)
kiss_reflex_persistent_kernel(
    volatile GuimSharedMemoryV2* __restrict__ d_ipc
) {
    const int tid = threadIdx.x;  // 32 threads in 1 warp

    // Coordinated shared memory for 16 motor neurons x 128 sensors
    __shared__ float s_weights[REFLEX_NEURONS][REFLEX_SENSORS];
    __shared__ float s_traces [REFLEX_NEURONS][REFLEX_SENSORS];
    __shared__ float s_hidden [REFLEX_NEURONS];

    // Parallel deterministic initialization (32 threads init 16x128 = 2048 synapses)
    for (int idx = tid; idx < REFLEX_NEURONS * REFLEX_SENSORS; idx += 32) {
        const int n = idx / REFLEX_SENSORS;
        const int s = idx % REFLEX_SENSORS;
        const float seed = static_cast<float>((n * 31 + s * 17) & 0xFF) / 255.0f - 0.5f;
        s_weights[n][s] = seed * 0.1f;
        s_traces [n][s] = 0.0f;
    }
    if (tid < REFLEX_NEURONS) {
        s_hidden[tid] = 0.0f;
    }
    __syncwarp();

    std::uint64_t local_seq = d_ipc->seq_in;
    constexpr unsigned long long WATCHDOG_TIMEOUT_NS = 2000000000ull;  // 2.0s

    while (!d_ipc->terminate) {
        // Spin-wait on seq_in with backoff + %globaltimer watchdog
        if (tid == 0) {
            const unsigned long long t_start = read_globaltimer_reflex_ns();
            while (d_ipc->seq_in <= local_seq && !d_ipc->terminate) {
                #if __CUDA_ARCH__ >= 700
                __nanosleep(100);
                #else
                asm volatile("nanosleep.u32 100;");
                #endif

                if (read_globaltimer_reflex_ns() - t_start > WATCHDOG_TIMEOUT_NS) {
                    d_ipc->terminate = true;
                    break;
                }
            }
            local_seq = d_ipc->seq_in;
        }
        __syncwarp();
        if (d_ipc->terminate) break;

        // Shared input unpacking
        __shared__ float s_sensors[REFLEX_SENSORS];
        for (int i = tid; i < REFLEX_SENSORS; i += 32) {
            const int pack_idx = i / 4;
            const int shift    = (i & 3) * 8;
            s_sensors[i] = static_cast<float>((d_ipc->real_sensors_q8[pack_idx] >> shift) & 0xFF) / 255.0f;
        }
        __syncwarp();

        // 16 neurons computed by first 16 threads (zero divergence)
        if (tid < REFLEX_NEURONS) {
            float accum = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < REFLEX_SENSORS; ++i) {
                accum += s_weights[tid][i] * s_sensors[i];
            }
            accum += s_hidden[tid] * 0.7f;
            s_hidden[tid] = accum;

            const float h_new = 1.0f / (1.0f + __expf(-accum));
            const float dsig  = h_new * (1.0f - h_new);
            const float td_error = d_ipc->dopamine;

            #pragma unroll 8
            for (int i = 0; i < REFLEX_SENSORS; ++i) {
                float tr = 0.891f * s_traces[tid][i] + dsig * s_sensors[i];
                s_traces[tid][i]   = tr;
                s_weights[tid][i] += 0.01f * td_error * tr;
            }

            d_ipc->reflex_motor_out[tid] = h_new;
        }

        __threadfence_system();
        __syncwarp();
        if (tid == 0) {
            d_ipc->seq_out = local_seq;
        }
    }
}