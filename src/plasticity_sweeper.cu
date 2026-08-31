// ============================================================================
//  plasticity_sweeper.cu — Asynchronous synaptic garbage collector
//  ---------------------------------------------------------------------------
//  Runs on a separate non-blocking CUDA stream to gently re-randomize synapses
//  that have remained dormant past the death threshold.
//
//  Author        : Guimlab Contributors
//  License       : AGPL-3.0
// ============================================================================

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdint>
#include <cstdio>
#include <thread>
#include <atomic>
#include <chrono>

#include "ipc_structs.h"

constexpr std::uint64_t DEATH_THRESHOLD_NS = 3'600'000'000'000ULL;  // 1 hour
constexpr float INJECTED_NOISE_AMP = 0.02f;

// ----------------------------------------------------------------------------
//  Plasticity sweeper kernel (float version)
// ----------------------------------------------------------------------------

__global__ void plasticity_sweeper_kernel(
    float*         __restrict__ d_weights,
    std::uint64_t* __restrict__ d_last_t,
    curandState*   __restrict__ d_rng,
    std::uint64_t               current_timestamp
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_synapses = GUIM_STATE_DIM * (GUIM_INPUT_DIM + GUIM_STATE_DIM);
    if (idx >= total_synapses) return;

    const std::uint64_t time_since = current_timestamp - d_last_t[idx];
    if (time_since > DEATH_THRESHOLD_NS) {
        curandState local_rng = d_rng[idx];
        const float u1 = curand_uniform(&local_rng) - 0.5f;
        d_weights[idx] += u1 * INJECTED_NOISE_AMP;
        d_last_t[idx] = current_timestamp;
        d_rng[idx]    = local_rng;
    }
}

// ----------------------------------------------------------------------------
//  cuRAND state init
// ----------------------------------------------------------------------------

__global__ void init_sweeper_rng_kernel(
    curandState* __restrict__ d_rng,
    std::uint64_t             seed
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= GUIM_STATE_DIM * (GUIM_INPUT_DIM + GUIM_STATE_DIM)) return;
    curand_init(seed, idx, 0, &d_rng[idx]);
}

// ----------------------------------------------------------------------------
//  Host-side background worker entry
// ----------------------------------------------------------------------------

extern "C" void plasticity_sweeper_thread_entry(
    float*             d_weights,
    std::uint64_t*     d_last_t,
    curandState*       d_rng,
    std::atomic<bool>* stop_flag,
    cudaStream_t       stream
) {
    constexpr int THREADS = 256;
    const int total_synapses = GUIM_STATE_DIM * (GUIM_INPUT_DIM + GUIM_STATE_DIM);
    const int blocks = (total_synapses + THREADS - 1) / THREADS;

    while (!stop_flag->load(std::memory_order_acquire)) {
        const auto now = std::chrono::steady_clock::now();
        const std::uint64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()
        ).count();

        plasticity_sweeper_kernel<<<blocks, THREADS, 0, stream>>>(
            d_weights, d_last_t, d_rng, now_ns
        );

        // Sleep in small increments to allow rapid shutdown
        for (int i = 0; i < 100 && !stop_flag->load(std::memory_order_acquire); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}