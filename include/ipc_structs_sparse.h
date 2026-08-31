// ============================================================================
//  ipc_structs_sparse.h — Delta-Driven IPC layout (DDWR)
//  ---------------------------------------------------------------------------
//  Sparse variant of GuimSharedMemory: instead of carrying the full input
//  vector every frame, the CPU transmits ONLY the indices of sensors whose
//  value has changed by more than epsilon.  This is the Delta-Driven Warp
//  Routing (DDWR) protocol — 95% reduction in VRAM bandwidth.
//
//  Concurrency model
//  ----------------
//    CPU:
//      1. SIMD scan the input vector (8 floats at a time via AVX2).
//      2. For each chunk, compute mask = abs(new - old) > epsilon.
//      3. For each bit set in mask, write delta_indices[i] and delta_values[i].
//      4. Set delta_count.
//      5. Set timestamp_ns.
//      6. Increment seq_in (release).
//
//    GPU (sparse_persistent_guim_kernel):
//      1. Spin-wait on seq_in.
//      2. Read delta_count.  If 0, skip everything (the warp is dormant).
//      3. For each delta, accumulate input contribution.  Other threads in
//         the same warp whose |accum| < THRESHOLD are marked inactive via
//         __ballot_sync.  The whole warp can skip the trace/weight read.
//      4. Compute trace update using CLOSED-FORM continuous-time formula:
//           E(t) = E(t_last) * exp(-(t - t_last) / tau) + new_gradient
//      5. Update TMD-ET meta-trace + alpha + weight.
//      6. Write delta_output[row] = h_new - h_old (CPU only needs the change).
//      7. __threadfence_system + seq_out++.
//
//  Why this delta is
//  ---------------
//    *  Audio: 2% of samples change between adjacent frames at 16 kHz →
//       98% bandwidth saved.
//    *  Video: object detection only changes pixels around moving objects →
//       99%+ bandwidth saved.
//    *  Robotics: tactile sensors only fire on contact → 99.9% bandwidth saved.
//
//  Capacity
//  --------
//    delta_indices/delta_values hold MAX_DELTAS = 64 indices/values.  With
//    GUIM_INPUT_DIM = 128, this means at most 50% of the input vector can
//    change per frame.  For higher-change workloads, increase the buffer.
//
//  Author        : Guillaume Meingan + Claude (ultracode session 2026-08-31)
//  License       : AGPL-3.0
// ============================================================================

#ifndef GUIM_IPC_STRUCTS_SPARSE_H
#define GUIM_IPC_STRUCTS_SPARSE_H

#include <cstdint>

#include "ipc_structs.h"  // for GUIM_STATE_DIM, GUIM_INPUT_DIM

// ----------------------------------------------------------------------------
//  Capacity
// ----------------------------------------------------------------------------

constexpr int GUIM_MAX_DELTAS = 64;   // max deltas per frame (out of 128)

// ----------------------------------------------------------------------------
//  Sparse shared structure
// ----------------------------------------------------------------------------

struct alignas(64) GuimSharedMemorySparse {
    // ----------------------- CPU → GPU channel ------------------------------
    alignas(64) volatile std::uint64_t seq_in;          // monotonic release counter
    alignas(64) volatile std::uint64_t timestamp_ns;    // monotonic wall clock (ns)
    alignas(64) volatile int           delta_count;      // 0..GUIM_MAX_DELTAS
    alignas(64) volatile int           delta_indices[GUIM_MAX_DELTAS];
    alignas(64) volatile float          delta_values [GUIM_MAX_DELTAS];
    alignas(64) volatile float          td_error;

    // ----------------------- GPU → CPU channel ------------------------------
    alignas(64) volatile std::uint64_t seq_out;         // monotonic release counter
    alignas(64) volatile float          delta_output[GUIM_STATE_DIM];
    alignas(64) volatile int           active_count;     // # of warps that fired

    // ----------------------- Control ----------------------------------------
    alignas(64) volatile bool          terminate;
};

// Compile-time sanity
static_assert(sizeof(GuimSharedMemorySparse) < (32 * 1024 * 1024),
              "GuimSharedMemorySparse exceeds 32 MiB POSIX SHM default");

#endif  // GUIM_IPC_STRUCTS_SPARSE_H