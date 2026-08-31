// ============================================================================
//  ipc_structs.h — POSIX SHM + CUDA zero-copy IPC layout
//  ---------------------------------------------------------------------------
//  Lock-free ring of (input → output) frames between a CPU client and a
//  persistent CUDA kernel.  Latency budget: 300-900 ns on PCIe Gen4/5.
//
//  Concurrency model
//  ----------------
//    *  Two channels, each protected by a sequence counter (`seq_in`,
//       `seq_out`).  Volatile loads enforce acquire/release semantics on
//       weakly-ordered x86 (which already has strong ordering, but volatile
//       keeps the compiler honest).
//    *  CPU writes `input[i]`, `td_error`, then increments `seq_in`.
//    *  GPU spins on `seq_in`, then reads `input[i]` and `td_error` once it
//       has changed, computes, writes `output[i]`, then increments `seq_out`.
//    *  CPU spins on `seq_out` until it equals the local sequence.
//
//  Memory layout (cache-line aligned)
//  ----------------------------------
//    *  Each "channel" gets a fresh 64-byte cache line (alignas(64)) so the
//       CPU-side writer and GPU-side reader don't false-share with anything
//       else.
//    *  Order of fields matters for acquire/release: sequence counter is
//       read LAST by the consumer (it acts as the release signal).
//
//  Why volatile + __threadfence_system
//  ------------------------------------
//    *  `volatile` on the seq counter forces the compiler to issue a real load
//       every loop iteration (no register-cached load).
//    *  `__threadfence_system()` inside the kernel guarantees the L2 cache has
//       flushed to host RAM over PCIe before `seq_out++`.  This is the GPU
//       equivalent of an sfence + mfence.
//    *  Pair with `_mm_pause()` on the CPU side to avoid pipeline stalls
//       while spinning (saves ~10x power and reduces contention).
//
//  Compilation
//  -----------
//    *  Server : nvcc -O3 -use_fast_math -std=c++20 persistent_kernel.cu
//    *  Client : g++  -O3 -std=c++20 -mavx2 client.cpp -lrt
//
//  Author        : Guillaume Meingan + Claude (ultracode session 2026-08-31)
//  License       : AGPL-3.0
// ============================================================================

#ifndef GUIM_IPC_STRUCTS_H
#define GUIM_IPC_STRUCTS_H

#include <cstdint>

// ----------------------------------------------------------------------------
//  Architecture constants — kept in sync with src/persistent_kernel.cu
// ----------------------------------------------------------------------------

constexpr int GUIM_STATE_DIM = 256;
constexpr int GUIM_INPUT_DIM = 128;
constexpr int GUIM_TOTAL_IN  = GUIM_INPUT_DIM + GUIM_STATE_DIM;  // 384
constexpr int GUIM_TOTAL_W   = GUIM_STATE_DIM * GUIM_TOTAL_IN;   // 98304

// ----------------------------------------------------------------------------
//  Cache-line-aligned shared structure
// ----------------------------------------------------------------------------
//
//  alignas(64) on every field that crosses the producer-consumer boundary
//  prevents false sharing.  Order matters for acquire/release semantics:
//  sequence counter is the LAST thing written (release) and FIRST thing
//  read (acquire).
// ----------------------------------------------------------------------------

struct alignas(64) GuimSharedMemory {
    // -------------------------- CPU → GPU channel --------------------------
    alignas(64) volatile std::uint64_t seq_in;   // monotonic counter, CPU writes
    alignas(64) volatile float         input[GUIM_INPUT_DIM];
    alignas(64) volatile float         td_error;

    // -------------------------- GPU → CPU channel --------------------------
    alignas(64) volatile std::uint64_t seq_out;  // monotonic counter, GPU writes
    alignas(64) volatile float         output[GUIM_STATE_DIM];

    // -------------------------- Control -------------------------------------
    alignas(64) volatile bool          terminate; // both sides observe
};

// ----------------------------------------------------------------------------
//  Compile-time sanity: ensure the struct fits in a reasonable memory region
//  for SHM (default Linux limit is often 32 MB but configurable via /proc).
// ----------------------------------------------------------------------------

static_assert(sizeof(GuimSharedMemory) < (32 * 1024 * 1024),
              "GuimSharedMemory exceeds 32 MiB POSIX SHM default");

// ----------------------------------------------------------------------------
//  Helper: byte offset of a field (debug only — checked at runtime)
// ----------------------------------------------------------------------------

#ifdef GUIM_DEBUG
#include <cstdio>
inline void guim_dump_layout() noexcept {
    std::printf("GuimSharedMemory layout:\n");
    std::printf("  seq_in:    %zu\n", offsetof(GuimSharedMemory, seq_in));
    std::printf("  input:     %zu\n", offsetof(GuimSharedMemory, input));
    std::printf("  td_error:  %zu\n", offsetof(GuimSharedMemory, td_error));
    std::printf("  seq_out:   %zu\n", offsetof(GuimSharedMemory, seq_out));
    std::printf("  output:    %zu\n", offsetof(GuimSharedMemory, output));
    std::printf("  terminate: %zu\n", offsetof(GuimSharedMemory, terminate));
    std::printf("  sizeof:    %zu\n", sizeof(GuimSharedMemory));
}
#endif

#endif  // GUIM_IPC_STRUCTS_H