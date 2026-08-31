// =============================================================================
//  guim_kernel.h - Public C ABI for the Guimlab continual learning kernel.
//
//  Author        : Guimlab contributors
//  SPDX-License   : MIT
//
//  Source of truth for kernel ABI versioning.  All numeric constants
//  (STATE_DIM, INPUT_DIM, learning rates, ...) live here exactly once.  The
//  modern C++ wrapper <guim/guim.h> includes this header and re-exports the
//  constants in a typed manner.
//
//  This ABI must stay C-compatible (extern "C", no exceptions, no STL types)
//  so the kernel can be consumed from C, Rust, Julia, or any FFI layer.
// =============================================================================
#ifndef GUIM_KERNEL_H
#define GUIM_KERNEL_H

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
//  Architecture constants
// ---------------------------------------------------------------------------
//  STATE_DIM : recurrent hidden state size  (Javed CBP: 64-256 typical)
//  INPUT_DIM : per-frame input feature size (audio embedding dim)
//  TOTAL_DIM : concatenated input vector [x; h_prev]
//  TOTAL_W   : full weight matrix size     (STATE_DIM * TOTAL_DIM)
//
//  These are pinned at compile time so the weight array can live in
//  register-friendly SoA layouts on the device.  See docs/01-architecture.md
//  for the derivation from CBP and Sutton TD(λ).
// ---------------------------------------------------------------------------
#define GUIM_STATE_DIM   64
#define GUIM_INPUT_DIM   32
#define GUIM_TOTAL_DIM   (GUIM_STATE_DIM + GUIM_INPUT_DIM)  // 96
#define GUIM_TOTAL_W     (GUIM_STATE_DIM * GUIM_TOTAL_DIM)  // 6144

// Plasticity preservation (Javed CBP)
#define GUIM_PLASTICITY_WINDOW       256

// Learning dynamics (Sutton TD + RTRL flavour)
#define GUIM_ALPHA                   0.005f
#define GUIM_LAMBDA                  0.85f
#define GUIM_WEIGHT_DECAY            0.0001f
#define GUIM_PLASTICITY_THRESHOLD    0.001f

// Numerical safety
#define GUIM_WEIGHT_CLIP             5.0f
#define GUIM_TRACE_CLIP              100.0f

// ---------------------------------------------------------------------------
//  Opaque handles
// ---------------------------------------------------------------------------
typedef struct GuimKernel GuimKernel;   // GPU implementation
typedef struct GuimCpuRef GuimCpuRef;   // host reference (no CUDA)

// ---------------------------------------------------------------------------
//  Lifecycle (GPU)
// ---------------------------------------------------------------------------
GuimKernel* guim_create   (std::uint32_t seed);
void        guim_destroy  (GuimKernel* k);

// ---------------------------------------------------------------------------
//  Per-frame update
//
//  input  : [GUIM_INPUT_DIM]  caller-owned
//  output : [GUIM_STATE_DIM]  caller-owned, written by step
//
//  Returns the magnitude of the TD-error used this frame, or NaN if the
//  kernel detected a numerical fault (NaN/Inf in input or hidden state).
// ---------------------------------------------------------------------------
float       guim_step      (GuimKernel* k, const float* input, float* output);

// ---------------------------------------------------------------------------
//  Introspection
// ---------------------------------------------------------------------------
void        guim_get_hidden (const GuimKernel* k, float* dst);
void        guim_get_weights(const GuimKernel* k, float* dst);
std::uint32_t guim_resets_count(const GuimKernel* k);
std::uint64_t guim_frames_run  (const GuimKernel* k);
std::size_t   guim_bytes_resident(const GuimKernel* k);

// ---------------------------------------------------------------------------
//  CPU reference (no CUDA) — used by tests/test_correctness.cpp
//  Same algorithm as the GPU path; identical NaN-sentinel contract.
// ---------------------------------------------------------------------------
GuimCpuRef* guim_cpu_create   (std::uint32_t seed);
void        guim_cpu_destroy  (GuimCpuRef* r);
float       guim_cpu_step     (GuimCpuRef* r, const float* input, float* output);
void        guim_cpu_get_hidden (const GuimCpuRef* r, float* dst);
void        guim_cpu_get_weights(const GuimCpuRef* r, float* dst);
std::uint32_t guim_cpu_resets_count(const GuimCpuRef* r);

#ifdef __cplusplus
}
#endif

#endif // GUIM_KERNEL_H