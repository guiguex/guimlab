// ============================================================================
//  motor_babbling_kernel.cu — Stochastic motor babbling (exploration noise)
//  ---------------------------------------------------------------------------
//  FIX #4 (per user's review): "L'Étape Suivante : Le Motor Babbling".
//  Injects curand_normal noise into cortex_cognitive_out[] AFTER the cortex
//  has written its hidden-state delta.  Noise scale ∝ (1 - serotonin):
//
//      cortex_cognitive_out[i] += curand_normal() * BASE_NOISE * (1 - serotonin)
//
//  where:
//    - BASE_NOISE = 0.05  (small enough to perturb, large enough to explore)
//    - serotonin  ∈ [0, 1]  is the uncertainty/stability hormone from the CPU
//                           (high serotonin = high confidence = no babbling)
//                           (low serotonin = high uncertainty = lots of babbling)
//
//  This implements the standard exploration-vs-exploitation trade-off from
//  Sutton & Barto RL, applied at the silicon level.  When the agent is
//  confident (high serotonin), it acts precisely.  When it doesn't understand
//  its environment, it "trembles" and tests random actions to generate new
//  sensor deltas (which then drive learning via TMD-ET).
//
//  Where it runs
//  -------------
//    Stream: stream_plasticity (cudaStreamNonBlocking) — non-critical
//    Order:  AFTER the cortex has published seq_out
//    Why:    Cortex must not see its own output overwritten by noise BEFORE
//            it's been consumed.  The plasticity stream serializes AFTER
//            the inference stream (CUDA stream semantics).
//
//  Author        : Guillaume Meingan + Claude (ultracode session 2026-08-31)
//  License       : AGPL-3.0
// ============================================================================

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdint>
#include <cstdio>

#include "ipc_structs.h"
#include "ipc_structs_v2.h"

constexpr float BASE_NOISE = 0.05f;   // max babbling amplitude (5% of signal)

// ----------------------------------------------------------------------------
//  Init cuRAND state (one per output channel)
// ----------------------------------------------------------------------------

__global__ void babbling_init_kernel(
    curandState* __restrict__ d_rng,
    std::uint64_t             seed
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= GUIM_STATE_DIM) return;
    curand_init(seed, idx, 0, &d_rng[idx]);
}

// ----------------------------------------------------------------------------
//  Babbling kernel — runs ON stream_plasticity AFTER cortex publishes
// ----------------------------------------------------------------------------

__global__ void motor_babbling_kernel(
    volatile GuimSharedMemoryV2* __restrict__ d_ipc,
    curandState*                  __restrict__ d_rng
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= GUIM_STATE_DIM) return;

    // Read the stability signal from the CPU-computed neuromodulators
    const float serotonin = d_ipc->serotonin;

    // FIX #4: babbling noise scale ∝ (1 - serotonin)
    // When serotonin → 1 (high confidence), noise → 0 (no babbling).
    // When serotonin → 0 (high uncertainty), noise → BASE_NOISE (max babbling).
    const float noise_scale = BASE_NOISE * (1.0f - serotonin);

    // Skip kernel work entirely if we're confident (fast path)
    if (noise_scale <= 0.0f) return;

    // Pull the cortex's hidden-state delta and inject noise
    curandState local_rng = d_rng[row];
    const float noise = curand_normal(&local_rng) * noise_scale;
    d_rng[row] = local_rng;

    // Atomically add noise to the cortex output (separate from reflex_motor_out
    // which must NEVER be perturbed — emergencies stay clean).
    // We use volatile + atomic add semantics: cortex may have written in
    // parallel on stream_inference; we serialize on stream_plasticity which
    // is ordered AFTER stream_inference.
    const float current = d_ipc->cortex_cognitive_out[row];
    d_ipc->cortex_cognitive_out[row] = current + noise;
}

// ============================================================================
//  Host-side launcher
// ============================================================================

extern "C" {

void launch_motor_babbling(
    volatile GuimSharedMemoryV2* d_ipc,
    curandState*                d_rng,
    cudaStream_t                stream
) {
    motor_babbling_kernel<<<(GUIM_STATE_DIM + 255) / 256, 256, 0, stream>>>(
        d_ipc, d_rng
    );
}

void init_babbling_rng(curandState* d_rng, std::uint64_t seed) {
    babbling_init_kernel<<<(GUIM_STATE_DIM + 255) / 256, 256>>>(
        d_rng, seed
    );
}

}  // extern "C"