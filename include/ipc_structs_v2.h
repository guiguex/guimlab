// ============================================================================
//  ipc_structs_v2.h — Guimlab IPC layout (INT8 packed, cache-aligned)
//  ---------------------------------------------------------------------------
//  Author        : Guimlab Contributors
//  License       : AGPL-3.0
// ============================================================================

#ifndef GUIM_IPC_STRUCTS_V2_H
#define GUIM_IPC_STRUCTS_V2_H

#include <cstdint>
#include "ipc_structs.h"

// ----------------------------------------------------------------------------
//  Architecture constants (must match kernel expectations)
// ----------------------------------------------------------------------------

#ifndef GUIM_PACKED_DIM
constexpr int GUIM_PACKED_DIM = GUIM_INPUT_DIM / 4;   // 4 sensors per uint32
#endif

#ifndef GUIM_REFLEX_MOTORS
constexpr int GUIM_REFLEX_MOTORS = 16;     // reflex motor output channels
#endif

// ----------------------------------------------------------------------------
//  Shared structure v2
// ----------------------------------------------------------------------------

struct alignas(64) GuimSharedMemoryV2 {
    // ---- CHANNEL 1: SENSORS (CPU -> GPU) ----
    alignas(64) std::uint32_t real_sensors_q8[GUIM_PACKED_DIM];
    alignas(64) std::uint32_t pred_sensors_q8[GUIM_PACKED_DIM];

    // ---- CHANNEL 2: NEUROMODULATORS (CPU -> GPU) ----
    alignas(64) float dopamine;
    alignas(64) float serotonin;

    // ---- CHANNEL 3: METADATA (CPU -> GPU) ----
    alignas(64) std::uint64_t timestamp_ns;
    alignas(64) volatile std::uint64_t seq_in;

    // ---- CHANNEL 4: MOTOR REFLEX (GPU Reflex -> CPU) ----
    alignas(64) float reflex_motor_out[GUIM_REFLEX_MOTORS];

    // ---- CHANNEL 5: CORTEX COGNITIVE (GPU Cortex -> CPU) ----
    alignas(64) float cortex_cognitive_out[GUIM_STATE_DIM];

    // ---- CHANNEL 6: GPU SEQUENCE (GPU -> CPU) ----
    alignas(64) volatile std::uint64_t seq_out;

    // ---- CONTROL ----
    alignas(64) volatile bool terminate;
};

#endif // GUIM_IPC_STRUCTS_V2_H