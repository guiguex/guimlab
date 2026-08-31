// ============================================================================
//  guim_studio.h — Next-Gen C++20 Neuromorphic Cockpit & Monitoring Studio
//  ============================================================================
//  High-frequency visual telemetry and interactive control harness:
//    - Lock-Free SHM Ingestion (POSIX / Win32 Shared Memory)
//    - Lovelace Cortex 256-Neuron Dynamic Thermal Heatmap & CBP Dead Units
//    - Symplectic Phase-Space Orbit (E_ij, P_ij) & Kuramoto Harmonic Alignment
//    - 16 kHz Real-Time Acoustic Oscilloscope & DDWR Sparse Delta Scope
//    - Per-Synapse TMD-ET / IDBD Learning Rate Dial & Attention Barometer
//    - Hardware Cycle-Accurate Latency Profiler & VRAM Leak Sentinel
//
//  Author  : GuimLab Core Team
//  License : AGPL-3.0 / MIT
// ============================================================================

#ifndef GUIM_STUDIO_H
#define GUIM_STUDIO_H

#include <cstdint>
#include <cstddef>
#include <array>
#include <vector>
#include <string>
#include <span>

#include "ipc_structs_v2.h"
#include "ipc_structs_sparse.h"

namespace guim::studio {

// ----------------------------------------------------------------------------
//  Display & Buffer Constants
// ----------------------------------------------------------------------------
constexpr std::size_t AUDIO_HISTORY_SAMPLES   = 4096;
constexpr std::size_t LATENCY_HISTORY_FRAMES  = 1024;
constexpr std::size_t TRACE_ORBIT_POINTS      = 512;
constexpr int         CORTEX_GRID_ROWS        = 16;
constexpr int         CORTEX_GRID_COLS        = 16; // 16x16 = 256 neurons

// ----------------------------------------------------------------------------
//  Circular Buffer (Zero-Allocation on Update)
// ----------------------------------------------------------------------------
template <typename T, std::size_t Capacity>
class CircularBuffer {
public:
    constexpr CircularBuffer() : head_(0), count_(0) {
        data_.fill(T{});
    }

    void push(T val) noexcept {
        data_[head_] = val;
        head_ = (head_ + 1) % Capacity;
        if (count_ < Capacity) {
            ++count_;
        }
    }

    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return Capacity; }
    [[nodiscard]] std::size_t head() const noexcept { return head_; }

    [[nodiscard]] T operator[](std::size_t idx) const noexcept {
        if (idx >= count_) return T{};
        std::size_t real_idx = (head_ + Capacity - count_ + idx) % Capacity;
        return data_[real_idx];
    }

    [[nodiscard]] const T* data() const noexcept { return data_.data(); }

private:
    std::array<T, Capacity> data_{};
    std::size_t head_{0};
    std::size_t count_{0};
};

// ----------------------------------------------------------------------------
//  Studio Telemetry Snapshot
// ----------------------------------------------------------------------------
struct StudioTelemetry {
    // Acoustic Waveform
    CircularBuffer<float, AUDIO_HISTORY_SAMPLES> audio_waveform;
    
    // Cortex Neuron Activations & Variances
    std::array<float, GUIM_STATE_DIM> cortex_activations{};
    std::array<float, GUIM_STATE_DIM> cortex_variances{};
    std::array<bool,  GUIM_STATE_DIM> cortex_dead_units{};

    // Symplectic Orbit (Sample synapse 0)
    CircularBuffer<float, TRACE_ORBIT_POINTS> symplectic_e;
    CircularBuffer<float, TRACE_ORBIT_POINTS> symplectic_p;

    // Reflex Motors
    std::array<float, GUIM_REFLEX_MOTORS> reflex_motors{};

    // Neuromodulators & Reward
    float dopamine{0.0f};
    float serotonin{0.0f};

    // Latency & Frame Stats
    CircularBuffer<float, LATENCY_HISTORY_FRAMES> latency_us;
    std::uint64_t total_frames{0};
    std::uint64_t last_timestamp_ns{0};
    float         fps{0.0f};
    float         p50_latency_us{0.0f};
    float         p99_latency_us{0.0f};
    float         ddwr_sparse_ratio{0.0f};
    std::size_t   vram_resident_bytes{0};

    // Status
    bool connected{false};
    bool paused{false};
};

// ----------------------------------------------------------------------------
//  IPC Shared Memory Client Interface
// ----------------------------------------------------------------------------
class ShmClient {
public:
    ShmClient();
    ~ShmClient();

    ShmClient(const ShmClient&) = delete;
    ShmClient& operator=(const ShmClient&) = delete;

    bool connect(const char* shm_name = "/guim_ipc_v2");
    void disconnect();
    bool is_connected() const noexcept { return is_connected_; }

    // Lock-free poll (returns true if fresh frame consumed)
    bool poll(StudioTelemetry& telemetry);

    // Optional bidirectional neuromodulator injection (CPU -> GPU)
    void inject_neuromodulators(float dopamine, float serotonin);
    void trigger_termination();

private:
    void* handle_{nullptr};
    GuimSharedMemoryV2* shm_ptr_{nullptr};
    std::uint64_t last_seq_out_{0};
    bool is_connected_{false};
};

// ----------------------------------------------------------------------------
//  Studio UI Theme & Styling
// ----------------------------------------------------------------------------
void apply_deeptech_theme();

// ----------------------------------------------------------------------------
//  Panel Renderers
// ----------------------------------------------------------------------------
void render_cortex_thermal_map(StudioTelemetry& telemetry);
void render_symplectic_phase_portrait(StudioTelemetry& telemetry);
void render_audio_oscilloscope(StudioTelemetry& telemetry);
void render_metagradients_inspector(StudioTelemetry& telemetry);
void render_telemetry_cockpit(StudioTelemetry& telemetry, ShmClient& client);

} // namespace guim::studio

#endif // GUIM_STUDIO_H
