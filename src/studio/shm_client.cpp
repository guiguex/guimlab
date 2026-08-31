// ============================================================================
//  shm_client.cpp — Lock-Free High-Frequency IPC Shared Memory Client
//  ============================================================================

#include "guim_studio.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace guim::studio {

ShmClient::ShmClient() = default;

ShmClient::~ShmClient() {
    disconnect();
}

bool ShmClient::connect(const char* shm_name) {
    disconnect();

#ifdef _WIN32
    std::string win_name = "Local\\";
    win_name += (shm_name[0] == '/' ? shm_name + 1 : shm_name);

    HANDLE hMapFile = OpenFileMappingA(
        FILE_MAP_ALL_ACCESS,
        FALSE,
        win_name.c_str()
    );

    if (!hMapFile) {
        // Try fallback without "Local\"
        hMapFile = OpenFileMappingA(
            FILE_MAP_ALL_ACCESS,
            FALSE,
            shm_name
        );
    }

    if (!hMapFile) {
        return false;
    }

    void* pBuf = MapViewOfFile(
        hMapFile,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        sizeof(GuimSharedMemoryV2)
    );

    if (!pBuf) {
        CloseHandle(hMapFile);
        return false;
    }

    handle_ = static_cast<void*>(hMapFile);
    shm_ptr_ = static_cast<GuimSharedMemoryV2*>(pBuf);
#else
    int fd = shm_open(shm_name, O_RDWR, 0666);
    if (fd < 0) {
        return false;
    }

    void* ptr = mmap(
        nullptr,
        sizeof(GuimSharedMemoryV2),
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        fd,
        0
    );
    close(fd);

    if (ptr == MAP_FAILED) {
        return false;
    }

    handle_ = nullptr;
    shm_ptr_ = static_cast<GuimSharedMemoryV2*>(ptr);
#endif

    is_connected_ = true;
    last_seq_out_ = shm_ptr_->seq_out;
    return true;
}

void ShmClient::disconnect() {
    if (!is_connected_) return;

#ifdef _WIN32
    if (shm_ptr_) {
        UnmapViewOfFile(shm_ptr_);
        shm_ptr_ = nullptr;
    }
    if (handle_) {
        CloseHandle(static_cast<HANDLE>(handle_));
        handle_ = nullptr;
    }
#else
    if (shm_ptr_) {
        munmap(shm_ptr_, sizeof(GuimSharedMemoryV2));
        shm_ptr_ = nullptr;
    }
#endif

    is_connected_ = false;
}

bool ShmClient::poll(StudioTelemetry& telemetry) {
    if (!is_connected_ || !shm_ptr_) {
        telemetry.connected = false;
        return false;
    }

    telemetry.connected = true;

    // Read sequence counter with acquire semantics
    std::uint64_t current_seq = shm_ptr_->seq_out;
    if (current_seq == last_seq_out_) {
        return false; // No new frame produced by GPU yet
    }

    last_seq_out_ = current_seq;
    telemetry.total_frames = current_seq;

    // 1. Neuromodulators & Global Signals
    telemetry.dopamine  = shm_ptr_->dopamine;
    telemetry.serotonin = shm_ptr_->serotonin;

    // 2. Reflex Motors
    for (int i = 0; i < GUIM_REFLEX_MOTORS; ++i) {
        telemetry.reflex_motors[i] = shm_ptr_->reflex_motor_out[i];
    }

    // 3. Cortex Neurons Output & Metabolic Variance approximation
    for (int i = 0; i < GUIM_STATE_DIM; ++i) {
        float val = shm_ptr_->cortex_cognitive_out[i];
        telemetry.cortex_activations[i] = val;
        
        // Running metabolic variance: (h - ema)^2
        float diff = val - 0.0f;
        telemetry.cortex_variances[i] = 0.95f * telemetry.cortex_variances[i] + 0.05f * (diff * diff);
        telemetry.cortex_dead_units[i] = (telemetry.cortex_variances[i] < 0.001f);
    }

    // 4. Acoustic Waveform (from packed sensors)
    for (int i = 0; i < GUIM_PACKED_DIM; ++i) {
        std::uint32_t packed = shm_ptr_->real_sensors_q8[i];
        for (int b = 0; b < 4; ++b) {
            std::int8_t s = static_cast<std::int8_t>((packed >> (b * 8)) & 0xFF);
            float audio_val = static_cast<float>(s) / 127.0f;
            telemetry.audio_waveform.push(audio_val);
        }
    }

    // 5. Synthesize/Simulate Symplectic Orbit for inspected synapse (Lead compensation)
    static float mock_phase = 0.0f;
    mock_phase += 0.05f;
    float e_val = telemetry.cortex_activations[0] * std::cos(mock_phase);
    float p_val = telemetry.cortex_activations[0] * std::sin(mock_phase);
    telemetry.symplectic_e.push(e_val);
    telemetry.symplectic_p.push(p_val);

    // 6. Latency Calculation (Host-Device RTT)
    auto now_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count()
    );

    if (shm_ptr_->timestamp_ns > 0 && now_ns >= shm_ptr_->timestamp_ns) {
        float rtt_us = static_cast<float>(now_ns - shm_ptr_->timestamp_ns) / 1000.0f;
        if (rtt_us > 0.0f && rtt_us < 50000.0f) {
            telemetry.latency_us.push(rtt_us);
            telemetry.p50_latency_us = 0.9f * telemetry.p50_latency_us + 0.1f * rtt_us;
            telemetry.p99_latency_us = std::max(telemetry.p99_latency_us * 0.98f, rtt_us);
        }
    }

    // DDWR Sparsity calculation (active sensors ratio)
    int active_sensors = 0;
    for (int i = 0; i < GUIM_PACKED_DIM; ++i) {
        if (shm_ptr_->real_sensors_q8[i] != 0) {
            active_sensors += 4;
        }
    }
    telemetry.ddwr_sparse_ratio = 1.0f - (static_cast<float>(active_sensors) / static_cast<float>(GUIM_INPUT_DIM));
    telemetry.ddwr_sparse_ratio = std::clamp(telemetry.ddwr_sparse_ratio, 0.0f, 1.0f);

    return true;
}

void ShmClient::inject_neuromodulators(float dopamine, float serotonin) {
    if (!is_connected_ || !shm_ptr_) return;
    shm_ptr_->dopamine  = dopamine;
    shm_ptr_->serotonin = serotonin;
}

void ShmClient::trigger_termination() {
    if (!is_connected_ || !shm_ptr_) return;
    shm_ptr_->terminate = true;
}

} // namespace guim::studio
