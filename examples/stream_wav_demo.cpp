// ============================================================================
//  stream_wav_demo.cpp — Native C++20 Streaming Audio Ingestion & Live Neuromorphic Engine
//  ============================================================================
//  Demonstrates continuous-time real-world speech processing on GuimLab:
//    1. Parses standard 16 kHz RIFF PCM .wav files (zero third-party libraries).
//    2. Extracts 64-channel continuous spectral latent features via native DSP.
//    3. Streams frames in real-time through the GPU-accelerated neuromorphic core.
//    4. Optionally broadcasts to zero-copy shared memory IPC for live TUI visualization.
//
//  Author  : Guillaume Meingan <guillaume@guig.dev>
//  License : AGPL-3.0 / MIT
// ============================================================================

#include "guim/guim.h"
#include "ipc_structs_v2.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

// ANSI colors
#define ANSI_RESET   "\033[0m"
#define ANSI_BOLD    "\033[1m"
#define ANSI_CYAN    "\033[36m"
#define ANSI_GREEN   "\033[32m"
#define ANSI_YELLOW  "\033[33m"
#define ANSI_MAGENTA "\033[35m"
#define ANSI_RED     "\033[31m"
#define ANSI_DIM     "\033[2m"

namespace {

// ---- RIFF WAV Header Parsing -----------------------------------------------
struct WavHeader {
    std::uint32_t sample_rate = 16000;
    std::uint16_t num_channels = 1;
    std::uint16_t bits_per_sample = 16;
    std::vector<float> samples;
};

bool load_wav(const std::string& path, WavHeader& out_wav) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::fprintf(stderr, "Error: Unable to open WAV file: %s\n", path.c_str());
        return false;
    }

    char chunk_id[4];
    file.read(chunk_id, 4);
    if (std::memcmp(chunk_id, "RIFF", 4) != 0) {
        std::fprintf(stderr, "Error: Invalid RIFF header in %s\n", path.c_str());
        return false;
    }

    file.seekg(8, std::ios::beg); // Skip chunk size
    file.read(chunk_id, 4);
    if (std::memcmp(chunk_id, "WAVE", 4) != 0) {
        std::fprintf(stderr, "Error: Invalid WAVE format in %s\n", path.c_str());
        return false;
    }

    std::uint32_t sample_rate = 16000;
    std::uint16_t num_channels = 1;
    std::uint16_t bits_per_sample = 16;
    bool found_fmt = false;
    bool found_data = false;

    while (file.read(chunk_id, 4)) {
        std::uint32_t chunk_size = 0;
        file.read(reinterpret_cast<char*>(&chunk_size), 4);

        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            std::uint16_t audio_format = 0;
            file.read(reinterpret_cast<char*>(&audio_format), 2);
            file.read(reinterpret_cast<char*>(&num_channels), 2);
            file.read(reinterpret_cast<char*>(&sample_rate), 4);
            file.seekg(6, std::ios::cur); // skip byte rate and block align
            file.read(reinterpret_cast<char*>(&bits_per_sample), 2);

            if (chunk_size > 16) {
                file.seekg(chunk_size - 16, std::ios::cur);
            }
            found_fmt = true;
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            if (!found_fmt) {
                std::fprintf(stderr, "Error: data chunk before fmt chunk\n");
                return false;
            }

            size_t num_samples = chunk_size / (bits_per_sample / 8);
            out_wav.samples.resize(num_samples / num_channels);

            if (bits_per_sample == 16) {
                std::vector<std::int16_t> raw_pcm(num_samples);
                file.read(reinterpret_cast<char*>(raw_pcm.data()), chunk_size);
                for (size_t i = 0; i < out_wav.samples.size(); ++i) {
                    float sum = 0.0f;
                    for (int c = 0; c < num_channels; ++c) {
                        sum += static_cast<float>(raw_pcm[i * num_channels + c]) / 32768.0f;
                    }
                    out_wav.samples[i] = sum / static_cast<float>(num_channels);
                }
            } else if (bits_per_sample == 32) {
                std::vector<float> raw_pcm(num_samples);
                file.read(reinterpret_cast<char*>(raw_pcm.data()), chunk_size);
                for (size_t i = 0; i < out_wav.samples.size(); ++i) {
                    float sum = 0.0f;
                    for (int c = 0; c < num_channels; ++c) {
                        sum += raw_pcm[i * num_channels + c];
                    }
                    out_wav.samples[i] = sum / static_cast<float>(num_channels);
                }
            } else {
                std::fprintf(stderr, "Error: Unsupported bit depth %u\n", bits_per_sample);
                return false;
            }
            found_data = true;
            break;
        } else {
            // Skip unknown chunk
            file.seekg(chunk_size, std::ios::cur);
        }
    }

    out_wav.sample_rate = sample_rate;
    out_wav.num_channels = num_channels;
    out_wav.bits_per_sample = bits_per_sample;
    return found_data;
}

// ---- Native Real-Time Spectral Latent Extractor (D_in = 64) ----------------
class AcousticFeatureExtractor {
public:
    static constexpr int kWindowSize = 512;  // 32 ms @ 16 kHz
    static constexpr int kHopSize    = 160;  // 10 ms hop (100 Hz frame rate)
    static constexpr int kNumBands   = GUIM_INPUT_DIM; // 64 filter channels

    AcousticFeatureExtractor() {
        // Pre-compute periodic Hann window
        window_.resize(kWindowSize);
        for (int i = 0; i < kWindowSize; ++i) {
            window_[i] = 0.5f * (1.0f - std::cos(2.0f * 3.14159265358979323846f * i / kWindowSize));
        }

        // Initialize triangular filterbank frequencies (100 Hz to 7500 Hz)
        band_centers_.resize(kNumBands);
        for (int i = 0; i < kNumBands; ++i) {
            float mel = 100.0f + static_cast<float>(i) * (7400.0f / static_cast<float>(kNumBands));
            band_centers_[i] = mel;
        }
    }

    void extract_frame(const float* pcm_window, float* out_features) const {
        // Apply window and compute localized spectral energy bands
        for (int b = 0; b < kNumBands; ++b) {
            float real_sum = 0.0f;
            float imag_sum = 0.0f;
            float freq_rad = 2.0f * 3.14159265358979323846f * band_centers_[b] / 16000.0f;

            for (int n = 0; n < kWindowSize; n += 2) { // 2x stride for ultra-fast DSP
                float sample = pcm_window[n] * window_[n];
                float angle = freq_rad * n;
                real_sum += sample * std::cos(angle);
                imag_sum -= sample * std::sin(angle);
            }

            float power = std::sqrt(real_sum * real_sum + imag_sum * imag_sum + 1e-6f);
            // Log-energy compression normalized to [-1.0, 1.0]
            float log_energy = std::log10(1.0f + 10.0f * power);
            out_features[b] = std::clamp(log_energy * 2.0f - 1.0f, -1.0f, 1.0f);
        }
    }

private:
    std::vector<float> window_;
    std::vector<float> band_centers_;
};

} // namespace

int main(int argc, char** argv) {
    std::string wav_path = "data/speech_16k.wav";
    bool use_ipc = false;
    bool loop_mode = false;
    float playback_speed = 1.0f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--ipc") {
            use_ipc = true;
        } else if (arg == "--loop") {
            loop_mode = true;
        } else if (arg == "--speed" && i + 1 < argc) {
            playback_speed = std::stof(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: stream_wav_demo [path_to_wav] [--ipc] [--loop] [--speed <float>]\n");
            std::printf("  --ipc    : Broadcast live audio and cortex activations to /guim_ipc_v3_shm for guim_monitor\n");
            std::printf("  --loop   : Continuously loop the audio stream\n");
            std::printf("  --speed  : Playback rate multiplier (default: 1.0 = real-time)\n");
            return 0;
        } else if (arg[0] != '-') {
            wav_path = arg;
        }
    }

    std::printf(ANSI_BOLD ANSI_CYAN "======================================================================\n" ANSI_RESET);
    std::printf(ANSI_BOLD ANSI_CYAN "   GUIMLAB :: REAL-TIME SPEECH STREAMING & NEUROMORPHIC INGESTION      \n" ANSI_RESET);
    std::printf(ANSI_BOLD ANSI_CYAN "======================================================================\n\n" ANSI_RESET);

    WavHeader wav;
    if (!load_wav(wav_path, wav)) {
        std::fprintf(stderr, "Failed to load audio from '%s'.\n", wav_path.c_str());
        return 1;
    }

    double duration_sec = static_cast<double>(wav.samples.size()) / wav.sample_rate;
    std::printf("  " ANSI_BOLD "Audio File    :" ANSI_RESET " %s\n", wav_path.c_str());
    std::printf("  " ANSI_BOLD "Sample Rate   :" ANSI_RESET " %u Hz\n", wav.sample_rate);
    std::printf("  " ANSI_BOLD "Channels      :" ANSI_RESET " %u (Mono Downmixed)\n", wav.num_channels);
    std::printf("  " ANSI_BOLD "Bit Depth     :" ANSI_RESET " %u-bit\n", wav.bits_per_sample);
    std::printf("  " ANSI_BOLD "Duration      :" ANSI_RESET " %.2f seconds (%zu samples)\n", duration_sec, wav.samples.size());
    std::printf("  " ANSI_BOLD "IPC Mode      :" ANSI_RESET " %s\n\n", use_ipc ? ANSI_GREEN "ACTIVE (/guim_ipc_v3_shm)" ANSI_RESET : ANSI_YELLOW "STANDALONE IN-PROCESS" ANSI_RESET);

    // Initialize IPC if requested
    volatile GuimSharedMemoryV2* ipc = nullptr;
#ifdef _WIN32
    HANDLE hMapFile = nullptr;
    if (use_ipc) {
        hMapFile = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, "Local\\guim_ipc_v3_shm");
        if (hMapFile) {
            ipc = reinterpret_cast<volatile GuimSharedMemoryV2*>(MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(GuimSharedMemoryV2)));
        }
        if (!ipc) {
            std::printf(ANSI_YELLOW "  [Note: Shared memory not found; running with local engine]\n" ANSI_RESET);
        }
    }
#else
    int shm_fd = -1;
    if (use_ipc) {
        shm_fd = ::shm_open("/guim_ipc_v3_shm", O_RDWR, 0666);
        if (shm_fd >= 0) {
            void* mapped = ::mmap(nullptr, sizeof(GuimSharedMemoryV2), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
            if (mapped != MAP_FAILED) {
                ipc = reinterpret_cast<volatile GuimSharedMemoryV2*>(mapped);
            }
        }
        if (!ipc) {
            std::printf(ANSI_YELLOW "  [Note: /dev/shm/guim_ipc_v3_shm not active; launching in-process engine]\n" ANSI_RESET);
        }
    }
#endif

    guim::Engine engine{42u};
    AcousticFeatureExtractor extractor;

    std::vector<float> input_latent(GUIM_INPUT_DIM, 0.0f);
    std::vector<float> output_cortex(GUIM_STATE_DIM, 0.0f);

    size_t hop_size = AcousticFeatureExtractor::kHopSize;
    size_t win_size = AcousticFeatureExtractor::kWindowSize;
    size_t total_frames = (wav.samples.size() > win_size) ? (wav.samples.size() - win_size) / hop_size : 0;

    std::chrono::microseconds frame_interval(static_cast<long long>(10000.0 / playback_speed)); // 10 ms nominal

    std::printf(ANSI_BOLD "Streaming %zu Audio Frames into Neuromorphic Substrate...\n\n" ANSI_RESET, total_frames);

    auto start_time = std::chrono::steady_clock::now();
    double total_latency_us = 0.0;
    size_t frame_count = 0;

    do {
        for (size_t offset = 0; offset + win_size < wav.samples.size(); offset += hop_size) {
            auto frame_start = std::chrono::steady_clock::now();

            // 1. Extract 64-channel acoustic latents from PCM window
            extractor.extract_frame(&wav.samples[offset], input_latent.data());

            // 2. Step the GPU Neuromorphic Substrate
            auto t0 = std::chrono::high_resolution_clock::now();
            (void)engine.step(
                std::span<const float, GUIM_INPUT_DIM>(input_latent.data(), GUIM_INPUT_DIM),
                std::span<float, GUIM_STATE_DIM>(output_cortex.data(), GUIM_STATE_DIM)
            );
            auto t1 = std::chrono::high_resolution_clock::now();
            double step_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            total_latency_us += step_us;
            frame_count++;

            // 3. Broadcast to Zero-Copy SHM if IPC enabled
            if (ipc) {
                for (int k = 0; k < GUIM_PACKED_DIM && (k * 4 + 3) < GUIM_INPUT_DIM; ++k) {
                    auto q = [](float v) -> std::uint8_t {
                        int val = static_cast<int>((std::clamp(v, -1.0f, 1.0f) * 0.5f + 0.5f) * 255.0f);
                        return static_cast<std::uint8_t>(val);
                    };
                    std::uint32_t packed = (q(input_latent[k * 4 + 0])) |
                                           (q(input_latent[k * 4 + 1]) << 8) |
                                           (q(input_latent[k * 4 + 2]) << 16) |
                                           (q(input_latent[k * 4 + 3]) << 24);
                    ipc->real_sensors_q8[k] = packed;
                }
                for (int k = 0; k < GUIM_STATE_DIM; ++k) {
                    ipc->cortex_cognitive_out[k] = output_cortex[k];
                }
                for (int k = 0; k < GUIM_REFLEX_MOTORS; ++k) {
                    ipc->reflex_motor_out[k] = output_cortex[k % GUIM_STATE_DIM];
                }
                ipc->seq_in = ipc->seq_in + 1;
                ipc->seq_out = ipc->seq_in;
            }

            // 4. Print progress bar and instant state
            if (frame_count % 10 == 0) {
                float progress = static_cast<float>(offset) / static_cast<float>(wav.samples.size());
                float rms = 0.0f;
                for (size_t j = 0; j < win_size; ++j) rms += wav.samples[offset + j] * wav.samples[offset + j];
                rms = std::sqrt(rms / win_size);

                std::printf("\r  " ANSI_BOLD "Time: %5.2fs" ANSI_RESET " [%-20s] " ANSI_MAGENTA "RMS: %4.2f" ANSI_RESET " | " ANSI_GREEN "Step Latency: %5.1f us" ANSI_RESET " | " ANSI_CYAN "Active Units: %3d/256" ANSI_RESET,
                            static_cast<double>(offset) / 16000.0,
                            std::string(static_cast<int>(progress * 20.0f), '=').c_str(),
                            rms,
                            step_us,
                            static_cast<int>(std::count_if(output_cortex.begin(), output_cortex.end(), [](float v){ return std::abs(v) > 0.2f; })));
                std::fflush(stdout);
            }

            // Maintain real-time pace
            auto frame_end = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start);
            if (elapsed < frame_interval) {
                std::this_thread::sleep_for(frame_interval - elapsed);
            }
        }
    } while (loop_mode);

    auto end_time = std::chrono::steady_clock::now();
    double total_wall_sec = std::chrono::duration<double>(end_time - start_time).count();

    std::printf("\n\n" ANSI_BOLD ANSI_GREEN "✔ Audio Ingestion Complete!" ANSI_RESET "\n");
    std::printf("  - Total Frames Executed : %zu\n", frame_count);
    std::printf("  - Mean Step Latency     : " ANSI_BOLD "%.2f µs" ANSI_RESET "\n", (frame_count > 0) ? (total_latency_us / frame_count) : 0.0);
    std::printf("  - Wall-Clock Run Time   : %.2f sec (Audio Duration: %.2f sec)\n", total_wall_sec, duration_sec);
    std::printf("  - Real-Time Factor (RTF): " ANSI_BOLD "%.4fx" ANSI_RESET " (Speedup Headroom: " ANSI_GREEN "%.1fx" ANSI_RESET ")\n\n",
                total_wall_sec / duration_sec,
                (duration_sec > 0 && total_latency_us > 0) ? (duration_sec / (total_latency_us * 1e-6)) : 0.0);

#ifdef _WIN32
    if (ipc) UnmapViewOfFile(const_cast<GuimSharedMemoryV2*>(ipc));
    if (hMapFile) CloseHandle(hMapFile);
#else
    if (ipc) ::munmap(const_cast<GuimSharedMemoryV2*>(ipc), sizeof(GuimSharedMemoryV2));
    if (shm_fd >= 0) ::close(shm_fd);
#endif

    return 0;
}
