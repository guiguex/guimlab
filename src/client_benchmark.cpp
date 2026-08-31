// ============================================================================
//  client_benchmark.cpp — Guimlab IPC client + nanosecond latency benchmark
//  ---------------------------------------------------------------------------
//  Measures round-trip time between CPU client and persistent GPU kernel.
//  Auto-detects v3 (/guim_ipc_v3_shm) and v1 (/guim_ipc_shm) server modes.
//
//  Author        : Guimlab Contributors
//  License       : AGPL-3.0
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <atomic>
#include <thread>
#include <cmath>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <immintrin.h>

#include "ipc_structs.h"
#include "ipc_structs_v2.h"

static inline std::uint32_t pack_int8(int8_t b0, int8_t b1, int8_t b2, int8_t b3) {
    return (static_cast<std::uint32_t>(static_cast<std::uint8_t>(b0))) |
           (static_cast<std::uint32_t>(static_cast<std::uint8_t>(b1)) << 8) |
           (static_cast<std::uint32_t>(static_cast<std::uint8_t>(b2)) << 16) |
           (static_cast<std::uint32_t>(static_cast<std::uint8_t>(b3)) << 24);
}

static bool is_valid_shm_name(const char* name) {
    if (!name) return false;
    std::size_t len = std::strlen(name);
    if (len == 0 || len > 255) return false;
    for (std::size_t i = 0; i < len; ++i) {
        char c = name[i];
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '/' || c == '_' || c == '-' || c == '.')) {
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    int target_fps = 0;
    const char* explicit_shm = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--fps") == 0 && i + 1 < argc) {
            target_fps = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--shm") == 0 && i + 1 < argc) {
            explicit_shm = argv[++i];
            if (!is_valid_shm_name(explicit_shm)) {
                std::fprintf(stderr, "[guim_client] ERROR: Invalid SHM segment name format: '%s'\n", explicit_shm);
                return 1;
            }
        }
    }

    // ---------- 1. Auto-detect SHM segment ----------
    const char* shm_candidates[] = {
        "/guim_ipc_v3_shm",
        "/guim_ipc_shm"
    };

    const char* selected_shm = explicit_shm;
    int shm_fd = -1;
    bool is_v3 = true;

    if (selected_shm != nullptr) {
        shm_fd = ::shm_open(selected_shm, O_RDWR, 0666);
        is_v3 = (std::strstr(selected_shm, "v3") != nullptr);
    } else {
        for (const char* cand : shm_candidates) {
            shm_fd = ::shm_open(cand, O_RDWR, 0666);
            if (shm_fd >= 0) {
                selected_shm = cand;
                is_v3 = (std::strstr(cand, "v3") != nullptr);
                break;
            }
        }
    }

    if (shm_fd < 0) {
        std::fprintf(stderr, "\n[guim_client] ERROR: Could not connect to any active SHM segment!\n");
        std::fprintf(stderr, "  Checked: '/guim_ipc_v3_shm' (guim_node_v3) and '/guim_ipc_shm' (guim_node)\n");
        std::fprintf(stderr, "\n  ==> Did you start the server first in another terminal?\n");
        std::fprintf(stderr, "      Run: ./build/bin/guim_node_v3\n\n");
        return 1;
    }

    std::printf("[guim_client] Connected to '%s' (%s mode)\n", selected_shm, is_v3 ? "v3 Two-Speed" : "v1 Single-Speed");

    if (is_v3) {
        void* mapped = ::mmap(nullptr, sizeof(GuimSharedMemoryV2),
                             PROT_READ | PROT_WRITE,
                             MAP_SHARED, shm_fd, 0);
        if (mapped == MAP_FAILED) {
            std::perror("mmap");
            ::close(shm_fd);
            return 1;
        }

        auto* ipc = reinterpret_cast<volatile GuimSharedMemoryV2*>(mapped);

        if (target_fps > 0) {
            std::printf("[guim_client] Streaming synthetic acoustic formants at %d FPS (Ctrl+C to stop)...\n", target_fps);
            const double frame_interval_us = 1e6 / static_cast<double>(target_fps);
            std::uint64_t seq = ipc->seq_in;
            std::uint64_t frame_count = 0;
            auto next_time = std::chrono::steady_clock::now();

            while (!ipc->terminate) {
                seq++;
                float t = static_cast<float>(frame_count) * (1.0f / static_cast<float>(target_fps));
                // Multi-harmonic audio formant signal (100 Hz, 220 Hz, 440 Hz)
                float s = 0.6f * std::sin(2.0f * 3.14159265f * 100.0f * t)
                        + 0.3f * std::sin(2.0f * 3.14159265f * 220.0f * t)
                        + 0.1f * std::sin(2.0f * 3.14159265f * 440.0f * t);
                
                int8_t val8 = static_cast<int8_t>(std::clamp(s * 127.0f, -128.0f, 127.0f));
                std::uint32_t packed = pack_int8(val8, val8, val8, val8);

                for (int j = 0; j < GUIM_PACKED_DIM; ++j) {
                    ipc->real_sensors_q8[j] = packed;
                    ipc->pred_sensors_q8[j] = 0;
                }
                ipc->dopamine = (frame_count % 200 == 0) ? 0.8f : -0.02f;
                ipc->serotonin = 0.5f;

                std::atomic_thread_fence(std::memory_order_release);
                ipc->seq_in = seq;

                auto spin_start = std::chrono::steady_clock::now();
                while (ipc->seq_out < seq && !ipc->terminate) {
                    #if defined(__x86_64__) || defined(_M_X64)
                    _mm_pause();
                    #endif
                    if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - spin_start).count() >= 2) {
                        std::fprintf(stderr, "\n[guim_client] WARNING: Timeout waiting for SHM server response (server may be terminated).\n");
                        break;
                    }
                }

                frame_count++;
                if (frame_count % static_cast<std::uint64_t>(target_fps) == 0) {
                    std::printf("[guim_client] Streamed %llu frames | Audio RMS: %.3f | Seq: %llu\n",
                                static_cast<unsigned long long>(frame_count),
                                std::abs(s),
                                static_cast<unsigned long long>(ipc->seq_out));
                }

                next_time += std::chrono::microseconds(static_cast<long long>(frame_interval_us));
                std::this_thread::sleep_until(next_time);
            }
        } else {
            // Full-speed benchmark (100k iterations)
            constexpr int WARMUP_ITERS = 5000;
            constexpr int BENCH_ITERS  = 100000;

            std::printf("[guim_client] Warm-up: %d iterations...\n", WARMUP_ITERS);
            std::uint64_t current_seq = ipc->seq_in;

            for (int i = 0; i < WARMUP_ITERS; ++i) {
                current_seq++;
                int8_t val8 = static_cast<int8_t>(i % 127);
                std::uint32_t packed = pack_int8(val8, val8, val8, val8);
                for (int j = 0; j < GUIM_PACKED_DIM; ++j) ipc->real_sensors_q8[j] = packed;
                ipc->dopamine  = 0.05f;
                ipc->serotonin = 0.5f;
                ipc->seq_in    = current_seq;

                auto warmup_spin_start = std::chrono::steady_clock::now();
                while (ipc->seq_out < current_seq) {
                    #if defined(__x86_64__) || defined(_M_X64)
                    _mm_pause();
                    #endif
                    if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - warmup_spin_start).count() >= 2) {
                        std::fprintf(stderr, "\n[guim_client] ERROR: Server response timeout during warm-up at iter %d!\n", i);
                        break;
                    }
                }
            }

            std::printf("[guim_client] Benchmark: %d iterations...\n", BENCH_ITERS);
            std::vector<double> latencies_ns;
            latencies_ns.reserve(BENCH_ITERS);

            for (int i = 0; i < BENCH_ITERS; ++i) {
                current_seq++;
                int8_t val8 = static_cast<int8_t>(i % 127);
                std::uint32_t packed = pack_int8(val8, val8, val8, val8);
                for (int j = 0; j < GUIM_PACKED_DIM; ++j) ipc->real_sensors_q8[j] = packed;
                ipc->dopamine = (i % 10 == 0) ? 1.0f : -0.01f;
                ipc->serotonin = 0.5f;

                std::atomic_thread_fence(std::memory_order_release);
                auto t_start = std::chrono::high_resolution_clock::now();
                ipc->seq_in = current_seq;

                auto bench_spin_start = std::chrono::steady_clock::now();
                while (ipc->seq_out < current_seq) {
                    #if defined(__x86_64__) || defined(_M_X64)
                    _mm_pause();
                    #endif
                    if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - bench_spin_start).count() >= 2) {
                        std::fprintf(stderr, "\n[guim_client] ERROR: Server response timeout during benchmark at iter %d!\n", i);
                        break;
                    }
                }

                auto t_end = std::chrono::high_resolution_clock::now();
                latencies_ns.push_back(
                    std::chrono::duration<double, std::nano>(t_end - t_start).count()
                );
            }

            std::sort(latencies_ns.begin(), latencies_ns.end());
            const double min_lat = latencies_ns.front();
            const double max_lat = latencies_ns.back();
            const double sum     = std::accumulate(latencies_ns.begin(), latencies_ns.end(), 0.0);
            const double avg_lat = sum / BENCH_ITERS;
            const double p50     = latencies_ns[BENCH_ITERS * 50 / 100];
            const double p95     = latencies_ns[BENCH_ITERS * 95 / 100];
            const double p99     = latencies_ns[BENCH_ITERS * 99 / 100];
            const double p999    = latencies_ns[BENCH_ITERS * 999 / 1000];

            std::printf("\n================ RESULTATS BENCHMARK IPC (v3) ================\n");
            std::printf("Itérations testées : %d\n", BENCH_ITERS);
            std::printf("Latence min        : %8.1f ns  (%6.3f µs)\n", min_lat, min_lat / 1000.0);
            std::printf("Latence p50        : %8.1f ns  (%6.3f µs)\n", p50,     p50     / 1000.0);
            std::printf("Latence moyenne    : %8.1f ns  (%6.3f µs)\n", avg_lat, avg_lat / 1000.0);
            std::printf("Latence p95        : %8.1f ns  (%6.3f µs)\n", p95,     p95     / 1000.0);
            std::printf("Latence p99        : %8.1f ns  (%6.3f µs)\n", p99,     p99     / 1000.0);
            std::printf("Latence p99.9      : %8.1f ns  (%6.3f µs)\n", p999,    p999    / 1000.0);
            std::printf("Latence max        : %8.1f ns  (%6.3f µs)\n", max_lat, max_lat / 1000.0);
            std::printf("Throughput         : %8.0f frames/sec\n", 1e9 / avg_lat);
            std::printf("==============================================================\n");
        }

        ::munmap(const_cast<GuimSharedMemoryV2*>(ipc), sizeof(GuimSharedMemoryV2));
    } else {
        // v1 Fallback
        void* mapped = ::mmap(nullptr, sizeof(GuimSharedMemory),
                             PROT_READ | PROT_WRITE,
                             MAP_SHARED, shm_fd, 0);
        if (mapped == MAP_FAILED) {
            std::perror("mmap");
            ::close(shm_fd);
            return 1;
        }

        auto* ipc = reinterpret_cast<volatile GuimSharedMemory*>(mapped);
        constexpr int BENCH_ITERS = 50000;
        std::uint64_t current_seq = ipc->seq_in;

        std::printf("[guim_client] Running v1 benchmark: %d iterations...\n", BENCH_ITERS);
        for (int i = 0; i < BENCH_ITERS; ++i) {
            current_seq++;
            ipc->input[0] = static_cast<float>(i % 100) * 0.01f;
            ipc->td_error = 0.05f;
            std::atomic_thread_fence(std::memory_order_release);
            ipc->seq_in = current_seq;
            while (ipc->seq_out < current_seq) {
                #if defined(__x86_64__) || defined(_M_X64)
                _mm_pause();
                #endif
            }
        }
        std::printf("[guim_client] v1 Benchmark done.\n");
        ::munmap(const_cast<GuimSharedMemory*>(ipc), sizeof(GuimSharedMemory));
    }

    ::close(shm_fd);
    return 0;
}