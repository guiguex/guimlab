// ============================================================================
//  client_benchmark.cpp — Guimlab IPC client + nanosecond latency benchmark
//  ---------------------------------------------------------------------------
//  Measures round-trip time between CPU client and persistent GPU kernel.
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

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <immintrin.h>

#include "ipc_structs.h"

int main(int /*argc*/, char** /*argv*/) {
    constexpr const char* SHM_NAME = "/guim_ipc_shm";

    std::printf("[guim_client] Connecting to SHM '%s'...\n", SHM_NAME);
    int shm_fd = ::shm_open(SHM_NAME, O_RDWR, 0666);
    if (shm_fd < 0) {
        std::fprintf(stderr, "[guim_client] shm_open failed: %s\n",
                     std::strerror(errno));
        std::fprintf(stderr, "[guim_client] Is the server (guim_node) running?\n");
        return 1;
    }

    void* mapped = ::mmap(nullptr, sizeof(GuimSharedMemory),
                         PROT_READ | PROT_WRITE,
                         MAP_SHARED, shm_fd, 0);
    if (mapped == MAP_FAILED) {
        std::perror("mmap");
        ::close(shm_fd);
        return 1;
    }

    auto* ipc = reinterpret_cast<volatile GuimSharedMemory*>(mapped);
    std::printf("[guim_client] Connected. SHM at %p (%zu bytes).\n",
                const_cast<void*>(static_cast<const volatile void*>(ipc)), sizeof(GuimSharedMemory));

    // ---------- 2. Prepare synthetic input ----------
    float test_input[GUIM_INPUT_DIM];
    for (int i = 0; i < GUIM_INPUT_DIM; ++i) {
        test_input[i] = static_cast<float>(i) * 0.01f;
    }

    // ---------- 3. Warm-up ----------
    constexpr int WARMUP_ITERS = 5000;
    constexpr int BENCH_ITERS  = 100000;

    std::printf("[guim_client] Warm-up: %d iterations...\n", WARMUP_ITERS);
    std::uint64_t current_seq = ipc->seq_in;

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        current_seq++;
        for (int j = 0; j < GUIM_INPUT_DIM; ++j) ipc->input[j] = test_input[j];
        ipc->td_error = 0.05f;
        ipc->seq_in   = current_seq;

        while (ipc->seq_out < current_seq) {
            #if defined(__x86_64__) || defined(_M_X64)
            _mm_pause();
            #endif
        }
    }

    // ---------- 4. Benchmark ----------
    std::printf("[guim_client] Benchmark: %d iterations...\n", BENCH_ITERS);
    std::vector<double> latencies_ns;
    latencies_ns.reserve(BENCH_ITERS);

    for (int i = 0; i < BENCH_ITERS; ++i) {
        current_seq++;
        ipc->input[0] = static_cast<float>(i % 100) * 0.01f;
        ipc->td_error = (i % 10 == 0) ? 1.0f : -0.01f;

        std::atomic_thread_fence(std::memory_order_release);

        auto t_start = std::chrono::high_resolution_clock::now();
        ipc->seq_in = current_seq;

        while (ipc->seq_out < current_seq) {
            #if defined(__x86_64__) || defined(_M_X64)
            _mm_pause();
            #endif
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        latencies_ns.push_back(
            std::chrono::duration<double, std::nano>(t_end - t_start).count()
        );
    }

    // ---------- 5. Stats ----------
    std::sort(latencies_ns.begin(), latencies_ns.end());

    const double min_lat = latencies_ns.front();
    const double max_lat = latencies_ns.back();
    const double sum     = std::accumulate(latencies_ns.begin(), latencies_ns.end(), 0.0);
    const double avg_lat = sum / BENCH_ITERS;
    const double p50     = latencies_ns[BENCH_ITERS * 50 / 100];
    const double p95     = latencies_ns[BENCH_ITERS * 95 / 100];
    const double p99     = latencies_ns[BENCH_ITERS * 99 / 100];
    const double p999    = latencies_ns[BENCH_ITERS * 999 / 1000];

    std::printf("\n================ RESULTATS BENCHMARK IPC ================\n");
    std::printf("Itérations testées : %d\n", BENCH_ITERS);
    std::printf("Latence min        : %8.1f ns  (%6.3f µs)\n", min_lat, min_lat / 1000.0);
    std::printf("Latence p50        : %8.1f ns  (%6.3f µs)\n", p50,     p50     / 1000.0);
    std::printf("Latence moyenne    : %8.1f ns  (%6.3f µs)\n", avg_lat, avg_lat / 1000.0);
    std::printf("Latence p95        : %8.1f ns  (%6.3f µs)\n", p95,     p95     / 1000.0);
    std::printf("Latence p99        : %8.1f ns  (%6.3f µs)\n", p99,     p99     / 1000.0);
    std::printf("Latence p99.9      : %8.1f ns  (%6.3f µs)\n", p999,    p999    / 1000.0);
    std::printf("Latence max        : %8.1f ns  (%6.3f µs)\n", max_lat, max_lat / 1000.0);
    std::printf("Throughput         : %8.0f frames/sec\n", 1e9 / avg_lat);
    std::printf("=========================================================\n");

    ::munmap(const_cast<GuimSharedMemory*>(ipc), sizeof(GuimSharedMemory));
    ::close(shm_fd);
    return 0;
}