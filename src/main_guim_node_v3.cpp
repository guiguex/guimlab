// ============================================================================
//  main_guim_node_v3.cpp — Host orchestrator (v3) with motor babbling + reflex
//  ---------------------------------------------------------------------------
//  Orchestrates the complete two-speed AGI substrate on a single shared POSIX SHM:
//    * Reflex (kiss_reflex_kernel)    — L0 register path, writes reflex_motor_out[16]
//    * Cortex (guim_lovelace_kernel)  — sparse + WMMA, writes cortex_cognitive_out[256]
//    * Babbling (motor_babbling_kernel) — exploration noise on stream_plasticity
//
//  Author        : Guimlab Contributors
//  License       : AGPL-3.0
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>

#include <atomic>
#include <chrono>
#include <thread>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <csignal>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <time.h>

#include "ipc_structs.h"
#include "ipc_structs_v2.h"

// ----------------------------------------------------------------------------
//  Globals for signal handling
// ----------------------------------------------------------------------------

static volatile GuimSharedMemoryV2* g_ipc = nullptr;
static std::atomic<bool> g_stop{false};

static void on_signal(int sig) {
    std::fprintf(stderr, "[guim_node_v3] Caught signal %d\n", sig);
    if (g_ipc) g_ipc->terminate = true;
    g_stop.store(true, std::memory_order_release);
}

// ----------------------------------------------------------------------------
//  CUDA error helper
// ----------------------------------------------------------------------------

#define CUDA_OK(expr)                                                        \
    do {                                                                     \
        cudaError_t _e = (expr);                                             \
        if (_e != cudaSuccess) {                                              \
            std::fprintf(stderr, "[CUDA ERROR] %s at %s:%d\n",              \
                          cudaGetErrorString(_e), __FILE__, __LINE__);      \
            std::exit(1);                                                     \
        }                                                                    \
    } while (0)

static std::uint64_t now_ns() {
    const auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()
    ).count();
}

// ----------------------------------------------------------------------------
//  External kernels
// ----------------------------------------------------------------------------

__global__ void kiss_reflex_persistent_kernel(
    volatile GuimSharedMemoryV2* __restrict__ d_ipc);

__global__ void guim_lovelace_init_kernel(
    half*          __restrict__ d_weights_fp16,
    half*          __restrict__ d_alphas_fp16,
    float*         __restrict__ d_hidden,
    float*         __restrict__ d_traces,
    float*         __restrict__ d_meta_traces,
    float*         __restrict__ d_betas,
    std::uint64_t* __restrict__ d_last_t,
    std::uint64_t               start_time_ns);

__global__ void guim_lovelace_fused_kernel(
    volatile GuimSharedMemoryV2* __restrict__ d_ipc,
    float*          __restrict__ d_hidden,
    half*           __restrict__ d_weights_fp16,
    half*           __restrict__ d_alphas_fp16,
    float*          __restrict__ d_traces,
    std::uint64_t*  __restrict__ d_last_t,
    float*          __restrict__ d_meta_traces,
    float*          __restrict__ d_betas);

__global__ void plasticity_sweeper_kernel(
    float*, std::uint64_t*, curandState*, std::uint64_t);
__global__ void init_sweeper_rng_kernel(curandState*, std::uint64_t);

__global__ void motor_babbling_kernel(
    volatile GuimSharedMemoryV2* __restrict__, curandState*);
__global__ void babbling_init_kernel(curandState*, std::uint64_t);

// ----------------------------------------------------------------------------
//  main
// ----------------------------------------------------------------------------

int main(int /*argc*/, char** /*argv*/) {
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    constexpr const char* SHM_NAME = "/guim_ipc_v3_shm";

    std::printf("[guim_node_v3] Two-speed AGI orchestrator (Reflex + Cortex + Babbling)\n");

    // ---------- 1. SHM create + mmap + cudaHostRegister ----------
    std::printf("[guim_node_v3] Creating POSIX SHM '%s'...\n", SHM_NAME);
    int shm_fd = ::shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd < 0) { std::perror("shm_open"); return 1; }
    if (::ftruncate(shm_fd, sizeof(GuimSharedMemoryV2)) != 0) {
        std::perror("ftruncate");
        ::close(shm_fd);
        return 1;
    }
    void* mapped = ::mmap(nullptr, sizeof(GuimSharedMemoryV2),
                           PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (mapped == MAP_FAILED) { std::perror("mmap"); ::close(shm_fd); return 1; }

    auto* ipc = reinterpret_cast<volatile GuimSharedMemoryV2*>(mapped);
    g_ipc = ipc;
    std::memset(const_cast<GuimSharedMemoryV2*>(ipc), 0, sizeof(*ipc));
    ipc->terminate = false;
    ipc->serotonin = 1.0f;  // start confident

    CUDA_OK(cudaHostRegister(const_cast<GuimSharedMemoryV2*>(ipc),
                             sizeof(GuimSharedMemoryV2),
                             cudaHostRegisterMapped));

    GuimSharedMemoryV2* d_ipc = nullptr;
    CUDA_OK(cudaHostGetDevicePointer(&d_ipc, const_cast<GuimSharedMemoryV2*>(ipc), 0));

    // ---------- 2. Allocate device-side state buffers ----------
    const int total_synapses = GUIM_STATE_DIM * GUIM_INPUT_DIM;
    float*         d_hidden       = nullptr;
    half*          d_weights_fp16 = nullptr;
    half*          d_alphas_fp16  = nullptr;
    float*         d_traces       = nullptr;
    std::uint64_t* d_last_t       = nullptr;
    float*         d_meta_traces  = nullptr;
    float*         d_betas        = nullptr;
    curandState*   d_rng_sweeper  = nullptr;
    curandState*   d_rng_babbler  = nullptr;

    CUDA_OK(cudaMalloc(&d_hidden,       GUIM_STATE_DIM * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_weights_fp16, total_synapses * sizeof(half)));
    CUDA_OK(cudaMalloc(&d_alphas_fp16,  total_synapses * sizeof(half)));
    CUDA_OK(cudaMalloc(&d_traces,       total_synapses * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_last_t,       total_synapses * sizeof(std::uint64_t)));
    CUDA_OK(cudaMalloc(&d_meta_traces,  total_synapses * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_betas,        total_synapses * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_rng_sweeper,  total_synapses * sizeof(curandState)));
    CUDA_OK(cudaMalloc(&d_rng_babbler,  GUIM_STATE_DIM * sizeof(curandState)));

    // ---------- 3. Initialize ----------
    CUDA_OK(cudaMemset(d_hidden,      0, GUIM_STATE_DIM * sizeof(float)));
    CUDA_OK(cudaMemset(d_traces,      0, total_synapses * sizeof(float)));
    CUDA_OK(cudaMemset(d_meta_traces, 0, total_synapses * sizeof(float)));
    CUDA_OK(cudaMemset(d_last_t,       0, total_synapses * sizeof(std::uint64_t)));

    {
        constexpr int THREADS = 256;
        const int blocks = (total_synapses + THREADS - 1) / THREADS;

        guim_lovelace_init_kernel<<<blocks, THREADS>>>(
            d_weights_fp16, d_alphas_fp16, d_hidden, d_traces,
            d_meta_traces, d_betas, d_last_t, now_ns()
        );

        init_sweeper_rng_kernel<<<blocks, THREADS>>>(d_rng_sweeper, 0xDEADBEEFull);
        babbling_init_kernel<<<(GUIM_STATE_DIM + 255) / 256, 256>>>(d_rng_babbler, 0xBABBEFull);
    }
    CUDA_OK(cudaDeviceSynchronize());

    // ---------- 4. Create separate non-blocking streams ----------
    cudaStream_t stream_reflex, stream_cortex, stream_plasticity;
    CUDA_OK(cudaStreamCreateWithFlags(&stream_reflex,     cudaStreamNonBlocking));
    CUDA_OK(cudaStreamCreateWithFlags(&stream_cortex,     cudaStreamNonBlocking));
    CUDA_OK(cudaStreamCreateWithFlags(&stream_plasticity, cudaStreamNonBlocking));

    // ---------- 5. Launch REFLEX & CORTEX in parallel on separate streams ----------
    std::printf("[guim_node_v3] Launching REFLEX kernel (L0 registers)...\n");
    kiss_reflex_persistent_kernel<<<1, 32, 0, stream_reflex>>>(d_ipc);
    CUDA_OK(cudaGetLastError());

    std::printf("[guim_node_v3] Launching CORTEX kernel (sparse + DDWR + TMD-ET)...\n");
    guim_lovelace_fused_kernel<<<1, GUIM_STATE_DIM, 0, stream_cortex>>>(
        d_ipc, d_hidden, d_weights_fp16, d_alphas_fp16, d_traces,
        d_last_t, d_meta_traces, d_betas
    );
    CUDA_OK(cudaGetLastError());

    // ---------- 6. Launch BABBLING thread ----------
    std::printf("[guim_node_v3] Starting motor babbling worker...\n");
    std::thread babbling_thread([&]() {
        while (!g_stop.load(std::memory_order_acquire)) {
            motor_babbling_kernel<<<(GUIM_STATE_DIM + 255) / 256, 256, 0, stream_plasticity>>>(
                d_ipc, d_rng_babbler
            );
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    // ---------- 7. Main loop: wait for termination ----------
    while (!g_stop.load(std::memory_order_acquire) && !ipc->terminate) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // ---------- 8. Teardown ----------
    std::printf("[guim_node_v3] Shutting down...\n");
    g_stop.store(true, std::memory_order_release);
    if (g_ipc) g_ipc->terminate = true;

    babbling_thread.join();
    CUDA_OK(cudaDeviceSynchronize());

    CUDA_OK(cudaStreamDestroy(stream_reflex));
    CUDA_OK(cudaStreamDestroy(stream_cortex));
    CUDA_OK(cudaStreamDestroy(stream_plasticity));

    CUDA_OK(cudaHostUnregister(const_cast<GuimSharedMemoryV2*>(ipc)));
    ::munmap(const_cast<GuimSharedMemoryV2*>(ipc), sizeof(*ipc));
    ::close(shm_fd);
    ::shm_unlink(SHM_NAME);

    CUDA_OK(cudaFree(d_hidden));
    CUDA_OK(cudaFree(d_weights_fp16));
    CUDA_OK(cudaFree(d_alphas_fp16));
    CUDA_OK(cudaFree(d_traces));
    CUDA_OK(cudaFree(d_last_t));
    CUDA_OK(cudaFree(d_meta_traces));
    CUDA_OK(cudaFree(d_betas));
    CUDA_OK(cudaFree(d_rng_sweeper));
    CUDA_OK(cudaFree(d_rng_babbler));

    std::printf("[guim_node_v3] Clean exit completed.\n");
    return 0;
}