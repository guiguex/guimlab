// ============================================================================
//  main_guim_node_sparse.cpp — Host orchestrator with multi-stream concurrency
//  ---------------------------------------------------------------------------
//  Spawns two CUDA streams:
//    * stream_inference  — runs sparse_persistent_guim_kernel forever
//    * stream_plasticity — runs plasticity_sweeper_kernel every 10s
//
//  Both streams are cudaStreamNonBlocking so the GC never blocks the hot path.
//  Plus the standard SHM + cudaHostRegister + zero-copy setup.
//
//  Usage
//  -----
//    Terminal 1:  taskset -c 0 ./guim_node_sparse
//    Terminal 2:  taskset -c 1 ./guim_delta   (the AVX2 client)
//
//  Author        : Guillaume Meingan + Claude (ultracode session 2026-08-31)
//  License       : AGPL-3.0
//  Compilation   : nvcc -O3 -use_fast_math -std=c++20 ddwr_kernel.cu
//                                          plasticity_sweeper.cu
//                                          main_guim_node_sparse.cpp
//                  -o guim_node_sparse -lrt -lpthread -lcurand
// ============================================================================

#include <cuda_runtime.h>
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
#include "ipc_structs_sparse.h"

// ----------------------------------------------------------------------------
//  Globals for signal handling
// ----------------------------------------------------------------------------

static volatile GuimSharedMemorySparse* g_ipc = nullptr;
static std::atomic<bool> g_stop{false};

static void on_signal(int sig) {
    std::fprintf(stderr, "[guim_node_sparse] Caught signal %d\n", sig);
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

// ----------------------------------------------------------------------------
//  Wall-clock helper
// ----------------------------------------------------------------------------

static std::uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ULL
         + static_cast<std::uint64_t>(ts.tv_nsec);
}

// ----------------------------------------------------------------------------
//  External kernels (declared in their respective .cu files)
// ----------------------------------------------------------------------------

__global__ void sparse_persistent_guim_kernel(
    volatile GuimSharedMemorySparse* __restrict__,
    float* __restrict__, float* __restrict__, float* __restrict__,
    float* __restrict__, float* __restrict__, float* __restrict__,
    std::uint64_t* __restrict__, float* __restrict__);

__global__ void plasticity_sweeper_kernel(
    float* __restrict__,
    std::uint64_t* __restrict__,
    curandState* __restrict__,
    std::uint64_t);

__global__ void init_sweeper_rng_kernel(curandState* __restrict__, std::uint64_t);

extern "C" void plasticity_sweeper_thread_entry(
    float*, std::uint64_t*, curandState*, std::atomic<bool>*, cudaStream_t);

// ----------------------------------------------------------------------------
//  main
// ----------------------------------------------------------------------------

int main(int /*argc*/, char** /*argv*/) {
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    constexpr const char* SHM_NAME = "/guim_ipc_shm_sparse";

    // ---------- 1. SHM create + mmap + cudaHostRegister ----------
    std::printf("[guim_node_sparse] Creating POSIX SHM '%s'...\n", SHM_NAME);
    int shm_fd = ::shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd < 0) { std::perror("shm_open"); return 1; }
    ::ftruncate(shm_fd, sizeof(GuimSharedMemorySparse));
    void* mapped = ::mmap(nullptr, sizeof(GuimSharedMemorySparse),
                           PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (mapped == MAP_FAILED) { std::perror("mmap"); ::close(shm_fd); return 1; }

    auto* ipc = reinterpret_cast<volatile GuimSharedMemorySparse*>(mapped);
    g_ipc = ipc;
    std::memset(const_cast<GuimSharedMemorySparse*>(ipc), 0, sizeof(*ipc));
    ipc->terminate = false;

    CUDA_OK(cudaHostRegister(const_cast<GuimSharedMemorySparse*>(ipc),
                             sizeof(GuimSharedMemorySparse),
                             cudaHostRegisterMapped));

    GuimSharedMemorySparse* d_ipc = nullptr;
    CUDA_OK(cudaHostGetDevicePointer(&d_ipc, const_cast<GuimSharedMemorySparse*>(ipc), 0));

    // ---------- 2. Allocate device-side state buffers ----------
    const int total_synapses = GUIM_STATE_DIM * (GUIM_INPUT_DIM + GUIM_STATE_DIM);
    float *d_hidden = nullptr, *d_weights = nullptr, *d_traces = nullptr;
    float *d_alphas = nullptr, *d_betas = nullptr, *d_meta_traces = nullptr;
    float* d_input_state = nullptr;
    std::uint64_t* d_last_t = nullptr;
    curandState*   d_rng_sweeper = nullptr;

    CUDA_OK(cudaMalloc(&d_hidden,      GUIM_STATE_DIM * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_weights,     total_synapses * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_traces,      total_synapses * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_alphas,      total_synapses * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_betas,       total_synapses * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_meta_traces, total_synapses * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_input_state, GUIM_INPUT_DIM * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_last_t,       total_synapses * sizeof(std::uint64_t)));
    CUDA_OK(cudaMalloc(&d_rng_sweeper, total_synapses * sizeof(curandState)));

    // ---------- 3. Initialize ----------
    CUDA_OK(cudaMemset(d_hidden,      0, GUIM_STATE_DIM * sizeof(float)));
    CUDA_OK(cudaMemset(d_traces,      0, total_synapses * sizeof(float)));
    CUDA_OK(cudaMemset(d_meta_traces, 0, total_synapses * sizeof(float)));
    CUDA_OK(cudaMemset(d_input_state, 0, GUIM_INPUT_DIM * sizeof(float)));
    CUDA_OK(cudaMemset(d_last_t,       0, total_synapses * sizeof(std::uint64_t)));

    {
        constexpr int THREADS = 256;
        const int blocks = (total_synapses + THREADS - 1) / THREADS;
        // Sparse init kernel — set alphas/betas/weights
        // (we re-use the sparse_init_kernel that lives in ddwr_kernel.cu)
        extern __global__ void sparse_init_kernel(
            float*, float*, float*, float*, float*, float*, std::uint64_t*, std::uint64_t);
        sparse_init_kernel<<<blocks, THREADS>>>(
            d_weights, d_alphas, d_betas, d_hidden, d_traces, d_meta_traces,
            d_last_t, now_ns()
        );
        init_sweeper_rng_kernel<<<blocks, THREADS>>>(d_rng_sweeper, 0xDEADBEEFull);
    }
    CUDA_OK(cudaDeviceSynchronize());

    // ---------- 4. Create two non-blocking streams ----------
    cudaStream_t stream_inference, stream_plasticity;
    CUDA_OK(cudaStreamCreateWithFlags(&stream_inference,  cudaStreamNonBlocking));
    CUDA_OK(cudaStreamCreateWithFlags(&stream_plasticity, cudaStreamNonBlocking));

    // ---------- 5. Launch persistent sparse kernel ----------
    std::printf("[guim_node_sparse] Launching sparse persistent kernel...\n");
    sparse_persistent_guim_kernel<<<1, GUIM_STATE_DIM, 0, stream_inference>>>(
        d_ipc, d_hidden, d_weights, d_traces, d_alphas, d_betas, d_meta_traces,
        d_last_t, d_input_state
    );
    CUDA_OK(cudaGetLastError());

    // ---------- 6. Spawn GC thread ----------
    std::printf("[guim_node_sparse] Spawning plasticity GC thread (every 10s)...\n");
    std::thread gc_thread(plasticity_sweeper_thread_entry,
                          d_weights, d_last_t, d_rng_sweeper, &g_stop, stream_plasticity);

    // ---------- 7. Wait for terminate ----------
    while (!g_stop.load(std::memory_order_acquire) && !ipc->terminate) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // ---------- 8. Teardown ----------
    std::printf("[guim_node_sparse] Shutting down...\n");
    g_stop.store(true, std::memory_order_release);
    if (g_ipc) g_ipc->terminate = true;

    gc_thread.join();
    CUDA_OK(cudaDeviceSynchronize());

    CUDA_OK(cudaStreamDestroy(stream_inference));
    CUDA_OK(cudaStreamDestroy(stream_plasticity));

    CUDA_OK(cudaHostUnregister(const_cast<GuimSharedMemorySparse*>(ipc)));
    ::munmap(const_cast<GuimSharedMemorySparse*>(ipc), sizeof(*ipc));
    ::close(shm_fd);
    ::shm_unlink(SHM_NAME);

    CUDA_OK(cudaFree(d_hidden));
    CUDA_OK(cudaFree(d_weights));
    CUDA_OK(cudaFree(d_traces));
    CUDA_OK(cudaFree(d_alphas));
    if (d_betas)       cudaFree(d_betas);
    if (d_meta_traces) cudaFree(d_meta_traces);
    if (d_input_state) cudaFree(d_input_state);
    if (d_last_t)      cudaFree(d_last_t);
    if (d_rng_sweeper) cudaFree(d_rng_sweeper);

    return 0;
}