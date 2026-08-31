// ============================================================================
//  main_guim_node.cpp — Persistent Guimlab node (CUDA server)
//  ---------------------------------------------------------------------------
//  Allocates POSIX SHM, pins it for zero-copy GPU access, initializes all
//  device-side state buffers, then launches `persistent_guim_kernel`.
//
//  Author        : Guimlab Contributors
//  License       : AGPL-3.0 / MIT
// ============================================================================

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <csignal>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "ipc_structs.h"

// Global pointer for signal handler
static volatile GuimSharedMemory* g_ipc = nullptr;

static void on_signal(int sig) {
    std::fprintf(stderr, "[guim_node] Caught signal %d, requesting terminate...\n", sig);
    if (g_ipc) {
        g_ipc->terminate = true;
    }
}

#define CUDA_OK(expr)                                                        \
    do {                                                                     \
        cudaError_t _e = (expr);                                             \
        if (_e != cudaSuccess) {                                              \
            std::fprintf(stderr, "[CUDA ERROR] %s (%s) at %s:%d\n",         \
                          cudaGetErrorString(_e), cudaGetErrorName(_e),     \
                          __FILE__, __LINE__);                               \
            std::exit(1);                                                     \
        }                                                                    \
    } while (0)

__global__ void guim_init_kernel(
    float*       __restrict__ d_weights,
    float*       __restrict__ d_alphas,
    float*       __restrict__ d_betas,
    float*       __restrict__ d_hidden,
    float*       __restrict__ d_traces,
    float*       __restrict__ d_meta_traces,
    float*       __restrict__ d_utility,
    float*       __restrict__ d_mean,
    curandState* __restrict__ d_rng,
    unsigned long long        seed
);

__global__ void persistent_guim_kernel(
    volatile GuimSharedMemory* __restrict__ ipc,
    float*        __restrict__ d_hidden,
    float*        __restrict__ d_weights,
    float*        __restrict__ d_traces,
    float*        __restrict__ d_alphas,
    float*        __restrict__ d_betas,
    float*        __restrict__ d_meta_traces,
    float*        __restrict__ d_utility,
    float*        __restrict__ d_mean,
    curandState*  __restrict__ d_rng
);

int main(int /*argc*/, char** /*argv*/) {
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    constexpr const char* SHM_NAME = "/guim_ipc_shm";
    std::printf("[guim_node] Creating POSIX SHM '%s'...\n", SHM_NAME);

    int shm_fd = ::shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd < 0) {
        std::perror("shm_open");
        return 1;
    }

    if (::ftruncate(shm_fd, sizeof(GuimSharedMemory)) != 0) {
        std::perror("ftruncate");
        ::close(shm_fd);
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
    g_ipc     = ipc;

    ipc->seq_in    = 0;
    ipc->seq_out   = 0;
    ipc->terminate = false;

    // Pin SHM for zero-copy GPU access
    std::printf("[guim_node] Pinning SHM as zero-copy (cudaHostRegisterMapped)...\n");
    CUDA_OK(cudaHostRegister(const_cast<GuimSharedMemory*>(ipc),
                          sizeof(GuimSharedMemory),
                          cudaHostRegisterMapped));

    void* d_ipc_void = nullptr;
    CUDA_OK(cudaHostGetDevicePointer(&d_ipc_void, const_cast<GuimSharedMemory*>(ipc), 0));
    auto* d_ipc = static_cast<GuimSharedMemory*>(d_ipc_void);
    std::printf("[guim_node] Device pointer for IPC: %p (host: %p)\n",
                static_cast<const void*>(d_ipc),
                static_cast<const void*>(const_cast<GuimSharedMemory*>(ipc)));

    // Allocate device buffers
    float *d_hidden = nullptr, *d_weights = nullptr, *d_traces = nullptr;
    float *d_alphas = nullptr, *d_betas = nullptr, *d_meta_traces = nullptr;
    float *d_utility = nullptr, *d_mean = nullptr;
    curandState *d_rng = nullptr;

    CUDA_OK(cudaMalloc(&d_hidden,      GUIM_STATE_DIM * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_weights,     GUIM_TOTAL_W   * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_traces,      GUIM_TOTAL_W   * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_alphas,      GUIM_TOTAL_W   * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_betas,       GUIM_TOTAL_W   * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_meta_traces, GUIM_TOTAL_W   * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_utility,     GUIM_STATE_DIM * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_mean,        GUIM_STATE_DIM * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_rng,         GUIM_STATE_DIM * sizeof(curandState)));

    // Initialize state on GPU
    std::printf("[guim_node] Initializing state (Xavier + alpha=0.01 + CBP tracking)...\n");
    {
        constexpr int THREADS = 256;
        const int blocks = (GUIM_TOTAL_W + THREADS - 1) / THREADS;
        guim_init_kernel<<<blocks, THREADS>>>(
            d_weights, d_alphas, d_betas, d_hidden, d_traces, d_meta_traces,
            d_utility, d_mean, d_rng, 0xC0FFEEull
        );
        CUDA_OK(cudaGetLastError());
    }
    CUDA_OK(cudaDeviceSynchronize());

    // Launch persistent kernel
    std::printf("[guim_node] Launching persistent kernel (1 block x %d threads)...\n",
                GUIM_STATE_DIM);
    std::printf("[guim_node] Waiting for IPC frames. Ctrl+C to exit.\n");

    persistent_guim_kernel<<<1, GUIM_STATE_DIM>>>(
        d_ipc, d_hidden, d_weights, d_traces, d_alphas, d_betas, d_meta_traces,
        d_utility, d_mean, d_rng
    );
    CUDA_OK(cudaGetLastError());

    // Wait for kernel to exit
    CUDA_OK(cudaDeviceSynchronize());
    std::printf("[guim_node] Kernel exited cleanly.\n");

    // Teardown
    CUDA_OK(cudaHostUnregister(const_cast<GuimSharedMemory*>(ipc)));
    ::munmap(const_cast<GuimSharedMemory*>(ipc), sizeof(GuimSharedMemory));
    ::close(shm_fd);
    ::shm_unlink(SHM_NAME);

    CUDA_OK(cudaFree(d_hidden));
    CUDA_OK(cudaFree(d_weights));
    CUDA_OK(cudaFree(d_traces));
    CUDA_OK(cudaFree(d_alphas));
    CUDA_OK(cudaFree(d_betas));
    CUDA_OK(cudaFree(d_meta_traces));
    CUDA_OK(cudaFree(d_utility));
    CUDA_OK(cudaFree(d_mean));
    CUDA_OK(cudaFree(d_rng));

    return 0;
}