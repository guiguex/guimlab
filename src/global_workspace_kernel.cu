// ============================================================================
//  global_workspace_kernel.cu — GPU Implementation of GWT Bottleneck & Broadcast
//  ============================================================================
//  Author        : Guimlab Contributors
//  License       : AGPL-3.0 / MIT
// ============================================================================

#include "global_workspace.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstring>

namespace guim {

// ----------------------------------------------------------------------------
//  GWT CUDA Kernels
// ----------------------------------------------------------------------------

__global__ void gwt_competition_and_broadcast_kernel(
    const float* __restrict__ d_proposals,  // [GWT_MODULES * GWT_IN_DIM]
    const float* __restrict__ d_saliences,  // [GWT_MODULES]
    float*       __restrict__ d_weights,    // [GWT_MODULES]
    float*       __restrict__ d_broadcast,  // [GWT_BROADCAST_DIM]
    float                     temp
) {
    const int tid = threadIdx.x;

    // 1. Softmax competition across modules (executed in shared memory / registers)
    __shared__ float s_weights[GWT_MODULES];

    if (tid < GWT_MODULES) {
        float max_s = d_saliences[0];
        for (int m = 1; m < GWT_MODULES; ++m) {
            if (d_saliences[m] > max_s) max_s = d_saliences[m];
        }

        float sum_exp = 0.0f;
        for (int m = 0; m < GWT_MODULES; ++m) {
            sum_exp += __expf(temp * (d_saliences[m] - max_s));
        }

        const float exp_val = __expf(temp * (d_saliences[tid] - max_s));
        const float w = (sum_exp > 1e-8f) ? (exp_val / sum_exp) : (1.0f / GWT_MODULES);
        s_weights[tid] = w;
        d_weights[tid] = w;
    }
    __syncthreads();

    // 2. Synthesize Global Broadcast Vector in parallel
    if (tid < GWT_BROADCAST_DIM) {
        float accum = 0.0f;
        #pragma unroll
        for (int m = 0; m < GWT_MODULES; ++m) {
            accum += s_weights[m] * d_proposals[m * GWT_IN_DIM + tid];
        }
        d_broadcast[tid] = tanhf(accum);
    }
}

// ----------------------------------------------------------------------------
//  GlobalWorkspace Class Implementation
// ----------------------------------------------------------------------------

GlobalWorkspace::GlobalWorkspace() {
    cudaStreamCreateWithFlags(&state_.stream, cudaStreamNonBlocking);

    cudaMalloc(&state_.d_proposals, GWT_MODULES * GWT_IN_DIM * sizeof(float));
    cudaMalloc(&state_.d_saliences, GWT_MODULES * sizeof(float));
    cudaMalloc(&state_.d_weights,   GWT_MODULES * sizeof(float));
    cudaMalloc(&state_.d_broadcast, GWT_BROADCAST_DIM * sizeof(float));

    cudaMemsetAsync(state_.d_proposals, 0, GWT_MODULES * GWT_IN_DIM * sizeof(float), state_.stream);
    cudaMemsetAsync(state_.d_saliences, 0, GWT_MODULES * sizeof(float), state_.stream);
    cudaMemsetAsync(state_.d_broadcast, 0, GWT_BROADCAST_DIM * sizeof(float), state_.stream);
    cudaStreamSynchronize(state_.stream);
}

GlobalWorkspace::~GlobalWorkspace() {
    if (state_.stream) {
        cudaStreamSynchronize(state_.stream);
        cudaStreamDestroy(state_.stream);
    }
    if (state_.d_proposals) cudaFree(state_.d_proposals);
    if (state_.d_saliences) cudaFree(state_.d_saliences);
    if (state_.d_weights)   cudaFree(state_.d_weights);
    if (state_.d_broadcast) cudaFree(state_.d_broadcast);
}

GlobalWorkspace::GlobalWorkspace(GlobalWorkspace&& other) noexcept
    : state_(other.state_)
{
    other.state_ = {};
}

GlobalWorkspace& GlobalWorkspace::operator=(GlobalWorkspace&& other) noexcept {
    if (this != &other) {
        this->~GlobalWorkspace();
        state_ = other.state_;
        other.state_ = {};
    }
    return *this;
}

void GlobalWorkspace::set_proposal(int module_idx,
                                  std::span<const float, GWT_IN_DIM> proposal,
                                  float salience) {
    if (module_idx < 0 || module_idx >= GWT_MODULES) return;

    h_saliences_[module_idx] = salience;

    const std::size_t offset = module_idx * GWT_IN_DIM * sizeof(float);
    cudaMemcpyAsync(reinterpret_cast<char*>(state_.d_proposals) + offset,
                    proposal.data(), GWT_IN_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, state_.stream);

    cudaMemcpyAsync(state_.d_saliences + module_idx,
                    &salience, sizeof(float),
                    cudaMemcpyHostToDevice, state_.stream);
}

void GlobalWorkspace::arbitrate_and_broadcast(
    std::span<float, GWT_BROADCAST_DIM> broadcast_out,
    float competition_temp
) {
    constexpr int THREADS = 64;
    gwt_competition_and_broadcast_kernel<<<1, THREADS, 0, state_.stream>>>(
        state_.d_proposals, state_.d_saliences, state_.d_weights,
        state_.d_broadcast, competition_temp
    );

    cudaMemcpyAsync(broadcast_out.data(), state_.d_broadcast,
                    GWT_BROADCAST_DIM * sizeof(float),
                    cudaMemcpyDeviceToHost, state_.stream);
    cudaStreamSynchronize(state_.stream);

    std::memcpy(h_broadcast_, broadcast_out.data(), sizeof(h_broadcast_));
}

int GlobalWorkspace::get_dominant_module() const {
    int dom = 0;
    float max_s = h_saliences_[0];
    for (int m = 1; m < GWT_MODULES; ++m) {
        if (h_saliences_[m] > max_s) {
            max_s = h_saliences_[m];
            dom = m;
        }
    }
    return dom;
}

}  // namespace guim
