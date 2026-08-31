// ============================================================================
//  episodic_vsa_kernel.cu — CUDA implementation of VSA & Modern Hopfield Memory
//  ============================================================================
//  Author        : Guimlab Contributors
//  License       : AGPL-3.0 / MIT
// ============================================================================

#include "episodic_vsa.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstring>

namespace guim {

// ----------------------------------------------------------------------------
//  VSA Basic Algebraic Kernels
// ----------------------------------------------------------------------------

__global__ void vsa_bind_kernel(
    const float* __restrict__ d_a,
    const float* __restrict__ d_b,
    float*       __restrict__ d_out,
    int          dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        d_out[idx] = d_a[idx] * d_b[idx];
    }
}

__global__ void vsa_bundle_kernel(
    const float* __restrict__ d_a,
    const float* __restrict__ d_b,
    float*       __restrict__ d_out,
    int          dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        d_out[idx] = 0.5f * (d_a[idx] + d_b[idx]);
    }
}

__global__ void vsa_permute_kernel(
    const float* __restrict__ d_in,
    float*       __restrict__ d_out,
    int          shift,
    int          dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        int target = (idx + shift) % dim;
        if (target < 0) target += dim;
        d_out[target] = d_in[idx];
    }
}

// ----------------------------------------------------------------------------
//  Modern Dense Hopfield Kernels (1 Warp per memory slot)
// ----------------------------------------------------------------------------

__global__ void hopfield_dot_products_kernel(
    const float* __restrict__ d_query,
    const float* __restrict__ d_keys,
    float*       __restrict__ d_similarities,
    int          count,
    int          dim
) {
    // Warp ID = Memory slot ID
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane_id = threadIdx.x & 31;

    if (warp_id >= count) return;

    const int k_base = warp_id * dim;
    float dot = 0.0f;

    // 32 threads in warp process dim=1024 floats (32 floats per thread)
    #pragma unroll 4
    for (int i = lane_id; i < dim; i += 32) {
        dot += d_query[i] * d_keys[k_base + i];
    }

    // Warp reduction via __shfl_down_sync
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xFFFFFFFFu, dot, offset);
    }

    if (lane_id == 0) {
        // True cosine dot product for unit vectors
        d_similarities[warp_id] = dot;
    }
}

__global__ void hopfield_softmax_weighted_recall_kernel(
    const float* __restrict__ d_similarities,
    const float* __restrict__ d_values,
    float*       __restrict__ d_recalled,
    int          count,
    int          dim,
    float        beta
) {
    const int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (dim_idx >= dim) return;

    if (count == 0) {
        d_recalled[dim_idx] = 0.0f;
        return;
    }

    // 1. Find max similarity for numeric stability
    float max_sim = -1e9f;
    for (int m = 0; m < count; ++m) {
        if (d_similarities[m] > max_sim) {
            max_sim = d_similarities[m];
        }
    }

    // 2. Compute softmax sum and weighted vector sum
    float sum_exp = 0.0f;
    float weighted_acc = 0.0f;

    for (int m = 0; m < count; ++m) {
        const float exp_val = __expf(beta * (d_similarities[m] - max_sim));
        sum_exp += exp_val;
        weighted_acc += exp_val * d_values[m * dim + dim_idx];
    }

    d_recalled[dim_idx] = (sum_exp > 1e-8f) ? (weighted_acc / sum_exp) : 0.0f;
}

// ----------------------------------------------------------------------------
//  EpisodicMemory Class Implementation
// ----------------------------------------------------------------------------

EpisodicMemory::EpisodicMemory(std::uint32_t capacity)
    : capacity_(capacity)
{
    state_.current_count = 0;
    cudaStreamCreateWithFlags(&state_.stream, cudaStreamNonBlocking);

    const std::size_t size_matrix = capacity_ * VSA_DIM * sizeof(float);
    const std::size_t size_vector = VSA_DIM * sizeof(float);
    const std::size_t size_sims   = capacity_ * sizeof(float);

    cudaMalloc(&state_.d_keys,         size_matrix);
    cudaMalloc(&state_.d_values,       size_matrix);
    cudaMalloc(&state_.d_similarities, size_sims);
    cudaMalloc(&state_.d_query,        size_vector);
    cudaMalloc(&state_.d_recalled,     size_vector);
    cudaMalloc(&state_.d_count,        sizeof(std::uint32_t));

    cudaMemsetAsync(state_.d_keys,         0, size_matrix, state_.stream);
    cudaMemsetAsync(state_.d_values,       0, size_matrix, state_.stream);
    cudaMemsetAsync(state_.d_similarities, 0, size_sims,   state_.stream);
    cudaMemsetAsync(state_.d_count,        0, sizeof(std::uint32_t), state_.stream);
    cudaStreamSynchronize(state_.stream);
}

EpisodicMemory::~EpisodicMemory() {
    if (state_.stream) {
        cudaStreamSynchronize(state_.stream);
        cudaStreamDestroy(state_.stream);
    }
    if (state_.d_keys)         cudaFree(state_.d_keys);
    if (state_.d_values)       cudaFree(state_.d_values);
    if (state_.d_similarities) cudaFree(state_.d_similarities);
    if (state_.d_query)        cudaFree(state_.d_query);
    if (state_.d_recalled)     cudaFree(state_.d_recalled);
    if (state_.d_count)        cudaFree(state_.d_count);
}

EpisodicMemory::EpisodicMemory(EpisodicMemory&& other) noexcept
    : state_(other.state_), capacity_(other.capacity_)
{
    other.state_ = {};
    other.capacity_ = 0;
}

EpisodicMemory& EpisodicMemory::operator=(EpisodicMemory&& other) noexcept {
    if (this != &other) {
        this->~EpisodicMemory();
        state_ = other.state_;
        capacity_ = other.capacity_;
        other.state_ = {};
        other.capacity_ = 0;
    }
    return *this;
}

void EpisodicMemory::store(std::span<const float, VSA_DIM> key,
                           std::span<const float, VSA_DIM> value) {
    if (state_.current_count >= capacity_) {
        // Ring-buffer overwrite when full
        state_.current_count = 0;
    }

    const std::uint32_t slot = state_.current_count;
    const std::size_t offset = slot * VSA_DIM * sizeof(float);

    cudaMemcpyAsync(reinterpret_cast<char*>(state_.d_keys) + offset,
                    key.data(), VSA_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, state_.stream);

    cudaMemcpyAsync(reinterpret_cast<char*>(state_.d_values) + offset,
                    value.data(), VSA_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, state_.stream);

    state_.current_count++;
    cudaStreamSynchronize(state_.stream);
}

float EpisodicMemory::recall(std::span<const float, VSA_DIM> query,
                             std::span<float, VSA_DIM> recalled_out,
                             float beta) {
    if (state_.current_count == 0) {
        std::fill(recalled_out.begin(), recalled_out.end(), 0.0f);
        return 0.0f;
    }

    cudaMemcpyAsync(state_.d_query, query.data(), VSA_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, state_.stream);

    // 1. Compute dot products across all stored slots
    const int count = state_.current_count;
    const int warps_needed = count;
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int blocks_sim = (warps_needed + warps_per_block - 1) / warps_per_block;

    hopfield_dot_products_kernel<<<blocks_sim, threads_per_block, 0, state_.stream>>>(
        state_.d_query, state_.d_keys, state_.d_similarities, count, VSA_DIM
    );

    // 2. Compute softmax weighted recall
    const int blocks_recall = (VSA_DIM + 255) / 256;
    hopfield_softmax_weighted_recall_kernel<<<blocks_recall, 256, 0, state_.stream>>>(
        state_.d_similarities, state_.d_values, state_.d_recalled, count, VSA_DIM, beta
    );

    cudaMemcpyAsync(recalled_out.data(), state_.d_recalled, VSA_DIM * sizeof(float),
                    cudaMemcpyDeviceToHost, state_.stream);
    cudaStreamSynchronize(state_.stream);

    return 1.0f;
}

void EpisodicMemory::bind(const float* d_a, const float* d_b, float* d_out, cudaStream_t stream) {
    vsa_bind_kernel<<<(VSA_DIM + 255) / 256, 256, 0, stream>>>(d_a, d_b, d_out, VSA_DIM);
}

void EpisodicMemory::bundle(const float* d_a, const float* d_b, float* d_out, cudaStream_t stream) {
    vsa_bundle_kernel<<<(VSA_DIM + 255) / 256, 256, 0, stream>>>(d_a, d_b, d_out, VSA_DIM);
}

void EpisodicMemory::permute(const float* d_in, float* d_out, int shift, cudaStream_t stream) {
    vsa_permute_kernel<<<(VSA_DIM + 255) / 256, 256, 0, stream>>>(d_in, d_out, shift, VSA_DIM);
}

}  // namespace guim
