// ============================================================================
//  world_model_kernel.cu — CUDA Implementation of Latent World Model & Dyna Rollout
//  ============================================================================
//  Author        : Guimlab Contributors
//  License       : AGPL-3.0 / MIT
// ============================================================================

#include "world_model.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstring>

namespace guim {

// ----------------------------------------------------------------------------
//  World Model CUDA Kernels
// ----------------------------------------------------------------------------

__global__ void wm_init_kernel(
    float*       __restrict__ d_weights,
    float*       __restrict__ d_traces,
    float*       __restrict__ d_reward_w,
    std::uint32_t seed
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < WM_TOTAL_W) {
        const float scale = sqrtf(2.0f / static_cast<float>(WM_TOTAL_IN));
        unsigned long long s = seed ^ (static_cast<unsigned long long>(idx) * 0x9E3779B97F4A7C15ull);
        s = (s ^ (s >> 30)) * 0xBF58476D1CE4E5B9ull;
        s = (s ^ (s >> 27)) * 0x94D049BB133111EBull;
        s = s ^ (s >> 31);
        const float u = (static_cast<float>(s & 0xFFFFFF) / static_cast<float>(0x1000000)) - 0.5f;

        d_weights[idx] = u * 2.0f * scale;
        d_traces[idx]  = 0.0f;
    }

    if (idx < WM_STATE_DIM) {
        d_reward_w[idx] = 0.1f;
    }
}

__global__ void wm_transition_and_learn_kernel(
    const float* __restrict__ d_state,
    const float* __restrict__ d_action,
    const float* __restrict__ d_target_next,
    float*       __restrict__ d_predicted_next,
    float*       __restrict__ d_weights,
    float*       __restrict__ d_traces,
    float*       __restrict__ d_loss_out
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= WM_STATE_DIM) return;

    const int w_base = row * WM_TOTAL_IN;

    // 1. Forward transition pass
    float accum = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < WM_STATE_DIM; ++i) {
        accum += d_weights[w_base + i] * d_state[i];
    }
    #pragma unroll 8
    for (int i = 0; i < WM_ACTION_DIM; ++i) {
        accum += d_weights[w_base + WM_STATE_DIM + i] * d_action[i];
    }

    const float s_pred = tanhf(accum);
    const float dtanh  = 1.0f - s_pred * s_pred;
    d_predicted_next[row] = s_pred;

    // 2. Prediction error
    const float err = d_target_next[row] - s_pred;

    // 3. Online gradient update
    constexpr float LR = 0.01f;
    #pragma unroll 8
    for (int col = 0; col < WM_TOTAL_IN; ++col) {
        const int idx = w_base + col;
        const float in_val = (col < WM_STATE_DIM) ? d_state[col]
                                                  : d_action[col - WM_STATE_DIM];

        // Trace + weight update
        float tr = 0.85f * d_traces[idx] + dtanh * in_val;
        d_traces[idx] = tr;
        d_weights[idx] += LR * err * tr;
    }

    if (row == 0) {
        *d_loss_out = fabsf(err);
    }
}

__global__ void wm_imagination_rollout_kernel(
    const float* __restrict__ d_init_state,
    const float* __restrict__ d_action_seq,
    const float* __restrict__ d_weights,
    const float* __restrict__ d_reward_w,
    float*       __restrict__ d_cum_reward_out,
    int                       horizon,
    float                     gamma
) {
    __shared__ float s_state[WM_STATE_DIM];
    __shared__ float s_next_state[WM_STATE_DIM];
    __shared__ float s_reward_step[WM_STATE_DIM];

    const int tid = threadIdx.x;

    // Load initial state into shared memory
    if (tid < WM_STATE_DIM) {
        s_state[tid] = d_init_state[tid];
    }
    __syncthreads();

    float total_discounted_reward = 0.0f;
    float current_gamma = 1.0f;

    for (int step = 0; step < horizon; ++step) {
        const int act_base = step * WM_ACTION_DIM;

        // Parallel forward transition for current step
        if (tid < WM_STATE_DIM) {
            float accum = 0.0f;
            const int w_base = tid * WM_TOTAL_IN;

            #pragma unroll 8
            for (int i = 0; i < WM_STATE_DIM; ++i) {
                accum += d_weights[w_base + i] * s_state[i];
            }
            #pragma unroll 8
            for (int i = 0; i < WM_ACTION_DIM; ++i) {
                accum += d_weights[w_base + WM_STATE_DIM + i] * d_action_seq[act_base + i];
            }

            const float s_next = tanhf(accum);
            s_next_state[tid]  = s_next;
            s_reward_step[tid] = s_next * d_reward_w[tid];
        }
        __syncthreads();

        // Accumulate reward across threads
        if (tid == 0) {
            float step_r = 0.0f;
            for (int i = 0; i < WM_STATE_DIM; ++i) {
                step_r += s_reward_step[i];
            }
            total_discounted_reward += current_gamma * step_r;
            current_gamma *= gamma;
        }

        // Advance state
        if (tid < WM_STATE_DIM) {
            s_state[tid] = s_next_state[tid];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *d_cum_reward_out = total_discounted_reward;
    }
}

// ----------------------------------------------------------------------------
//  WorldModel Class Implementation
// ----------------------------------------------------------------------------

WorldModel::WorldModel(std::uint32_t seed) {
    cudaStreamCreateWithFlags(&state_.stream, cudaStreamNonBlocking);

    cudaMalloc(&state_.d_weights,       WM_TOTAL_W * sizeof(float));
    cudaMalloc(&state_.d_traces,        WM_TOTAL_W * sizeof(float));
    cudaMalloc(&state_.d_reward_w,      WM_STATE_DIM * sizeof(float));
    cudaMalloc(&state_.d_latent_state,  WM_STATE_DIM * sizeof(float));
    cudaMalloc(&state_.d_imagined_traj, WM_MAX_HORIZON * WM_STATE_DIM * sizeof(float));
    cudaMalloc(&state_.d_action_seq,    WM_MAX_HORIZON * WM_ACTION_DIM * sizeof(float));
    cudaMalloc(&state_.d_cum_reward,    sizeof(float));

    constexpr int THREADS = 256;
    const int blocks = (WM_TOTAL_W + THREADS - 1) / THREADS;
    wm_init_kernel<<<blocks, THREADS, 0, state_.stream>>>(
        state_.d_weights, state_.d_traces, state_.d_reward_w, seed
    );
    cudaStreamSynchronize(state_.stream);
}

WorldModel::~WorldModel() {
    if (state_.stream) {
        cudaStreamSynchronize(state_.stream);
        cudaStreamDestroy(state_.stream);
    }
    if (state_.d_weights)       cudaFree(state_.d_weights);
    if (state_.d_traces)        cudaFree(state_.d_traces);
    if (state_.d_reward_w)      cudaFree(state_.d_reward_w);
    if (state_.d_latent_state)  cudaFree(state_.d_latent_state);
    if (state_.d_imagined_traj) cudaFree(state_.d_imagined_traj);
    if (state_.d_action_seq)    cudaFree(state_.d_action_seq);
    if (state_.d_cum_reward)    cudaFree(state_.d_cum_reward);
}

WorldModel::WorldModel(WorldModel&& other) noexcept
    : state_(other.state_)
{
    other.state_ = {};
}

WorldModel& WorldModel::operator=(WorldModel&& other) noexcept {
    if (this != &other) {
        this->~WorldModel();
        state_ = other.state_;
        other.state_ = {};
    }
    return *this;
}

float WorldModel::step(
    std::span<const float, WM_STATE_DIM> state,
    std::span<const float, WM_ACTION_DIM> action,
    std::span<const float, WM_STATE_DIM> next_state_target,
    float reward_target,
    std::span<float, WM_STATE_DIM> predicted_next_state
) {
    (void)reward_target;
    float* d_in_s = state_.d_latent_state;
    float* d_in_a = state_.d_action_seq;
    float* d_next_tgt = state_.d_imagined_traj;
    float* d_pred = state_.d_imagined_traj + WM_STATE_DIM;
    float* d_loss = state_.d_cum_reward;

    cudaMemcpyAsync(d_in_s, state.data(), WM_STATE_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, state_.stream);
    cudaMemcpyAsync(d_in_a, action.data(), WM_ACTION_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, state_.stream);
    cudaMemcpyAsync(d_next_tgt, next_state_target.data(), WM_STATE_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, state_.stream);

    wm_transition_and_learn_kernel<<<1, WM_STATE_DIM, 0, state_.stream>>>(
        d_in_s, d_in_a, d_next_tgt, d_pred, state_.d_weights, state_.d_traces, d_loss
    );

    float loss_val = 0.0f;
    cudaMemcpyAsync(predicted_next_state.data(), d_pred, WM_STATE_DIM * sizeof(float),
                    cudaMemcpyDeviceToHost, state_.stream);
    cudaMemcpyAsync(&loss_val, d_loss, sizeof(float),
                    cudaMemcpyDeviceToHost, state_.stream);
    cudaStreamSynchronize(state_.stream);

    return loss_val;
}

float WorldModel::imagine_rollout(
    std::span<const float, WM_STATE_DIM> initial_state,
    std::span<const float> candidate_actions,
    int horizon,
    float gamma
) {
    if (horizon <= 0 || horizon > WM_MAX_HORIZON) return 0.0f;

    cudaMemcpyAsync(state_.d_latent_state, initial_state.data(), WM_STATE_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, state_.stream);
    cudaMemcpyAsync(state_.d_action_seq, candidate_actions.data(),
                    horizon * WM_ACTION_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, state_.stream);

    wm_imagination_rollout_kernel<<<1, WM_STATE_DIM, 0, state_.stream>>>(
        state_.d_latent_state, state_.d_action_seq, state_.d_weights,
        state_.d_reward_w, state_.d_cum_reward, horizon, gamma
    );

    float cum_reward = 0.0f;
    cudaMemcpyAsync(&cum_reward, state_.d_cum_reward, sizeof(float),
                    cudaMemcpyDeviceToHost, state_.stream);
    cudaStreamSynchronize(state_.stream);

    return cum_reward;
}

}  // namespace guim
