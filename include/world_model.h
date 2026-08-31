// ============================================================================
//  world_model.h — Generative Latent World Model & Imagination Rollout Engine
//  ============================================================================
//  Implements:
//    1. Online continuous transition dynamics: S_{t+1}, R_{t+1} = M(S_t, A_t)
//    2. Pure-VRAM Imagination Rollouts (Dyna / Active Inference)
//    3. Model-based trajectory evaluation in microsecond budget
//
//  Author        : Guimlab Contributors
//  License       : AGPL-3.0 / MIT
// ============================================================================

#ifndef GUIM_WORLD_MODEL_H
#define GUIM_WORLD_MODEL_H

#include <cuda_runtime.h>
#include <cstdint>
#include <span>

namespace guim {

constexpr int WM_STATE_DIM   = 64;
constexpr int WM_ACTION_DIM  = 16;
constexpr int WM_TOTAL_IN    = WM_STATE_DIM + WM_ACTION_DIM;  // 80
constexpr int WM_TOTAL_W     = WM_STATE_DIM * WM_TOTAL_IN;    // 5120
constexpr int WM_MAX_HORIZON = 32;

struct WorldModelState {
    float*       d_weights       = nullptr;  // [WM_TOTAL_W]
    float*       d_traces        = nullptr;  // [WM_TOTAL_W]
    float*       d_reward_w      = nullptr;  // [WM_STATE_DIM]
    float*       d_latent_state  = nullptr;  // [WM_STATE_DIM]
    float*       d_imagined_traj = nullptr;  // [WM_MAX_HORIZON * WM_STATE_DIM]
    float*       d_action_seq    = nullptr;  // [WM_MAX_HORIZON * WM_ACTION_DIM]
    float*       d_cum_reward    = nullptr;  // [1]
    cudaStream_t stream          = nullptr;
};

class WorldModel {
public:
    explicit WorldModel(std::uint32_t seed = 42);
    ~WorldModel();

    WorldModel(const WorldModel&) = delete;
    WorldModel& operator=(const WorldModel&) = delete;
    WorldModel(WorldModel&& other) noexcept;
    WorldModel& operator=(WorldModel&& other) noexcept;

    // Online single-step transition & learning from real experience
    float step(std::span<const float, WM_STATE_DIM> state,
               std::span<const float, WM_ACTION_DIM> action,
               std::span<const float, WM_STATE_DIM> next_state_target,
               float reward_target,
               std::span<float, WM_STATE_DIM> predicted_next_state);

    // Imagination mode: simulate K steps in pure VRAM without host interaction
    float imagine_rollout(std::span<const float, WM_STATE_DIM> initial_state,
                          std::span<const float> candidate_actions,  // [horizon * WM_ACTION_DIM]
                          int horizon,
                          float gamma = 0.95f);

private:
    WorldModelState state_;
};

}  // namespace guim

#endif // GUIM_WORLD_MODEL_H
