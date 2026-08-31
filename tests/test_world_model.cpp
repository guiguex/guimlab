// =============================================================================
//  tests/test_world_model.cpp — Tests for Latent World Model & Imagination Rollouts
// =============================================================================

#include "world_model.h"

#include <gtest/gtest.h>
#include <cmath>
#include <array>
#include <vector>
#include <random>

TEST(WorldModel, ConstructAndStep) {
    guim::WorldModel wm{42};

    std::array<float, guim::WM_STATE_DIM> s_t{};
    std::array<float, guim::WM_ACTION_DIM> a_t{};
    std::array<float, guim::WM_STATE_DIM> s_next_target{};
    std::array<float, guim::WM_STATE_DIM> s_pred{};

    std::fill(s_t.begin(), s_t.end(), 0.5f);
    std::fill(a_t.begin(), a_t.end(), 0.1f);
    std::fill(s_next_target.begin(), s_next_target.end(), 0.6f);

    float initial_err = wm.step(s_t, a_t, s_next_target, 1.0f, s_pred);
    EXPECT_TRUE(std::isfinite(initial_err));

    // Train on repetitive transition
    float final_err = initial_err;
    for (int i = 0; i < 50; ++i) {
        final_err = wm.step(s_t, a_t, s_next_target, 1.0f, s_pred);
    }

    // Prediction error must decrease with online learning
    EXPECT_LT(final_err, initial_err);
}

TEST(WorldModel, ImaginationRolloutPureVRAM) {
    guim::WorldModel wm{123};

    std::array<float, guim::WM_STATE_DIM> s0{};
    std::fill(s0.begin(), s0.end(), 0.2f);

    constexpr int HORIZON = 16;
    std::vector<float> action_seq(HORIZON * guim::WM_ACTION_DIM, 0.05f);

    float cumulative_reward = wm.imagine_rollout(s0, action_seq, HORIZON, 0.95f);
    EXPECT_TRUE(std::isfinite(cumulative_reward));
}
