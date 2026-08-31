// =============================================================================
//  tests/test_correctness.cpp  -  Numerical-correctness sanity tests.
// =============================================================================
#include "guim.h"

#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <limits>
#include <span>
#include <vector>
#include <array>
#include <algorithm>

namespace {

std::vector<float> make_finite_input(std::uint32_t seed)
{
    std::mt19937 rng{seed};
    std::normal_distribution<float> dist{0.0f, 1.0f};
    std::vector<float> v(GUIM_INPUT_DIM);
    for (auto& x : v) x = dist(rng);
    return v;
}

}  // namespace

TEST(GuimCorrectness, ConstructAndDestroy)
{
    guim::Engine engine{42u};
    EXPECT_NE(engine.handle(), nullptr);
    EXPECT_EQ(engine.frames_run(), 0u);
    EXPECT_EQ(engine.resets_count(), 0u);
    EXPECT_GT(engine.bytes_resident(), 0u);
}

TEST(GuimCorrectness, StepIncrementsCounter)
{
    guim::Engine engine{42u};
    auto input = make_finite_input(1);
    std::vector<float> output(GUIM_STATE_DIM);

    for (int i = 0; i < 16; ++i) {
        float err = engine.step(
            std::span<const float, GUIM_INPUT_DIM>(input.data(), GUIM_INPUT_DIM),
            std::span<float, GUIM_STATE_DIM>(output.data(), GUIM_STATE_DIM));
        EXPECT_TRUE(std::isfinite(err));
    }
    EXPECT_EQ(engine.frames_run(), 16u);
}

TEST(GuimCorrectness, NaNInputTriggersReset)
{
    guim::Engine engine{7u};
    std::vector<float> bad_input(GUIM_INPUT_DIM, 0.0f);
    bad_input[3] = std::nanf("");
    std::vector<float> output(GUIM_STATE_DIM);

    float err = engine.step(
        std::span<const float, GUIM_INPUT_DIM>(bad_input.data(), GUIM_INPUT_DIM),
        std::span<float, GUIM_STATE_DIM>(output.data(), GUIM_STATE_DIM));

    EXPECT_TRUE(std::isnan(err));
    EXPECT_GE(engine.resets_count(), 1u);
    for (float v : output) EXPECT_TRUE(std::isnan(v));
}

TEST(GuimCorrectness, InfInputTriggersReset)
{
    guim::Engine engine{7u};
    std::vector<float> bad_input(GUIM_INPUT_DIM, 0.0f);
    bad_input[10] = std::numeric_limits<float>::infinity();
    std::vector<float> output(GUIM_STATE_DIM);

    float err = engine.step(
        std::span<const float, GUIM_INPUT_DIM>(bad_input.data(), GUIM_INPUT_DIM),
        std::span<float, GUIM_STATE_DIM>(output.data(), GUIM_STATE_DIM));

    EXPECT_TRUE(std::isnan(err));
    EXPECT_GE(engine.resets_count(), 1u);
}

TEST(GuimCorrectness, OutputSpanIsFullyWritten)
{
    guim::Engine engine{0xDEADBEEFu};
    auto input = make_finite_input(2);
    std::vector<float> output(GUIM_STATE_DIM);

    std::fill(output.begin(), output.end(), 42.0f);

    float err = engine.step(
        std::span<const float, GUIM_INPUT_DIM>(input.data(), GUIM_INPUT_DIM),
        std::span<float, GUIM_STATE_DIM>(output.data(), GUIM_STATE_DIM));

    EXPECT_TRUE(std::isfinite(err));
    bool touched = false;
    for (float v : output) if (v != 42.0f && std::isfinite(v)) { touched = true; break; }
    EXPECT_TRUE(touched);
}

TEST(GuimCorrectness, CopyHiddenAndWeights)
{
    guim::Engine engine{123u};
    auto input = make_finite_input(3);
    std::vector<float> output(GUIM_STATE_DIM);
    (void)engine.step(
        std::span<const float, GUIM_INPUT_DIM>(input.data(), GUIM_INPUT_DIM),
        std::span<float, GUIM_STATE_DIM>(output.data(), GUIM_STATE_DIM));

    std::array<float, GUIM_STATE_DIM> hidden{};
    std::array<float, GUIM_STATE_DIM * GUIM_TOTAL_DIM> weights{};
    engine.copy_hidden(hidden);
    engine.copy_weights(weights);

    bool non_zero_weights = false;
    for (float w : weights) if (w != 0.0f) { non_zero_weights = true; break; }
    EXPECT_TRUE(non_zero_weights);
}