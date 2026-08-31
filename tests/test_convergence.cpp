// =============================================================================
//  tests/test_convergence.cpp  -  Synthetic sine-prediction learning test.
// =============================================================================
#include "guim/guim.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <vector>

namespace {

constexpr int kFrames = 5'000;
constexpr int kDelay  = 8;
constexpr int kHalfIn = GUIM_INPUT_DIM / 2;

void fill_sine_frame(std::vector<float>& input, int t, int delay)
{
    const float cur = std::sin(static_cast<float>(t)        * 0.01f);
    const float del = std::sin(static_cast<float>(t - delay) * 0.01f);
    for (int i = 0; i < kHalfIn; ++i) {
        input[i]           = cur * (0.6f + 0.4f * std::sin(static_cast<float>(i)));
        input[kHalfIn + i] = del * (0.6f + 0.4f * std::cos(static_cast<float>(i)));
    }
}

}  // namespace

TEST(GuimConvergence, FramesRunIsMonotonic)
{
    guim::Engine engine{123u};
    std::vector<float> input(GUIM_INPUT_DIM);
    std::vector<float> output(GUIM_STATE_DIM);

    std::uint64_t prev = engine.frames_run();
    for (int t = 0; t < 1000; ++t) {
        fill_sine_frame(input, t, kDelay);
        engine.step(
            std::span<const float, GUIM_INPUT_DIM>(input.data(), GUIM_INPUT_DIM),
            std::span<float, GUIM_STATE_DIM>(output.data(), GUIM_STATE_DIM));
        EXPECT_GE(engine.frames_run(), prev + 1);
        prev = engine.frames_run();
    }
    EXPECT_EQ(engine.frames_run(), 1000u);
}

TEST(GuimConvergence, HiddenStaysFiniteOverFrames)
{
    guim::Engine engine{0xCAFEu};
    std::vector<float> input(GUIM_INPUT_DIM);
    std::vector<float> output(GUIM_STATE_DIM);

    for (int t = 0; t < kFrames; ++t) {
        fill_sine_frame(input, t, kDelay);
        float err = engine.step(
            std::span<const float, GUIM_INPUT_DIM>(input.data(), GUIM_INPUT_DIM),
            std::span<float, GUIM_STATE_DIM>(output.data(), GUIM_STATE_DIM));

        ASSERT_TRUE(std::isfinite(err));
        for (int i = 0; i < GUIM_STATE_DIM; ++i) {
            ASSERT_TRUE(std::isfinite(output[i]));
        }
    }
}

TEST(GuimConvergence, ValidInputsProduceFiniteUpdates)
{
    guim::Engine engine{7u};
    std::vector<float> input(GUIM_INPUT_DIM);
    std::vector<float> output(GUIM_STATE_DIM);

    for (int t = 0; t < 512; ++t) {
        fill_sine_frame(input, t, kDelay);
        float td = engine.step(
            std::span<const float, GUIM_INPUT_DIM>(input.data(), GUIM_INPUT_DIM),
            std::span<float, GUIM_STATE_DIM>(output.data(), GUIM_STATE_DIM));
        EXPECT_TRUE(std::isfinite(td));
    }
}