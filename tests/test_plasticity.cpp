// =============================================================================
//  tests/test_plasticity.cpp  -  CBP plasticity-reset path smoke test.
// =============================================================================
#include "guim.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <vector>

TEST(GuimPlasticity, CounterIsMonotonic)
{
    guim::Engine engine{1u};
    std::vector<float> in(GUIM_INPUT_DIM, 0.0f);
    std::vector<float> out(GUIM_STATE_DIM, 0.0f);

    std::uint32_t prev = engine.resets_count();
    for (int i = 0; i < 32; ++i) {
        (void)engine.step(
            std::span<const float, GUIM_INPUT_DIM>(in.data(), GUIM_INPUT_DIM),
            std::span<float, GUIM_STATE_DIM>(out.data(), GUIM_STATE_DIM));
        EXPECT_GE(engine.resets_count(), prev);
        prev = engine.resets_count();
    }
}

TEST(GuimPlasticity, BadInputsAlwaysReset)
{
    guim::Engine engine{2u};
    std::vector<float> in(GUIM_INPUT_DIM, 0.0f);
    std::vector<float> out(GUIM_STATE_DIM, 0.0f);

    // First call: NaN input.
    in[0] = std::nanf("");
    (void)engine.step(
        std::span<const float, GUIM_INPUT_DIM>(in.data(), GUIM_INPUT_DIM),
        std::span<float, GUIM_STATE_DIM>(out.data(), GUIM_STATE_DIM));
    auto r1 = engine.resets_count();

    // Second call: +Inf input.
    in[0] = 0.0f;
    in[5] = std::numeric_limits<float>::infinity();
    (void)engine.step(
        std::span<const float, GUIM_INPUT_DIM>(in.data(), GUIM_INPUT_DIM),
        std::span<float, GUIM_STATE_DIM>(out.data(), GUIM_STATE_DIM));
    auto r2 = engine.resets_count();

    EXPECT_EQ(r1, 1u);
    EXPECT_EQ(r2, 2u);
}