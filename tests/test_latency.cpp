// =============================================================================
//  tests/test_latency.cpp  -  Wall-clock latency sanity check.
// =============================================================================
#include "guim.h"

#include <chrono>
#include <gtest/gtest.h>
#include <span>
#include <vector>

namespace {

constexpr std::uint64_t kWarmup = 200;
constexpr std::uint64_t kFrames = 2'000;
constexpr double        kMaxMeanUs = 5'000.0;   // 5 ms - very loose upper bound

}  // namespace

TEST(GuimLatency, SustainedStepUnderBound)
{
    guim::Engine engine{99u};
    std::vector<float> input(GUIM_INPUT_DIM, 0.0f);
    std::vector<float> output(GUIM_STATE_DIM, 0.0f);

    auto run = [&](std::uint64_t n) {
        for (std::uint64_t i = 0; i < n; ++i) {
            (void)engine.step(
                std::span<const float, GUIM_INPUT_DIM>(input.data(), GUIM_INPUT_DIM),
                std::span<float, GUIM_STATE_DIM>(output.data(), GUIM_STATE_DIM));
        }
    };

    run(kWarmup);

    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    run(kFrames);
    auto t1 = clock::now();

    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double mean_us = total_us / static_cast<double>(kFrames);

    EXPECT_LT(mean_us, kMaxMeanUs)
        << "mean per-frame latency " << mean_us << " us exceeded soft bound "
        << kMaxMeanUs << " us";

    RecordProperty("mean_us_per_frame", mean_us);
}