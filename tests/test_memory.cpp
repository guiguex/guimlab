// =============================================================================
//  tests/test_memory.cpp  -  Long-run memory stability & no-leak checks.
// =============================================================================
#include "guim.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <vector>

namespace {

constexpr std::uint64_t kFrames = 100'000;

bool get_free_bytes(std::size_t& free_b, std::size_t& total_b)
{
    std::size_t f = 0, t = 0;
    cudaError_t e = cudaMemGetInfo(&f, &t);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "cudaMemGetInfo: %s\n", cudaGetErrorString(e));
        return false;
    }
    free_b  = f;
    total_b = t;
    return true;
}

void fill_random(std::vector<float>& v, std::uint32_t seed)
{
    std::uint64_t s = seed * 268582165355123ULL + 1ULL;
    for (auto& x : v) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        std::uint32_t bits = static_cast<std::uint32_t>((s >> 40) & 0xffffffu);
        x = (bits / 8388607.5f) - 1.0f;
    }
}

}  // namespace

TEST(GuimMemory, ArenaSizeIsStable)
{
    guim::Engine engine{42u};
    const std::size_t before = engine.bytes_resident();
    EXPECT_GT(before, 0u);

    std::vector<float> in(GUIM_INPUT_DIM);
    std::vector<float> out(GUIM_STATE_DIM);
    fill_random(in, 1u);

    for (std::uint64_t i = 0; i < 1'000; ++i) {
        (void)engine.step(
            std::span<const float, GUIM_INPUT_DIM>(in.data(), GUIM_INPUT_DIM),
            std::span<float, GUIM_STATE_DIM>(out.data(), GUIM_STATE_DIM));
        in[0] = std::sin(static_cast<float>(i) * 0.001f);
    }
    EXPECT_EQ(engine.bytes_resident(), before);
}

TEST(GuimMemory, NoDeviceLeakOverManyFrames)
{
    std::size_t free_before = 0, total_before = 0;
    ASSERT_TRUE(get_free_bytes(free_before, total_before));

    guim::Engine engine{99u};
    std::vector<float> in(GUIM_INPUT_DIM);
    std::vector<float> out(GUIM_STATE_DIM);
    fill_random(in, 17u);

    for (std::uint64_t i = 0; i < kFrames; ++i) {
        in[0] = std::sin(static_cast<float>(i) * 0.001f);
        float err = engine.step(
            std::span<const float, GUIM_INPUT_DIM>(in.data(), GUIM_INPUT_DIM),
            std::span<float, GUIM_STATE_DIM>(out.data(), GUIM_STATE_DIM));
        ASSERT_TRUE(std::isfinite(err));
        ASSERT_TRUE(std::isfinite(out[0]));
    }
    EXPECT_EQ(engine.frames_run(), kFrames);

    std::size_t free_after = 0, total_after = 0;
    ASSERT_TRUE(get_free_bytes(free_after, total_after));

    constexpr std::size_t kSlack = 4ull * 1024 * 1024;
    if (free_before > free_after + kSlack) {
        FAIL() << "device free memory dropped from " << free_before
               << " to " << free_after << " (delta "
               << (free_before - free_after) << " bytes)";
    }
}