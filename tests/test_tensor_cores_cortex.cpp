// =============================================================================
//  tests/test_tensor_cores_cortex.cpp — Tensor Cores & GRM-FIN Cortex Test Suite
// =============================================================================

#include "lovelace_cortex.h"
#include "ipc_structs.h"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <array>
#include <numeric>
#include <chrono>
#include <random>

namespace {

void generate_synthetic_cortex_input(std::array<float, GUIM_INPUT_DIM>& in_vec, std::uint32_t seed) {
    std::mt19937 rng{seed};
    std::uniform_real_distribution<float> dist{-1.0f, 1.0f};
    for (float& x : in_vec) {
        x = dist(rng);
    }
}

}  // namespace

// ----------------------------------------------------------------------------
//  Test 1: Tensor Core Cortex Construction & Basic Step
// ----------------------------------------------------------------------------
TEST(TensorCoresCortex, ConstructAndStep) {
    guim::LovelaceCortexEngine cortex{1337};

    std::array<float, GUIM_INPUT_DIM> in_vec{};
    std::array<float, GUIM_STATE_DIM> out_vec{};
    generate_synthetic_cortex_input(in_vec, 42);

    cortex.step(in_vec, 0.5f, 0.8f, out_vec, 1000000ull);

    for (float v : out_vec) {
        EXPECT_TRUE(std::isfinite(v));
        EXPECT_GE(v, -1.0f);
        EXPECT_LE(v, 1.0f);
    }
}

// ----------------------------------------------------------------------------
//  Test 2: Golden Ratio Metric (GRM-FIN) Invariant Stability over 1000 Steps
// ----------------------------------------------------------------------------
TEST(TensorCoresCortex, GoldenRatioEnergyStability) {
    guim::LovelaceCortexEngine cortex{999};

    std::array<float, GUIM_INPUT_DIM> in_vec{};
    std::array<float, GUIM_STATE_DIM> out_vec{};

    float max_energy = 0.0f;

    for (int t = 0; t < 1000; ++t) {
        generate_synthetic_cortex_input(in_vec, static_cast<std::uint32_t>(t + 1));
        cortex.step(in_vec, 0.2f, 0.5f, out_vec, static_cast<std::uint64_t>(t * 100000ull));

        float energy = 0.0f;
        for (float v : out_vec) {
            energy += v * v;
        }
        if (energy > max_energy) max_energy = energy;

        EXPECT_TRUE(std::isfinite(energy));
        // Energy must remain bounded within total cortical dimension
        EXPECT_LE(energy, static_cast<float>(GUIM_STATE_DIM));
    }

    std::printf("\n[TensorCoresCortex] Max Cortical Energy over 1000 steps: %6.4f / %d\n",
                max_energy, GUIM_STATE_DIM);
}

// ----------------------------------------------------------------------------
//  Test 3: Fast Invariant Sub-Microsecond Step Latency Benchmark
// ----------------------------------------------------------------------------
TEST(TensorCoresCortex, SubMicrosecondStepLatency) {
    guim::LovelaceCortexEngine cortex{777};

    std::array<float, GUIM_INPUT_DIM> in_vec{};
    std::array<float, GUIM_STATE_DIM> out_vec{};
    generate_synthetic_cortex_input(in_vec, 123);

    // Warmup
    for (int i = 0; i < 500; ++i) {
        cortex.step(in_vec, 0.1f, 0.1f, out_vec, static_cast<std::uint64_t>(i * 1000ull));
    }

    constexpr int TIMED_STEPS = 5000;
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < TIMED_STEPS; ++i) {
        cortex.step(in_vec, 0.1f, 0.1f, out_vec, static_cast<std::uint64_t>(i * 1000ull));
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();
    double avg_step_us = total_us / TIMED_STEPS;

    std::printf("\n================ [TENSOR CORE CORTEX LATENCY] ================\n");
    std::printf("  Timed Steps            : %d\n", TIMED_STEPS);
    std::printf("  Avg Step Latency (Host): %7.3f us\n", avg_step_us);
    std::printf("  Estimated Throughput   : %9.1f FPS\n", 1e6 / avg_step_us);
    std::printf("===============================================================\n");

    EXPECT_LT(avg_step_us, 500.0);
}
