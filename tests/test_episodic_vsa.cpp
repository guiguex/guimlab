// =============================================================================
//  tests/test_episodic_vsa.cpp — Tests for Episodic VSA & Hopfield Memory
// =============================================================================

#include "episodic_vsa.h"

#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <array>
#include <vector>

namespace {

void make_random_unit_vector(std::array<float, guim::VSA_DIM>& v, std::uint32_t seed) {
    std::mt19937 rng{seed};
    std::normal_distribution<float> dist{0.0f, 1.0f};
    float norm_sq = 0.0f;
    for (auto& x : v) {
        x = dist(rng);
        norm_sq += x * x;
    }
    float inv_norm = 1.0f / std::sqrt(norm_sq);
    for (auto& x : v) {
        x *= inv_norm;
    }
}

}  // namespace

TEST(EpisodicVSA, ConstructAndStore) {
    guim::EpisodicMemory mem{64};
    EXPECT_EQ(mem.size(), 0u);
    EXPECT_EQ(mem.capacity(), 64u);

    std::array<float, guim::VSA_DIM> k1{}, v1{};
    make_random_unit_vector(k1, 101);
    make_random_unit_vector(v1, 202);

    mem.store(k1, v1);
    EXPECT_EQ(mem.size(), 1u);
}

TEST(EpisodicVSA, ExactAssociativeRecall) {
    guim::EpisodicMemory mem{128};

    std::array<float, guim::VSA_DIM> k1{}, v1{};
    std::array<float, guim::VSA_DIM> k2{}, v2{};
    std::array<float, guim::VSA_DIM> k3{}, v3{};

    make_random_unit_vector(k1, 1);
    make_random_unit_vector(v1, 10);
    make_random_unit_vector(k2, 2);
    make_random_unit_vector(v2, 20);
    make_random_unit_vector(k3, 3);
    make_random_unit_vector(v3, 30);

    mem.store(k1, v1);
    mem.store(k2, v2);
    mem.store(k3, v3);

    // Query with key 2
    std::array<float, guim::VSA_DIM> recalled{};
    mem.recall(k2, recalled, 16.0f);

    // Compute dot product between recalled and true v2
    float dot = 0.0f;
    for (int i = 0; i < guim::VSA_DIM; ++i) {
        dot += recalled[i] * v2[i];
    }

    // Modern Hopfield should achieve high fidelity recall (> 0.90)
    EXPECT_GT(dot, 0.90f);
}

TEST(EpisodicVSA, RobustRecallUnderNoise) {
    guim::EpisodicMemory mem{128};

    std::array<float, guim::VSA_DIM> k1{}, v1{};
    std::array<float, guim::VSA_DIM> k2{}, v2{};

    make_random_unit_vector(k1, 11);
    make_random_unit_vector(v1, 111);
    make_random_unit_vector(k2, 22);
    make_random_unit_vector(v2, 222);

    mem.store(k1, v1);
    mem.store(k2, v2);

    // Add Gaussian noise to k1
    std::array<float, guim::VSA_DIM> noisy_k1 = k1;
    std::mt19937 rng{999};
    std::normal_distribution<float> noise{0.0f, 0.05f};
    for (auto& x : noisy_k1) {
        x += noise(rng);
    }

    std::array<float, guim::VSA_DIM> recalled{};
    mem.recall(noisy_k1, recalled, 12.0f);

    float dot = 0.0f;
    for (int i = 0; i < guim::VSA_DIM; ++i) {
        dot += recalled[i] * v1[i];
    }

    EXPECT_GT(dot, 0.85f);
}
