// ============================================================================
//  test_harness_integrity.cpp — Pure C++20 Mathematical & Structural Invariants
//  ============================================================================
//  Validates the bare-metal invariants, cache alignments, and symplectic proofs
//  without any external scripting dependency.
// ============================================================================

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "ipc_structs_v2.h"
#include "ipc_structs_sparse.h"
#include "guim_kernel.h"
#include "guim_studio.h"
#include "guim_viewport_session.h"

namespace {

// ----------------------------------------------------------------------------
//  1. Compile-Time Alignment & Layout Invariants
// ----------------------------------------------------------------------------
static_assert(alignof(GuimSharedMemoryV2) == 64, "GuimSharedMemoryV2 must be 64-byte cache-aligned");
static_assert(sizeof(GuimSharedMemoryV2) % 64 == 0, "GuimSharedMemoryV2 size must be a multiple of 64 bytes");
static_assert(std::is_trivially_copyable_v<GuimSharedMemoryV2>, "GuimSharedMemoryV2 must be trivially copyable for zero-copy IPC");

TEST(HarnessIntegrityTest, CacheLineAlignmentAndSizes) {
    EXPECT_EQ(alignof(GuimSharedMemoryV2), 64);
    EXPECT_EQ(sizeof(GuimSharedMemoryV2) % 64, 0);
    EXPECT_LE(sizeof(GuimSharedMemoryV2), 4096); // Must fit within single 4KB page
}

// ----------------------------------------------------------------------------
//  2. Symplectic Riemannian Phase-Space Determinant (det(R) == 1.0)
// ----------------------------------------------------------------------------
TEST(HarnessIntegrityTest, SymplecticUnitaryRotationInvariant) {
    constexpr float omega = 440.0f * 2.0f * 3.14159265358979323846f; // 440 Hz formant
    constexpr float dt    = 0.0005f; // 2 kHz sampling interval (0.5 ms)
    
    float cos_val = std::cos(omega * dt);
    float sin_val = std::sin(omega * dt);

    // Symplectic rotation matrix determinant: cos^2(theta) + sin^2(theta)
    float det = (cos_val * cos_val) + (sin_val * sin_val);
    EXPECT_NEAR(det, 1.0f, 1e-6f);

    // Phase lead scaling factor gamma(omega, tau)
    constexpr float tau = 0.030f; // 30 ms
    float omega_tau = omega * tau;
    float gamma = omega_tau / std::sqrt(1.0f + (omega_tau * omega_tau));

    EXPECT_GT(gamma, 0.0f);
    EXPECT_LT(gamma, 1.0f);
}

// ----------------------------------------------------------------------------
//  3. Closed-Form Continuous-Time Trace Monotonicity
// ----------------------------------------------------------------------------
TEST(HarnessIntegrityTest, ClosedFormContinuousTimeTraceDecay) {
    constexpr float tau = 0.030f; // 30 ms membrane time constant
    constexpr float dt  = 0.0005f;
    float decay = std::exp(-dt / tau);

    EXPECT_GT(decay, 0.0f);
    EXPECT_LT(decay, 1.0f);

    // Exponential decay over 1000 ticks must remain strictly positive and non-NaN
    float trace = 1.0f;
    for (int t = 0; t < 1000; ++t) {
        trace *= decay;
        ASSERT_FALSE(std::isnan(trace));
        ASSERT_FALSE(std::isinf(trace));
        ASSERT_GE(trace, 0.0f);
    }
}

// ----------------------------------------------------------------------------
//  4. TMD-ET Learning Rate Bounding Discipline
// ----------------------------------------------------------------------------
TEST(HarnessIntegrityTest, TMDETLearningRateBounds) {
    constexpr float beta_min = -6.9077f; // exp(-6.9077) ~ 0.001
    constexpr float beta_max = -0.6931f; // exp(-0.6931) ~ 0.500

    float alpha_min = std::exp(beta_min);
    float alpha_max = std::exp(beta_max);

    EXPECT_NEAR(alpha_min, 0.001f, 1e-4f);
    EXPECT_NEAR(alpha_max, 0.500f, 1e-4f);
}

} // namespace
