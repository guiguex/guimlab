// ============================================================================
//  test_physics_constants.cpp — Unit Tests for Universal Physics Constants
//  ============================================================================

#include <gtest/gtest.h>
#include "guim_physics_constants.h"
#include <cmath>

namespace {

using namespace guim::physics;

TEST(PhysicsConstantsTest, GoldenRatioKAMProperties) {
    // 1 / Phi == Phi - 1
    EXPECT_NEAR(1.0f / GOLDEN_RATIO_PHI, GOLDEN_RATIO_PHI - 1.0f, 1e-6f);
    // Phi^2 == Phi + 1
    EXPECT_NEAR(GOLDEN_RATIO_PHI * GOLDEN_RATIO_PHI, GOLDEN_RATIO_PHI + 1.0f, 1e-6f);
    // Silver ratio property: (delta_S - 1)^2 == 2
    EXPECT_NEAR((SILVER_RATIO_DELTA - 1.0f) * (SILVER_RATIO_DELTA - 1.0f), 2.0f, 1e-5f);
}

TEST(PhysicsConstantsTest, CoulombYukawaScreeningDecay) {
    float q1 = 1.0f, q2 = 1.0f;
    
    // Near distance: r = 0.1
    float r_near = 0.1f;
    float v_near = COULOMB_K_CONSTANT * (q1 * q2 / (r_near + DIELECTRIC_EPSILON)) * std::exp(-DEBYE_SCREENING_KAPPA * r_near);

    // Far distance: r = 2.0
    float r_far = 2.0f;
    float v_far = COULOMB_K_CONSTANT * (q1 * q2 / (r_far + DIELECTRIC_EPSILON)) * std::exp(-DEBYE_SCREENING_KAPPA * r_far);

    EXPECT_GT(v_near, v_far);
    EXPECT_GT(v_near, 0.0f);
    EXPECT_LT(v_far, 0.1f); // Screened out at distance
}

TEST(PhysicsConstantsTest, FeigenbaumCriticalityDecay) {
    EXPECT_GT(CRITICAL_EMA_HEURISTIC, 0.90f);
    EXPECT_LT(CRITICAL_EMA_HEURISTIC, 0.95f);

    // Verify Lyapunov stability in 1D discrete map near critical parameter
    float lambda_crit = 1.0f + (1.0f / FEIGENBAUM_DELTA);
    EXPECT_GT(lambda_crit, 1.0f);
    EXPECT_LT(lambda_crit, 1.3f);
}

TEST(PhysicsConstantsTest, SymplecticActionQuantumBounds) {
    float omega = 0.0f; // Near zero singular frequency
    float u_dot = 1.0f;
    float tau = 0.030f;

    // Regularized canonical momentum without division by zero
    float p_reg = u_dot / std::sqrt((omega * omega) + (SYNAPTIC_ACTION_HBAR / (tau * tau)));
    
    ASSERT_FALSE(std::isnan(p_reg));
    ASSERT_FALSE(std::isinf(p_reg));
    EXPECT_GT(p_reg, 0.0f);
    EXPECT_LT(p_reg, 1000.0f); // Bounded
}

} // namespace
