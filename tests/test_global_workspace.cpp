// =============================================================================
//  tests/test_global_workspace.cpp — Tests for Global Workspace Theory (GWT)
// =============================================================================

#include "global_workspace.h"

#include <gtest/gtest.h>
#include <cmath>
#include <array>
#include <algorithm>

TEST(GlobalWorkspace, ConstructAndArbitrate) {
    guim::GlobalWorkspace gwt;

    std::array<float, guim::GWT_IN_DIM> prop0{};
    std::array<float, guim::GWT_IN_DIM> prop1{};
    std::array<float, guim::GWT_IN_DIM> prop2{};
    std::array<float, guim::GWT_IN_DIM> prop3{};

    std::fill(prop0.begin(), prop0.end(), 0.1f);
    std::fill(prop1.begin(), prop1.end(), 0.2f);
    std::fill(prop2.begin(), prop2.end(), 0.8f);  // Module 2 has high value
    std::fill(prop3.begin(), prop3.end(), 0.4f);

    // Module 2 is given overwhelming salience (e.g. emergency sensory trigger)
    gwt.set_proposal(0, prop0, 0.1f);
    gwt.set_proposal(1, prop1, 0.2f);
    gwt.set_proposal(2, prop2, 10.0f);
    gwt.set_proposal(3, prop3, 0.3f);

    EXPECT_EQ(gwt.get_dominant_module(), 2);

    std::array<float, guim::GWT_BROADCAST_DIM> broadcast{};
    gwt.arbitrate_and_broadcast(broadcast, 10.0f);

    // Broadcast should be close to tanh(prop2) = tanh(0.8) ≈ 0.664
    const float expected = std::tanh(0.8f);
    for (float v : broadcast) {
        EXPECT_NEAR(v, expected, 0.05f);
    }
}
