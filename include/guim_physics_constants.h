// ============================================================================
//  guim_physics_constants.h — Universal Mathematical & Physical Constants
//  ============================================================================
//  Strictly typed, compile-time verified constants for next-gen continuous
//  symplectic dynamics, Yukawa-Coulomb field attention, KAM spectral invariance,
//  and Feigenbaum edge-of-chaos plasticity regulation.
//
//  Author  : GuimLab Research & Engineering
//  License : AGPL-3.0 / MIT
// ============================================================================

#ifndef GUIM_PHYSICS_CONSTANTS_H
#define GUIM_PHYSICS_CONSTANTS_H

#include <cstdint>
#include <type_traits>

namespace guim::physics {

// ----------------------------------------------------------------------------
//  1. Electrostatic & Field Potential Constants (Yukawa-Coulomb Topology)
// ----------------------------------------------------------------------------
// Normalized Coulomb electrostatic constant k_e = 1 / (4 * pi * eps_0)
constexpr float COULOMB_K_CONSTANT      = 8.9875517923e-1f; 
// Debye screening parameter kappa for topological manifold localization
constexpr float DEBYE_SCREENING_KAPPA   = 1.6180339887e0f;
// Dielectric regularizer to avoid singularity at r -> 0
constexpr float DIELECTRIC_EPSILON      = 1.0000000000e-4f;

// ----------------------------------------------------------------------------
//  2. KAM Theory & Quasi-Periodic Spectral Invariants (Kolmogorov-Arnold-Moser)
// ----------------------------------------------------------------------------
// Golden Ratio Phi = (1 + sqrt(5)) / 2 (Most irrational number, max KAM stability)
constexpr float GOLDEN_RATIO_PHI        = 1.6180339887498948482f;
// Golden Ratio Conjugate / Inverse phi = 1 / Phi = Phi - 1
constexpr float GOLDEN_RATIO_INV        = 0.6180339887498948482f;
// Silver Ratio delta_S = 1 + sqrt(2)
constexpr float SILVER_RATIO_DELTA      = 2.4142135623730950488f;
// Plasticity Noble Frequency Exponent
constexpr float KAM_NOBLE_EXPONENT      = 0.6180339887498948482f;

// ----------------------------------------------------------------------------
//  3. Critical Plasticity Heuristics (Empirical — NOT Feigenbaum-derived)
// ----------------------------------------------------------------------------
// Feigenbaum delta: universal period-doubling scaling factor (kept for reference)
//   NOTE: Feigenbaum delta describes bifurcation INTERVAL scaling (mu_n - mu_{n-1})
//   in 1D maps; it is NOT a derivation source for EMA decay constants.
constexpr float FEIGENBAUM_DELTA        = 4.6692016091029906718f;
// Feigenbaum alpha: universal phase-space scaling factor (kept for reference)
//   NOTE: Feigenbaum alpha describes phase-space ATTRACTOR scaling; combining
//   it arithmetically with delta yields a number with no theoretical meaning.
constexpr float FEIGENBAUM_ALPHA        = 2.5029078750958928222f;
// Empirical critical EMA decay constant for CBP variance tracking.
//   HONEST CAVEAT: This value is an arithmetically assembled heuristic
//   (1 - 1/(delta*alpha) ≈ 0.91443), NOT derived from Feigenbaum period-
//   doubling theory. It coincidentally falls in the empirical [0.9, 0.95]
//   band used in critical plasticity literature. The actual Feigenbaum-derived
//   decay rate at the accumulation point is 1 - 1/delta ≈ 0.7858 (Lyapunov
//   decay), which is mathematically different. Treat CRITICAL_EMA_HEURISTIC
//   as an empirical tuning constant; do not cite it as theoretically grounded.
constexpr float CRITICAL_EMA_HEURISTIC  = 1.0f - (1.0f / (FEIGENBAUM_DELTA * FEIGENBAUM_ALPHA)); // ~0.91443

// ----------------------------------------------------------------------------
//  4. Continuous Analysis & Spectral Summation Invariants
// ----------------------------------------------------------------------------
// Euler-Mascheroni constant gamma for continuum harmonic limits
constexpr float EULER_MASCHERONI_GAMMA  = 0.5772156649015328606f;
// Dirac root scale 1 / sqrt(2)
constexpr float INV_SQRT_2              = 0.7071067811865475244f;
// Pi and Pi / 2
constexpr float PI_CONST                = 3.1415926535897932384f;
constexpr float HALF_PI_CONST           = 1.5707963267948966192f;

// ----------------------------------------------------------------------------
//  5. Symplectic Action Quantum & Thermodynamic Noise Bounds
// ----------------------------------------------------------------------------
// Dimensionless symplectic action quantum (prevents phase space orbit singularity)
constexpr float SYNAPTIC_ACTION_HBAR    = 1.054571817e-4f;
// Dimensionless thermal Boltzmann constant for synaptic stochastic resonance
constexpr float BOLTZMANN_THERMAL_KB    = 1.380649000e-3f;

// ----------------------------------------------------------------------------
//  Compile-Time Sanity Assertions
// ----------------------------------------------------------------------------
static_assert(GOLDEN_RATIO_PHI > 1.618f && GOLDEN_RATIO_PHI < 1.619f, "Golden ratio precision fault");
static_assert(GOLDEN_RATIO_INV > 0.618f && GOLDEN_RATIO_INV < 0.619f, "Golden ratio inverse precision fault");
static_assert(CRITICAL_EMA_HEURISTIC > 0.90f && CRITICAL_EMA_HEURISTIC < 0.95f, "Critical EMA heuristic out of bounds");
static_assert(COULOMB_K_CONSTANT > 0.0f, "Coulomb constant must be strictly positive");

} // namespace guim::physics

#endif // GUIM_PHYSICS_CONSTANTS_H
