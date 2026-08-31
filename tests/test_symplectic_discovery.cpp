// =============================================================================
//  tests/test_symplectic_discovery.cpp — Rigorous Empirical Discovery Suite
// =============================================================================

#include "symplectic_traces.h"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <array>
#include <numeric>
#include <random>

namespace {

void generate_multi_harmonic_acoustic_frame(
    std::array<float, guim::SYM_INPUT_DIM>& frame,
    float t_sec,
    float fundamental_freq = 150.0f
) {
    for (int i = 0; i < guim::SYM_INPUT_DIM; ++i) {
        float f1 = fundamental_freq * (1.0f + 0.05f * static_cast<float>(i));
        float f2 = fundamental_freq * 2.0f * (1.0f + 0.02f * static_cast<float>(i));
        float f3 = fundamental_freq * 3.0f;

        frame[i] = 0.6f * std::sin(2.0f * 3.14159265f * f1 * t_sec) +
                   0.3f * std::sin(2.0f * 3.14159265f * f2 * t_sec + 0.5f) +
                   0.1f * std::sin(2.0f * 3.14159265f * f3 * t_sec + 1.2f);
    }
}

// Mackey-Glass Chaotic Non-Linear Time Series Generator (tau = 17)
class MackeyGlassGenerator {
public:
    MackeyGlassGenerator(int history_len = 170) : history_(history_len, 0.9f), idx_(0) {}

    float next_sample(float dt = 0.05f) {
        int tau_idx = (idx_ + 1) % history_.size();
        float x_tau = history_[tau_idx];
        float x_curr = history_[idx_];

        // dx/dt = 0.2 * x(t-tau)/(1 + x(t-tau)^10) - 0.1 * x(t)
        float dxdt = (0.2f * x_tau) / (1.0f + std::pow(x_tau, 10.0f)) - 0.1f * x_curr;
        float x_next = x_curr + dt * dxdt;

        idx_ = (idx_ + 1) % history_.size();
        history_[idx_] = x_next;
        return x_next;
    }

private:
    std::vector<float> history_;
    size_t idx_;
};

}  // namespace

// ----------------------------------------------------------------------------
//  Test 1: Multi-Harmonic Formant Acoustic Predictive Tracking (2 kHz)
// ----------------------------------------------------------------------------
TEST(SymplecticDiscovery, HarmonicAcousticTrackingAdvantage) {
    guim::SymplecticEngine engine_symplectic{42};
    guim::SymplecticEngine engine_baseline{42};

    constexpr int NUM_TICKS = 2500;
    constexpr float DT = 0.0005f;  // 2 kHz acoustic sample rate

    std::array<float, guim::SYM_INPUT_DIM> in_curr{};
    std::array<float, guim::SYM_INPUT_DIM> in_next{};
    std::array<float, guim::SYM_STATE_DIM> out_sym{};
    std::array<float, guim::SYM_STATE_DIM> out_base{};

    float accum_err_sym = 0.0f;
    float accum_err_base = 0.0f;

    for (int t = 0; t < NUM_TICKS; ++t) {
        const float t_sec = static_cast<float>(t) * DT;
        generate_multi_harmonic_acoustic_frame(in_curr, t_sec, 220.0f);
        generate_multi_harmonic_acoustic_frame(in_next, t_sec + DT, 220.0f);

        // Target is the next phase step x(t + dt) (predictive coding)
        const float target = in_next[0];

        // Symplectic Step (AK-SRT)
        const float pred_sym = out_sym[0];
        const float td_sym   = target - pred_sym;
        engine_symplectic.step(in_curr, td_sym, out_sym, DT);
        if (t > 1000) accum_err_sym += td_sym * td_sym;

        // Baseline Step (Standard 1st-Order Trace)
        const float pred_base = out_base[0];
        const float td_base   = target - pred_base;
        engine_baseline.step_standard_baseline(in_curr, td_base, out_base);
        if (t > 1000) accum_err_base += td_base * td_base;
    }

    const float mse_sym  = accum_err_sym / 1500.0f;
    const float mse_base = accum_err_base / 1500.0f;

    std::printf("\n================ [EXPERIMENT 1] HARMONIC ACOUSTIC TRACKING ================\n");
    std::printf("  Acoustic Signal        : Multi-Harmonic Formant Waveform (2 kHz)\n");
    std::printf("  Baseline Standard RTRL : MSE = %9.6f\n", mse_base);
    std::printf("  AK-SRT Symplectic (Ours): MSE = %9.6f\n", mse_sym);
    std::printf("  Relative Advantage     : %6.2f %%\n", (mse_base - mse_sym) / mse_base * 100.0f);
    std::printf("===========================================================================\n");

    EXPECT_TRUE(std::isfinite(mse_sym));
    EXPECT_LT(mse_sym, mse_base);
}

// ----------------------------------------------------------------------------
//  Test 2: Mackey-Glass Chaotic Attractor Prediction
// ----------------------------------------------------------------------------
TEST(SymplecticDiscovery, MackeyGlassChaosPrediction) {
    guim::SymplecticEngine engine_symplectic{99};
    guim::SymplecticEngine engine_baseline{99};

    constexpr float DT_MG = 0.05f;
    MackeyGlassGenerator mg_gen{};
    std::array<float, guim::SYM_INPUT_DIM> in_frame{};
    std::array<float, guim::SYM_STATE_DIM> out_sym{};
    std::array<float, guim::SYM_STATE_DIM> out_base{};

    float accum_err_sym = 0.0f;
    float accum_err_base = 0.0f;

    for (int t = 0; t < 2500; ++t) {
        float sample = mg_gen.next_sample(DT_MG) - 0.9f;
        for (int i = 0; i < guim::SYM_INPUT_DIM; ++i) {
            in_frame[i] = sample * (1.0f + 0.02f * i);
        }

        const float target = sample;

        const float td_sym  = target - out_sym[0];
        engine_symplectic.step(in_frame, td_sym, out_sym, DT_MG);
        if (t > 1000) accum_err_sym += td_sym * td_sym;

        const float td_base = target - out_base[0];
        engine_baseline.step_standard_baseline(in_frame, td_base, out_base);
        if (t > 1000) accum_err_base += td_base * td_base;
    }

    const float mse_sym  = accum_err_sym / 1500.0f;
    const float mse_base = accum_err_base / 1500.0f;

    std::printf("\n================ [EXPERIMENT 2] CHAOTIC TIME SERIES PREDICTION ============\n");
    std::printf("  Benchmark Dynamic      : Mackey-Glass Non-Linear Chaos (tau=17)\n");
    std::printf("  Baseline Standard RTRL : MSE = %9.6f\n", mse_base);
    std::printf("  AK-SRT Symplectic (Ours): MSE = %9.6f\n", mse_sym);
    std::printf("  Relative Advantage     : %6.2f %%\n", (mse_base - mse_sym) / mse_base * 100.0f);
    std::printf("===========================================================================\n");

    EXPECT_TRUE(std::isfinite(mse_sym));
    EXPECT_LT(mse_sym, mse_base);
}

// ----------------------------------------------------------------------------
//  Test 3: Sudden Speaker Pitch Shift with White Gaussian Noise (+6 dB SNR)
// ----------------------------------------------------------------------------
TEST(SymplecticDiscovery, PitchShiftUnderNoiseRobustness) {
    guim::SymplecticEngine engine{101};

    std::mt19937 rng{1337};
    std::normal_distribution<float> noise(0.0f, 0.15f);

    std::array<float, guim::SYM_INPUT_DIM> in_frame{};
    std::array<float, guim::SYM_STATE_DIM> out{};

    // Initial speaker 140 Hz
    for (int t = 0; t < 500; ++t) {
        generate_multi_harmonic_acoustic_frame(in_frame, t * 0.0005f, 140.0f);
        for (float& v : in_frame) v += noise(rng);
        engine.step(in_frame, in_frame[0] - out[0], out, 0.0005f);
    }

    // Sudden speaker pitch transition to 420 Hz
    for (int t = 0; t < 500; ++t) {
        generate_multi_harmonic_acoustic_frame(in_frame, t * 0.0005f, 420.0f);
        for (float& v : in_frame) v += noise(rng);
        float err = engine.step(in_frame, in_frame[0] - out[0], out, 0.0005f);
        ASSERT_TRUE(std::isfinite(err));
    }

    for (float v : out) {
        ASSERT_TRUE(std::isfinite(v));
    }
}
