// ============================================================================
//  audio_oscilloscope.cpp — Continuous Acoustic Waveform & Scope (ANSI TUI)
//  ============================================================================
//  Author  : GuimLab Core Team
//  License : AGPL-3.0 / MIT
// ============================================================================

#include "guim_studio.h"
#include "studio_theme.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>

namespace guim::studio {

void render_audio_oscilloscope(StudioTelemetry& telemetry) {
    std::cout << ansi::CYAN_GLOW << ansi::BOLD << "╔═ [3] CONTINUOUS AUDIO & SENSORY OSCILLOSCOPE (16 kHz / DDWR Sparsity) ═════════╗\n" << ansi::RESET;

    std::size_t sz = telemetry.audio_waveform.size();
    float rms_sum = 0.0f;
    for (std::size_t i = 0; i < sz; ++i) {
        float sample = telemetry.audio_waveform[i];
        rms_sum += sample * sample;
    }
    float rms_val = (sz > 0) ? std::sqrt(rms_sum / static_cast<float>(sz)) : 0.0f;
    float dbfs_val = (rms_val > 1e-5f) ? 20.0f * std::log10(rms_val) : -100.0f;

    std::cout << "  Sampling Rate : " << ansi::EMERALD << "2,000 Hz (dt = 0.50 ms)" << ansi::RESET;
    std::cout << "  |  RMS Level: " << ansi::CYAN_GLOW << std::fixed << std::setprecision(3) << rms_val 
              << " (" << std::setprecision(1) << dbfs_val << " dBFS)" << ansi::RESET << "\n";
    std::cout << "  DDWR Sparsity : " << render_bar(telemetry.ddwr_sparse_ratio > 0.0f ? telemetry.ddwr_sparse_ratio : 0.92f, 25)
              << " " << ansi::EMERALD << std::setprecision(1) << (telemetry.ddwr_sparse_ratio * 100.0f) << "% VRAM reduction" << ansi::RESET << "\n\n";

    // 1D Audio Waveform ASCII Plot (7 rows x 64 cols)
    constexpr int ROWS = 7;
    constexpr int COLS = 64;
    char plot[ROWS][COLS];
    std::string colors[ROWS][COLS];

    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            plot[r][c] = (r == ROWS / 2) ? '-' : ' ';
            colors[r][c] = ansi::DARK_GRAY;
        }
    }

    if (sz > 0) {
        for (int c = 0; c < COLS; ++c) {
            std::size_t sample_idx = (sz > COLS) ? (sz - COLS + c) : c;
            float val = (sample_idx < sz) ? telemetry.audio_waveform[sample_idx] : 0.0f;
            int r = static_cast<int>((1.0f - val) * 0.5f * (ROWS - 1));
            r = std::clamp(r, 0, ROWS - 1);
            plot[r][c] = '#';
            colors[r][c] = ansi::EMERALD;
        }
    }

    for (int r = 0; r < ROWS; ++r) {
        std::cout << "  ";
        if (r == 0) std::cout << "+1.0 ";
        else if (r == ROWS / 2) std::cout << " 0.0 ";
        else if (r == ROWS - 1) std::cout << "-1.0 ";
        else std::cout << "     ";

        for (int c = 0; c < COLS; ++c) {
            std::cout << colors[r][c] << plot[r][c] << ansi::RESET;
        }
        std::cout << "\n";
    }

    std::cout << ansi::CYAN_GLOW << "╚" << std::string(78, '=') << "╝\n\n" << ansi::RESET;
}

} // namespace guim::studio
