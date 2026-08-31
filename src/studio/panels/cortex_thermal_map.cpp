// ============================================================================
//  cortex_thermal_map.cpp — Lovelace Cortex 256-Neuron Matrix (ANSI TUI)
//  ============================================================================
//  Author  : GuimLab Core Team
//  License : AGPL-3.0 / MIT
// ============================================================================

#include "guim_studio.h"
#include "studio_theme.hpp"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>

namespace guim::studio {

void render_cortex_thermal_map(StudioTelemetry& telemetry) {
    std::cout << ansi::CYAN_GLOW << ansi::BOLD << "╔═ [1] LOVELACE CORTEX 256-NEURON MATRIX (TrueColor Heatmap & CBP Plasticity) ═╗\n" << ansi::RESET;

    int dead_count = 0;
    for (int i = 0; i < GUIM_STATE_DIM; ++i) {
        if (telemetry.cortex_dead_units[i]) dead_count++;
    }

    std::cout << "  Active Units : " << ansi::EMERALD << (GUIM_STATE_DIM - dead_count) << "/" << GUIM_STATE_DIM << ansi::RESET;
    std::cout << "  |  Dead/Saturated: ";
    if (dead_count > 0) {
        std::cout << ansi::CRIMSON << ansi::BOLD << dead_count << " (Triggering CBP Neurogenesis)" << ansi::RESET;
    } else {
        std::cout << ansi::EMERALD << "0 (100% Plasticity)" << ansi::RESET;
    }
    std::cout << "  |  Utility Threshold: eps = 1e-4\n\n";

    // 16x16 Heatmap Grid (8x32 display characters)
    for (int row = 0; row < 16; ++row) {
        std::cout << "  ";
        for (int col = 0; col < 16; ++col) {
            int idx = row * 16 + col;
            float val = (idx < GUIM_STATE_DIM) ? telemetry.cortex_activations[idx] : 0.0f;
            bool is_dead = (idx < GUIM_STATE_DIM) && telemetry.cortex_dead_units[idx];

            std::cout << ansi::heatmap_color(val);
            if (is_dead) {
                std::cout << "!!";
            } else if (std::abs(val) < 0.05f) {
                std::cout << "  ";
            } else if (val > 0.0f) {
                std::cout << "++";
            } else {
                std::cout << "--";
            }
            std::cout << ansi::RESET;
        }
        std::cout << "  " << ansi::DARK_GRAY << "Row " << std::setw(2) << row << ansi::RESET << "\n";
    }

    std::cout << "\n  Legend: " 
              << ansi::bg_rgb(0, 180, 220) << ansi::fg_rgb(0, 0, 0) << " -1.0 (Inhibited) " << ansi::RESET << "  "
              << ansi::bg_rgb(20, 25, 32)  << ansi::fg_rgb(200, 200, 200) << "  0.0 (Quiescent) " << ansi::RESET << "  "
              << ansi::bg_rgb(255, 190, 20) << ansi::fg_rgb(0, 0, 0) << " +1.0 (Excited) " << ansi::RESET << "  "
              << ansi::CRIMSON << "[!!] Dead Unit" << ansi::RESET << "\n";
    std::cout << ansi::CYAN_GLOW << "╚" << std::string(78, '=') << "╝\n\n" << ansi::RESET;
}

} // namespace guim::studio
