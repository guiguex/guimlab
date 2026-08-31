// ============================================================================
//  metagradients_inspector.cpp — TMD-ET / IDBD Learning Rates (ANSI TUI)
//  ============================================================================
//  Author  : GuimLab Core Team
//  License : AGPL-3.0 / MIT
// ============================================================================

#include "guim_studio.h"
#include "studio_theme.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>

namespace guim::studio {

void render_metagradients_inspector(StudioTelemetry& telemetry) {
    std::cout << ansi::AMBER << ansi::BOLD << "╔═ [4] TMD-ET META-GRADIENTS & CHEMICAL ATTENTION DIAL ═════════════════════════╗\n" << ansi::RESET;

    std::cout << "  Dopamine delta(t) (TD-Error): ";
    if (telemetry.dopamine >= 0.0f) {
        std::cout << ansi::EMERALD << std::fixed << std::setprecision(3) << "+" << telemetry.dopamine << ansi::RESET;
    } else {
        std::cout << ansi::CRIMSON << std::fixed << std::setprecision(3) << telemetry.dopamine << ansi::RESET;
    }
    std::cout << " " << render_bar(telemetry.dopamine, 16, true);

    std::cout << "  |  Serotonin: " << ansi::PURPLE << std::setprecision(3) << telemetry.serotonin << ansi::RESET
              << " " << render_bar(telemetry.serotonin, 16, false) << "\n";

    std::cout << "  Effective Meta-Gradient     : " << ansi::GOLD << std::setprecision(5) 
              << (telemetry.dopamine * telemetry.serotonin * 0.015f) << ansi::RESET << "\n\n";

    // 16 Spinal Motor Channels
    std::cout << "  Reflex L0 Spinal Motors (16 Channels, < 100 ns):\n";
    for (int i = 0; i < GUIM_REFLEX_MOTORS; i += 2) {
        std::cout << "    M" << std::setw(2) << i << ": " 
                  << render_bar(telemetry.reflex_motors[i], 16, true) << " "
                  << std::setw(5) << std::setprecision(2) << telemetry.reflex_motors[i];

        if (i + 1 < GUIM_REFLEX_MOTORS) {
            std::cout << "  │  M" << std::setw(2) << (i + 1) << ": " 
                      << render_bar(telemetry.reflex_motors[i + 1], 16, true) << " "
                      << std::setw(5) << std::setprecision(2) << telemetry.reflex_motors[i + 1];
        }
        std::cout << "\n";
    }

    std::cout << ansi::AMBER << "╚" << std::string(78, '=') << "╝\n\n" << ansi::RESET;
}

} // namespace guim::studio
