// ============================================================================
//  metagradients_inspector.cpp — TMD-ET / IDBD Learning Rates (ImPlot)
//  ============================================================================
//  Author  : GuimLab Core Team
//  License : AGPL-3.0 / MIT
// ============================================================================

#include "guim_studio.h"
#include "studio_theme.hpp"
#include "imgui.h"
#include "implot.h"

#include <vector>
#include <cmath>
#include <cstdio>

namespace guim::studio {

void render_metagradients_inspector(StudioTelemetry& telemetry) {
    ImGui::Begin("TMD-ET Meta-Gradients & Attention Barometer", nullptr, ImGuiWindowFlags_NoCollapse);

    ImGui::TextColored(colors::AMBER, "Per-Synapse Learning Rate Adaptation alpha_ij = exp(beta_ij)");
    ImGui::SameLine();
    ImGui::TextDisabled("| IDBD Sutton 1992 / Javed 2024");

    ImGui::Separator();

    // Neuromodulator Chemical State (Dopamine & Serotonin)
    ImGui::Columns(3, "NeuromodCols", false);
    ImGui::TextDisabled("Dopamine delta(t) (TD Error)");
    ImGui::TextColored(telemetry.dopamine >= 0.0f ? colors::EMERALD : colors::CRIMSON, 
                       "%+.4f (Reward Signal)", telemetry.dopamine);
    ImGui::NextColumn();

    ImGui::TextDisabled("Serotonin (Plasticity Gate)");
    ImGui::TextColored(colors::PURPLE, "%.4f (Modulation Gate)", telemetry.serotonin);
    ImGui::NextColumn();

    ImGui::TextDisabled("Effective Meta-Gradient");
    ImGui::TextColored(colors::CYBER_GOLD, "%.5f", telemetry.dopamine * telemetry.serotonin * 0.015f);
    ImGui::NextColumn();
    ImGui::Columns(1);

    ImGui::Separator();

    // Reflex Motor Outputs (16 channels)
    ImGui::Text("Reflex L0 Spinal Motor Outputs (< 100 ns SRAM Execution):");
    static float motor_indices[GUIM_REFLEX_MOTORS];
    static float motor_values[GUIM_REFLEX_MOTORS];
    for (int i = 0; i < GUIM_REFLEX_MOTORS; ++i) {
        motor_indices[i] = static_cast<float>(i);
        motor_values[i]  = telemetry.reflex_motors[i];
    }

    if (ImPlot::BeginPlot("##ReflexMotors", ImVec2(-1, 160))) {
        ImPlot::SetupAxes("Motor Channel (0..15: Formant Dynamics / Turn-Taking Latents)", "Normalized Torque (-1.0 to +1.0)", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
        ImPlot::SetupAxesLimits(-0.5, GUIM_REFLEX_MOTORS - 0.5, -1.1, 1.1, ImPlotCond_Always);

        // Zero Baseline
        static float zero_x[2] = {-0.5f, GUIM_REFLEX_MOTORS - 0.5f};
        static float zero_y[2] = {0.0f, 0.0f};
        ImPlot::PlotLine("Zero Baseline", zero_x, zero_y, 2,
                        {ImPlotProp_LineColor, ImVec4(0.3f, 0.4f, 0.5f, 0.4f), ImPlotProp_LineWeight, 1.0f});

        ImPlot::PlotBars("Motor Activations", motor_indices, motor_values, GUIM_REFLEX_MOTORS, 0.55,
                        {ImPlotProp_FillColor, colors::NEON_CYAN});
        ImPlot::EndPlot();
    }

    ImGui::End();
}

} // namespace guim::studio
