// ============================================================================
//  metagradients_inspector.cpp — TMD-ET / IDBD Learning Rates & Reflex Motors
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

void render_metagradients_inspector_contents(StudioTelemetry& telemetry) {
    // 1. Neuromodulator Chemical State
    ImGui::Columns(3, "NeuromodCols", false);
    ImGui::TextDisabled("Dopamine delta(t)");
    ImGui::TextColored(telemetry.dopamine >= 0.0f ? colors::EMERALD : colors::CRIMSON, 
                       "%+.3f (Reward)", telemetry.dopamine);
    ImGui::NextColumn();

    ImGui::TextDisabled("Serotonin Gate");
    ImGui::TextColored(colors::PURPLE, "%.3f (Gate)", telemetry.serotonin);
    ImGui::NextColumn();

    ImGui::TextDisabled("TMD-ET Gradient");
    ImGui::TextColored(colors::CYBER_GOLD, "%.4f", telemetry.dopamine * telemetry.serotonin * 0.015f);
    ImGui::NextColumn();
    ImGui::Columns(1);

    ImGui::Spacing();

    // 2. Reflex Motor Outputs (16 channels)
    ImGui::TextDisabled("Reflex L0 Spinal Motors (< 100 ns SRAM / 16 Channels):");
    static float motor_indices[GUIM_REFLEX_MOTORS];
    static float motor_values[GUIM_REFLEX_MOTORS];
    for (int i = 0; i < GUIM_REFLEX_MOTORS; ++i) {
        motor_indices[i] = static_cast<float>(i);
        motor_values[i]  = telemetry.reflex_motors[i];
    }

    if (ImPlot::BeginPlot("##ReflexMotors", ImVec2(-1, -1), ImPlotFlags_NoLegend | ImPlotFlags_NoMenus)) {
        ImPlot::SetupAxes("Channel (0..15)", "Torque (-1..+1)", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
        ImPlot::SetupAxesLimits(-0.5, GUIM_REFLEX_MOTORS - 0.5, -1.05, 1.05, ImPlotCond_Always);

        static float zero_x[2] = {-0.5f, GUIM_REFLEX_MOTORS - 0.5f};
        static float zero_y[2] = {0.0f, 0.0f};
        ImPlot::PlotLine("Zero", zero_x, zero_y, 2,
                        {ImPlotProp_LineColor, ImVec4(0.3f, 0.4f, 0.5f, 0.4f), ImPlotProp_LineWeight, 1.0f});

        ImPlot::PlotBars("Motor Activations", motor_indices, motor_values, GUIM_REFLEX_MOTORS, 0.55,
                        {ImPlotProp_FillColor, colors::NEON_CYAN});
        ImPlot::EndPlot();
    }
}

void render_metagradients_inspector(StudioTelemetry& telemetry) {
    begin_card("CardMetagradients", "TMD-ET META-GRADIENTS & REFLEX MOTORS", "IDBD / Javed 2024", "16 MOTORS", colors::AMBER);
    render_metagradients_inspector_contents(telemetry);
    end_card();
}

} // namespace guim::studio
