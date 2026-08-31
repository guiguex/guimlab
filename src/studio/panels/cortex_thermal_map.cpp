// ============================================================================
//  cortex_thermal_map.cpp — Lovelace Cortex 256-Neuron Dynamic Thermal Heatmap
//  ============================================================================
//  Author  : GuimLab Core Team
//  License : AGPL-3.0 / MIT
// ============================================================================

#include "guim_studio.h"
#include "studio_theme.hpp"
#include "imgui.h"
#include "implot.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace guim::studio {

static int s_cortex_view_mode = 0; // 0 = Heatmap, 1 = Variance Distribution

void render_cortex_thermal_map_contents(StudioTelemetry& telemetry) {
    static float heatmap_values[16 * 16];
    static float variance_values[GUIM_STATE_DIM];
    static float neuron_indices[GUIM_STATE_DIM];
    int dead_count = 0;

    for (int i = 0; i < 256; ++i) {
        if (i < GUIM_STATE_DIM) {
            heatmap_values[i]   = telemetry.cortex_activations[i];
            variance_values[i]  = telemetry.cortex_variances[i];
            neuron_indices[i]   = static_cast<float>(i);
            if (telemetry.cortex_dead_units[i]) {
                dead_count++;
            }
        } else {
            heatmap_values[i] = 0.0f;
            variance_values[i] = 0.0f;
            neuron_indices[i] = static_cast<float>(i);
        }
    }

    // 1. Status Bar & View Mode Selector
    ImGui::Columns(3, "CortexStatusCols", false);
    ImGui::TextDisabled("Active Units");
    ImGui::TextColored(colors::EMERALD, "%d / 256 (100%%)", 256 - dead_count);
    ImGui::NextColumn();

    ImGui::TextDisabled("CBP Neurogenesis");
    ImGui::TextColored(dead_count == 0 ? colors::EMERALD : colors::AMBER, 
                       "%d Resets", dead_count);
    ImGui::NextColumn();

    // Mode Toggle Tabs
    if (ImGui::SmallButton(s_cortex_view_mode == 0 ? "[ 16x16 Heatmap ]" : "  16x16 Heatmap  ")) {
        s_cortex_view_mode = 0;
    }
    ImGui::SameLine();
    if (ImGui::SmallButton(s_cortex_view_mode == 1 ? "[ Variance σ² ]" : "  Variance σ²  ")) {
        s_cortex_view_mode = 1;
    }
    ImGui::NextColumn();
    ImGui::Columns(1);

    ImGui::Spacing();

    // 2. Main Visualization View
    if (s_cortex_view_mode == 0) {
        // 16x16 Viridis Heatmap
        if (ImPlot::BeginPlot("##CortexHeatmap", ImVec2(-1, -1), ImPlotFlags_Equal | ImPlotFlags_NoMenus)) {
            ImPlot::SetupAxes(nullptr, nullptr, 
                              ImPlotAxisFlags_NoDecorations, 
                              ImPlotAxisFlags_NoDecorations | ImPlotAxisFlags_Invert);
            ImPlot::SetupAxesLimits(0, 16, 0, 16, ImPlotCond_Always);

            ImPlot::PushColormap(ImPlotColormap_Viridis);
            ImPlot::PlotHeatmap("Activations", heatmap_values, 16, 16, -1.0, 1.0, "%.2f", 
                                ImPlotPoint(0, 0), ImPlotPoint(16, 16));
            ImPlot::PopColormap();

            ImPlot::EndPlot();
        }
    } else {
        // Metabolic Variance Distribution (CBP Plasticity)
        if (ImPlot::BeginPlot("##CortexVariances", ImVec2(-1, -1), ImPlotFlags_NoMenus)) {
            ImPlot::SetupAxes("Neuron ID (0..255)", "Metabolic Variance σ²", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
            ImPlot::SetupAxesLimits(0, 256, 0.0, 0.5, ImPlotCond_Always);

            // Plasticity threshold line ε = 0.001
            static float eps_x[2] = {0.0f, 256.0f};
            static float eps_y[2] = {0.001f, 0.001f};
            ImPlot::PlotLine("Plasticity Floor ε", eps_x, eps_y, 2,
                            {ImPlotProp_LineColor, colors::CRIMSON, ImPlotProp_LineWeight, 1.2f});

            ImPlot::PlotBars("Variance σ²", neuron_indices, variance_values, 256, 0.8,
                            {ImPlotProp_FillColor, colors::PURPLE});

            ImPlot::EndPlot();
        }
    }
}

void render_cortex_thermal_map(StudioTelemetry& telemetry) {
    begin_card("CardCortex", "16x16 LOVELACE CORTEX MATRIX", "Tensor Cores / 256 Neurons", "256 UNITS", colors::PURPLE);
    render_cortex_thermal_map_contents(telemetry);
    end_card();
}

} // namespace guim::studio
