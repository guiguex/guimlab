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

void render_cortex_thermal_map(StudioTelemetry& telemetry) {
    ImGui::Begin("Lovelace Cortex 256-Neuron Matrix", nullptr, ImGuiWindowFlags_NoCollapse);

    ImGui::TextColored(colors::NEON_CYAN, "Tensor Core Cortical Activations h_i & CBP Metabolic Plasticity");
    ImGui::SameLine();
    ImGui::TextDisabled("| Dohare & Sutton 2024");

    ImGui::Separator();

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
        }
    }

    // Status Summary Banner
    ImGui::BeginGroup();
    ImGui::Text("Active Units: %d / %d", GUIM_STATE_DIM - dead_count, GUIM_STATE_DIM);
    ImGui::SameLine(180);
    if (dead_count > 0) {
        ImGui::TextColored(colors::CRIMSON, "[!] Dead Units: %d (Triggering CBP Neurogenesis)", dead_count);
    } else {
        ImGui::TextColored(colors::EMERALD, "[OK] Plasticity: 100%% Active (Zero Saturated Units)");
    }
    ImGui::SameLine(580);
    ImGui::TextDisabled("CBP Utility Threshold: eps = 1e-4");
    ImGui::EndGroup();

    ImGui::Separator();

    // 16x16 Heatmap via ImPlot with Viridis Colormap
    if (ImPlot::BeginPlot("##CortexHeatmap", ImVec2(-1, 260), ImPlotFlags_Equal)) {
        ImPlot::SetupAxes("Cortical Columns (0..15)", "Cortical Rows (0..15)", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
        ImPlot::SetupAxesLimits(0, 16, 0, 16, ImPlotCond_Always);
        
        ImPlot::PushColormap(ImPlotColormap_Viridis);
        ImPlot::PlotHeatmap("##Heatmap", heatmap_values, 16, 16, -1.0, 1.0, "%.2f", ImPlotPoint(0, 0), ImPlotPoint(16, 16));
        ImPlot::PopColormap();
        
        ImPlot::EndPlot();
    }

    // Metabolic Variance Distribution Bar Chart
    ImGui::TextDisabled("Neuron Metabolic Variance sigma_i^2 (Plasticity Watchdog)");
    if (ImPlot::BeginPlot("##VarianceDistribution", ImVec2(-1, 100))) {
        ImPlot::SetupAxes("Neuron ID", "Variance sigma^2", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
        ImPlot::SetupAxesLimits(-0.5, GUIM_STATE_DIM - 0.5, 0.0, 0.25, ImPlotCond_Always);
        ImPlot::PlotBars("Variance", neuron_indices, variance_values, GUIM_STATE_DIM, 0.6, 
                         {ImPlotProp_FillColor, colors::EMERALD});
        ImPlot::EndPlot();
    }

    ImGui::End();
}

} // namespace guim::studio
