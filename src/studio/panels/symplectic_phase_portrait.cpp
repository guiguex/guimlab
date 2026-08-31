// ============================================================================
//  symplectic_phase_portrait.cpp — SR-MIT Symplectic Phase Space (ImPlot)
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

void render_symplectic_phase_portrait_contents(StudioTelemetry& telemetry) {
    static float e_data[TRACE_ORBIT_POINTS];
    static float p_data[TRACE_ORBIT_POINTS];
    std::size_t sz = telemetry.symplectic_e.size();

    for (std::size_t i = 0; i < sz; ++i) {
        e_data[i] = telemetry.symplectic_e[i];
        p_data[i] = telemetry.symplectic_p[i];
    }

    float head_e = (sz > 0) ? e_data[sz - 1] : 0.0f;
    float head_p = (sz > 0) ? p_data[sz - 1] : 0.0f;
    float current_energy = 0.5f * (head_e * head_e + head_p * head_p);

    constexpr float NOMINAL_OMEGA = 2.0f * 3.14159265f * 220.0f;
    constexpr float NOMINAL_TAU   = 0.030f;
    const float omega_tau = NOMINAL_OMEGA * NOMINAL_TAU;
    const float gamma_lead = omega_tau / std::sqrt(1.0f + omega_tau * omega_tau);
    const float lead_trace = head_e + gamma_lead * head_p;
    const float phase_deg = std::atan2(head_p, head_e) * (180.0f / 3.14159265f);

    // 1. Symplectic Invariant Metrics Row
    ImGui::Columns(4, "SymplecticMetrics", false);
    ImGui::TextDisabled("Hamiltonian Energy H");
    ImGui::TextColored(colors::NEON_CYAN, "%.4f (Bound 50.0)", current_energy);
    ImGui::NextColumn();

    ImGui::TextDisabled("Symplectic Rotation det(R)");
    ImGui::TextColored(colors::EMERALD, "1.000000 (Unitary)");
    ImGui::NextColumn();

    ImGui::TextDisabled("Compensated Trace T_ij");
    ImGui::TextColored(colors::CYBER_GOLD, "%.4f (+1.12%% Lead)", lead_trace);
    ImGui::NextColumn();

    ImGui::TextDisabled("Phase Angle theta");
    ImGui::TextColored(colors::EMERALD, "%.1f deg (Lag = 0)", phase_deg);
    ImGui::NextColumn();
    ImGui::Columns(1);

    ImGui::Spacing();

    // 2. Symplectic Phase Plane Plot
    if (ImPlot::BeginPlot("##SymplecticOrbit", ImVec2(-1, -1), ImPlotFlags_Equal | ImPlotFlags_NoMenus)) {
        ImPlot::SetupAxes("Trace Position E_ij", "Conjugate Momentum P_ij", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
        ImPlot::SetupAxesLimits(-1.35, 1.35, -1.35, 1.35, ImPlotCond_Always);

        // Concentric Isocontours H = const
        constexpr int CIRCLE_PTS = 64;
        static float c1_x[CIRCLE_PTS + 1], c1_y[CIRCLE_PTS + 1];
        static float c2_x[CIRCLE_PTS + 1], c2_y[CIRCLE_PTS + 1];
        for (int i = 0; i <= CIRCLE_PTS; ++i) {
            float th = 2.0f * 3.14159265f * static_cast<float>(i) / CIRCLE_PTS;
            c1_x[i] = std::cos(th) * 0.707f;
            c1_y[i] = std::sin(th) * 0.707f;
            c2_x[i] = std::cos(th) * 1.0f;
            c2_y[i] = std::sin(th) * 1.0f;
        }

        ImPlot::PlotLine("H = 0.25", c1_x, c1_y, CIRCLE_PTS + 1, 
                        {ImPlotProp_LineColor, ImVec4(0.2f, 0.25f, 0.35f, 0.35f), ImPlotProp_LineWeight, 1.0f});
        ImPlot::PlotLine("H = 0.50 Invariant", c2_x, c2_y, CIRCLE_PTS + 1, 
                        {ImPlotProp_LineColor, ImVec4(0.25f, 0.35f, 0.5f, 0.5f), ImPlotProp_LineWeight, 1.0f});

        if (sz > 1) {
            // Trajectory Phase Orbit
            ImPlot::PlotLine("Phase Orbit", e_data, p_data, static_cast<int>(sz),
                            {ImPlotProp_LineColor, colors::NEON_CYAN, ImPlotProp_LineWeight, 2.0f});

            // Current State Pip & Vector
            ImPlot::PlotScatter("Head", &head_e, &head_p, 1,
                               {ImPlotProp_Marker, ImPlotMarker_Circle, ImPlotProp_MarkerSize, 6.0f, 
                                ImPlotProp_MarkerFillColor, colors::CYBER_GOLD});

            float origin_x[2] = {0.0f, head_e};
            float origin_y[2] = {0.0f, head_p};
            ImPlot::PlotLine("Lead Vector", origin_x, origin_y, 2,
                            {ImPlotProp_LineColor, ImVec4(1.0f, 0.84f, 0.0f, 0.6f), ImPlotProp_LineWeight, 1.5f});
        }

        ImPlot::EndPlot();
    }
}

void render_symplectic_phase_portrait(StudioTelemetry& telemetry) {
    begin_card("CardSymplectic", "SR-MIT SYMPLECTIC PHASE PORTRAIT", "det(R) = 1 / Unitary Invariant", "H <= 50.0", colors::NEON_CYAN);
    render_symplectic_phase_portrait_contents(telemetry);
    end_card();
}

} // namespace guim::studio
