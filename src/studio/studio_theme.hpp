// ============================================================================
//  studio_theme.hpp — GuimLab Studio: AAA Deep-Tech Obsidian Design System
//  ============================================================================
//  Author  : GuimLab Core Team
//  License : AGPL-3.0 / MIT
// ============================================================================

#ifndef GUIM_STUDIO_THEME_HPP
#define GUIM_STUDIO_THEME_HPP

#include "imgui.h"
#include "implot.h"

namespace guim::studio {

namespace colors {
    constexpr ImVec4 BG_OBSIDIAN     = ImVec4(0.043f, 0.055f, 0.078f, 1.000f); // #0B0E14
    constexpr ImVec4 BG_CARD         = ImVec4(0.071f, 0.090f, 0.133f, 1.000f); // #121722
    constexpr ImVec4 BG_HEADER       = ImVec4(0.102f, 0.125f, 0.180f, 1.000f); // #1A202E
    constexpr ImVec4 BORDER_SUBTLE   = ImVec4(0.137f, 0.169f, 0.243f, 0.850f); // #232B3E

    constexpr ImVec4 NEON_CYAN       = ImVec4(0.000f, 0.898f, 1.000f, 1.000f); // #00E5FF
    constexpr ImVec4 EMERALD         = ImVec4(0.000f, 0.902f, 0.463f, 1.000f); // #00E676
    constexpr ImVec4 CYBER_GOLD      = ImVec4(1.000f, 0.843f, 0.000f, 1.000f); // #FFD700
    constexpr ImVec4 AMBER           = ImVec4(1.000f, 0.569f, 0.000f, 1.000f); // #FF9100
    constexpr ImVec4 CRIMSON         = ImVec4(1.000f, 0.090f, 0.267f, 1.000f); // #FF1744
    constexpr ImVec4 PURPLE          = ImVec4(0.702f, 0.533f, 1.000f, 1.000f); // #B388FF

    constexpr ImVec4 TEXT_PRIMARY    = ImVec4(0.941f, 0.961f, 0.980f, 1.000f); // #F0F5FA
    constexpr ImVec4 TEXT_SECONDARY  = ImVec4(0.600f, 0.650f, 0.720f, 1.000f); // #99A6B8
    constexpr ImVec4 TEXT_MUTED      = ImVec4(0.400f, 0.450f, 0.520f, 1.000f); // #667385
}

inline void apply_deeptech_theme() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* c = style.Colors;

    // Window & Surfaces
    c[ImGuiCol_WindowBg]             = colors::BG_OBSIDIAN;
    c[ImGuiCol_ChildBg]              = colors::BG_CARD;
    c[ImGuiCol_PopupBg]              = colors::BG_CARD;
    c[ImGuiCol_Border]               = colors::BORDER_SUBTLE;
    c[ImGuiCol_BorderShadow]         = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

    // Headers & Titles
    c[ImGuiCol_TitleBg]              = colors::BG_HEADER;
    c[ImGuiCol_TitleBgActive]        = ImVec4(0.12f, 0.16f, 0.24f, 1.00f);
    c[ImGuiCol_TitleBgCollapsed]      = colors::BG_OBSIDIAN;
    c[ImGuiCol_MenuBarBg]            = colors::BG_HEADER;

    // Tabs & Docking
    c[ImGuiCol_Tab]                  = colors::BG_CARD;
    c[ImGuiCol_TabHovered]           = ImVec4(0.15f, 0.22f, 0.32f, 1.00f);
    c[ImGuiCol_TabActive]            = ImVec4(0.10f, 0.18f, 0.28f, 1.00f);
    c[ImGuiCol_TabUnfocused]         = colors::BG_OBSIDIAN;
    c[ImGuiCol_TabUnfocusedActive]   = colors::BG_CARD;
    c[ImGuiCol_DockingPreview]       = ImVec4(0.00f, 0.90f, 1.00f, 0.35f);
    c[ImGuiCol_DockingEmptyBg]       = colors::BG_OBSIDIAN;

    // Controls & Interactions
    c[ImGuiCol_FrameBg]              = ImVec4(0.08f, 0.11f, 0.16f, 1.00f);
    c[ImGuiCol_FrameBgHovered]       = ImVec4(0.12f, 0.17f, 0.25f, 1.00f);
    c[ImGuiCol_FrameBgActive]        = ImVec4(0.15f, 0.22f, 0.32f, 1.00f);
    c[ImGuiCol_Button]               = ImVec4(0.10f, 0.15f, 0.22f, 1.00f);
    c[ImGuiCol_ButtonHovered]        = ImVec4(0.00f, 0.75f, 0.85f, 0.85f);
    c[ImGuiCol_ButtonActive]         = ImVec4(0.00f, 0.90f, 1.00f, 1.00f);
    c[ImGuiCol_Header]               = ImVec4(0.10f, 0.15f, 0.22f, 0.80f);
    c[ImGuiCol_HeaderHovered]        = ImVec4(0.15f, 0.22f, 0.32f, 0.90f);
    c[ImGuiCol_HeaderActive]         = ImVec4(0.00f, 0.90f, 1.00f, 0.80f);

    // Sliders & Checkmarks
    c[ImGuiCol_CheckMark]            = colors::NEON_CYAN;
    c[ImGuiCol_SliderGrab]           = colors::NEON_CYAN;
    c[ImGuiCol_SliderGrabActive]     = colors::EMERALD;

    // Scrollbars
    c[ImGuiCol_ScrollbarBg]          = colors::BG_OBSIDIAN;
    c[ImGuiCol_ScrollbarGrab]        = colors::BORDER_SUBTLE;
    c[ImGuiCol_ScrollbarGrabHovered] = colors::TEXT_MUTED;
    c[ImGuiCol_ScrollbarGrabActive]  = colors::NEON_CYAN;

    // Text & Separators
    c[ImGuiCol_Text]                 = colors::TEXT_PRIMARY;
    c[ImGuiCol_TextDisabled]         = colors::TEXT_MUTED;
    c[ImGuiCol_Separator]            = colors::BORDER_SUBTLE;
    c[ImGuiCol_SeparatorHovered]     = colors::NEON_CYAN;
    c[ImGuiCol_SeparatorActive]      = colors::EMERALD;

    // Geometrical Ergonomics (Modern Clean Bevels)
    style.WindowPadding     = ImVec2(12.0f, 12.0f);
    style.FramePadding      = ImVec2(8.0f, 5.0f);
    style.ItemSpacing       = ImVec2(8.0f, 7.0f);
    style.ItemInnerSpacing  = ImVec2(6.0f, 4.0f);
    style.WindowRounding    = 6.0f;
    style.ChildRounding     = 5.0f;
    style.FrameRounding     = 4.0f;
    style.PopupRounding     = 5.0f;
    style.ScrollbarRounding = 9.0f;
    style.GrabRounding      = 4.0f;
    style.TabRounding       = 4.0f;
    style.WindowBorderSize  = 1.0f;
    style.FrameBorderSize   = 1.0f;

    // ImPlot Theme Setup
    ImPlotStyle& plot_style = ImPlot::GetStyle();
    plot_style.PlotPadding       = ImVec2(10, 10);
    plot_style.LabelPadding      = ImVec2(4, 4);
    plot_style.LegendPadding     = ImVec2(6, 6);
    plot_style.PlotBorderSize    = 1.0f;
    plot_style.MajorGridSize     = ImVec2(1.0f, 1.0f);
    plot_style.MinorGridSize     = ImVec2(0.5f, 0.5f);
    plot_style.MajorTickLen      = ImVec2(5, 5);
    plot_style.MinorTickLen      = ImVec2(2, 2);

    plot_style.Colors[ImPlotCol_PlotBg]     = colors::BG_CARD;
    plot_style.Colors[ImPlotCol_PlotBorder] = colors::BORDER_SUBTLE;
    plot_style.Colors[ImPlotCol_LegendBg]   = ImVec4(0.05f, 0.07f, 0.10f, 0.90f);
    plot_style.Colors[ImPlotCol_LegendBorder]= colors::BORDER_SUBTLE;
    plot_style.Colors[ImPlotCol_LegendText] = colors::TEXT_PRIMARY;
    plot_style.Colors[ImPlotCol_TitleText]  = colors::NEON_CYAN;
    plot_style.Colors[ImPlotCol_InlayText]  = colors::TEXT_SECONDARY;
    plot_style.Colors[ImPlotCol_AxisText]   = colors::TEXT_SECONDARY;
    plot_style.Colors[ImPlotCol_AxisGrid]   = ImVec4(0.12f, 0.15f, 0.22f, 0.70f);
}

} // namespace guim::studio

#endif // GUIM_STUDIO_THEME_HPP
