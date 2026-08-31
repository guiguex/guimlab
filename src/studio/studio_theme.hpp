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

#include <cmath>
#include <algorithm>
#include <cstdio>

namespace guim::studio {

namespace colors {
    constexpr ImVec4 BG_OBSIDIAN     = ImVec4(0.035f, 0.043f, 0.063f, 1.000f); // #090B10
    constexpr ImVec4 BG_CARD         = ImVec4(0.063f, 0.078f, 0.118f, 1.000f); // #10141E
    constexpr ImVec4 BG_CARD_HOVER   = ImVec4(0.086f, 0.106f, 0.157f, 1.000f); // #161B28
    constexpr ImVec4 BG_HEADER       = ImVec4(0.082f, 0.102f, 0.149f, 1.000f); // #151A26
    constexpr ImVec4 BORDER_SUBTLE   = ImVec4(0.125f, 0.161f, 0.235f, 0.850f); // #20293C
    constexpr ImVec4 BORDER_GLOW     = ImVec4(0.000f, 0.898f, 1.000f, 0.350f); // #00E5FF59

    constexpr ImVec4 NEON_CYAN       = ImVec4(0.000f, 0.898f, 1.000f, 1.000f); // #00E5FF
    constexpr ImVec4 EMERALD         = ImVec4(0.000f, 0.902f, 0.463f, 1.000f); // #00E676
    constexpr ImVec4 CYBER_GOLD      = ImVec4(1.000f, 0.843f, 0.000f, 1.000f); // #FFD700
    constexpr ImVec4 AMBER           = ImVec4(1.000f, 0.569f, 0.000f, 1.000f); // #FF9100
    constexpr ImVec4 CRIMSON         = ImVec4(1.000f, 0.090f, 0.267f, 1.000f); // #FF1744
    constexpr ImVec4 PURPLE          = ImVec4(0.702f, 0.533f, 1.000f, 1.000f); // #B388FF

    constexpr ImVec4 TEXT_PRIMARY    = ImVec4(0.941f, 0.961f, 0.980f, 1.000f); // #F0F5FA
    constexpr ImVec4 TEXT_SECONDARY  = ImVec4(0.600f, 0.650f, 0.720f, 1.000f); // #99A6B8
    constexpr ImVec4 TEXT_MUTED      = ImVec4(0.380f, 0.430f, 0.500f, 1.000f); // #616E80
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
    c[ImGuiCol_TitleBgActive]        = ImVec4(0.10f, 0.14f, 0.22f, 1.00f);
    c[ImGuiCol_TitleBgCollapsed]     = colors::BG_OBSIDIAN;
    c[ImGuiCol_MenuBarBg]            = colors::BG_HEADER;

    // Tabs & Docking
    c[ImGuiCol_Tab]                  = colors::BG_CARD;
    c[ImGuiCol_TabHovered]           = ImVec4(0.14f, 0.20f, 0.30f, 1.00f);
    c[ImGuiCol_TabActive]            = ImVec4(0.10f, 0.16f, 0.26f, 1.00f);
    c[ImGuiCol_TabUnfocused]         = colors::BG_OBSIDIAN;
    c[ImGuiCol_TabUnfocusedActive]   = colors::BG_CARD;
    c[ImGuiCol_DockingPreview]       = ImVec4(0.00f, 0.90f, 1.00f, 0.35f);
    c[ImGuiCol_DockingEmptyBg]       = colors::BG_OBSIDIAN;

    // Controls & Interactions
    c[ImGuiCol_FrameBg]              = ImVec4(0.07f, 0.09f, 0.14f, 1.00f);
    c[ImGuiCol_FrameBgHovered]       = ImVec4(0.11f, 0.15f, 0.22f, 1.00f);
    c[ImGuiCol_FrameBgActive]        = ImVec4(0.14f, 0.20f, 0.30f, 1.00f);
    c[ImGuiCol_Button]               = ImVec4(0.09f, 0.13f, 0.20f, 1.00f);
    c[ImGuiCol_ButtonHovered]        = ImVec4(0.00f, 0.70f, 0.82f, 0.85f);
    c[ImGuiCol_ButtonActive]         = ImVec4(0.00f, 0.85f, 1.00f, 1.00f);
    c[ImGuiCol_Header]               = ImVec4(0.09f, 0.13f, 0.20f, 0.80f);
    c[ImGuiCol_HeaderHovered]        = ImVec4(0.14f, 0.20f, 0.30f, 0.90f);
    c[ImGuiCol_HeaderActive]         = ImVec4(0.00f, 0.85f, 1.00f, 0.80f);

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

    // Geometrical Ergonomics (Modern Clean Rounded Shapes)
    style.WindowPadding     = ImVec2(8.0f, 8.0f);
    style.FramePadding      = ImVec2(7.0f, 4.0f);
    style.ItemSpacing       = ImVec2(6.0f, 5.0f);
    style.ItemInnerSpacing  = ImVec2(5.0f, 4.0f);
    style.WindowRounding    = 8.0f;
    style.ChildRounding     = 7.0f;
    style.FrameRounding     = 5.0f;
    style.PopupRounding     = 6.0f;
    style.ScrollbarRounding = 9.0f;
    style.GrabRounding      = 5.0f;
    style.TabRounding       = 5.0f;
    style.WindowBorderSize  = 1.0f;
    style.ChildBorderSize   = 1.0f;
    style.FrameBorderSize   = 1.0f;
}

// ----------------------------------------------------------------------------
//  Next-Gen UI Widgets & Micro-Renderers
// ----------------------------------------------------------------------------

inline void draw_pulsing_led(ImDrawList* draw, ImVec2 pos, ImVec4 color, float time_sec, float radius = 4.0f) {
    float pulse = 0.5f + 0.5f * std::sin(time_sec * 4.0f);
    ImU32 col_core = ImGui::ColorConvertFloat4ToU32(color);
    ImVec4 glow_col = color;
    glow_col.w = 0.25f * pulse;
    ImU32 col_glow = ImGui::ColorConvertFloat4ToU32(glow_col);

    draw->AddCircleFilled(pos, radius + 3.0f * pulse, col_glow);
    draw->AddCircleFilled(pos, radius, col_core);
}

inline void draw_metric_pill(const char* label, const char* value, ImVec4 val_color, const char* unit = nullptr, float width = 0.0f) {
    ImDrawList* draw = ImGui::GetWindowDrawList();
    ImVec2 p0 = ImGui::GetCursorScreenPos();
    float h = 26.0f;
    float w = (width > 0.0f) ? width : 120.0f;
    ImVec2 p1 = ImVec2(p0.x + w, p0.y + h);

    // Pill Background & Border
    draw->AddRectFilled(p0, p1, ImGui::ColorConvertFloat4ToU32(colors::BG_HEADER), 5.0f);
    draw->AddRect(p0, p1, ImGui::ColorConvertFloat4ToU32(colors::BORDER_SUBTLE), 5.0f);

    // Label & Value
    char buf[64];
    if (unit) {
        std::snprintf(buf, sizeof(buf), "%s %s", value, unit);
    } else {
        std::snprintf(buf, sizeof(buf), "%s", value);
    }

    ImVec2 text_pos_lbl = ImVec2(p0.x + 6.0f, p0.y + 2.0f);
    draw->AddText(text_pos_lbl, ImGui::ColorConvertFloat4ToU32(colors::TEXT_MUTED), label);

    ImVec2 text_pos_val = ImVec2(p0.x + 6.0f, p0.y + 12.0f);
    draw->AddText(text_pos_val, ImGui::ColorConvertFloat4ToU32(val_color), buf);

    ImGui::Dummy(ImVec2(w, h));
}

inline void begin_card(const char* str_id, const char* title, const char* subtitle, 
                       const char* badge_text = nullptr, ImVec4 badge_color = colors::NEON_CYAN, 
                       ImVec2 size = ImVec2(0, 0)) {
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 7.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, colors::BG_CARD);
    ImGui::PushStyleColor(ImGuiCol_Border, colors::BORDER_SUBTLE);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 8.0f));

    ImGui::BeginChild(str_id, size, true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor(2);

    // Card Header Bar with Subtle Gradient
    ImDrawList* draw = ImGui::GetWindowDrawList();
    ImVec2 cp = ImGui::GetCursorScreenPos();
    ImVec2 header_end = ImVec2(cp.x + ImGui::GetWindowWidth() - 16.0f, cp.y + 18.0f);
    
    // Top indicator pip
    draw->AddRectFilled(ImVec2(cp.x, cp.y + 3.0f), ImVec2(cp.x + 3.0f, cp.y + 15.0f), 
                        ImGui::ColorConvertFloat4ToU32(badge_color), 1.5f);

    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 10.0f);
    ImGui::TextColored(colors::TEXT_PRIMARY, "%s", title);
    
    if (subtitle && subtitle[0] != '\0') {
        ImGui::SameLine();
        ImGui::TextDisabled("| %s", subtitle);
    }

    if (badge_text && badge_text[0] != '\0') {
        ImVec2 badge_sz = ImGui::CalcTextSize(badge_text);
        float badge_x = ImGui::GetWindowWidth() - badge_sz.x - 22.0f;
        if (badge_x > ImGui::GetCursorPosX() + 8.0f) {
            ImGui::SameLine(badge_x);
            // Badge background pill
            ImVec2 bp0 = ImVec2(cp.x + badge_x - 6.0f, cp.y);
            ImVec2 bp1 = ImVec2(bp0.x + badge_sz.x + 12.0f, bp0.y + 16.0f);
            ImVec4 bg_badge = badge_color;
            bg_badge.w = 0.15f;
            draw->AddRectFilled(bp0, bp1, ImGui::ColorConvertFloat4ToU32(bg_badge), 3.0f);
            draw->AddRect(bp0, bp1, ImGui::ColorConvertFloat4ToU32(badge_color), 3.0f);
            ImGui::TextColored(badge_color, "%s", badge_text);
        }
    }

    ImGui::PushStyleColor(ImGuiCol_Separator, colors::BORDER_SUBTLE);
    ImGui::Separator();
    ImGui::PopStyleColor();
}

inline void end_card() {
    ImGui::EndChild();
}

} // namespace guim::studio

#endif // GUIM_STUDIO_THEME_HPP
