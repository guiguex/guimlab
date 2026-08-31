// ============================================================================
//  studio_theme.hpp — GuimLab Studio: ANSI TrueColor & Box Drawing Primitives
//  ============================================================================
//  Author  : GuimLab Core Team
//  License : AGPL-3.0 / MIT
// ============================================================================

#ifndef GUIM_STUDIO_THEME_HPP
#define GUIM_STUDIO_THEME_HPP

#include <string>
#include <sstream>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace guim::studio {

namespace ansi {
    // Reset & Style
    constexpr const char* RESET       = "\033[0m";
    constexpr const char* BOLD        = "\033[1m";
    constexpr const char* DIM         = "\033[2m";
    constexpr const char* ITALIC      = "\033[3m";
    constexpr const char* UNDERLINE   = "\033[4m";

    // Cursor / Screen Control
    constexpr const char* CLEAR_SCREEN = "\033[2J";
    constexpr const char* CURSOR_HOME  = "\033[H";
    constexpr const char* HIDE_CURSOR  = "\033[?25l";
    constexpr const char* SHOW_CURSOR  = "\033[?25h";

    // Deep-Tech Obsidian TrueColor Palette (FG)
    constexpr const char* CYAN_GLOW   = "\033[38;2;0;220;240m";
    constexpr const char* EMERALD     = "\033[38;2;40;225;140m";
    constexpr const char* GOLD        = "\033[38;2;255;215;0m";
    constexpr const char* AMBER       = "\033[38;2;255;165;0m";
    constexpr const char* CRIMSON     = "\033[38;2;255;75;75m";
    constexpr const char* PURPLE      = "\033[38;2;180;120;255m";
    constexpr const char* MUTED_GRAY  = "\033[38;2;120;130;145m";
    constexpr const char* DARK_GRAY   = "\033[38;2;60;70;80m";
    constexpr const char* WHITE       = "\033[38;2;240;245;250m";

    // Background TrueColors
    constexpr const char* BG_OBSIDIAN = "\033[48;2;12;15;20m";
    constexpr const char* BG_CARD     = "\033[48;2;20;25;32m";
    constexpr const char* BG_HEADER   = "\033[48;2;28;35;45m";

    // Helpers
    inline std::string fg_rgb(uint8_t r, uint8_t g, uint8_t b) {
        return "\033[38;2;" + std::to_string(r) + ";" + std::to_string(g) + ";" + std::to_string(b) + "m";
    }

    inline std::string bg_rgb(uint8_t r, uint8_t g, uint8_t b) {
        return "\033[48;2;" + std::to_string(r) + ";" + std::to_string(g) + ";" + std::to_string(b) + "m";
    }

    // Diverging Heatmap Color ([-1.0, +1.0] -> RGB Cyan/Blue <-> Obsidian <-> Amber/Gold)
    inline std::string heatmap_color(float val) {
        val = std::clamp(val, -1.0f, 1.0f);
        uint8_t r = 0, g = 0, b = 0;
        if (val < 0.0f) {
            // Negative: Obsidian (0, 15, 25) -> Deep Cyan (0, 200, 240)
            float t = -val;
            r = static_cast<uint8_t>(20.0f * (1.0f - t));
            g = static_cast<uint8_t>(20.0f + 190.0f * t);
            b = static_cast<uint8_t>(35.0f + 215.0f * t);
        } else {
            // Positive: Obsidian (0, 15, 25) -> Vibrant Gold/Amber (255, 190, 20)
            float t = val;
            r = static_cast<uint8_t>(20.0f + 235.0f * t);
            g = static_cast<uint8_t>(25.0f + 165.0f * t);
            b = static_cast<uint8_t>(35.0f * (1.0f - t));
        }
        return bg_rgb(r, g, b) + fg_rgb(r > 150 ? 10 : 240, g > 150 ? 10 : 240, b > 150 ? 10 : 240);
    }
} // namespace ansi

// Box Drawing Characters
namespace box {
    constexpr const char* H_LINE  = "─";
    constexpr const char* V_LINE  = "│";
    constexpr const char* TL_CORNER = "┌";
    constexpr const char* TR_CORNER = "┐";
    constexpr const char* BL_CORNER = "└";
    constexpr const char* BR_CORNER = "┘";
    constexpr const char* T_DOWN  = "┬";
    constexpr const char* T_UP    = "┴";
    constexpr const char* T_RIGHT = "├";
    constexpr const char* T_LEFT  = "┤";
    constexpr const char* CROSS   = "┼";

    // Double Borders
    constexpr const char* D_HLINE = "═";
    constexpr const char* D_VLINE = "║";
    constexpr const char* D_TLCORNER = "╔";
    constexpr const char* D_TRCORNER = "╗";
    constexpr const char* D_BLCORNER = "╚";
    constexpr const char* D_BRCORNER = "╝";
} // namespace box

// Horizontal Progress / Level Bar Generator
inline std::string render_bar(float normalized_val, int width = 20, bool bidirectional = false) {
    if (!bidirectional) {
        float val = std::clamp(normalized_val, 0.0f, 1.0f);
        int filled = static_cast<int>(val * static_cast<float>(width));
        std::string out = "[";
        out += ansi::EMERALD;
        for (int i = 0; i < filled; ++i) out += "█";
        out += ansi::DARK_GRAY;
        for (int i = filled; i < width; ++i) out += "░";
        out += ansi::RESET;
        out += "]";
        return out;
    } else {
        float val = std::clamp(normalized_val, -1.0f, 1.0f);
        int half = width / 2;
        std::string out = "[";
        int pos = static_cast<int>(val * static_cast<float>(half));
        for (int i = -half; i < half; ++i) {
            if (i == 0) {
                out += ansi::WHITE;
                out += "│";
            } else if (pos >= 0 && i > 0 && i <= pos) {
                out += ansi::CYAN_GLOW;
                out += "█";
            } else if (pos < 0 && i < 0 && i >= pos) {
                out += ansi::AMBER;
                out += "█";
            } else {
                out += ansi::DARK_GRAY;
                out += "░";
            }
        }
        out += ansi::RESET;
        out += "]";
        return out;
    }
}

} // namespace guim::studio

#endif // GUIM_STUDIO_THEME_HPP
