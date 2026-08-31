// ============================================================================
//  main_studio.cpp — GuimLab Studio: Pure ANSI TUI Neuromorphic Cockpit
//  ============================================================================
//  A 100% C++20 Zero-Dependency ANSI TrueColor Terminal Cockpit:
//    - Lovelace Cortex 256-Neuron Dynamic Thermal Heatmap & CBP Plasticity
//    - Symplectic Riemannian Phase-Space Orbit (E_ij, P_ij)
//    - 16 kHz Real-Time Acoustic & DDWR Delta Scope
//    - Per-Synapse TMD-ET Meta-Gradients & Reflex L0 Motor Channels
//    - Hardware Sub-Microsecond Latency Profiler & VRAM Sentinel
//
//  Author  : GuimLab Core Team
//  License : AGPL-3.0 / MIT
// ============================================================================

#include "guim_studio.h"
#include "studio_theme.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <csignal>
#include <atomic>
#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
#include <conio.h>
#include <windows.h>
#else
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#endif

namespace {

std::atomic<bool> g_running{true};
int g_active_view = 0; // 0 = Master Cockpit, 1 = Cortex, 2 = Symplectic, 3 = Audio, 4 = Meta, 5 = Latency

void handle_signal(int) {
    g_running.store(false);
}

#ifndef _WIN32
struct TermiosGuard {
    struct termios orig_termios{};
    bool active{false};

    void enable_raw() {
        if (tcgetattr(STDIN_FILENO, &orig_termios) == -1) return;
        struct termios raw = orig_termios;
        raw.c_lflag &= ~(ECHO | ICANON);
        raw.c_cc[VMIN]  = 0;
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSANOW, &raw);
        active = true;
    }

    void disable_raw() {
        if (active) {
            tcsetattr(STDIN_FILENO, TCSANOW, &orig_termios);
            active = false;
        }
    }

    ~TermiosGuard() { disable_raw(); }
};

int nonblocking_getch() {
    char ch = 0;
    if (read(STDIN_FILENO, &ch, 1) > 0) {
        return static_cast<int>(ch);
    }
    return -1;
}
#else
int nonblocking_getch() {
    if (_kbhit()) {
        return _getch();
    }
    return -1;
}
#endif

} // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT,  handle_signal);
    std::signal(SIGTERM, handle_signal);

#ifndef _WIN32
    TermiosGuard term_guard;
    term_guard.enable_raw();
#endif

    std::cout << guim::studio::ansi::HIDE_CURSOR;
    std::cout << guim::studio::ansi::CLEAR_SCREEN << guim::studio::ansi::CURSOR_HOME;

    // Connect to Shared Memory
    guim::studio::ShmClient shm_client;
    guim::studio::StudioTelemetry telemetry;

    const char* shm_path = (argc > 1) ? argv[1] : "/guim_ipc_v3_shm";
    shm_client.connect(shm_path);

    float inject_dopamine = 0.0f;
    float inject_serotonin = 0.5f;

    auto last_render = std::chrono::steady_clock::now();

    while (g_running.load()) {
        // 1. High-frequency SHM polling
        if (!shm_client.is_connected()) {
            shm_client.connect(shm_path);
        }
        if (!telemetry.paused) {
            shm_client.poll(telemetry);
        }

        // 2. Non-blocking keyboard event handling
        int key = nonblocking_getch();
        if (key > 0) {
            if (key == 'q' || key == 'Q' || key == 27) { // q / ESC
                g_running.store(false);
                break;
            } else if (key == '0') {
                g_active_view = 0;
            } else if (key >= '1' && key <= '5') {
                g_active_view = key - '0';
            } else if (key == 'p' || key == 'P' || key == ' ') {
                telemetry.paused = !telemetry.paused;
            } else if (key == '+' || key == '=') {
                inject_dopamine = std::clamp(inject_dopamine + 0.1f, -1.0f, 1.0f);
                shm_client.inject_neuromodulators(inject_dopamine, inject_serotonin);
            } else if (key == '-' || key == '_') {
                inject_dopamine = std::clamp(inject_dopamine - 0.1f, -1.0f, 1.0f);
                shm_client.inject_neuromodulators(inject_dopamine, inject_serotonin);
            } else if (key == 'r' || key == 'R') {
                shm_client.connect(shm_path);
            }
        }

        // 3. Render at ~30 FPS
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_render).count() >= 33) {
            last_render = now;

            std::cout << guim::studio::ansi::CURSOR_HOME;

            // Global Master Header
            std::cout << guim::studio::ansi::BG_HEADER << guim::studio::ansi::WHITE << guim::studio::ansi::BOLD 
                      << "  GUIMLAB STUDIO — Continuous Neuromorphic Substrate TUI Cockpit (C++20 Bare-Metal)  " 
                      << guim::studio::ansi::RESET << "\n";

            std::cout << "  Views: "
                      << (g_active_view == 0 ? guim::studio::ansi::GOLD : guim::studio::ansi::MUTED_GRAY) << "[0] Master  "
                      << (g_active_view == 1 ? guim::studio::ansi::GOLD : guim::studio::ansi::MUTED_GRAY) << "[1] Cortex  "
                      << (g_active_view == 2 ? guim::studio::ansi::GOLD : guim::studio::ansi::MUTED_GRAY) << "[2] Symplectic  "
                      << (g_active_view == 3 ? guim::studio::ansi::GOLD : guim::studio::ansi::MUTED_GRAY) << "[3] Audio Scope  "
                      << (g_active_view == 4 ? guim::studio::ansi::GOLD : guim::studio::ansi::MUTED_GRAY) << "[4] Meta-Gradients  "
                      << (g_active_view == 5 ? guim::studio::ansi::GOLD : guim::studio::ansi::MUTED_GRAY) << "[5] Latency Profiler"
                      << guim::studio::ansi::RESET << "\n";

            std::cout << "  Keys : " << guim::studio::ansi::DARK_GRAY 
                      << "[0-5] Switch View | [SPACE] Pause | [+/-] Dopamine | [r] Reconnect | [q] Quit" 
                      << guim::studio::ansi::RESET << "\n\n";

            // Render view according to selection
            if (g_active_view == 0) {
                // Master Cockpit (all panels)
                guim::studio::render_telemetry_cockpit(telemetry, shm_client);
                guim::studio::render_cortex_thermal_map(telemetry);
                guim::studio::render_symplectic_phase_portrait(telemetry);
                guim::studio::render_audio_oscilloscope(telemetry);
                guim::studio::render_metagradients_inspector(telemetry);
            } else if (g_active_view == 1) {
                guim::studio::render_cortex_thermal_map(telemetry);
            } else if (g_active_view == 2) {
                guim::studio::render_symplectic_phase_portrait(telemetry);
            } else if (g_active_view == 3) {
                guim::studio::render_audio_oscilloscope(telemetry);
            } else if (g_active_view == 4) {
                guim::studio::render_metagradients_inspector(telemetry);
            } else if (g_active_view == 5) {
                guim::studio::render_telemetry_cockpit(telemetry, shm_client);
            }

            std::cout << std::flush;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

#ifndef _WIN32
    term_guard.disable_raw();
#endif
    std::cout << guim::studio::ansi::SHOW_CURSOR << "\n[Studio] Graceful shutdown complete.\n";
    return 0;
}
