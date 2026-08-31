// ============================================================================
//  main_studio.cpp — GuimLab Studio: AAA Single-Header Sokol / ImGui / ImPlot
//  ============================================================================
//  Author  : GuimLab Core Team
//  License : AGPL-3.0 / MIT
// ============================================================================

#define SOKOL_IMPL
#if defined(_WIN32)
#define SOKOL_D3D11
#elif defined(__APPLE__)
#define SOKOL_METAL
#else
#define SOKOL_GLCORE
#endif

#include "sokol/sokol_app.h"
#include "sokol/sokol_gfx.h"
#include "sokol/sokol_glue.h"
#include "sokol/sokol_time.h"
#include "sokol/sokol_log.h"

#include "imgui.h"
#include "implot.h"

#define SOKOL_IMGUI_IMPL
#include "sokol/sokol_imgui.h"

#include "guim_studio.h"
#include "studio_theme.hpp"

#include <cstdio>
#include <vector>

static guim::studio::ShmClient g_shm_client;
static guim::studio::StudioTelemetry g_telemetry;
static sg_pass_action g_pass_action;

static void init_cb() {
    // 1. Initialize High-Resolution Microsecond Timer
    stm_setup();

    // 2. Initialize Sokol Gfx Pipeline
    sg_desc desc = {};
    desc.environment = sglue_environment();
    desc.logger.func = slog_func;
    sg_setup(&desc);

    // 3. Initialize Sokol ImGui Bridge
    simgui_desc_t simgui_desc = {};
    simgui_desc.logger.func = slog_func;
    simgui_setup(&simgui_desc);

    // 4. Initialize ImPlot Context & Apply AAA Deep-Tech Theme
    ImPlot::CreateContext();
    guim::studio::apply_deeptech_theme();

    // 5. Connect to Zero-Copy Lock-Free Shared Memory Segment
    g_shm_client.connect("/guim_ipc_v3_shm");

    // 6. Setup Frame Clear Action (Deep Obsidian Slate)
    g_pass_action.colors[0].load_action = SG_LOADACTION_CLEAR;
    g_pass_action.colors[0].clear_value = { 0.043f, 0.055f, 0.078f, 1.0f };
}

static void frame_cb() {
    // 1. Ingest IPC Telemetry from GPU SHM (Zero-Copy Ring-Buffer)
    g_shm_client.poll(g_telemetry);

    // 2. Begin ImGui Frame via Sokol
    const int width = sapp_width();
    const int height = sapp_height();
    simgui_frame_desc_t frame_desc = {};
    frame_desc.width = width;
    frame_desc.height = height;
    frame_desc.delta_time = sapp_frame_duration();
    frame_desc.dpi_scale = sapp_dpi_scale();
    simgui_new_frame(&frame_desc);

    // 3. Setup DockSpace for AAA Multi-Window Drag-and-Drop
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);
    
    ImGuiWindowFlags host_window_flags = 0;
    host_window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDocking;
    host_window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_MenuBar;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("GuimLabStudioDockSpaceHost", nullptr, host_window_flags);
    ImGui::PopStyleVar(3);

    // Top Main Menu Bar
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("Substrate")) {
            if (ImGui::MenuItem("Reconnect SHM (/guim_ipc_v3_shm)")) {
                g_shm_client.connect("/guim_ipc_v3_shm");
            }
            if (ImGui::MenuItem("Toggle Telemetry Stream Pause", "SPACE")) {
                g_telemetry.paused = !g_telemetry.paused;
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Quit Studio", "Ctrl+Q")) {
                sapp_request_quit();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Neuromodulation")) {
            if (ImGui::MenuItem("Inject Dopamine Spike (+1.0)")) {
                g_shm_client.inject_neuromodulators(1.0f, 0.5f);
            }
            if (ImGui::MenuItem("Inhibit Dopamine (-1.0)")) {
                g_shm_client.inject_neuromodulators(-1.0f, 0.5f);
            }
            if (ImGui::MenuItem("Reset Chemical Modulators")) {
                g_shm_client.inject_neuromodulators(0.0f, 0.0f);
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Viewports")) {
            ImGui::MenuItem("Lovelace Cortex 256-Neuron Matrix", nullptr, true);
            ImGui::MenuItem("SR-MIT Symplectic Phase Portrait", nullptr, true);
            ImGui::MenuItem("Continuous Audio Oscilloscope", nullptr, true);
            ImGui::MenuItem("TMD-ET Meta-Gradients Inspector", nullptr, true);
            ImGui::MenuItem("Hardware Telemetry Cockpit", nullptr, true);
            ImGui::EndMenu();
        }
        ImGui::SameLine(ImGui::GetWindowWidth() - 320);
        ImGui::TextColored(guim::studio::colors::NEON_CYAN, "GUIMLAB STUDIO | Sokol Bare-Metal C++20");
        ImGui::EndMenuBar();
    }

    ImGuiID dockspace_id = ImGui::GetID("GuimLabStudioDockSpace");
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::End();

    // 4. Render All 5 Dedicated Studio Panels
    guim::studio::render_telemetry_cockpit(g_telemetry, g_shm_client);
    guim::studio::render_cortex_thermal_map(g_telemetry);
    guim::studio::render_symplectic_phase_portrait(g_telemetry);
    guim::studio::render_audio_oscilloscope(g_telemetry);
    guim::studio::render_metagradients_inspector(g_telemetry);

    // 5. Render Sokol Pass
    sg_pass pass = {};
    pass.action = g_pass_action;
    pass.swapchain = sglue_swapchain();

    sg_begin_pass(&pass);
    simgui_render();
    sg_end_pass();
    sg_commit();
}

static void event_cb(const sapp_event* ev) {
    if (ev->type == SAPP_EVENTTYPE_KEY_DOWN) {
        if (ev->key_code == SAPP_KEYCODE_ESCAPE || (ev->key_code == SAPP_KEYCODE_Q && (ev->modifiers & SAPP_MODIFIER_CTRL))) {
            sapp_request_quit();
        } else if (ev->key_code == SAPP_KEYCODE_SPACE) {
            g_telemetry.paused = !g_telemetry.paused;
        }
    }
    simgui_handle_event(ev);
}

static void cleanup_cb() {
    g_shm_client.disconnect();
    ImPlot::DestroyContext();
    simgui_shutdown();
    sg_shutdown();
}

sapp_desc sokol_main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    sapp_desc desc = {};
    desc.init_cb    = init_cb;
    desc.frame_cb   = frame_cb;
    desc.cleanup_cb = cleanup_cb;
    desc.event_cb   = event_cb;
    desc.width      = 1680;
    desc.height     = 1020;
    desc.window_title = "GUIMLAB STUDIO — AAA High-Density Neuromorphic Cockpit";
    desc.icon.sokol_default = true;
    desc.logger.func = slog_func;
    desc.high_dpi   = true;
    return desc;
}
