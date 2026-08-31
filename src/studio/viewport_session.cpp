// ============================================================================
//  viewport_session.cpp — Viewport Session Recording & Replay Engine
//  ============================================================================

#include "guim_viewport_session.h"
#include <iostream>
#include <iomanip>
#include <numeric>
#include <cmath>

namespace guim::studio {

ViewportSessionRecorder::ViewportSessionRecorder(std::string output_path)
    : output_path_(std::move(output_path)) {}

ViewportSessionRecorder::~ViewportSessionRecorder() {
    if (is_recording_) {
        stop();
    }
}

void ViewportSessionRecorder::start() {
    frames_.clear();
    frames_.reserve(10000);
    is_recording_ = true;
}

void ViewportSessionRecorder::record_frame(const StudioTelemetry& telemetry) {
    if (!is_recording_) return;

    SessionFrameData frame;
    frame.frame_index    = telemetry.total_frames;
    frame.timestamp_ns   = telemetry.last_timestamp_ns;
    frame.dopamine       = telemetry.dopamine;
    frame.serotonin      = telemetry.serotonin;
    frame.rtt_latency_us = telemetry.p50_latency_us;

    // Calculate total cortical activation energy
    float energy = 0.0f;
    for (float val : telemetry.cortex_activations) {
        energy += (val * val);
    }
    frame.cortex_energy = energy;

    frames_.push_back(frame);
}

void ViewportSessionRecorder::stop() {
    if (!is_recording_) return;
    is_recording_ = false;

    if (output_path_.empty() || frames_.empty()) {
        return;
    }

    std::ofstream out(output_path_);
    if (!out.is_open()) {
        std::cerr << "[ViewportSession] Failed to open output file: " << output_path_ << "\n";
        return;
    }

    out << "{\n";
    out << "  \"total_frames\": " << frames_.size() << ",\n";
    out << "  \"frames\": [\n";

    for (std::size_t i = 0; i < frames_.size(); ++i) {
        const auto& f = frames_[i];
        out << "    {\"frame\": " << f.frame_index
            << ", \"dopamine\": " << f.dopamine
            << ", \"serotonin\": " << f.serotonin
            << ", \"rtt_us\": " << f.rtt_latency_us
            << ", \"energy\": " << f.cortex_energy
            << "}" << (i + 1 < frames_.size() ? ",\n" : "\n");
    }

    out << "  ]\n";
    out << "}\n";
    out.close();

    std::cout << "[ViewportSession] Session saved with " << frames_.size() << " frames to " << output_path_ << "\n";
}

} // namespace guim::studio
