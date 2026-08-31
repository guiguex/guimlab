// ============================================================================
//  guim_viewport_session.h — Viewport Session Recording & Replay Engine
//  ============================================================================

#ifndef GUIM_VIEWPORT_SESSION_H
#define GUIM_VIEWPORT_SESSION_H

#include "guim_studio.h"
#include <string>
#include <vector>
#include <fstream>

namespace guim::studio {

struct SessionFrameData {
    std::uint64_t frame_index{0};
    std::uint64_t timestamp_ns{0};
    float dopamine{0.0f};
    float serotonin{0.0f};
    float rtt_latency_us{0.0f};
    float cortex_energy{0.0f};
};

class ViewportSessionRecorder {
public:
    explicit ViewportSessionRecorder(std::string output_path);
    ~ViewportSessionRecorder();

    void start();
    void record_frame(const StudioTelemetry& telemetry);
    void stop();

    [[nodiscard]] bool is_recording() const noexcept { return is_recording_; }
    [[nodiscard]] std::size_t recorded_frames_count() const noexcept { return frames_.size(); }

private:
    std::string output_path_;
    std::vector<SessionFrameData> frames_;
    bool is_recording_{false};
};

} // namespace guim::studio

#endif // GUIM_VIEWPORT_SESSION_H
