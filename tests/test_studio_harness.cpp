// ============================================================================
//  test_studio_harness.cpp — Headless Validation for GuimLab Studio
//  ============================================================================

#include <gtest/gtest.h>
#include "guim_studio.h"
#include "guim_viewport_session.h"

#include <chrono>
#include <cmath>

namespace {

TEST(StudioHarnessTest, CircularBufferOperations) {
    guim::studio::CircularBuffer<float, 8> buf;
    EXPECT_EQ(buf.size(), 0);
    EXPECT_EQ(buf.capacity(), 8);

    for (int i = 1; i <= 8; ++i) {
        buf.push(static_cast<float>(i));
    }
    EXPECT_EQ(buf.size(), 8);
    EXPECT_FLOAT_EQ(buf[0], 1.0f);
    EXPECT_FLOAT_EQ(buf[7], 8.0f);

    // Push 2 more to test wrap-around
    buf.push(9.0f);
    buf.push(10.0f);
    EXPECT_EQ(buf.size(), 8);
    EXPECT_FLOAT_EQ(buf[0], 3.0f);
    EXPECT_FLOAT_EQ(buf[7], 10.0f);
}

TEST(StudioHarnessTest, TelemetryStructInitialization) {
    guim::studio::StudioTelemetry telemetry;
    EXPECT_FALSE(telemetry.connected);
    EXPECT_EQ(telemetry.total_frames, 0);
    EXPECT_FLOAT_EQ(telemetry.dopamine, 0.0f);
    EXPECT_FLOAT_EQ(telemetry.serotonin, 0.0f);

    EXPECT_EQ(telemetry.cortex_activations.size(), GUIM_STATE_DIM);
    EXPECT_EQ(telemetry.reflex_motors.size(), GUIM_REFLEX_MOTORS);
}

TEST(StudioHarnessTest, ShmClientDisconnectSafe) {
    guim::studio::ShmClient client;
    EXPECT_FALSE(client.is_connected());
    
    // Disconnecting a non-connected client must not crash
    client.disconnect();
    EXPECT_FALSE(client.is_connected());

    // Polling disconnected client must return false gracefully
    guim::studio::StudioTelemetry telemetry;
    bool polled = client.poll(telemetry);
    EXPECT_FALSE(polled);
    EXPECT_FALSE(telemetry.connected);
}

TEST(StudioHarnessTest, ViewportSessionRecorderRecording) {
    guim::studio::ViewportSessionRecorder recorder("test_session_output.json");
    EXPECT_FALSE(recorder.is_recording());
    EXPECT_EQ(recorder.recorded_frames_count(), 0);

    recorder.start();
    EXPECT_TRUE(recorder.is_recording());

    guim::studio::StudioTelemetry telemetry;
    telemetry.total_frames = 100;
    telemetry.dopamine = 0.5f;
    telemetry.serotonin = 0.2f;
    telemetry.p50_latency_us = 310.0f;
    telemetry.cortex_activations[0] = 0.8f;

    recorder.record_frame(telemetry);
    EXPECT_EQ(recorder.recorded_frames_count(), 1);

    recorder.stop();
    EXPECT_FALSE(recorder.is_recording());
}

} // namespace
