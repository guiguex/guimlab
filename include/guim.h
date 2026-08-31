// =============================================================================
//  guim/guim.h  -  Modern C++20 wrapper around the guim_kernel C ABI.
//
//  Author        : Guimlab contributors
//  SPDX-License   : MIT
//
//  This header provides a const-correct, noexcept-friendly, RAII-friendly
//  facade over the lower-level guim_kernel.h C ABI (see ../guim_kernel.h).
//  The C ABI remains the source of truth for kernel versioning and ABI
//  stability; this header is sugar for the C++ test/bench code.
//
//  All numeric / behavioral constants (alpha, lambda, decay, dimensions) are
//  declared exactly once in <guim_kernel.h>.  Do NOT duplicate them here.
// =============================================================================
#ifndef GUIM_GUIM_H
#define GUIM_GUIM_H

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "guim_kernel.h"  // C ABI + canonical constants

namespace guim {

// ---- Strong types for input / output / weight buffers ----------------------
using InputView  = std::span<const float, GUIM_INPUT_DIM>;
using OutputView = std::span<float,       GUIM_STATE_DIM>;

// Forward declaration: the C ABI returns an opaque struct.
struct EngineState;

// -----------------------------------------------------------------------------
//  Engine
//
//  Thin RAII wrapper around GuimKernel*.  All operations are noexcept because
//  the C ABI performs all error checking internally and surfaces failures via
//  NaN return values; we propagate those rather than throwing (this is a
//  real-time kernel, no allocator churn on the hot path).
// -----------------------------------------------------------------------------
class Engine {
public:
    // Construct a fresh engine.  `seed` feeds the PRNG for weight init and
    // must be non-zero for reproducibility.  Throws std::bad_alloc only if the
    // host-side metadata allocation fails.
    explicit Engine(std::uint32_t seed) noexcept;

    // Non-copyable, movable (cheap; the GPU handle is heap-allocated and the
    // move just transfers ownership of the pointer).
    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&& other) noexcept;
    Engine& operator=(Engine&& other) noexcept;

    ~Engine() noexcept;

    // ---- Per-frame update --------------------------------------------------
    //  input   : GUIM_INPUT_DIM floats  (caller-owned)
    //  output  : GUIM_STATE_DIM floats  (caller-owned, written by step)
    //  returns : TD-error magnitude for the frame, or NaN on numerical fault.
    //
    //  `step` is const because the device memory is mutable via the opaque
    //  pointer; the Engine object itself holds only the handle.
    [[nodiscard]] float step(InputView input, OutputView output) const noexcept;

    // ---- Introspection (cheap) --------------------------------------------
    [[nodiscard]] std::uint32_t resets_count()  const noexcept;
    [[nodiscard]] std::uint64_t frames_run()    const noexcept;
    [[nodiscard]] std::size_t   bytes_resident() const noexcept;

    // Bulk read-back of internal state.  Throws std::bad_alloc only.
    void copy_hidden (std::span<float, GUIM_STATE_DIM> dst) const noexcept;
    void copy_weights(std::span<float, GUIM_STATE_DIM * GUIM_TOTAL_DIM> dst) const noexcept;

    // ---- Access to the raw C handle (for tests / harness) ----------------
    GuimKernel* handle() noexcept { return state_; }
    const GuimKernel* handle() const noexcept { return state_; }

private:
    GuimKernel* state_;   // owned; nullptr only after move
};

// =============================================================================
//  Inlined trivial accessors.
//  Out-of-line definitions live in src/guim_engine.cpp (not strictly required
//  by the build; we keep this header self-contained so eager compilers see
//  the contracts).
// =============================================================================
inline Engine::Engine(std::uint32_t seed) noexcept
    : state_(guim_create(seed))
{}

inline Engine::Engine(Engine&& other) noexcept
    : state_(other.state_)
{
    other.state_ = nullptr;
}

inline Engine& Engine::operator=(Engine&& other) noexcept
{
    if (this != &other) {
        if (state_) guim_destroy(state_);
        state_ = other.state_;
        other.state_ = nullptr;
    }
    return *this;
}

inline Engine::~Engine() noexcept
{
    if (state_) guim_destroy(state_);
}

inline float Engine::step(InputView input, OutputView output) const noexcept
{
    return guim_step(state_, input.data(), output.data());
}

inline std::uint32_t Engine::resets_count() const noexcept
{
    return guim_resets_count(state_);
}

inline std::uint64_t Engine::frames_run() const noexcept
{
    return guim_frames_run(state_);
}

inline std::size_t Engine::bytes_resident() const noexcept
{
    return guim_bytes_resident(state_);
}

inline void Engine::copy_hidden(std::span<float, GUIM_STATE_DIM> dst) const noexcept
{
    guim_get_hidden(state_, dst.data());
}

inline void Engine::copy_weights(
    std::span<float, GUIM_STATE_DIM * GUIM_TOTAL_DIM> dst) const noexcept
{
    guim_get_weights(state_, dst.data());
}

}  // namespace guim

#endif  // GUIM_GUIM_H