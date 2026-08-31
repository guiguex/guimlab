// ============================================================================
//  sarts_ring_buffer.h — Sokol Asynchronous Ring-Buffer Telemetry Surface (SARTS)
//  ============================================================================
//  Lock-free, zero-allocation, cache-line aligned (64 bytes) SPSC/MPMC telemetry
//  ring buffer bridging bare-metal CUDA AGI kernels with the Sokol UI surface.
//
//  Author  : GuimLab Core Team
//  License : AGPL-3.0 / MIT
// ============================================================================

#ifndef GUIM_SARTS_RING_BUFFER_H
#define GUIM_SARTS_RING_BUFFER_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace guim::sarts {

constexpr std::size_t CACHE_LINE_BYTES = 64;

// ----------------------------------------------------------------------------
//  Telemetry Sample (POD, deterministic memory layout, zero copy)
// ----------------------------------------------------------------------------
struct alignas(CACHE_LINE_BYTES) SartsTelemetrySample {
    std::uint64_t timestamp_ns;     // Monotonic host/device timestamp
    float         latency_p50_us;   // Median round-trip latency
    float         latency_p99_us;   // 99th percentile latency
    float         hamiltonian_energy;// SR-MIT symplectic invariant H = 0.5*(E^2+P^2)
    float         dopamine;         // TD-error reward modulator delta(t)
    float         serotonin;        // Plasticity threshold modulator
    float         ddwr_sparsity;    // VRAM bandwidth reduction ratio (0.0 to 1.0)
    std::uint32_t active_units;     // Number of active Lovelace neurons (0..256)
    std::uint32_t dead_units;       // CBP neurogenesis trigger count
    std::uint64_t seq;              // Monotonic sequence counter
    std::uint64_t reserved[2];      // Future expansion padding
};

static_assert(sizeof(SartsTelemetrySample) == 64, "SartsTelemetrySample must fit exactly in 1 cache line (64 bytes)");
static_assert(std::is_trivially_copyable_v<SartsTelemetrySample>, "Must be trivially copyable for bare-metal SHM transfers");

// ----------------------------------------------------------------------------
//  Lock-Free Bounded Ring Buffer (Vyukov SPSC/MPMC design)
// ----------------------------------------------------------------------------
template<std::size_t CAPACITY = 1024>
class alignas(CACHE_LINE_BYTES) SartsRingBuffer {
    static_assert((CAPACITY & (CAPACITY - 1)) == 0, "SARTS CAPACITY must be a power of 2 for bitwise masking");
    static constexpr std::size_t MASK = CAPACITY - 1;

    // Buffer storage
    SartsTelemetrySample m_storage[CAPACITY];

    // Head (Producer) & Tail (Consumer) on strictly isolated cache lines
    alignas(CACHE_LINE_BYTES) std::atomic<std::size_t> m_head{0};
    alignas(CACHE_LINE_BYTES) std::atomic<std::size_t> m_tail{0};
    alignas(CACHE_LINE_BYTES) std::atomic<std::uint64_t> m_dropped_frames{0};

public:
    constexpr SartsRingBuffer() noexcept = default;

    // Non-blocking try_push: Drops sample if full (Backpressure isolation)
    bool try_push(const SartsTelemetrySample& sample) noexcept {
        const std::size_t head = m_head.load(std::memory_order_relaxed);
        const std::size_t tail = m_tail.load(std::memory_order_acquire);

        if (head - tail >= CAPACITY) {
            m_dropped_frames.fetch_add(1, std::memory_order_relaxed);
            return false; // Backpressure drop: Never blocks producer kernel
        }

        m_storage[head & MASK] = sample;
        m_head.store(head + 1, std::memory_order_release);
        return true;
    }

    // Overwriting push: Always succeeds by overwriting stale data (Ideal for real-time 144Hz telemetry)
    void push_overwrite(const SartsTelemetrySample& sample) noexcept {
        const std::size_t head = m_head.load(std::memory_order_relaxed);
        std::size_t tail = m_tail.load(std::memory_order_relaxed);

        if (head - tail >= CAPACITY) {
            m_tail.store(tail + 1, std::memory_order_relaxed);
            m_dropped_frames.fetch_add(1, std::memory_order_relaxed);
        }

        m_storage[head & MASK] = sample;
        m_head.store(head + 1, std::memory_order_release);
    }

    // Non-blocking try_pop: Reads oldest sample
    bool try_pop(SartsTelemetrySample& out_sample) noexcept {
        const std::size_t tail = m_tail.load(std::memory_order_relaxed);
        const std::size_t head = m_head.load(std::memory_order_acquire);

        if (tail >= head) {
            return false; // Queue is empty
        }

        out_sample = m_storage[tail & MASK];
        m_tail.store(tail + 1, std::memory_order_release);
        return true;
    }

    // Drain latest sample (skips stale intermediate frames)
    bool pop_latest(SartsTelemetrySample& out_sample) noexcept {
        const std::size_t head = m_head.load(std::memory_order_acquire);
        const std::size_t tail = m_tail.load(std::memory_order_relaxed);

        if (tail >= head) {
            return false;
        }

        out_sample = m_storage[(head - 1) & MASK];
        m_tail.store(head, std::memory_order_release);
        return true;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        const std::size_t head = m_head.load(std::memory_order_relaxed);
        const std::size_t tail = m_tail.load(std::memory_order_relaxed);
        return (head >= tail) ? (head - tail) : 0;
    }

    [[nodiscard]] bool empty() const noexcept {
        return m_head.load(std::memory_order_relaxed) == m_tail.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::uint64_t dropped_count() const noexcept {
        return m_dropped_frames.load(std::memory_order_relaxed);
    }

    void reset() noexcept {
        m_head.store(0, std::memory_order_relaxed);
        m_tail.store(0, std::memory_order_relaxed);
        m_dropped_frames.store(0, std::memory_order_relaxed);
    }
};

} // namespace guim::sarts

#endif // GUIM_SARTS_RING_BUFFER_H
