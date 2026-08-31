// ============================================================================
//  sarts_ring_buffer.hpp — Sokol Asynchronous Ring-Buffer Telemetry Surface
//  ============================================================================
//  SARTS — Dmitry Vyukov Bounded Lock-Free MPMC Queue for Engine -> Studio
//  telemetry. Wire-format stable, zero allocation, cache-line padded.
//
//  Author        : Guimlab Contributors
//  License       : AGPL-3.0 / MIT
// ============================================================================

#ifndef GUIM_SARTS_RING_BUFFER_HPP
#define GUIM_SARTS_RING_BUFFER_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace guim::sarts {

// Cache line size for x86_64 and aarch64 architectures
constexpr std::size_t CACHE_LINE = 64;

// ----------------------------------------------------------------------------
//  TelemetrySample — 24-byte wire-format stable POD sample
// ----------------------------------------------------------------------------
struct alignas(8) TelemetrySample {
    std::uint64_t timestamp_ns;
    float         latency_p99_ns;
    float         drift_hamilton;
    std::uint32_t seq;
    std::uint32_t flags;
};

static_assert(sizeof(TelemetrySample) == 24,
    "TelemetrySample wire-format must remain 24 bytes for SARTS ABI stability");
static_assert(std::is_trivially_copyable_v<TelemetrySample>,
    "TelemetrySample must be trivially copyable");

// ----------------------------------------------------------------------------
//  AsyncRingBuffer — Dmitry Vyukov Lock-Free Bounded MPMC Queue
// ----------------------------------------------------------------------------
template<std::size_t CAPACITY>
class alignas(CACHE_LINE) AsyncRingBuffer {
    static_assert(CAPACITY >= 2, "AsyncRingBuffer CAPACITY must be >= 2");
    static_assert((CAPACITY & (CAPACITY - 1)) == 0, "AsyncRingBuffer CAPACITY must be a power of two");

    static constexpr std::size_t mask = CAPACITY - 1;

    struct Cell {
        std::atomic<std::size_t> sequence;
        TelemetrySample data;
    };

public:
    using sample_type = TelemetrySample;
    static constexpr std::size_t capacity = CAPACITY;

    AsyncRingBuffer() noexcept {
        for (std::size_t i = 0; i < CAPACITY; ++i) {
            m_buf[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    ~AsyncRingBuffer() = default;

    AsyncRingBuffer(const AsyncRingBuffer&) = delete;
    AsyncRingBuffer& operator=(const AsyncRingBuffer&) = delete;
    AsyncRingBuffer(AsyncRingBuffer&&) = delete;
    AsyncRingBuffer& operator=(AsyncRingBuffer&&) = delete;

    [[nodiscard]] bool try_push(const TelemetrySample& s) noexcept {
        Cell* cell;
        std::size_t pos = m_head.load(std::memory_order_relaxed);
        for (;;) {
            cell = &m_buf[pos & mask];
            std::size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
            if (diff == 0) {
                if (m_head.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                m_dropped.fetch_add(1, std::memory_order_relaxed);
                return false; // Backpressure drop: Queue is full
            } else {
                pos = m_head.load(std::memory_order_relaxed);
            }
        }

        cell->data = s;
        cell->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    [[nodiscard]] bool try_pop(TelemetrySample& out) noexcept {
        Cell* cell;
        std::size_t pos = m_tail.load(std::memory_order_relaxed);
        for (;;) {
            cell = &m_buf[pos & mask];
            std::size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
            if (diff == 0) {
                if (m_tail.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false; // Queue is empty
            } else {
                pos = m_tail.load(std::memory_order_relaxed);
            }
        }

        out = cell->data;
        cell->sequence.store(pos + mask + 1, std::memory_order_release);
        return true;
    }

    [[nodiscard]] std::size_t approx_size() const noexcept {
        const std::size_t head = m_head.load(std::memory_order_relaxed);
        const std::size_t tail = m_tail.load(std::memory_order_relaxed);
        return (head >= tail) ? (head - tail) : 0;
    }

    [[nodiscard]] constexpr std::size_t max_capacity() const noexcept {
        return CAPACITY;
    }

    [[nodiscard]] std::uint64_t dropped_count() const noexcept {
        return m_dropped.load(std::memory_order_relaxed);
    }

    void reset() noexcept {
        m_head.store(0, std::memory_order_relaxed);
        m_tail.store(0, std::memory_order_relaxed);
        m_dropped.store(0, std::memory_order_relaxed);
        for (std::size_t i = 0; i < CAPACITY; ++i) {
            m_buf[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

private:
    Cell m_buf[CAPACITY];

    alignas(CACHE_LINE) std::atomic<std::size_t> m_head{0};
    alignas(CACHE_LINE) std::atomic<std::size_t> m_tail{0};
    alignas(CACHE_LINE) std::atomic<std::uint64_t> m_dropped{0};
};

} // namespace guim::sarts

#endif // GUIM_SARTS_RING_BUFFER_HPP
