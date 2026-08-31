// ============================================================================
//  test_sarts_ring_buffer.cpp - SARTS MPMC ring-buffer verification
//  ============================================================================
//  Non-invasive unit tests for guim::sarts::AsyncRingBuffer.
//  Validates wire-format stability, lock-free semantics, back-pressure drop,
//  and FIFO ordering under single- and multi-threaded producers/consumers.
// ============================================================================

#include <gtest/gtest.h>

#include "sarts_ring_buffer.hpp"

#include <atomic>
#include <chrono>
#include <random>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

using namespace guim::sarts;

// ----------------------------------------------------------------------------
//  Wire-format ABI stability - MUST remain 24 bytes across all platforms
// ----------------------------------------------------------------------------
TEST(SartsRingBuffer, TelemetrySampleSizeIs24Bytes) {
    EXPECT_EQ(sizeof(TelemetrySample), 24u)
        << "SARTS wire-format ABI break - downstream panels assume 24 bytes";
}

TEST(SartsRingBuffer, TelemetrySampleIsTriviallyCopyable) {
    static_assert(std::is_trivially_copyable_v<TelemetrySample>,
        "TelemetrySample must be trivially copyable for zero-copy push/pop");
}

// ----------------------------------------------------------------------------
//  Basic single-threaded push/pop
// ----------------------------------------------------------------------------
TEST(SartsRingBuffer, EmptyQueueTryPopReturnsFalse) {
    AsyncRingBuffer<8> rb;
    TelemetrySample out{};
    EXPECT_FALSE(rb.try_pop(out));
    EXPECT_EQ(rb.approx_size(), 0u);
}

TEST(SartsRingBuffer, SinglePushPopRoundTrip) {
    AsyncRingBuffer<8> rb;
    TelemetrySample in{};
    in.timestamp_ns = 1234567890ull;
    in.latency_p99_ns = 87.5f;
    in.drift_hamilton = 0.00123f;
    in.seq = 42u;
    in.flags = 0u;

    ASSERT_TRUE(rb.try_push(in));
    EXPECT_EQ(rb.approx_size(), 1u);

    TelemetrySample out{};
    ASSERT_TRUE(rb.try_pop(out));
    EXPECT_EQ(out.timestamp_ns, 1234567890ull);
    EXPECT_FLOAT_EQ(out.latency_p99_ns, 87.5f);
    EXPECT_FLOAT_EQ(out.drift_hamilton, 0.00123f);
    EXPECT_EQ(out.seq, 42u);
    EXPECT_EQ(rb.approx_size(), 0u);
}

TEST(SartsRingBuffer, FifoOrderingPreserved) {
    AsyncRingBuffer<64> rb;
    constexpr std::size_t N = 32;

    for (std::size_t i = 0; i < N; ++i) {
        TelemetrySample s{};
        s.seq = static_cast<std::uint32_t>(i);
        ASSERT_TRUE(rb.try_push(s));
    }

    for (std::size_t i = 0; i < N; ++i) {
        TelemetrySample out{};
        ASSERT_TRUE(rb.try_pop(out));
        EXPECT_EQ(out.seq, static_cast<std::uint32_t>(i));
    }
}

// ----------------------------------------------------------------------------
//  Back-pressure: drop when full, never block
// ----------------------------------------------------------------------------
TEST(SartsRingBuffer, FullQueueTryPushReturnsFalse) {
    AsyncRingBuffer<4> rb;

    for (std::size_t i = 0; i < rb.max_capacity(); ++i) {
        TelemetrySample s{};
        s.seq = static_cast<std::uint32_t>(i);
        EXPECT_TRUE(rb.try_push(s))
            << "push " << i << " should succeed (capacity=" << rb.max_capacity() << ")";
    }

    TelemetrySample overflow{};
    overflow.seq = 999u;
    EXPECT_FALSE(rb.try_push(overflow))
        << "push to full ring must return false (back-pressure drop)";
}

TEST(SartsRingBuffer, ApproxSizeNonZeroWhenNonEmpty) {
    AsyncRingBuffer<16> rb;
    for (int i = 0; i < 5; ++i) {
        TelemetrySample s{};
        ASSERT_TRUE(rb.try_push(s));
    }
    EXPECT_GE(rb.approx_size(), 5u);
    EXPECT_LE(rb.approx_size(), 6u);
}

// ----------------------------------------------------------------------------
//  Capacity boundary - wrap-around correctness
// ----------------------------------------------------------------------------
TEST(SartsRingBuffer, WrapAroundMaintainsFifoOrder) {
    AsyncRingBuffer<4> rb;

    for (int round = 0; round < 4; ++round) {
        for (int i = 0; i < 3; ++i) {
            TelemetrySample s{};
            s.seq = static_cast<std::uint32_t>(round * 10 + i);
            ASSERT_TRUE(rb.try_push(s));
        }
        for (int i = 0; i < 3; ++i) {
            TelemetrySample out{};
            ASSERT_TRUE(rb.try_pop(out));
            EXPECT_EQ(out.seq, static_cast<std::uint32_t>(round * 10 + i));
        }
    }
}

// ----------------------------------------------------------------------------
//  Multi-threaded stress - 2 producers x 2 consumers
// ----------------------------------------------------------------------------
TEST(SartsRingBuffer, ConcurrentProducersConsumersStress) {
    constexpr std::size_t CAP = 1024;
    constexpr int N_PER_PRODUCER = 5000;
    AsyncRingBuffer<CAP> rb;

    std::atomic<int> produced_count{0};
    std::atomic<int> consumed_count{0};

    auto producer = [&](int id) {
        for (int i = 0; i < N_PER_PRODUCER; ++i) {
            TelemetrySample s{};
            s.seq = static_cast<std::uint32_t>(id * N_PER_PRODUCER + i);
            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (!rb.try_push(s)) {
                if (std::chrono::steady_clock::now() > deadline) {
                    ADD_FAILURE() << "producer " << id << " timed out at i=" << i;
                    return;
                }
                std::this_thread::yield();
            }
            produced_count.fetch_add(1, std::memory_order_relaxed);
        }
    };

    auto consumer = [&]() {
        TelemetrySample out{};
        while (consumed_count.load() < 2 * N_PER_PRODUCER) {
            if (rb.try_pop(out)) {
                consumed_count.fetch_add(1, std::memory_order_relaxed);
            } else {
                std::this_thread::yield();
            }
        }
    };

    std::thread p0(producer, 0);
    std::thread p1(producer, 1);
    std::thread c0(consumer);
    std::thread c1(consumer);

    p0.join(); p1.join();
    c0.join(); c1.join();

    EXPECT_EQ(produced_count.load(), 2 * N_PER_PRODUCER);
    EXPECT_EQ(consumed_count.load(), 2 * N_PER_PRODUCER);
    EXPECT_EQ(rb.approx_size(), 0u);
}

// ----------------------------------------------------------------------------
//  Compile-time guards - power-of-two & minimum capacity
// ----------------------------------------------------------------------------
TEST(SartsRingBuffer, StaticAssertionsCompileTime) {
    AsyncRingBuffer<2> rb2;
    AsyncRingBuffer<8> rb8;
    AsyncRingBuffer<1024> rb1024;
    EXPECT_EQ(rb2.max_capacity(), 2u);
    EXPECT_EQ(rb8.max_capacity(), 8u);
    EXPECT_EQ(rb1024.max_capacity(), 1024u);
}

} // namespace
