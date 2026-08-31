// ============================================================================
//  delta_filter.cpp — AVX2 branchless sensor delta router (CPU client side)
//  ---------------------------------------------------------------------------
//  Scans sensor vector, computes deltas > EPSILON, writes to sparse IPC.
//  Uses AVX2 vectorization + branchless bitscan (C++20 std::countr_zero).
//
//  Author        : Guimlab Contributors
//  License       : AGPL-3.0
// ============================================================================

#include <immintrin.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <bit>
#include <atomic>
#include <chrono>

#include "ipc_structs.h"
#include "ipc_structs_sparse.h"

class SensorDeltaRouter {
private:
    alignas(32) float previous_state_[GUIM_INPUT_DIM];
    GuimSharedMemorySparse* ipc_;

public:
    explicit SensorDeltaRouter(GuimSharedMemorySparse* shared_mem)
        : ipc_(shared_mem) {
        std::memset(previous_state_, 0, sizeof(previous_state_));
    }

    int route_deltas(const float* current_sensor_state,
                     std::uint64_t current_timestamp_ns) {
        #if defined(__AVX2__)
        const __m256 v_epsilon   = _mm256_set1_ps(1e-4f);
        const __m256 v_abs_mask  = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

        int delta_count = 0;

        for (int i = 0; i < GUIM_INPUT_DIM; i += 8) {
            const __m256 v_new = _mm256_loadu_ps(&current_sensor_state[i]);
            const __m256 v_old = _mm256_load_ps(&previous_state_[i]);

            const __m256 v_diff     = _mm256_sub_ps(v_new, v_old);
            const __m256 v_abs_diff = _mm256_and_ps(v_diff, v_abs_mask);

            const __m256 v_cmp = _mm256_cmp_ps(v_abs_diff, v_epsilon, _CMP_GT_OQ);
            unsigned int mask = static_cast<unsigned int>(_mm256_movemask_ps(v_cmp));

            while (mask != 0) {
                const int bit_idx = std::countr_zero(mask);
                const int global_idx = i + bit_idx;

                if (delta_count < GUIM_MAX_DELTAS) {
                    ipc_->delta_indices[delta_count] = global_idx;
                    ipc_->delta_values[delta_count]  = current_sensor_state[global_idx] - previous_state_[global_idx];
                    ++delta_count;
                }

                mask &= (mask - 1);
            }

            _mm256_store_ps(&previous_state_[i], v_new);
        }

        flush_to_ipc(delta_count, current_timestamp_ns);
        return delta_count;
        #else
        // Scalar fallback
        int delta_count = 0;
        for (int i = 0; i < GUIM_INPUT_DIM; ++i) {
            float diff = current_sensor_state[i] - previous_state_[i];
            if (std::abs(diff) > 1e-4f) {
                if (delta_count < GUIM_MAX_DELTAS) {
                    ipc_->delta_indices[delta_count] = i;
                    ipc_->delta_values[delta_count]  = diff;
                    ++delta_count;
                }
            }
            previous_state_[i] = current_sensor_state[i];
        }
        flush_to_ipc(delta_count, current_timestamp_ns);
        return delta_count;
        #endif
    }

private:
    void flush_to_ipc(int delta_count, std::uint64_t current_timestamp_ns) {
        std::atomic_thread_fence(std::memory_order_release);
        ipc_->delta_count  = delta_count;
        ipc_->timestamp_ns = current_timestamp_ns;
        ipc_->seq_in       = ipc_->seq_in + 1;
    }
};

#if defined(GUIM_DELTA_STANDALONE)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

int main() {
    int shm_fd = ::shm_open("/guim_ipc_shm_sparse", O_RDWR, 0666);
    if (shm_fd < 0) {
        std::fprintf(stderr, "[delta_filter] shm_open failed; is the server running?\n");
        return 1;
    }
    void* m = ::mmap(nullptr, sizeof(GuimSharedMemorySparse), PROT_READ | PROT_WRITE,
                     MAP_SHARED, shm_fd, 0);
    if (m == MAP_FAILED) {
        std::perror("mmap");
        ::close(shm_fd);
        return 1;
    }

    auto* ipc = reinterpret_cast<GuimSharedMemorySparse*>(m);
    SensorDeltaRouter router(ipc);

    float sensors[GUIM_INPUT_DIM] = {0};
    for (int step = 0; step < 1000; ++step) {
        sensors[step % GUIM_INPUT_DIM] += 0.1f;
        const auto now = std::chrono::steady_clock::now();
        const std::uint64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()
        ).count();

        router.route_deltas(sensors, now_ns);
    }

    ::munmap(m, sizeof(GuimSharedMemorySparse));
    ::close(shm_fd);
    return 0;
}
#endif