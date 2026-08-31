// ============================================================================
//  episodic_vsa.h — Hippocampal Episodic Memory via VSA & Dense Hopfield
//  ============================================================================
//  Provides:
//    1. Vector Symbolic Architecture (VSA / HDC) algebraic operations:
//       - Bind: Circular / Hadamard multiplication
//       - Bundle: Normalized superposition
//       - Permute: Cyclic coordinate permutation for sequence encoding
//    2. Modern Dense Hopfield associative memory:
//       - Softmax-weighted exponential associative retrieval, complexity O(N*D) per query (N=stored, D=VSA_DIM)
//
//  Author        : Guimlab Contributors
//  License       : AGPL-3.0 / MIT
// ============================================================================

#ifndef GUIM_EPISODIC_VSA_H
#define GUIM_EPISODIC_VSA_H

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <span>

namespace guim {

constexpr int VSA_DIM           = 1024;
constexpr int HOPFIELD_CAPACITY = 1024;

struct HopfieldState {
    float*         d_keys         = nullptr;  // [HOPFIELD_CAPACITY * VSA_DIM]
    float*         d_values       = nullptr;  // [HOPFIELD_CAPACITY * VSA_DIM]
    float*         d_similarities = nullptr;  // [HOPFIELD_CAPACITY]
    float*         d_query        = nullptr;  // [VSA_DIM]
    float*         d_recalled     = nullptr;  // [VSA_DIM]
    std::uint32_t* d_count        = nullptr;  // [1]
    cudaStream_t   stream         = nullptr;
    std::uint32_t  current_count  = 0;
};

// C++20 Clean API
class EpisodicMemory {
public:
    explicit EpisodicMemory(std::uint32_t capacity = HOPFIELD_CAPACITY);
    ~EpisodicMemory();

    EpisodicMemory(const EpisodicMemory&) = delete;
    EpisodicMemory& operator=(const EpisodicMemory&) = delete;
    EpisodicMemory(EpisodicMemory&& other) noexcept;
    EpisodicMemory& operator=(EpisodicMemory&& other) noexcept;

    // Store key-value episode in one-shot
    void store(std::span<const float, VSA_DIM> key,
               std::span<const float, VSA_DIM> value);

    // Retrieve episode via softmax associative recall
    float recall(std::span<const float, VSA_DIM> query,
                 std::span<float, VSA_DIM> recalled_out,
                 float beta = 8.0f);

    // VSA algebraic operations
    static void bind(const float* d_a, const float* d_b, float* d_out, cudaStream_t stream = nullptr);
    static void bundle(const float* d_a, const float* d_b, float* d_out, cudaStream_t stream = nullptr);
    static void permute(const float* d_in, float* d_out, int shift, cudaStream_t stream = nullptr);

    [[nodiscard]] std::uint32_t size() const noexcept { return state_.current_count; }
    [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }

private:
    HopfieldState state_;
    std::uint32_t capacity_;
};

}  // namespace guim

#endif // GUIM_EPISODIC_VSA_H



