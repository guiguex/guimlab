// ============================================================================
//  lovelace_cortex.h — Lovelace Cortex with Warp Tensor Cores (mma.sync) & GRM-FIN
//  ============================================================================
//  Author        : Guimlab Contributors
//  License       : AGPL-3.0 / MIT
// ============================================================================

#ifndef GUIM_LOVELACE_CORTEX_H
#define GUIM_LOVELACE_CORTEX_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <span>
#include "ipc_structs_v2.h"

namespace guim {

// Golden Ratio Invariant Constants (KAM Spectral Invariance)
constexpr float GOLDEN_RATIO_PHI = 1.61803398875f;
constexpr float GOLDEN_RATIO_INV = 0.61803398875f;  // phi = 1/Phi

struct LovelaceCortexState {
    half*          d_weights_fp16 = nullptr;  // [GUIM_STATE_DIM * GUIM_INPUT_DIM]
    half*          d_alphas_fp16  = nullptr;  // [GUIM_STATE_DIM * GUIM_INPUT_DIM]
    float*         d_hidden       = nullptr;  // [GUIM_STATE_DIM]
    float*         d_traces       = nullptr;  // [GUIM_STATE_DIM * GUIM_INPUT_DIM]
    float*         d_meta_traces  = nullptr;  // [GUIM_STATE_DIM * GUIM_INPUT_DIM]
    float*         d_betas        = nullptr;  // [GUIM_STATE_DIM * GUIM_INPUT_DIM]
    std::uint64_t* d_last_t       = nullptr;  // [GUIM_STATE_DIM * GUIM_INPUT_DIM]
    cudaStream_t   stream         = nullptr;
};

// Host wrapper class for standalone validation & benchmarking
class LovelaceCortexEngine {
public:
    explicit LovelaceCortexEngine(std::uint32_t seed = 0xC0FFEE);
    ~LovelaceCortexEngine();

    LovelaceCortexEngine(const LovelaceCortexEngine&) = delete;
    LovelaceCortexEngine& operator=(const LovelaceCortexEngine&) = delete;

    // Tensor Core accelerated cortical step
    void step(
        std::span<const float, GUIM_INPUT_DIM> input,
        float dopamine,
        float serotonin,
        std::span<float, GUIM_STATE_DIM> output,
        std::uint64_t timestamp_ns
    );

    // Scalar reference step (for numerical parity validation)
    void step_reference(
        std::span<const float, GUIM_INPUT_DIM> input,
        float dopamine,
        float serotonin,
        std::span<float, GUIM_STATE_DIM> output,
        std::uint64_t timestamp_ns
    );

private:
    LovelaceCortexState state_;
    float* d_input_f32_ = nullptr;
    float* d_output_f32_ = nullptr;
};

}  // namespace guim

#endif // GUIM_LOVELACE_CORTEX_H
