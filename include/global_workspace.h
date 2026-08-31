// ============================================================================
//  global_workspace.h — Global Workspace Theory (GWT) Attention & Broadcast
//  ============================================================================
//  Implements Baars/Dehaene Global Workspace Architecture on GPU:
//    1. GWT Bottleneck: Competitive soft-topK inhibition across modules
//    2. Global Broadcast Vector: Unifies sensory, cortical, and memory streams
//    3. Top-down neuromodulatory feedback to L0/L1/L2 layers
//
//  Author        : Guimlab Contributors
//  License       : AGPL-3.0 / MIT
// ============================================================================

#ifndef GUIM_GLOBAL_WORKSPACE_H
#define GUIM_GLOBAL_WORKSPACE_H

#include <cuda_runtime.h>
#include <cstdint>
#include <span>

namespace guim {

constexpr int GWT_MODULES       = 4;   // 0: Sensory, 1: Reflex L0, 2: Cortex L1/L2, 3: Episodic
constexpr int GWT_IN_DIM        = 64;  // Dimensions per module
constexpr int GWT_BROADCAST_DIM = 64;  // Central broadcast vector

struct GlobalWorkspaceState {
    float*       d_proposals    = nullptr;  // [GWT_MODULES * GWT_IN_DIM]
    float*       d_saliences    = nullptr;  // [GWT_MODULES]
    float*       d_weights      = nullptr;  // [GWT_MODULES] (softmax weights)
    float*       d_broadcast    = nullptr;  // [GWT_BROADCAST_DIM]
    cudaStream_t stream         = nullptr;
};

class GlobalWorkspace {
public:
    GlobalWorkspace();
    ~GlobalWorkspace();

    GlobalWorkspace(const GlobalWorkspace&) = delete;
    GlobalWorkspace& operator=(const GlobalWorkspace&) = delete;
    GlobalWorkspace(GlobalWorkspace&& other) noexcept;
    GlobalWorkspace& operator=(GlobalWorkspace&& other) noexcept;

    // Set proposal vector and salience for a specific module
    void set_proposal(int module_idx,
                      std::span<const float, GWT_IN_DIM> proposal,
                      float salience);

    // Run competitive GWT arbitration and compute broadcast state
    void arbitrate_and_broadcast(std::span<float, GWT_BROADCAST_DIM> broadcast_out,
                                float competition_temp = 5.0f);

    // Get the winning module index
    [[nodiscard]] int get_dominant_module() const;

private:
    GlobalWorkspaceState state_;
    float h_saliences_[GWT_MODULES] = {0};
    float h_broadcast_[GWT_BROADCAST_DIM] = {0};
};

}  // namespace guim

#endif // GUIM_GLOBAL_WORKSPACE_H
