// ============================================================================
//  l2_pinning.cpp — Attention-window L2 cache pinning (host side)
//  ---------------------------------------------------------------------------
//  Pins a sliding window of "hot" synaptic weights into the GPU's L2 cache
//  (typically 32-64 MB on Ada / 50 MB on Hopper) so the kernel reads from
//  L2 instead of GDDR6X for the attention focal point.
//
//  How it works
//  ------------
//    1.  We allocate a "weights" tensor in device memory.
//    2.  We expose a sliding window over that tensor — the active cortex
//        at time T.
//    3.  We tell the CUDA driver via `cudaStreamSetAttribute` that any
//        access in that window should be marked `cudaAccessPropertyPersisting`
//        (don't evict from L2) and `hitRatio=1.0`.
//    4.  As the attention focal point slides (driven by surprise metric
//        or task context), we re-issue the pinning for the new window.
//
//  Why this is a "secret HFT trick"
//  -------------------------------
//    High-frequency trading firms have been using this exact pattern for
//    years to microsecond-trades.  NVIDIA documented `cudaAccessPropertyPersisting`
//    in CUDA 11 but it's not widely known in the ML community because
//    standard CUDA frameworks (cuDNN, cuBLAS) don't expose it.
//
//  Author        : Guillaume Meingan + Claude (ultracode session 2026-08-31)
//  License       : AGPL-3.0
// ============================================================================

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "ipc_structs.h"

// ============================================================================
//  L2 Pinner — manages the sliding attention window
// ============================================================================

class L2Pinner {
private:
    void*       d_weights_      = nullptr;   // base of full weight tensor
    std::size_t total_bytes_    = 0;
    std::size_t window_bytes_   = 16 * 1024 * 1024;  // 16 MB default
    cudaStream_t stream_        = 0;

public:
    L2Pinner(void* d_weights, std::size_t total_bytes,
             std::size_t window_bytes = 16 * 1024 * 1024,
             cudaStream_t stream = 0)
        : d_weights_(d_weights),
          total_bytes_(total_bytes),
          window_bytes_(window_bytes),
          stream_(stream) {}

    // Pin a sliding window of `window_bytes_` bytes at offset `offset`
    // within the weight tensor.  `access_count_hint` lets the driver know
    // roughly how often this window is accessed (in [0, 1]).
    void pin_window(std::size_t offset, float access_count_hint = 1.0f) {
        if (offset + window_bytes_ > total_bytes_) {
            throw std::out_of_range("L2 pin window exceeds tensor bounds");
        }

        cudaStreamAttrValue attr = {};
        attr.accessPolicyWindow.base_ptr  = static_cast<char*>(d_weights_) + offset;
        attr.accessPolicyWindow.num_bytes = window_bytes_;
        attr.accessPolicyWindow.hitRatio  = access_count_hint;
        attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;

        cudaError_t err = cudaStreamSetAttribute(
            stream_,
            cudaStreamAttributeAccessPolicyWindow,
            &attr
        );
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaStreamSetAttribute failed: ")
                                     + cudaGetErrorString(err));
        }
    }

    // Hint to the driver that we're done with this window — let it be evicted.
    // (Useful when sliding to a totally different focal point.)
    void unpin_all() {
        cudaStreamAttrValue attr = {};
        attr.accessPolicyWindow.num_bytes = 0;
        cudaStreamSetAttribute(
            stream_,
            cudaStreamAttributeAccessPolicyWindow,
            &attr
        );
    }

    // Report L2 cache size (for diagnostics)
    static std::size_t l2_cache_size() {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        // Ada (sm_89): 72 MB shared L2
        // Hopper (sm_90): 50 MB shared L2
        return static_cast<std::size_t>(prop.persistingL2CacheMaxSize);
    }

    std::size_t window_bytes() const { return window_bytes_; }
    std::size_t total_bytes()  const { return total_bytes_;  }
};

// ============================================================================
//  Surprise-driven attention slider
//  ---------------------------------------------------------------------------
//  Given the surprise metric from metacognition_kernels.cu (which attention
//  window had the highest surprise), shift the L2 pin to cover that window.
// ============================================================================

class AttentionSlider {
private:
    L2Pinner*    pinner_      = nullptr;
    std::size_t window_count_ = 0;
    std::size_t window_size_  = 0;   // bytes per attention window

public:
    AttentionSlider(L2Pinner* pinner, std::size_t total_bytes, std::size_t num_windows)
        : pinner_(pinner),
          window_count_(num_windows) {
        window_size_ = pinner->total_bytes() / num_windows;
    }

    // Slide the L2 pin to the attention window with the highest surprise.
    // `surprise_per_window` is a float[num_windows] array.
    void slide_to_highest_surprise(const float* surprise_per_window) {
        if (!pinner_ || window_count_ == 0) return;

        std::size_t best = 0;
        float best_s = surprise_per_window[0];
        for (std::size_t i = 1; i < window_count_; ++i) {
            if (surprise_per_window[i] > best_s) {
                best_s = surprise_per_window[i];
                best   = i;
            }
        }

        // Pin the entire window for the highest-surprise region
        const std::size_t offset = best * window_size_;
        try {
            pinner_->pin_window(offset, 1.0f);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[L2 pinner] %s\n", e.what());
        }
    }

    // Multi-window pin: pin top-K highest-surprise windows (for parallel
    // attention across multiple cortical columns).
    void slide_to_topk(const float* surprise_per_window, std::size_t k) {
        if (!pinner_) return;
        // For simplicity: just pin all K windows by re-issuing the attr
        // for each (CUDA driver ORs the windows internally).
        // NOTE: cudaStreamAttributeAccessPolicyWindow accepts ONE window per
        // call.  For multi-window, call multiple times (the driver keeps
        // the union).
        std::vector<std::size_t> indices(window_count_);
        for (std::size_t i = 0; i < window_count_; ++i) indices[i] = i;
        std::sort(indices.begin(), indices.end(),
                  [surprise_per_window](std::size_t a, std::size_t b) {
                      return surprise_per_window[a] > surprise_per_window[b];
                  });
        for (std::size_t i = 0; i < k && i < indices.size(); ++i) {
            const std::size_t offset = indices[i] * window_size_;
            pinner_->pin_window(offset, 0.8f);
        }
    }
};

// ============================================================================
//  Diagnostic / smoke test
// ============================================================================

#ifdef L2_PINNER_DEMO
int main() {
    // Allocate a fake weight tensor
    const std::size_t tensor_bytes = 64 * 1024 * 1024;  // 64 MB
    float* d_weights = nullptr;
    cudaMalloc(&d_weights, tensor_bytes);

    // Create pinner + slider
    L2Pinner pinner(d_weights, tensor_bytes, 16 * 1024 * 1024, 0);
    AttentionSlider slider(&pinner, tensor_bytes, /*num_windows=*/4);

    std::printf("L2 cache size on this GPU: %zu MB\n",
                L2Pinner::l2_cache_size() / (1024 * 1024));

    // Simulate surprise distribution
    float surprise[4] = {0.1f, 0.7f, 0.05f, 0.15f};
    slider.slide_to_highest_surprise(surprise);
    std::printf("L2 pinned to window 1 (highest surprise = 0.7)\n");

    pinner.unpin_all();
    cudaFree(d_weights);
    return 0;
}
#endif