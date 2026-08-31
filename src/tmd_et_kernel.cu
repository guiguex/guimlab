// ============================================================================
//  tmd_et_kernel.cu — Temporal Meta-Descent with Eligibility Traces (TMD-ET)
//  ---------------------------------------------------------------------------
//  Per-synapse learning rate adaptation (IDBD) + RTRL credit assignment.
//
//  Author        : Guimlab Contributors
//  License       : AGPL-3.0
// ============================================================================

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

// ----------------------------------------------------------------------------
//  Constants
// ----------------------------------------------------------------------------

constexpr int TMD_STATE_DIM   = 256;
constexpr int TMD_INPUT_DIM   = 128;
constexpr int TMD_TOTAL_IN    = TMD_INPUT_DIM + TMD_STATE_DIM;  // 384
constexpr int TMD_TOTAL_W     = TMD_STATE_DIM * TMD_TOTAL_IN;   // 98304

__constant__ float C_GAMMA     = 0.99f;
__constant__ float C_LAMBDA    = 0.90f;
__constant__ float C_MU        = 0.001f;
__constant__ float C_ALPHA_MAX = 0.10f;
__constant__ float C_BETA_MIN  = -10.0f;
__constant__ float C_BETA_MAX  =  10.0f;

// ----------------------------------------------------------------------------
//  Fused TMD-ET Kernel (Double-buffered hidden state)
// ----------------------------------------------------------------------------

__global__ void fused_guim_meta_step(
    const float* __restrict__ d_input,       // [TMD_INPUT_DIM]
    const float* __restrict__ d_hidden_in,   // [TMD_STATE_DIM] read-only
    float*       __restrict__ d_hidden_out,  // [TMD_STATE_DIM] write-only
    float*       __restrict__ d_weights,     // [TMD_TOTAL_W]
    float*       __restrict__ d_traces,      // [TMD_TOTAL_W]
    float*       __restrict__ d_alphas,      // [TMD_TOTAL_W]
    float*       __restrict__ d_betas,       // [TMD_TOTAL_W]
    float*       __restrict__ d_meta_traces, // [TMD_TOTAL_W]
    const float              td_error
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= TMD_STATE_DIM) return;

    // Forward pass
    float accum = 0.0f;
    const int w_base = row * TMD_TOTAL_IN;

    #pragma unroll 8
    for (int i = 0; i < TMD_INPUT_DIM; ++i) {
        accum += d_weights[w_base + i] * d_input[i];
    }

    #pragma unroll 8
    for (int i = 0; i < TMD_STATE_DIM; ++i) {
        accum += d_weights[w_base + TMD_INPUT_DIM + i] * d_hidden_in[i];
    }

    const float h_new = tanhf(accum);
    const float dtanh = 1.0f - h_new * h_new;

    // Per-synapse meta-updates
    #pragma unroll 8
    for (int col = 0; col < TMD_TOTAL_IN; ++col) {
        const int idx = w_base + col;
        const float in_val = (col < TMD_INPUT_DIM) ? d_input[col]
                                                   : d_hidden_in[col - TMD_INPUT_DIM];

        // A. Eligibility trace
        float e_ij = (C_LAMBDA * C_GAMMA * d_traces[idx]) + (dtanh * in_val);
        d_traces[idx] = e_ij;

        // B. Meta-trace
        float alpha_ij = d_alphas[idx];
        float m_ij     = d_meta_traces[idx];
        const float grad = td_error * e_ij;

        m_ij = m_ij * (1.0f - alpha_ij * e_ij * e_ij) + alpha_ij * grad;
        d_meta_traces[idx] = m_ij;

        // C. Meta-gradient on beta
        float beta_ij = d_betas[idx];
        beta_ij += C_MU * grad * m_ij;

        if (beta_ij < C_BETA_MIN) beta_ij = C_BETA_MIN;
        if (beta_ij > C_BETA_MAX) beta_ij = C_BETA_MAX;
        d_betas[idx] = beta_ij;

        // D. Recompute alpha
        alpha_ij = __expf(beta_ij);
        if (alpha_ij > C_ALPHA_MAX) alpha_ij = C_ALPHA_MAX;
        d_alphas[idx] = alpha_ij;

        // E. Weight update
        d_weights[idx] += alpha_ij * grad;
    }

    d_hidden_out[row] = h_new;
}

// ----------------------------------------------------------------------------
//  Init kernel
// ----------------------------------------------------------------------------

__global__ void init_alpha_and_beta_kernel(
    float* __restrict__ d_alphas,
    float* __restrict__ d_betas,
    unsigned long long  seed
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TMD_TOTAL_W) return;

    d_betas[idx]  = -4.6f;   // exp(-4.6) ≈ 0.01
    d_alphas[idx] = 0.01f;
}