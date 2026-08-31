// ============================================================================
//  symplectic_traces_kernel.cu — Adaptive Kuramoto-Symplectic Traces (AK-SRT)
//  ============================================================================
//  Author        : Guimlab Contributors
//  License       : AGPL-3.0 / MIT
// ============================================================================

#include "symplectic_traces.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstring>

namespace guim {

// ----------------------------------------------------------------------------
//  AK-SRT CUDA Kernels
// ----------------------------------------------------------------------------

__global__ void symplectic_init_kernel(
    float*       __restrict__ d_weights,
    float*       __restrict__ d_trace_real,
    float*       __restrict__ d_trace_imag,
    float*       __restrict__ d_omega,
    float*       __restrict__ d_alphas,
    float*       __restrict__ d_betas,
    float*       __restrict__ d_meta_traces,
    float*       __restrict__ d_hidden,
    std::uint32_t seed
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < SYM_TOTAL_W) {
        const float scale = sqrtf(2.0f / static_cast<float>(SYM_TOTAL_IN));
        unsigned long long s = seed ^ (static_cast<unsigned long long>(idx) * 0x9E3779B97F4A7C15ull);
        s = (s ^ (s >> 30)) * 0xBF58476D1CE4E5B9ull;
        s = (s ^ (s >> 27)) * 0x94D049BB133111EBull;
        s = s ^ (s >> 31);
        const float u1 = (static_cast<float>(s & 0xFFFFFF) / static_cast<float>(0x1000000)) - 0.5f;

        d_weights[idx]     = u1 * 2.0f * scale;
        d_trace_real[idx]  = 0.0f;
        d_trace_imag[idx]  = 0.0f;
        d_meta_traces[idx] = 0.0f;

        const float u2 = static_cast<float>((s >> 8) & 0xFFFFFF) / static_cast<float>(0x1000000);
        d_omega[idx]   = 50.0f + 300.0f * u2;
        d_betas[idx]   = -3.912f;  // exp(-3.912) = 0.020f (identical to baseline)
        d_alphas[idx]  = 0.020f;
    }

    if (idx < SYM_STATE_DIM) {
        d_hidden[idx] = 0.0f;
    }
}

__global__ void __launch_bounds__(SYM_STATE_DIM, 1)
symplectic_step_kernel(
    const float* __restrict__ d_input_curr,
    const float* __restrict__ d_input_prev,
    const float* __restrict__ d_hidden_prev,
    float*       __restrict__ d_hidden_curr,
    float*       __restrict__ d_weights,
    float*       __restrict__ d_trace_real,
    float*       __restrict__ d_trace_imag,
    float*       __restrict__ d_omega,
    float*       __restrict__ d_alphas,
    float*       __restrict__ d_betas,
    float*       __restrict__ d_meta_traces,
    float                     td_error,
    float                     dt_sec,
    float*       __restrict__ d_loss_out
) {
    const int row = threadIdx.x;
    if (row >= SYM_STATE_DIM) return;

    const int w_base = row * SYM_TOTAL_IN;

    // 1. Forward recurrent pass
    float accum = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < SYM_INPUT_DIM; ++i) {
        accum += d_weights[w_base + i] * d_input_curr[i];
    }
    #pragma unroll 8
    for (int i = 0; i < SYM_STATE_DIM; ++i) {
        accum += d_weights[w_base + SYM_INPUT_DIM + i] * d_hidden_prev[i];
    }

    const float h_new = tanhf(accum);
    const float dtanh = 1.0f - h_new * h_new;

    // 2. Symplectic Phase-Space Evolution
    constexpr float TAU = 0.03f;
    const float safe_dt = fmaxf(dt_sec, 1e-6f);
    const float decay = __expf(-safe_dt / TAU);

    #pragma unroll 8
    for (int col = 0; col < SYM_TOTAL_IN; ++col) {
        const int idx = w_base + col;

        float u_pos = 0.0f;
        float u_vel = 0.0f;

        if (col < SYM_INPUT_DIM) {
            u_pos = d_input_curr[col];
            u_vel = (d_input_curr[col] - d_input_prev[col]) / safe_dt;
        } else {
            const int h_idx = col - SYM_INPUT_DIM;
            u_pos = d_hidden_prev[h_idx];
            u_vel = (h_new - d_hidden_prev[h_idx]) / safe_dt;
        }

        float omega = d_omega[idx];
        const float theta = omega * safe_dt;
        const float cos_t = __cosf(theta);
        const float sin_t = __sinf(theta);

        const float e_old = d_trace_real[idx];
        const float p_old = d_trace_imag[idx];

        // Complex Symplectic Unitary Rotation: (E + i*P) * e^{-dt/tau - i*theta}
        const float e_rot = (cos_t * e_old - sin_t * p_old) * decay;
        const float p_rot = (sin_t * e_old + cos_t * p_old) * decay;

        const float norm_vel = u_vel / (omega + 1e-4f);
        float e_new = e_rot + dtanh * u_pos;
        float p_new = p_rot + dtanh * norm_vel;

        // Hamiltonian Energy Bounding
        constexpr float H_MAX = 50.0f;
        const float h_energy = 0.5f * (e_new * e_new + p_new * p_new);
        if (h_energy > H_MAX) {
            const float s = sqrtf(H_MAX / h_energy);
            e_new *= s;
            p_new *= s;
        }

        d_trace_real[idx] = e_new;
        d_trace_imag[idx] = p_new;

        // 3. Online Kuramoto Eigenfrequency Tuning
        constexpr float MU_OMEGA = 0.005f;
        omega += MU_OMEGA * td_error * (p_new * u_pos - e_new * norm_vel);
        if (omega < 10.0f)   omega = 10.0f;
        if (omega > 2000.0f) omega = 2000.0f;
        d_omega[idx] = omega;

        // 4. Exact Mathematical Phase-Lead Compensation: omega*tau / sqrt(1 + (omega*tau)^2)
        const float omega_tau = omega * TAU;
        const float lead_coeff = omega_tau / sqrtf(1.0f + omega_tau * omega_tau);
        const float phase_lead_trace = e_new + lead_coeff * p_new;

        // 5. TMD-ET / IDBD Synaptic Update with Controlled Meta-Step
        const float grad = td_error * phase_lead_trace;
        float alpha_ij = d_alphas[idx];
        float m_ij     = d_meta_traces[idx];

        const float damp_factor = fmaxf(1.0f - alpha_ij * phase_lead_trace * phase_lead_trace, 0.0f);
        m_ij = m_ij * damp_factor + alpha_ij * grad;
        d_meta_traces[idx] = m_ij;

        float beta_ij = d_betas[idx];
        beta_ij += 0.0001f * grad * m_ij;
        if (beta_ij < -6.0f) beta_ij = -6.0f;
        if (beta_ij > -2.5f) beta_ij = -2.5f;  // max alpha = exp(-2.5) = 0.082f
        d_betas[idx] = beta_ij;

        alpha_ij = __expf(beta_ij);
        d_alphas[idx] = alpha_ij;

        d_weights[idx] += alpha_ij * grad - 0.00005f * d_weights[idx];
    }

    d_hidden_curr[row] = h_new;

    if (row == 0) {
        *d_loss_out = fabsf(td_error);
    }
}

__global__ void standard_baseline_step_kernel(
    const float* __restrict__ d_input_curr,
    const float* __restrict__ d_hidden_prev,
    float*       __restrict__ d_hidden_curr,
    float*       __restrict__ d_weights,
    float*       __restrict__ d_trace_real,
    float                     td_error,
    float*       __restrict__ d_loss_out
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= SYM_STATE_DIM) return;

    const int w_base = row * SYM_TOTAL_IN;

    float accum = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < SYM_INPUT_DIM; ++i) {
        accum += d_weights[w_base + i] * d_input_curr[i];
    }
    #pragma unroll 8
    for (int i = 0; i < SYM_STATE_DIM; ++i) {
        accum += d_weights[w_base + SYM_INPUT_DIM + i] * d_hidden_prev[i];
    }

    const float h_new = tanhf(accum);
    const float dtanh = 1.0f - h_new * h_new;

    #pragma unroll 8
    for (int col = 0; col < SYM_TOTAL_IN; ++col) {
        const int idx = w_base + col;
        const float in_val = (col < SYM_INPUT_DIM) ? d_input_curr[col]
                                                   : d_hidden_prev[col - SYM_INPUT_DIM];

        // Standard uncompensated RC lag trace
        float e = 0.90f * d_trace_real[idx] + dtanh * in_val;
        d_trace_real[idx] = e;

        d_weights[idx] += 0.02f * td_error * e - 0.00005f * d_weights[idx];
    }

    d_hidden_curr[row] = h_new;
    if (row == 0) *d_loss_out = fabsf(td_error);
}

// ----------------------------------------------------------------------------
//  SymplecticEngine Implementation
// ----------------------------------------------------------------------------

SymplecticEngine::SymplecticEngine(std::uint32_t seed) {
    cudaStreamCreateWithFlags(&state_.stream, cudaStreamNonBlocking);

    cudaMalloc(&state_.d_weights,     SYM_TOTAL_W * sizeof(float));
    cudaMalloc(&state_.d_trace_real,  SYM_TOTAL_W * sizeof(float));
    cudaMalloc(&state_.d_trace_imag,  SYM_TOTAL_W * sizeof(float));
    cudaMalloc(&state_.d_omega,       SYM_TOTAL_W * sizeof(float));
    cudaMalloc(&state_.d_alphas,      SYM_TOTAL_W * sizeof(float));
    cudaMalloc(&state_.d_betas,       SYM_TOTAL_W * sizeof(float));
    cudaMalloc(&state_.d_meta_traces, SYM_TOTAL_W * sizeof(float));
    cudaMalloc(&state_.d_hidden_prev, SYM_STATE_DIM * sizeof(float));
    cudaMalloc(&state_.d_hidden_curr, SYM_STATE_DIM * sizeof(float));
    cudaMalloc(&state_.d_input_prev,  SYM_INPUT_DIM * sizeof(float));
    cudaMalloc(&state_.d_input_curr,  SYM_INPUT_DIM * sizeof(float));
    cudaMalloc(&state_.d_loss_out,    sizeof(float));

    constexpr int THREADS = 256;
    const int blocks = (SYM_TOTAL_W + THREADS - 1) / THREADS;
    symplectic_init_kernel<<<blocks, THREADS, 0, state_.stream>>>(
        state_.d_weights, state_.d_trace_real, state_.d_trace_imag,
        state_.d_omega, state_.d_alphas, state_.d_betas, state_.d_meta_traces,
        state_.d_hidden_prev, seed
    );
    cudaMemsetAsync(state_.d_input_prev, 0, SYM_INPUT_DIM * sizeof(float), state_.stream);
    cudaStreamSynchronize(state_.stream);
}

SymplecticEngine::~SymplecticEngine() {
    if (state_.stream) {
        cudaStreamSynchronize(state_.stream);
        cudaStreamDestroy(state_.stream);
    }
    if (state_.d_weights)     cudaFree(state_.d_weights);
    if (state_.d_trace_real)  cudaFree(state_.d_trace_real);
    if (state_.d_trace_imag)  cudaFree(state_.d_trace_imag);
    if (state_.d_omega)       cudaFree(state_.d_omega);
    if (state_.d_alphas)      cudaFree(state_.d_alphas);
    if (state_.d_betas)       cudaFree(state_.d_betas);
    if (state_.d_meta_traces) cudaFree(state_.d_meta_traces);
    if (state_.d_hidden_prev) cudaFree(state_.d_hidden_prev);
    if (state_.d_hidden_curr) cudaFree(state_.d_hidden_curr);
    if (state_.d_input_prev)  cudaFree(state_.d_input_prev);
    if (state_.d_input_curr)  cudaFree(state_.d_input_curr);
    if (state_.d_loss_out)    cudaFree(state_.d_loss_out);
}

float SymplecticEngine::step(
    std::span<const float, SYM_INPUT_DIM> input,
    float td_error,
    std::span<float, SYM_STATE_DIM> output,
    float dt_sec
) {
    cudaMemcpyAsync(state_.d_input_curr, input.data(), SYM_INPUT_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, state_.stream);

    symplectic_step_kernel<<<1, SYM_STATE_DIM, 0, state_.stream>>>(
        state_.d_input_curr, state_.d_input_prev, state_.d_hidden_prev,
        state_.d_hidden_curr, state_.d_weights, state_.d_trace_real,
        state_.d_trace_imag, state_.d_omega, state_.d_alphas,
        state_.d_betas, state_.d_meta_traces,
        td_error, dt_sec, state_.d_loss_out
    );

    cudaMemcpyAsync(output.data(), state_.d_hidden_curr, SYM_STATE_DIM * sizeof(float),
                    cudaMemcpyDeviceToHost, state_.stream);

    cudaMemcpyAsync(state_.d_input_prev, state_.d_input_curr, SYM_INPUT_DIM * sizeof(float),
                    cudaMemcpyDeviceToDevice, state_.stream);
    cudaMemcpyAsync(state_.d_hidden_prev, state_.d_hidden_curr, SYM_STATE_DIM * sizeof(float),
                    cudaMemcpyDeviceToDevice, state_.stream);

    cudaStreamSynchronize(state_.stream);
    return td_error;
}

float SymplecticEngine::step_standard_baseline(
    std::span<const float, SYM_INPUT_DIM> input,
    float td_error,
    std::span<float, SYM_STATE_DIM> output
) {
    cudaMemcpyAsync(state_.d_input_curr, input.data(), SYM_INPUT_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, state_.stream);

    standard_baseline_step_kernel<<<1, SYM_STATE_DIM, 0, state_.stream>>>(
        state_.d_input_curr, state_.d_hidden_prev, state_.d_hidden_curr,
        state_.d_weights, state_.d_trace_real, td_error, state_.d_loss_out
    );

    cudaMemcpyAsync(output.data(), state_.d_hidden_curr, SYM_STATE_DIM * sizeof(float),
                    cudaMemcpyDeviceToHost, state_.stream);
    cudaMemcpyAsync(state_.d_hidden_prev, state_.d_hidden_curr, SYM_STATE_DIM * sizeof(float),
                    cudaMemcpyDeviceToDevice, state_.stream);

    cudaStreamSynchronize(state_.stream);
    return td_error;
}

}  // namespace guim
