// =============================================================================
//  guim_cpu_ref.cpp  -  Host-only CPU reference implementation.
//  =============================================================================
//  Mirrors the CUDA algorithm in C++20 for CPU validation.
//
//  Author        : Guimlab Contributors
//  SPDX-License   : MIT
// =============================================================================

#include "guim_kernel.h"

#include <cmath>
#include <cstring>
#include <new>
#include <algorithm>

struct GuimCpuRef {
    float  weights    [GUIM_TOTAL_W]   = {0};
    float  alphas     [GUIM_TOTAL_W]   = {0};
    float  betas      [GUIM_TOTAL_W]   = {0};
    float  traces     [GUIM_TOTAL_W]   = {0};
    float  meta_traces[GUIM_TOTAL_W]   = {0};
    float  hidden_in  [GUIM_STATE_DIM] = {0};
    float  hidden_out [GUIM_STATE_DIM] = {0};
    float  utility    [GUIM_STATE_DIM] = {0};
    float  mean       [GUIM_STATE_DIM] = {0};

    std::uint32_t resets_count = 0;
    std::uint64_t frames_run   = 0;
    std::uint32_t seed         = 0;
};

extern "C" GuimCpuRef* guim_cpu_create(std::uint32_t seed)
{
    auto* r = new (std::nothrow) GuimCpuRef;
    if (!r) return nullptr;
    r->seed = seed;

    const float scale = std::sqrt(2.0f / static_cast<float>(GUIM_TOTAL_DIM));
    for (int idx = 0; idx < GUIM_TOTAL_W; ++idx) {
        unsigned long long s = seed ^ (static_cast<unsigned long long>(idx) * 0x9E3779B97F4A7C15ull);
        s = (s ^ (s >> 30)) * 0xBF58476D1CE4E5B9ull;
        s = (s ^ (s >> 27)) * 0x94D049BB133111EBull;
        s = s ^ (s >> 31);
        const float u = (static_cast<float>(s & 0xFFFFFF) / static_cast<float>(0x1000000)) - 0.5f;

        r->weights[idx]     = u * 2.0f * scale;
        r->alphas[idx]      = GUIM_ALPHA;
        r->betas[idx]       = std::log(GUIM_ALPHA);
        r->traces[idx]      = 0.0f;
        r->meta_traces[idx] = 0.0f;
    }

    for (int i = 0; i < GUIM_STATE_DIM; ++i) {
        r->hidden_in[i]  = 0.0f;
        r->hidden_out[i] = 0.0f;
        r->utility[i]    = 0.1f;
        r->mean[i]       = 0.0f;
    }

    return r;
}

extern "C" void guim_cpu_destroy(GuimCpuRef* r)
{
    delete r;
}

extern "C" float guim_cpu_step(GuimCpuRef* r, const float* input, float* output)
{
    if (!r || !input || !output) return std::nanf("");

    // Guard against NaN/Inf
    for (int i = 0; i < GUIM_INPUT_DIM; ++i) {
        if (!std::isfinite(input[i])) {
            ++r->resets_count;
            for (int o = 0; o < GUIM_STATE_DIM; ++o) output[o] = std::nanf("");
            return std::nanf("");
        }
    }

    float td_error_mag = 0.0f;

    for (int row = 0; row < GUIM_STATE_DIM; ++row) {
        const int w_base = row * GUIM_TOTAL_DIM;

        float accum = 0.0f;
        for (int i = 0; i < GUIM_INPUT_DIM; ++i) {
            accum += r->weights[w_base + i] * input[i];
        }
        for (int i = 0; i < GUIM_STATE_DIM; ++i) {
            accum += r->weights[w_base + GUIM_INPUT_DIM + i] * r->hidden_in[i];
        }

        const float h_new = std::tanh(accum);
        const float dtanh = 1.0f - h_new * h_new;

        const float td_error = input[0] - h_new;
        if (row == 0) td_error_mag = std::abs(td_error);

        // CBP tracking
        constexpr float BETA_EMA = 0.99f;
        r->mean[row] = BETA_EMA * r->mean[row] + (1.0f - BETA_EMA) * h_new;
        const float var = (h_new - r->mean[row]) * (h_new - r->mean[row]);
        r->utility[row] = BETA_EMA * r->utility[row] + (1.0f - BETA_EMA) * var;

        if (r->utility[row] < GUIM_PLASTICITY_THRESHOLD) {
            ++r->resets_count;
            r->utility[row] = 0.1f;
            for (int col = 0; col < GUIM_TOTAL_DIM; ++col) {
                const int idx = w_base + col;
                r->weights[idx]     = 0.0f;
                r->traces[idx]      = 0.0f;
                r->meta_traces[idx] = 0.0f;
                r->alphas[idx]      = GUIM_ALPHA;
                r->betas[idx]       = std::log(GUIM_ALPHA);
            }
        } else {
            for (int col = 0; col < GUIM_TOTAL_DIM; ++col) {
                const int idx = w_base + col;
                const float in_val = (col < GUIM_INPUT_DIM)
                                     ? input[col]
                                     : r->hidden_in[col - GUIM_INPUT_DIM];

                float e_ij = (GUIM_LAMBDA * r->traces[idx]) + (dtanh * in_val);
                if (std::abs(e_ij) > GUIM_TRACE_CLIP) e_ij = std::copysign(GUIM_TRACE_CLIP, e_ij);
                r->traces[idx] = e_ij;

                const float grad = td_error * e_ij;
                float alpha_ij   = r->alphas[idx];
                float m_ij       = r->meta_traces[idx];

                m_ij = m_ij * (1.0f - alpha_ij * e_ij * e_ij) + alpha_ij * grad;
                r->meta_traces[idx] = m_ij;

                float beta_ij = r->betas[idx] + 0.001f * grad * m_ij;
                beta_ij = std::clamp(beta_ij, -10.0f, 5.0f);
                r->betas[idx] = beta_ij;

                alpha_ij = std::min(std::exp(beta_ij), 0.05f);
                r->alphas[idx] = alpha_ij;

                float w = r->weights[idx] + alpha_ij * grad - GUIM_WEIGHT_DECAY * r->weights[idx];
                if (std::abs(w) > GUIM_WEIGHT_CLIP) w = std::copysign(GUIM_WEIGHT_CLIP, w);
                r->weights[idx] = w;
            }
        }

        r->hidden_out[row] = h_new;
        output[row]        = h_new;
    }

    std::memcpy(r->hidden_in, r->hidden_out, sizeof(r->hidden_in));
    ++r->frames_run;
    return td_error_mag;
}

extern "C" void guim_cpu_get_hidden(const GuimCpuRef* r, float* dst)
{
    if (!r || !dst) return;
    std::memcpy(dst, r->hidden_in, sizeof(r->hidden_in));
}

extern "C" void guim_cpu_get_weights(const GuimCpuRef* r, float* dst)
{
    if (!r || !dst) return;
    std::memcpy(dst, r->weights, sizeof(r->weights));
}

extern "C" std::uint32_t guim_cpu_resets_count(const GuimCpuRef* r)
{
    return r ? r->resets_count : 0u;
}