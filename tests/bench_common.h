// bench_common.h - shared utilities for the benchmark harness
//
// Tiny self-contained helpers: percentile, timer, JSON writer.  No external
// deps. Header-only; pull in from any test file.

#ifndef BENCH_COMMON_H
#define BENCH_COMMON_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace bench {

struct LatencyStats {
    double avg_us = 0.0;
    double p50_us = 0.0;
    double p99_us = 0.0;
    double p999_us = 0.0;
    double max_us = 0.0;
    double min_us = 0.0;
    double fps    = 0.0;
    uint64_t n = 0;
};

// Percentile on a *sorted* vector of doubles (microseconds).
inline LatencyStats compute_stats(std::vector<double>& samples_us) {
    LatencyStats s;
    s.n = samples_us.size();
    if (s.n == 0) return s;
    std::sort(samples_us.begin(), samples_us.end());

    double sum = 0.0;
    for (double v : samples_us) sum += v;
    s.avg_us = sum / (double)s.n;
    s.min_us = samples_us.front();
    s.max_us = samples_us.back();

    auto pct = [&](double p) {
        size_t idx = (size_t)std::floor((p / 100.0) * (s.n - 1));
        return samples_us[idx];
    };
    s.p50_us  = pct(50.0);
    s.p99_us  = pct(99.0);
    s.p999_us = pct(99.9);
    s.fps     = 1e6 / s.avg_us;
    return s;
}

inline void print_stats(const char* label, const LatencyStats& s) {
    std::printf("\n[%s]\n", label);
    std::printf("  frames    : %llu\n", (unsigned long long)s.n);
    std::printf("  avg       : %8.2f us\n", s.avg_us);
    std::printf("  p50       : %8.2f us\n", s.p50_us);
    std::printf("  p99       : %8.2f us\n", s.p99_us);
    std::printf("  p99.9     : %8.2f us\n", s.p999_us);
    std::printf("  max       : %8.2f us\n", s.max_us);
    std::printf("  min       : %8.2f us\n", s.min_us);
    std::printf("  throughput: %8.0f fps\n", s.fps);
}

// ----- Minimal JSON writer (no deps). ------------------------------------
// Produces valid JSON with deterministic key order via std::map.
struct Json {
    std::map<std::string, std::string> scalars;   // key -> json-encoded value
    std::map<std::string, Json>         objects;  // nested objects
    std::vector<Json>                   arrays;   // optional single array

    void put(const std::string& k, const std::string& v)   { scalars[k] = v; }
    void put(const std::string& k, const char* v)          { scalars[k] = std::string("\"") + v + "\""; }
    void put(const std::string& k, double v) {
        char buf[64]; std::snprintf(buf, sizeof(buf), "%.6g", v);
        scalars[k] = buf;
    }
    void put(const std::string& k, int64_t v) {
        scalars[k] = std::to_string(v);
    }
    void put(const std::string& k, bool v) {
        scalars[k] = v ? "true" : "false";
    }
    void put(const std::string& k, const Json& j) {
        objects[k] = j;
    }
    void put_array(const std::string& k, const std::vector<Json>& arr) {
        Json j;
        j.arrays = arr;
        objects[k] = j;
    }

    std::string dump(int indent = 0) const {
        std::ostringstream os;
        std::string pad(indent * 2, ' ');
        os << "{\n";
        bool first = true;
        for (auto& [k, v] : scalars) {
            if (!first) os << ",\n";
            first = false;
            os << pad << "  \"" << k << "\": " << v;
        }
        for (auto& [k, j] : objects) {
            if (!first) os << ",\n";
            first = false;
            os << pad << "  \"" << k << "\": " << j.dump(indent + 1);
        }
        if (!arrays.empty()) {
            if (!first) os << ",\n";
            first = false;
            os << pad << "  \"values\": [";
            for (size_t i = 0; i < arrays.size(); ++i) {
                if (i) os << ", ";
                os << arrays[i].dump(indent + 1);
            }
            os << "]";
        }
        os << "\n" << pad << "}";
        return os.str();
    }
};

inline bool write_file(const std::string& path, const std::string& content) {
    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << content;
    return f.good();
}

// ----- Tiny RNG (deterministic, testable) --------------------------------
struct LCG {
    uint64_t s;
    explicit LCG(uint64_t seed = 0xdeadbeefcafeULL) : s(seed) {}
    float next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        // 24-bit precision float in [-1, 1]
        uint32_t bits = (uint32_t)((s >> 40) & 0xffffffu);
        return (bits / 8388607.5f) - 1.0f;
    }
};

// ----- Synthetic sine task -----------------------------------------------
// Generate an input frame that DELAYS the sine by N frames; kernel must
// predict the current sine.  This is the standard test for RTRL/TD.
inline float sine_task(int t, int delay_frames) {
    float phase = (float)((t + delay_frames) % 360) * (3.14159265f / 180.f);
    return std::sin(phase);
}

// ----- NaN/Inf scanner ---------------------------------------------------
template <typename T>
inline bool any_nonfinite(const T* p, size_t n) {
    for (size_t i = 0; i < n; ++i) if (!std::isfinite(p[i])) return true;
    return false;
}

} // namespace bench

#endif // BENCH_COMMON_H