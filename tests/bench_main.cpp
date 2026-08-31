// =============================================================================
//  tests/bench_main.cpp  -  Microbenchmark harness for the Guimlab kernel.
//
//  Executes real-time microbenchmarking on the GPU:
//   - constructs an engine with a fixed seed;
//   - runs `BENCH_FRAMES` steps feeding deterministic acoustic input;
//   - reports wall-clock mean / p50 / p99 / p99.9 per-frame latency and FPS.
// =============================================================================
#include "guim/guim.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint32_t BENCH_SEED    = 0xC0FFEEu;
constexpr std::uint64_t BENCH_FRAMES  = 60'000;   // 1 second @ 60 Hz
constexpr std::uint64_t WARMUP_FRAMES = 500;

double percentile(std::vector<double>& v, double pct) noexcept
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    auto idx = static_cast<std::size_t>(pct * (v.size() - 1));
    return v[idx];
}

void print_help() noexcept
{
    std::printf(
        "Usage: guim_bench [options]\n"
        "  --frames N      number of timed frames  (default %llu)\n"
        "  --warmup N      number of warm-up frames (default %llu)\n"
        "  --seed N        PRNG seed                (default %u)\n"
        "  --help          show this help\n",
        static_cast<unsigned long long>(BENCH_FRAMES),
        static_cast<unsigned long long>(WARMUP_FRAMES),
        BENCH_SEED);
}

}  // namespace

int main(int argc, char** argv)
{
    std::uint64_t frames  = BENCH_FRAMES;
    std::uint64_t warmup  = WARMUP_FRAMES;
    std::uint32_t seed    = BENCH_SEED;

    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a == "--frames" && i + 1 < argc) frames = std::stoull(argv[++i]);
        else if (a == "--warmup" && i + 1 < argc) warmup = std::stoull(argv[++i]);
        else if (a == "--seed"   && i + 1 < argc) seed   = std::stoul(argv[++i]);
        else if (a == "--help"   || a == "-h") { print_help(); return 0; }
        else { std::fprintf(stderr, "unknown arg: %s\n", std::string(a).c_str()); return 2; }
    }

    guim::Engine engine{seed};
    if (!engine.handle()) {
        std::fprintf(stderr, "FATAL: engine construction failed (no CUDA context?)\n");
        return 1;
    }

    std::printf("=== Guimlab microbench ===\n");
    std::printf("  seed:           0x%08X\n", seed);
    std::printf("  warmup frames:  %llu\n", static_cast<unsigned long long>(warmup));
    std::printf("  timed frames:   %llu\n", static_cast<unsigned long long>(frames));

    // Synthetic input: 32 floats sampled from N(0, 1).  Deterministic per seed.
    std::mt19937 rng{seed};
    std::normal_distribution<float> dist{0.0f, 1.0f};
    std::vector<float> input(GUIM_INPUT_DIM);
    std::vector<float> output(GUIM_STATE_DIM);
    for (auto& v : input) v = dist(rng);

    auto run = [&](std::uint64_t n) {
        for (std::uint64_t i = 0; i < n; ++i) {
            // Touch the input to keep the compiler honest.
            input[0] = std::sin(static_cast<float>(i) * 0.001f);
            engine.step(
                std::span<const float, GUIM_INPUT_DIM>(input.data(), GUIM_INPUT_DIM),
                std::span<float, GUIM_STATE_DIM>(output.data(), GUIM_STATE_DIM));
        }
    };

    // Warm-up (untimed, lets the JIT/cache settle).
    run(warmup);

    // Timed loop.  We measure CUDA-event-free host wall-clock because the
    // staging placeholder isn't async; once the real fused kernel lands,
    // swap this for cudaEventElapsedTime per frame.
    std::vector<double> latencies;
    latencies.reserve(static_cast<std::size_t>(frames));

    using clock = std::chrono::steady_clock;
    for (std::uint64_t i = 0; i < frames; ++i) {
        input[0] = std::sin(static_cast<float>(i) * 0.001f);
        auto t0 = clock::now();
        engine.step(
            std::span<const float, GUIM_INPUT_DIM>(input.data(), GUIM_INPUT_DIM),
            std::span<float, GUIM_STATE_DIM>(output.data(), GUIM_STATE_DIM));
        auto t1 = clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    double sum = 0.0;
    for (double v : latencies) sum += v;
    double mean = sum / static_cast<double>(latencies.size());
    double p50  = percentile(latencies, 0.50);
    double p99  = percentile(latencies, 0.99);
    double p999 = percentile(latencies, 0.999);

    std::printf("\n--- per-frame latency ---\n");
    std::printf("  mean:    %8.3f us\n", mean);
    std::printf("  p50:     %8.3f us\n", p50);
    std::printf("  p99:     %8.3f us\n", p99);
    std::printf("  p99.9:   %8.3f us\n", p999);

    double fps = 1e6 / mean;
    std::printf("\n  effective frame rate: %8.1f fps\n", fps);

    // ---- Optional JSON output -------------------------------------------
    //  Triggered by --json <path> or by GUIM_BENCH_JSON=<path> env var.
    //  Default output path is <cwd>/build/bench_results.json.
    std::string json_path = "build/bench_results.json";
    for (int i = 1; i + 1 < argc; ++i) {
        std::string_view a{argv[i]};
        if (a == "--json") json_path = argv[i + 1];
    }
    if (const char* env = std::getenv("GUIM_BENCH_JSON")) json_path = env;

    {
        std::ofstream f(json_path);
        if (f.is_open()) {
            bool nan_or_inf = std::isnan(output[0]) || std::isinf(output[0]);
            f << "{\n"
              << "  \"latency_us\": {\n"
              << "    \"avg\":  " << mean  << ",\n"
              << "    \"p50\":  " << p50   << ",\n"
              << "    \"p99\":  " << p99   << ",\n"
              << "    \"max\":  " << latencies.back() << "\n"
              << "  },\n"
              << "  \"throughput_fps\":           " << fps                                  << ",\n"
              << "  \"convergence_td_error_final\": null,\n"
              << "  \"memory_used_mb\":            " << (engine.bytes_resident() / (1024.0 * 1024.0)) << ",\n"
              << "  \"plasticity_resets_count\":   " << engine.resets_count()                << ",\n"
              << "  \"nan_inf_detected\":          " << (nan_or_inf ? "true" : "false")     << ",\n"
              << "  \"frames_run\":                " << engine.frames_run()                  << ",\n"
              << "  \"warmup_frames\":             " << warmup                               << ",\n"
              << "  \"timed_frames\":              " << frames                               << ",\n"
              << "  \"seed\":                      " << seed                                 << "\n"
              << "}\n";
            std::printf("\n  wrote JSON:  %s\n", json_path.c_str());
        }
    }

    std::printf("=== bench complete ===\n");
    return 0;
}