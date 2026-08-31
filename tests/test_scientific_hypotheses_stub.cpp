// Standalone runner for test_scientific_hypotheses.cpp — stubs gtest so we can
// run the actual test code without the full GoogleTest infrastructure.
#include <cstdio>
#include <cstdlib>
#include <cmath>

static int g_failures = 0;

#define EXPECT_GT(a, b) do { \
    auto _va = (a); auto _vb = (b); \
    if (!(_va > _vb)) { std::fprintf(stderr, "FAIL: EXPECT_GT(%s, %s) at %s:%d  (got %g vs %g)\n", #a, #b, __FILE__, __LINE__, (double)_va, (double)_vb); g_failures++; } \
} while(0)

#define EXPECT_LT(a, b) do { \
    auto _va = (a); auto _vb = (b); \
    if (!(_va < _vb)) { std::fprintf(stderr, "FAIL: EXPECT_LT(%s, %s) at %s:%d  (got %g vs %g)\n", #a, #b, __FILE__, __LINE__, (double)_va, (double)_vb); g_failures++; } \
} while(0)

#define ASSERT_FALSE(x) do { \
    if ((x)) { std::fprintf(stderr, "FAIL: ASSERT_FALSE(%s) at %s:%d\n", #x, __FILE__, __LINE__); g_failures++; } \
} while(0)

#define TEST(suite, name) \
    static void run_##suite##_##name(); \
    struct Reg_##suite##_##name { Reg_##suite##_##name() { \
        std::printf("RUN  %s.%s\n", #suite, #name); \
        int before = g_failures; \
        run_##suite##_##name(); \
        if (g_failures == before) std::printf("PASS %s.%s\n", #suite, #name); \
        else std::printf("FAIL %s.%s  (%d failures)\n", #suite, #name, g_failures-before); \
    } } reg_##suite##_##name; \
    static void run_##suite##_##name()

// Pull in the real test file
#include "test_scientific_hypotheses.cpp"

int main() {
    // Static initializers in the TEST macros already ran
    std::printf("\n=== Summary: %d total failures ===\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
