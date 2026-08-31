// Minimal gtest stub sufficient to compile test_scientific_hypotheses.cpp
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace testing { inline int InitGoogleTest(int*, char**) { return 0; } }

extern int gtest_failures;

#define TEST(suite, name) \
    static void run_test_##suite##_##name(); \
    struct Reg_##suite##_##name { \
        Reg_##suite##_##name() { \
            std::printf("[ RUN      ] %s.%s\n", #suite, #name); \
            int before = gtest_failures; \
            run_test_##suite##_##name(); \
            if (gtest_failures == before) std::printf("[       OK ] %s.%s\n", #suite, #name); \
            else std::printf("[  FAILED  ] %s.%s  (%d failures)\n", #suite, #name, gtest_failures - before); \
        } \
    } reg_instance_##suite##_##name; \
    static void run_test_##suite##_##name()

#define EXPECT_GT(a, b) do { \
    auto _va = (a); auto _vb = (b); \
    if (!(_va > _vb)) { \
        std::fprintf(stderr, "  FAIL: EXPECT_GT(%s > %s) at line %d  (got %g vs %g)\n", #a, #b, __LINE__, (double)_va, (double)_vb); \
        gtest_failures++; \
    } \
} while(0)

#define EXPECT_LT(a, b) do { \
    auto _va = (a); auto _vb = (b); \
    if (!(_va < _vb)) { \
        std::fprintf(stderr, "  FAIL: EXPECT_LT(%s < %s) at line %d  (got %g vs %g)\n", #a, #b, __LINE__, (double)_va, (double)_vb); \
        gtest_failures++; \
    } \
} while(0)

#define EXPECT_NEAR(a, b, eps) do { \
    auto _va = (a); auto _vb = (b); auto _ve = (eps); \
    if (std::abs(_va - _vb) > _ve) { \
        std::fprintf(stderr, "  FAIL: EXPECT_NEAR(%s ~= %s, eps=%g) at line %d (got diff %g)\n", #a, #b, (double)_ve, __LINE__, (double)std::abs(_va - _vb)); \
        gtest_failures++; \
    } \
} while(0)

#define ASSERT_FALSE(x) do { \
    if ((x)) { \
        std::fprintf(stderr, "  FAIL: ASSERT_FALSE(%s) at line %d\n", #x, __LINE__); \
        gtest_failures++; \
    } \
} while(0)

#define ASSERT_TRUE(x) do { \
    if (!(x)) { \
        std::fprintf(stderr, "  FAIL: ASSERT_TRUE(%s) at line %d\n", #x, __LINE__); \
        gtest_failures++; \
    } \
} while(0)

inline int RunAllTests() {
    // All tests already ran via static initializers
    return gtest_failures;
}
