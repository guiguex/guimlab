// Runner that supplies gtest_failures and main()
#include "gtest/gtest.h"
#include <cstdio>

int gtest_failures = 0;

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    testing::InitGoogleTest(&argc, argv);
    int rc = RunAllTests();
    std::printf("\n========================================\n");
    std::printf("Total failures: %d\n", rc);
    std::printf("========================================\n");
    return rc == 0 ? 0 : 1;
}
