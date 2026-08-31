# =============================================================================
#  tests/CMakeLists.txt - Build the Guimlab test & bench suite.
#
#  Generates 6 targets:
#    - guim_kernel_tests    : gtest binary that runs ALL *_test.cpp
#    - guim_bench           : the microbench orchestrator (writes JSON when
#                             GUIM_BENCH_JSON=1 or --json <path>)
#
#  Required toolchain:
#    - nvcc  (CUDA Toolkit, >= 11.0)
#    - CMake >= 3.18
#    - GoogleTest (auto-fetched via FetchContent if not installed)
#
#  Quick build:
#    cmake -S tests -B build/tests -DCMAKE_CUDA_ARCHITECTURES=89
#    cmake --build build/tests -j
# =============================================================================

cmake_minimum_required(VERSION 3.18)
project(guim_bench CXX CUDA)

# ---- Hard requirement: nvcc available ------------------------------------
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default GPU arch; override with -DCMAKE_CUDA_ARCHITECTURES=86 (Ampere) etc.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)  # Ada Lovelace (RTX 40xx)
endif()

# Strict + fast by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)
add_compile_definitions(NDEBUG)

# ---- GoogleTest (auto-fetch) ---------------------------------------------
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
)
# Stop GoogleTest from clobbering our warning flags.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ---- Paths ---------------------------------------------------------------
set(GUIM_ROOT      "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(GUIM_INCLUDE   "${GUIM_ROOT}/include")
set(GUIM_SRC       "${GUIM_ROOT}/src")

# ---- Kernel + ref (build as a static lib so benches/tests link it) ------
add_library(guim_kernel STATIC
    "${GUIM_SRC}/guim_kernel.cu"
    "${GUIM_SRC}/guim_cpu_ref.cpp"
    "${GUIM_SRC}/guim_engine.cpp"
)
target_include_directories(guim_kernel PUBLIC "${GUIM_INCLUDE}")
target_compile_features(guim_kernel PUBLIC cxx_std_20)

# The .cu uses <span>, <curand_kernel.h> — nvcc provides both.  Add
# C++20 explicitly here as well for safety.
set_target_properties(guim_kernel PROPERTIES
    CUDA_STANDARD 20
    POSITION_INDEPENDENT_CODE ON
)

# ---- Test binary ---------------------------------------------------------
add_executable(guim_kernel_tests
    test_correctness.cpp
    test_latency.cpp
    test_convergence.cpp
    test_plasticity.cpp
    test_memory.cpp
)
target_link_libraries(guim_kernel_tests
    PRIVATE guim_kernel GTest::gtest_main
)
target_include_directories(guim_kernel_tests PRIVATE "${GUIM_INCLUDE}")

# Register tests with gtest's discovery runner.
include(GoogleTest)
gtest_discover_tests(guim_kernel_tests)

# ---- Standalone bench binary --------------------------------------------
add_executable(guim_bench bench_main.cpp)
target_link_libraries(guim_bench PRIVATE guim_kernel)
target_include_directories(guim_bench PRIVATE "${GUIM_INCLUDE}")

# ---- Convenience: build everything --------------------------------------
add_custom_target(guim_all
    DEPENDS guim_kernel_tests guim_bench
)

# ---- Install (optional, for system-wide usage) ---------------------------
install(TARGETS guim_kernel_tests guim_bench
        RUNTIME DESTINATION bin
        OPTIONAL)

message(STATUS "")
message(STATUS "  Guimlab test+bench build")
message(STATUS "  -----------------------------")
message(STATUS "    CUDA arch : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "    Build typ : ${CMAKE_BUILD_TYPE}")
message(STATUS "    Source    : ${GUIM_SRC}")
message(STATUS "    Headers   : ${GUIM_INCLUDE}")
message(STATUS "")