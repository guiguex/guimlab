# ============================================================================
#  CMakeLists.txt — Build configuration for Guimlab substrate
# ============================================================================
cmake_minimum_required(VERSION 3.22)
project(guimlab
    VERSION     0.2.0
    DESCRIPTION "Real-time continual learning Voice-to-Voice neuromorphic substrate"
    LANGUAGES   CXX CUDA
)

# ----------------------------------------------------------------------------
#  Standards & Options
# ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD          20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD         20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Default to RTX 3090 (sm_86) + Volta, Ampere, Ada Lovelace, Hopper
if(NOT DEFINED GUIM_CUDA_ARCH)
    set(GUIM_CUDA_ARCH "86" CACHE STRING "CUDA compute capabilities")
endif()
set(CMAKE_CUDA_ARCHITECTURES "${GUIM_CUDA_ARCH}")

# ----------------------------------------------------------------------------
#  Modular Helpers & Options
# ----------------------------------------------------------------------------
include(cmake/CudaArchitectures.cmake)
include(cmake/Sanitizers.cmake)
include(cmake/Warnings.cmake)

guim_setup_cuda_architectures()

# ----------------------------------------------------------------------------
#  Flags
# ----------------------------------------------------------------------------
if(MSVC)
    set(GUIM_CXX_FLAGS /O2 /W4 /EHsc)
    set(GUIM_CUDA_FLAGS -O3 --use_fast_math -Xptxas -v)
else()
    set(GUIM_CXX_FLAGS -O3 -Wall -Wextra -Wno-unused-parameter -mavx2)
    set(GUIM_CUDA_FLAGS -O3 --use_fast_math -Xptxas -v)
endif()

# ----------------------------------------------------------------------------
#  Core Library Target: guim
# ----------------------------------------------------------------------------
set(GUIM_SOURCES
    src/guim_kernel.cu
    src/guim_cpu_ref.cpp
    src/guim_engine.cpp
    src/episodic_vsa_kernel.cu
    src/global_workspace_kernel.cu
    src/world_model_kernel.cu
    src/symplectic_traces_kernel.cu
    src/guim_lovelace_kernel.cu
)

set(GUIM_HEADERS
    include/guim.h
    include/guim_kernel.h
    include/ipc_structs.h
    include/ipc_structs_v2.h
    include/ipc_structs_sparse.h
    include/episodic_vsa.h
    include/global_workspace.h
    include/world_model.h
    include/symplectic_traces.h
    include/lovelace_cortex.h
)

add_library(guim STATIC ${GUIM_SOURCES} ${GUIM_HEADERS})
target_include_directories(guim PUBLIC include)
target_compile_options(guim PRIVATE ${GUIM_CXX_FLAGS})
target_compile_options(guim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${GUIM_CUDA_FLAGS}>)
set_target_properties(guim PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ----------------------------------------------------------------------------
#  Benchmark Executable: guim_bench
# ----------------------------------------------------------------------------
add_executable(guim_bench tests/bench_main.cpp)
target_link_libraries(guim_bench PRIVATE guim)
target_include_directories(guim_bench PRIVATE include)
target_compile_options(guim_bench PRIVATE ${GUIM_CXX_FLAGS})
set_target_properties(guim_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ----------------------------------------------------------------------------
#  Real-World Audio Streaming Demo: stream_wav_demo
# ----------------------------------------------------------------------------
add_executable(stream_wav_demo examples/stream_wav_demo.cpp)
target_link_libraries(stream_wav_demo PRIVATE guim)
target_include_directories(stream_wav_demo PRIVATE include)
target_compile_options(stream_wav_demo PRIVATE ${GUIM_CXX_FLAGS})
if(UNIX)
    target_link_libraries(stream_wav_demo PRIVATE rt pthread)
endif()
set_target_properties(stream_wav_demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ----------------------------------------------------------------------------
#  Next-Gen Pure C++20 Live Presentation Dashboard: guim_dashboard
# ----------------------------------------------------------------------------
add_executable(guim_dashboard src/guim_dashboard.cpp)
target_link_libraries(guim_dashboard PRIVATE guim)
target_include_directories(guim_dashboard PRIVATE include)
target_compile_options(guim_dashboard PRIVATE ${GUIM_CXX_FLAGS})
if(UNIX)
    target_link_libraries(guim_dashboard PRIVATE rt pthread)
endif()
set_target_properties(guim_dashboard PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ----------------------------------------------------------------------------
#  GuimLab Studio — AAA High-Density Neuromorphic Cockpit (Sokol / ImGui / ImPlot)
# ----------------------------------------------------------------------------
add_executable(guim_studio
    src/studio/main_studio.cpp
    src/studio/shm_client.cpp
    src/studio/viewport_session.cpp
    src/studio/panels/cortex_thermal_map.cpp
    src/studio/panels/symplectic_phase_portrait.cpp
    src/studio/panels/audio_oscilloscope.cpp
    src/studio/panels/metagradients_inspector.cpp
    src/studio/panels/telemetry_cockpit.cpp
    src/studio/third_party/imgui/imgui.cpp
    src/studio/third_party/imgui/imgui_draw.cpp
    src/studio/third_party/imgui/imgui_tables.cpp
    src/studio/third_party/imgui/imgui_widgets.cpp
    src/studio/third_party/implot/implot.cpp
    src/studio/third_party/implot/implot_items.cpp
)

target_include_directories(guim_studio PRIVATE
    include
    include/sokol
    src/studio
    src/studio/third_party/imgui
    src/studio/third_party/implot
)

target_compile_options(guim_studio PRIVATE ${GUIM_CXX_FLAGS})
target_link_libraries(guim_studio PRIVATE guim)
if(UNIX)
    target_link_libraries(guim_studio PRIVATE GL X11 Xi Xcursor dl m rt pthread)
elseif(WIN32)
    target_link_libraries(guim_studio PRIVATE d3d11 dxgi)
endif()
set_target_properties(guim_studio PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ----------------------------------------------------------------------------
#  GoogleTest Suite: guim_tests
# ----------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(guim_tests
    tests/test_correctness.cpp
    tests/test_latency.cpp
    tests/test_plasticity.cpp
    tests/test_convergence.cpp
    tests/test_memory.cpp
    tests/test_episodic_vsa.cpp
    tests/test_global_workspace.cpp
    tests/test_world_model.cpp
    tests/test_symplectic_discovery.cpp
    tests/test_tensor_cores_cortex.cpp
    tests/test_studio_harness.cpp
    tests/test_harness_integrity.cpp
    tests/test_physics_constants.cpp
    tests/test_scientific_hypotheses.cpp
    tests/test_sarts_ring_buffer.cpp
    src/studio/shm_client.cpp
    src/studio/viewport_session.cpp
)

target_link_libraries(guim_tests PRIVATE guim gtest_main)
target_include_directories(guim_tests PRIVATE include tests src/studio)
target_compile_options(guim_tests PRIVATE ${GUIM_CXX_FLAGS})
if(UNIX)
    target_link_libraries(guim_tests PRIVATE rt pthread)
endif()
set_target_properties(guim_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include(GoogleTest)
gtest_discover_tests(guim_tests)

# ----------------------------------------------------------------------------
#  Server & Client Nodes & Visualizers (Linux / POSIX)
# ----------------------------------------------------------------------------
if(UNIX)
    # v1 Persistent Server: guim_node
    set_source_files_properties(src/main_guim_node.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(guim_node
        src/main_guim_node.cpp
        src/persistent_kernel.cu
    )
    target_include_directories(guim_node PRIVATE include)
    target_compile_options(guim_node PRIVATE ${GUIM_CXX_FLAGS})
    target_compile_options(guim_node PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${GUIM_CUDA_FLAGS}>)
    target_link_libraries(guim_node PRIVATE pthread rt curand)
    set_target_properties(guim_node PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    # v3 Two-Speed Server: guim_node_v3
    set_source_files_properties(src/main_guim_node_v3.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(guim_node_v3
        src/main_guim_node_v3.cpp
        src/kiss_reflex_kernel.cu
        src/guim_lovelace_kernel.cu
        src/plasticity_sweeper.cu
        src/motor_babbling_kernel.cu
    )
    target_include_directories(guim_node_v3 PRIVATE include)
    target_compile_options(guim_node_v3 PRIVATE ${GUIM_CXX_FLAGS})
    target_compile_options(guim_node_v3 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${GUIM_CUDA_FLAGS}>)
    target_link_libraries(guim_node_v3 PRIVATE pthread rt curand)
    set_target_properties(guim_node_v3 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    # Delta Filter Executable: guim_delta
    add_executable(guim_delta src/delta_filter.cpp)
    target_include_directories(guim_delta PRIVATE include)
    target_compile_definitions(guim_delta PRIVATE GUIM_DELTA_STANDALONE)
    target_compile_options(guim_delta PRIVATE ${GUIM_CXX_FLAGS})
    target_link_libraries(guim_delta PRIVATE rt)
    set_target_properties(guim_delta PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    # Client Benchmark Executable: guim_client
    add_executable(guim_client src/client_benchmark.cpp)
    target_include_directories(guim_client PRIVATE include)
    target_compile_options(guim_client PRIVATE ${GUIM_CXX_FLAGS})
    target_link_libraries(guim_client PRIVATE rt)
    set_target_properties(guim_client PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    # TUI Visualizer: guim_monitor
    add_executable(guim_monitor src/guim_monitor.cpp)
    target_include_directories(guim_monitor PRIVATE include)
    target_compile_options(guim_monitor PRIVATE ${GUIM_CXX_FLAGS})
    target_link_libraries(guim_monitor PRIVATE rt pthread)
    set_target_properties(guim_monitor PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

# Cross-platform TUI Visualizer: guim_monitor (Win32 + POSIX)
if(NOT UNIX)
    add_executable(guim_monitor src/guim_monitor.cpp)
    target_include_directories(guim_monitor PRIVATE include)
    target_compile_options(guim_monitor PRIVATE ${GUIM_CXX_FLAGS})
    set_target_properties(guim_monitor PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

# ----------------------------------------------------------------------------
#  Summary
# ----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "Guimlab build configured:")
message(STATUS "  Build Type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Arch:         ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  C++ Standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Standard:     ${CMAKE_CUDA_STANDARD}")
message(STATUS "  Source Dir:        ${CMAKE_SOURCE_DIR}")
message(STATUS "  Binary Dir:        ${CMAKE_BINARY_DIR}")
message(STATUS "")