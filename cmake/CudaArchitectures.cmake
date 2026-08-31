# ============================================================================
#  cmake/CudaArchitectures.cmake — Architecture configuration for CUDA kernels
# ============================================================================

function(guim_setup_cuda_architectures)
    if(NOT DEFINED GUIM_CUDA_ARCH)
        # Default target: RTX 3090 / Ampere (sm_86) + forward compatibility
        set(GUIM_CUDA_ARCH "86" CACHE STRING "Target CUDA architecture capabilities" FORCE)
    endif()

    set(CMAKE_CUDA_ARCHITECTURES "${GUIM_CUDA_ARCH}" PARENT_SCOPE)
    message(STATUS "[CUDA] Targeted Compute Architectures: ${GUIM_CUDA_ARCH}")
endfunction()
