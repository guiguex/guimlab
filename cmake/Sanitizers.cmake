# ============================================================================
#  cmake/Sanitizers.cmake — Sanitizer configuration for host C++ code
# ============================================================================

function(guim_enable_sanitizers TARGET_NAME)
    option(GUIM_ENABLE_ASAN "Enable AddressSanitizer & UndefinedBehaviorSanitizer" OFF)

    if(GUIM_ENABLE_ASAN)
        if(MSVC)
            target_compile_options(${TARGET_NAME} PRIVATE /fsanitize=address)
        else()
            target_compile_options(${TARGET_NAME} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(${TARGET_NAME} PRIVATE -fsanitize=address,undefined)
        endif()
        message(STATUS "[Sanitizers] ASan & UBSan enabled on target ${TARGET_NAME}")
    endif()
endfunction()
