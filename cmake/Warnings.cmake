# ============================================================================
#  cmake/Warnings.cmake — Strict compiler warning configuration
# ============================================================================

function(guim_set_strict_warnings TARGET_NAME)
    if(MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${TARGET_NAME} PRIVATE 
            -Wall 
            -Wextra 
            -Wpedantic 
            -Wno-unused-parameter
            -Wconversion
            -Wsign-conversion
            -Wshadow
        )
    endif()
endfunction()
