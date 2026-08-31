@echo off
REM ============================================================================
REM  scripts/check_gpu.bat — verify CUDA installation and GPU detection (Windows)
REM  ============================================================================
REM  Exit 0 if a usable GPU is found, exit 1 otherwise.
REM  Prints diagnostic info to stdout.
REM ============================================================================

setlocal EnableDelayedExpansion

echo ============================================================
echo   Guimlab ^— CUDA/GPU environment check
echo ============================================================

REM ------------------ nvcc available? ------------------
where nvcc >nul 2>&1
if errorlevel 1 (
    echo FATAL: nvcc not found in PATH.
    echo Install CUDA Toolkit 12.x from https://developer.nvidia.com/cuda-toolkit
    exit /b 1
)

for /f "tokens=5 delims=," %%i in ('nvcc --version ^| findstr release') do (
    set "NVCC_VER=%%i"
)
echo nvcc version:         %NVCC_VER%

REM ------------------ nvidia-smi available? ------------------
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo FATAL: nvidia-smi not found.
    echo Install NVIDIA driver matching CUDA Toolkit.
    exit /b 1
)

REM ------------------ Driver + GPU info ------------------
for /f "delims=" %%i in ('nvidia-smi --query-gpu=driver_version --format^=csv^,noheader 2^>nul') do (
    set "DRIVER_VER=%%i"
)
for /f "delims=" %%i in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do (
    set "GPU_NAME=%%i"
)
for /f "delims=" %%i in ('nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader 2^>nul') do (
    set "COMPUTE_CAP=%%i"
)

echo Driver version:       %DRIVER_VER%
echo GPU:                  %GPU_NAME%
echo Compute capability:   %COMPUTE_CAP%

REM ------------------ Compute capability check ------------------
if not "%COMPUTE_CAP%"=="" (
    for /f "tokens=1 delims=." %%i in ("%COMPUTE_CAP%") do (
        set "MAJOR=%%i"
    )
    if %MAJOR% LSS 7 (
        echo.
        echo WARN: Compute capability ^< 7.0 ^(Volta+^).
        echo       Performance may be suboptimal.
    )
)

echo.
echo [OK^] CUDA environment looks usable.
exit /b 0