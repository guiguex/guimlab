@echo off
REM ============================================================================
REM  scripts/build.bat — configure + build + benchmark Guimlab (Windows)
REM  ============================================================================
REM  Usage: scripts\build.bat [Release^|Debug^|RelWithDebInfo]
REM ============================================================================

setlocal EnableDelayedExpansion

if "%1"=="" set "BUILD_TYPE=Release"
if not "%1"=="" set "BUILD_TYPE=%1"

set "BUILD_DIR=build"

echo ============================================================
echo   Guimlab ^— build (%BUILD_TYPE%^)
echo ============================================================

REM Step 1: check GPU
call "%~dp0check_gpu.bat"
if errorlevel 1 exit /b 1

REM Step 2: configure CMake
echo.
echo [1/3^] Configuring CMake...
cmake -B "%BUILD_DIR%" -S . -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
if errorlevel 1 exit /b 1

REM Step 3: build
echo.
echo [2/3^] Building...
cmake --build "%BUILD_DIR%" --parallel
if errorlevel 1 exit /b 1

REM Step 4: run benchmark (Release only)
if /i "%BUILD_TYPE%"=="Release" (
    echo.
    echo [3/3^] Running benchmark...
    if exist "%BUILD_DIR%\bin\guim_bench.exe" (
        "%BUILD_DIR%\bin\guim_bench.exe"
    ) else (
        echo WARN: guim_bench.exe not found.
    )
)

echo.
echo [OK^] Build complete. Output in %BUILD_DIR%\bin\
endlocal