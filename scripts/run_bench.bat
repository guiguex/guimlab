@echo off
REM ============================================================================
REM  Guimlab  -  run benchmark executable (Windows)
REM  Builds if needed, then runs guim_bench.
REM ============================================================================
setlocal EnableDelayedExpansion
set "SCRIPT_DIR=%~dp0"
set "APP_DIR=%SCRIPT_DIR%.."
for %%I in ("%APP_DIR%") do set "APP_DIR=%%~fI"
set "BUILD_DIR=%APP_DIR%\build"
set "EXE=%BUILD_DIR%\Release\guim_bench.exe"

if not exist "%EXE%" (
    if exist "%BUILD_DIR%\guim_bench.exe" set "EXE=%BUILD_DIR%\guim_bench.exe"
)

if not exist "%EXE%" (
    echo guim_bench not found - invoking build.bat first
    call "%SCRIPT_DIR%build.bat" %*
    if exist "%BUILD_DIR%\Release\guim_bench.exe" set "EXE=%BUILD_DIR%\Release\guim_bench.exe"
    if exist "%BUILD_DIR%\guim_bench.exe"       set "EXE=%BUILD_DIR%\guim_bench.exe"
)

echo ==^> running %EXE%
echo.
"%EXE%" %*
endlocal