@echo off
setlocal EnableExtensions

REM ============================================
REM Ultra Fast Image Gen Launcher (Windows)
REM ============================================
REM Version: 4.0 - Quick launcher (no verification)
REM Run Verify.bat first to check your setup
REM ============================================

echo.
echo ============================================
echo        Ultra Fast Image Gen (Windows)
echo ============================================
echo.

REM Change to script directory for consistent path handling
cd /d "%~dp0" 2>nul
if errorlevel 1 (
    echo ERROR: Failed to change to script directory.
    goto :error_exit
)

set "PYTHON_EXE=%CD%\venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo ERROR: Virtual environment Python was not found at:
    echo   %PYTHON_EXE%
    echo Run Install.bat to create the environment, then Verify.bat if startup still fails.
    goto :error_exit
)

REM ============================================
REM ENVIRONMENT CONFIGURATION
REM ============================================

echo.
echo [1/2] Configuring fast launch environment...

REM Create cache directories for persistent torch.compile caching
if not exist "%CD%\cache\torch_inductor" mkdir "%CD%\cache\torch_inductor"
if not exist "%CD%\cache\triton" mkdir "%CD%\cache\triton"

REM PyTorch optimizations
set "TORCHINDUCTOR_MAX_AUTOTUNE=1"
set "TORCHINDUCTOR_FREEZING=1"
set "TORCHINDUCTOR_CACHE_DIR=%CD%\cache\torch_inductor"
set "TORCHINDUCTOR_FX_GRAPH_CACHE=1"
set "TORCHINDUCTOR_AUTOTUNE_CACHE=1"
set "TORCHINDUCTOR_FORCE_DISABLE_CACHES=0"
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,roundup_power2_divisions:32"
set "PYTORCH_ALLOC_CONF=%PYTORCH_CUDA_ALLOC_CONF%"
if "%UFIG_OPTIMIZATION_PROFILE%"=="" set "UFIG_OPTIMIZATION_PROFILE=max_speed"
if "%UFIG_ENABLE_OPTIONAL_ACCELERATORS%"=="" set "UFIG_ENABLE_OPTIONAL_ACCELERATORS=0"
REM SDNQ Triton optimizations now auto-detected via triton-windows package
REM To force-disable: set "SDNQ_USE_TORCH_COMPILE=0" and set "SDNQ_USE_TRITON_MM=0"

REM HuggingFace optimizations
set "HF_HUB_DISABLE_PROGRESS_BARS=0"
set "HF_HUB_ENABLE_HF_TRANSFER=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

REM Fast-launch flags: trust verified installs and defer heavyweight startup checks.
set "SKIP_DEPENDENCY_CHECK=1"
set "UFIG_STARTUP_MODEL_PREFLIGHT=0"
set "UFIG_OPEN_BROWSER=1"
set "UFIG_GRADIO_PORT=7860"

echo Environment configuration complete.

REM ============================================
REM LAUNCH APPLICATION
REM ============================================

echo.
echo [2/2] Starting Ultra Fast Image Gen...

echo.
echo ============================================
echo Starting Gradio UI...
echo Local URL: http://localhost:%UFIG_GRADIO_PORT%
echo ^(Press Ctrl+C to stop the server^)
echo ============================================
echo.

for /f "tokens=5" %%P in ('netstat -ano -p tcp ^| findstr /R /C:":%UFIG_GRADIO_PORT% .*LISTENING"') do (
    if not "%%P"=="" taskkill /F /PID %%P >nul 2>&1
)

REM Start the application
echo Starting Python application...
"%PYTHON_EXE%" app.py
set APP_EXIT_CODE=%errorlevel%

if %APP_EXIT_CODE% neq 0 (
    echo.
    echo ERROR: Application exited with code %APP_EXIT_CODE%
    echo Check the output above for details. Run Verify.bat if this keeps failing.
    goto :error_exit
)

echo.
echo Application exited normally.
goto :success_exit

REM ============================================
REM ERROR HANDLING
REM ============================================

:error_exit
echo.
echo ============================================
echo              LAUNCH FAILED
echo ============================================
echo Please run Verify.bat to check your setup.
echo.
pause
endlocal
exit /b 1

REM ============================================
REM SUCCESS HANDLING
REM ============================================

:success_exit
endlocal
exit /b 0
