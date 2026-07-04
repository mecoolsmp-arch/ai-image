@echo off
setlocal EnableExtensions

echo ============================================
echo        Ultra Fast Image Gen (Windows)
echo ============================================
echo.

set "PY_CMD=python"
set "PY_ARGS="
set "FORCE_REPAIR=0"
set "INSTALL_PROFILE=cu130-windows-v5"
set "CUDA_INDEX_URL=https://download.pytorch.org/whl/cu130"
set "REQ_FILE=requirements-lock-cu130.txt"
set "ENABLE_OPTIONAL_ACCELERATORS=0"

for %%A in (%*) do (
  if /I "%%~A"=="--repair" set "FORCE_REPAIR=1"
  if /I "%%~A"=="--with-optional-accelerators" set "ENABLE_OPTIONAL_ACCELERATORS=1"
)

py -0p >nul 2>&1
if not errorlevel 1 set "PY_CMD=py" & set "PY_ARGS=-3"

where %PY_CMD% >nul 2>&1
if errorlevel 1 goto no_python

echo.
echo Checking CUDA toolkit...
where nvcc >nul 2>&1
if errorlevel 1 (
  echo WARNING: CUDA toolkit not detected via nvcc.
  echo Target profile is CUDA 13. Continue only if your NVIDIA driver supports CUDA 13 runtime.
) else (
  for /f "tokens=5" %%v in ('nvcc --version ^| findstr /i "release"') do set CUDA_VERSION=%%v
  echo Detected CUDA version: %CUDA_VERSION%
  echo %CUDA_VERSION% | findstr /r "^13\." >nul
  if errorlevel 1 (
    echo WARNING: CUDA toolkit is not 13.x. This installer targets CUDA 13 wheels.
  ) else (
    echo CUDA 13 detected - target match.
  )
)

echo.
echo Checking git-lfs...
where git-lfs >nul 2>&1
if errorlevel 1 (
  echo WARNING: git-lfs not installed. Some models may require manual download.
) else (
  git lfs install >nul 2>&1
  echo git-lfs initialized.
)

echo Using: %PY_CMD% %PY_ARGS%
%PY_CMD% %PY_ARGS% -V

if not exist venv (
  echo Creating virtual environment...
  %PY_CMD% %PY_ARGS% -m venv venv
)

call venv\Scripts\activate.bat
if errorlevel 1 goto no_venv
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

if not exist "%REQ_FILE%" (
  set "REQ_FILE=requirements.txt"
)

set "INSTALL_STAMP=venv\.installed_profile"
set "CURRENT_PROFILE="
if exist "%INSTALL_STAMP%" set /p CURRENT_PROFILE=<"%INSTALL_STAMP%"

if "%FORCE_REPAIR%"=="1" goto install_all
if not exist venv\.installed goto install_all
if /I not "%CURRENT_PROFILE%"=="%INSTALL_PROFILE%" goto install_all

echo Dependencies already installed for profile %INSTALL_PROFILE%.
echo Refreshing Python package set from %REQ_FILE%...
python -m pip install -r %REQ_FILE%
goto verify

:install_all
echo Installing dependencies for profile %INSTALL_PROFILE%...
python -m pip install --upgrade pip setuptools wheel

REM Only remove packages that conflict with CUDA 13 install BEFORE installing.
REM Everything else gets overwritten naturally by pip upgrade/install.
python -m pip uninstall -y xformers flash-attn 2>nul

echo Installing CUDA 13 PyTorch stack (2.10.0)...
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url %CUDA_INDEX_URL%
if errorlevel 1 goto install_fail

echo Installing Triton for Windows (enables torch.compile + SDNQ optimizations)...
REM Triton 3.6.x works with PyTorch 2.10.x. Triton 3.7+ requires PyTorch 2.11+.
python -m pip install -U "triton-windows>=3.6,<3.7"
if not errorlevel 1 goto triton_done
echo WARNING: triton-windows install failed. torch.compile and SDNQ quantized matmul will be disabled.
echo Generation will be significantly slower. Try: pip install -U "triton-windows>=3.6,<3.7"
:triton_done

echo Enabling Windows long path support (required for torch.compile temp files)...
python -c "import winreg; k=winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'SYSTEM\CurrentControlSet\Control\FileSystem',0,winreg.KEY_READ|winreg.KEY_WRITE); v=winreg.QueryValueEx(k,'LongPathsEnabled')[0]; print('Long paths already enabled.') if v==1 else None" 2>nul
if not errorlevel 1 goto longpath_done
python -c "import winreg; k=winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'SYSTEM\CurrentControlSet\Control\FileSystem',0,winreg.KEY_WRITE); winreg.SetValueEx(k,'LongPathsEnabled',0,winreg.REG_DWORD,1); print('Long paths enabled. Reboot may be required.')" 2>nul
if not errorlevel 1 goto longpath_done
echo WARNING: Could not enable long paths. Run Install.bat as Administrator.
echo   Or manually: reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
:longpath_done

echo Installing project requirements from %REQ_FILE%...
python -m pip install -r %REQ_FILE%
if errorlevel 1 goto install_fail

echo Ensuring Hugging Face Xet support is installed...
python -m pip install --upgrade hf_xet
if errorlevel 1 goto install_fail

echo Installing SageAttention (2-5x attention speedup for FLUX models)...
REM SageAttention cannot build from source on Windows + CUDA 13 (NVCC parsing bug).
REM Use woct0rdho ABI3 wheel — single wheel works across Python >=3.9 and PyTorch >=2.9.
python -m pip install "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post4/sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl" --no-deps
if not errorlevel 1 goto sage_done
echo woct0rdho ABI3 wheel failed. Trying sdbds prebuilt cu130 wheel...
for /f "tokens=2 delims=." %%a in ('python -c "import sys; print(sys.version)" 2^>nul') do set "PY_MINOR=%%a"
set "CP_TAG=cp3%PY_MINOR%"
echo   Detected Python minor version: %PY_MINOR% (wheel tag: %CP_TAG%)
python -m pip install "https://github.com/sdbds/SageAttention-for-windows/releases/download/sageattn_torch2100%%2Bcu130/sageattention-2.2.0%%2Bcu130torch2.10.0-%CP_TAG%-%CP_TAG%-win_amd64.whl" --no-deps 2>nul
if not errorlevel 1 goto sage_done
echo Prebuilt cu130 wheel failed. Falling back to Triton-only SageAttention 1.0...
python -m pip install sageattention==1.0.6
if not errorlevel 1 goto sage_done
echo WARNING: SageAttention install failed. Attention will use default SDPA (slower).
echo This is non-critical — generation will still work, just slower attention.
:sage_done

echo %INSTALL_PROFILE%> "%INSTALL_STAMP%"
type nul > venv\.installed

echo.
echo Post-install cleanup: removing stale files to save disk space...
REM Remove packages that conflict with CUDA 13 or are no longer used
python -m pip uninstall -y xformers flash-attn whisperx torchaudio qwen-tts pyannote-audio speechbrain librosa imageio-ffmpeg soundfile moviepy basicsr realesrgan facexlib insightface controlnet-aux mediapipe pyannote-metrics pyannote-pipeline pytorch-metric-learning torch-audiomentations torch-pitch-shift 2>nul
REM Clear pip download cache (wheels are installed, cache is dead weight)
python -m pip cache purge 2>nul
REM Remove orphaned packages that are no longer dependencies of anything
python -m pip autoremove -y 2>nul
REM Clear stale torch compile cache from previous PyTorch versions
if exist "%CD%\cache\torch_inductor" (
  echo Clearing old torch compile cache...
  rmdir /s /q "%CD%\cache\torch_inductor" 2>nul
)
REM Clean up any broken/invalid package dirs from partial uninstalls
for /d %%d in (venv\Lib\site-packages\~*) do rmdir /s /q "%%d" 2>nul
echo Cleanup complete.

if "%ENABLE_OPTIONAL_ACCELERATORS%"=="1" (
  call :install_optional_accelerators
)

:verify
echo.
echo Verifying critical dependencies...
python -m pip check
if errorlevel 1 (
  echo WARNING: pip dependency issues detected. Run Install.bat --repair to reconcile.
)

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda)"
if exist "scripts\verify_install.py" (
  python scripts\verify_install.py
  if errorlevel 1 goto install_fail
) else (
  echo WARNING: verify_install.py not found, skipping verification.
)

if exist "scripts\write_dependency_metadata.py" (
  python scripts\write_dependency_metadata.py --requirements "%REQ_FILE%" --output ".dependencies_verified"
  if errorlevel 1 (
    echo WARNING: Failed to write dependency metadata.
    goto install_fail
  )
) else (
  echo WARNING: write_dependency_metadata.py not found, skipping metadata generation.
)

echo.
echo Installation complete!
goto done

:install_optional_accelerators
echo Optional accelerators mode enabled.
echo Attempting CUDA 13 xformers installation from ABI3 wheel...
python -m pip install "https://huggingface.co/Wildminder/AI-windows-whl/resolve/main/xformers-0.0.34+torch2.10cu130-cp39-abi3-win_amd64.whl" --no-deps
if errorlevel 1 (
  echo WARNING: xformers CUDA 13 wheel failed or is unavailable for this runtime. Continuing with SageAttention/SDPA.
)
exit /b 0

:install_fail
echo.
echo ERROR: Installation failed.
goto done

:no_python
echo Python 3.10+ is required but was not found.
echo Please install Python from https://www.python.org/downloads/ and try again.
goto done

:no_venv
echo ERROR: Failed to activate virtual environment.

:done
echo.
pause
endlocal
exit /b 0
