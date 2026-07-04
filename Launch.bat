@echo off
setlocal EnableExtensions

echo ============================================
echo         ComfyUI Local Image App Launcher
echo ============================================
echo.

cd /d "%~dp0" 2>nul
if errorlevel 1 (
  echo ERROR: Could not change to the repository folder.
  exit /b 1
)

if not exist "ComfyUI\main.py" (
  echo ERROR: ComfyUI was not found.
  echo Run Install.bat first.
  exit /b 1
)

if not exist "venv\Scripts\python.exe" (
  echo ERROR: The virtual environment was not found.
  echo Run Install.bat first.
  exit /b 1
)

call "venv\Scripts\activate.bat"
if errorlevel 1 (
  echo ERROR: Could not activate the virtual environment.
  exit /b 1
)

REM Load HF_TOKEN from .env so the ComfyUI subprocess (and its custom nodes)
REM download models authenticated (higher rate limits + xet acceleration).
if exist ".env" (
  for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
    if /I "%%A"=="HF_TOKEN" set "HF_TOKEN=%%B"
  )
)

set "COMFYUI_HOST=127.0.0.1"
set "COMFYUI_PORT=8188"
set "COMFYUI_UI_HOST=127.0.0.1"
set "COMFYUI_UI_PORT=7861"
set "PYTHONWARNINGS=ignore::DeprecationWarning"
set "EXTRA_FLAGS="
for /f "usebackq delims=" %%A in (`python -c "from comfyui_app.vram import detect_vram, select_tier; gb, _, _ = detect_vram(); print(' '.join(select_tier(gb).extra_launch_flags))"`) do set "EXTRA_FLAGS=%%A"
set "SAGE_FLAG="
for /f "usebackq delims=" %%A in (`python -c "import sageattention; print('--use-sage-attention')" 2^>nul`) do set "SAGE_FLAG=%%A"

REM RTX 3070 / Ampere: scope --fast to fp16_accumulation (real Ampere speedup);
REM fp8_matrix_mult is a no-op on Ampere. --reserve-vram leaves headroom for the
REM Windows display to avoid OOM/offload stalls. --fast-disk speeds offload on NVMe.
echo Starting ComfyUI...
start "ComfyUI" /b python "ComfyUI\main.py" --listen %COMFYUI_HOST% --port %COMFYUI_PORT% --fast fp16_accumulation --reserve-vram 0.8 --fast-disk --preview-method latent2rgb %SAGE_FLAG% %EXTRA_FLAGS%

echo Waiting for ComfyUI to become ready...
python -c "from comfyui_app.comfy_client import ComfyClient; from comfyui_app.config import COMFYUI_HOST, COMFYUI_PORT; ComfyClient(COMFYUI_HOST, COMFYUI_PORT).wait_until_up(timeout=180)"
if errorlevel 1 (
  echo ERROR: ComfyUI did not start.
  exit /b 1
)

echo Launching the local image app...
echo Open the app at http://%COMFYUI_HOST%:%COMFYUI_UI_PORT%  (ComfyUI engine runs separately at http://%COMFYUI_HOST%:%COMFYUI_PORT%)
start "" "http://%COMFYUI_HOST%:%COMFYUI_UI_PORT%"
python -m comfyui_app.app
set "APP_EXIT=%errorlevel%"

if not "%APP_EXIT%"=="0" (
  echo ERROR: The app exited with code %APP_EXIT%.
  exit /b %APP_EXIT%
)

endlocal
exit /b 0
