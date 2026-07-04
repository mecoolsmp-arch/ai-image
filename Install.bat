@echo off
setlocal EnableExtensions

echo ============================================
echo         ComfyUI Local Image App Setup
echo ============================================
echo.

cd /d "%~dp0" 2>nul
if errorlevel 1 (
  echo ERROR: Could not change to the repository folder.
  exit /b 1
)

set "PY_CMD=python"
set "PY_ARGS="
py -0p >nul 2>&1
if not errorlevel 1 (
  set "PY_CMD=py"
  set "PY_ARGS=-3"
)

where %PY_CMD% >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python was not found on PATH.
  exit /b 1
)

if not exist "venv" (
  echo Creating the virtual environment...
  %PY_CMD% %PY_ARGS% -m venv venv
  if errorlevel 1 exit /b 1
)

call "venv\Scripts\activate.bat"
if errorlevel 1 (
  echo ERROR: Could not activate the virtual environment.
  exit /b 1
)

echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 exit /b 1

echo Installing the CUDA PyTorch stack...
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 exit /b 1

echo Installing ComfyUI helper requirements...
python -m pip install -r requirements-comfyui.txt
if errorlevel 1 exit /b 1

echo Trying to add SageAttention 2 for faster attention...
python -m pip install --upgrade sageattention
if errorlevel 1 (
  echo SageAttention package install did not work. Trying the Windows wheel next...
  python -m pip install "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post4/sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl" --no-deps
)
if errorlevel 1 (
  echo WARNING: SageAttention 2 could not be installed. ComfyUI will use the default attention path.
  echo You can try again later from the SageAttention releases page.
)

echo Running the ComfyUI setup tool...
python -m comfyui_app.installer %*
if errorlevel 1 (
  echo.
  echo Setup did not finish successfully.
  exit /b 1
)

echo.
echo ComfyUI setup is complete.
endlocal
exit /b 0
