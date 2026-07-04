@echo off
setlocal EnableExtensions

echo ============================================
echo         ComfyUI Local Image App Update
echo ============================================
echo.

cd /d "%~dp0" 2>nul
if errorlevel 1 (
  echo ERROR: Could not change to the repository folder.
  exit /b 1
)

if not exist "venv\Scripts\activate.bat" (
  echo ERROR: The virtual environment was not found.
  echo Run Install.bat first.
  exit /b 1
)

call "venv\Scripts\activate.bat"
if errorlevel 1 (
  echo ERROR: Could not activate the virtual environment.
  exit /b 1
)

echo Pulling the latest repository changes...
git pull
if errorlevel 1 (
  echo WARNING: git pull did not complete cleanly. Continuing with the local update.
)

echo Refreshing the ComfyUI helper dependencies...
python -m pip install -r requirements-comfyui.txt
if errorlevel 1 (
  echo ERROR: Dependency refresh failed.
  exit /b 1
)

echo Refreshing ComfyUI, custom nodes, and models...
python -m comfyui_app.installer %*
if errorlevel 1 (
  echo ERROR: The ComfyUI update step failed.
  exit /b 1
)

echo.
echo Update complete.
endlocal
exit /b 0
