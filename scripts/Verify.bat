@echo off
setlocal EnableExtensions

echo.
echo ============================================
echo     Manga to Realistic Verification
echo ============================================
echo.

cd /d "%~dp0.." 2>nul
if errorlevel 1 goto fail

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: venv not found. Run Install.bat first.
    goto fail
)

call venv\Scripts\activate.bat
if errorlevel 1 goto fail

python --version
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda)"
if errorlevel 1 goto fail

python -c "import gradio, transformers, diffusers, accelerate, safetensors, PIL, scipy, peft, requests; print('Core image dependencies OK')"
if errorlevel 1 goto fail

python scripts\verify_install.py
if errorlevel 1 goto fail

echo.
echo Verification passed. Run Launch.bat to start the image editor.
echo.
pause
endlocal
exit /b 0

:fail
echo.
echo Verification failed. Run Install.bat --repair if dependencies are missing.
echo.
pause
endlocal
exit /b 1
