@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
title Safetensors Model Merge - MiniMax H3

rem ============================================================
rem  start.bat - Launches krea-2-merge-tool.py
rem  Checks dependencies and asks whether to install them
rem  if any are missing.
rem ============================================================

cd /d "%~dp0"

set "PYTHON=python"

rem ------------------------------------------------------------
rem  Locate Python (falls back to "py" if "python" is missing)
rem ------------------------------------------------------------
where python >nul 2>nul
if errorlevel 1 (
    where py >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python was not found on this system.
        echo         Install Python from https://www.python.org/downloads/
        echo         and make sure to check "Add Python to PATH".
        pause
        exit /b 1
    )
    set "PYTHON=py"
)

echo ============================================================
echo  Safetensors Model Merge - Krea 2
echo ============================================================
echo.

rem ------------------------------------------------------------
rem  Check dependencies
rem ------------------------------------------------------------
set "MISSING="

%PYTHON% -c "import torch" >nul 2>nul
if errorlevel 1 set "MISSING=!MISSING! torch"

%PYTHON% -c "import safetensors" >nul 2>nul
if errorlevel 1 set "MISSING=!MISSING! safetensors"

if defined MISSING (
    echo The following dependencies are missing:
    echo   !MISSING!
    echo.
    set /p "INSTALL=Do you want to install them now? (Y/N): "
    if /i "!INSTALL!"=="Y" goto :install
    if /i "!INSTALL!"=="y" goto :install
    echo.
    echo Installation cancelled. The program will now exit.
    pause
    exit /b 1
)

goto :run

:install
echo.
echo Installing dependencies... (this may take a few minutes)
echo.
%PYTHON% -m pip install --upgrade pip
%PYTHON% -m pip install torch safetensors
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install the dependencies.
    echo         Try running it manually:
    echo         %PYTHON% -m pip install torch safetensors
    pause
    exit /b 1
)
echo.
echo Dependencies installed successfully!
echo.

:run
echo Starting the program...
echo.
%PYTHON% "%~dp0krea-2-merge-tool.py" %*
if errorlevel 1 (
    echo.
    echo The program exited with an error.
    pause
)

endlocal