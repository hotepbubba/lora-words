@echo off
setlocal ENABLEDELAYEDEXPANSION

:: Directory for the virtual environment (defaults to .venv)
set VENV_DIR=.venv
if not "%~1"=="" (
    set VENV_DIR=%~1
)

if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Virtual environment already exists at %VENV_DIR%.
) else (
    echo Creating virtual environment in %VENV_DIR%...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create virtual environment. Ensure Python 3.9+ is installed and on PATH.
        exit /b 1
    )
)

echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo Failed to activate virtual environment.
    exit /b 1
)

echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip.
    exit /b 1
)

echo Installing requirements from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install required packages.
    exit /b 1
)

echo Installation complete. Virtual environment available at %VENV_DIR%.
exit /b 0
