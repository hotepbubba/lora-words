@echo off
setlocal

set VENV_DIR=.venv
if /I "%~1"=="--venv" (
    if "%~2"=="" (
        echo Missing value for --venv option.
        exit /b 1
    )
    set VENV_DIR=%~2
    shift
    shift
)

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Virtual environment not found at %VENV_DIR%.
    echo Run install.bat first (optionally specifying the same virtual environment path).
    exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo Failed to activate virtual environment at %VENV_DIR%.
    exit /b 1
)

if "%~1"=="" (
    echo Running lora_trigger_sync.py with --help.
    python lora_trigger_sync.py --help
) else (
    python lora_trigger_sync.py %*
)

exit /b %errorlevel%
