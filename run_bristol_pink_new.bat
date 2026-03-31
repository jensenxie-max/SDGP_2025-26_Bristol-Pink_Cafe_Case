@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "BASE_DIR=%~dp0"
set "VENV_DIR=%BASE_DIR%.venv"
set "REQ_FILE=%BASE_DIR%requirements.txt"
set "APP_FILE="

if exist "%BASE_DIR%Bristol-Pink_improved_fixed_colors.py" (
    set "APP_FILE=%BASE_DIR%Bristol-Pink_improved_fixed_colors.py"
) else if exist "%BASE_DIR%Bristol-Pink_improved.py" (
    set "APP_FILE=%BASE_DIR%Bristol-Pink_improved.py"
)

echo ========================================
echo Bristol-Pink Streamlit Launcher
echo ========================================
echo.

if not defined APP_FILE (
    echo [ERROR] Could not find either of these files:
    echo   %BASE_DIR%Bristol-Pink_improved_fixed_colors.py
    echo   %BASE_DIR%Bristol-Pink_improved.py
    echo.
    pause
    exit /b 1
)

echo [INFO] App file:
echo %APP_FILE%
echo.

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [INFO] Creating virtual environment...
    py -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        echo Make sure Python is installed and the py launcher is available.
        echo.
        pause
        exit /b 1
    )
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    echo.
    pause
    exit /b 1
)

echo [INFO] Python in use:
python --version
where python
echo.

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] pip upgrade failed.
    echo.
    pause
    exit /b 1
)

echo.
if exist "%REQ_FILE%" (
    echo [INFO] Installing dependencies from requirements.txt...
    python -m pip install -r "%REQ_FILE%"
) else (
    echo [INFO] requirements.txt not found. Installing default packages...
    python -m pip install streamlit pandas numpy plotly scikit-learn
)
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    echo.
    pause
    exit /b 1
)

echo.
echo [INFO] Optional: if the page still looks old, press Ctrl+F5 in the browser.
echo [INFO] Starting Streamlit...
echo.

python -m streamlit run "%APP_FILE%" --server.headless false

echo.
echo [INFO] Streamlit has stopped.
pause
