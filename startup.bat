@echo off
REM DeClickify - Full Stack Startup Script

echo ============================================
echo   DeClickify - Full Stack Startup
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    exit /b 1
)

echo [1/3] Setting up Backend...
echo.

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install backend dependencies
echo Installing backend dependencies...
pip install -r requirements.txt --quiet

echo.
echo [2/3] Setting up Frontend...
echo.

cd frontend

REM Install frontend dependencies
if not exist node_modules (
    echo Installing frontend dependencies...
    call npm install --quiet
)

cd ..

echo.
echo [3/3] Starting Services...
echo.

echo.
echo ============================================
echo   Services Starting...
echo ============================================
echo.
echo Backend API:  http://localhost:5000
echo Frontend UI:  http://localhost:5173
echo.
echo Press Ctrl+C to stop services
echo ============================================
echo.

REM Start backend in a new window
echo Starting Backend API Server...
start "DeClickify Backend" cmd /k "python api_server.py"

timeout /t 2

REM Start frontend in a new window
echo Starting Frontend Development Server...
start "DeClickify Frontend" cmd /k "cd frontend && npm run dev"

REM Keep main window open
pause
