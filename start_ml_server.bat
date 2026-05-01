@echo off
REM Start the ML inference server for DeClickify
echo Starting DeClickify ML Server...
echo.
echo The server will run on http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
python ml_server.py
pause
