@REM Cleanup unwanted API files
@REM This removes the old API-related files that are no longer needed

@echo off
echo Removing old API-related files...
echo.

if exist api_server.py (
    del api_server.py
    echo ✓ Removed api_server.py
)

if exist run_api.py (
    del run_api.py
    echo ✓ Removed run_api.py
)

if exist app.py (
    del app.py
    echo ✓ Removed app.py (Streamlit app - no longer needed)
)

if exist import_batch_check.py (
    del import_batch_check.py
    echo ✓ Removed import_batch_check.py
)

if exist import_check.py (
    del import_check.py
    echo ✓ Removed import_check.py
)

if exist import_test2.py (
    del import_test2.py
    echo ✓ Removed import_test2.py
)

if exist import_verify.py (
    del import_verify.py
    echo ✓ Removed import_verify.py
)

if exist model_loader.py (
    del model_loader.py
    echo ✓ Removed model_loader.py
)

if exist model_loader_py314.py (
    del model_loader_py314.py
    echo ✓ Removed model_loader_py314.py
)

if exist startup.bat (
    del startup.bat
    echo ✓ Removed startup.bat
)

echo.
echo Cleanup complete! Old API files have been removed.
echo.
echo Next steps:
echo 1. Run: python ml_server.py (to start the ML server)
echo 2. Run: cd frontend && npm install && npm run dev (to start the frontend)
pause
