@echo off
REM Batch Process Audio from JSON Workflow
REM 
REM This script:
REM 1. Generates assets_without_embeddings.json from the database
REM 2. Runs batch_process_from_json.py to process all Audio assets
REM
REM Prerequisites:
REM - SSH tunnel must be running to database (run ssh_tunnel.py first)
REM - Python virtual environment at ./venv
REM - AWS authentication (will auto-prompt for 'aws sso login' if needed)

echo ================================================================================
echo BATCH PROCESS AUDIO FROM JSON - Full Workflow
echo ================================================================================
echo.
echo Prerequisites Check:
echo   1. SSH tunnel running? (python database_navigator/ssh_tunnel.py)
echo   2. AWS authenticated? (will prompt if needed)
echo.
echo Step 1: Generating assets_without_embeddings.json from database...
echo        (This will export all Audio assets without embeddings)
echo.

REM Run get_assets_without_embeddings.py with option 2 (export all)
echo 2 | venv\Scripts\python.exe database_navigator\get_assets_without_embeddings.py

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to generate assets_without_embeddings.json
    echo         Make sure SSH tunnel is running: python database_navigator\ssh_tunnel.py
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Step 2: Starting batch processing from assets_without_embeddings.json...
echo         (Processing Audio assets with MULTI-THREADING - up to 5 parallel)
echo ================================================================================
echo.
echo Press Ctrl+C to cancel, or any key to continue...
pause

REM Run batch_process_from_json.py with auto-continue enabled and 5 threads
REM Add -d cuda:0 or -d cuda:1 to specify which GPU to use
REM Add --skip N to skip the first N oldest assets (e.g., --skip 27)
venv\Scripts\python.exe batch_process_from_json.py -y -t 5

if errorlevel 1 (
    echo.
    echo [WARNING] Batch processing encountered errors or was stopped
    echo.
) else (
    echo.
    echo ================================================================================
    echo [SUCCESS] All audio assets have been processed!
    echo ================================================================================
)

echo.
pause
