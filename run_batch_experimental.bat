@echo off
REM Batch Process Audio from JSON - EXPERIMENTAL GPU LOAD BALANCING
REM 
REM This script uses intelligent GPU load balancing to distribute work
REM across cuda:0 and cuda:1 based on real-time GPU utilization.
REM
REM Prerequisites:
REM - SSH tunnel must be running to database (run ssh_tunnel.py first)
REM - Python virtual environment at ./venv
REM - AWS authentication (will auto-prompt for 'aws sso login' if needed)
REM - TWO NVIDIA GPUs available (cuda:0 and cuda:1)

echo ================================================================================
echo EXPERIMENTAL MODE - GPU LOAD BALANCING
echo ================================================================================
echo.
echo This mode will:
echo   - Monitor GPU utilization on cuda:0 and cuda:1
echo   - Dynamically assign tasks to the GPU with lower load
echo   - Use FP16 precision (2.5GB VRAM per task for medium model)
echo   - Support up to 10 concurrent tasks (5 per GPU max)
echo   - Automatically balance workload in real-time
echo.
echo Prerequisites Check:
echo   1. SSH tunnel running? (python database_navigator/ssh_tunnel.py)
echo   2. AWS authenticated? (will prompt if needed)
echo   3. Two NVIDIA GPUs available? (cuda:0 and cuda:1)
echo.
echo ================================================================================

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
echo Step 2: Starting EXPERIMENTAL GPU load balancing mode...
echo         Processing with 4 threads across 2 GPUs (2 per GPU max)
echo         FP16 enabled: ~2.5GB VRAM per task
echo ================================================================================
echo.
echo Press Ctrl+C to cancel, or any key to continue...
pause

REM Run batch_process_from_json.py with experimental GPU load balancing
REM -t 4: 4 total threads
REM --experimental: Enable GPU load balancing
REM --max-per-gpu 2: Maximum 2 tasks per GPU
REM Add --skip N to skip the first N oldest assets (e.g., --skip 27)
venv\Scripts\python.exe batch_process_from_json.py -y -t 4 --experimental --max-per-gpu 2

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
