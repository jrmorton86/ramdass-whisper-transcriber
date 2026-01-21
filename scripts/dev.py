#!/usr/bin/env python3
"""
Development script to run both the API server and Next.js frontend.

Usage:
    python scripts/dev.py          # Run both services
    python scripts/dev.py api      # Run only API
    python scripts/dev.py frontend # Run only frontend
"""

import subprocess
import sys
import os
import signal
import time
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
API_DIR = ROOT_DIR / "packages" / "api"
FRONTEND_DIR = ROOT_DIR / "packages" / "frontend"

# Detect venv Python executable
if sys.platform == "win32":
    API_PYTHON = API_DIR / "venv" / "Scripts" / "python.exe"
else:
    API_PYTHON = API_DIR / "venv" / "bin" / "python"

processes = []


def run_api():
    """Run the FastAPI backend server."""
    print("\n[API] Starting FastAPI server...")

    # Check if venv exists
    if not API_PYTHON.exists():
        print(f"[API] ERROR: Virtual environment not found at {API_DIR / 'venv'}")
        print("[API] Please create it with: python -m venv packages/api/venv")
        print("[API] Then install deps: packages/api/venv/Scripts/pip install -r packages/api/requirements.txt")
        sys.exit(1)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR / "packages" / "pipeline")

    proc = subprocess.Popen(
        [str(API_PYTHON), "-m", "uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
        cwd=API_DIR,
        env=env,
    )
    processes.append(("API", proc))
    return proc


def run_frontend():
    """Run the Next.js development server."""
    print("\n[FRONTEND] Starting Next.js server...")

    # Check if node_modules exists
    node_modules = FRONTEND_DIR / "node_modules"
    if not node_modules.exists():
        print("[FRONTEND] Installing dependencies...")
        subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, check=True, shell=True)

    proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=FRONTEND_DIR,
        shell=True,
    )
    processes.append(("FRONTEND", proc))
    return proc


def cleanup(signum=None, frame=None):
    """Clean up all running processes."""
    print("\n\nShutting down services...")
    for name, proc in processes:
        print(f"  Stopping {name}...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("All services stopped.")
    sys.exit(0)


def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Parse arguments
    args = sys.argv[1:]
    run_api_only = "api" in args
    run_frontend_only = "frontend" in args

    print("=" * 60)
    print("  Transcription Dashboard - Development Server")
    print("=" * 60)

    if run_api_only:
        run_api()
    elif run_frontend_only:
        run_frontend()
    else:
        # Run both
        run_api()
        time.sleep(2)  # Give API time to start
        run_frontend()

    print("\n" + "=" * 60)
    if not run_api_only and not run_frontend_only:
        print("  API:      http://localhost:8000")
        print("  Frontend: http://localhost:3000")
        print("  API Docs: http://localhost:8000/docs")
    elif run_api_only:
        print("  API:      http://localhost:8000")
        print("  API Docs: http://localhost:8000/docs")
    else:
        print("  Frontend: http://localhost:3000")
    print("=" * 60)
    print("\nPress Ctrl+C to stop all services\n")

    # Wait for processes
    try:
        while True:
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"\n[{name}] Process exited with code {proc.returncode}")
                    cleanup()
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
