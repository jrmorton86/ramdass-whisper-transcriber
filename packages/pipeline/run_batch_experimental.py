#!/usr/bin/env python3
"""
Run batch processing with experimental GPU load balancing.

This Python script replicates run_batch_experimental.bat functionality
but with command-line arguments for flexibility.
"""

import subprocess
import sys
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Run experimental GPU load balancing batch process'
    )
    parser.add_argument('-t', '--threads', type=int, default=4,
                        help='Number of parallel threads (default: 4)')
    parser.add_argument('--max-per-gpu', type=int, default=2,
                        help='Max concurrent tasks per GPU (default: 2)')
    parser.add_argument('--skip', type=int, default=0,
                        help='Number of assets to skip from the beginning (default: 0)')
    parser.add_argument('--no-regenerate', action='store_true',
                        help='Skip regenerating JSON from database')
    args = parser.parse_args()
    
    venv_python = Path(__file__).parent / "venv" / "Scripts" / "python.exe"
    
    print("=" * 80)
    print("EXPERIMENTAL MODE - GPU LOAD BALANCING")
    print("=" * 80)
    print()
    print("This mode will:")
    print("  - Monitor GPU utilization on cuda:0 and cuda:1")
    print("  - Dynamically assign tasks to the GPU with lower load")
    print("  - Use FP16 precision (2.5GB VRAM per task for medium model)")
    print(f"  - Run {args.threads} concurrent tasks ({args.max_per_gpu} per GPU max)")
    print("  - Automatically balance workload in real-time")
    if args.skip > 0:
        print(f"  - Skip first {args.skip} oldest assets")
    print()
    print("Prerequisites Check:")
    print("  1. SSH tunnel running? (python database_navigator/ssh_tunnel.py)")
    print("  2. AWS authenticated? (will prompt if needed)")
    print("  3. Two NVIDIA GPUs available? (cuda:0 and cuda:1)")
    print()
    print("=" * 80)
    print()
    
    if not args.no_regenerate:
        print("Step 1: Generating assets_without_embeddings.json from database...")
        print("        (This will export all Audio assets without embeddings)")
        print()
        
        # Run get_assets_without_embeddings.py with option 2 (export all)
        cmd = [str(venv_python), "database_navigator/get_assets_without_embeddings.py"]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        stdout, _ = proc.communicate(input="2\n")
        print(stdout)
        
        if proc.returncode != 0:
            print()
            print("[ERROR] Failed to generate assets_without_embeddings.json")
            print("        Make sure SSH tunnel is running: python database_navigator/ssh_tunnel.py")
            print()
            return 1
    else:
        print("Skipping JSON regeneration (using existing file)")
        print()
    
    print("=" * 80)
    print("Step 2: Starting EXPERIMENTAL GPU load balancing mode...")
    print(f"        Processing with {args.threads} threads across 2 GPUs ({args.max_per_gpu} per GPU max)")
    print("        FP16 enabled: ~2.5GB VRAM per task")
    if args.skip > 0:
        print(f"        Skipping first {args.skip} oldest assets")
    print("=" * 80)
    print()
    
    # Build command
    cmd = [
        str(venv_python),
        "batch_process_from_json.py",
        "-y",
        "-t", str(args.threads),
        "--experimental",
        "--max-per-gpu", str(args.max_per_gpu)
    ]
    
    if args.skip > 0:
        cmd.extend(["--skip", str(args.skip)])
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    # Run batch_process_from_json.py
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print()
        print("[WARNING] Batch processing encountered errors or was stopped")
        print()
        return result.returncode
    else:
        print()
        print("=" * 80)
        print("[SUCCESS] All audio assets have been processed!")
        print("=" * 80)
        return 0


if __name__ == '__main__':
    sys.exit(main())
