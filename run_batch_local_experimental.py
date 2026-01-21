#!/usr/bin/env python3
"""
Run batch local file transcription with experimental GPU load balancing.

This script processes local files with multi-GPU support and load balancing.
"""

import subprocess
import sys
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Run experimental GPU load balancing for local file batch transcription'
    )
    parser.add_argument('--downloads', type=str,
                        default=r"C:\Users\jrmor\OneDrive\Documents\Programming\ramdass.io\transcriber\downloads",
                        help='Path to downloads folder')
    parser.add_argument('--model', type=str, default='medium',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size (default: medium)')
    parser.add_argument('-t', '--threads', type=int, default=4,
                        help='Number of parallel threads (default: 4)')
    parser.add_argument('--max-per-gpu', type=int, default=2,
                        help='Max concurrent tasks per GPU (default: 2)')
    parser.add_argument('--force', action='store_true',
                        help='Reprocess files even if .json already exists')
    parser.add_argument('--skip-claude', action='store_true',
                        help='Skip Claude refinement step')
    parser.add_argument('--skip-embeddings', action='store_true',
                        help='Skip embedding generation step')
    args = parser.parse_args()
    
    venv_python = Path(__file__).parent / "venv" / "Scripts" / "python.exe"
    
    print("=" * 80)
    print("EXPERIMENTAL MODE - GPU LOAD BALANCED LOCAL FILE TRANSCRIPTION")
    print("=" * 80)
    print()
    print("This mode will:")
    print("  - Monitor GPU utilization on cuda:0 and cuda:1")
    print("  - Dynamically assign tasks to the GPU with lower load")
    print("  - Use FP16 precision (2.5GB VRAM per task for medium model)")
    print(f"  - Run {args.threads} concurrent tasks ({args.max_per_gpu} per GPU max)")
    print("  - Process local files from: {args.downloads}")
    print(f"  - Model: {args.model}")
    if args.skip_claude:
        print("  - SKIP Claude refinement")
    if args.skip_embeddings:
        print("  - SKIP embeddings generation")
    if args.force:
        print("  - FORCE reprocessing of all files")
    print()
    print("Prerequisites Check:")
    print("  1. AWS authenticated? (for Claude - will prompt if needed)")
    print("  2. Two NVIDIA GPUs available? (cuda:0 and cuda:1)")
    print()
    print("=" * 80)
    print()
    
    # Build command
    cmd = [
        str(venv_python),
        "batch_transcribe_local_experimental.py",
        "--downloads", args.downloads,
        "--model", args.model,
        "-t", str(args.threads),
        "--max-per-gpu", str(args.max_per_gpu)
    ]
    
    if args.force:
        cmd.append("--force")
    if args.skip_claude:
        cmd.append("--skip-claude")
    if args.skip_embeddings:
        cmd.append("--skip-embeddings")
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    # Run batch_transcribe_local_experimental.py
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print()
        print("[WARNING] Batch processing encountered errors or was stopped")
        print()
        return result.returncode
    else:
        print()
        print("=" * 80)
        print("[SUCCESS] All local files have been processed!")
        print("=" * 80)
        return 0


if __name__ == '__main__':
    sys.exit(main())
