#!/usr/bin/env python3
"""
Batch process local files with experimental GPU load balancing.

Uses ThreadPoolExecutor and GPU load balancer to distribute work across multiple GPUs.
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from gpu_load_balancer import GPULoadBalancer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(threadName)-10s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Supported audio/video extensions
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
ALL_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def find_media_files(downloads_dir: Path) -> list:
    """Find all audio/video files in downloads directory and subdirectories."""
    media_files = set()  # Use set to avoid duplicates
    
    for ext in ALL_EXTENSIONS:
        media_files.update(downloads_dir.rglob(f"*{ext}"))
        media_files.update(downloads_dir.rglob(f"*{ext.upper()}"))
    
    return sorted(media_files)


def process_single_file(media_file: Path, idx: int, total: int, model: str, 
                       skip_embeddings: bool, 
                       gpu_balancer: GPULoadBalancer | None = None) -> dict:
    """
    Process a single media file.
    Returns dict with status info: {'file': str, 'success': bool, 'error': str or None}
    """
    import threading
    thread_name = threading.current_thread().name
    
    # Determine which device to use
    assigned_device = None
    if gpu_balancer:
        assigned_device = gpu_balancer.get_best_device()
        logger.info(f"🎮 GPU Balancer Status: {gpu_balancer.get_status()}")
    
    logger.info("=" * 80)
    logger.info(f"[{idx}/{total}] Processing: {media_file.name}")
    logger.info(f"File: {media_file}")
    logger.info(f"Thread: {thread_name}")
    if assigned_device:
        logger.info(f"Device: {assigned_device}")
    logger.info("=" * 80)
    
    venv_python = Path(__file__).parent / "venv" / "Scripts" / "python.exe"
    
    cmd = [
        str(venv_python),
        "transcribe_local_file.py",
        str(media_file),
        "--model", model
    ]
    
    if assigned_device:
        cmd.extend(["--device", assigned_device])
    if skip_embeddings:
        cmd.append("--skip-embeddings")
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            logger.info(f"✅ SUCCESS [{idx}/{total}] - {media_file.name}")
            return {'file': str(media_file), 'success': True, 'error': None}
        else:
            error_msg = f"Exit code: {result.returncode}"
            logger.error(f"❌ FAILED [{idx}/{total}] - {error_msg}")
            return {'file': str(media_file), 'success': False, 'error': error_msg}
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ ERROR [{idx}/{total}]: {error_msg}")
        return {'file': str(media_file), 'success': False, 'error': error_msg}
    
    finally:
        # Release device back to balancer when task completes
        if gpu_balancer and assigned_device:
            gpu_balancer.release_device(assigned_device)


def main():
    parser = argparse.ArgumentParser(
        description='Batch transcribe local files with GPU load balancing'
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
    parser.add_argument('--skip-embeddings', action='store_true',
                        help='Skip embedding generation step')
    args = parser.parse_args()
    
    downloads_dir = Path(args.downloads)
    
    if not downloads_dir.exists():
        logger.error(f"❌ Downloads folder not found: {downloads_dir}")
        return 1
    
    # Initialize GPU load balancer
    gpu_balancer = GPULoadBalancer(gpu_ids=[0, 1], max_tasks_per_gpu=args.max_per_gpu)
    
    logger.info("=" * 80)
    logger.info("⚡ EXPERIMENTAL MODE: GPU LOAD BALANCED LOCAL FILE TRANSCRIPTION")
    logger.info("=" * 80)
    logger.info(f"Downloads folder: {downloads_dir}")
    logger.info(f"Model: {args.model}")
    logger.info(f"🧵 Total threads: {args.threads}")
    logger.info(f"🎮 Load balancing between cuda:0 and cuda:1")
    logger.info(f"📊 Max tasks per GPU: {args.max_per_gpu}")
    if args.skip_embeddings:
        logger.info("⏭️  Skipping embeddings generation")
    logger.info("=" * 80)
    logger.info("")
    
    # Find all media files
    media_files = find_media_files(downloads_dir)
    
    if not media_files:
        logger.info(f"No media files found in {downloads_dir}")
        return 0
    
    logger.info(f"Found {len(media_files)} media files")
    
    # Filter out files that already have JSON (unless --force)
    files_to_process = []
    skipped = []
    
    for media_file in media_files:
        json_file = media_file.with_suffix('.json')
        if json_file.exists() and not args.force:
            skipped.append(media_file)
        else:
            files_to_process.append(media_file)
    
    if skipped:
        logger.info(f"⏭️  Skipping {len(skipped)} files (already processed)")
        logger.info("")
    
    if not files_to_process:
        logger.info("✅ All files already processed! Use --force to reprocess.")
        return 0
    
    logger.info(f"📋 Processing {len(files_to_process)} files")
    logger.info("")
    
    # Process files in parallel using ThreadPoolExecutor
    results = []
    completed_count = 0
    failed_count = 0
    
    try:
        with ThreadPoolExecutor(max_workers=args.threads, thread_name_prefix='Worker') as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(
                    process_single_file,
                    media_file,
                    idx,
                    len(files_to_process),
                    args.model,
                    args.skip_embeddings,
                    gpu_balancer
                ): media_file
                for idx, media_file in enumerate(files_to_process, 1)
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_file):
                media_file = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        completed_count += 1
                    else:
                        failed_count += 1
                    
                    # Progress update
                    total_processed = completed_count + failed_count
                    logger.info(f"\n📊 Progress: {total_processed}/{len(files_to_process)} "
                              f"(✅ {completed_count} | ❌ {failed_count})")
                    
                except Exception as e:
                    logger.error(f"❌ Thread exception for {media_file.name}: {e}")
                    failed_count += 1
    
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Interrupted by user - waiting for current tasks to complete...")
        return 1
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("🎉 BATCH PROCESSING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Total processed: {len(files_to_process):,}")
    logger.info(f"✅ Successful: {completed_count:,}")
    logger.info(f"❌ Failed: {failed_count:,}")
    logger.info("=" * 80)
    
    return 0 if failed_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
