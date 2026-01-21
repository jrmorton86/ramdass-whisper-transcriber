#!/usr/bin/env python3
"""
Batch process all audio/video files in the downloads folder.

Processes each file and creates a .json and .srt file next to it.
Skips files that already have a .json output.

Usage:
    python batch_transcribe_local.py
    python batch_transcribe_local.py --model large
    python batch_transcribe_local.py --force  # Reprocess even if .json exists
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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


def main():
    parser = argparse.ArgumentParser(
        description='Batch transcribe all local audio/video files in downloads folder'
    )
    parser.add_argument('--downloads', type=str,
                        default=r"C:\Users\jrmor\OneDrive\Documents\Programming\ramdass.io\transcriber\downloads",
                        help='Path to downloads folder')
    parser.add_argument('--model', type=str, default='medium',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size (default: medium)')
    parser.add_argument('-d', '--device', type=str,
                        help='CUDA device to use (e.g., cuda:0, cuda:1)')
    parser.add_argument('--force', action='store_true',
                        help='Reprocess files even if .json already exists')
    parser.add_argument('--skip-embeddings', action='store_true',
                        help='Skip embedding generation step')
    args = parser.parse_args()
    
    downloads_dir = Path(args.downloads)
    
    if not downloads_dir.exists():
        logger.error(f"❌ Downloads folder not found: {downloads_dir}")
        return 1
    
    # Find all media files
    media_files = find_media_files(downloads_dir)
    
    if not media_files:
        logger.info(f"No media files found in {downloads_dir}")
        return 0
    
    logger.info("=" * 80)
    logger.info("BATCH LOCAL FILE TRANSCRIPTION")
    logger.info("=" * 80)
    logger.info(f"Downloads folder: {downloads_dir}")
    logger.info(f"Found {len(media_files)} media files")
    logger.info(f"Model: {args.model}")
    if args.device:
        logger.info(f"Device: {args.device}")
    logger.info("=" * 80)
    logger.info("")
    
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
        logger.info(f"⏭️  Skipping {len(skipped)} files (already processed):")
        for f in skipped:
            logger.info(f"   - {f.name}")
        logger.info("")
    
    if not files_to_process:
        logger.info("✅ All files already processed! Use --force to reprocess.")
        return 0
    
    logger.info(f"📋 Processing {len(files_to_process)} files:")
    for f in files_to_process:
        logger.info(f"   - {f.name}")
    logger.info("")
    
    # Process each file
    venv_python = Path(__file__).parent / "venv" / "Scripts" / "python.exe"
    success_count = 0
    failed_count = 0
    failed_files = []
    
    for idx, media_file in enumerate(files_to_process, 1):
        logger.info("=" * 80)
        logger.info(f"[{idx}/{len(files_to_process)}] Processing: {media_file.name}")
        logger.info("=" * 80)
        
        # Build command
        cmd = [
            str(venv_python),
            "transcribe_local_file.py",
            str(media_file),
            "--model", args.model
        ]
        
        if args.device:
            cmd.extend(["--device", args.device])
        if args.skip_embeddings:
            cmd.append("--skip-embeddings")
        
        # Run transcription
        import subprocess
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            success_count += 1
            logger.info(f"✅ SUCCESS [{idx}/{len(files_to_process)}]")
        else:
            failed_count += 1
            failed_files.append(media_file.name)
            logger.error(f"❌ FAILED [{idx}/{len(files_to_process)}]")
        
        logger.info("")
    
    # Final summary
    logger.info("=" * 80)
    logger.info("🎉 BATCH PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total files: {len(files_to_process)}")
    logger.info(f"✅ Successful: {success_count}")
    logger.info(f"❌ Failed: {failed_count}")
    
    if failed_files:
        logger.info("")
        logger.info("Failed files:")
        for f in failed_files:
            logger.info(f"  - {f}")
    
    logger.info("=" * 80)
    
    return 0 if failed_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
