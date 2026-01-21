#!/usr/bin/env python3
"""
Batch Process Audio with Worker Pool

High-performance batch processor using persistent GPU workers.
Model loads once per GPU, eliminating per-asset loading overhead.

Usage:
    python batch_process_pool.py [options]

Examples:
    python batch_process_pool.py -t 2 --gpus 0,1
    python batch_process_pool.py -t 1 --gpus 0 -m large
"""

import json
import sys
import logging
import subprocess
import time
import threading
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent))
from worker_pool import WorkerPool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lock for JSON file updates
json_lock = threading.Lock()


def check_aws_authentication():
    """Check if AWS credentials are valid."""
    try:
        result = subprocess.run(
            ['aws', 'sts', 'get-caller-identity'],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def run_aws_sso_login():
    """Run AWS SSO login interactively."""
    logger.warning("=" * 80)
    logger.warning("AWS Authentication Required")
    logger.warning("Running: aws sso login")
    logger.warning("=" * 80)

    try:
        result = subprocess.run(['aws', 'sso', 'login'], check=False)
        if result.returncode == 0:
            logger.info("[OK] AWS SSO login successful")
            time.sleep(2)
            return True
        return False
    except Exception as e:
        logger.error(f"AWS SSO login failed: {e}")
        return False


def remove_asset_from_json(json_file: Path, asset_uuid: str) -> bool:
    """Thread-safe removal of asset from JSON file."""
    with json_lock:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            assets = data.get('assets', [])
            original_count = len(assets)
            data['assets'] = [a for a in assets if a.get('id') != asset_uuid]

            if len(data['assets']) == original_count:
                return False

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            logger.error(f"Error updating JSON: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Batch process audio with persistent GPU workers'
    )
    parser.add_argument('-t', '--threads', type=int, default=2,
                        help='Number of workers/GPUs (default: 2)')
    parser.add_argument('-m', '--model', default='medium',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model (default: medium)')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Auto-continue on errors')
    parser.add_argument('--gpus', default='0,1',
                        help='GPU IDs to use (comma-separated, default: 0,1)')
    parser.add_argument('--skip', type=int, default=0,
                        help='Number of assets to skip')

    args = parser.parse_args()

    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')][:args.threads]

    json_file = Path("assets_without_embeddings.json")

    if not json_file.exists():
        logger.error(f"JSON file not found: {json_file}")
        logger.info("Generate with: echo '2' | python database_navigator/get_assets_without_embeddings.py")
        return 1

    # Check AWS authentication
    logger.info("Checking AWS authentication...")
    if not check_aws_authentication():
        if not run_aws_sso_login() or not check_aws_authentication():
            logger.error("AWS authentication failed")
            return 1
    logger.info("[OK] AWS credentials valid")

    # Load assets
    logger.info(f"\nLoading: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_assets = data.get('assets', [])
    audio_assets = [
        a for a in all_assets
        if a.get('asset_type') == 'Audio'
        and a.get('folder_path') != 'Resources/To Be Sorted'
    ]

    # Reverse for oldest-first processing
    audio_assets.reverse()

    # Apply skip
    if args.skip > 0:
        if args.skip >= len(audio_assets):
            logger.error(f"Cannot skip {args.skip} - only {len(audio_assets)} available")
            return 1
        audio_assets = audio_assets[args.skip:]
        logger.info(f"Skipped {args.skip} assets")

    logger.info(f"\n{'='*80}")
    logger.info("WORKER POOL BATCH PROCESSOR")
    logger.info(f"{'='*80}")
    logger.info(f"Assets to process: {len(audio_assets)}")
    logger.info(f"Workers: {len(gpu_ids)} (GPUs: {gpu_ids})")
    logger.info(f"Model: {args.model}")
    logger.info(f"{'='*80}\n")

    if not audio_assets:
        logger.info("No assets to process")
        return 0

    # Start worker pool
    pool = WorkerPool(gpu_ids=gpu_ids, model_name=args.model)

    try:
        pool.start()

        # Submit all assets
        for asset in audio_assets:
            uuid = asset['id']
            name = asset.get('name') or asset.get('file_name', 'Unknown')

            # Remove from JSON before submitting
            remove_asset_from_json(json_file, uuid)
            pool.submit(uuid, name)

        # Collect results
        completed = 0
        failed = 0
        total = len(audio_assets)

        while completed + failed < total:
            result = pool.get_result(timeout=600)  # 10 min timeout per asset

            if result is None:
                logger.warning("Timeout waiting for result")
                failed += 1
                continue

            if result['status'] == 'success':
                completed += 1
                logger.info(f"[OK] {result['name']}")
            else:
                failed += 1
                logger.error(f"[FAIL] {result['name']}: {result.get('error', 'Unknown')}")

            logger.info(f"Progress: {completed + failed}/{total} | OK: {completed} | FAIL: {failed}")

        # Summary
        logger.info(f"\n{'='*80}")
        logger.info("BATCH COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total: {total}")
        logger.info(f"Success: {completed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"{'='*80}")

        return 0 if failed == 0 else 1

    except KeyboardInterrupt:
        logger.info("\nInterrupted - shutting down...")
        return 1

    finally:
        pool.shutdown()


if __name__ == '__main__':
    sys.exit(main())
