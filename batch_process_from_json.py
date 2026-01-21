#!/usr/bin/env python3
"""
Process audio assets from JSON file in reverse order.

Reads assets_without_embeddings.json (in transcriber folder), filters for Audio type,
and processes them from bottom to top (oldest to newest).

Prerequisites:
  1. SSH tunnel running: python database_navigator/ssh_tunnel.py
  2. JSON file generated: echo "2" | venv/Scripts/python.exe database_navigator/get_assets_without_embeddings.py
  3. AWS SSO authenticated: aws sso login (will auto-run if needed)
  
Or simply run: run_batch_from_json.bat

Automatically excludes assets in "Resources/To Be Sorted" folder.
"""

import json
import sys
import subprocess
import logging
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from gpu_load_balancer import GPULoadBalancer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(threadName)-10s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global lock for JSON file access
json_lock = threading.Lock()


def check_aws_authentication():
    """
    Check if AWS credentials are valid by trying to get caller identity.
    Returns True if authenticated, False otherwise.
    """
    try:
        result = subprocess.run(
            ['aws', 'sts', 'get-caller-identity'],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception as e:
        logger.debug(f"AWS auth check failed: {e}")
        return False


def run_aws_sso_login():
    """
    Run 'aws sso login' interactively and wait for completion.
    Returns True if successful, False otherwise.
    """
    logger.warning("="*80)
    logger.warning("AWS Authentication Required")
    logger.warning("="*80)
    logger.warning("Running: aws sso login")
    logger.warning("Please complete the authentication in your browser...")
    logger.warning("="*80)
    
    try:
        # Run aws sso login interactively
        result = subprocess.run(['aws', 'sso', 'login'], check=False)
        
        if result.returncode == 0:
            logger.info("✅ AWS SSO login successful!")
            # Give AWS a moment to propagate credentials
            time.sleep(2)
            return True
        else:
            logger.error("❌ AWS SSO login failed")
            return False
            
    except FileNotFoundError:
        logger.error("❌ AWS CLI not found. Please install AWS CLI first.")
        logger.error("   Download from: https://aws.amazon.com/cli/")
        return False
    except Exception as e:
        logger.error(f"❌ Error during AWS SSO login: {e}")
        return False


def remove_asset_from_json(json_file: Path, asset_uuid: str) -> bool:
    """
    Thread-safe removal of an asset from the JSON file.
    Returns True if asset was found and removed, False otherwise.
    """
    with json_lock:
        try:
            # Read current JSON
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Find and remove the asset
            assets = data.get('assets', [])
            original_count = len(assets)
            
            # Filter out the asset with matching UUID
            data['assets'] = [a for a in assets if a.get('id') != asset_uuid]
            
            if len(data['assets']) == original_count:
                # Asset not found
                return False
            
            # Write updated JSON back
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Removed UUID {asset_uuid} from JSON ({original_count} → {len(data['assets'])} assets)")
            return True
            
        except Exception as e:
            logger.error(f"Error removing asset from JSON: {e}")
            return False


def process_single_asset(asset: dict, json_file: Path, idx: int, total: int, auto_continue: bool, 
                         device: str | None = None, gpu_balancer: GPULoadBalancer | None = None) -> dict:
    """
    Process a single audio asset.
    Returns dict with status info: {'uuid': str, 'success': bool, 'error': str or None}
    """
    uuid = asset['id']
    name = asset.get('name') or asset.get('file_name', 'Unknown')
    thread_name = threading.current_thread().name
    
    # Determine which device to use
    assigned_device = device
    if gpu_balancer:
        # Experimental mode: dynamically select best GPU
        assigned_device = gpu_balancer.get_best_device()
        logger.info(f"🎮 GPU Balancer Status: {gpu_balancer.get_status()}")
    
    logger.info("="*80)
    logger.info(f"[{idx}/{total}] Processing: {name}")
    logger.info(f"UUID: {uuid}")
    logger.info(f"Thread: {thread_name}")
    if assigned_device:
        logger.info(f"Device: {assigned_device}")
    logger.info("="*80)
    
    # FIRST: Remove from JSON to prevent duplicate processing
    if not remove_asset_from_json(json_file, uuid):
        logger.warning(f"⚠️  Asset {uuid} not found in JSON (may have been processed already)")
        if gpu_balancer and assigned_device:
            gpu_balancer.release_device(assigned_device)
        return {'uuid': uuid, 'success': False, 'error': 'Asset not in JSON'}
    
    # NOW: Process the asset
    venv_python = Path(__file__).parent / "venv" / "Scripts" / "python.exe"
    cmd = [
        str(venv_python),
        "batch_process_audio.py",
        "--uuid", uuid,
        "--model", "medium"
    ]
    
    # Add device if specified
    if assigned_device:
        cmd.extend(["--device", assigned_device])
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            logger.info(f"✅ SUCCESS [{idx}/{total}] - {name}")
            return {'uuid': uuid, 'success': True, 'error': None}
        else:
            error_msg = f"Exit code: {result.returncode}"
            logger.error(f"❌ FAILED [{idx}/{total}] - {error_msg}")
            return {'uuid': uuid, 'success': False, 'error': error_msg}
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ ERROR [{idx}/{total}]: {error_msg}")
        return {'uuid': uuid, 'success': False, 'error': error_msg}
    
    finally:
        # Release device back to balancer when task completes
        if gpu_balancer and assigned_device:
            gpu_balancer.release_device(assigned_device)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process audio assets from JSON file with multi-threading')
    parser.add_argument('-y', '--yes', action='store_true', 
                        help='Auto-continue on errors without prompting')
    parser.add_argument('-t', '--threads', type=int, default=5,
                        help='Number of parallel threads (default: 5)')
    parser.add_argument('-d', '--device', type=str,
                        help='CUDA device to use (e.g., cuda:0, cuda:1)')
    parser.add_argument('--experimental', action='store_true',
                        help='⚡ EXPERIMENTAL: Enable GPU load balancing across cuda:0 and cuda:1 (ignores -d)')
    parser.add_argument('--max-per-gpu', type=int, default=5,
                        help='Max concurrent tasks per GPU in experimental mode (default: 5)')
    parser.add_argument('--skip', type=int, default=0,
                        help='Number of assets to skip from the beginning of the list (default: 0)')
    args = parser.parse_args()
    
    json_file = Path("assets_without_embeddings.json")
    
    if not json_file.exists():
        logger.error(f"JSON file not found: {json_file}")
        logger.info("Generate it with: echo '2' | venv/Scripts/python.exe database_navigator/get_assets_without_embeddings.py")
        logger.info("Or run: run_batch_from_json.bat")
        return 1
    
    # Check AWS authentication before starting
    logger.info("Checking AWS authentication...")
    if not check_aws_authentication():
        logger.warning("AWS credentials not found or expired")
        
        # Auto-login with AWS SSO
        if not run_aws_sso_login():
            logger.error("Failed to authenticate with AWS. Cannot proceed.")
            logger.error("Please run 'aws sso login' manually and try again.")
            return 1
        
        # Verify authentication succeeded
        if not check_aws_authentication():
            logger.error("AWS authentication still invalid after SSO login")
            return 1
    else:
        logger.info("✅ AWS credentials valid")
    
    # Initialize GPU load balancer if experimental mode
    gpu_balancer = None
    if args.experimental:
        logger.info("="*80)
        logger.info("⚡ EXPERIMENTAL MODE: GPU LOAD BALANCING ENABLED")
        logger.info("="*80)
        gpu_balancer = GPULoadBalancer(gpu_ids=[0, 1], max_tasks_per_gpu=args.max_per_gpu)
        logger.info(f"🧵 Total threads: {args.threads}")
        logger.info(f"🎮 Load balancing between cuda:0 and cuda:1")
        logger.info(f"📊 Max tasks per GPU: {args.max_per_gpu}")
        logger.info("="*80)
    else:
        logger.info("="*80)
        logger.info("BATCH PROCESS AUDIO FROM JSON (MULTI-THREADED)")
        logger.info("="*80)
        if args.yes:
            logger.info("⚡ Auto-continue mode enabled (-y)")
        logger.info(f"🧵 Parallel threads: {args.threads}")
        if args.device:
            logger.info(f"🎮 CUDA device: {args.device}")
        logger.info("="*80)
    
    # Load JSON
    logger.info(f"\nLoading: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter for Audio assets only, excluding "Resources/To Be Sorted"
    all_assets = data.get('assets', [])
    audio_assets = [
        a for a in all_assets 
        if a.get('asset_type') == 'Audio' 
        and a.get('folder_path') != 'Resources/To Be Sorted'
    ]
    
    logger.info(f"Total assets in file: {len(all_assets):,}")
    logger.info(f"Audio assets (excluding 'To Be Sorted'): {len(audio_assets):,}")
    
    if not audio_assets:
        logger.info("No Audio assets found!")
        return 0
    
    # Reverse the list (bottom to top = oldest to newest)
    audio_assets.reverse()
    
    # Skip assets if requested - do this LAST so we skip from the final processing list
    if args.skip > 0:
        if args.skip >= len(audio_assets):
            logger.error(f"❌ Cannot skip {args.skip} assets - only {len(audio_assets)} available!")
            return 1
        
        logger.info(f"\n⏭️  Skipping first {args.skip} assets (oldest)")
        logger.info(f"   Skipped range: {audio_assets[0]['id']} ... {audio_assets[args.skip-1]['id']}")
        
        # Remove skipped assets from the list we'll process
        audio_assets = audio_assets[args.skip:]
        
        logger.info(f"   ✅ Skipped {args.skip} assets - will process remaining {len(audio_assets)}")
    
    logger.info(f"\n🎯 Processing {len(audio_assets):,} Audio assets in REVERSE order")
    logger.info(f"   First UUID: {audio_assets[0]['id']}")
    logger.info(f"   Last UUID: {audio_assets[-1]['id']}")
    logger.info(f"\n⚡ Starting multi-threaded processing with {args.threads} workers...")
    
    # Process assets in parallel using ThreadPoolExecutor
    results = []
    completed_count = 0
    failed_count = 0
    
    try:
        with ThreadPoolExecutor(max_workers=args.threads, thread_name_prefix='Worker') as executor:
            # Submit all tasks
            future_to_asset = {
                executor.submit(
                    process_single_asset, 
                    asset, 
                    json_file, 
                    idx, 
                    len(audio_assets), 
                    args.yes,
                    args.device if not args.experimental else None,
                    gpu_balancer
                ): asset 
                for idx, asset in enumerate(audio_assets, 1)
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_asset):
                asset = future_to_asset[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        completed_count += 1
                    else:
                        failed_count += 1
                    
                    # Progress update
                    total_processed = completed_count + failed_count
                    logger.info(f"\n📊 Progress: {total_processed}/{len(audio_assets)} "
                              f"(✅ {completed_count} | ❌ {failed_count})")
                    
                except Exception as e:
                    logger.error(f"❌ Thread exception for {asset.get('id', 'unknown')}: {e}")
                    failed_count += 1
    
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Interrupted by user - waiting for current tasks to complete...")
        return 1
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("🎉 BATCH PROCESSING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Total processed: {len(audio_assets):,}")
    logger.info(f"✅ Successful: {completed_count:,}")
    logger.info(f"❌ Failed: {failed_count:,}")
    logger.info("="*80)
    
    return 0 if failed_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
