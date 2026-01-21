#!/usr/bin/env python3
"""
Batch Audio Asset Processor

Complete pipeline for each audio asset:
1. Download audio (S3 or Intelligence Bank)
2. Transcribe with LOCAL WHISPER AI (vocabulary-enhanced)
3. Generate SRT/TXT formats with Claude refinement
4. Creates Claude summary + keywords
5. Generates 4 embeddings (transcript, summary, keywords, metadata)
6. Runs AWS Comprehend analysis
7. Uploads all files to S3 + stores in database

Usage:
    python batch_process_audio.py [--limit N] [--skip N] [--model MODEL]

Examples:
    python batch_process_audio.py --limit 10
    python batch_process_audio.py --limit 100 --skip 50
    python batch_process_audio.py --model medium --limit 5
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import shutil
import asyncio
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database_navigator.db import get_connection as get_db_connection
from post_process_transcript import post_process_transcript

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_uuid(uuid_str: str) -> str:
    """Normalize UUID to add hyphens if needed (format: 8-4-4-4-12)."""
    uuid_str = uuid_str.strip().lower()
    if '-' in uuid_str:
        return uuid_str  # Already has hyphens
    if len(uuid_str) != 32:
        raise ValueError(f"Invalid UUID: expected 32 hex chars, got {len(uuid_str)}")
    # Add hyphens: 8-4-4-4-12
    return f"{uuid_str[0:8]}-{uuid_str[8:12]}-{uuid_str[12:16]}-{uuid_str[16:20]}-{uuid_str[20:32]}"


def get_audio_assets_without_embeddings(limit=None, offset=0):
    """
    Query database for Audio assets that don't have embeddings yet.
    
    Args:
        limit: Maximum number of assets to return (None = all)
        offset: Number of assets to skip
        
    Returns:
        List of dicts with asset info: {id, name, file_name, asset_type, etc}
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            query = """
                SELECT a.*
                FROM assets a
                WHERE a.id NOT IN (
                    SELECT DISTINCT resource_id 
                    FROM asset_embeddings
                )
                AND a.asset_type = 'Audio'
                ORDER BY a.created_at DESC
            """
            
            params = []
            if limit:
                query += " LIMIT %s"
                params.append(limit)
            if offset:
                query += " OFFSET %s"
                params.append(offset)
            
            cur.execute(query, params)
            columns = [desc[0] for desc in cur.description]
            results = []
            
            for row in cur.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
    finally:
        conn.close()


def main():
    """Main batch processing orchestrator."""
    parser = argparse.ArgumentParser(
        description='Batch process audio assets through complete transcription + embedding pipeline'
    )
    parser.add_argument('--uuid', help='Process specific asset by UUID (with or without hyphens)')
    parser.add_argument('--limit', type=int, help='Maximum number of assets to process')
    parser.add_argument('--skip', type=int, default=0, help='Number of assets to skip')
    parser.add_argument('--model', default='base', help='Whisper model (tiny, base, small, medium, large)')
    parser.add_argument('--device', help='CUDA device to use (e.g., cuda:0, cuda:1)')
    parser.add_argument('--silent', action='store_true', help='Silent mode - disable verbose output and Claude thinking display')
    args = parser.parse_args()
    
    # Get assets to process
    logger.info("=" * 80)
    logger.info("BATCH AUDIO ASSET PROCESSOR")
    logger.info("=" * 80)
    logger.info("\n⚠️  IMPORTANT: Ensure SSH tunnel is running:")
    logger.info("   python database_navigator/ssh_tunnel.py")
    logger.info("   OR in another terminal: ssh -i ~/.ssh/ramdass-bastion-temp.pem -N -L 5433:dam-ramdass-io-rds-instance-1.c7ecmfdohgux.us-east-1.rds.amazonaws.com:5432 ec2-user@54.175.205.16")
    
    # Handle specific UUID if provided
    if args.uuid:
        normalized_uuid = normalize_uuid(args.uuid)
        logger.info(f"\n🎯 Processing specific UUID: {normalized_uuid}")
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM assets WHERE id = %s AND asset_type = 'Audio'",
                    (normalized_uuid,)
                )
                columns = [desc[0] for desc in cur.description]
                row = cur.fetchone()
                if not row:
                    logger.error(f"❌ Asset not found: {normalized_uuid}")
                    sys.exit(1)
                assets = [dict(zip(columns, row))]
        finally:
            conn.close()
    else:
        # Query assets without embeddings
        assets = get_audio_assets_without_embeddings(limit=args.limit, offset=args.skip)
    
    if not assets:
        logger.info("✅ No assets found to process!")
        return
    
    logger.info(f"\n📋 Found {len(assets)} assets to process")
    logger.info(f"   Whisper model: {args.model}")
    
    # Process each asset
    results = {
        'total': len(assets),
        'successful': 0,
        'failed': 0,
        'assets': []
    }
    
    start_time = datetime.now()
    
    for idx, asset in enumerate(assets, 1):
        asset_uuid = asset['id']
        asset_name = asset['name'] or asset.get('file_name', 'Unknown')
        
        logger.info("\n" + "=" * 80)
        logger.info(f"[{idx}/{len(assets)}] {asset_name}")
        logger.info(f"UUID: {asset_uuid}")
        logger.info("=" * 80)
        
        temp_dir = None
        try:
            # Create temp directory: ./tmp/{uuid}/
            app_root = os.getcwd()
            tmp_base = os.path.join(app_root, 'tmp')
            os.makedirs(tmp_base, exist_ok=True)
            temp_dir = os.path.join(tmp_base, asset_uuid)
            os.makedirs(temp_dir, exist_ok=True)
            logger.info(f"Temp dir: {temp_dir}")
            
            # Step 1: Download audio
            logger.info("\n📥 Step 1/7: Downloading audio...")
            from intelligencebank_utils.download_asset import download_asset
            audio_path = asyncio.run(download_asset(asset_uuid))
            logger.info(f"✅ Downloaded: {audio_path}")
            
            # Move to temp_dir for processing
            audio_in_temp = os.path.join(temp_dir, os.path.basename(str(audio_path)))
            shutil.move(str(audio_path), audio_in_temp)
            audio_path = audio_in_temp
            
            # Step 2-4: Run Whisper transcription pipeline
            logger.info("\n🎙️  Steps 2-4/7: Transcribing with Whisper + Claude...")
            base_filename = Path(audio_path).stem
            output_dir = temp_dir
            
            # Run transcribe_pipeline.py from its directory to ensure relative imports work
            transcribe_script = os.path.join(os.getcwd(), 'transcribe_pipeline', 'transcribe_pipeline.py')
            cmd = [
                sys.executable,
                transcribe_script,
                str(audio_path),
                '--model', args.model,
                '--output', output_dir
            ]
            
            # Add device if specified
            if args.device:
                cmd.extend(['--device', args.device])
            
            # Add silent flag if enabled (verbose is default)
            if args.silent:
                cmd.append('--silent')
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=False, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Transcription failed with return code {result.returncode}")
            
            logger.info("✅ Transcription pipeline complete")
            
            # Step 5-7: Post-process (embeddings + Comprehend + DB + S3)
            logger.info("\n🚀 Steps 5-7/7: Post-processing (embeddings + Comprehend + DB + S3)...")
            transcript_base = os.path.join(output_dir, base_filename)
            post_result = post_process_transcript(asset_uuid, transcript_base)
            
            logger.info(f"\n✅ Post-processing complete!")
            logger.info(f"   📊 Embeddings: {post_result['embeddings_count']}")
            logger.info(f"   🏷️  Entities: {post_result['entities_count']}")
            logger.info(f"   🔑 Key phrases: {post_result['key_phrases_count']}")
            
            results['successful'] += 1
            results['assets'].append({
                'uuid': asset_uuid,
                'name': asset_name,
                'status': 'success'
            })
            
            logger.info(f"\n✅ SUCCESS ({results['successful']}/{len(assets)})")
            
        except Exception as e:
            error_msg = str(e)
            results['failed'] += 1
            results['assets'].append({
                'uuid': asset_uuid,
                'name': asset_name,
                'status': 'failed',
                'error': error_msg
            })
            
            logger.error(f"\n❌ FAILED ({results['failed']}/{len(assets)}): {error_msg}")
            import traceback
            logger.error(traceback.format_exc())
        
        finally:
            # Cleanup temp directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"Cleaned up: {temp_dir}")
    
    # Summary
    duration = datetime.now() - start_time
    
    logger.info("\n" + "=" * 80)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total:     {results['total']}")
    logger.info(f"✅ Success: {results['successful']}")
    logger.info(f"❌ Failed:  {results['failed']}")
    logger.info(f"⏱️  Duration: {duration}")
    if assets:
        logger.info(f"⚡ Avg/asset: {duration / len(assets)}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"batch_audio_results_{timestamp}.json"
    
    results['start_time'] = start_time.isoformat()
    results['end_time'] = datetime.now().isoformat()
    results['duration_seconds'] = duration.total_seconds()
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n📄 Results saved: {results_file}")
    sys.exit(1 if results['failed'] > 0 else 0)


if __name__ == '__main__':
    main()
