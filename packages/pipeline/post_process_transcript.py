#!/usr/bin/env python3
"""
Post-Transcription Processing: Embeddings + Comprehend + Database + S3

Takes existing transcript files from the transcribe_pipeline.py output and:
1. Generates 4 embeddings (transcript, summary, keywords, metadata)
2. Runs AWS Comprehend analysis
3. Stores in database
4. Uploads all files to S3

Usage:
    python post_process_transcript.py <asset_uuid> <transcript_base_path>

Example:
    python post_process_transcript.py abc123-uuid downloads/Clip_1_1969
    
This expects these files to exist:
    - {base_path}.json           (Whisper JSON output)
    - {base_path}_formatted_refined.txt  (Claude refined text)
    - {base_path}_refined.srt    (Refined SRT)
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from comprehend_utils.comprehend import (
    analyze_transcript_with_comprehend,
    generate_embedding,
    store_embeddings,
    store_comprehend_analysis
)
from comprehend_utils.generate_audio_embeddings import generate_summary_and_keywords
from database_navigator.db import get_connection
import boto3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# AWS Configuration
S3_BUCKET = os.getenv('S3_BUCKET', 'dam-ramdass-io-assets')
s3_client = boto3.client('s3', region_name='us-east-1')


def upload_to_s3(local_path: str, s3_key: str) -> str:
    """Upload file to S3"""
    s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
    logger.info(f"Uploading {local_path} to {s3_uri}...")
    s3_client.upload_file(local_path, S3_BUCKET, s3_key)
    file_size = os.path.getsize(local_path)
    logger.info(f"✅ Uploaded ({file_size:,} bytes)")
    return s3_uri


def post_process_transcript(asset_uuid: str, transcript_base_path: str):
    """
    Post-process transcript files: embeddings + Comprehend + DB + S3
    
    Args:
        asset_uuid: Asset UUID
        transcript_base_path: Base path to transcript files (without extension)
    """
    base_path = Path(transcript_base_path)
    base_name = base_path.name
    
    # Expected files
    json_file = base_path.parent / f"{base_name}.json"
    refined_txt = base_path.parent / f"{base_name}_formatted_refined.txt"
    refined_srt = base_path.parent / f"{base_name}_refined.srt"
    
    logger.info("=" * 80)
    logger.info(f"POST-PROCESSING TRANSCRIPT: {asset_uuid}")
    logger.info("=" * 80)
    
    # Step 1: Load transcript text
    logger.info("Step 1/5: Loading transcript files...")
    if not refined_txt.exists():
        raise FileNotFoundError(f"Refined transcript not found: {refined_txt}")
    
    with open(refined_txt, 'r', encoding='utf-8') as f:
        plain_text = f.read()
    
    logger.info(f"✅ Loaded transcript ({len(plain_text):,} chars)")
    
    # Get asset info from database
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM assets WHERE id = %s", (asset_uuid,))
    result = cur.fetchone()
    if not result:
        raise ValueError(f"Asset not found in database: {asset_uuid}")
    
    asset_id = result[0]
    asset_name = result[1] or base_name
    logger.info(f"Asset: {asset_name}")
    
    # Step 2: Generate summary + keywords
    logger.info("Step 2/5: Generating summary and keywords with Claude...")
    summary_text, keywords = generate_summary_and_keywords(plain_text, asset_name)
    logger.info(f"✅ Summary: {len(summary_text)} chars")
    logger.info(f"✅ Keywords: {', '.join(keywords[:5])}...")
    
    # Step 3: Generate 4 embeddings
    logger.info("Step 3/5: Generating 4 embeddings with Titan...")
    
    embeddings_data = []
    
    # Embedding 1: Transcript (first 8000 chars)
    logger.info("  ⏳ Generating transcript embedding (1/4)...")
    transcript_embedding = generate_embedding(plain_text[:8000])
    embeddings_data.append({
        'content_type': 'transcript',
        'text': plain_text[:8000],
        'embedding': transcript_embedding,
        'metadata': {}
    })
    logger.info("  ✅ Transcript embedding generated")
    
    # Embedding 2: Summary
    logger.info("  ⏳ Generating summary embedding (2/4)...")
    summary_embedding = generate_embedding(summary_text)
    embeddings_data.append({
        'content_type': 'summary',
        'text': summary_text,
        'embedding': summary_embedding,
        'metadata': {}
    })
    logger.info("  ✅ Summary embedding generated")
    
    # Embedding 3: Keywords
    logger.info("  ⏳ Generating keywords embedding (3/4)...")
    keywords_text = ', '.join(keywords)
    keywords_embedding = generate_embedding(keywords_text)
    embeddings_data.append({
        'content_type': 'keywords',
        'text': keywords_text,
        'embedding': keywords_embedding,
        'metadata': {}
    })
    logger.info("  ✅ Keywords embedding generated")
    
    # Embedding 4: Metadata
    logger.info("  ⏳ Generating metadata embedding (4/4)...")
    metadata_text = f"{asset_name}\n\nKeywords: {keywords_text}\n\nType: Audio"
    metadata_embedding = generate_embedding(metadata_text)
    embeddings_data.append({
        'content_type': 'metadata',
        'text': metadata_text,
        'embedding': metadata_embedding,
        'metadata': {
            'asset_name': asset_name,
            'asset_type': 'Audio',
            'keywords': keywords
        }
    })
    logger.info("  ✅ Metadata embedding generated")
    
    # Step 4: Run AWS Comprehend analysis
    logger.info("Step 4/5: Running AWS Comprehend analysis...")
    comprehend_analysis = analyze_transcript_with_comprehend(plain_text)
    logger.info(f"✅ Found {len(comprehend_analysis.get('entities', []))} entities")
    logger.info(f"✅ Found {len(comprehend_analysis.get('key_phrases', []))} key phrases")
    logger.info(f"✅ Sentiment: {comprehend_analysis.get('sentiment', {}).get('Sentiment', 'N/A')}")
    
    # Step 5: Store in database
    logger.info("Step 5/5: Storing in database...")
    
    # Store embeddings
    count = store_embeddings(conn, asset_id, embeddings_data)
    logger.info(f"✅ Stored {count} embeddings")
    
    # Store Comprehend analysis
    store_comprehend_analysis(conn, asset_id, comprehend_analysis)
    logger.info("✅ Stored Comprehend analysis")
    
    conn.close()
    
    # Step 6: Upload to S3
    logger.info("Step 6/6: Uploading files to S3...")
    
    # Upload JSON
    if json_file.exists():
        s3_key = f"audio/{asset_uuid}/{base_name}.json"
        upload_to_s3(str(json_file), s3_key)
    
    # Upload refined TXT
    if refined_txt.exists():
        s3_key = f"audio/{asset_uuid}/{base_name}.txt"
        upload_to_s3(str(refined_txt), s3_key)
    
    # Upload refined SRT
    if refined_srt.exists():
        s3_key = f"audio/{asset_uuid}/{base_name}.srt"
        upload_to_s3(str(refined_srt), s3_key)
    
    logger.info("=" * 80)
    logger.info("✅ POST-PROCESSING COMPLETE")
    logger.info("=" * 80)
    
    return {
        'status': 'success',
        'asset_uuid': asset_uuid,
        'embeddings_count': len(embeddings_data),
        'entities_count': len(comprehend_analysis.get('entities', [])),
        'key_phrases_count': len(comprehend_analysis.get('key_phrases', []))
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Post-process transcript: embeddings + Comprehend + DB + S3'
    )
    parser.add_argument('asset_uuid', help='Asset UUID')
    parser.add_argument('transcript_base_path', help='Base path to transcript files (without extension)')
    
    args = parser.parse_args()
    
    try:
        result = post_process_transcript(args.asset_uuid, args.transcript_base_path)
        print(f"\n✅ SUCCESS: {result}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ FAILED: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
 