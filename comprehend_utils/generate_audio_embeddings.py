"""
Generate Audio Asset Embeddings

This module generates embeddings for audio assets by:
1. Checking if transcript exists in S3, if not:
   a. Download audio from S3 or Intelligence Bank
   b. Transcribe with AWS Transcribe
   c. Generate refined SRT and TXT files
2. Generate Claude summary and keywords  
3. Create 4 embeddings (transcript, summary, keywords, metadata)
4. Run AWS Comprehend analysis
5. Store everything in the database

Usage:
    from comprehend_utils.generate_audio_embeddings import generate_audio_embeddings
    
    result = generate_audio_embeddings('asset-uuid-here')
"""

import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import boto3
from botocore.exceptions import ClientError

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database_navigator import get_connection
from .config import AWS_REGION, get_bedrock_runtime_client, get_comprehend_client
from .comprehend import (
    analyze_transcript_with_comprehend,
    generate_embedding,
    store_comprehend_analysis,
    store_embeddings,
)

# Setup logging
logger = logging.getLogger(__name__)

# AWS clients
s3_client = boto3.client('s3', region_name=AWS_REGION)
S3_BUCKET = 'dam-ramdass-io-assets'


def get_asset_info(asset_uuid: str) -> Optional[Dict[str, Any]]:
    """
    Fetch asset information from database.
    
    Args:
        asset_uuid: UUID of the asset
        
    Returns:
        Dictionary with asset info or None if not found
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, asset_type, description
                FROM assets
                WHERE id = %s AND asset_type = 'Audio'
            """, (asset_uuid,))
            
            row = cur.fetchone()
            if not row:
                return None
            
            return {
                'id': row[0],
                'name': row[1],
                'asset_type': row[2],
                'description': row[3]
            }


def check_transcript_exists_in_s3(asset_uuid: str) -> bool:
    """
    Check if transcript JSON already exists in S3.
    
    Args:
        asset_uuid: UUID of the asset
        
    Returns:
        True if transcript exists, False otherwise
    """
    prefix = f"audio/{asset_uuid}/"
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=prefix,
            MaxKeys=50
        )
        
        if 'Contents' not in response:
            return False
        
        # Look for JSON transcript
        for obj in response['Contents']:
            key = obj['Key']
            if key.endswith('.json') and 'aws_transcribe' not in key.lower():
                # Found transcript, verify it's not empty
                try:
                    resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
                    transcript_json = json.loads(resp['Body'].read())
                    plain_text = transcript_json['results']['transcripts'][0]['transcript']
                    return len(plain_text.strip()) > 10
                except Exception as e:
                    logger.warning(f"Error checking transcript {key}: {e}")
                    return False
        
        return False
        
    except ClientError as e:
        logger.warning(f"Error checking S3 for transcript: {e}")
        return False


def run_transcription_pipeline(asset_uuid: str) -> Optional[str]:
    """
    Run the full transcription pipeline for an asset.
    Uses the existing process_audio_asset.py pipeline.
    
    Args:
        asset_uuid: UUID of the asset
        
    Returns:
        Plain text transcript or None if failed
    """
    logger.info("🎙️ Running full transcription pipeline...")
    
    try:
        # Import the existing pipeline
        from transcribe_pipeline.process_audio_asset import process_audio_asset
        
        # Run the pipeline
        result = process_audio_asset(asset_uuid, verbose=False)
        
        if result.get('status') == 'completed':
            logger.info("✓ Transcription pipeline completed successfully")
            
            # Fetch the generated transcript from S3
            return fetch_transcript_from_s3(asset_uuid)
        else:
            logger.error(f"❌ Transcription pipeline failed: {result.get('errors', [])}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error running transcription pipeline: {e}", exc_info=True)
        return None


def fetch_transcript_from_s3(asset_uuid: str) -> Optional[str]:
    """
    Fetch transcript text from S3.
    
    Args:
        asset_uuid: UUID of the asset
        
    Returns:
        Plain text transcript or None if not found
    """
    prefix = f"audio/{asset_uuid}/"
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            return None
        
        # Find JSON transcript
        for obj in response['Contents']:
            key = obj['Key']
            if key.endswith('.json') and 'aws_transcribe' not in key.lower():
                # Download and parse
                logger.info(f"Fetching transcript from s3://{S3_BUCKET}/{key}")
                resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
                transcript_json = json.loads(resp['Body'].read())
                
                # Extract plain text
                plain_text = transcript_json['results']['transcripts'][0]['transcript']
                
                if plain_text and len(plain_text.strip()) > 10:
                    logger.info(f"✓ Retrieved transcript ({len(plain_text)} chars)")
                    return plain_text
                else:
                    logger.warning(f"Transcript is too short or empty ({len(plain_text)} chars)")
                    return None
        
        logger.warning(f"No valid transcript found in S3 for asset {asset_uuid}")
        return None
        
    except Exception as e:
        logger.error(f"Error fetching transcript from S3: {e}")
        return None


def generate_summary_and_keywords(plain_text: str, asset_name: str) -> Tuple[str, str]:
    """
    Generate summary and keywords using Claude via Bedrock.
    
    Args:
        plain_text: Full transcript text
        asset_name: Name of the asset
        
    Returns:
        Tuple of (summary, keywords)
    """
    bedrock_runtime = get_bedrock_runtime_client()
    
    # Truncate to 100K chars to fit Claude context window
    text_sample = plain_text[:100000]
    
    prompt = f"""Analyze this Ram Dass lecture transcript and provide:

1. A 3-5 paragraph summary capturing the main teachings and themes
2. 10-15 keywords/key phrases that capture the essence

Transcript: "{text_sample}"

Asset name: {asset_name}

Respond ONLY with valid JSON in this exact format:
{{
  "summary": "Your 3-5 paragraph summary here...",
  "keywords": "keyword1, keyword2, keyword3, ..."
}}"""

    logger.info("⏳ Generating summary and keywords with Claude...")
    
    response = bedrock_runtime.converse(
        modelId='us.anthropic.claude-sonnet-4-20250514-v1:0',
        messages=[
            {
                'role': 'user',
                'content': [{'text': prompt}]
            }
        ],
        inferenceConfig={
            'maxTokens': 2000,
            'temperature': 0.5
        }
    )
    
    # Parse response
    result_text = response['output']['message']['content'][0]['text']
    
    # Remove markdown code blocks if present
    json_text = result_text.strip()
    if json_text.startswith('```'):
        lines = json_text.split('\n')
        json_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else json_text
        json_text = json_text.replace('```json', '').replace('```', '').strip()
    
    result = json.loads(json_text)
    summary = result.get('summary', '')
    keywords = result.get('keywords', '')
    
    logger.info(f"✓ Generated summary ({len(summary)} chars) and keywords")
    return (summary, keywords)


def check_existing_embeddings(asset_uuid: str) -> bool:
    """
    Check if embeddings already exist for this asset.
    
    Args:
        asset_uuid: UUID of the asset
        
    Returns:
        True if embeddings exist, False otherwise
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM asset_embeddings
                WHERE resource_id = %s
            """, (asset_uuid,))
            
            row = cur.fetchone()
            count = row[0] if row else 0
            return count > 0


def generate_audio_embeddings(
    asset_uuid: str,
    force: bool = False,
    skip_comprehend: bool = False
) -> Dict[str, Any]:
    """
    Generate and store embeddings for an audio asset.
    
    This creates 4 embeddings:
    1. transcript - Full transcript text (first 8000 chars)
    2. summary - Claude-generated summary
    3. keywords - Claude-generated keywords
    4. metadata - Asset name + keywords + type
    
    Also runs AWS Comprehend analysis (entities, key phrases, sentiment).
    
    Args:
        asset_uuid: UUID of the audio asset
        force: If True, regenerate even if embeddings exist
        skip_comprehend: If True, skip Comprehend analysis
        
    Returns:
        Dictionary with processing results
    """
    logger.info(f"🚀 Starting embedding generation for asset: {asset_uuid}")
    
    result = {
        'asset_uuid': asset_uuid,
        'success': False,
        'embeddings_created': 0,
        'comprehend_entities': 0,
        'comprehend_phrases': 0,
        'comprehend_sentiment': False,
        'errors': []
    }
    
    try:
        # Step 1: Check if embeddings already exist
        if not force and check_existing_embeddings(asset_uuid):
            logger.info("⚠️  Embeddings already exist for this asset. Use force=True to regenerate.")
            result['errors'].append('Embeddings already exist')
            return result
        
        # Step 2: Get asset info
        logger.info("Step 1/6: Fetching asset information...")
        asset = get_asset_info(asset_uuid)
        if not asset:
            error_msg = f"Asset {asset_uuid} not found or is not Audio type"
            logger.error(f"❌ {error_msg}")
            result['errors'].append(error_msg)
            return result
        
        asset_id = asset['id']
        asset_name = asset['name']
        asset_type = asset['asset_type']
        
        logger.info(f"✓ Found asset: {asset_name} (type: {asset_type})")
        
        # Step 3: Get or create transcript
        logger.info("Step 2/6: Checking for existing transcript...")
        
        plain_text = None
        
        # First check if transcript already exists in S3
        if check_transcript_exists_in_s3(asset_uuid):
            logger.info("✓ Transcript exists in S3, fetching...")
            plain_text = fetch_transcript_from_s3(asset_uuid)
        
        # If no transcript exists, run the full transcription pipeline
        if not plain_text:
            logger.info("⚠️  No transcript found, running full transcription pipeline...")
            plain_text = run_transcription_pipeline(asset_uuid)
        
        # Validate we have a transcript
        if not plain_text:
            error_msg = "Failed to obtain transcript (not in S3 and transcription failed)"
            logger.error(f"❌ {error_msg}")
            result['errors'].append(error_msg)
            return result
        
        if len(plain_text.strip()) < 10:
            error_msg = f"Transcript too short ({len(plain_text)} chars). Minimum 10 chars required."
            logger.error(f"❌ {error_msg}")
            result['errors'].append(error_msg)
            return result
        
        logger.info(f"✓ Have valid transcript ({len(plain_text)} chars)")
        
        # Step 4: Generate summary and keywords
        logger.info("Step 3/6: Generating summary and keywords with Claude...")
        summary_text, keywords = generate_summary_and_keywords(plain_text, asset_name)
        
        # Step 5: Create embeddings
        logger.info("Step 4/6: Generating 4 embeddings...")
        embeddings_data = []
        
        # Embedding 1: Transcript
        logger.info("  ⏳ Generating transcript embedding (1/4)...")
        transcript_embedding = generate_embedding(plain_text[:8000])
        embeddings_data.append({
            'content_type': 'transcript',
            'embedding': transcript_embedding,
        })
        
        # Embedding 2: Summary
        logger.info("  ⏳ Generating summary embedding (2/4)...")
        summary_embedding = generate_embedding(summary_text)
        embeddings_data.append({
            'content_type': 'summary',
            'embedding': summary_embedding,
        })
        
        # Embedding 3: Keywords
        logger.info("  ⏳ Generating keywords embedding (3/4)...")
        keywords_embedding = generate_embedding(keywords)
        embeddings_data.append({
            'content_type': 'keywords',
            'embedding': keywords_embedding,
        })
        
        # Embedding 4: Metadata
        logger.info("  ⏳ Generating metadata embedding (4/4)...")
        metadata_text = f"{asset_name}. Keywords: {keywords}. Type: {asset_type}"
        metadata_embedding = generate_embedding(metadata_text)
        embeddings_data.append({
            'content_type': 'metadata',
            'embedding': metadata_embedding,
        })
        
        # Step 6: Store embeddings in database
        logger.info("Step 5/6: Storing embeddings in database...")
        with get_connection() as conn:
            num_stored = store_embeddings(conn, asset_id, embeddings_data)
            result['embeddings_created'] = num_stored
        
        logger.info(f"✓ Stored {num_stored} embeddings")
        
        # Step 7: AWS Comprehend analysis (optional)
        if not skip_comprehend:
            logger.info("Step 6/6: Running AWS Comprehend analysis...")
            comprehend_analysis = analyze_transcript_with_comprehend(plain_text)
            
            with get_connection() as conn:
                counts = store_comprehend_analysis(conn, asset_id, comprehend_analysis)
                result['comprehend_entities'] = counts['entities']
                result['comprehend_phrases'] = counts['key_phrases']
                result['comprehend_sentiment'] = counts['sentiment'] > 0
            
            logger.info(f"✓ Stored {counts['entities']} entities, {counts['key_phrases']} key phrases")
        else:
            logger.info("Step 6/6: Skipping Comprehend analysis (skip_comprehend=True)")
        
        result['success'] = True
        logger.info(f"✅ Successfully generated embeddings for {asset_name}")
        
    except Exception as e:
        error_msg = f"Error processing asset: {str(e)}"
        logger.error(f"❌ {error_msg}", exc_info=True)
        result['errors'].append(error_msg)
    
    return result


if __name__ == '__main__':
    # Setup logging for CLI usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m comprehend_utils.generate_audio_embeddings <asset_uuid> [--force] [--skip-comprehend]")
        sys.exit(1)
    
    asset_uuid = sys.argv[1]
    force = '--force' in sys.argv
    skip_comprehend = '--skip-comprehend' in sys.argv
    
    result = generate_audio_embeddings(asset_uuid, force=force, skip_comprehend=skip_comprehend)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(json.dumps(result, indent=2))
    
    sys.exit(0 if result['success'] else 1)
