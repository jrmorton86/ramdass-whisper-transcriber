#!/usr/bin/env python3
"""
Audio Asset Processing Pipeline

Downloads audio from Intelligence Bank, transcribes with LOCAL WHISPER AI,
generates 3 transcript formats (SRT, formatted .txt, raw JSON), creates embeddings, and
stores everything in the database.

Usage:
    python3 process_audio_asset.py <asset_uuid> [--verbose] [--model base]

Example:
    python3 process_audio_asset.py 550e8400-e29b-41d4-a716-446655440000 --verbose --model medium
"""

import os
import sys
import json
import time
import logging
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

# Add parent directory to path to import backend modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
from botocore.exceptions import ClientError
import psycopg2

# Import local Whisper transcription
from whisper_with_vocab import VocabularyEnhancedTranscriber

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# AWS Configuration
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
S3_BUCKET = os.getenv('S3_BUCKET', 'dam-ramdass-io-assets')

# Initialize AWS clients
s3_client = boto3.client('s3', region_name=AWS_REGION)
transcribe_client = boto3.client('transcribe', region_name=AWS_REGION)
bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)


def get_db_connection():
    """Get database connection using same approach as database_navigator."""
    # Import here to avoid circular dependency
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from database_navigator.db import get_connection
    return get_connection()


def transcribe_with_whisper(audio_path: str, asset_uuid: str, base_filename: str, model_name: str = "base") -> Dict:
    """
    Transcribe audio using local Whisper AI with vocabulary enhancement.
    
    Args:
        audio_path: Path to local audio file
        asset_uuid: Asset UUID for S3 upload path
        base_filename: Base filename for output files
        model_name: Whisper model size (tiny, base, small, medium, large)
    
    Returns:
        dict: Whisper transcription result with segments and metadata
    """
    logger.info(f"🎙️  Transcribing with Whisper AI (model: {model_name})...")
    logger.info(f"   Using vocabulary-enhanced transcription")
    
    # Initialize Whisper transcriber with vocabulary
    transcriber = VocabularyEnhancedTranscriber()
    
    # Transcribe
    result = transcriber.transcribe(
        audio_path,
        model_name=model_name,
        language="en",
        apply_vocab_corrections=True,
        remove_fillers=False
    )
    
    logger.info(f"✅ Transcription complete ({len(result['text'])} chars)")
    
    # Save transcript JSON to temp location (will upload to S3 later)
    temp_json_path = os.path.join(os.path.dirname(audio_path), f"{base_filename}.json")
    with open(temp_json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    # Upload JSON to S3
    json_s3_key = f"audio/{asset_uuid}/{base_filename}.json"
    upload_to_s3(temp_json_path, json_s3_key)
    logger.info(f"✅ Uploaded transcript JSON to S3: {json_s3_key}")
    
    return result


def get_ib_session():
    """
    Authenticate with Intelligence Bank using 2-step process.
    Returns: (session_id, api_v3_url) tuple
    """
    import requests
    from dotenv import load_dotenv
    
    load_dotenv('/srv/config-repo/.env')
    
    ib_email = os.getenv('IB_API_EMAIL')
    ib_password = os.getenv('IB_API_PASSWORD')
    platform = os.getenv('IB_PLATFORM_URL', 'ramdass.intelligencebank.com')
    
    logger.info("Authenticating with Intelligence Bank API...")
    
    try:
        # Step 1: Get API address
        logger.debug("Step 1: Getting API address from platform...")
        step1_response = requests.get(
            f"https://{platform}/v1/auth/app/getYapiAddress",
            timeout=30
        )
        step1_response.raise_for_status()
        step1_data = step1_response.json()
        
        api_v2_url = step1_data.get('content')
        if not api_v2_url:
            raise ValueError(f"Step 1 response missing API URL: {step1_data}")
        
        logger.debug(f"Got API V2 URL: {api_v2_url}")
        
        # Step 2: Login with credentials
        logger.debug("Step 2: Logging in with credentials...")
        form_data = {
            "p70": ib_email,
            "p80": ib_password,
            "p90": platform.replace('https://', '').replace('http://', '')
        }
        
        step2_response = requests.post(
            f"{api_v2_url}/webapp/1.0/login",
            data=form_data,
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            timeout=30
        )
        step2_response.raise_for_status()
        
        result = step2_response.json()
        
        session_id = result.get('sid')
        api_v3_url = result.get('apiV3url')
        
        if not session_id:
            raise ValueError(f"Authentication response missing session ID")
        
        logger.info(f"Successfully authenticated with IB (session: {session_id[:8]}...)")
        return session_id, api_v3_url
        
    except requests.exceptions.RequestException as e:
        logger.error(f"IB authentication failed: {e}")
        raise RuntimeError(f"Failed to authenticate with Intelligence Bank: {e}")


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to be safe for filesystem use.
    From ib_download_resource.py
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename safe for use on most filesystems
    """
    import re
    # Remove or replace characters that are problematic on various filesystems
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    
    # Limit filename length (most filesystems support 255 bytes)
    max_length = 200
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:max_length - len(ext)] + ext
    
    # Ensure we have something left
    if not sanitized or sanitized == '':
        sanitized = 'download'
    
    return sanitized


def extract_filename_from_content_disposition(content_disposition: str) -> Optional[str]:
    """
    Extract filename from Content-Disposition header.
    From ib_download_resource.py
    
    Args:
        content_disposition: Content-Disposition header value
    
    Returns:
        Extracted filename or None
    """
    import re
    from urllib.parse import unquote
    
    if not content_disposition:
        return None
    
    # Try RFC 5987 format first: filename*=UTF-8''filename.ext
    rfc5987_match = re.search(r"filename\*=(?:UTF-8|utf-8)''([^;]+)", content_disposition)
    if rfc5987_match:
        filename = unquote(rfc5987_match.group(1))
        return sanitize_filename(filename)
    
    # Try standard format: filename="filename.ext" or filename=filename.ext
    standard_match = re.search(r'filename="?([^";]+)"?', content_disposition, re.IGNORECASE)
    if standard_match:
        filename = standard_match.group(1).strip('"')
        filename = unquote(filename)
        return sanitize_filename(filename)
    
    return None


def check_audio_in_s3(asset_uuid: str) -> Optional[Tuple[str, str, int]]:
    """
    Check if audio file already exists in S3 and get its size without downloading.
    
    Args:
        asset_uuid: UUID of the asset
        
    Returns:
        Tuple of (s3_uri, filename, size_bytes) if file exists, None otherwise
    """
    prefix = f"audio/{asset_uuid}/"
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=prefix,
            MaxKeys=50
        )
        
        if 'Contents' not in response:
            return None
        
        # Look for audio files (not transcripts)
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma', '.mp4', '.mov', '.avi', '.mkv'}
        
        for obj in response['Contents']:
            key = obj['Key']
            filename = os.path.basename(key)
            size_bytes = obj['Size']
            
            # Skip transcript files
            if any(key.endswith(ext) for ext in ['.json', '.srt', '.txt']):
                continue
            
            # Check if it's an audio file
            _, ext = os.path.splitext(filename.lower())
            if ext in audio_extensions:
                s3_uri = f"s3://{S3_BUCKET}/{key}"
                size_mb = size_bytes / (1024 * 1024)
                logger.info(f"Found existing audio file in S3: {s3_uri} ({size_mb:.1f} MB)")
                return s3_uri, filename, size_bytes
        
        return None
        
    except ClientError as e:
        logger.debug(f"Error checking S3 for audio file: {e}")
        return None


def download_from_s3(s3_uri: str, output_dir: str) -> str:
    """
    Download file from S3 to local directory.
    
    Args:
        s3_uri: S3 URI (s3://bucket/key)
        output_dir: Local directory to save file
        
    Returns:
        Local file path
    """
    # Parse S3 URI
    parts = s3_uri.replace('s3://', '').split('/', 1)
    bucket = parts[0]
    key = parts[1]
    filename = os.path.basename(key)
    local_path = os.path.join(output_dir, filename)
    
    logger.info(f"Downloading from S3: {s3_uri}")
    s3_client.download_file(bucket, key, local_path)
    logger.info(f"Downloaded to {local_path}")
    
    return local_path


def download_audio_from_ib(asset_uuid: str, output_dir: str) -> Tuple[str, str]:
    """
    Download audio file from Intelligence Bank.
    
    Args:
        asset_uuid: UUID of the asset in IB
        output_dir: Directory to save the downloaded file
        
    Returns:
        Tuple of (local_file_path, original_filename)
    """
    import requests
    
    logger.info(f"Downloading audio asset {asset_uuid} from Intelligence Bank...")
    
    session_id, api_v3_url = get_ib_session()
    
    # Get client ID from environment or use default
    client_id = os.getenv('IB_CLIENT_ID', 'vJgXP3')
    
    # IB API expects UUID without hyphens
    ib_resource_id = asset_uuid.replace('-', '')
    
    # Build download URL using correct IB API format
    resource_url = f"{api_v3_url}/api/3.0.0/{client_id}/resource/{ib_resource_id}?action=download&sid={session_id}"
    headers = {'sid': session_id}
    
    logger.debug(f"Downloading from: {resource_url}")
    
    response = requests.get(resource_url, headers=headers, stream=True, timeout=300)
    response.raise_for_status()
    
    # Extract filename using comprehensive logic from ib_download_resource.py
    content_disposition = response.headers.get('Content-Disposition', '')
    filename = extract_filename_from_content_disposition(content_disposition)
    
    if not filename:
        # Fallback: use UUID with extension from Content-Type
        content_type = response.headers.get('Content-Type', 'audio/mpeg')
        ext = content_type.split('/')[-1].split(';')[0]
        filename = f"{asset_uuid}.{ext}"
    
    logger.info(f"Downloaded filename: {filename}")
    local_path = os.path.join(output_dir, filename)
    
    # Download with progress logging
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0 and downloaded % (1024 * 1024) == 0:  # Log every MB
                progress = (downloaded / total_size) * 100
                logger.debug(f"Download progress: {progress:.1f}%")
    
    logger.info(f"Downloaded to {local_path} ({downloaded} bytes)")
    return local_path, filename


def check_s3_file_exists(s3_key: str) -> Optional[Dict]:
    """
    Check if file exists in S3 and get its metadata.
    
    Args:
        s3_key: S3 key (path within bucket)
        
    Returns:
        Dict with file metadata if exists (ContentLength, LastModified, ContentType), None otherwise
    """
    try:
        response = s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
        return {
            'size': response['ContentLength'],
            'last_modified': response['LastModified'],
            'content_type': response.get('ContentType', 'unknown')
        }
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return None
        else:
            logger.warning(f"Error checking S3 file {s3_key}: {e}")
            return None


def upload_to_s3(local_path: str, s3_key: str, skip_if_exists: bool = True) -> str:
    """
    Upload file to S3.
    
    Args:
        local_path: Path to local file
        s3_key: S3 key (path within bucket)
        skip_if_exists: If True, skip upload if file already exists in S3
        
    Returns:
        S3 URI (s3://bucket/key)
    """
    s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
    
    # Check if file already exists
    if skip_if_exists:
        existing = check_s3_file_exists(s3_key)
        if existing:
            logger.info(f"File already exists in S3: {s3_uri} ({existing['size']} bytes)")
            return s3_uri
    
    logger.info(f"Uploading {local_path} to {s3_uri}...")
    
    s3_client.upload_file(local_path, S3_BUCKET, s3_key)
    
    logger.info(f"Uploaded to {s3_uri}")
    return s3_uri


def is_video_file(file_path: str) -> bool:
    """
    Check if a file is a video file based on extension and MIME type.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is a video
    """
    import mimetypes
    
    # Video extensions (including common ones miscategorized as audio)
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mpg', '.mpeg']
    
    # Check extension
    ext = os.path.splitext(file_path)[1].lower()
    if ext in video_extensions:
        return True
    
    # Check MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith('video/'):
        return True
    
    return False


def compress_to_mp3(input_path: str, output_dir: str, bitrate: str = '128k') -> str:
    """
    Convert audio/video file to compressed MP3 using ffmpeg.
    Handles both audio files and video files (extracts audio track).
    
    Args:
        input_path: Path to input file (audio or video)
        output_dir: Directory to save compressed file
        bitrate: MP3 bitrate (default: 128k for good quality/size balance)
        
    Returns:
        Path to compressed MP3 file
    """
    import subprocess
    
    # Get base filename and create MP3 output path
    base_filename = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base_filename}_compressed.mp3")
    
    logger.info(f"Compressing to MP3 (bitrate: {bitrate})...")
    logger.info(f"  Input:  {input_path}")
    logger.info(f"  Output: {output_path}")
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("ffmpeg not found. Please install: sudo apt-get install ffmpeg")
    
    # FFmpeg command to convert to MP3
    # -i: input file
    # -vn: no video (audio only)
    # -acodec libmp3lame: use LAME MP3 encoder
    # -ab: audio bitrate
    # -ar: audio sample rate (44100 Hz = CD quality)
    # -ac: audio channels (2 = stereo)
    # -y: overwrite output file if exists
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-vn',  # No video
        '-acodec', 'libmp3lame',
        '-ab', bitrate,
        '-ar', '44100',
        '-ac', '2',
        '-y',
        output_path
    ]
    
    try:
        # Run ffmpeg with progress output suppressed
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Get file sizes for comparison
        input_size = os.path.getsize(input_path)
        output_size = os.path.getsize(output_path)
        compression_ratio = (1 - output_size / input_size) * 100
        
        logger.info(f"✅ Compression complete!")
        logger.info(f"  Original size: {input_size / (1024**3):.2f} GB")
        logger.info(f"  Compressed size: {output_size / (1024**3):.2f} GB")
        logger.info(f"  Reduction: {compression_ratio:.1f}%")
        
        return output_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr}")
        raise RuntimeError(f"Failed to compress audio: {e.stderr}")


def start_transcription_job(asset_uuid: str, audio_s3_uri: str, base_filename: str) -> str:
    """
    Start AWS Transcribe job with speaker diarization, custom vocabulary, and subtitle generation.
    
    NOTE: Using en-US with custom vocabulary (RamDassVocabulary) instead of multi-language detection
    because AWS Transcribe doesn't support custom vocabularies with IdentifyMultipleLanguages.
    Most Ram Dass content is in English with occasional Sanskrit/Hindi terms which the custom
    vocabulary handles well.
    
    Args:
        asset_uuid: UUID of the asset
        audio_s3_uri: S3 URI of the audio file
        base_filename: Base filename (without extension) for the transcript output
        
    Returns:
        Job name
    """
    job_name = f"audio-{asset_uuid}-{int(time.time())}"
    
    logger.info(f"Starting transcription job {job_name} with advanced features...")
    logger.info(f"  - Language: English (en-US)")
    logger.info(f"  - Speaker diarization enabled (max 10 speakers)")
    logger.info(f"  - Custom vocabulary: RamDassVocabulary")
    logger.info(f"  - Vocabulary filter: RamDassVocabularyFilter")
    logger.info(f"  - SRT subtitle generation enabled")
    
    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': audio_s3_uri},
        MediaFormat='mp3',  # Adjust based on actual format
        LanguageCode='en-US',
        Settings={
            'ShowSpeakerLabels': True,
            'MaxSpeakerLabels': 10,
            'VocabularyName': 'RamDassVocabulary',
            'VocabularyFilterName': 'RamDassVocabularyFilter',
            'VocabularyFilterMethod': 'mask'
        },
        Subtitles={
            'Formats': ['srt'],
            'OutputStartIndex': 1
        },
        OutputBucketName=S3_BUCKET,
        OutputKey=f"audio/{asset_uuid}/{base_filename}.json"
    )
    
    logger.info(f"Transcription job started: {job_name}")
    return job_name


def wait_for_transcription(job_name: str, asset_uuid: str, base_filename: str, timeout: int = 3600) -> Optional[Dict]:
    """
    Wait for transcription JSON to appear in S3 by polling S3 directly.
    This is faster than waiting for job status COMPLETED since the transcript
    file is written to S3 before AWS updates the job metadata.
    
    Args:
        job_name: Transcription job name
        asset_uuid: UUID of the asset (for S3 key)
        base_filename: Base filename (without extension) for the transcript output
        timeout: Maximum time to wait in seconds (default: 1 hour)
        
    Returns:
        Transcription result JSON
    """
    logger.info(f"Polling S3 for transcription output from job {job_name}...")
    
    # S3 key where Transcribe writes the output
    transcript_s3_key = f"audio/{asset_uuid}/{base_filename}.json"
    
    poll_interval = 15  # Poll every 15 seconds
    elapsed = 0
    last_status = None
    heartbeat_interval = 30  # Log heartbeat every 30 seconds
    last_heartbeat = 0
    
    while elapsed < timeout:
        time.sleep(poll_interval)
        elapsed += poll_interval
        
        # Heartbeat logging to prevent watchdog timeout
        if elapsed - last_heartbeat >= heartbeat_interval:
            logger.info(f"⏳ Still waiting for transcription... (elapsed: {elapsed}s, timeout: {timeout}s)")
            last_heartbeat = elapsed
        
        # First check if file exists in S3
        try:
            logger.debug(f"Checking S3 for {transcript_s3_key}...")
            response = s3_client.head_object(Bucket=S3_BUCKET, Key=transcript_s3_key)
            
            # File exists! Download it
            logger.info(f"✅ Transcript file found in S3 after {elapsed}s")
            
            # Download the transcript JSON from S3
            s3_response = s3_client.get_object(Bucket=S3_BUCKET, Key=transcript_s3_key)
            transcript_json = json.loads(s3_response['Body'].read())
            
            # Also log the Transcribe job status for reference (non-blocking)
            try:
                job_response = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
                job_status = job_response['TranscriptionJob']['TranscriptionJobStatus']
                logger.info(f"📊 Transcribe job status: {job_status} (transcript already retrieved from S3)")
            except Exception as e:
                logger.debug(f"Could not get job status: {e}")
            
            return transcript_json
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                # File not in S3 yet, check Transcribe job status for progress/errors
                try:
                    job_response = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
                    status = job_response['TranscriptionJob']['TranscriptionJobStatus']
                    
                    if status == 'FAILED':
                        failure_reason = job_response['TranscriptionJob'].get('FailureReason', 'Unknown')
                        
                        # Check if it's a parse failure (corrupted/invalid audio file)
                        if 'Failed to parse audio file' in failure_reason or 'parse' in failure_reason.lower():
                            logger.error(f"❌ Transcription job {job_name} failed due to corrupted/invalid audio: {failure_reason}")
                            # Return None to signal this asset should be skipped
                            return None
                        
                        raise Exception(f"Transcription job {job_name} failed: {failure_reason}")
                    
                    # Log status change
                    if status != last_status:
                        logger.info(f"⏳ Transcription status: {status}, polling S3 for output (elapsed: {elapsed}s)")
                        last_status = status
                    
                except Exception as transcribe_error:
                    # Check if the error message indicates a failed job
                    error_msg = str(transcribe_error)
                    if 'failed:' in error_msg.lower() and 'parse' in error_msg.lower():
                        logger.error(f"❌ Transcription job {job_name} failed due to corrupted/invalid audio")
                        # Return None to signal this asset should be skipped
                        return None
                    logger.warning(f"Could not check Transcribe job status: {transcribe_error}")
            else:
                # Some other S3 error
                logger.warning(f"S3 error while checking for transcript: {e}")
    
    # Timeout reached
    raise TimeoutError(f"Transcription output not found in S3 after {timeout} seconds")


def transcribe_to_srt(transcript_json: Dict) -> str:
    """
    Convert AWS Transcribe JSON to SRT format with timecodes.
    
    Args:
        transcript_json: AWS Transcribe output JSON
        
    Returns:
        SRT formatted string
    """
    logger.info("Converting transcript to SRT format...")
    
    items = transcript_json['results']['items']
    speaker_segments = transcript_json['results'].get('speaker_labels', {}).get('segments', [])
    
    srt_entries = []
    entry_num = 1
    
    for segment in speaker_segments:
        speaker = segment.get('speaker_label', 'SPEAKER_00')
        start_time = float(segment['start_time'])
        end_time = float(segment['end_time'])
        
        # Get words for this segment
        segment_items = segment.get('items', [])
        words = []
        
        for item in segment_items:
            # Find the word in items
            for word_item in items:
                if word_item.get('start_time') == item.get('start_time'):
                    words.append(word_item.get('alternatives', [{}])[0].get('content', ''))
                    break
        
        text = ' '.join(words)
        
        # Format timestamps as SRT (HH:MM:SS,mmm)
        start_srt = format_srt_timestamp(start_time)
        end_srt = format_srt_timestamp(end_time)
        
        # Create SRT entry
        srt_entry = f"{entry_num}\n{start_srt} --> {end_srt}\n{speaker}: {text}\n"
        srt_entries.append(srt_entry)
        entry_num += 1
    
    return '\n'.join(srt_entries)


def format_srt_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        SRT formatted timestamp
    """
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def transcribe_to_formatted_text(transcript_json: Dict) -> str:
    """
    Convert AWS Transcribe JSON to formatted text with intelligent paragraph breaks.
    Uses statistical analysis of sentence timing delays to determine natural paragraph boundaries.
    
    Args:
        transcript_json: AWS Transcribe output JSON
        
    Returns:
        Formatted text with speaker labels and paragraph breaks
    """
    logger.info("Converting transcript to formatted text with intelligent paragraphing...")
    
    # Check if transcript has valid content
    if not transcript_json or not transcript_json.get('results'):
        logger.warning("⚠️  Transcript is empty or has no results - likely no speech content in audio")
        return None
    
    items = transcript_json['results'].get('items', [])
    if not items:
        logger.warning("⚠️  No items found in transcript - audio contains no speech")
        return None
    
    speaker_segments = transcript_json['results'].get('speaker_labels', {}).get('segments', [])
    
    # Step 1: Build sentences with metadata
    sentences = []
    current_sentence = []
    current_start = None
    current_end = None
    current_speaker = None
    
    for item in items:
        if item['type'] == 'pronunciation':
            word = item['alternatives'][0]['content']
            start_time = float(item.get('start_time', 0))
            end_time = float(item.get('end_time', 0))
            
            # Find speaker for this timestamp
            speaker = 'SPEAKER_00'
            for segment in speaker_segments:
                seg_start = float(segment['start_time'])
                seg_end = float(segment['end_time'])
                if seg_start <= start_time <= seg_end:
                    speaker = segment['speaker_label']
                    break
            
            if current_start is None:
                current_start = start_time
            current_end = end_time
            current_speaker = speaker
            current_sentence.append(word)
            
        elif item['type'] == 'punctuation':
            punct = item['alternatives'][0]['content']
            current_sentence.append(punct)
            
            # Sentence boundary markers
            if punct in ['.', '!', '?']:
                if current_sentence:
                    # Join words with spaces, then fix punctuation spacing
                    text = ' '.join(current_sentence)
                    text = text.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?')
                    text = text.replace(' ;', ';').replace(' :', ':')
                    
                    sentences.append({
                        'text': text,
                        'start_time': current_start,
                        'end_time': current_end,
                        'speaker': current_speaker,
                        'word_count': len([w for w in current_sentence if w not in ['.', ',', '!', '?', ';', ':']])
                    })
                    current_sentence = []
                    current_start = None
                    current_end = None
    
    # Add any remaining sentence
    if current_sentence:
        # Join words with spaces, then fix punctuation spacing
        text = ' '.join(current_sentence)
        text = text.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?')
        text = text.replace(' ;', ';').replace(' :', ':')
        
        sentences.append({
            'text': text,
            'start_time': current_start or 0,
            'end_time': current_end or 0,
            'speaker': current_speaker or 'SPEAKER_00',
            'word_count': len([w for w in current_sentence if w not in ['.', ',', '!', '?', ';', ':']])
        })
    
    if not sentences:
        return ""
    
    # Step 2: Calculate start delays between sentences
    for i in range(1, len(sentences)):
        delay = sentences[i]['start_time'] - sentences[i-1]['end_time']
        sentences[i]['start_delay'] = max(0, delay)
    sentences[0]['start_delay'] = 0
    
    # Step 3: Statistical analysis of delays (using median + IQR approach)
    delays = [s['start_delay'] for s in sentences[1:] if s['start_delay'] > 0]
    
    if delays:
        delays_sorted = sorted(delays)
        n = len(delays_sorted)
        median = delays_sorted[n // 2]
        q1 = delays_sorted[n // 4]
        q3 = delays_sorted[3 * n // 4]
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        
        # Use median as threshold (adjustable strategy)
        delay_threshold = max(median, 1.0)  # At least 1 second
        logger.debug(f"Delay threshold: {delay_threshold:.2f}s (median: {median:.2f}s, upper_fence: {upper_fence:.2f}s)")
    else:
        delay_threshold = 2.0  # Default fallback
    
    # Step 4: Mark paragraph breaks based on delays and speaker changes
    for i, sentence in enumerate(sentences):
        if i == 0:
            sentence['is_paragraph_start'] = True
        elif sentence['speaker'] != sentences[i-1]['speaker']:
            sentence['is_paragraph_start'] = True
        elif sentence['start_delay'] > delay_threshold:
            sentence['is_paragraph_start'] = True
        else:
            sentence['is_paragraph_start'] = False
    
    # Step 5: Build initial paragraphs
    paragraphs = []
    current_para_sentences = []
    current_para_speaker = None
    
    for sentence in sentences:
        if sentence['is_paragraph_start'] and current_para_sentences:
            # Save current paragraph
            paragraphs.append({
                'speaker': current_para_speaker,
                'sentences': current_para_sentences[:],
                'word_count': sum(s['word_count'] for s in current_para_sentences)
            })
            current_para_sentences = []
        
        current_para_speaker = sentence['speaker']
        current_para_sentences.append(sentence)
    
    # Add final paragraph
    if current_para_sentences:
        paragraphs.append({
            'speaker': current_para_speaker,
            'sentences': current_para_sentences[:],
            'word_count': sum(s['word_count'] for s in current_para_sentences)
        })
    
    # Step 6: Split overly long paragraphs
    word_counts = [p['word_count'] for p in paragraphs]
    if word_counts:
        wc_sorted = sorted(word_counts)
        n = len(wc_sorted)
        q3_words = wc_sorted[3 * n // 4] if n >= 4 else 150
        iqr_words = wc_sorted[3 * n // 4] - wc_sorted[n // 4] if n >= 4 else 50
        max_words = q3_words + 1.5 * iqr_words
        max_words = max(max_words, 200)  # At least 200 words
        
        logger.debug(f"Max words per paragraph: {max_words:.0f}")
        
        split_paragraphs = []
        for para in paragraphs:
            if para['word_count'] > max_words and len(para['sentences']) > 1:
                # Find sentence with max delay within paragraph
                max_delay_idx = 0
                max_delay = 0
                for i, sent in enumerate(para['sentences'][1:], 1):
                    if sent['start_delay'] > max_delay:
                        max_delay = sent['start_delay']
                        max_delay_idx = i
                
                if max_delay_idx > 0:
                    # Split at that point
                    split_paragraphs.append({
                        'speaker': para['speaker'],
                        'sentences': para['sentences'][:max_delay_idx],
                        'word_count': sum(s['word_count'] for s in para['sentences'][:max_delay_idx])
                    })
                    split_paragraphs.append({
                        'speaker': para['speaker'],
                        'sentences': para['sentences'][max_delay_idx:],
                        'word_count': sum(s['word_count'] for s in para['sentences'][max_delay_idx:])
                    })
                else:
                    split_paragraphs.append(para)
            else:
                split_paragraphs.append(para)
        
        paragraphs = split_paragraphs
    
    # Step 7: Format output
    output_lines = []
    current_speaker_label = None
    
    for para in paragraphs:
        speaker = para['speaker']
        speaker_num = int(speaker.split('_')[-1]) + 1
        speaker_label = f"Speaker {speaker_num}"
        
        # Add speaker label if changed
        if speaker_label != current_speaker_label:
            if output_lines:  # Add blank line before new speaker
                output_lines.append("")
            output_lines.append(f"{speaker_label}:")
            current_speaker_label = speaker_label
        
        # Build paragraph text
        para_text = ' '.join(s['text'] for s in para['sentences'])
        output_lines.append(para_text)
        output_lines.append("")  # Blank line after paragraph
    
    return '\n'.join(output_lines)


def extract_plain_text(transcript_json: Dict) -> str:
    """
    Extract plain text from transcript (for embeddings).
    
    Args:
        transcript_json: AWS Transcribe output JSON
        
    Returns:
        Plain text string
    """
    return transcript_json['results']['transcripts'][0]['transcript']


def generate_summary_and_keywords(plain_text: str, asset_name: str) -> Tuple[str, str]:
    """
    Use Claude to generate summary and keywords for the audio.
    
    Args:
        plain_text: Plain text transcript
        asset_name: Name of the asset
        
    Returns:
        Tuple of (summary, keywords)
    """
    logger.info("Generating summary and keywords with Claude...")
    
    prompt = f"""Analyze this audio transcript titled "{asset_name}".

Transcript:
{plain_text[:10000]}  # Limit to first 10k chars for token efficiency

Provide:
1. A comprehensive summary (3-5 paragraphs) covering main themes and key points
2. A list of 10-15 relevant keywords and topics

Format your response as JSON:
{{
  "summary": "...",
  "keywords": "keyword1, keyword2, keyword3, ..."
}}"""
    
    # Use converse API with Claude Sonnet 4.5 (extended context by default)
    response = bedrock_runtime.converse(
        modelId='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
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
    
    # Parse converse API response format
    result_text = response['output']['message']['content'][0]['text']
    
    # Parse JSON response - handle markdown code blocks
    try:
        # Remove markdown code blocks if present
        json_text = result_text.strip()
        if json_text.startswith('```'):
            # Extract JSON from code block
            lines = json_text.split('\n')
            json_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else json_text
            json_text = json_text.replace('```json', '').replace('```', '').strip()
        
        result = json.loads(json_text)
        summary = result.get('summary', '')
        keywords = result.get('keywords', '')
    except json.JSONDecodeError as e:
        logger.warning(f"Claude didn't return valid JSON: {e}, parsing as plain text")
        # Fallback if Claude doesn't return valid JSON
        summary = result_text
        keywords = ''
    
    logger.info("Summary and keywords generated")
    return summary, keywords


def analyze_transcript_with_comprehend(plain_text: str) -> Dict:
    """
    Analyze transcript with AWS Comprehend to extract entities and key phrases.
    
    AWS Comprehend has a 5000 byte limit for synchronous operations,
    so we'll analyze in chunks and aggregate results.
    
    Args:
        plain_text: Plain text transcript
        
    Returns:
        Dictionary with entities and key phrases
    """
    logger.info("Analyzing transcript with AWS Comprehend...")
    
    # Initialize Comprehend client
    comprehend_client = boto3.client('comprehend', region_name=AWS_REGION)
    
    # Analyze in 4500 char chunks (leave buffer for multi-byte characters)
    chunk_size = 4500
    all_entities = []
    all_key_phrases = []
    sentiment_scores = []
    
    # Process text in chunks
    num_chunks = (len(plain_text) + chunk_size - 1) // chunk_size
    logger.info(f"  Processing {num_chunks} chunks for Comprehend analysis...")
    
    for i in range(0, len(plain_text), chunk_size):
        chunk = plain_text[i:i + chunk_size]
        chunk_num = i // chunk_size + 1
        
        # Heartbeat logging for long operations
        if chunk_num % 5 == 0:
            logger.info(f"⏳ Processing Comprehend chunk {chunk_num}/{num_chunks}...")
        
        try:
            # Detect entities
            entities_response = comprehend_client.detect_entities(
                Text=chunk,
                LanguageCode='en'
            )
            all_entities.extend(entities_response['Entities'])
            
            # Extract key phrases
            key_phrases_response = comprehend_client.detect_key_phrases(
                Text=chunk,
                LanguageCode='en'
            )
            all_key_phrases.extend(key_phrases_response['KeyPhrases'])
            
            # Get sentiment for first chunk only (representative)
            if i == 0:
                sentiment_response = comprehend_client.detect_sentiment(
                    Text=chunk,
                    LanguageCode='en'
                )
                sentiment_scores.append({
                    'sentiment': sentiment_response['Sentiment'],
                    'scores': sentiment_response['SentimentScore']
                })
        
        except Exception as e:
            logger.warning(f"  Error analyzing chunk {i//chunk_size + 1}: {e}")
            continue
    
    # Deduplicate and rank entities by confidence
    entities_dict = {}
    for entity in all_entities:
        key = (entity['Text'].lower(), entity['Type'])
        if key not in entities_dict or entity['Score'] > entities_dict[key]['Score']:
            entities_dict[key] = entity
    
    ranked_entities = sorted(entities_dict.values(), key=lambda x: x['Score'], reverse=True)
    
    # Deduplicate and rank key phrases
    phrases_dict = {}
    for phrase in all_key_phrases:
        key = phrase['Text'].lower()
        if key not in phrases_dict or phrase['Score'] > phrases_dict[key]['Score']:
            phrases_dict[key] = phrase
    
    ranked_phrases = sorted(phrases_dict.values(), key=lambda x: x['Score'], reverse=True)
    
    logger.info(f"  Found {len(ranked_entities)} unique entities")
    logger.info(f"  Found {len(ranked_phrases)} unique key phrases")
    
    # Return top results
    return {
        'entities': ranked_entities[:100],  # Top 100 entities
        'key_phrases': ranked_phrases[:50],  # Top 50 key phrases
        'sentiment': sentiment_scores[0] if sentiment_scores else None
    }


def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding vector using Amazon Titan Embeddings.
    
    Args:
        text: Text to embed
        
    Returns:
        List of floats (1024-dimensional vector)
    """
    response = bedrock_runtime.invoke_model(
        modelId='amazon.titan-embed-text-v2:0',
        body=json.dumps({
            'inputText': text[:8000]  # Titan limit
        })
    )
    
    response_body = json.loads(response['body'].read())
    return response_body['embedding']


def store_comprehend_analysis(conn, asset_id: str, comprehend_analysis: Dict) -> Dict:
    """
    Store AWS Comprehend analysis results in the database.
    
    Args:
        conn: Database connection
        asset_id: UUID of the asset
        comprehend_analysis: Dictionary with entities, key_phrases, and sentiment
        
    Returns:
        Dictionary with counts of stored items
    """
    cursor = conn.cursor()
    
    counts = {'entities': 0, 'key_phrases': 0, 'sentiment': 0}
    
    try:
        # 1. Store entities
        entities = comprehend_analysis.get('entities', [])
        if entities:
            # Delete existing entities for this asset
            cursor.execute("DELETE FROM comprehend_entities WHERE resource_id = %s", (asset_id,))
            
            # Insert new entities
            for entity in entities:
                cursor.execute("""
                    INSERT INTO comprehend_entities 
                    (resource_id, entity_text, entity_type, confidence_score, begin_offset, end_offset)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    asset_id,
                    entity['Text'],
                    entity['Type'],
                    entity['Score'],
                    entity.get('BeginOffset'),
                    entity.get('EndOffset')
                ))
                counts['entities'] += 1
        
        # 2. Store key phrases
        key_phrases = comprehend_analysis.get('key_phrases', [])
        if key_phrases:
            # Delete existing key phrases for this asset
            cursor.execute("DELETE FROM comprehend_key_phrases WHERE resource_id = %s", (asset_id,))
            
            # Insert new key phrases
            for phrase in key_phrases:
                cursor.execute("""
                    INSERT INTO comprehend_key_phrases 
                    (resource_id, phrase_text, confidence_score, begin_offset, end_offset)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    asset_id,
                    phrase['Text'],
                    phrase['Score'],
                    phrase.get('BeginOffset'),
                    phrase.get('EndOffset')
                ))
                counts['key_phrases'] += 1
        
        # 3. Store sentiment
        sentiment = comprehend_analysis.get('sentiment')
        if sentiment:
            # Delete existing sentiment for this asset
            cursor.execute("DELETE FROM comprehend_sentiment WHERE resource_id = %s", (asset_id,))
            
            # Insert new sentiment
            cursor.execute("""
                INSERT INTO comprehend_sentiment 
                (resource_id, sentiment, positive_score, negative_score, neutral_score, mixed_score)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                asset_id,
                sentiment['sentiment'],
                sentiment['scores']['Positive'],
                sentiment['scores']['Negative'],
                sentiment['scores']['Neutral'],
                sentiment['scores']['Mixed']
            ))
            counts['sentiment'] = 1
        
        conn.commit()
        logger.info(f"Stored Comprehend analysis: {counts['entities']} entities, {counts['key_phrases']} key phrases, {counts['sentiment']} sentiment")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing Comprehend analysis: {e}")
        raise
    
    finally:
        cursor.close()
    
    return counts


def store_embeddings(conn, asset_id: str, embeddings_data: List[Dict]) -> int:
    """
    Store embeddings in the database.
    
    Args:
        conn: Database connection
        asset_id: UUID of the asset
        embeddings_data: List of embedding dictionaries with keys:
            - content_type: Type of content (e.g., 'transcript', 'summary')
            - text: The text that was embedded
            - embedding: The vector embedding
            - metadata: Additional JSONB metadata
            
    Returns:
        Number of embeddings stored
    """
    cursor = conn.cursor()
    
    for data in embeddings_data:
        cursor.execute("""
            INSERT INTO asset_embeddings 
            (resource_id, content_type, embedding, embedding_model, created_at, updated_at)
            VALUES (%s, %s, %s, %s, NOW(), NOW())
            ON CONFLICT (resource_id, content_type, embedding_model)
            DO UPDATE SET
                embedding = EXCLUDED.embedding,
                updated_at = NOW()
        """, (
            asset_id,
            data['content_type'],
            data['embedding'],
            'amazon.titan-embed-text-v2:0'
        ))
    
    conn.commit()
    logger.info(f"Stored {len(embeddings_data)} embeddings for asset {asset_id}")
    
    return len(embeddings_data)


def process_audio_asset(asset_uuid: str, verbose: bool = False) -> Dict:
    """
    Main processing function for audio assets.
    
    Args:
        asset_uuid: UUID of the asset to process (with or without hyphens)
        verbose: Enable verbose logging
        
    Returns:
        Processing summary dictionary
    """
    # Strip hyphens from input UUID for consistent formatting
    asset_uuid = asset_uuid.replace('-', '')
    
    # Format as standard UUID with hyphens for database operations
    # Format: 8-4-4-4-12
    if len(asset_uuid) == 32:
        asset_uuid = f"{asset_uuid[0:8]}-{asset_uuid[8:12]}-{asset_uuid[12:16]}-{asset_uuid[16:20]}-{asset_uuid[20:32]}"
    
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info(f"=== Processing Audio Asset {asset_uuid} ===")
    
    processing_start = time.time()
    summary = {
        'asset_uuid': asset_uuid,
        'status': 'processing',
        'steps_completed': [],
        'errors': []
    }
    
    try:
        # Connect to database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get asset info
        cursor.execute("""
            SELECT id, name, asset_type 
            FROM assets 
            WHERE id = %s
        """, (asset_uuid,))
        
        asset_row = cursor.fetchone()
        if not asset_row:
            raise ValueError(f"Asset {asset_uuid} not found in database")
        
        asset_id, asset_name, asset_type = asset_row
        logger.info(f"Found asset: {asset_name} (type: {asset_type})")
        
        # Create temp directory - will be automatically cleaned up even on exception
        temp_dir = tempfile.mkdtemp(prefix='audio_processing_')
        
        try:
            # Step 1: Check S3 first, then download from IB if needed
            logger.info("Step 1/7: Checking S3 for existing audio file...")
            s3_check = check_audio_in_s3(asset_uuid)
            
            if s3_check:
                # File exists in S3, get metadata without downloading yet
                audio_s3_uri, filename, file_size = s3_check
                logger.info(f"Using existing S3 file: {audio_s3_uri}")
                summary['audio_source'] = 's3'
                summary['audio_s3_uri'] = audio_s3_uri
            else:
                # File not in S3, download from IB
                logger.info("No existing S3 file found, downloading from Intelligence Bank...")
                local_path, filename = download_audio_from_ib(asset_id, temp_dir)
                file_size = os.path.getsize(local_path)
                summary['audio_source'] = 'intelligence_bank'
                summary['steps_completed'].append('download_from_ib')
                
                # Upload original to S3 for future use
                logger.info("Uploading original to S3 for future reuse...")
                audio_s3_key = f"audio/{asset_uuid}/{filename}"
                audio_s3_uri = upload_to_s3(local_path, audio_s3_key)
                summary['audio_s3_uri'] = audio_s3_uri
                summary['steps_completed'].append('s3_upload')
            
            # Step 1b: Check if compression needed (large files or video files)
            # Note: We check file_size from S3 metadata to avoid downloading if not needed
            file_size_gb = file_size / (1024**3)
            
            # Check if it's a video file by extension
            _, ext = os.path.splitext(filename.lower())
            video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mpg', '.mpeg'}
            is_video = ext in video_extensions
            
            needs_compression = file_size > (1024**3) or is_video  # >1GB or video file
            
            # Only download from S3 if we need to compress it
            if s3_check and needs_compression:
                logger.info("Large file or video detected, downloading from S3 for compression...")
                local_path = download_from_s3(audio_s3_uri, temp_dir)
                summary['steps_completed'].append('download_from_s3')
            
            if needs_compression:
                logger.info("=" * 60)
                if is_video:
                    logger.info(f"⚠️  VIDEO FILE DETECTED: {filename}")
                    logger.info(f"File size: {file_size_gb:.2f} GB")
                    logger.info("This appears to be a video file miscategorized as audio.")
                    logger.info("Converting to MP3 for audio processing...")
                else:
                    logger.info(f"⚠️  LARGE FILE DETECTED: {file_size_gb:.2f} GB")
                    logger.info("File exceeds 1GB threshold. Creating compressed MP3 version...")
                logger.info("=" * 60)
                
                # Compress to MP3
                compressed_path = compress_to_mp3(local_path, temp_dir, bitrate='128k')
                compressed_filename = os.path.basename(compressed_path)
                
                # Upload compressed version to S3
                compressed_s3_key = f"audio/{asset_uuid}/{compressed_filename}"
                compressed_s3_uri = upload_to_s3(compressed_path, compressed_s3_key)
                
                logger.info("✅ Both original and compressed versions stored in S3:")
                logger.info(f"  Original:   {audio_s3_uri}")
                logger.info(f"  Compressed: {compressed_s3_uri}")
                
                # Use compressed version for transcription (smaller, faster, cheaper)
                transcription_audio_uri = compressed_s3_uri
                transcription_filename = compressed_filename
                summary['compressed_audio_s3_uri'] = compressed_s3_uri
                summary['compression_applied'] = True
                summary['steps_completed'].append('compression')
            else:
                logger.info(f"File size: {file_size_gb:.2f} GB - within limits, no compression needed")
                transcription_audio_uri = audio_s3_uri
                transcription_filename = filename
                summary['compression_applied'] = False
            
            # Extract base filename (without extension) for transcript files
            base_filename = os.path.splitext(transcription_filename)[0]
            logger.info(f"Base filename for transcripts: {base_filename}")
            
            # Step 2: Transcribe with LOCAL Whisper AI
            logger.info("Step 2/7: Transcribing with LOCAL Whisper AI...")
            whisper_model = 'base'  # Options: tiny, base, small, medium, large
            logger.info(f"   Model: {whisper_model}")
            logger.info(f"   Using vocabulary-enhanced transcription")
            
            try:
                transcript_result = transcribe_with_whisper(
                    transcription_local_path if 'transcription_local_path' in locals() else audio_local_path,
                    asset_uuid, 
                    base_filename,
                    model_name=whisper_model
                )
                summary['transcription_method'] = f'whisper_{whisper_model}'
                summary['steps_completed'].append('transcription_complete')
                
                # Whisper returns different format - store for compatibility
                transcript_json = transcript_result  # Use Whisper result directly
                
            except Exception as e:
                logger.error(f"❌ Whisper transcription failed: {e}")
                summary['status'] = 'failed'
                summary['message'] = f'Whisper transcription error: {str(e)}'
                summary['steps_completed'].append('transcription_failed')
                shutil.rmtree(temp_dir, ignore_errors=True)
                return summary
            
            # Step 4: Generate transcript formats
            logger.info("Step 4/7: Generating transcript formats...")
            
            # SRT format - AWS Transcribe generates this automatically now
            # The SRT file is placed in the same directory as the JSON output
            # AWS Transcribe names it: {base_filename}.srt
            srt_s3_key = f"audio/{asset_uuid}/{base_filename}.srt"
            logger.info(f"SRT subtitle file auto-generated by AWS Transcribe at: s3://{S3_BUCKET}/{srt_s3_key}")
            
            # Verify SRT file exists in S3
            try:
                s3_client.head_object(Bucket=S3_BUCKET, Key=srt_s3_key)
                logger.info("✅ SRT file confirmed in S3")
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    logger.warning("⚠️  SRT file not found, falling back to manual generation")
                    # Fallback: generate SRT manually if auto-generation failed
                    srt_content = transcribe_to_srt(transcript_json)
                    srt_path = os.path.join(temp_dir, f"{base_filename}.srt")
                    with open(srt_path, 'w') as f:
                        f.write(srt_content)
                    upload_to_s3(srt_path, srt_s3_key)
                else:
                    raise
            
            # Formatted text with intelligent paragraphs
            formatted_text = transcribe_to_formatted_text(transcript_json)
            
            # Check if transcript is empty (no speech content)
            if formatted_text is None:
                logger.warning("⚠️  No speech content found in audio file - skipping AI processing")
                logger.info("This file appears to be a sound effect or instrumental with no dialogue")
                
                # Mark as skipped, not failed
                summary['status'] = 'skipped'
                summary['message'] = 'No speech content found - likely sound effect or instrumental'
                summary['steps_completed'].append('transcript_check_failed')
                
                # Clean up temp directory
                shutil.rmtree(temp_dir, ignore_errors=True)
                return summary
            
            formatted_path = os.path.join(temp_dir, f"{base_filename}.txt")
            with open(formatted_path, 'w') as f:
                f.write(formatted_text)
            formatted_s3_key = f"audio/{asset_uuid}/{base_filename}.txt"
            upload_to_s3(formatted_path, formatted_s3_key)
            
            # Raw JSON (already in S3 from Transcribe)
            summary['srt_file'] = f"s3://{S3_BUCKET}/{srt_s3_key}"
            summary['formatted_transcript'] = f"s3://{S3_BUCKET}/{formatted_s3_key}"
            summary['raw_json'] = f"s3://{S3_BUCKET}/audio/{asset_uuid}/{base_filename}.json"
            summary['steps_completed'].append('transcript_formats')
            
            # Step 5: Extract text and generate summary/keywords
            logger.info("Step 5/7: Extracting text and generating summary/keywords...")
            plain_text = extract_plain_text(transcript_json)
            summary_text, keywords = generate_summary_and_keywords(plain_text, asset_name)
            
            # Run AWS Comprehend analysis
            logger.info("Step 5b/7: Analyzing transcript with AWS Comprehend...")
            comprehend_analysis = analyze_transcript_with_comprehend(plain_text)
            
            # Create enhanced transcript with summary, keywords, plain text, and Comprehend analysis
            enhanced_data = {
                'asset_uuid': asset_uuid,
                'asset_name': asset_name,
                'summary': summary_text,
                'keywords': keywords,
                'plain_text': plain_text,
                'comprehend_analysis': {
                    'entities': comprehend_analysis.get('entities', []),
                    'key_phrases': comprehend_analysis.get('key_phrases', []),
                    'sentiment': comprehend_analysis.get('sentiment')
                },
                'transcript_files': {
                    'json': f"s3://{S3_BUCKET}/audio/{asset_uuid}/{base_filename}.json",
                    'srt': f"s3://{S3_BUCKET}/audio/{asset_uuid}/{base_filename}.srt",
                    'txt': f"s3://{S3_BUCKET}/audio/{asset_uuid}/{base_filename}.txt"
                },
                'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            }
            
            # Save enhanced data to separate file in S3
            enhanced_json_path = os.path.join(temp_dir, f"{base_filename}_enhanced.json")
            with open(enhanced_json_path, 'w') as f:
                json.dump(enhanced_data, f, indent=2)
            upload_to_s3(enhanced_json_path, f"audio/{asset_uuid}/{base_filename}_enhanced.json")
            logger.info("Saved enhanced transcript data (summary, keywords, plain text) to S3")
            
            summary['steps_completed'].append('summary_keywords')
            
            # Step 6: Generate embeddings and store
            logger.info("Step 6/7: Generating embeddings and storing in database...")
            
            embeddings_data = []
            
            # Embedding 1: Full transcript
            logger.info("⏳ Generating transcript embedding (1/4)...")
            transcript_embedding = generate_embedding(plain_text[:8000])
            embeddings_data.append({
                'content_type': 'transcript',
                'text': plain_text[:8000],
                'embedding': transcript_embedding,
                'metadata': {
                    'srt_file': summary['srt_file'],
                    'formatted_transcript': summary['formatted_transcript'],
                    'speaker_count': len(set([s.get('speaker_label') for s in transcript_json['results'].get('speaker_labels', {}).get('segments', [])])),
                    'duration': transcript_json['results'].get('audio_segments', [{}])[0].get('end_time', 0) if transcript_json['results'].get('audio_segments') else 0
                }
            })
            
            # Embedding 2: Summary
            logger.info("⏳ Generating summary embedding (2/4)...")
            summary_embedding = generate_embedding(summary_text)
            embeddings_data.append({
                'content_type': 'summary',
                'text': summary_text,
                'embedding': summary_embedding,
                'metadata': {}
            })
            
            # Embedding 3: Keywords
            logger.info("⏳ Generating keywords embedding (3/4)...")
            keywords_embedding = generate_embedding(keywords)
            embeddings_data.append({
                'content_type': 'keywords',
                'text': keywords,
                'embedding': keywords_embedding,
                'metadata': {}
            })
            
            # Embedding 4: Metadata (name + keywords + basic info)
            logger.info("⏳ Generating metadata embedding (4/4)...")
            metadata_text = f"{asset_name}. Keywords: {keywords}. Type: {asset_type}"
            metadata_embedding = generate_embedding(metadata_text)
            embeddings_data.append({
                'content_type': 'metadata',
                'text': metadata_text,
                'embedding': metadata_embedding,
                'metadata': {
                    'asset_name': asset_name,
                    'asset_type': asset_type
                }
            })
            
            # Store embeddings in database
            num_embeddings = store_embeddings(conn, asset_id, embeddings_data)
            summary['embeddings_stored'] = num_embeddings
            summary['steps_completed'].append('embeddings_stored')
            
            # Store Comprehend analysis in database
            logger.info("Step 6b/7: Storing Comprehend analysis in database...")
            comprehend_counts = store_comprehend_analysis(conn, asset_id, comprehend_analysis)
            summary['comprehend_stored'] = comprehend_counts
            summary['steps_completed'].append('comprehend_stored')
        
        finally:
            # ALWAYS cleanup temp directory, even on error
            try:
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temp directory: {temp_dir}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp directory {temp_dir}: {cleanup_error}")
        
        # Success
        summary['status'] = 'complete'
        summary['processing_time_seconds'] = time.time() - processing_start
        
        logger.info(f"=== Audio Asset {asset_uuid} Processing Complete ===")
        logger.info(f"Processing time: {summary['processing_time_seconds']:.2f} seconds")
        logger.info(f"Embeddings stored: {num_embeddings}")
        logger.info(f"Comprehend analysis stored: {comprehend_counts}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error processing audio asset {asset_uuid}: {e}", exc_info=True)
        summary['status'] = 'failed'
        summary['errors'].append(str(e))
        summary['processing_time_seconds'] = time.time() - processing_start
        
        # Cleanup temp directory on exception too
        if 'temp_dir' in locals():
            try:
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temp directory after error: {temp_dir}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp directory after error: {cleanup_error}")
    
    return summary


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Process audio asset for deep embeddings')
    parser.add_argument('asset_uuid', help='UUID of the audio asset to process')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Process the asset
    result = process_audio_asset(args.asset_uuid, args.verbose)
    
    # Print summary
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(json.dumps(result, indent=2))
    print("="*80 + "\n")
    
    # Exit with appropriate code
    sys.exit(0 if result['status'] == 'complete' else 1)


if __name__ == '__main__':
    main()
