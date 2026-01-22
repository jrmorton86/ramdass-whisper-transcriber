"""
S3 Upload Service - Scaffold for uploading job results to S3.

This module provides functions to upload transcription results to S3
and clean up local working files after successful processing.
"""

import asyncio
import logging
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Retry settings for Windows file locking issues
CLEANUP_RETRIES = 3
CLEANUP_RETRY_DELAY = 1.0  # seconds


async def upload_results_to_s3(job_id: str, output_dir: Path) -> dict:
    """
    Scaffold for uploading job results to S3.

    Currently just logs what would be uploaded.
    TODO: Implement actual S3 upload when ready.

    Args:
        job_id: Job ID for logging/S3 key prefix
        output_dir: Directory containing transcription output files

    Returns:
        dict with 'success' and 'files' (list of file paths that would be uploaded)
    """
    files_to_upload = []

    if output_dir.exists():
        # Collect all files in the output directory
        for file_path in output_dir.iterdir():
            if file_path.is_file():
                files_to_upload.append(str(file_path))

    logger.info(f"[S3 Scaffold] Job {job_id}: Would upload {len(files_to_upload)} files to S3")
    for file_path in files_to_upload:
        logger.info(f"[S3 Scaffold]   - {file_path}")

    # TODO: Implement actual S3 upload
    # Example implementation:
    # import boto3
    # s3_client = boto3.client('s3')
    # bucket = settings.s3_bucket
    # for file_path in files_to_upload:
    #     key = f"transcriptions/{job_id}/{Path(file_path).name}"
    #     s3_client.upload_file(file_path, bucket, key)

    return {
        "success": True,
        "files": files_to_upload,
    }


async def cleanup_job_files(job_id: str, input_file: Path, output_dir: Path) -> bool:
    """
    Delete job working files after successful processing.

    Includes retry logic for Windows file locking issues (OneDrive, antivirus, etc.)

    Args:
        job_id: Job ID for logging
        input_file: The input audio file to delete
        output_dir: The output directory to delete (contains transcripts)

    Returns:
        True if cleanup succeeded, False otherwise
    """
    success = True

    # Small delay to allow file handles to close
    await asyncio.sleep(0.5)

    # Delete input file with retries
    if input_file.exists():
        for attempt in range(CLEANUP_RETRIES):
            try:
                input_file.unlink()
                logger.info(f"[Cleanup] Job {job_id}: Deleted input file {input_file}")
                break
            except OSError as e:
                if attempt < CLEANUP_RETRIES - 1:
                    logger.warning(f"[Cleanup] Job {job_id}: Retry {attempt + 1}/{CLEANUP_RETRIES} for input file (locked)")
                    await asyncio.sleep(CLEANUP_RETRY_DELAY)
                else:
                    logger.error(f"[Cleanup] Job {job_id}: Failed to delete input file {input_file}: {e}")
                    success = False

    # Delete output directory with retries
    if output_dir.exists():
        for attempt in range(CLEANUP_RETRIES):
            try:
                shutil.rmtree(output_dir)
                logger.info(f"[Cleanup] Job {job_id}: Deleted output directory {output_dir}")
                break
            except OSError as e:
                if attempt < CLEANUP_RETRIES - 1:
                    logger.warning(f"[Cleanup] Job {job_id}: Retry {attempt + 1}/{CLEANUP_RETRIES} for output dir (locked)")
                    await asyncio.sleep(CLEANUP_RETRY_DELAY)
                else:
                    # Final attempt: try with ignore_errors as last resort
                    shutil.rmtree(output_dir, ignore_errors=True)
                    if output_dir.exists():
                        logger.error(f"[Cleanup] Job {job_id}: Failed to delete output directory {output_dir}: {e}")
                        success = False
                    else:
                        logger.warning(f"[Cleanup] Job {job_id}: Deleted output directory (with some errors ignored)")

    return success
