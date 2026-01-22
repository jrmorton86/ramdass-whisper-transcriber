"""
Files API router - File upload handling.
"""

import uuid
import logging
import aiofiles
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..database import get_session
from ..schemas.job import Job, JobStatus
from ..models.job import JobResponse
from ..services.job_manager import job_manager
from ..services.ffmpeg import convert_to_mp3

logger = logging.getLogger(__name__)
router = APIRouter()

# Temp directory for files before conversion
TEMP_DIR = settings.upload_dir / "temp"

# Allowed audio/video file extensions
ALLOWED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma",
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
}


@router.post("/upload", response_model=JobResponse, status_code=201)
async def upload_file(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
    batch_id: Optional[str] = Query(None, description="Optional batch ID for grouping jobs"),
):
    """
    Upload an audio or video file and create a transcription job.

    The file will be converted to MP3 format (optimized for Whisper) if not already MP3.

    Args:
        file: The audio or video file to upload
        batch_id: Optional batch ID for grouping multiple jobs (not yet implemented)

    Returns the created job.
    """
    logger.info(f"=== UPLOAD START: {file.filename} ===")
    print(f"=== UPLOAD START: {file.filename} ===")
    if batch_id:
        logger.info(f"Batch ID: {batch_id}")
        print(f"Batch ID: {batch_id}")

    # Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Generate unique file ID
    file_id = str(uuid.uuid4())

    # Ensure temp directory exists
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Save uploaded file to temp location with original extension
    temp_filename = f"{file_id}{ext}"
    temp_path = TEMP_DIR / temp_filename
    logger.info(f"Saving to temp: {temp_path}")
    print(f"Saving to temp: {temp_path}")

    # Stream to disk
    try:
        async with aiofiles.open(temp_path, "wb") as f:
            bytes_written = 0
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                await f.write(chunk)
                bytes_written += len(chunk)
        logger.info(f"File saved: {bytes_written} bytes")
        print(f"File saved: {bytes_written} bytes")
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        print(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Determine final MP3 path
    final_filename = f"{file_id}.mp3"
    final_path = settings.upload_dir / final_filename

    # Convert to MP3 if not already MP3
    if ext == ".mp3":
        # Already MP3, just move from temp to upload dir
        logger.info("File is already MP3, moving to uploads")
        print("File is already MP3, moving to uploads")
        try:
            temp_path.rename(final_path)
        except Exception as e:
            # If rename fails (e.g., cross-device), copy and delete
            import shutil
            shutil.move(str(temp_path), str(final_path))
    else:
        # Convert to MP3 using FFmpeg
        logger.info(f"Converting {ext} to MP3...")
        print(f"Converting {ext} to MP3...")

        success, error = await convert_to_mp3(temp_path, final_path)

        # Clean up temp file regardless of conversion result
        try:
            temp_path.unlink()
            logger.info("Temp file deleted")
            print("Temp file deleted")
        except Exception as e:
            logger.warning(f"Failed to delete temp file: {e}")
            print(f"Failed to delete temp file: {e}")

        if not success:
            logger.error(f"FFmpeg conversion failed: {error}")
            print(f"FFmpeg conversion failed: {error}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to convert file to MP3: {error}",
            )

        logger.info("Conversion complete")
        print("Conversion complete")

    # Create a job for this file
    logger.info("Creating job in database...")
    print("Creating job in database...")
    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        name=file.filename,
        type="file",
        input=str(final_path),
        status=JobStatus.PENDING.value,
        progress=0,
    )
    # Note: batch_id will be used once the Job model supports it (Task 3)

    session.add(job)
    await session.commit()
    await session.refresh(job)
    logger.info(f"Job created: {job_id}")
    print(f"Job created: {job_id}")

    # Submit to job manager for processing
    logger.info("Submitting to job manager...")
    print("Submitting to job manager...")
    await job_manager.submit_job(job_id, "file", str(final_path))
    logger.info("Job submitted!")
    print("Job submitted!")

    job_dict = job.to_dict()
    logger.info(f"Returning JobResponse: {job_dict}")
    print(f"Returning JobResponse: {job_dict}")

    return JobResponse(**job_dict)


@router.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete an uploaded file."""
    # All uploaded files are now converted to MP3
    file_path = settings.upload_dir / f"{file_id}.mp3"
    if file_path.exists():
        file_path.unlink()
        return {"success": True}

    # Fallback: check for any extension (for backwards compatibility)
    for ext in ALLOWED_EXTENSIONS:
        file_path = settings.upload_dir / f"{file_id}{ext}"
        if file_path.exists():
            file_path.unlink()
            return {"success": True}

    raise HTTPException(status_code=404, detail="File not found")
