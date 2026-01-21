"""
Files API router - File upload handling.
"""

import uuid
import logging
import aiofiles
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..database import get_session
from ..schemas.job import Job, JobStatus
from ..models.job import JobResponse
from ..services.job_manager import job_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# Allowed audio/video file extensions
ALLOWED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma",
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
}


@router.post("/upload", response_model=JobResponse, status_code=201)
async def upload_file(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    """
    Upload an audio or video file and create a transcription job.

    Returns the created job.
    """
    logger.info(f"=== UPLOAD START: {file.filename} ===")
    print(f"=== UPLOAD START: {file.filename} ===")

    # Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Generate unique filename
    file_id = str(uuid.uuid4())
    save_filename = f"{file_id}{ext}"
    save_path = settings.upload_dir / save_filename
    logger.info(f"Saving to: {save_path}")
    print(f"Saving to: {save_path}")

    # Stream to disk
    try:
        async with aiofiles.open(save_path, "wb") as f:
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

    # Create a job for this file
    logger.info("Creating job in database...")
    print("Creating job in database...")
    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        name=file.filename,
        type="file",
        input=str(save_path),
        status=JobStatus.PENDING.value,
        progress=0,
    )

    session.add(job)
    await session.commit()
    await session.refresh(job)
    logger.info(f"Job created: {job_id}")
    print(f"Job created: {job_id}")

    # Submit to job manager for processing
    logger.info("Submitting to job manager...")
    print("Submitting to job manager...")
    await job_manager.submit_job(job_id, "file", str(save_path))
    logger.info("Job submitted!")
    print("Job submitted!")

    job_dict = job.to_dict()
    logger.info(f"Returning JobResponse: {job_dict}")
    print(f"Returning JobResponse: {job_dict}")

    return JobResponse(**job_dict)


@router.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete an uploaded file."""
    # Find the file by ID (could have any extension)
    for ext in ALLOWED_EXTENSIONS:
        file_path = settings.upload_dir / f"{file_id}{ext}"
        if file_path.exists():
            file_path.unlink()
            return {"success": True}

    raise HTTPException(status_code=404, detail="File not found")
