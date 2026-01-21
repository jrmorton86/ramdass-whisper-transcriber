"""
Files API router - File upload handling.
"""

import uuid
import aiofiles
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..database import get_session
from ..schemas.job import Job, JobStatus
from ..models.job import JobResponse
from ..services.job_manager import job_manager

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

    # Stream to disk
    try:
        async with aiofiles.open(save_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                await f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Create a job for this file
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

    # Submit to job manager for processing
    await job_manager.submit_job(job_id, "file", str(save_path))

    return JobResponse(**job.to_dict())


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
