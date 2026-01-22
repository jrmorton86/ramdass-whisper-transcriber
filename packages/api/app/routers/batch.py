"""
Batch API router - CRUD operations for batch uploads.
"""

import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..schemas.batch import Batch
from ..schemas.job import Job, JobStatus
from ..models.batch import BatchResponse, BatchListResponse
from ..models.job import JobResponse
from ..services.job_manager import job_manager

router = APIRouter()


def _compute_batch_status(jobs: list[Job]) -> str:
    """
    Compute overall batch status based on jobs.

    - pending: all jobs pending
    - processing: any job processing, or mix of pending/completed
    - completed: all jobs completed successfully
    - failed: all jobs failed
    - partial: some completed, some failed (no pending/processing)
    """
    if not jobs:
        return "pending"

    statuses = [job.status for job in jobs]

    # Check if any job is still processing or pending
    has_pending = JobStatus.PENDING.value in statuses
    has_processing = JobStatus.PROCESSING.value in statuses

    if has_processing:
        return "processing"

    if has_pending:
        # Some pending, check if others are done
        non_pending = [s for s in statuses if s != JobStatus.PENDING.value]
        if non_pending:
            return "processing"  # Mix of pending and completed/failed
        return "pending"

    # All jobs are in a terminal state (completed, failed, or cancelled)
    completed_count = statuses.count(JobStatus.COMPLETED.value)
    failed_count = statuses.count(JobStatus.FAILED.value) + statuses.count(JobStatus.CANCELLED.value)

    if failed_count == 0:
        return "completed"
    if completed_count == 0:
        return "failed"
    return "partial"


def _build_batch_response(batch: Batch, jobs: list[Job]) -> BatchResponse:
    """Build a BatchResponse from a batch and its jobs."""
    completed_count = sum(1 for j in jobs if j.status == JobStatus.COMPLETED.value)
    failed_count = sum(1 for j in jobs if j.status in [JobStatus.FAILED.value, JobStatus.CANCELLED.value])

    # Determine completedAt - latest completed_at among all jobs if batch is done
    batch_status = _compute_batch_status(jobs)
    completed_at = None
    if batch_status in ["completed", "failed", "partial"]:
        completed_times = [j.completed_at for j in jobs if j.completed_at]
        if completed_times:
            completed_at = max(completed_times)

    return BatchResponse(
        id=batch.id,
        status=batch_status,
        totalFiles=len(jobs),
        completedFiles=completed_count,
        failedFiles=failed_count,
        jobIds=[j.id for j in jobs],
        createdAt=batch.created_at,
        completedAt=completed_at,
    )


@router.post("/batch", response_model=BatchResponse, status_code=201)
async def create_batch(
    name: Optional[str] = Query(None, description="Optional display name for the batch"),
    session: AsyncSession = Depends(get_session),
):
    """
    Create a new batch for grouping multiple file uploads.

    Returns a batch with a generated UUID that can be used when uploading files.
    """
    batch_id = str(uuid.uuid4())

    batch = Batch(
        id=batch_id,
        name=name,
    )

    session.add(batch)
    await session.commit()
    await session.refresh(batch)

    # Return empty batch (no jobs yet)
    return BatchResponse(
        id=batch.id,
        status="pending",
        totalFiles=0,
        completedFiles=0,
        failedFiles=0,
        jobIds=[],
        createdAt=batch.created_at,
        completedAt=None,
    )


@router.get("/batch/{batch_id}", response_model=BatchResponse)
async def get_batch(
    batch_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Get a batch by ID with summary of job statuses.

    Returns batch info including counts of total, completed, and failed files.
    """
    # Get the batch
    result = await session.execute(select(Batch).where(Batch.id == batch_id))
    batch = result.scalar_one_or_none()

    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Get all jobs in this batch
    jobs_result = await session.execute(
        select(Job).where(Job.batch_id == batch_id).order_by(Job.created_at)
    )
    jobs = list(jobs_result.scalars().all())

    return _build_batch_response(batch, jobs)


@router.get("/batch/{batch_id}/jobs", response_model=list[JobResponse])
async def get_batch_jobs(
    batch_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Get all jobs in a batch.

    Returns a list of all jobs associated with the batch.
    """
    # Verify batch exists
    result = await session.execute(select(Batch).where(Batch.id == batch_id))
    batch = result.scalar_one_or_none()

    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Get all jobs in this batch
    jobs_result = await session.execute(
        select(Job).where(Job.batch_id == batch_id).order_by(Job.created_at)
    )
    jobs = jobs_result.scalars().all()

    return [JobResponse(**job.to_dict()) for job in jobs]


@router.post("/batch/{batch_id}/retry-failed", response_model=BatchResponse)
async def retry_failed_jobs(
    batch_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Retry all failed jobs in a batch.

    Resets failed jobs to pending status and resubmits them for processing.
    """
    # Verify batch exists
    result = await session.execute(select(Batch).where(Batch.id == batch_id))
    batch = result.scalar_one_or_none()

    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Get all failed jobs in this batch
    jobs_result = await session.execute(
        select(Job).where(
            Job.batch_id == batch_id,
            Job.status.in_([JobStatus.FAILED.value, JobStatus.CANCELLED.value])
        )
    )
    failed_jobs = list(jobs_result.scalars().all())

    if not failed_jobs:
        raise HTTPException(status_code=400, detail="No failed jobs to retry")

    # Reset and resubmit each failed job
    for job in failed_jobs:
        job.status = JobStatus.PENDING.value
        job.progress = 0
        job.current_stage = None
        job.error = None
        job.started_at = None
        job.completed_at = None
        job.duration = None
        job.result_text = None
        job.result_segments = None
        job.result_metadata = None
        job.logs = None

        # Resubmit to job manager
        await job_manager.submit_job(job.id, job.type, job.input)

    await session.commit()

    # Get updated job list
    all_jobs_result = await session.execute(
        select(Job).where(Job.batch_id == batch_id).order_by(Job.created_at)
    )
    all_jobs = list(all_jobs_result.scalars().all())

    return _build_batch_response(batch, all_jobs)


@router.get("/batches", response_model=BatchListResponse)
async def list_batches(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    """
    List all batches with pagination.

    Returns batches ordered by creation date (newest first).
    """
    # Get total count
    count_result = await session.execute(select(func.count()).select_from(Batch))
    total = count_result.scalar() or 0

    # Get batches with pagination
    batches_result = await session.execute(
        select(Batch).order_by(desc(Batch.created_at)).offset(offset).limit(limit)
    )
    batches = batches_result.scalars().all()

    # Fetch all jobs for all batches in a single query to avoid N+1 problem
    batch_ids = [batch.id for batch in batches]
    jobs_by_batch: dict[str, list[Job]] = {bid: [] for bid in batch_ids}

    if batch_ids:
        all_jobs_result = await session.execute(
            select(Job).where(Job.batch_id.in_(batch_ids))
        )
        all_jobs = all_jobs_result.scalars().all()

        # Group jobs by batch_id
        for job in all_jobs:
            jobs_by_batch[job.batch_id].append(job)

    # Build response for each batch
    batch_responses = []
    for batch in batches:
        jobs = jobs_by_batch.get(batch.id, [])
        batch_responses.append(_build_batch_response(batch, jobs))

    return BatchListResponse(
        batches=batch_responses,
        total=total,
    )
