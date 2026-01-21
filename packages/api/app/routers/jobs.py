"""
Jobs API router - CRUD operations for transcription jobs.
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..schemas.job import Job, JobStatus
from ..models.job import JobCreate, JobResponse, JobListResponse, AnalyticsResponse, StatusCount, JobsPerDay
from ..services.job_manager import job_manager

router = APIRouter()


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    type: Optional[str] = Query(None, description="Filter by type"),
    search: Optional[str] = Query(None, description="Search in name"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    """List all jobs with optional filters."""
    query = select(Job)

    # Apply filters
    if status:
        query = query.where(Job.status == status)
    if type:
        query = query.where(Job.type == type)
    if search:
        query = query.where(Job.name.ilike(f"%{search}%"))

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await session.execute(count_query)
    total = total_result.scalar() or 0

    # Apply pagination and ordering
    query = query.order_by(desc(Job.created_at)).offset(offset).limit(limit)

    result = await session.execute(query)
    jobs = result.scalars().all()

    return JobListResponse(
        jobs=[JobResponse(**job.to_dict()) for job in jobs],
        total=total,
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get a single job by ID."""
    result = await session.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(**job.to_dict())


@router.post("/jobs", response_model=JobResponse, status_code=201)
async def create_job(
    job_data: JobCreate,
    session: AsyncSession = Depends(get_session),
):
    """Create a new transcription job."""
    job_id = str(uuid.uuid4())

    # Generate name if not provided
    name = job_data.name
    if not name:
        if job_data.type == "uuid":
            name = f"UUID: {job_data.input[:8]}..."
        else:
            name = job_data.input

    job = Job(
        id=job_id,
        name=name,
        type=job_data.type,
        input=job_data.input,
        status=JobStatus.PENDING.value,
        progress=0,
    )

    session.add(job)
    await session.commit()
    await session.refresh(job)

    # Submit to job manager for processing
    await job_manager.submit_job(job_id, job_data.type, job_data.input)

    return JobResponse(**job.to_dict())


@router.delete("/jobs/{job_id}")
async def delete_job(
    job_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Delete a job."""
    result = await session.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Cancel if running
    if job.status in [JobStatus.PENDING.value, JobStatus.PROCESSING.value]:
        await job_manager.cancel_job(job_id)

    await session.delete(job)
    await session.commit()

    return {"success": True}


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Cancel a running or pending job."""
    result = await session.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status not in [JobStatus.PENDING.value, JobStatus.PROCESSING.value]:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")

    # Cancel the job
    await job_manager.cancel_job(job_id)

    # Update status
    job.status = JobStatus.CANCELLED.value
    job.completed_at = datetime.utcnow()
    await session.commit()

    return {"success": True}


@router.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics(
    session: AsyncSession = Depends(get_session),
):
    """Get analytics data for the dashboard."""
    # Total jobs
    total_result = await session.execute(select(func.count()).select_from(Job))
    total_jobs = total_result.scalar() or 0

    # Completed jobs
    completed_result = await session.execute(
        select(func.count()).select_from(Job).where(Job.status == JobStatus.COMPLETED.value)
    )
    completed_jobs = completed_result.scalar() or 0

    # Success rate
    success_rate = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0

    # Average duration
    avg_result = await session.execute(
        select(func.avg(Job.duration)).where(Job.duration.isnot(None))
    )
    avg_duration = avg_result.scalar() or 0

    # Status distribution
    status_counts = []
    for status in JobStatus:
        count_result = await session.execute(
            select(func.count()).select_from(Job).where(Job.status == status.value)
        )
        count = count_result.scalar() or 0
        status_counts.append(StatusCount(status=status.value, count=count))

    # Jobs per day (last 7 days)
    jobs_per_day = []
    for i in range(6, -1, -1):
        date = datetime.utcnow().date() - timedelta(days=i)
        next_date = date + timedelta(days=1)

        count_result = await session.execute(
            select(func.count())
            .select_from(Job)
            .where(Job.created_at >= datetime.combine(date, datetime.min.time()))
            .where(Job.created_at < datetime.combine(next_date, datetime.min.time()))
        )
        count = count_result.scalar() or 0
        jobs_per_day.append(
            JobsPerDay(date=date.strftime("%b %d"), count=count)
        )

    return AnalyticsResponse(
        totalJobs=total_jobs,
        successRate=round(success_rate, 1),
        avgDuration=round(avg_duration, 2),
        jobsPerDay=jobs_per_day,
        statusDistribution=status_counts,
    )
