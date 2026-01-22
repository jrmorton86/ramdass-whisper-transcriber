"""
Pydantic models for Job API.
"""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


class TranscriptionSegment(BaseModel):
    """A segment of the transcription with timing."""
    start: float
    end: float
    text: str


class TranscriptionMetadata(BaseModel):
    """Metadata about the transcription."""
    duration: float
    language: str = "en"
    model: str = "whisper-medium"


class TranscriptionResult(BaseModel):
    """Full transcription result."""
    text: str
    segments: list[TranscriptionSegment] = []
    metadata: TranscriptionMetadata


class JobCreate(BaseModel):
    """Request model for creating a job."""
    type: Literal["file", "uuid"]
    input: str = Field(..., description="Filename or UUID")
    name: Optional[str] = Field(None, description="Display name for the job")


class LogEntry(BaseModel):
    """A log entry from job execution."""
    type: str = "log"
    timestamp: str
    level: Optional[str] = "info"
    message: Optional[str] = None  # Not present in progress/status entries
    ansi: Optional[str] = None
    stage: Optional[str] = None
    step: Optional[int] = None
    totalSteps: Optional[int] = None
    progress: Optional[int] = None  # For progress entries
    status: Optional[str] = None  # For status entries
    error: Optional[str] = None


class JobResponse(BaseModel):
    """Response model for a job."""
    id: str
    name: str
    type: Literal["file", "uuid"]
    input: str
    status: Literal["pending", "processing", "completed", "failed", "cancelled"]
    progress: int = 0
    currentStage: Optional[str] = None
    createdAt: datetime
    startedAt: Optional[datetime] = None
    completedAt: Optional[datetime] = None
    duration: Optional[float] = None
    error: Optional[str] = None
    result: Optional[TranscriptionResult] = None
    logs: Optional[list[LogEntry]] = None

    class Config:
        from_attributes = True


class JobListResponse(BaseModel):
    """Response model for job list."""
    jobs: list[JobResponse]
    total: int


class StatusCount(BaseModel):
    """Status distribution item."""
    status: str
    count: int


class JobsPerDay(BaseModel):
    """Jobs per day item."""
    date: str
    count: int


class AnalyticsResponse(BaseModel):
    """Response model for analytics."""
    totalJobs: int
    successRate: float
    avgDuration: float
    jobsPerDay: list[JobsPerDay]
    statusDistribution: list[StatusCount]
