"""
Pydantic models for Batch API.
"""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel


class BatchResponse(BaseModel):
    """Response model for a batch."""
    id: str
    status: Literal["pending", "processing", "completed", "failed", "partial"]
    totalFiles: int
    completedFiles: int
    failedFiles: int
    jobIds: list[str]
    createdAt: datetime
    completedAt: Optional[datetime] = None

    class Config:
        from_attributes = True


class BatchListResponse(BaseModel):
    """Response model for batch list."""
    batches: list[BatchResponse]
    total: int
