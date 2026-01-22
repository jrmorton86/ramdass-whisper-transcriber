"""
SQLAlchemy model for Job.
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Enum
from sqlalchemy.sql import func
import enum

from ..database import Base


class JobStatus(str, enum.Enum):
    """Job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, enum.Enum):
    """Job type enumeration."""
    FILE = "file"
    UUID = "uuid"


class Job(Base):
    """Job database model."""

    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    type = Column(String(10), nullable=False)  # 'file' or 'uuid'
    input = Column(String(1024), nullable=False)  # filename or UUID
    status = Column(String(20), default=JobStatus.PENDING.value)
    progress = Column(Integer, default=0)
    current_stage = Column(String(100), nullable=True)

    created_at = Column(DateTime, server_default=func.now())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    duration = Column(Float, nullable=True)

    error = Column(Text, nullable=True)

    # Result stored as JSON strings
    result_text = Column(Text, nullable=True)
    result_segments = Column(Text, nullable=True)  # JSON array
    result_metadata = Column(Text, nullable=True)  # JSON object

    # Logs stored as JSON array for persistence after job completion
    logs = Column(Text, nullable=True)  # JSON array of log entries

    def to_dict(self):
        """Convert to dictionary."""
        import json

        result = None
        if self.result_text:
            result = {
                "text": self.result_text,
                "segments": json.loads(self.result_segments) if self.result_segments else [],
                "metadata": json.loads(self.result_metadata) if self.result_metadata else {},
            }

        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "input": self.input,
            "status": self.status,
            "progress": self.progress,
            "currentStage": self.current_stage,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
            "startedAt": self.started_at.isoformat() if self.started_at else None,
            "completedAt": self.completed_at.isoformat() if self.completed_at else None,
            "duration": self.duration,
            "error": self.error,
            "result": result,
            "logs": json.loads(self.logs) if self.logs else None,
        }
