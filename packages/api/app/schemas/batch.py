"""
SQLAlchemy model for Batch.
"""

from sqlalchemy import Column, String, DateTime
from sqlalchemy.sql import func

from ..database import Base


class Batch(Base):
    """Batch database model for grouping multiple jobs."""

    __tablename__ = "batches"

    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=True)  # Optional display name
    created_at = Column(DateTime, server_default=func.now())

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
        }
