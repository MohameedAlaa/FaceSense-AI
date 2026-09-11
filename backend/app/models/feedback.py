"""
FeedbackRecord ORM Model.
Maps to the 'feedback' table in PostgreSQL.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from sqlalchemy import DateTime, Float, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base


class FeedbackRecord(Base):
    """
    Persistent feedback record model.
    Stores emotion prediction feedback, user corrections, confidence, image references,
    and associated metadata.
    """
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feedback_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    state: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    predicted_emotion: Mapped[str] = mapped_column(String(30), index=True, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    corrected_emotion: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    image_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    bounding_box: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    extra_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert ORM record into dictionary representation."""
        return {
            "id": self.id,
            "feedback_id": self.feedback_id,
            "state": self.state,
            "predicted_emotion": self.predicted_emotion,
            "confidence": self.confidence,
            "corrected_emotion": self.corrected_emotion,
            "image_path": self.image_path,
            "bounding_box": self.bounding_box,
            "notes": self.notes,
            "model_version": self.model_version,
            "metadata": self.extra_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
