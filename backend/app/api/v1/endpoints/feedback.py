from typing import Optional
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.models.user import User
from backend.app.core.dependencies import get_optional_current_user, require_admin
from backend.app.core.rate_limit import rate_limit_feedback
from backend.app.schemas.feedback import (
    FeedbackCreateRequest,
    FeedbackCreateResponse,
    FeedbackStatsResponse,
)
from backend.app.services.feedback_service import feedback_service

router = APIRouter()


@router.post("", response_model=FeedbackCreateResponse, status_code=201, tags=["Feedback"], dependencies=[Depends(rate_limit_feedback)])
def submit_feedback(
    request: FeedbackCreateRequest,
    db: Optional[Session] = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
) -> FeedbackCreateResponse:
    """
    Submits user feedback (correct, incorrect, or uncertain) for model predictions,
    persisting to structured JSONL logs and PostgreSQL database (if connected).
    """
    user_id = current_user.id if current_user else None
    return feedback_service.create_feedback(request, db=db, user_id=user_id)


@router.get("/stats", response_model=FeedbackStatsResponse, tags=["Feedback"], dependencies=[Depends(require_admin)])
def get_feedback_stats(
    db: Optional[Session] = Depends(get_db),
) -> FeedbackStatsResponse:
    """
    Returns aggregate feedback metrics, totals by feedback state, and class breakdowns.
    Uses database when available, with automatic JSONL fallback.
    """
    return feedback_service.get_stats(db=db)
