from fastapi import APIRouter
from backend.app.schemas.feedback import (
    FeedbackCreateRequest,
    FeedbackCreateResponse,
    FeedbackStatsResponse,
)
from backend.app.services.feedback_service import feedback_service

router = APIRouter()


@router.post("", response_model=FeedbackCreateResponse, status_code=201, tags=["Feedback"])
def submit_feedback(request: FeedbackCreateRequest) -> FeedbackCreateResponse:
    """
    Submits user feedback (correct, incorrect, or uncertain) for model predictions,
    optionally saving face crops and updating logs.
    """
    return feedback_service.create_feedback(request)


@router.get("/stats", response_model=FeedbackStatsResponse, tags=["Feedback"])
def get_feedback_stats() -> FeedbackStatsResponse:
    """
    Returns aggregate feedback metrics, totals by feedback state, and class breakdowns.
    """
    return feedback_service.get_stats()
