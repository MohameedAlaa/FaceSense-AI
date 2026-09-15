from typing import Optional
from pathlib import Path
from fastapi import APIRouter, Depends, Request, HTTPException, Query, Path as FPath, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.models.user import User
from backend.app.core.dependencies import get_optional_current_user, require_admin
from backend.app.core.rate_limit import rate_limit_feedback
from backend.app.schemas.feedback import (
    FeedbackCreateRequest,
    FeedbackCreateResponse,
    FeedbackStatsResponse,
    FeedbackReviewOverviewResponse,
    FeedbackReviewListResponse,
    FeedbackReviewDecisionRequest,
    FeedbackReviewDecisionResponse,
)
from backend.app.services.feedback_service import feedback_service
from backend.app.core.config import settings

router = APIRouter()

MAX_FEEDBACK_REQUEST_BYTES = 4 * 1024 * 1024  # 4 MB payload guard
VALID_FER2013_EMOTIONS = {"angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"}


def verify_feedback_payload_size(request: Request) -> None:
    """Rejects incoming feedback requests exceeding maximum allowed body size before parsing."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            length = int(content_length)
            if length > MAX_FEEDBACK_REQUEST_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Feedback payload too large. Maximum size is {MAX_FEEDBACK_REQUEST_BYTES // (1024 * 1024)}MB.",
                )
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header.")


@router.post(
    "",
    response_model=FeedbackCreateResponse,
    status_code=201,
    tags=["Feedback"],
    dependencies=[Depends(rate_limit_feedback), Depends(verify_feedback_payload_size)],
)
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


@router.get(
    "/review/overview",
    response_model=FeedbackReviewOverviewResponse,
    tags=["Feedback Review"],
    dependencies=[Depends(require_admin)],
)
def get_feedback_review_overview(
    db: Optional[Session] = Depends(get_db),
) -> FeedbackReviewOverviewResponse:
    """
    Admin-only: Returns feedback counts grouped by the 7 FER2013 emotions,
    categorized by review status (pending, approved, rejected).
    """
    data = feedback_service.get_review_overview(db=db)
    return FeedbackReviewOverviewResponse(**data)


@router.get(
    "/review/emotion/{emotion}",
    response_model=FeedbackReviewListResponse,
    tags=["Feedback Review"],
    dependencies=[Depends(require_admin)],
)
def get_feedback_review_by_emotion(
    emotion: str = FPath(..., description="One of the 7 FER2013 emotion classes"),
    status: Optional[str] = Query(None, description="Optional status filter: pending, approved, rejected, or all"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Optional[Session] = Depends(get_db),
) -> FeedbackReviewListResponse:
    """
    Admin-only: Lists submitted feedback items for a specific emotion class.
    """
    emo_clean = emotion.strip().lower()
    if emo_clean not in VALID_FER2013_EMOTIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY if hasattr(status, "HTTP_422_UNPROCESSABLE_ENTITY") else 422,
            detail=f"Invalid emotion '{emotion}'. Must be one of {sorted(VALID_FER2013_EMOTIONS)}.",
        )

    if status:
        st_clean = status.strip().lower()
        if st_clean not in {"pending", "approved", "rejected", "all"}:
            raise HTTPException(
                status_code=422,
                detail="Invalid status filter. Must be one of: pending, approved, rejected, all.",
            )

    result = feedback_service.list_review_feedback(
        emotion=emo_clean,
        status_filter=status,
        limit=limit,
        offset=offset,
        db=db,
    )

    # Attach image_url if image_path exists and is within outputs/feedback/images
    base_img_dir = (settings.FEEDBACK_DIR / "images").resolve()
    for item in result["items"]:
        img_p = item.get("image_path")
        if img_p:
            clean_p = str(img_p).replace("\\", "/")
            p_name = clean_p.split("/")[-1]
            target_file = (base_img_dir / p_name).resolve()
            if p_name and str(target_file).startswith(str(base_img_dir)) and target_file.is_file():
                item["image_url"] = f"{settings.API_V1_STR}/feedback/images/{p_name}"
            else:
                item["image_url"] = None

    return FeedbackReviewListResponse(**result)


@router.patch(
    "/review/{feedback_id}/decision",
    response_model=FeedbackReviewDecisionResponse,
    tags=["Feedback Review"],
)
def update_feedback_decision(
    feedback_id: str = FPath(..., description="ID of feedback record to review"),
    request: FeedbackReviewDecisionRequest = ...,
    db: Optional[Session] = Depends(get_db),
    admin_user: User = Depends(require_admin),
) -> FeedbackReviewDecisionResponse:
    """
    Admin-only: Authoritatively approves or rejects a feedback record.
    The final training label is set by the admin (or resolved from user correction if accepted).
    """
    record = feedback_service.get_by_feedback_id(feedback_id, db=db)
    if not record:
        raise HTTPException(status_code=404, detail=f"Feedback record '{feedback_id}' not found.")

    res = feedback_service.update_review_decision(
        feedback_id=feedback_id,
        decision=request.decision,
        final_label=request.final_label,
        reviewer_id=admin_user.id,
        db=db,
    )
    return FeedbackReviewDecisionResponse(**res)


@router.get(
    "/images/{image_name}",
    tags=["Feedback Review"],
    dependencies=[Depends(require_admin)],
)
def get_feedback_image(
    image_name: str = FPath(..., description="Filename of stored feedback crop"),
):
    """
    Admin-only: Safely serves stored face crop images for administrative review.
    Guards against path traversal attacks.
    """
    if ".." in image_name or "/" in image_name or "\\" in image_name:
        raise HTTPException(status_code=400, detail="Invalid image filename.")

    img_path = (settings.FEEDBACK_DIR / "images" / image_name).resolve()
    base_dir = (settings.FEEDBACK_DIR / "images").resolve()

    if not str(img_path).startswith(str(base_dir)) or not img_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")

    return FileResponse(path=img_path, media_type="image/jpeg")

