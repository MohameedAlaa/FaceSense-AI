from typing import Optional
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.session import get_db, check_db_connection
from backend.app.schemas.common import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def get_health(db: Optional[Session] = Depends(get_db)) -> HealthResponse:
    """Returns application health status, API version, and database connectivity."""
    db_status = "not_configured"
    if settings.DATABASE_URL and settings.DATABASE_URL.strip():
        db_status = "connected" if check_db_connection(db) else "disconnected"

    return HealthResponse(
        status="ok",
        service=settings.PROJECT_NAME,
        version=settings.VERSION,
        database=db_status,
    )
