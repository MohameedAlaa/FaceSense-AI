from fastapi import APIRouter
from backend.app.schemas.model import ModelInfoResponse
from backend.app.services.model_service import model_info_service

router = APIRouter()


@router.get("/info", response_model=ModelInfoResponse, tags=["Model"])
def get_model_info() -> ModelInfoResponse:
    """Returns current active model metadata, architecture details, and classes."""
    return model_info_service.get_model_info()
