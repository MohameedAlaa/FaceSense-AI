from typing import Optional
from fastapi import APIRouter, File, UploadFile, Query

from backend.app.schemas.predict import ImagePredictionResponse
from backend.app.services.prediction_service import prediction_service

router = APIRouter()


@router.post("/image", response_model=ImagePredictionResponse, tags=["Prediction"])
async def predict_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, etc.) to analyze"),
    confidence_threshold: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Minimum confidence threshold (0.0 to 1.0)"
    ),
    allow_fallback: bool = Query(
        False, description="If True, treats entire image as face when face detector detects no faces"
    ),
) -> ImagePredictionResponse:
    """
    Detects faces in the uploaded image and predicts facial expressions/emotions
    for each detected face.
    """
    contents = await file.read()
    return prediction_service.predict_image(
        image_bytes=contents,
        confidence_threshold=confidence_threshold,
        allow_fallback=allow_fallback,
    )
