from typing import List, Dict
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: int = Field(..., description="X coordinate of top-left corner")
    y: int = Field(..., description="Y coordinate of top-left corner")
    w: int = Field(..., description="Width of bounding box")
    h: int = Field(..., description="Height of bounding box")


class FacePredictionItem(BaseModel):
    box: BoundingBox
    emotion: str
    confidence: float
    all_probabilities: Dict[str, float]


class ImagePredictionResponse(BaseModel):
    faces_detected: int
    predictions: List[FacePredictionItem]
    processing_time_ms: float
