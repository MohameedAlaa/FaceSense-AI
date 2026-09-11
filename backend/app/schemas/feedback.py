from typing import Optional, Dict
from pydantic import BaseModel, Field, field_validator


class FeedbackCreateRequest(BaseModel):
    feedback_type: str = Field(..., description="'correct', 'incorrect', or 'uncertain'")
    predicted_emotion: str = Field(..., description="Predicted emotion label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    corrected_emotion: Optional[str] = Field(None, description="Corrected emotion if feedback_type is 'incorrect'")
    image_base64: Optional[str] = Field(None, description="Optional Base64-encoded face crop image")
    notes: Optional[str] = Field(None, description="Optional user notes or remarks")

    @field_validator("feedback_type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        valid_types = {"correct", "incorrect", "uncertain"}
        v_clean = v.strip().lower()
        if v_clean not in valid_types:
            raise ValueError(f"feedback_type must be one of {sorted(valid_types)}")
        return v_clean


class FeedbackCreateResponse(BaseModel):
    status: str = "success"
    record_id: str
    message: str


class FeedbackStatsResponse(BaseModel):
    total_records: int
    correct: int
    incorrect: int
    uncertain: int
    emotions: Dict[str, int]
    accuracy_rate: Optional[float] = None
