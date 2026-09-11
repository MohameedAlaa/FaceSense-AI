import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

# Base workspace path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseModel):
    PROJECT_NAME: str = "FaceSense AI API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DESCRIPTION: str = "REST API foundation for FaceSense AI Emotion Detection and Feedback Management"

    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    # Database placeholder (PostgreSQL in future phases)
    DATABASE_URL: Optional[str] = Field(
        default=os.getenv("DATABASE_URL", None),
        description="Placeholder for future PostgreSQL database connection URL"
    )

    # ML Configuration & Paths
    MODEL_CHECKPOINT_PATH: Path = WORKSPACE_ROOT / "ml" / "models" / "checkpoints" / "final" / "best_model.pt"
    MODEL_METADATA_PATH: Path = WORKSPACE_ROOT / "ml" / "models" / "checkpoints" / "final" / "model_metadata.json"
    HAAR_CASCADE_PATH: Optional[Path] = None
    FEEDBACK_DIR: Path = WORKSPACE_ROOT / "outputs" / "feedback"
    FEEDBACK_LOG_FILE: str = "feedback_records.jsonl"

    # Inference defaults
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.50

    model_config = ConfigDict(arbitrary_types_allowed=True)


settings = Settings()
