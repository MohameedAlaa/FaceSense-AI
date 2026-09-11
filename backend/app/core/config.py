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
    CORS_ALLOWED_ORIGINS: str = Field(
        default=os.getenv("CORS_ALLOWED_ORIGINS", "*"),
        description="Comma-separated list of allowed origins (or * for all)"
    )
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    @property
    def get_cors_origins(self) -> List[str]:
        if self.CORS_ALLOWED_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ALLOWED_ORIGINS.split(",")]

    # Database settings (PostgreSQL)
    DATABASE_URL: Optional[str] = Field(
        default=os.getenv("DATABASE_URL", None),
        description="PostgreSQL database connection URL (e.g. postgresql+psycopg://user:pass@host:5432/dbname)"
    )
    DATABASE_ECHO: bool = Field(
        default=os.getenv("DATABASE_ECHO", "false").lower() in ("true", "1", "yes"),
        description="Whether SQLAlchemy should log SQL statements"
    )
    DATABASE_POOL_SIZE: int = Field(
        default=int(os.getenv("DATABASE_POOL_SIZE", "5")),
        description="The number of connections to keep open inside the connection pool"
    )
    DATABASE_MAX_OVERFLOW: int = Field(
        default=int(os.getenv("DATABASE_MAX_OVERFLOW", "10")),
        description="The number of connections to allow in overflow"
    )
    DATABASE_POOL_TIMEOUT: int = Field(
        default=int(os.getenv("DATABASE_POOL_TIMEOUT", "30")),
        description="The number of seconds to wait before giving up on returning a connection"
    )

    # Security settings
    JWT_SECRET_KEY: str = Field(
        default=os.getenv("JWT_SECRET_KEY", "insecure-dev-secret-key-please-change-in-prod"),
        description="Secret key for JWT generation"
    )
    JWT_ALGORITHM: str = Field(default=os.getenv("JWT_ALGORITHM", "HS256"))
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60")))

    # Upload settings
    MAX_IMAGE_UPLOAD_MB: int = Field(default=int(os.getenv("MAX_IMAGE_UPLOAD_MB", "10")))
    ALLOWED_IMAGE_MIMES: List[str] = ["image/jpeg", "image/png", "image/webp"]

    # Rate limiting
    RATE_LIMIT_ENABLED: bool = Field(default=os.getenv("RATE_LIMIT_ENABLED", "true").lower() in ("true", "1", "yes"))
    RATE_LIMIT_PREDICT: str = Field(default=os.getenv("RATE_LIMIT_PREDICT", "30/minute"))
    RATE_LIMIT_FEEDBACK: str = Field(default=os.getenv("RATE_LIMIT_FEEDBACK", "60/minute"))
    RATE_LIMIT_AUTH: str = Field(default=os.getenv("RATE_LIMIT_AUTH", "10/minute"))

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
