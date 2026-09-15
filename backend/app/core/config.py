import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

# Base workspace path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _load_env_file() -> None:
    """Load .env from the workspace root into os.environ (keys already set are not overwritten).

    This is a lightweight alternative to python-dotenv. It only handles
    simple KEY=VALUE lines; comments (#) and blank lines are skipped.
    It runs before the Settings singleton is constructed so that
    JWT_SECRET_KEY (and other secrets) can be provided via .env in
    development and CI without any additional dependencies.
    """
    env_file = WORKSPACE_ROOT / ".env"
    if not env_file.is_file():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip optional surrounding quotes
        if len(value) >= 2 and value[0] in ('"', "'") and value[0] == value[-1]:
            value = value[1:-1]
        # Never overwrite values already set in the real environment
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()


class Settings(BaseModel):
    PROJECT_NAME: str = "FaceSense AI API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DESCRIPTION: str = "REST API foundation for FaceSense AI Emotion Detection and Feedback Management"

    # CORS settings
    CORS_ALLOWED_ORIGINS: str = Field(
        default=os.getenv(
            "CORS_ALLOWED_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ),
        description="Comma-separated list of allowed origins (explicit origins required when credentials enabled)",
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(
        default=os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() in ("true", "1", "yes"),
        description="Whether to allow credentials in cross-origin requests",
    )
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    @property
    def get_cors_origins(self) -> List[str]:
        raw = self.CORS_ALLOWED_ORIGINS
        if not raw or not raw.strip():
            return []
        origins = [
            origin.strip().rstrip("/") if origin.strip() != "/" else "/"
            for origin in raw.split(",")
            if origin.strip()
        ]
        if "*" in origins and self.CORS_ALLOW_CREDENTIALS:
            raise ValueError(
                "Unsafe CORS configuration: wildcard '*' origin cannot be used when CORS credentials are enabled."
            )
        return origins

    @model_validator(mode="after")
    def validate_cors_credentials_and_origins(self) -> "Settings":
        raw = self.CORS_ALLOWED_ORIGINS
        if raw is not None:
            origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
            if "*" in origins and self.CORS_ALLOW_CREDENTIALS:
                raise ValueError(
                    "Unsafe CORS configuration: wildcard '*' origin cannot be used when CORS credentials are enabled. "
                    "Configure explicit origins (e.g. 'http://localhost:5173,http://127.0.0.1:5173')."
                )
        return self

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
    # JWT_SECRET_KEY is mandatory. Set it via the JWT_SECRET_KEY environment variable.
    # The application will refuse to start if it is missing or empty.
    JWT_SECRET_KEY: Optional[str] = Field(
        default=os.getenv("JWT_SECRET_KEY") or None,
        description="Secret key for JWT generation (mandatory – must be set via JWT_SECRET_KEY env var)"
    )

    @field_validator("JWT_SECRET_KEY")
    @classmethod
    def validate_jwt_secret_key(cls, v: Optional[str]) -> str:
        """Reject missing, empty, whitespace-only, or known-insecure placeholder values at startup."""
        if not v or not v.strip():
            raise ValueError(
                "JWT_SECRET_KEY is required. "
                "Set it via the JWT_SECRET_KEY environment variable (see .env.example)."
            )
        if v == "insecure-dev-secret-key-please-change-in-prod":
            raise ValueError(
                "JWT_SECRET_KEY must not use the insecure placeholder value. "
                "Generate a strong random secret and set it via the JWT_SECRET_KEY environment variable."
            )
        return v

    JWT_ALGORITHM: str = Field(default=os.getenv("JWT_ALGORITHM", "HS256"))
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60")))

    # Admin bootstrap settings
    # ADMIN_BOOTSTRAP_KEY is an optional one-time secret used to create the first admin account.
    # If absent or empty, the admin bootstrap endpoint is effectively disabled.
    # Generate with: python -c "import secrets; print(secrets.token_hex(32))"
    # Never hardcode a real value here; set it only in .env (gitignored).
    ADMIN_BOOTSTRAP_KEY: Optional[str] = Field(
        default=os.getenv("ADMIN_BOOTSTRAP_KEY") or None,
        description="One-time secret key required to create the initial admin account (optional)"
    )


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
