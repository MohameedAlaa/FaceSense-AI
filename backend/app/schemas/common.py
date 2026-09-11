from typing import Optional
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "FaceSense AI API"
    version: str = "1.0.0"
    database: Optional[str] = None
