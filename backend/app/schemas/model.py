from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class ModelInfoResponse(BaseModel):
    model_name: str
    architecture: str
    num_classes: int
    classes: List[str]
    device: str
    input_shape: List[int]
    status: str
    metrics: Optional[Dict[str, Any]] = None
