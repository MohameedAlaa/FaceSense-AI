import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch

from backend.app.core.config import settings
from backend.app.schemas.model import ModelInfoResponse

logger = logging.getLogger(__name__)

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class ModelInfoService:
    def __init__(self, metadata_path: Optional[Path] = None):
        self.metadata_path = metadata_path or settings.MODEL_METADATA_PATH

    def get_model_info(self) -> ModelInfoResponse:
        """
        Loads production model metadata and returns structured model info.
        Prevents local absolute filesystem paths from leaking in API responses.
        """
        raw_metadata: Dict[str, Any] = {}
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    raw_metadata = json.load(f)
            except Exception as e:
                logger.warning("Failed to read model metadata: %s", e)

        # Sanitize metrics and remove local path references
        sanitized_metrics = {}
        for key, val in raw_metadata.items():
            if "path" in key.lower() or "backup" in key.lower():
                continue
            if key not in {"model_architecture", "model_version"}:
                sanitized_metrics[key] = val

        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        model_version = raw_metadata.get("model_version", "ResidualEmotionCNN-Candidate-epoch39")
        architecture = raw_metadata.get("model_architecture", "ResidualEmotionCNN")

        return ModelInfoResponse(
            model_name=model_version,
            architecture=architecture,
            num_classes=len(EMOTION_CLASSES),
            classes=EMOTION_CLASSES,
            device=device_name,
            input_shape=[1, 1, 48, 48],
            status="production",
            metrics=sanitized_metrics if sanitized_metrics else None,
        )


model_info_service = ModelInfoService()
