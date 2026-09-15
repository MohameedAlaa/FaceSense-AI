import time
import logging
from typing import Optional, List, Tuple
import cv2
import numpy as np

from backend.app.core.config import settings
from backend.app.schemas.predict import BoundingBox, FacePredictionItem, ImagePredictionResponse
from ml.detection.detector import FaceDetector
from ml.inference.predictor import EmotionPredictor

logger = logging.getLogger(__name__)


class BackendError(Exception):
    """Base exception for backend service errors."""
    pass


class ImageDecodeError(BackendError):
    """Raised when uploaded image bytes cannot be decoded."""
    pass


class NoFaceDetectedError(BackendError):
    """Raised when no face is found in an image and fallback is not permitted."""
    pass


class ImagePredictionService:
    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        predictor: Optional[EmotionPredictor] = None,
    ):
        self._detector = detector
        self._predictor = predictor

    @property
    def detector(self) -> FaceDetector:
        if self._detector is None:
            cascade_p = str(settings.HAAR_CASCADE_PATH) if settings.HAAR_CASCADE_PATH else None
            self._detector = FaceDetector(
                cascade_path=cascade_p
            )
        return self._detector

    @property
    def predictor(self) -> EmotionPredictor:
        if self._predictor is None:
            self._predictor = EmotionPredictor(
                checkpoint_path=str(settings.MODEL_CHECKPOINT_PATH),
                confidence_threshold=settings.DEFAULT_CONFIDENCE_THRESHOLD,
            )
        return self._predictor

    def predict_image(
        self,
        image_bytes: bytes,
        confidence_threshold: Optional[float] = None,
        allow_fallback: bool = False,
    ) -> ImagePredictionResponse:
        """
        Runs face detection and emotion classification on an image provided as bytes.

        Args:
            image_bytes: Raw bytes of the uploaded image.
            confidence_threshold: Optional threshold override.
            allow_fallback: If True, falls back to full frame if no faces detected.

        Returns:
            ImagePredictionResponse containing detected faces and emotion predictions.

        Raises:
            ImageDecodeError: If image bytes cannot be decoded.
            NoFaceDetectedError: If no face is detected and allow_fallback is False.
        """
        if not image_bytes:
            raise ImageDecodeError("Empty image file provided.")

        start_time = time.perf_counter()

        np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_bgr is None or img_bgr.size == 0:
            raise ImageDecodeError("Could not decode image content. Please provide a valid JPEG or PNG file.")

        bboxes = self.detector.detect_faces(img_bgr)
        is_fallback = False

        if len(bboxes) == 0:
            if allow_fallback:
                h, w = img_bgr.shape[:2]
                bboxes = [(0, 0, int(w), int(h))]
                is_fallback = True
            else:
                raise NoFaceDetectedError("No face detected in the provided image.")

        threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.DEFAULT_CONFIDENCE_THRESHOLD
        )

        face_crops = []
        for bbox in bboxes:
            if is_fallback:
                face_crop = img_bgr.copy()
            else:
                face_crop = self.detector.crop_face(img_bgr, bbox, margin_ratio=0.05)
            face_crops.append(face_crop)

        batch_results = self.predictor.predict_batch(
            face_crops,
            confidence_threshold=threshold,
        )

        predictions: List[FacePredictionItem] = []
        for bbox, pred_result in zip(bboxes, batch_results):
            x, y, w, h = bbox
            predictions.append(
                FacePredictionItem(
                    box=BoundingBox(x=int(x), y=int(y), w=int(w), h=int(h)),
                    emotion=pred_result["predicted_emotion"],
                    confidence=float(pred_result["confidence"]),
                    all_probabilities={
                        k: float(v) for k, v in pred_result.get("probabilities", {}).items()
                    },
                )
            )

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return ImagePredictionResponse(
            faces_detected=len(predictions),
            predictions=predictions,
            processing_time_ms=round(duration_ms, 2),
        )


prediction_service = ImagePredictionService()
