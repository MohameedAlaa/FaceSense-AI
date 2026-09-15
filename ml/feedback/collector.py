"""
FaceSense AI - Feedback Collector
Collects, validates, and records prediction feedback (correct, incorrect, uncertain)
in structured JSON Lines (JSONL) and optional CSV format.
"""

from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import numpy as np
from PIL import Image

# Default valid FER2013 emotion classes
DEFAULT_VALID_EMOTIONS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
UNCERTAIN_LABEL = "uncertain"
DEFAULT_METADATA_PATH = "ml/models/checkpoints/final/model_metadata.json"


def get_default_model_version(metadata_path: Union[str, Path] = DEFAULT_METADATA_PATH) -> str:
    """Reads the current promoted model version from model_metadata.json, with fallback."""
    meta_p = Path(metadata_path)
    if meta_p.exists():
        try:
            with open(meta_p, "r", encoding="utf-8") as f:
                data = json.load(f)
                v = data.get("model_version")
                if v:
                    return str(v)
        except Exception:
            pass
    return "ResidualEmotionCNN-Candidate-epoch39"


class FeedbackState(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNCERTAIN = "uncertain"


class FeedbackCollector:
    """
    Manages collection, validation, storage, and querying of user feedback records.
    Every feedback record stores:
      - feedback_id: Unique identifier
      - state: 'correct', 'incorrect', or 'uncertain'
      - image_path: Path to the stored face or frame image
      - bounding_box: [x, y, w, h] integer coordinates
      - predicted_emotion: Emotion predicted by the model (or 'uncertain')
      - confidence: Prediction confidence score [0.0, 1.0]
      - corrected_emotion: True/user-verified emotion label (or None)
      - timestamp: ISO 8601 formatted timestamp string
      - model_version: Model identifier and checkpoint metadata
      - metadata: Optional arbitrary dictionary for environment, camera_id, etc.
    """

    def __init__(
        self,
        feedback_dir: Union[str, Path] = "outputs/feedback",
        log_filename: str = "feedback_records.jsonl",
        valid_emotions: Optional[Union[List[str], Tuple[str, ...]]] = None,
        default_model_version: Optional[str] = None,
    ):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.feedback_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.feedback_dir / log_filename
        self.valid_emotions = tuple(valid_emotions) if valid_emotions else DEFAULT_VALID_EMOTIONS
        self.default_model_version = (
            str(default_model_version)
            if default_model_version is not None
            else get_default_model_version()
        )

    def validate_record(
        self,
        image_path: Optional[Union[str, Path]],
        predicted_emotion: str,
        confidence: float,
        corrected_emotion: Optional[str] = None,
        bounding_box: Optional[Union[List[int], Tuple[int, ...]]] = None,
        timestamp: Optional[str] = None,
        state: Union[str, FeedbackState] = FeedbackState.CORRECT,
    ) -> Dict[str, Any]:
        """
        Validates all constraints on feedback components before persistence.

        Raises:
            FileNotFoundError: If image_path is provided but does not exist on disk.
            ValueError: For invalid emotions, invalid confidence range, invalid bounding box, or invalid timestamp.
        """
        # 1. Image path validation
        stored_img_path: Optional[str] = None
        if image_path is not None:
            img_p = Path(image_path)
            if not img_p.exists() or not img_p.is_file():
                raise FileNotFoundError(f"Feedback image path does not exist: {img_p.resolve()}")
            stored_img_path = str(img_p.resolve())

        # 2. Predicted emotion validation
        pred_clean = str(predicted_emotion).strip().lower()
        if pred_clean not in self.valid_emotions and pred_clean != UNCERTAIN_LABEL:
            raise ValueError(
                f"Invalid predicted emotion '{predicted_emotion}'. Must be one of {self.valid_emotions} or '{UNCERTAIN_LABEL}'."
            )

        # 3. Confidence score validation
        try:
            conf_val = float(confidence)
        except (TypeError, ValueError):
            raise ValueError(f"Confidence must be a numeric float, got '{confidence}'.")
        if not (0.0 <= conf_val <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0 inclusive, got {conf_val}.")

        # 4. Corrected emotion validation
        corr_clean: Optional[str] = None
        if corrected_emotion is not None and str(corrected_emotion).strip() != "":
            corr_clean = str(corrected_emotion).strip().lower()
            if corr_clean not in self.valid_emotions:
                raise ValueError(
                    f"Invalid corrected emotion '{corrected_emotion}'. Must be one of {self.valid_emotions}."
                )

        # 5. State validation
        if isinstance(state, FeedbackState):
            state_val = state.value
        else:
            state_val = str(state).strip().lower()
            valid_states = [s.value for s in FeedbackState]
            if state_val not in valid_states:
                raise ValueError(f"Invalid feedback state '{state}'. Expected one of {valid_states}.")

        # For incorrect feedback, a corrected emotion is strongly expected or recommended
        if state_val == FeedbackState.INCORRECT.value and corr_clean is None:
            # We allow it with a fallback or warning, but validate if provided
            pass

        # 6. Bounding box validation
        bbox_list: Optional[List[int]] = None
        if bounding_box is not None:
            if not isinstance(bounding_box, (list, tuple)) or len(bounding_box) != 4:
                raise ValueError(f"Bounding box must be a 4-element sequence [x, y, w, h], got {bounding_box}.")
            try:
                bbox_list = [int(v) for v in bounding_box]
            except (TypeError, ValueError):
                raise ValueError(f"Bounding box elements must be integers, got {bounding_box}.")
            x, y, w, h = bbox_list
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                raise ValueError(f"Bounding box coordinates must be non-negative with positive width/height: [x={x}, y={y}, w={w}, h={h}].")

        # 7. Timestamp validation
        if timestamp is not None:
            try:
                datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                ts_val = timestamp
            except Exception as e:
                raise ValueError(f"Timestamp must be a valid ISO 8601 string, got '{timestamp}': {e}")
        else:
            ts_val = datetime.now(timezone.utc).isoformat()

        return {
            "image_path": stored_img_path,
            "predicted_emotion": pred_clean,
            "confidence": conf_val,
            "corrected_emotion": corr_clean,
            "bounding_box": bbox_list,
            "timestamp": ts_val,
            "state": state_val,
        }

    def add_feedback(
        self,
        image_path: Optional[Union[str, Path]],
        predicted_emotion: str,
        confidence: float,
        corrected_emotion: Optional[str] = None,
        bounding_box: Optional[Union[List[int], Tuple[int, ...]]] = None,
        state: Union[str, FeedbackState] = FeedbackState.CORRECT,
        model_version: Optional[str] = None,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        copy_image: bool = False,
    ) -> Dict[str, Any]:
        """
        Validates, records, and persists a single feedback entry.

        Args:
            image_path: Path to the image file (optional).
            predicted_emotion: Label predicted by model.
            confidence: Float confidence score in [0.0, 1.0].
            corrected_emotion: User ground-truth corrected label.
            bounding_box: [x, y, w, h] face coordinates.
            state: FeedbackState or string ('correct', 'incorrect', 'uncertain').
            model_version: Identifier string of model producing prediction.
            timestamp: ISO formatted timestamp (optional, defaults to now UTC).
            metadata: Additional contextual info (e.g. inference time, device, notes).
            copy_image: If True, copies the image to outputs/feedback/images/ for permanent retention.

        Returns:
            The complete recorded feedback dictionary.
        """
        validated = self.validate_record(
            image_path=image_path,
            predicted_emotion=predicted_emotion,
            confidence=confidence,
            corrected_emotion=corrected_emotion,
            bounding_box=bounding_box,
            timestamp=timestamp,
            state=state,
        )

        if validated["image_path"] is not None:
            resolved_img = Path(validated["image_path"])
            if copy_image:
                img_filename = f"fb_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}_{resolved_img.name}"
                dest_path = self.images_dir / img_filename
                import shutil
                shutil.copy2(resolved_img, dest_path)
                stored_img_path = str(dest_path.resolve())
            else:
                stored_img_path = validated["image_path"]
        else:
            stored_img_path = None

        resolved_model_version = str(model_version) if model_version is not None else self.default_model_version

        record = {
            "feedback_id": f"fb_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:6]}",
            "state": validated["state"],
            "image_path": stored_img_path,
            "bounding_box": validated["bounding_box"],
            "predicted_emotion": validated["predicted_emotion"],
            "confidence": validated["confidence"],
            "corrected_emotion": validated["corrected_emotion"],
            "timestamp": validated["timestamp"],
            "model_version": resolved_model_version,
            "metadata": metadata or {},
        }

        # Append to JSONL file
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record

    def save_face_crop_and_add_feedback(
        self,
        frame_or_face: Union[np.ndarray, Image.Image],
        predicted_emotion: str,
        confidence: float,
        corrected_emotion: Optional[str] = None,
        bounding_box: Optional[Union[List[int], Tuple[int, ...]]] = None,
        state: Union[str, FeedbackState] = FeedbackState.CORRECT,
        model_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Saves a numpy array or PIL image directly to the feedback images directory,
        then records the feedback. Useful for webcam or runtime live streams.
        """
        ts_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        img_name = f"crop_{ts_str}.jpg"
        save_path = self.images_dir / img_name

        if isinstance(frame_or_face, Image.Image):
            frame_or_face.save(save_path)
        elif isinstance(frame_or_face, np.ndarray):
            import cv2
            if frame_or_face.ndim == 2:
                cv2.imwrite(str(save_path), frame_or_face)
            elif frame_or_face.ndim == 3 and frame_or_face.shape[2] == 3:
                cv2.imwrite(str(save_path), frame_or_face)
            else:
                raise ValueError(f"Unsupported numpy array dimensions: {frame_or_face.shape}")
        else:
            raise TypeError(f"Expected PIL Image or numpy array, got {type(frame_or_face)}")

        return self.add_feedback(
            image_path=save_path,
            predicted_emotion=predicted_emotion,
            confidence=confidence,
            corrected_emotion=corrected_emotion,
            bounding_box=bounding_box,
            state=state,
            model_version=model_version,
            metadata=metadata,
            copy_image=False,
        )

    def load_feedback_records(
        self,
        filter_state: Optional[Union[str, FeedbackState]] = None,
        filter_model_version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Loads all feedback records from the JSONL log file with optional filtering.
        """
        if not self.log_path.exists():
            return []

        records = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    rec = json.loads(line_str)
                    records.append(rec)
                except json.JSONDecodeError:
                    continue

        if filter_state is not None:
            target_state = filter_state.value if isinstance(filter_state, FeedbackState) else str(filter_state).lower()
            records = [r for r in records if r.get("state") == target_state]

        if filter_model_version is not None:
            records = [r for r in records if r.get("model_version") == filter_model_version]

        return records

    def export_to_csv(self, output_csv_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Exports all current feedback records to a CSV spreadsheet.
        """
        import csv

        csv_path = Path(output_csv_path) if output_csv_path else self.feedback_dir / "feedback_records.csv"
        records = self.load_feedback_records()

        fieldnames = [
            "feedback_id",
            "state",
            "image_path",
            "bounding_box",
            "predicted_emotion",
            "confidence",
            "corrected_emotion",
            "timestamp",
            "model_version",
            "metadata",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                row = dict(rec)
                if isinstance(row.get("bounding_box"), list):
                    row["bounding_box"] = json.dumps(row["bounding_box"])
                if isinstance(row.get("metadata"), dict):
                    row["metadata"] = json.dumps(row["metadata"])
                writer.writerow(row)

        return csv_path

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Computes count summaries by feedback state, emotions, and model version.
        """
        records = self.load_feedback_records()
        stats: Dict[str, Any] = {
            "total_records": len(records),
            "by_state": {
                FeedbackState.CORRECT.value: 0,
                FeedbackState.INCORRECT.value: 0,
                FeedbackState.UNCERTAIN.value: 0,
            },
            "by_predicted_emotion": {},
            "by_corrected_emotion": {},
            "by_model_version": {},
        }

        for r in records:
            st = r.get("state", "unknown")
            stats["by_state"][st] = stats["by_state"].get(st, 0) + 1

            p_emo = r.get("predicted_emotion")
            if p_emo:
                stats["by_predicted_emotion"][p_emo] = stats["by_predicted_emotion"].get(p_emo, 0) + 1

            c_emo = r.get("corrected_emotion")
            if c_emo:
                stats["by_corrected_emotion"][c_emo] = stats["by_corrected_emotion"].get(c_emo, 0) + 1

            mv = r.get("model_version", "unknown")
            stats["by_model_version"][mv] = stats["by_model_version"].get(mv, 0) + 1

        return stats
