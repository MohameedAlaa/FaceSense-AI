"""
FaceSense AI - Image Inference Pipeline
Provides ImageInferencePipeline for running face detection, cropping,
emotion prediction, and visual annotation on static images and directories.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from ml.detection.detector import FaceDetector
from ml.inference.predictor import EmotionPredictor

# Consistent emotion color palette (BGR) matching webcam HUD
EMOTION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "angry": (0, 0, 255),        # Red
    "disgust": (0, 140, 255),    # Orange/Dark Gold
    "fear": (128, 0, 128),       # Purple
    "happy": (0, 255, 0),        # Green
    "neutral": (200, 200, 200),  # Light Gray
    "sad": (255, 150, 0),        # Blue/Teal
    "surprise": (0, 255, 255),   # Yellow
    "uncertain": (100, 100, 100) # Dark Gray
}


def load_image_bgr(image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Safely loads or converts an input into a BGR numpy array uint8 format.

    Args:
        image_input: File path (str/Path), PIL Image, or numpy ndarray.

    Returns:
        np.ndarray: BGR image array with shape (H, W, 3).

    Raises:
        FileNotFoundError: If image file path does not exist.
        ValueError: If file cannot be read or array format is invalid.
    """
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            raise FileNotFoundError(f"Image file not found: {p.resolve()}")
        if not p.is_file():
            raise ValueError(f"Path is not a regular file: {p.resolve()}")

        # Use imdecode to be resilient across Windows unicode paths
        try:
            with open(p, "rb") as f:
                bytes_data = bytearray(f.read())
            np_arr = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not decode image content from '{p}'.")
            return img
        except Exception as e:
            raise ValueError(f"Failed loading image '{p}': {e}")

    elif isinstance(image_input, Image.Image):
        rgb_img = image_input.convert("RGB")
        return cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)

    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            raise ValueError("Input numpy array is empty.")
        if image_input.ndim == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        elif image_input.ndim == 3:
            if image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
            elif image_input.shape[2] == 3:
                return image_input.copy()
            elif image_input.shape[2] == 1:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        raise ValueError(f"Unsupported numpy array dimensions: {image_input.shape}")

    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")


class ImageInferencePipeline:
    """
    Modular image inference pipeline:
      Image -> Face Detection -> Crop each face -> Emotion Prediction -> Annotation
    """

    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        predictor: Optional[EmotionPredictor] = None,
        confidence_threshold: float = 0.35,
    ):
        """
        Args:
            detector: FaceDetector instance (instantiates default if None).
            predictor: EmotionPredictor instance (instantiates default if None).
            confidence_threshold: Confidence score cutoff below which prediction is labeled 'uncertain'.
        """
        self.detector = detector or FaceDetector()
        self.predictor = predictor or EmotionPredictor(confidence_threshold=confidence_threshold)
        self.confidence_threshold = confidence_threshold

    def process_image(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        margin_ratio: float = 0.05,
        fallback_to_full_frame: bool = True,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Runs face detection and emotion prediction across all detected faces in an image.

        Args:
            image_input: File path, PIL Image, or numpy array.
            margin_ratio: Margin to add around bounding box when cropping faces.
            fallback_to_full_frame: If True and no faces are detected by Haar cascade,
                                   treats the entire image as a single face [0, 0, W, H].

        Returns:
            Tuple of:
              - frame_bgr: The original full image in BGR format.
              - predictions: List of prediction dictionaries for every detected face:
                  - face_idx: 1-based index (1, 2, ...)
                  - bbox: (x, y, w, h) bounding box coordinates
                  - predicted_emotion: Predicted emotion label (or 'uncertain')
                  - confidence: Confidence score float in [0.0, 1.0]
                  - probabilities: Dict mapping all classes to probabilities
                  - is_uncertain: Boolean flag
                  - raw_predicted_emotion: Top class before thresholding
                  - model_name: Name of model
                  - model_version: Version identifier string
                  - face_crop: Cropped face BGR ndarray
        """
        frame_bgr = load_image_bgr(image_input)
        bboxes = self.detector.detect_faces(frame_bgr)
        is_fallback = False

        if len(bboxes) == 0 and fallback_to_full_frame:
            h, w = frame_bgr.shape[:2]
            bboxes = [(0, 0, int(w), int(h))]
            is_fallback = True

        predictions: List[Dict[str, Any]] = []

        for i, bbox in enumerate(bboxes):
            try:
                if is_fallback:
                    face_crop = frame_bgr.copy()
                else:
                    face_crop = self.detector.crop_face(frame_bgr, bbox, margin_ratio=margin_ratio)
                pred_result = self.predictor.predict(
                    face_crop,
                    confidence_threshold=self.confidence_threshold,
                )
                pred_result["face_idx"] = i + 1
                pred_result["bbox"] = bbox
                pred_result["face_crop"] = face_crop
                pred_result["is_fallback"] = is_fallback
                predictions.append(pred_result)
            except Exception as e:
                # Log or handle unexpected crop error
                pred_result = {
                    "face_idx": i + 1,
                    "bbox": bbox,
                    "predicted_emotion": "error",
                    "confidence": 0.0,
                    "probabilities": {},
                    "is_uncertain": True,
                    "error": str(e),
                    "face_crop": None,
                }
                predictions.append(pred_result)

        return frame_bgr, predictions

    def annotate_image(
        self,
        frame: np.ndarray,
        predictions: List[Dict[str, Any]],
        selected_face_idx: int = 0,
    ) -> np.ndarray:
        """
        Renders bounding boxes, emotion labels, confidence scores, and selection markers.

        Args:
            frame: Input BGR image array.
            predictions: List of face prediction dicts from process_image().
            selected_face_idx: 0-based index of the currently selected face (highlighted).

        Returns:
            np.ndarray: Annotated BGR image copy.
        """
        annotated = frame.copy()

        for i, pred in enumerate(predictions):
            bbox = pred.get("bbox")
            if not bbox:
                continue

            x, y, w, h = bbox
            emotion = pred.get("predicted_emotion", "unknown").lower()
            conf = pred.get("confidence", 0.0) * 100.0
            is_selected = (i == selected_face_idx)

            color = EMOTION_COLORS.get(emotion, (0, 255, 0))

            # 1. Bounding box rectangle
            box_thickness = 3 if is_selected else 2
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, box_thickness)

            # 2. Selected face accent corners
            if is_selected:
                corner_len = max(8, min(20, w // 4, h // 4))
                accent_col = (0, 215, 255)  # Gold highlight
                # Top-left
                cv2.line(annotated, (x - 3, y - 3), (x - 3 + corner_len, y - 3), accent_col, 2)
                cv2.line(annotated, (x - 3, y - 3), (x - 3, y - 3 + corner_len), accent_col, 2)
                # Top-right
                cv2.line(annotated, (x + w + 3, y - 3), (x + w + 3 - corner_len, y - 3), accent_col, 2)
                cv2.line(annotated, (x + w + 3, y - 3), (x + w + 3, y - 3 + corner_len), accent_col, 2)
                # Bottom-left
                cv2.line(annotated, (x - 3, y + h + 3), (x - 3 + corner_len, y + h + 3), accent_col, 2)
                cv2.line(annotated, (x - 3, y + h + 3), (x - 3, y + h + 3 - corner_len), accent_col, 2)
                # Bottom-right
                cv2.line(annotated, (x + w + 3, y + h + 3), (x + w + 3 - corner_len, y + h + 3), accent_col, 2)
                cv2.line(annotated, (x + w + 3, y + h + 3), (x + w + 3, y + h + 3 - corner_len), accent_col, 2)

            # 3. Label tag
            face_num = pred.get("face_idx", i + 1)
            sel_tag = f"[{face_num}*] " if is_selected else f"[{face_num}] "
            if pred.get("is_uncertain", False):
                label_text = f"{sel_tag}Uncertain ({conf:.1f}%)"
            else:
                label_text = f"{sel_tag}{emotion.capitalize()} ({conf:.1f}%)"

            (text_w, text_h), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            pill_y1 = max(0, y - text_h - 10)
            pill_y2 = y
            pill_x2 = min(annotated.shape[1], x + text_w + 10)

            cv2.rectangle(annotated, (x, pill_y1), (pill_x2, pill_y2), color, cv2.FILLED)
            text_color = (0, 0, 0) if emotion in ["happy", "surprise", "neutral"] else (255, 255, 255)
            cv2.putText(
                annotated,
                label_text,
                (x + 5, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                text_color,
                2,
                cv2.LINE_AA,
            )

        return annotated

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_annotated: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Runs batch face detection and emotion prediction on all supported images in a directory.

        Args:
            input_dir: Path to directory containing images.
            output_dir: Optional path to save annotated images.
            save_annotated: Whether to save annotated images to output_dir.

        Returns:
            List of structured dictionary summaries for each processed image.
        """
        dir_p = Path(input_dir)
        if not dir_p.exists() or not dir_p.is_dir():
            raise FileNotFoundError(f"Input directory not found: {dir_p.resolve()}")

        out_p: Optional[Path] = None
        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = sorted([
            f for f in dir_p.iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions
        ])

        results: List[Dict[str, Any]] = []

        for img_file in image_files:
            try:
                frame, preds = self.process_image(img_file)
                annotated_path = None
                if save_annotated and out_p is not None:
                    annotated_frame = self.annotate_image(frame, preds)
                    annotated_path = str((out_p / f"annotated_{img_file.name}").resolve())
                    cv2.imwrite(annotated_path, annotated_frame)

                # Clean face crops from serialization summary
                cleaned_preds = []
                for p in preds:
                    cp = dict(p)
                    cp.pop("face_crop", None)
                    cleaned_preds.append(cp)

                results.append({
                    "image_file": img_file.name,
                    "image_path": str(img_file.resolve()),
                    "faces_detected": len(preds),
                    "predictions": cleaned_preds,
                    "annotated_output": annotated_path,
                })
            except Exception as e:
                results.append({
                    "image_file": img_file.name,
                    "image_path": str(img_file.resolve()),
                    "faces_detected": 0,
                    "predictions": [],
                    "error": str(e),
                })

        return results
