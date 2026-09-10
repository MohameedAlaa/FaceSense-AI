"""
FaceSense AI - Real-Time Webcam Facial Expression Recognition Application
Pipeline:
  Camera frame -> Face Detection -> Face Crop -> EmotionPredictor -> Overlays (BBox, Label, Confidence, FPS)

Keyboard Controls:
  - Q: Quit application
  - S: Save snapshot frame with annotations
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.detection.detector import FaceDetector
from ml.inference.predictor import EmotionPredictor


# Emotion color map for visual cues (BGR format)
EMOTION_COLORS = {
    "angry": (0, 0, 255),       # Red
    "disgust": (0, 140, 255),   # Orange/Dark Gold
    "fear": (128, 0, 128),      # Purple
    "happy": (0, 255, 0),       # Green
    "neutral": (200, 200, 200), # Light Gray
    "sad": (255, 150, 0),       # Blue/Teal
    "surprise": (0, 255, 255),  # Yellow
    "uncertain": (100, 100, 100)# Dark Gray
}


class WebcamEmotionApp:
    """
    Real-Time Webcam Face & Emotion Recognition Application.
    Orchestrates camera capture, multi-face detection, CNN prediction, and annotated rendering.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "ml/models/checkpoints/final/best_model.pt",
        camera_id: int = 0,
        confidence_threshold: float = 0.35,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        snapshots_dir: Union[str, Path] = "outputs/snapshots",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        self.snapshots_dir = Path(snapshots_dir)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        print("[FaceSense AI] Initializing Face Detector...")
        self.detector = FaceDetector(
            scale_factor=scale_factor,
            min_neighbors=min_neighbors,
        )

        print(f"[FaceSense AI] Loading Emotion Predictor from {self.checkpoint_path}...")
        self.predictor = EmotionPredictor(
            checkpoint_path=self.checkpoint_path,
            confidence_threshold=self.confidence_threshold,
        )

        self.fps = 0.0
        self._prev_time = time.time()

    def process_frame(
        self, frame: np.ndarray, draw_overlay: bool = True
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Processes a single BGR frame: detects faces, runs emotion predictions, and draws overlays.

        Args:
            frame: Input BGR frame.
            draw_overlay: Whether to draw bounding boxes and annotations onto returned frame.

        Returns:
            annotated_frame: Output BGR frame with visual overlays.
            face_predictions: List of dicts with bounding boxes and prediction results.
        """
        if frame is None or frame.size == 0:
            return frame, []

        display_frame = frame.copy() if draw_overlay else frame
        face_bboxes = self.detector.detect_faces(frame)
        face_predictions = []

        for bbox in face_bboxes:
            x, y, w, h = bbox
            try:
                face_crop = self.detector.crop_face(frame, bbox, margin_ratio=0.05)
                pred_result = self.predictor.predict(
                    face_crop, confidence_threshold=self.confidence_threshold
                )
                pred_result["bbox"] = bbox
                face_predictions.append(pred_result)

                if draw_overlay:
                    self._draw_face_annotation(display_frame, bbox, pred_result)
            except Exception as e:
                # If face crop or prediction fails for a single box, log and continue
                if draw_overlay:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(
                        display_frame,
                        "Error",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

        if draw_overlay:
            self._draw_status_bar(display_frame, num_faces=len(face_bboxes))

        return display_frame, face_predictions

    def _draw_face_annotation(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int], pred: dict
    ) -> None:
        """Draws bounding box, label tag, and confidence on a detected face."""
        x, y, w, h = bbox
        emotion = pred["predicted_emotion"].lower()
        conf = pred["confidence"] * 100.0

        color = EMOTION_COLORS.get(emotion, (0, 255, 0))

        # 1. Main bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # 2. Text label
        label_text = f"{emotion.capitalize()} ({conf:.1f}%)"
        if pred["is_uncertain"]:
            label_text = f"Uncertain ({conf:.1f}%)"

        # Label background pill
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        pill_y1 = max(0, y - text_h - 10)
        pill_y2 = y
        pill_x2 = min(frame.shape[1], x + text_w + 10)

        cv2.rectangle(frame, (x, pill_y1), (pill_x2, pill_y2), color, cv2.FILLED)
        # Black text for high contrast on bright pills, white on dark
        text_color = (0, 0, 0) if emotion in ["happy", "surprise", "neutral"] else (255, 255, 255)
        cv2.putText(
            frame,
            label_text,
            (x + 5, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2,
            cv2.LINE_AA,
        )

    def _draw_status_bar(self, frame: np.ndarray, num_faces: int) -> None:
        """Draws FPS and control hints overlay on top header of frame."""
        # Calculate current FPS
        curr_time = time.time()
        dt = curr_time - self._prev_time
        self._prev_time = curr_time
        if dt > 0:
            current_fps = 1.0 / dt
            self.fps = 0.9 * self.fps + 0.1 * current_fps if self.fps > 0 else current_fps

        # Header overlay banner
        h, w = frame.shape[:2]
        header_h = 35
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_h), (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Status text
        status_text = (
            f"FaceSense AI | Faces: {num_faces} | FPS: {self.fps:.1f} | "
            f"Thresh: {self.confidence_threshold:.2f} | [S] Save [Q] Quit"
        )
        cv2.putText(
            frame,
            status_text,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def run(self) -> None:
        """Starts real-time video capture loop from webcam."""
        print(f"[FaceSense AI] Connecting to camera ID {self.camera_id}...")
        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            print(
                f"[Error] Could not open webcam with ID {self.camera_id}.\n"
                f"Please check camera connections, privacy permissions, or try --camera 1.",
                file=sys.stderr,
            )
            return

        window_name = "FaceSense AI - Real-Time Facial Expression Recognition"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("\n" + "=" * 65)
        print("  FaceSense AI Live Webcam Running")
        print("  Controls:")
        print("    [Q] Quit application")
        print("    [S] Save annotated snapshot")
        print("=" * 65 + "\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[Warning] Failed to receive frame from webcam. Retrying...", file=sys.stderr)
                    time.sleep(0.1)
                    continue

                # Process detection and inference
                annotated_frame, preds = self.process_frame(frame, draw_overlay=True)

                cv2.imshow(window_name, annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord("Q") or key == 27:  # Q or ESC
                    print("[FaceSense AI] Quitting application.")
                    break
                elif key == ord("s") or key == ord("S"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snap_path = self.snapshots_dir / f"snapshot_{timestamp}.jpg"
                    cv2.imwrite(str(snap_path), annotated_frame)
                    print(f"[FaceSense AI] Saved snapshot to: {snap_path}")

        finally:
            cap.release()
            cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="FaceSense AI - Real-Time Webcam Facial Expression Recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--camera",
        "-c",
        type=int,
        default=0,
        help="Webcam device ID (default 0).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ml/models/checkpoints/final/best_model.pt",
        help="Path to ResidualEmotionCNN final checkpoint.",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.35,
        help="Confidence threshold below which prediction is labeled 'Uncertain'.",
    )
    parser.add_argument(
        "--snapshots-dir",
        type=str,
        default="outputs/snapshots",
        help="Directory to store captured snapshots.",
    )
    parser.add_argument(
        "--test-image",
        type=str,
        default=None,
        help="If specified, runs a headless single-frame test on the provided image and saves output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    app = WebcamEmotionApp(
        checkpoint_path=args.checkpoint,
        camera_id=args.camera,
        confidence_threshold=args.threshold,
        snapshots_dir=args.snapshots_dir,
    )

    if args.test_image:
        test_path = Path(args.test_image)
        if not test_path.exists():
            print(f"[Error] Test image '{test_path}' not found.", file=sys.stderr)
            sys.exit(1)
        img = cv2.imread(str(test_path))
        if img is None:
            print(f"[Error] Failed to read test image '{test_path}'.", file=sys.stderr)
            sys.exit(1)
        annotated, preds = app.process_frame(img, draw_overlay=True)
        out_path = Path(args.snapshots_dir) / f"test_output_{test_path.name}"
        cv2.imwrite(str(out_path), annotated)
        print(f"[FaceSense AI] Test frame processed: detected {len(preds)} face(s). Saved to {out_path}")
        for i, p in enumerate(preds, 1):
            print(f"  Face {i}: {p['predicted_emotion']} ({p['confidence']*100:.1f}%) | BBox: {p['bbox']}")
    else:
        app.run()


if __name__ == "__main__":
    main()
