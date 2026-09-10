"""
FaceSense AI - Real-Time Webcam Facial Expression Recognition & Interactive Feedback Application
Pipeline:
  Camera frame -> Face Detection -> Face Crop -> EmotionPredictor -> Interactive Feedback -> Overlays

Keyboard Controls:
  - [C]: Mark selected face prediction as CORRECT
  - [W]: Mark selected face prediction as INCORRECT (terminal prompt for corrected emotion)
  - [U]: Mark selected face prediction as UNCERTAIN
  - [1-9]: Select specific face index in multi-face scenes
  - [S]: Save snapshot frame with annotations
  - [Q] / [ESC]: Quit application
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.detection.detector import FaceDetector
from ml.inference.predictor import EmotionPredictor
from ml.feedback.webcam_feedback import WebcamFeedbackHandler


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
    Real-Time Webcam Face & Emotion Recognition Application with Interactive Feedback.
    Orchestrates camera capture, multi-face detection, CNN prediction, interactive feedback
    recording (C/W/U keys), face selection (1-9), and annotated rendering.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "ml/models/checkpoints/final/best_model.pt",
        camera_id: int = 0,
        confidence_threshold: float = 0.35,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        snapshots_dir: Union[str, Path] = "outputs/snapshots",
        feedback_dir: Union[str, Path] = "outputs/feedback",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        self.snapshots_dir = Path(snapshots_dir)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_dir = Path(feedback_dir)

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

        self.feedback_handler = WebcamFeedbackHandler(
            feedback_dir=self.feedback_dir,
            model_version=self.predictor.model_version,
        )

        self.selected_face_idx = 0
        self.last_frame: Optional[np.ndarray] = None
        self.last_predictions: List[dict] = []
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

        self.last_frame = frame.copy()
        display_frame = frame.copy() if draw_overlay else frame
        face_bboxes = self.detector.detect_faces(frame)
        face_predictions = []

        # Maintain selected face index in bounds
        if face_bboxes:
            self.selected_face_idx = min(self.selected_face_idx, len(face_bboxes) - 1)
        else:
            self.selected_face_idx = 0

        for i, bbox in enumerate(face_bboxes):
            x, y, w, h = bbox
            try:
                face_crop = self.detector.crop_face(frame, bbox, margin_ratio=0.05)
                pred_result = self.predictor.predict(
                    face_crop, confidence_threshold=self.confidence_threshold
                )
                pred_result["bbox"] = bbox
                pred_result["face_idx"] = i + 1
                face_predictions.append(pred_result)

                if draw_overlay:
                    is_selected = (i == self.selected_face_idx)
                    self._draw_face_annotation(
                        display_frame, bbox, pred_result, face_idx=i + 1, is_selected=is_selected
                    )
            except Exception as e:
                if draw_overlay:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(
                        display_frame,
                        f"Error: {e}",
                        (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )

        self.last_predictions = face_predictions

        if draw_overlay:
            self._draw_status_bar(display_frame, num_faces=len(face_bboxes))

        return display_frame, face_predictions

    def _draw_face_annotation(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        pred: dict,
        face_idx: int = 1,
        is_selected: bool = False,
    ) -> None:
        """Draws bounding box, label tag, selection indicator, and confidence on a detected face."""
        x, y, w, h = bbox
        emotion = pred["predicted_emotion"].lower()
        conf = pred["confidence"] * 100.0

        color = EMOTION_COLORS.get(emotion, (0, 255, 0))

        # 1. Main bounding box
        box_thickness = 3 if is_selected else 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, box_thickness)

        # 2. Selected face border accent
        if is_selected:
            # Draw subtle gold/white corners or outer marker
            corner_len = min(15, w // 4, h // 4)
            accent_col = (0, 215, 255)  # Gold in BGR
            cv2.line(frame, (x - 4, y - 4), (x - 4 + corner_len, y - 4), accent_col, 2)
            cv2.line(frame, (x - 4, y - 4), (x - 4, y - 4 + corner_len), accent_col, 2)
            cv2.line(frame, (x + w + 4, y - 4), (x + w + 4 - corner_len, y - 4), accent_col, 2)
            cv2.line(frame, (x + w + 4, y - 4), (x + w + 4, y - 4 + corner_len), accent_col, 2)
            cv2.line(frame, (x - 4, y + h + 4), (x - 4 + corner_len, y + h + 4), accent_col, 2)
            cv2.line(frame, (x - 4, y + h + 4), (x - 4, y + h + 4 - corner_len), accent_col, 2)
            cv2.line(frame, (x + w + 4, y + h + 4), (x + w + 4 - corner_len, y + h + 4), accent_col, 2)
            cv2.line(frame, (x + w + 4, y + h + 4), (x + w + 4, y + h + 4 - corner_len), accent_col, 2)

        # 3. Text label
        sel_prefix = f"[{face_idx}*] " if is_selected else f"[{face_idx}] "
        if pred["is_uncertain"]:
            label_text = f"{sel_prefix}Uncertain ({conf:.1f}%)"
        else:
            label_text = f"{sel_prefix}{emotion.capitalize()} ({conf:.1f}%)"

        # Label background pill
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        pill_y1 = max(0, y - text_h - 10)
        pill_y2 = y
        pill_x2 = min(frame.shape[1], x + text_w + 10)

        cv2.rectangle(frame, (x, pill_y1), (pill_x2, pill_y2), color, cv2.FILLED)
        text_color = (0, 0, 0) if emotion in ["happy", "surprise", "neutral"] else (255, 255, 255)
        cv2.putText(
            frame,
            label_text,
            (x + 5, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
            2,
            cv2.LINE_AA,
        )

    def _draw_status_bar(self, frame: np.ndarray, num_faces: int) -> None:
        """Draws FPS, control hints, and temporary feedback status toasts on header."""
        # Calculate current FPS
        curr_time = time.time()
        dt = curr_time - self._prev_time
        self._prev_time = curr_time
        if dt > 0:
            current_fps = 1.0 / dt
            self.fps = 0.9 * self.fps + 0.1 * current_fps if self.fps > 0 else current_fps

        h, w = frame.shape[:2]
        header_h = 50
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_h), (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Status text line 1
        selected_display = (self.selected_face_idx + 1) if num_faces > 0 else 0
        line1 = (
            f"FaceSense AI | Model: {self.predictor.model_version} | "
            f"Faces: {num_faces} (Selected: {selected_display}) | FPS: {self.fps:.1f}"
        )
        cv2.putText(
            frame,
            line1,
            (10, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Status text line 2 (Controls)
        line2 = "[C] Correct  [W] Wrong  [U] Uncertain  [1-9] Select Face  [S] Save  [Q] Quit"
        cv2.putText(
            frame,
            line2,
            (10, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 215, 255),
            1,
            cv2.LINE_AA,
        )

        # ── Feedback Status Toast Overlay ──────────────────────────
        hud_info = self.feedback_handler.get_hud_status()
        if hud_info:
            msg, is_error = hud_info
            toast_h = 28
            toast_y = h - toast_h - 10
            toast_bg = (0, 0, 180) if is_error else (30, 140, 30)  # Red for error, Green for success
            toast_overlay = frame.copy()
            cv2.rectangle(toast_overlay, (0, toast_y), (w, toast_y + toast_h), toast_bg, cv2.FILLED)
            cv2.addWeighted(toast_overlay, 0.85, frame, 0.15, 0, frame)
            cv2.putText(
                frame,
                f" {msg}",
                (15, toast_y + 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def handle_feedback(
        self,
        action: str,  # "correct", "incorrect", "uncertain"
        frame: Optional[np.ndarray] = None,
        predictions: Optional[List[dict]] = None,
        face_idx: Optional[int] = None,
        corrected_emotion: Optional[str] = None,
        input_func: Callable[[str], str] = input,
    ) -> Optional[Dict[str, Any]]:
        """
        Processes interactive or programmatic feedback for the selected face.

        Args:
            action: 'correct', 'incorrect', or 'uncertain'.
            frame: Input frame ndarray (defaults to last processed frame).
            predictions: Face predictions list (defaults to last predictions).
            face_idx: 0-based face index to target (defaults to self.selected_face_idx).
            corrected_emotion: Ground-truth label for 'incorrect' (if None, prompts via terminal).
            input_func: Custom input function for terminal prompt (defaults to input).

        Returns:
            The recorded feedback record dict, or None if skipped/cancelled.
        """
        target_frame = frame if frame is not None else self.last_frame
        target_preds = predictions if predictions is not None else self.last_predictions

        if target_frame is None or not target_preds:
            msg = "No detected face available to submit feedback"
            self.feedback_handler.set_hud_status(msg, is_error=True)
            print(f"\n[FaceSense AI Feedback] {msg}.")
            return None

        # Resolve face index
        sel_idx = face_idx if face_idx is not None else self.selected_face_idx
        sel_idx = max(0, min(sel_idx, len(target_preds) - 1))
        target_pred = target_preds[sel_idx]
        bbox = target_pred.get("bbox")

        # Extract face crop
        try:
            if bbox is not None:
                face_crop = self.detector.crop_face(target_frame, bbox, margin_ratio=0.05)
            else:
                face_crop = target_frame
        except Exception as e:
            print(f"[FaceSense AI Feedback] Face cropping failed: {e}")
            face_crop = target_frame

        pred_emotion = target_pred.get("raw_predicted_emotion", target_pred.get("predicted_emotion", "unknown"))
        conf = float(target_pred.get("confidence", 1.0))
        act = action.strip().lower()

        if act == "correct":
            return self.feedback_handler.record_feedback(
                face_image=face_crop,
                state="correct",
                predicted_emotion=pred_emotion,
                confidence=conf,
                bounding_box=bbox,
                face_idx=sel_idx + 1,
            )

        elif act == "uncertain":
            return self.feedback_handler.record_feedback(
                face_image=face_crop,
                state="uncertain",
                predicted_emotion=pred_emotion,
                confidence=conf,
                bounding_box=bbox,
                face_idx=sel_idx + 1,
            )

        elif act == "incorrect":
            if corrected_emotion is None:
                corrected_emotion = self.feedback_handler.prompt_for_corrected_emotion(
                    face_idx=sel_idx + 1,
                    predicted_emotion=pred_emotion,
                    input_func=input_func,
                )

            if not corrected_emotion:
                self.feedback_handler.set_hud_status("Feedback Cancelled", is_error=True)
                return None

            try:
                return self.feedback_handler.record_feedback(
                    face_image=face_crop,
                    state="incorrect",
                    predicted_emotion=pred_emotion,
                    confidence=conf,
                    corrected_emotion=corrected_emotion,
                    bounding_box=bbox,
                    face_idx=sel_idx + 1,
                )
            except Exception as e:
                self.feedback_handler.set_hud_status(f"Error: {e}", is_error=True)
                print(f"[FaceSense AI Feedback] Failed to record feedback: {e}")
                return None

        else:
            raise ValueError(f"Unknown feedback action '{action}'. Must be 'correct', 'incorrect', or 'uncertain'.")

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

        print("\n" + "=" * 68)
        print("  FaceSense AI Live Webcam Running")
        print(f"  Model Version : {self.predictor.model_version}")
        print("  Controls:")
        print("    [C]   Mark selected face as CORRECT prediction")
        print("    [W]   Mark selected face as WRONG (incorrect) prediction")
        print("    [U]   Mark selected face as UNCERTAIN prediction")
        print("    [1-9] Select face index in multi-face scenes")
        print("    [S]   Save annotated snapshot")
        print("    [Q]   Quit application")
        print("=" * 68 + "\n")

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
                    self.feedback_handler.set_hud_status(f"Snapshot Saved: {snap_path.name}")
                    print(f"[FaceSense AI] Saved snapshot to: {snap_path}")
                elif key == ord("c") or key == ord("C"):
                    self.handle_feedback("correct")
                elif key == ord("u") or key == ord("U"):
                    self.handle_feedback("uncertain")
                elif key == ord("w") or key == ord("W"):
                    self.handle_feedback("incorrect")
                elif ord("1") <= key <= ord("9"):
                    num = key - ord("1")
                    if self.last_predictions:
                        self.selected_face_idx = min(num, len(self.last_predictions) - 1)
                        self.feedback_handler.set_hud_status(f"Selected Face {self.selected_face_idx + 1}")
                        print(f"[FaceSense AI] Selected Face {self.selected_face_idx + 1}")

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
        "--feedback-dir",
        type=str,
        default="outputs/feedback",
        help="Directory to store logged user feedback.",
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
        feedback_dir=args.feedback_dir,
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
