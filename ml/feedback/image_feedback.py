"""
FaceSense AI - Interactive Image Feedback Handler
Manages interactive and programmatic feedback collection from user-provided images,
associating confirmations and corrections with specific detected faces, storing face crops,
and automatically tagging records with the active production model version.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from ml.feedback.collector import (
    DEFAULT_VALID_EMOTIONS,
    DEFAULT_METADATA_PATH,
    FeedbackCollector,
    FeedbackState,
    get_default_model_version,
)


class ImageFeedbackHandler:
    """
    Handles feedback operations for detected faces in static images:
      - [C] Mark selected face prediction as CORRECT
      - [W] Mark selected face prediction as INCORRECT (prompts for ground-truth label)
      - [U] Mark selected face prediction as UNCERTAIN
      - [1-9] Select face index in multi-face scenes
      - [Q] Quit feedback session
    """

    def __init__(
        self,
        feedback_dir: Union[str, Path] = "outputs/feedback",
        collector: Optional[FeedbackCollector] = None,
        model_version: Optional[str] = None,
        metadata_path: Union[str, Path] = DEFAULT_METADATA_PATH,
    ):
        self.feedback_dir = Path(feedback_dir)
        self.metadata_path = Path(metadata_path)
        # Automatically load current production model version from metadata if not overridden
        self.model_version = model_version or get_default_model_version(self.metadata_path)
        self.collector = collector or FeedbackCollector(
            feedback_dir=self.feedback_dir,
            default_model_version=self.model_version,
        )
        self.valid_emotions = DEFAULT_VALID_EMOTIONS

    def prompt_for_corrected_emotion(
        self,
        face_idx: int = 1,
        predicted_emotion: str = "unknown",
        input_func: Callable[[str], str] = input,
    ) -> Optional[str]:
        """
        Presents an interactive prompt asking user for the ground-truth emotion label.
        Validates against valid FER2013 emotions.
        """
        print("\n" + "=" * 65)
        print(f"[FaceSense AI Feedback] Correcting Prediction for Face {face_idx}")
        print(f"  Current Model Prediction : {predicted_emotion.upper()}")
        print(f"  Valid Options            : {', '.join(self.valid_emotions)}")
        print("  Enter corrected emotion (or 'c' / 'cancel' to abort):")
        print("=" * 65)

        try:
            user_input = input_func(">> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[FaceSense AI Feedback] Feedback entry aborted.")
            return None

        if not user_input or user_input in ["c", "cancel", "q", "quit", "abort", "exit"]:
            print("[FaceSense AI Feedback] Feedback entry cancelled.")
            return None

        if user_input not in self.valid_emotions:
            print(
                f"[FaceSense AI Feedback] Error: '{user_input}' is not a valid emotion.\n"
                f"Must be one of: {', '.join(self.valid_emotions)}"
            )
            return None

        return user_input

    def record_feedback(
        self,
        face_image: Union[np.ndarray, Image.Image, str, Path],
        state: Union[str, FeedbackState],
        predicted_emotion: str,
        confidence: float,
        corrected_emotion: Optional[str] = None,
        bounding_box: Optional[Union[List[int], Tuple[int, ...]]] = None,
        face_idx: int = 1,
        model_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Records a validated feedback entry for a face crop.

        Args:
            face_image: numpy array (BGR/Grayscale), PIL Image, or image path.
            state: 'correct', 'incorrect', or 'uncertain'.
            predicted_emotion: Label predicted by model.
            confidence: Confidence score float [0.0 - 1.0].
            corrected_emotion: User ground-truth label (required for 'incorrect').
            bounding_box: [x, y, w, h] face bounding box.
            face_idx: 1-based index of face in image.
            model_version: Optional model version override.
            metadata: Additional contextual dictionary.

        Returns:
            The recorded feedback dictionary.
        """
        state_str = state.value if isinstance(state, FeedbackState) else str(state).lower()
        version = model_version or self.model_version

        # Validate corrected emotion if state is incorrect
        if state_str == FeedbackState.INCORRECT.value:
            if not corrected_emotion or corrected_emotion.lower() not in self.valid_emotions:
                raise ValueError(
                    f"A valid corrected_emotion from {self.valid_emotions} is required for incorrect feedback."
                )
            corrected_emotion = corrected_emotion.lower()

        base_meta = {"source": "image_interactive", "face_index": face_idx}
        if metadata:
            base_meta.update(metadata)

        # Handle face_image saving
        if isinstance(face_image, (np.ndarray, Image.Image)):
            record = self.collector.save_face_crop_and_add_feedback(
                frame_or_face=face_image,
                predicted_emotion=predicted_emotion,
                confidence=confidence,
                corrected_emotion=corrected_emotion,
                bounding_box=bounding_box,
                state=state_str,
                model_version=version,
                metadata=base_meta,
            )
        else:
            img_path = Path(face_image)
            record = self.collector.add_feedback(
                image_path=img_path,
                predicted_emotion=predicted_emotion,
                confidence=confidence,
                corrected_emotion=corrected_emotion,
                bounding_box=bounding_box,
                state=state_str,
                model_version=version,
                metadata=base_meta,
                copy_image=True,
            )

        # Print user feedback confirmation
        if state_str == FeedbackState.CORRECT.value:
            print(
                f"[FaceSense AI Feedback] Saved CORRECT feedback for Face {face_idx} "
                f"({predicted_emotion.upper()}, {confidence * 100:.1f}%) [Model: {version}]"
            )
        elif state_str == FeedbackState.UNCERTAIN.value:
            print(
                f"[FaceSense AI Feedback] Saved UNCERTAIN feedback for Face {face_idx} "
                f"({predicted_emotion.upper()}, {confidence * 100:.1f}%) [Model: {version}]"
            )
        elif state_str == FeedbackState.INCORRECT.value:
            print(
                f"[FaceSense AI Feedback] Saved INCORRECT feedback for Face {face_idx} "
                f"(Corrected: {corrected_emotion.upper()}, Predicted: {predicted_emotion.upper()}) [Model: {version}]"
            )

        return record

    def display_face_predictions(
        self,
        predictions: List[Dict[str, Any]],
        selected_face_idx: int = 0,
    ) -> None:
        """
        Prints a formatted table of all detected faces, highlighting the selected face.
        """
        print("\n" + "=" * 70)
        print("FaceSense AI - Detected Faces & Emotion Predictions")
        print("=" * 70)
        if not predictions:
            print("  No faces detected in the image.")
            print("=" * 70)
            return

        print(f" {'Face #':<8} {'Selected':<10} {'Predicted':<12} {'Confidence':<12} {'Bounding Box [x, y, w, h]'}")
        print("-" * 70)
        for i, pred in enumerate(predictions):
            f_idx = pred.get("face_idx", i + 1)
            is_sel = (i == selected_face_idx)
            sel_marker = "[SELECTED]" if is_sel else ""
            emotion = pred.get("predicted_emotion", "unknown").upper()
            conf = pred.get("confidence", 0.0) * 100.0
            bbox = pred.get("bbox", [])
            bbox_str = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]" if len(bbox) == 4 else str(bbox)
            print(f"  Face {f_idx:<3} {sel_marker:<10} {emotion:<12} {conf:6.2f}%     {bbox_str}")
        print("=" * 70)

    def run_interactive_feedback(
        self,
        predictions: List[Dict[str, Any]],
        original_frame: Optional[np.ndarray] = None,
        image_source: Optional[Union[str, Path]] = None,
        input_func: Callable[[str], str] = input,
    ) -> List[Dict[str, Any]]:
        """
        Runs an interactive terminal loop for user feedback on detected faces.

        Controls:
          [C]   Mark selected face as CORRECT
          [W]   Mark selected face as INCORRECT (prompts for correction)
          [U]   Mark selected face as UNCERTAIN
          [1-9] Select face index
          [Q]   Finish / Quit feedback session
        """
        if not predictions:
            print("\n[FaceSense AI] Cannot collect feedback: No faces detected in image.")
            return []

        selected_idx = 0
        recorded_feedbacks: List[Dict[str, Any]] = []

        while True:
            self.display_face_predictions(predictions, selected_face_idx=selected_idx)
            target_pred = predictions[selected_idx]
            current_face_num = target_pred.get("face_idx", selected_idx + 1)

            print(f"\nActive Face: Face {current_face_num} "
                  f"(Emotion: {target_pred.get('predicted_emotion', '').upper()}, "
                  f"Confidence: {target_pred.get('confidence', 0.0)*100:.1f}%)")
            print("Feedback Actions:")
            print("  [C]   Mark prediction as CORRECT")
            print("  [W]   Mark prediction as WRONG (incorrect)")
            print("  [U]   Mark prediction as UNCERTAIN")
            if len(predictions) > 1:
                print(f"  [1-{len(predictions)}] Select another face")
            print("  [Q]   Done / Exit feedback")

            try:
                action = input_func("Enter choice >> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n[FaceSense AI] Feedback session ended.")
                break

            if not action or action in ["q", "quit", "exit", "done"]:
                print("[FaceSense AI] Feedback session ended.")
                break

            # Face selection by number 1-9
            if action.isdigit():
                num = int(action)
                if 1 <= num <= len(predictions):
                    selected_idx = num - 1
                    print(f"\n-> Switched active target to Face {num}")
                    continue
                else:
                    print(f"\n[Error] Invalid face number '{num}'. Choose between 1 and {len(predictions)}.")
                    continue

            face_crop = target_pred.get("face_crop")
            if face_crop is None:
                # If face_crop wasn't stored, fallback to original frame crop if bbox exists
                bbox = target_pred.get("bbox")
                if original_frame is not None and bbox is not None:
                    x, y, w, h = bbox
                    face_crop = original_frame[y:y+h, x:x+w]
                else:
                    face_crop = original_frame

            pred_emotion = target_pred.get("raw_predicted_emotion", target_pred.get("predicted_emotion", "unknown"))
            conf = float(target_pred.get("confidence", 1.0))
            bbox = target_pred.get("bbox")

            meta = {
                "source": "image_interactive",
                "image_source": str(image_source) if image_source else None,
            }

            if action == "c":
                rec = self.record_feedback(
                    face_image=face_crop,
                    state=FeedbackState.CORRECT,
                    predicted_emotion=pred_emotion,
                    confidence=conf,
                    bounding_box=bbox,
                    face_idx=current_face_num,
                    metadata=meta,
                )
                recorded_feedbacks.append(rec)

            elif action == "u":
                rec = self.record_feedback(
                    face_image=face_crop,
                    state=FeedbackState.UNCERTAIN,
                    predicted_emotion=pred_emotion,
                    confidence=conf,
                    bounding_box=bbox,
                    face_idx=current_face_num,
                    metadata=meta,
                )
                recorded_feedbacks.append(rec)

            elif action == "w":
                corr = self.prompt_for_corrected_emotion(
                    face_idx=current_face_num,
                    predicted_emotion=pred_emotion,
                    input_func=input_func,
                )
                if corr:
                    rec = self.record_feedback(
                        face_image=face_crop,
                        state=FeedbackState.INCORRECT,
                        predicted_emotion=pred_emotion,
                        confidence=conf,
                        corrected_emotion=corr,
                        bounding_box=bbox,
                        face_idx=current_face_num,
                        metadata=meta,
                    )
                    recorded_feedbacks.append(rec)

            else:
                print(f"\n[Error] Unrecognized choice '{action}'. Enter C, W, U, 1-{len(predictions)}, or Q.")

            # If only 1 face exists, auto-exit after feedback is recorded unless user wants to continue
            if len(predictions) == 1 and recorded_feedbacks:
                break

        return recorded_feedbacks
