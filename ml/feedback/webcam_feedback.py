"""
FaceSense AI - Interactive Webcam Feedback Handler
Manages interactive feedback collection during live webcam sessions, associating
user confirmations and corrections with specific detected faces, storing face crops,
and automatically tagging records with the active production model version.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
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


class WebcamFeedbackHandler:
    """
    Handles interactive feedback operations triggered via keyboard shortcuts in the webcam app:
      - [C] Mark selected face prediction as CORRECT
      - [W] Mark selected face prediction as INCORRECT (prompts for ground-truth label)
      - [U] Mark selected face prediction as UNCERTAIN
      - Number keys [1-9]: Select specific face in multi-face scenes
    """

    def __init__(
        self,
        feedback_dir: Union[str, Path] = "outputs/feedback",
        collector: Optional[FeedbackCollector] = None,
        model_version: Optional[str] = None,
        metadata_path: Union[str, Path] = DEFAULT_METADATA_PATH,
        status_duration: float = 3.5,
    ):
        self.feedback_dir = Path(feedback_dir)
        self.metadata_path = Path(metadata_path)
        self.model_version = model_version or get_default_model_version(self.metadata_path)
        self.collector = collector or FeedbackCollector(
            feedback_dir=self.feedback_dir,
            default_model_version=self.model_version,
        )
        self.valid_emotions = DEFAULT_VALID_EMOTIONS
        self.status_duration = status_duration
        self.last_status_message: Optional[str] = None
        self.last_status_time: float = 0.0
        self.last_status_is_error: bool = False

    def get_hud_status(self) -> Optional[Tuple[str, bool]]:
        """
        Returns (message, is_error) if a recent feedback status message is active, otherwise None.
        """
        if self.last_status_message is None:
            return None
        elapsed = time.time() - self.last_status_time
        if elapsed <= self.status_duration:
            return self.last_status_message, self.last_status_is_error
        return None

    def set_hud_status(self, message: str, is_error: bool = False) -> None:
        """Sets a temporary status message to be rendered on the webcam HUD."""
        self.last_status_message = message
        self.last_status_time = time.time()
        self.last_status_is_error = is_error

    def prompt_for_corrected_emotion(
        self,
        face_idx: int = 1,
        predicted_emotion: str = "unknown",
        input_func: Callable[[str], str] = input,
    ) -> Optional[str]:
        """
        Presents a simple interactive terminal prompt asking the user for the true emotion label.

        Args:
            face_idx: 1-based index of the target face.
            predicted_emotion: The model's current predicted label.
            input_func: Input function (defaults to built-in input, mockable for testing).

        Returns:
            Validated emotion string (e.g. 'happy') or None if cancelled.
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
    ) -> Dict[str, Any]:
        """
        Records a validated feedback entry for a face crop.

        Args:
            face_image: numpy array (BGR/Grayscale), PIL Image, or existing image path.
            state: 'correct', 'incorrect', or 'uncertain'.
            predicted_emotion: Label predicted by model.
            confidence: Confidence score [0.0 - 1.0].
            corrected_emotion: User ground-truth label (required for 'incorrect').
            bounding_box: [x, y, w, h] face bounding box.
            face_idx: 1-based index of the selected face in current frame.
            model_version: Optional model version override.

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
                metadata={"source": "webcam_interactive", "face_index": face_idx},
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
                metadata={"source": "webcam_interactive", "face_index": face_idx},
                copy_image=True,
            )

        # Set HUD status toast
        if state_str == FeedbackState.CORRECT.value:
            msg = f"Feedback Saved: CORRECT (Face {face_idx}: {predicted_emotion})"
            self.set_hud_status(msg, is_error=False)
            print(f"\n[FaceSense AI Feedback] Saved CORRECT feedback for Face {face_idx} ({predicted_emotion}, {confidence*100:.1f}%) [Model: {version}]")
        elif state_str == FeedbackState.UNCERTAIN.value:
            msg = f"Feedback Saved: UNCERTAIN (Face {face_idx})"
            self.set_hud_status(msg, is_error=False)
            print(f"\n[FaceSense AI Feedback] Saved UNCERTAIN feedback for Face {face_idx} ({predicted_emotion}, {confidence*100:.1f}%) [Model: {version}]")
        elif state_str == FeedbackState.INCORRECT.value:
            msg = f"Feedback Saved: WRONG -> {corrected_emotion.upper()} (Face {face_idx})"
            self.set_hud_status(msg, is_error=False)
            print(f"\n[FaceSense AI Feedback] Saved INCORRECT feedback for Face {face_idx} (True: {corrected_emotion}, Pred: {predicted_emotion}) [Model: {version}]")

        return record
