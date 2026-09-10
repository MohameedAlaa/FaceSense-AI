"""
FaceSense AI - Face Detection Module
Provides lightweight FaceDetector class using OpenCV Haar Cascades for multi-face detection.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np


class FaceDetector:
    """
    Lightweight Face Detector using OpenCV Haar Cascade Classifier.
    Detects single and multiple faces in RGB/BGR frames or grayscale images.
    Returns standardized bounding box tuples: (x, y, w, h).
    """

    def __init__(
        self,
        cascade_path: Optional[Union[str, Path]] = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
    ):
        """
        Args:
            cascade_path: Path to Haar Cascade XML file. If None, uses default OpenCV frontal face cascade.
            scale_factor: Parameter specifying how much the image size is reduced at each image scale.
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it.
            min_size: Minimum possible object size. Objects smaller than that are ignored.
        """
        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.min_size = tuple(min_size)

        if cascade_path is None:
            # Check local repository cascade directory first
            local_path = Path(__file__).resolve().parent / "cascades" / "haarcascade_frontalface_default.xml"
            cv2_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml" if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades") else None
            
            if local_path.exists():
                self.cascade_path = local_path
            elif cv2_path and cv2_path.exists():
                self.cascade_path = cv2_path
            else:
                raise FileNotFoundError(
                    f"Haar Cascade XML not found at local '{local_path}' or cv2 '{cv2_path}'."
                )
        else:
            self.cascade_path = Path(cascade_path)
            if not self.cascade_path.exists():
                raise FileNotFoundError(f"Custom Haar Cascade file not found at: {self.cascade_path.resolve()}")

        self.face_cascade = cv2.CascadeClassifier(str(self.cascade_path))
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar Cascade from '{self.cascade_path}'.")

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects faces in a BGR, RGB, or grayscale image array.

        Args:
            frame: Numpy array representing image frame.

        Returns:
            List of bounding box tuples in format [(x, y, w, h), ...].
            Returns empty list if no faces are detected or frame is empty/invalid.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        # Convert to grayscale for detection if frame has color channels
        if frame.ndim == 3:
            if frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif frame.shape[2] == 4:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            else:
                gray = frame[:, :, 0]
        elif frame.ndim == 2:
            gray = frame
        else:
            return []

        # Equalize histogram or use directly for robust detection
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            return []

        # Convert detected ndarray to list of tuples: (x, y, w, h)
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def crop_face(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        margin_ratio: float = 0.0,
    ) -> np.ndarray:
        """
        Crops a detected face region from frame with optional margin padding.

        Args:
            frame: Input image array.
            bbox: (x, y, w, h) bounding box.
            margin_ratio: Fraction of width/height to pad around the face (default 0.0).

        Returns:
            Cropped image region as numpy array.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty.")

        h_img, w_img = frame.shape[:2]
        x, y, w, h = bbox

        if margin_ratio > 0.0:
            pad_w = int(w * margin_ratio)
            pad_h = int(h * margin_ratio)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w_img, x + w + pad_w)
            y2 = min(h_img, y + h + pad_h)
        else:
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_img, x + w)
            y2 = min(h_img, y + h)

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid crop coordinates: [{y1}:{y2}, {x1}:{x2}] from bbox {bbox}.")

        return frame[y1:y2, x1:x2]
