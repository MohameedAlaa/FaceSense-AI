"""
FaceSense AI - Face Detection Module
Provides lightweight FaceDetector class using OpenCV Haar Cascades for multi-face detection.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np


DEFAULT_MAX_DETECTOR_DIM: int = 640


class FaceDetector:
    """
    Lightweight Face Detector using OpenCV Haar Cascade Classifier.
    Detects single and multiple faces in RGB/BGR frames or grayscale images.
    Returns standardized bounding box tuples: (x, y, w, h).

    Conservative defaults are chosen to minimise false positives from background
    clutter (shelves, windows, objects) while preserving detection of real human
    faces at typical webcam/upload distances:

      - min_size=(80, 80):    Any face close enough to classify emotions is
                               substantially larger than 30×30 px.  The previous
                               default of 30×30 allowed tiny background patches to
                               trigger the cascade (root cause of the upper-right
                               false positive observed in webcam captures).

      - min_neighbors=6:      Each candidate rectangle must have at least 6
                               neighbouring detections rather than 5, requiring
                               stronger consensus before a positive is kept.

      - use_clahe=True:       Applies CLAHE (Contrast Limited Adaptive Histogram
                               Equalisation) to the grayscale frame before running
                               the cascade.  This normalises uneven lighting from
                               bright windows or overhead lights — a common source
                               of background Haar-feature matches — while keeping
                               the true face signal strong.

      - max_detector_dim=640: Limits the maximum image dimension fed into the Haar
                               cascade detector. Images larger than this dimension
                               are proportionally downscaled before detection, and
                               bounding boxes are remapped to original coordinates.
    """

    def __init__(
        self,
        cascade_path: Optional[Union[str, Path]] = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 6,
        min_size: Tuple[int, int] = (80, 80),
        max_size: Tuple[int, int] = (),
        use_clahe: bool = True,
        max_detector_dim: Optional[int] = DEFAULT_MAX_DETECTOR_DIM,
    ):
        """
        Args:
            cascade_path:      Path to Haar Cascade XML file.  If None, the local
                               repository cascade is used, falling back to the OpenCV
                               bundled cascade.
            scale_factor:      Image pyramid reduction factor per scale step.
            min_neighbors:     Minimum number of overlapping detection windows a
                               candidate must accumulate before being kept as a face.
                               Higher values reduce false positives at the cost of
                               slightly lower recall on very small or partial faces.
            min_size:          Minimum bounding-box dimensions (pixels).  Objects
                               smaller than this are ignored.  Default (80, 80) removes
                               tiny background false positives that cannot possibly
                               be a detectable human face at usable resolution.
            max_size:          Maximum bounding-box dimensions (pixels).  Empty tuple
                               means no upper limit (default).
            use_clahe:         If True (default), applies CLAHE to the grayscale image
                               before cascade detection.  Normalises local contrast to
                               reduce spurious matches from high-contrast backgrounds.
            max_detector_dim:  Maximum dimension (width or height) fed to the detector.
                               If an input frame exceeds this dimension, it is
                               downscaled proportionally while preserving aspect ratio.
                               Detected boxes are remapped to original image coordinates.
                               Default is 640. Set to None to disable downscaling.
        """
        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.min_size = tuple(min_size)
        self.max_size = tuple(max_size)
        self.use_clahe = bool(use_clahe)
        self.max_detector_dim = (
            int(max_detector_dim)
            if max_detector_dim is not None and int(max_detector_dim) > 0
            else None
        )

        if cascade_path is None:
            # Check local repository cascade directory first
            local_path = Path(__file__).resolve().parent / "cascades" / "haarcascade_frontalface_default.xml"
            cv2_path = (
                Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
                if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades")
                else None
            )

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

        # Pre-allocate CLAHE processor (reuse across frames for efficiency)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert frame to grayscale and optionally apply CLAHE.

        Grayscale conversion follows OpenCV channel conventions:
          3-channel → BGR→GRAY, 4-channel → BGRA→GRAY, 2-D → already gray.

        CLAHE (Contrast Limited Adaptive Histogram Equalisation) subdivides the
        image into small tiles, equalises each independently, and blends them via
        bilinear interpolation.  This suppresses the effect of bright background
        regions (windows, lights) that would otherwise generate high-gradient edge
        patterns matching early Haar cascade stages for a face.
        """
        # Convert to grayscale
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
            return None  # type: ignore[return-value]

        # Optionally apply CLAHE for contrast normalisation
        if self.use_clahe:
            gray = self._clahe.apply(gray)

        return gray

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects faces in a BGR, RGB, or grayscale image array.

        If the maximum dimension of frame exceeds max_detector_dim, the image
        is proportionally downscaled before running Haar cascade detection.
        All detected bounding boxes are mapped back to original image coordinates.

        Args:
            frame: Numpy array representing image frame.

        Returns:
            List of bounding box tuples in format [(x, y, w, h), ...] in the
            original coordinate space of the input frame.
            Returns empty list if no faces are detected or frame is empty/invalid.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        if frame.ndim < 2:
            return []

        h_orig, w_orig = frame.shape[:2]
        if h_orig <= 0 or w_orig <= 0:
            return []

        max_dim = max(h_orig, w_orig)
        is_downscaled = False
        det_frame = frame

        if self.max_detector_dim is not None and max_dim > self.max_detector_dim:
            scale_ratio = self.max_detector_dim / float(max_dim)
            new_w = max(1, int(round(w_orig * scale_ratio)))
            new_h = max(1, int(round(h_orig * scale_ratio)))
            det_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            is_downscaled = True

        gray = self._preprocess(det_frame)
        if gray is None:
            return []

        # Build kwargs for detectMultiScale; only pass maxSize when set
        kwargs = dict(
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if self.max_size:
            kwargs["maxSize"] = self.max_size

        faces = self.face_cascade.detectMultiScale(gray, **kwargs)

        if len(faces) == 0:
            return []

        if not is_downscaled:
            return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

        inv_scale_x = float(w_orig) / float(new_w)
        inv_scale_y = float(h_orig) / float(new_h)

        rescaled_faces: List[Tuple[int, int, int, int]] = []
        for (x, y, w, h) in faces:
            orig_x = int(round(x * inv_scale_x))
            orig_y = int(round(y * inv_scale_y))
            orig_w = int(round(w * inv_scale_x))
            orig_h = int(round(h * inv_scale_y))

            # Clamp bounding box to ensure it remains inside original frame boundaries
            orig_x = max(0, min(orig_x, w_orig - 1))
            orig_y = max(0, min(orig_y, h_orig - 1))
            orig_w = max(1, min(orig_w, w_orig - orig_x))
            orig_h = max(1, min(orig_h, h_orig - orig_y))

            rescaled_faces.append((orig_x, orig_y, orig_w, orig_h))

        return rescaled_faces

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
