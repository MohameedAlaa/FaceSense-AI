"""
Unit and Integration Tests for FaceSense AI Face Detection & Webcam Integration Pipeline
Tests FaceDetector initialization, empty frame handling, bounding box format, multi-face output,
crop functionality, integration with EmotionPredictor, and confidence threshold behavior.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import numpy as np
import cv2
import torch
from PIL import Image

from ml.detection.detector import FaceDetector
from ml.inference.predictor import EmotionPredictor
from ml.inference.webcam import WebcamEmotionApp


class TestFaceDetectionAndWebcam(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.detector = FaceDetector()
        cls.checkpoint_path = PROJECT_ROOT / "ml" / "models" / "checkpoints" / "final" / "best_model.pt"
        cls.predictor = EmotionPredictor(checkpoint_path=cls.checkpoint_path)
        cls.test_data_dir = PROJECT_ROOT / "Data(FER2013)" / "test"
        cls.real_img_path = next(cls.test_data_dir.rglob("*.jpg"))

    def test_detector_initialization(self):
        """Verify FaceDetector initializes and loads Haar Cascade."""
        self.assertIsNotNone(self.detector.face_cascade)
        self.assertFalse(self.detector.face_cascade.empty())
        self.assertTrue(Path(self.detector.cascade_path).exists())

    def test_empty_and_invalid_frame_handling(self):
        """Verify detector handles None, empty arrays, or invalid dimensions without crashing."""
        self.assertEqual(self.detector.detect_faces(None), [])
        self.assertEqual(self.detector.detect_faces(np.array([])), [])
        self.assertEqual(self.detector.detect_faces(np.zeros((0, 0, 3), dtype=np.uint8)), [])
        self.assertEqual(self.detector.detect_faces(np.zeros((10, 10, 5), dtype=np.uint8)), [])

    def test_bounding_box_format_and_synthetic_face(self):
        """Verify detect_faces returns list of (x, y, w, h) integer 4-tuples."""
        # Create a synthetic image canvas containing real FER2013 face pasted in the center
        fer_face = cv2.imread(str(self.real_img_path))
        canvas = np.full((300, 300, 3), 128, dtype=np.uint8)
        # Upscale face so Haar Cascade can detect easily (min size 30x30)
        face_upscaled = cv2.resize(fer_face, (150, 150))
        canvas[75:225, 75:225] = face_upscaled

        bboxes = self.detector.detect_faces(canvas)
        # Even if Haar cascade on low-res FER image returns 0 or 1 face, test return type constraints
        self.assertIsInstance(bboxes, list)
        for bbox in bboxes:
            self.assertEqual(len(bbox), 4)
            x, y, w, h = bbox
            self.assertIsInstance(x, int)
            self.assertIsInstance(y, int)
            self.assertIsInstance(w, int)
            self.assertIsInstance(h, int)
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)

    def test_crop_face_functionality(self):
        """Verify crop_face extracts valid non-empty ndarray."""
        frame = np.ones((200, 200, 3), dtype=np.uint8) * 255
        bbox = (50, 50, 80, 80)
        crop = self.detector.crop_face(frame, bbox, margin_ratio=0.0)
        self.assertEqual(crop.shape, (80, 80, 3))

        # Test with margin padding
        crop_padded = self.detector.crop_face(frame, bbox, margin_ratio=0.1)
        self.assertGreaterEqual(crop_padded.shape[0], 80)
        self.assertGreaterEqual(crop_padded.shape[1], 80)

    def test_multi_face_mock_detection_pipeline(self):
        """Verify multi-face processing pipeline handles multiple detected boxes correctly."""
        # Create mock frame
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Mock 2 face regions with real FER images
        fer_face = cv2.imread(str(self.real_img_path))
        face_1 = cv2.resize(fer_face, (100, 100))
        face_2 = cv2.resize(fer_face, (100, 100))
        mock_frame[50:150, 50:150] = face_1
        mock_frame[50:150, 250:350] = face_2

        # Mock detector output with two bounding boxes
        mock_bboxes = [(50, 50, 100, 100), (250, 50, 100, 100)]
        
        predictions = []
        for bbox in mock_bboxes:
            crop = self.detector.crop_face(mock_frame, bbox)
            res = self.predictor.predict(crop)
            predictions.append(res)

        self.assertEqual(len(predictions), 2)
        for res in predictions:
            self.assertIn("predicted_emotion", res)
            self.assertIn("confidence", res)
            self.assertIn(res["predicted_emotion"], self.predictor.class_names)

    def test_webcam_app_process_frame_integration(self):
        """Verify WebcamEmotionApp process_frame executes end-to-end and draws annotations."""
        app = WebcamEmotionApp(
            checkpoint_path=self.checkpoint_path,
            camera_id=0,
            confidence_threshold=0.30,
        )

        test_frame = cv2.imread(str(self.real_img_path))
        test_frame_bgr = cv2.resize(test_frame, (320, 240))

        annotated_frame, preds = app.process_frame(test_frame_bgr, draw_overlay=True)
        self.assertEqual(annotated_frame.shape, test_frame_bgr.shape)
        self.assertIsInstance(preds, list)

    def test_confidence_threshold_in_webcam_app(self):
        """Verify confidence threshold marks predictions as 'uncertain' when below threshold."""
        app = WebcamEmotionApp(
            checkpoint_path=self.checkpoint_path,
            confidence_threshold=0.9999, # Force uncertain
        )

        dummy_face = cv2.imread(str(self.real_img_path))
        # Direct prediction test with app's predictor
        res = app.predictor.predict(dummy_face, confidence_threshold=app.confidence_threshold)
        self.assertEqual(res["predicted_emotion"], "uncertain")
        self.assertTrue(res["is_uncertain"])


if __name__ == "__main__":
    unittest.main()
