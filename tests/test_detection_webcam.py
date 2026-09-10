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

    def test_webcam_feedback_correct_recording(self):
        """Verify pressing 'C' in webcam records correct feedback with proper metadata."""
        test_fb_dir = PROJECT_ROOT / "tests" / "test_feedback_webcam_correct"
        test_fb_dir.mkdir(parents=True, exist_ok=True)
        try:
            app = WebcamEmotionApp(
                checkpoint_path=self.checkpoint_path,
                feedback_dir=test_fb_dir,
            )

            # Mock a frame with predictions
            test_frame = np.zeros((200, 200, 3), dtype=np.uint8)
            preds = [{
                "predicted_emotion": "happy",
                "raw_predicted_emotion": "happy",
                "confidence": 0.95,
                "bbox": (20, 20, 100, 100),
            }]

            record = app.handle_feedback("correct", frame=test_frame, predictions=preds, face_idx=0)
            self.assertIsNotNone(record)
            self.assertEqual(record["state"], "correct")
            self.assertEqual(record["predicted_emotion"], "happy")
            self.assertAlmostEqual(record["confidence"], 0.95, places=2)
            self.assertIsNone(record["corrected_emotion"])
            self.assertTrue(Path(record["image_path"]).exists())
            self.assertEqual(record["model_version"], app.predictor.model_version)
        finally:
            import shutil
            if test_fb_dir.exists():
                shutil.rmtree(test_fb_dir, ignore_errors=True)

    def test_webcam_feedback_incorrect_with_corrected_label(self):
        """Verify recording incorrect feedback with user corrected emotion label."""
        test_fb_dir = PROJECT_ROOT / "tests" / "test_feedback_webcam_incorrect"
        test_fb_dir.mkdir(parents=True, exist_ok=True)
        try:
            app = WebcamEmotionApp(
                checkpoint_path=self.checkpoint_path,
                feedback_dir=test_fb_dir,
            )

            test_frame = np.zeros((200, 200, 3), dtype=np.uint8)
            preds = [{
                "predicted_emotion": "sad",
                "raw_predicted_emotion": "sad",
                "confidence": 0.60,
                "bbox": (10, 10, 80, 80),
            }]

            record = app.handle_feedback(
                "incorrect",
                frame=test_frame,
                predictions=preds,
                face_idx=0,
                corrected_emotion="neutral",
            )
            self.assertIsNotNone(record)
            self.assertEqual(record["state"], "incorrect")
            self.assertEqual(record["predicted_emotion"], "sad")
            self.assertEqual(record["corrected_emotion"], "neutral")
            self.assertTrue(Path(record["image_path"]).exists())
        finally:
            import shutil
            if test_fb_dir.exists():
                shutil.rmtree(test_fb_dir, ignore_errors=True)

    def test_webcam_feedback_uncertain_recording(self):
        """Verify recording uncertain feedback from webcam application."""
        test_fb_dir = PROJECT_ROOT / "tests" / "test_feedback_webcam_uncertain"
        test_fb_dir.mkdir(parents=True, exist_ok=True)
        try:
            app = WebcamEmotionApp(
                checkpoint_path=self.checkpoint_path,
                feedback_dir=test_fb_dir,
            )

            test_frame = np.zeros((200, 200, 3), dtype=np.uint8)
            preds = [{
                "predicted_emotion": "uncertain",
                "raw_predicted_emotion": "fear",
                "confidence": 0.28,
                "bbox": (15, 15, 60, 60),
            }]

            record = app.handle_feedback("uncertain", frame=test_frame, predictions=preds, face_idx=0)
            self.assertIsNotNone(record)
            self.assertEqual(record["state"], "uncertain")
            self.assertTrue(Path(record["image_path"]).exists())
            self.assertEqual(record["model_version"], app.predictor.model_version)
        finally:
            import shutil
            if test_fb_dir.exists():
                shutil.rmtree(test_fb_dir, ignore_errors=True)

    def test_webcam_feedback_model_version_tracking(self):
        """Verify webcam feedback automatically reads and records the active production model version."""
        test_fb_dir = PROJECT_ROOT / "tests" / "test_feedback_webcam_mv"
        test_fb_dir.mkdir(parents=True, exist_ok=True)
        try:
            app = WebcamEmotionApp(
                checkpoint_path=self.checkpoint_path,
                feedback_dir=test_fb_dir,
            )

            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            preds = [{"predicted_emotion": "happy", "confidence": 0.9, "bbox": (0, 0, 50, 50)}]

            record = app.handle_feedback("correct", frame=test_frame, predictions=preds, face_idx=0)
            self.assertIsNotNone(record)
            # Must match promoted model version
            self.assertEqual(record["model_version"], "ResidualEmotionCNN-Candidate-epoch39")
        finally:
            import shutil
            if test_fb_dir.exists():
                shutil.rmtree(test_fb_dir, ignore_errors=True)

    def test_webcam_feedback_invalid_corrected_label(self):
        """Verify providing an invalid corrected label is rejected cleanly."""
        test_fb_dir = PROJECT_ROOT / "tests" / "test_feedback_webcam_invalid"
        test_fb_dir.mkdir(parents=True, exist_ok=True)
        try:
            app = WebcamEmotionApp(
                checkpoint_path=self.checkpoint_path,
                feedback_dir=test_fb_dir,
            )

            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            preds = [{"predicted_emotion": "sad", "confidence": 0.5, "bbox": (0, 0, 50, 50)}]

            # Mock input function providing invalid emotion then cancelling
            inputs = iter(["invalid_label", "cancel"])
            res = app.handle_feedback("incorrect", frame=test_frame, predictions=preds, face_idx=0, input_func=lambda _: next(inputs))
            self.assertIsNone(res)
        finally:
            import shutil
            if test_fb_dir.exists():
                shutil.rmtree(test_fb_dir, ignore_errors=True)

    def test_webcam_feedback_multi_face_selection(self):
        """Verify feedback target correctly points to the selected face in multi-face frames."""
        test_fb_dir = PROJECT_ROOT / "tests" / "test_feedback_webcam_multiface"
        test_fb_dir.mkdir(parents=True, exist_ok=True)
        try:
            app = WebcamEmotionApp(
                checkpoint_path=self.checkpoint_path,
                feedback_dir=test_fb_dir,
            )

            test_frame = np.zeros((300, 300, 3), dtype=np.uint8)
            preds = [
                {"predicted_emotion": "happy", "confidence": 0.90, "bbox": (10, 10, 50, 50)},
                {"predicted_emotion": "angry", "confidence": 0.85, "bbox": (100, 100, 60, 60)},
            ]

            # Submit feedback for face 1 (index 0)
            rec1 = app.handle_feedback("correct", frame=test_frame, predictions=preds, face_idx=0)
            self.assertEqual(rec1["predicted_emotion"], "happy")
            self.assertEqual(rec1["metadata"]["face_index"], 1)

            # Submit feedback for face 2 (index 1)
            rec2 = app.handle_feedback("correct", frame=test_frame, predictions=preds, face_idx=1)
            self.assertEqual(rec2["predicted_emotion"], "angry")
            self.assertEqual(rec2["metadata"]["face_index"], 2)
        finally:
            import shutil
            if test_fb_dir.exists():
                shutil.rmtree(test_fb_dir, ignore_errors=True)

    def test_webcam_feedback_image_persistence(self):
        """Verify saved face crop image exists on disk and is readable."""
        test_fb_dir = PROJECT_ROOT / "tests" / "test_feedback_webcam_persist"
        test_fb_dir.mkdir(parents=True, exist_ok=True)
        try:
            app = WebcamEmotionApp(
                checkpoint_path=self.checkpoint_path,
                feedback_dir=test_fb_dir,
            )

            test_frame = np.full((120, 120, 3), 180, dtype=np.uint8)
            preds = [{"predicted_emotion": "surprise", "confidence": 0.88, "bbox": (10, 10, 60, 60)}]

            record = app.handle_feedback("correct", frame=test_frame, predictions=preds, face_idx=0)
            img_path = Path(record["image_path"])
            self.assertTrue(img_path.exists())
            saved_cv = cv2.imread(str(img_path))
            self.assertIsNotNone(saved_cv)
            self.assertGreater(saved_cv.size, 0)
        finally:
            import shutil
            if test_fb_dir.exists():
                shutil.rmtree(test_fb_dir, ignore_errors=True)

    def test_confidence_threshold_in_webcam_app(self):
        """Verify confidence threshold marks predictions as 'uncertain' when below threshold."""
        app = WebcamEmotionApp(
            checkpoint_path=self.checkpoint_path,
            confidence_threshold=0.9999,  # Force uncertain
        )

        dummy_face = cv2.imread(str(self.real_img_path))
        res = app.predictor.predict(dummy_face, confidence_threshold=app.confidence_threshold)
        self.assertEqual(res["predicted_emotion"], "uncertain")
        self.assertTrue(res["is_uncertain"])


if __name__ == "__main__":
    unittest.main()


