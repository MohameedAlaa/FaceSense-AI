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

    # ── Input Downscaling Optimization & Regression Tests ──────────

    def test_image_smaller_or_equal_max_dimension_bypasses_resize(self):
        """Verify images with max dimension <= 640px are not resized."""
        from unittest.mock import patch

        img_640x480 = np.full((480, 640, 3), 128, dtype=np.uint8)
        img_320x240 = np.full((240, 320, 3), 128, dtype=np.uint8)

        with patch("cv2.resize", wraps=cv2.resize) as spy_resize:
            self.detector.detect_faces(img_640x480)
            self.detector.detect_faces(img_320x240)
            # Neither image should trigger detector-input downscaling resize
            self.assertEqual(spy_resize.call_count, 0)

    def test_image_greater_than_max_dimension_triggers_downscaling(self):
        """Verify images with max dimension > 640px trigger proportional downscaling."""
        from unittest.mock import patch

        img_1280x720 = np.full((720, 1280, 3), 128, dtype=np.uint8)

        with patch("cv2.resize", wraps=cv2.resize) as spy_resize:
            self.detector.detect_faces(img_1280x720)
            # cv2.resize should be called exactly once for downscaling
            self.assertEqual(spy_resize.call_count, 1)
            call_args = spy_resize.call_args[0]
            # Verify target size (new_w, new_h) = (640, 360)
            target_size = call_args[1]
            self.assertEqual(target_size, (640, 360))

    def test_aspect_ratio_preservation_non_standard_resolution(self):
        """Verify non-standard aspect ratios are scaled proportionally without distortion."""
        from unittest.mock import patch

        # 1200x400 (3:1 aspect ratio)
        img_wide = np.full((400, 1200, 3), 128, dtype=np.uint8)

        with patch("cv2.resize", wraps=cv2.resize) as spy_resize:
            self.detector.detect_faces(img_wide)
            self.assertEqual(spy_resize.call_count, 1)
            target_size = spy_resize.call_args[0][1]
            # 640 max dim -> (640, 213)
            self.assertEqual(target_size[0], 640)
            self.assertEqual(target_size[1], int(round(400 * (640 / 1200))))

    def test_original_input_image_not_mutated(self):
        """Verify detect_faces does not mutate the input array in-place."""
        img = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
        img_copy = img.copy()

        self.detector.detect_faces(img)
        self.assertTrue(np.array_equal(img, img_copy))

    def test_bounding_box_mapping_accuracy_and_validity(self):
        """Verify bounding boxes on a large 1280x720 image map back accurately to original space."""
        fer_face = cv2.imread(str(self.real_img_path))
        canvas = np.full((720, 1280, 3), 160, dtype=np.uint8)
        face_resized = cv2.resize(fer_face, (200, 200))
        # Place face in bottom-right quadrant: x in [700, 900], y in [350, 550]
        canvas[350:550, 700:900] = face_resized

        bboxes = self.detector.detect_faces(canvas)
        self.assertGreaterEqual(len(bboxes), 1)

        bbox = bboxes[0]
        x, y, w, h = bbox

        # Bounding box must be in original coordinate space
        self.assertGreater(x, 640, "Detected x coordinate should be in the right half of the 1280px image")
        self.assertGreater(y, 250, "Detected y coordinate should be in the lower half of the 720px image")
        self.assertGreater(w, 100)
        self.assertGreater(h, 100)

        # Must be strictly within canvas bounds
        self.assertLessEqual(x + w, 1280)
        self.assertLessEqual(y + h, 720)

        # Crop must succeed with original image
        crop = self.detector.crop_face(canvas, bbox, margin_ratio=0.0)
        self.assertGreater(crop.shape[0], 0)
        self.assertGreater(crop.shape[1], 0)

    def test_multi_face_detection_and_ordering(self):
        """Verify multi-face detection on 1280x720 image returns all faces with preserved ordering."""
        fer_face = cv2.imread(str(self.real_img_path))
        canvas = np.full((720, 1280, 3), 160, dtype=np.uint8)
        face_1 = cv2.resize(fer_face, (180, 180))
        face_2 = cv2.resize(fer_face, (180, 180))

        # Face 1 on left, Face 2 on right
        canvas[250:430, 200:380] = face_1
        canvas[250:430, 800:980] = face_2

        bboxes = self.detector.detect_faces(canvas)
        self.assertEqual(len(bboxes), 2)

        # Ordering must be consistent across multiple calls
        bboxes_second_call = self.detector.detect_faces(canvas)
        self.assertEqual(bboxes, bboxes_second_call)

        # Verify coordinates of both faces map to original space
        x_coords = [b[0] for b in bboxes]
        self.assertTrue(any(x < 640 for x in x_coords), "Should detect a face in left half")
        self.assertTrue(any(x > 640 for x in x_coords), "Should detect a face in right half")

    def test_no_face_image_returns_empty_list(self):
        """Verify an image without faces returns an empty list."""
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        bboxes = self.detector.detect_faces(blank)
        self.assertEqual(bboxes, [])

    def test_custom_and_disabled_max_detector_dim(self):
        """Verify configurable max_detector_dim parameter behavior."""
        from unittest.mock import patch

        # Disabled downscaling
        detector_disabled = FaceDetector(max_detector_dim=None)
        img_large = np.full((720, 1280, 3), 128, dtype=np.uint8)

        with patch("cv2.resize", wraps=cv2.resize) as spy_resize:
            detector_disabled.detect_faces(img_large)
            self.assertEqual(spy_resize.call_count, 0)

        # Custom smaller dimension
        detector_custom = FaceDetector(max_detector_dim=320)
        with patch("cv2.resize", wraps=cv2.resize) as spy_resize:
            detector_custom.detect_faces(img_large)
            self.assertEqual(spy_resize.call_count, 1)
            target_size = spy_resize.call_args[0][1]
            self.assertEqual(target_size, (320, 180))


if __name__ == "__main__":
    unittest.main()



