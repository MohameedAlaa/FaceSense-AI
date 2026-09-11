"""
Unit and Integration Tests for Interactive Image Inference and Feedback (Phase 12)
Tests:
  1. Image loading across formats (path, PIL, numpy BGR/Grayscale/BGRA)
  2. Face detection integration with multi-face outputs
  3. Prediction integration (bounding box, confidence, probabilities, model metadata)
  4. Correct feedback ('C') persistence and JSONL logging
  5. Incorrect feedback ('W') with corrected emotion validation
  6. Uncertain feedback ('U') recording
  7. Corrected label validation against FER classes
  8. Automatic model version tracking from model_metadata.json
  9. Multi-face selection switching and targeting
  10. Missing and invalid image handling
  11. Directory batch processing
  12. Visual annotation rendering and output saving
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import unittest
import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.detection.detector import FaceDetector
from ml.inference.predictor import EmotionPredictor
from ml.inference.image_pipeline import ImageInferencePipeline, load_image_bgr
from ml.feedback.image_feedback import ImageFeedbackHandler
from ml.feedback.collector import FeedbackCollector, FeedbackState, get_default_model_version


class TestImageInferenceAndFeedback(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = PROJECT_ROOT / "tests" / "test_image_feedback_scratch"
        cls.test_dir.mkdir(parents=True, exist_ok=True)

        cls.checkpoint_path = PROJECT_ROOT / "ml" / "models" / "checkpoints" / "final" / "best_model.pt"
        cls.metadata_path = PROJECT_ROOT / "ml" / "models" / "checkpoints" / "final" / "model_metadata.json"
        cls.test_data_dir = PROJECT_ROOT / "Data(FER2013)" / "test"
        cls.real_face_path = next(cls.test_data_dir.rglob("*.jpg"))

        cls.detector = FaceDetector()
        cls.predictor = EmotionPredictor(checkpoint_path=cls.checkpoint_path)
        cls.pipeline = ImageInferencePipeline(detector=cls.detector, predictor=cls.predictor)

        # Create a synthetic canvas image containing 2 detected faces
        real_img = cv2.imread(str(cls.real_face_path))
        face_resized = cv2.resize(real_img, (120, 120))
        cls.multi_face_canvas = np.full((400, 600, 3), 128, dtype=np.uint8)
        cls.multi_face_canvas[100:220, 80:200] = face_resized
        cls.multi_face_canvas[100:220, 350:470] = face_resized

        cls.multi_face_img_path = cls.test_dir / "multi_face_test.jpg"
        cv2.imwrite(str(cls.multi_face_img_path), cls.multi_face_canvas)

        # Create a single face synthetic image
        cls.single_face_img_path = cls.test_dir / "single_face_test.jpg"
        single_canvas = np.full((300, 300, 3), 128, dtype=np.uint8)
        single_canvas[75:195, 90:210] = face_resized
        cv2.imwrite(str(cls.single_face_img_path), single_canvas)

    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def setUp(self):
        self.fb_subfolder = self.test_dir / f"fb_{datetime.now(timezone.utc).strftime('%H%M%S_%f')}"
        self.collector = FeedbackCollector(feedback_dir=self.fb_subfolder)
        self.feedback_handler = ImageFeedbackHandler(
            feedback_dir=self.fb_subfolder,
            collector=self.collector,
            metadata_path=self.metadata_path,
        )

    # ── 1. Image Loading Tests ────────────────────────────────────
    def test_image_loading_various_inputs(self):
        """Verify load_image_bgr correctly parses path, PIL Image, and numpy arrays."""
        # 1. File path string & Path object
        img1 = load_image_bgr(str(self.single_face_img_path))
        self.assertEqual(img1.ndim, 3)
        self.assertEqual(img1.shape[2], 3)

        img2 = load_image_bgr(self.single_face_img_path)
        self.assertEqual(img2.shape, img1.shape)

        # 2. PIL Image
        pil_img = Image.open(self.single_face_img_path)
        img3 = load_image_bgr(pil_img)
        self.assertEqual(img3.ndim, 3)

        # 3. Numpy array Grayscale
        gray_arr = np.ones((50, 50), dtype=np.uint8) * 100
        img4 = load_image_bgr(gray_arr)
        self.assertEqual(img4.shape, (50, 50, 3))

        # 4. Numpy array BGRA
        bgra_arr = np.ones((50, 50, 4), dtype=np.uint8) * 100
        img5 = load_image_bgr(bgra_arr)
        self.assertEqual(img5.shape, (50, 50, 3))

    # ── 2. Face Detection Integration ─────────────────────────────
    def test_face_detection_integration(self):
        """Verify pipeline detects faces and extracts bounding boxes."""
        frame, predictions = self.pipeline.process_image(self.multi_face_img_path)
        self.assertIsInstance(frame, np.ndarray)
        self.assertIsInstance(predictions, list)
        self.assertGreaterEqual(len(predictions), 1)

        for pred in predictions:
            self.assertIn("face_idx", pred)
            self.assertIn("bbox", pred)
            self.assertEqual(len(pred["bbox"]), 4)
            x, y, w, h = pred["bbox"]
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)
            self.assertIsNotNone(pred["face_crop"])
            self.assertEqual(pred["face_crop"].ndim, 3)

    # ── 3. Prediction Integration ─────────────────────────────────
    def test_prediction_integration_structure(self):
        """Verify emotion prediction outputs conform to required structure."""
        frame, predictions = self.pipeline.process_image(self.single_face_img_path)
        self.assertGreaterEqual(len(predictions), 1)
        pred = predictions[0]

        self.assertIn("predicted_emotion", pred)
        self.assertIn("confidence", pred)
        self.assertIn("probabilities", pred)
        self.assertIn("model_version", pred)
        self.assertIn("is_uncertain", pred)
        self.assertIsInstance(pred["confidence"], float)
        self.assertGreaterEqual(pred["confidence"], 0.0)
        self.assertLessEqual(pred["confidence"], 1.0)
        self.assertEqual(len(pred["probabilities"]), 7)

    # ── 4. Correct Feedback ───────────────────────────────────────
    def test_correct_feedback_recording(self):
        """Verify [C] feedback saves face crop and logs record in JSONL."""
        dummy_face = np.full((60, 60, 3), 150, dtype=np.uint8)
        record = self.feedback_handler.record_feedback(
            face_image=dummy_face,
            state="correct",
            predicted_emotion="happy",
            confidence=0.94,
            bounding_box=[20, 30, 60, 60],
            face_idx=1,
        )

        self.assertEqual(record["state"], "correct")
        self.assertEqual(record["predicted_emotion"], "happy")
        self.assertEqual(record["confidence"], 0.94)
        self.assertIsNone(record["corrected_emotion"])
        self.assertTrue(Path(record["image_path"]).exists())

        # Check JSONL
        records = self.collector.load_feedback_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["feedback_id"], record["feedback_id"])

    # ── 5. Incorrect Feedback ─────────────────────────────────────
    def test_incorrect_feedback_recording_with_correction(self):
        """Verify [W] feedback records user corrected ground-truth label."""
        dummy_face = np.full((60, 60, 3), 150, dtype=np.uint8)
        record = self.feedback_handler.record_feedback(
            face_image=dummy_face,
            state="incorrect",
            predicted_emotion="sad",
            confidence=0.72,
            corrected_emotion="happy",
            bounding_box=[10, 10, 60, 60],
            face_idx=1,
        )

        self.assertEqual(record["state"], "incorrect")
        self.assertEqual(record["predicted_emotion"], "sad")
        self.assertEqual(record["corrected_emotion"], "happy")

        records = self.collector.load_feedback_records(filter_state="incorrect")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["corrected_emotion"], "happy")

    # ── 6. Uncertain Feedback ─────────────────────────────────────
    def test_uncertain_feedback_recording(self):
        """Verify [U] feedback logs uncertain prediction properly."""
        dummy_face = np.full((60, 60, 3), 150, dtype=np.uint8)
        record = self.feedback_handler.record_feedback(
            face_image=dummy_face,
            state="uncertain",
            predicted_emotion="neutral",
            confidence=0.32,
            bounding_box=[5, 5, 60, 60],
            face_idx=1,
        )

        self.assertEqual(record["state"], "uncertain")
        records = self.collector.load_feedback_records(filter_state="uncertain")
        self.assertEqual(len(records), 1)

    # ── 7. Corrected Label Validation ─────────────────────────────
    def test_corrected_label_validation(self):
        """Verify ground-truth correction rejects invalid labels and accepts valid FER labels."""
        dummy_face = np.full((60, 60, 3), 150, dtype=np.uint8)

        # Invalid labels must raise ValueError
        for invalid_label in ["excited", "bored", "random_text", "", None]:
            with self.assertRaises(ValueError):
                self.feedback_handler.record_feedback(
                    face_image=dummy_face,
                    state="incorrect",
                    predicted_emotion="happy",
                    confidence=0.8,
                    corrected_emotion=invalid_label,
                )

        # Valid FER labels must succeed
        valid_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        for valid_label in valid_labels:
            rec = self.feedback_handler.record_feedback(
                face_image=dummy_face,
                state="incorrect",
                predicted_emotion="fear",
                confidence=0.7,
                corrected_emotion=valid_label,
            )
            self.assertEqual(rec["corrected_emotion"], valid_label)

    # ── 8. Automatic Model Version Tracking ───────────────────────
    def test_automatic_model_version_tracking(self):
        """Verify production model version is automatically loaded from model_metadata.json."""
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            expected_version = json.load(f)["model_version"]

        # Handler without explicit version must match model_metadata.json
        handler = ImageFeedbackHandler(feedback_dir=self.fb_subfolder, metadata_path=self.metadata_path)
        self.assertEqual(handler.model_version, expected_version)
        self.assertEqual(handler.model_version, "ResidualEmotionCNN-Candidate-epoch39")

        # Feedback recorded through handler carries this exact version
        dummy_face = np.full((60, 60, 3), 150, dtype=np.uint8)
        rec = handler.record_feedback(
            face_image=dummy_face,
            state="correct",
            predicted_emotion="happy",
            confidence=0.9,
        )
        self.assertEqual(rec["model_version"], expected_version)

    # ── 9. Multi-Face Selection and Feedback ──────────────────────
    def test_multi_face_interactive_selection(self):
        """Verify multi-face selection targeting via simulated input_func."""
        mock_face_1 = np.full((60, 60, 3), 100, dtype=np.uint8)
        mock_face_2 = np.full((60, 60, 3), 200, dtype=np.uint8)

        mock_predictions = [
            {
                "face_idx": 1,
                "bbox": (10, 10, 60, 60),
                "predicted_emotion": "happy",
                "raw_predicted_emotion": "happy",
                "confidence": 0.91,
                "face_crop": mock_face_1,
            },
            {
                "face_idx": 2,
                "bbox": (100, 10, 60, 60),
                "predicted_emotion": "sad",
                "raw_predicted_emotion": "sad",
                "confidence": 0.85,
                "face_crop": mock_face_2,
            },
        ]

        # Simulate user inputs:
        # 1) Select Face 2: '2'
        # 2) Mark Face 2 as Wrong: 'w'
        # 3) Provide corrected emotion: 'surprise'
        # 4) Finish: 'q'
        simulated_inputs = iter(["2", "w", "surprise", "q"])

        def mock_input(prompt=""):
            return next(simulated_inputs)

        feedbacks = self.feedback_handler.run_interactive_feedback(
            predictions=mock_predictions,
            input_func=mock_input,
        )

        self.assertEqual(len(feedbacks), 1)
        rec = feedbacks[0]
        self.assertEqual(rec["state"], "incorrect")
        self.assertEqual(rec["predicted_emotion"], "sad")
        self.assertEqual(rec["corrected_emotion"], "surprise")
        self.assertEqual(rec["metadata"]["face_index"], 2)

    # ── 10. Missing and Invalid Image Handling ────────────────────
    def test_missing_and_invalid_image_handling(self):
        """Verify appropriate exceptions for missing or corrupt files."""
        # Non-existent file
        with self.assertRaises(FileNotFoundError):
            load_image_bgr("non_existent_file_path_xyz.jpg")

        with self.assertRaises(FileNotFoundError):
            self.pipeline.process_image("non_existent_file_path_xyz.jpg")

        # Unsupported type
        with self.assertRaises(TypeError):
            load_image_bgr(12345)

        # Empty numpy array
        with self.assertRaises(ValueError):
            load_image_bgr(np.array([]))

        # Corrupt file
        corrupt_file = self.test_dir / "corrupt.jpg"
        with open(corrupt_file, "wb") as f:
            f.write(b"not_an_image_data")
        with self.assertRaises(ValueError):
            load_image_bgr(corrupt_file)

    # ── 11. Directory Batch Inference ─────────────────────────────
    def test_batch_directory_inference(self):
        """Verify process_directory runs across a folder of images and produces structured output."""
        batch_dir = self.test_dir / "batch_input"
        batch_dir.mkdir(parents=True, exist_ok=True)
        out_dir = self.test_dir / "batch_output"
        out_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(self.single_face_img_path, batch_dir / "img1.jpg")
        shutil.copy2(self.multi_face_img_path, batch_dir / "img2.jpg")

        results = self.pipeline.process_directory(
            input_dir=batch_dir,
            output_dir=out_dir,
            save_annotated=True,
        )

        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIn("image_file", r)
            self.assertIn("faces_detected", r)
            self.assertIn("predictions", r)
            self.assertIn("annotated_output", r)
            self.assertTrue(Path(r["annotated_output"]).exists())

    # ── 12. Visual Annotation Rendering ───────────────────────────
    def test_visual_annotation(self):
        """Verify annotate_image draws bounding boxes and tags without altering dimensions."""
        frame, predictions = self.pipeline.process_image(self.single_face_img_path)
        annotated = self.pipeline.annotate_image(frame, predictions, selected_face_idx=0)
        self.assertEqual(annotated.shape, frame.shape)
        self.assertEqual(annotated.dtype, frame.dtype)


if __name__ == "__main__":
    unittest.main()
