"""
Unit and Integration Tests for FaceSense AI Feedback Module (Phase 10)
Tests FeedbackCollector record creation, validation checks (labels, confidence, bounding box,
timestamp, existing image path), saving and loading JSONL, CSV export, summary statistics,
model version tracking, and FeedbackDatasetBuilder dataset generation.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import unittest
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.feedback.collector import FeedbackCollector, FeedbackState
from ml.feedback.dataset_builder import FeedbackDatasetBuilder
from ml.inference.predictor import EmotionPredictor


class TestFeedbackModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = PROJECT_ROOT / "tests" / "test_feedback_outputs"
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        cls.checkpoint_path = PROJECT_ROOT / "ml" / "models" / "checkpoints" / "final" / "best_model.pt"
        cls.test_data_dir = PROJECT_ROOT / "Data(FER2013)" / "test"
        cls.real_img_path = next(cls.test_data_dir.rglob("*.jpg"))

        # Create a synthetic image in test_dir
        cls.synthetic_img_path = cls.test_dir / "sample_face.jpg"
        dummy_img = Image.new("L", (48, 48), color=128)
        dummy_img.save(cls.synthetic_img_path)

    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def setUp(self):
        # Create a clean per-test feedback collector folder
        self.fb_subfolder = self.test_dir / f"fb_{datetime.now(timezone.utc).strftime('%H%M%S_%f')}"
        self.collector = FeedbackCollector(feedback_dir=self.fb_subfolder)

    def test_feedback_record_creation_correct_state(self):
        """Verify recording a correct prediction stores proper fields and appends to JSONL."""
        record = self.collector.add_feedback(
            image_path=self.synthetic_img_path,
            predicted_emotion="happy",
            confidence=0.92,
            corrected_emotion=None,
            bounding_box=[10, 15, 48, 48],
            state=FeedbackState.CORRECT,
            model_version="ResidualEmotionCNN-epoch24",
        )

        self.assertIn("feedback_id", record)
        self.assertEqual(record["state"], "correct")
        self.assertEqual(record["predicted_emotion"], "happy")
        self.assertEqual(record["confidence"], 0.92)
        self.assertIsNone(record["corrected_emotion"])
        self.assertEqual(record["bounding_box"], [10, 15, 48, 48])
        self.assertEqual(record["model_version"], "ResidualEmotionCNN-epoch24")
        self.assertTrue(Path(record["image_path"]).exists())

        # Verify JSONL log on disk
        loaded = self.collector.load_feedback_records()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["feedback_id"], record["feedback_id"])

    def test_feedback_record_creation_incorrect_state(self):
        """Verify recording an incorrect prediction with user corrected label."""
        record = self.collector.add_feedback(
            image_path=self.synthetic_img_path,
            predicted_emotion="sad",
            confidence=0.65,
            corrected_emotion="angry",
            bounding_box=[0, 0, 48, 48],
            state=FeedbackState.INCORRECT,
            model_version="ResidualEmotionCNN-epoch24",
        )

        self.assertEqual(record["state"], "incorrect")
        self.assertEqual(record["predicted_emotion"], "sad")
        self.assertEqual(record["corrected_emotion"], "angry")
        self.assertEqual(record["confidence"], 0.65)

    def test_feedback_record_creation_uncertain_state(self):
        """Verify recording an uncertain prediction."""
        record = self.collector.add_feedback(
            image_path=self.synthetic_img_path,
            predicted_emotion="uncertain",
            confidence=0.25,
            corrected_emotion="fear",
            bounding_box=[5, 5, 40, 40],
            state=FeedbackState.UNCERTAIN,
            model_version="ResidualEmotionCNN-epoch24",
        )

        self.assertEqual(record["state"], "uncertain")
        self.assertEqual(record["predicted_emotion"], "uncertain")
        self.assertEqual(record["corrected_emotion"], "fear")
        self.assertEqual(record["confidence"], 0.25)

    def test_validation_invalid_emotion_label(self):
        """Verify ValueError is raised when predicted or corrected emotion is not recognized."""
        with self.assertRaises(ValueError):
            self.collector.add_feedback(
                image_path=self.synthetic_img_path,
                predicted_emotion="joyful_unknown",
                confidence=0.8,
            )

        with self.assertRaises(ValueError):
            self.collector.add_feedback(
                image_path=self.synthetic_img_path,
                predicted_emotion="happy",
                confidence=0.8,
                corrected_emotion="invalid_emotion_label",
                state=FeedbackState.INCORRECT,
            )

    def test_validation_invalid_confidence_range(self):
        """Verify ValueError is raised when confidence is outside [0.0, 1.0]."""
        with self.assertRaises(ValueError):
            self.collector.add_feedback(
                image_path=self.synthetic_img_path,
                predicted_emotion="happy",
                confidence=1.5,
            )

        with self.assertRaises(ValueError):
            self.collector.add_feedback(
                image_path=self.synthetic_img_path,
                predicted_emotion="happy",
                confidence=-0.1,
            )

    def test_validation_missing_image_path(self):
        """Verify FileNotFoundError is raised when image path does not exist on disk."""
        non_existent_img = self.test_dir / "does_not_exist_face.jpg"
        with self.assertRaises(FileNotFoundError):
            self.collector.add_feedback(
                image_path=non_existent_img,
                predicted_emotion="happy",
                confidence=0.8,
            )

    def test_validation_invalid_bounding_box(self):
        """Verify ValueError is raised for invalid bounding box dimensions."""
        with self.assertRaises(ValueError):
            self.collector.add_feedback(
                image_path=self.synthetic_img_path,
                predicted_emotion="happy",
                confidence=0.8,
                bounding_box=[10, 20],  # only 2 elements instead of 4
            )

        with self.assertRaises(ValueError):
            self.collector.add_feedback(
                image_path=self.synthetic_img_path,
                predicted_emotion="happy",
                confidence=0.8,
                bounding_box=[10, 20, -5, 48],  # negative width
            )

    def test_multiple_feedback_records_and_summary_statistics(self):
        """Verify logging multiple records and extracting summary statistics."""
        # 1. Correct record
        self.collector.add_feedback(
            image_path=self.synthetic_img_path,
            predicted_emotion="happy",
            confidence=0.9,
            state=FeedbackState.CORRECT,
            model_version="Model-v1",
        )
        # 2. Incorrect record
        self.collector.add_feedback(
            image_path=self.synthetic_img_path,
            predicted_emotion="neutral",
            confidence=0.6,
            corrected_emotion="sad",
            state=FeedbackState.INCORRECT,
            model_version="Model-v1",
        )
        # 3. Uncertain record
        self.collector.add_feedback(
            image_path=self.synthetic_img_path,
            predicted_emotion="uncertain",
            confidence=0.3,
            corrected_emotion="surprise",
            state=FeedbackState.UNCERTAIN,
            model_version="Model-v2",
        )

        all_records = self.collector.load_feedback_records()
        self.assertEqual(len(all_records), 3)

        stats = self.collector.get_summary_statistics()
        self.assertEqual(stats["total_records"], 3)
        self.assertEqual(stats["by_state"]["correct"], 1)
        self.assertEqual(stats["by_state"]["incorrect"], 1)
        self.assertEqual(stats["by_state"]["uncertain"], 1)
        self.assertEqual(stats["by_model_version"]["Model-v1"], 2)
        self.assertEqual(stats["by_model_version"]["Model-v2"], 1)

    def test_csv_export_functionality(self):
        """Verify exporting feedback records to CSV."""
        self.collector.add_feedback(
            image_path=self.synthetic_img_path,
            predicted_emotion="disgust",
            confidence=0.88,
            state=FeedbackState.CORRECT,
        )

        csv_file = self.collector.export_to_csv()
        self.assertTrue(csv_file.exists())
        self.assertGreater(csv_file.stat().st_size, 0)

        with open(csv_file, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("feedback_id", content)
            self.assertIn("disgust", content)

    def test_save_face_crop_directly_from_numpy(self):
        """Verify save_face_crop_and_add_feedback persists numpy arrays into feedback store."""
        synthetic_crop = np.full((48, 48, 3), 200, dtype=np.uint8)
        record = self.collector.save_face_crop_and_add_feedback(
            frame_or_face=synthetic_crop,
            predicted_emotion="surprise",
            confidence=0.77,
            state=FeedbackState.CORRECT,
        )
        self.assertTrue(Path(record["image_path"]).exists())
        self.assertEqual(record["predicted_emotion"], "surprise")

    def test_dataset_builder_pipeline(self):
        """Verify FeedbackDatasetBuilder extracts valid labels and constructs train/val splits."""
        # Add 4 records across different emotions
        self.collector.add_feedback(
            image_path=self.synthetic_img_path,
            predicted_emotion="happy",
            confidence=0.95,
            state=FeedbackState.CORRECT,
        )
        self.collector.add_feedback(
            image_path=self.synthetic_img_path,
            predicted_emotion="sad",
            confidence=0.55,
            corrected_emotion="angry",
            state=FeedbackState.INCORRECT,
        )
        self.collector.add_feedback(
            image_path=self.synthetic_img_path,
            predicted_emotion="uncertain",
            confidence=0.2,
            corrected_emotion="fear",
            state=FeedbackState.UNCERTAIN,
        )

        dataset_out = self.fb_subfolder / "test_built_dataset"
        builder = FeedbackDatasetBuilder(
            feedback_collector=self.collector,
            feedback_dir=self.fb_subfolder,
        )

        manifest = builder.build_feedback_dataset(
            output_dataset_dir=dataset_out,
            val_split_ratio=0.33,
            seed=42,
        )

        self.assertEqual(manifest["total_samples"], 3)
        self.assertEqual(manifest["train_samples"] + manifest["val_samples"], 3)
        self.assertTrue((dataset_out / "manifest.json").exists())
        self.assertTrue((dataset_out / "train").exists())
        self.assertTrue((dataset_out / "val").exists())

    def test_model_version_tracking_integration_with_predictor(self):
        """Verify EmotionPredictor outputs model_version and feeds cleanly into collector."""
        predictor = EmotionPredictor(checkpoint_path=self.checkpoint_path)
        pred_res = predictor.predict(self.real_img_path)

        self.assertIn("model_version", pred_res)
        self.assertEqual(pred_res["model_version"], f"ResidualEmotionCNN-epoch{predictor.best_epoch}")

        record = self.collector.add_feedback(
            image_path=self.real_img_path,
            predicted_emotion=pred_res["predicted_emotion"],
            confidence=pred_res["confidence"],
            state=FeedbackState.CORRECT,
            model_version=pred_res["model_version"],
        )

        self.assertEqual(record["model_version"], pred_res["model_version"])


if __name__ == "__main__":
    unittest.main()
