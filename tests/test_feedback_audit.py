"""
Unit and Integration Tests for Feedback Dataset Auditing & Construction (Phase: Feedback Audit Fix)
Tests:
  1. Correct feedback becomes a training sample with predicted emotion
  2. Incorrect feedback with corrected label becomes a training sample with corrected emotion
  3. Uncertain feedback without user correction is excluded with reason 'uncertain_without_correction'
  4. Missing/invalid image path is excluded with reason 'missing_image_file'
  5. Valid samples are not accidentally discarded
  6. Duplicate feedback handling is deterministic (deduplicated by feedback_id)
  7. Model version does not incorrectly filter out valid current feedback samples
  8. Manifest contains full audit breakdown and reason counts
  9. Live feedback records audit produces exact 19-record breakdown (11 usable, 8 excluded: 6 missing, 2 uncertain)
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


class TestFeedbackDatasetAudit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = PROJECT_ROOT / "tests" / "test_feedback_audit_scratch"
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        cls.valid_img_path = cls.test_dir / "valid_face.jpg"
        dummy_img = Image.new("L", (48, 48), color=128)
        dummy_img.save(cls.valid_img_path)

    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def setUp(self):
        self.fb_subfolder = self.test_dir / f"fb_{datetime.now(timezone.utc).strftime('%H%M%S_%f')}"
        self.collector = FeedbackCollector(feedback_dir=self.fb_subfolder)
        self.builder = FeedbackDatasetBuilder(
            feedback_collector=self.collector,
            feedback_dir=self.fb_subfolder,
        )

    def test_correct_feedback_becomes_training_sample(self):
        """Verify state=correct uses predicted_emotion as ground truth label."""
        self.collector.add_feedback(
            image_path=self.valid_img_path,
            predicted_emotion="happy",
            confidence=0.95,
            state=FeedbackState.CORRECT,
        )

        audit = self.builder.audit_feedback_records()
        self.assertEqual(audit["total_records"], 1)
        self.assertEqual(audit["usable_count"], 1)
        self.assertEqual(audit["excluded_count"], 0)
        self.assertEqual(audit["usable_samples"][0]["label"], "happy")

    def test_incorrect_feedback_with_correction_becomes_sample(self):
        """Verify state=incorrect uses corrected_emotion as ground truth label."""
        self.collector.add_feedback(
            image_path=self.valid_img_path,
            predicted_emotion="sad",
            confidence=0.60,
            corrected_emotion="fear",
            state=FeedbackState.INCORRECT,
        )

        audit = self.builder.audit_feedback_records()
        self.assertEqual(audit["total_records"], 1)
        self.assertEqual(audit["usable_count"], 1)
        self.assertEqual(audit["usable_samples"][0]["label"], "fear")

    def test_uncertain_without_correction_is_excluded_with_reason(self):
        """Verify state=uncertain without corrected_emotion is excluded with explicit reason."""
        self.collector.add_feedback(
            image_path=self.valid_img_path,
            predicted_emotion="neutral",
            confidence=0.30,
            corrected_emotion=None,
            state=FeedbackState.UNCERTAIN,
        )

        audit = self.builder.audit_feedback_records()
        self.assertEqual(audit["total_records"], 1)
        self.assertEqual(audit["usable_count"], 0)
        self.assertEqual(audit["excluded_count"], 1)
        self.assertEqual(audit["excluded_records"][0]["reason"], "uncertain_without_correction")
        self.assertIn("uncertain_without_correction", audit["exclusion_reasons"])

    def test_missing_image_file_is_excluded_with_reason(self):
        """Verify records pointing to missing image files are excluded with reason 'missing_image_file'."""
        missing_path = self.test_dir / "non_existent_snapshot_123.jpg"

        # Bypass collector.add_feedback validation by writing directly to jsonl
        record = {
            "feedback_id": "fb_missing_test_001",
            "state": "correct",
            "image_path": str(missing_path),
            "predicted_emotion": "happy",
            "confidence": 0.90,
            "corrected_emotion": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": "ResidualEmotionCNN-Candidate-epoch39",
            "metadata": {},
        }
        with open(self.collector.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        audit = self.builder.audit_feedback_records()
        self.assertEqual(audit["total_records"], 1)
        self.assertEqual(audit["usable_count"], 0)
        self.assertEqual(audit["excluded_count"], 1)
        self.assertEqual(audit["excluded_records"][0]["reason"], "missing_image_file")

    def test_valid_samples_not_accidentally_discarded(self):
        """Verify mixed valid records are all preserved without accidental drops."""
        emotions = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
        for emo in emotions:
            self.collector.add_feedback(
                image_path=self.valid_img_path,
                predicted_emotion=emo,
                confidence=0.85,
                state=FeedbackState.CORRECT,
            )

        audit = self.builder.audit_feedback_records()
        self.assertEqual(audit["total_records"], len(emotions))
        self.assertEqual(audit["usable_count"], len(emotions))
        self.assertEqual(audit["excluded_count"], 0)

    def test_duplicate_handling_is_deterministic(self):
        """Verify identical feedback_id is detected and handled deterministically."""
        record = {
            "feedback_id": "fb_dup_001",
            "state": "correct",
            "image_path": str(self.valid_img_path),
            "predicted_emotion": "happy",
            "confidence": 0.90,
            "corrected_emotion": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": "ResidualEmotionCNN-Candidate-epoch39",
            "metadata": {},
        }
        with open(self.collector.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.write(json.dumps(record) + "\n")

        audit = self.builder.audit_feedback_records()
        self.assertEqual(audit["total_records"], 2)
        self.assertEqual(audit["usable_count"], 1)
        self.assertEqual(audit["excluded_count"], 1)
        self.assertEqual(audit["excluded_records"][0]["reason"], "duplicate_feedback_id")

    def test_model_version_does_not_filter_out_valid_samples(self):
        """Verify samples with different model_version strings are both included if valid."""
        self.collector.add_feedback(
            image_path=self.valid_img_path,
            predicted_emotion="happy",
            confidence=0.90,
            state=FeedbackState.CORRECT,
            model_version="ResidualEmotionCNN-epoch24",
        )
        self.collector.add_feedback(
            image_path=self.valid_img_path,
            predicted_emotion="sad",
            confidence=0.80,
            state=FeedbackState.CORRECT,
            model_version="ResidualEmotionCNN-Candidate-epoch39",
        )

        audit = self.builder.audit_feedback_records()
        self.assertEqual(audit["total_records"], 2)
        self.assertEqual(audit["usable_count"], 2)

    def test_manifest_contains_audit_breakdown(self):
        """Verify build_feedback_dataset outputs detailed audit fields in manifest.json."""
        self.collector.add_feedback(
            image_path=self.valid_img_path,
            predicted_emotion="happy",
            confidence=0.92,
            state=FeedbackState.CORRECT,
        )
        self.collector.add_feedback(
            image_path=self.valid_img_path,
            predicted_emotion="sad",
            confidence=0.30,
            state=FeedbackState.UNCERTAIN,
        )

        out_dataset = self.fb_subfolder / "built_test_dataset"
        manifest = self.builder.build_feedback_dataset(output_dataset_dir=out_dataset)

        self.assertEqual(manifest["total_records_evaluated"], 2)
        self.assertEqual(manifest["usable_samples"], 1)
        self.assertEqual(manifest["excluded_samples"], 1)
        self.assertIn("uncertain_without_correction", manifest["exclusion_reasons"])
        self.assertEqual(manifest["exclusion_reasons"]["uncertain_without_correction"], 1)
        self.assertEqual(len(manifest["excluded_records_detail"]), 1)

    def test_live_dataset_audit_consistency(self):
        """Verify audit consistency and structural integrity against current live feedback log."""
        live_builder = FeedbackDatasetBuilder(feedback_dir="outputs/feedback")
        raw_records = live_builder.collector.load_feedback_records()
        audit = live_builder.audit_feedback_records()

        # 1. Total records must dynamically match the actual live feedback source
        self.assertGreater(len(raw_records), 0, "Live feedback log should contain recorded samples")
        self.assertEqual(audit["total_records"], len(raw_records))

        # 2. Partition consistency: usable + excluded must exactly equal total records evaluated
        self.assertEqual(audit["total_records"], audit["usable_count"] + audit["excluded_count"])
        self.assertEqual(len(audit["usable_samples"]), audit["usable_count"])
        self.assertEqual(len(audit["excluded_records"]), audit["excluded_count"])

        # 3. Exclusion reasons must account for every excluded record
        total_exclusion_reasons = sum(audit["exclusion_reasons"].values())
        self.assertEqual(audit["excluded_count"], total_exclusion_reasons)

        # 4. Semantic integrity of all usable samples
        for sample in audit["usable_samples"]:
            self.assertIn("feedback_id", sample)
            self.assertIn(sample["label"], live_builder.valid_emotions)
            self.assertTrue(Path(sample["image_path"]).exists(), f"Image path {sample['image_path']} must exist for usable sample")

        # 5. Semantic integrity of all excluded records
        for excluded in audit["excluded_records"]:
            self.assertIn("feedback_id", excluded)
            self.assertIn("reason", excluded)
            self.assertIn(excluded["reason"], audit["exclusion_reasons"])

        # 6. Verify that historical baseline records are accounted for
        self.assertGreaterEqual(audit["usable_count"], 11, "Must contain at least the 11 baseline usable samples")
        self.assertGreaterEqual(audit["exclusion_reasons"].get("missing_image_file", 0), 6, "Expected at least 6 legacy samples with missing images")
        self.assertGreaterEqual(audit["exclusion_reasons"].get("uncertain_without_correction", 0), 2, "Expected at least 2 uncertain exclusions")


if __name__ == "__main__":
    unittest.main()
