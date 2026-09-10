"""
Unit and Integration Tests for Phase 11 Retraining, Model Comparison, and Safe Promotion
Tests:
  - MixedEmotionDataset creation and source distribution
  - build_retraining_dataloaders integration
  - ModelComparator calculation and gate logic
  - Rejection path when candidate is worse
  - Successful promotion path when candidate is superior
  - Preservation and backup of previous production checkpoint
  - model_metadata.json tracking and version lineage
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import unittest
import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.dataloader import load_yaml_config
from ml.models.residual_cnn import ResidualEmotionCNN
from ml.training.retrain_dataset import MixedEmotionDataset, build_retraining_dataloaders
from ml.training.model_comparator import ModelComparator, evaluate_model_on_dataloader


class TestRetrainingAndPromotionPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = PROJECT_ROOT / "tests" / "test_retrain_outputs"
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        cls.config_path = PROJECT_ROOT / "configs" / "config.yaml"
        cls.config = load_yaml_config(cls.config_path)
        cls.class_names = cls.config["classes"]["names"]

        # Synthetic test images
        cls.dummy_img_1 = cls.test_dir / "face_base.jpg"
        cls.dummy_img_2 = cls.test_dir / "face_feedback.jpg"
        Image.new("L", (48, 48), color=100).save(cls.dummy_img_1)
        Image.new("L", (48, 48), color=200).save(cls.dummy_img_2)

    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_mixed_emotion_dataset_construction_and_weights(self):
        """Verify dataset combines base and feedback samples and handles feedback_weight."""
        base_samples = [(self.dummy_img_1, 3)] * 10  # 10 'happy'
        feedback_samples = [(self.dummy_img_2, 0)] * 5  # 5 'angry'

        # Test weight = 1.0 (exact combination -> 15 samples)
        mixed_ds = MixedEmotionDataset(
            base_samples=base_samples,
            feedback_samples=feedback_samples,
            class_names=self.class_names,
            feedback_weight=1.0,
        )
        self.assertEqual(len(mixed_ds), 15)
        dist = mixed_ds.get_source_distribution()
        self.assertEqual(dist["base"], 10)
        self.assertEqual(dist["feedback"], 5)

        # Test weight = 2.0 (oversampled feedback -> 10 + 10 = 20 samples)
        mixed_ds_weighted = MixedEmotionDataset(
            base_samples=base_samples,
            feedback_samples=feedback_samples,
            class_names=self.class_names,
            feedback_weight=2.0,
        )
        self.assertEqual(len(mixed_ds_weighted), 20)
        self.assertEqual(mixed_ds_weighted.get_source_distribution()["feedback"], 10)

        # Test item shape
        img, target = mixed_ds[0]
        self.assertEqual(img.shape, (1, 48, 48))
        self.assertIsInstance(target, int)

    def test_build_retraining_dataloaders_pipeline(self):
        """Verify build_retraining_dataloaders builds train, val, test loaders without modifying FER2013."""
        train_loader, val_loader, test_loader, meta = build_retraining_dataloaders(self.config)

        self.assertGreater(meta["base_train_count"], 0)
        self.assertGreaterEqual(meta["feedback_samples_used"], 0)
        self.assertEqual(meta["val_count"], 4307)
        self.assertEqual(meta["test_count"], 7178)
        self.assertEqual(len(meta["class_names"]), 7)

        # Ensure batch shape matches
        images, targets = next(iter(train_loader))
        self.assertEqual(images.shape[1:], (1, 48, 48))

    def test_promotion_gate_rejection_path(self):
        """Verify comparator rejects candidate when Macro F1 is worse."""
        comparator = ModelComparator(
            config_path=self.config_path,
            min_macro_f1_delta=0.5,  # Requires +0.5% improvement
        )

        mock_report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "baseline": {
                "version": "ResidualEmotionCNN-epoch24",
                "checkpoint_path": str(self.test_dir / "base.pt"),
                "test_loss": 1.25,
                "accuracy": 56.0,
                "macro_f1": 51.0,
                "weighted_f1": 54.0,
            },
            "candidate": {
                "version": "ResidualEmotionCNN-Candidate-1",
                "checkpoint_path": str(self.test_dir / "cand.pt"),
                "test_loss": 1.30,
                "accuracy": 55.0,
                "macro_f1": 49.5,  # Worse
                "weighted_f1": 53.0,
            },
            "differences": {
                "accuracy_diff": -1.0,
                "macro_f1_diff": -1.5,
                "weighted_f1_diff": -1.0,
                "loss_diff": 0.05,
                "per_class_diff": {},
            },
            "decision": "REJECT",
            "rejection_reasons": ["Candidate Macro F1 did not improve."],
        }

        # Setup mock production dir
        mock_prod_dir = self.test_dir / "mock_prod_rejection"
        mock_prod_dir.mkdir(parents=True, exist_ok=True)
        prod_best = mock_prod_dir / "best_model.pt"
        torch.save({"dummy": "base_weights"}, prod_best)

        # Apply promotion
        res = comparator.apply_promotion(mock_report, production_dir=mock_prod_dir)
        self.assertFalse(res["promoted"])
        self.assertEqual(res["status"], "REJECTED")

        # Verify best_model.pt was untouched
        loaded = torch.load(prod_best)
        self.assertEqual(loaded.get("dummy"), "base_weights")

    def test_promotion_gate_successful_promotion_and_backup(self):
        """Verify successful promotion copies candidate and backs up previous production checkpoint."""
        comparator = ModelComparator(
            config_path=self.config_path,
            min_macro_f1_delta=0.0,
        )

        mock_prod_dir = self.test_dir / "mock_prod_success"
        mock_prod_dir.mkdir(parents=True, exist_ok=True)
        prod_best = mock_prod_dir / "best_model.pt"
        torch.save({"epoch": 24, "weights": "original_production_weights"}, prod_best)

        cand_ckpt = self.test_dir / "cand_winner.pt"
        torch.save({"epoch": 30, "weights": "better_candidate_weights"}, cand_ckpt)

        meta_file = mock_prod_dir / "model_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({"best_epoch": 24, "experiment_name": "v1_original"}, f)

        mock_report = {
            "decision": "PROMOTE",
            "baseline": {"version": "ResidualEmotionCNN-epoch24"},
            "candidate": {
                "version": "ResidualEmotionCNN-Candidate-winner",
                "checkpoint_path": str(cand_ckpt),
                "accuracy": 58.5,
                "macro_f1": 53.2,
                "weighted_f1": 56.1,
                "test_loss": 1.18,
            },
            "differences": {
                "accuracy_diff": 2.5,
                "macro_f1_diff": 2.2,
                "weighted_f1_diff": 2.1,
                "loss_diff": -0.07,
            },
        }

        # Apply promotion
        res = comparator.apply_promotion(mock_report, production_dir=mock_prod_dir)
        self.assertTrue(res["promoted"])
        self.assertEqual(res["status"], "PROMOTED")
        self.assertIsNotNone(res["backup_checkpoint"])
        self.assertTrue(Path(res["backup_checkpoint"]).exists())

        # Verify new production weights
        new_loaded = torch.load(prod_best)
        self.assertEqual(new_loaded["weights"], "better_candidate_weights")

        # Verify old backup weights
        old_backup = torch.load(res["backup_checkpoint"])
        self.assertEqual(old_backup["weights"], "original_production_weights")

        # Verify metadata JSON updated
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            self.assertEqual(meta["model_version"], "ResidualEmotionCNN-Candidate-winner")
            self.assertEqual(meta["test_macro_f1"], 53.2)


if __name__ == "__main__":
    unittest.main()
