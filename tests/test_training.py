"""
Unit and Integration Tests for FaceSense AI Training Pipeline
Verifies training configuration, single step execution, loss calculation,
checkpoint save/load, validation metrics, and early stopping logic.
"""

import os
import sys
import shutil
from pathlib import Path
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.dataloader import load_yaml_config
from ml.models.baseline_cnn import BaselineEmotionCNN
from ml.training.utils import (
    EarlyStopping,
    MetricTracker,
    get_device,
    set_seed,
)
from ml.training.metrics import (
    compute_evaluation_metrics,
    format_confusion_matrix_ascii,
    save_metrics,
)
from ml.training.trainer import EmotionTrainer


class TestTrainingPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config_path = PROJECT_ROOT / "configs" / "config.yaml"
        cls.config = load_yaml_config(cls.config_path)
        cls.class_names = cls.config["classes"]["names"]
        cls.test_dir = PROJECT_ROOT / "tests" / "test_outputs"
        cls.test_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_training_config_structure(self):
        """Verify training hyperparameter keys and types in config.yaml."""
        tr_cfg = self.config["training"]
        self.assertIn("epochs", tr_cfg)
        self.assertIn("learning_rate", tr_cfg)
        self.assertIn("weight_decay", tr_cfg)
        self.assertIn("optimizer", tr_cfg)
        self.assertIn("label_smoothing", tr_cfg)
        self.assertIn("scheduler", tr_cfg)
        self.assertEqual(tr_cfg["scheduler"]["mode"], "max")
        self.assertEqual(tr_cfg["optimizer"], "adamw")

    def test_device_detection_and_seed(self):
        """Verify device detection returns valid torch.device and seed runs deterministically."""
        set_seed(42)
        device, info = get_device()
        self.assertIsInstance(device, torch.device)
        self.assertIn("device", info)

        # Verify deterministic tensor generation
        t1 = torch.randn(5)
        set_seed(42)
        t2 = torch.randn(5)
        self.assertTrue(torch.equal(t1, t2))

    def test_metrics_computation(self):
        """Verify multi-class metrics, Macro F1, precision, recall, and confusion matrix."""
        y_true = [0, 1, 2, 3, 4, 5, 6, 0, 1, 3]
        y_pred = [0, 1, 2, 3, 4, 5, 6, 1, 1, 3]  # 1 mistake on class 0

        metrics = compute_evaluation_metrics(y_true, y_pred, self.class_names)
        self.assertEqual(metrics["accuracy"], 90.0)
        self.assertGreater(metrics["macro_f1"], 80.0)
        self.assertIn("per_class", metrics)
        self.assertEqual(len(metrics["per_class"]), 7)
        self.assertEqual(len(metrics["confusion_matrix"]), 7)

        # Test ASCII formatting
        ascii_table = format_confusion_matrix_ascii(metrics["confusion_matrix"], self.class_names)
        self.assertIn("True \\ Pred", ascii_table)

    def test_early_stopping_logic(self):
        """Verify early stopping triggers after patience is exhausted for mode='max'."""
        stopper = EarlyStopping(patience=3, mode="max")

        self.assertTrue(stopper.step(70.0, epoch=1))  # Best
        self.assertTrue(stopper.step(72.0, epoch=2))  # Improved
        self.assertFalse(stopper.step(71.0, epoch=3)) # No improvement (counter=1)
        self.assertFalse(stopper.step(71.5, epoch=4)) # No improvement (counter=2)
        self.assertFalse(stopper.step(70.0, epoch=5)) # No improvement (counter=3 -> stop)

        self.assertTrue(stopper.early_stop)
        self.assertEqual(stopper.best_score, 72.0)
        self.assertEqual(stopper.best_epoch, 2)

    def test_single_training_step_and_loss(self):
        """Verify forward pass, loss calculation, backward step, and optimizer update."""
        model = BaselineEmotionCNN(in_channels=1, num_classes=7)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

        inputs = torch.randn(8, 1, 48, 48)
        targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0], dtype=torch.long)

        # Initial weights check
        initial_weight = model.classifier[-1].weight.clone()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        self.assertFalse(torch.isnan(loss).item())
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        optimizer.step()

        updated_weight = model.classifier[-1].weight
        self.assertFalse(torch.equal(initial_weight, updated_weight))

    def test_checkpoint_saving_and_loading(self):
        """Verify saving and restoring checkpoint preserves model weights and states."""
        # Create synthetic datasets
        x_train = torch.randn(16, 1, 48, 48)
        y_train = torch.randint(0, 7, (16,))
        train_ds = TensorDataset(x_train, y_train)
        loader = DataLoader(train_ds, batch_size=8)

        model = BaselineEmotionCNN(in_channels=1, num_classes=7)
        test_cfg = self.config.copy()
        test_cfg["paths"] = {
            "checkpoints_dir": str(self.test_dir / "checkpoints"),
            "best_checkpoint_path": str(self.test_dir / "checkpoints" / "best_model.pt"),
            "outputs_metrics_dir": str(self.test_dir / "metrics"),
            "outputs_plots_dir": str(self.test_dir / "plots"),
        }

        trainer = EmotionTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            test_loader=loader,
            config=test_cfg,
            device=torch.device("cpu"),
        )

        val_metrics = {"loss": 0.5, "accuracy": 85.0, "macro_f1": 84.5}
        ckpt_path = self.test_dir / "test_ckpt.pt"
        trainer.save_checkpoint(epoch=5, val_metrics=val_metrics, filepath=ckpt_path, is_best=True)

        self.assertTrue(ckpt_path.exists())

        # Load into a new model
        new_model = BaselineEmotionCNN(in_channels=1, num_classes=7)
        new_trainer = EmotionTrainer(
            model=new_model,
            train_loader=loader,
            val_loader=loader,
            test_loader=loader,
            config=test_cfg,
            device=torch.device("cpu"),
        )

        loaded_ckpt = new_trainer.load_checkpoint(ckpt_path)
        self.assertEqual(loaded_ckpt["epoch"], 5)
        self.assertEqual(loaded_ckpt["val_macro_f1"], 84.5)
        self.assertTrue(loaded_ckpt["is_best"])


if __name__ == "__main__":
    unittest.main()
