"""
Unit and Integration Tests for FaceSense AI Data Pipeline
Verifies dataset loading, image shapes, num_classes, reproducible stratified splits,
and dataloader batch shapes.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import torch
import torchvision.transforms as T
from PIL import Image

from ml.data.dataset import (
    FER2013Dataset,
    get_image_paths_and_targets,
    create_train_val_split,
)
from ml.data.dataloader import (
    load_yaml_config,
    get_transforms,
    compute_class_weights,
    create_weighted_sampler,
    build_dataloaders,
)


class TestDataPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
        cls.config = load_yaml_config(config_path)
        cls.train_dir = PROJECT_ROOT / cls.config["dataset"]["train_split_dir"]
        cls.test_dir = PROJECT_ROOT / cls.config["dataset"]["test_split_dir"]
        cls.class_mapping = cls.config["classes"]["mapping"]
        cls.class_names = cls.config["classes"]["names"]

    def test_config_loading(self):
        """Verify configuration is loaded and has expected keys."""
        self.assertEqual(self.config["classes"]["num_classes"], 7)
        self.assertEqual(len(self.class_names), 7)
        self.assertEqual(len(self.class_mapping), 7)
        self.assertEqual(self.config["dataset"]["image_size"], [48, 48])

    def test_dataset_loading_and_counts(self):
        """Verify reading image paths and target labels without mutating original files."""
        train_samples = get_image_paths_and_targets(self.train_dir, self.class_mapping)
        test_samples = get_image_paths_and_targets(self.test_dir, self.class_mapping)

        self.assertEqual(len(train_samples), 28709)
        self.assertEqual(len(test_samples), 7178)

    def test_train_validation_stratified_split(self):
        """Verify 85% train and 15% validation split with exact class proportions."""
        val_ratio = 0.15
        seed = 42
        train_samples, val_samples = create_train_val_split(
            self.train_dir,
            self.class_mapping,
            val_ratio=val_ratio,
            seed=seed,
        )

        expected_total = 28709
        self.assertEqual(len(train_samples) + len(val_samples), expected_total)
        # sklearn stratified split produces 4307 val (15.002%) and 24402 train (84.998%)
        self.assertEqual(len(val_samples), 4307)
        self.assertEqual(len(train_samples), 24402)

        # Check stratification per class (each class in val should be ~15%)
        train_ds = FER2013Dataset(train_samples, self.class_names)
        val_ds = FER2013Dataset(val_samples, self.class_names)

        train_counts = train_ds.get_class_counts()
        val_counts = val_ds.get_class_counts()

        for cls_name in self.class_names:
            total_cls = train_counts[cls_name] + val_counts[cls_name]
            val_pct = val_counts[cls_name] / total_cls
            self.assertAlmostEqual(val_pct, 0.15, delta=0.01)

    def test_dataset_item_shape_and_grayscale(self):
        """Verify FER2013Dataset returns 1x48x48 tensor and integer label."""
        transform = get_transforms(img_size=(48, 48), is_train=False)
        test_samples = get_image_paths_and_targets(self.test_dir, self.class_mapping)
        dataset = FER2013Dataset(test_samples[:10], self.class_names, transform=transform)

        image, label = dataset[0]
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape, (1, 48, 48))
        self.assertIsInstance(label, int)
        self.assertTrue(0 <= label < 7)

    def test_augmentation_transforms(self):
        """Verify training augmentation produces valid 1x48x48 tensor."""
        aug_transform = get_transforms(
            img_size=(48, 48),
            is_train=True,
            hflip_prob=0.5,
            rotation_deg=10,
            crop_scale=(0.85, 1.0),
        )
        test_samples = get_image_paths_and_targets(self.test_dir, self.class_mapping)
        dataset = FER2013Dataset(test_samples[:5], self.class_names, transform=aug_transform)

        image, label = dataset[0]
        self.assertEqual(image.shape, (1, 48, 48))
        self.assertTrue(torch.is_tensor(image))

    def test_class_weights_and_sampler(self):
        """Verify computed class weights and WeightedRandomSampler."""
        train_samples, _ = create_train_val_split(
            self.train_dir,
            self.class_mapping,
            val_ratio=0.15,
            seed=42,
        )
        train_ds = FER2013Dataset(train_samples, self.class_names)
        labels = train_ds.get_labels()

        weights = compute_class_weights(labels, num_classes=7, mode="balanced")
        self.assertEqual(len(weights), 7)

        # Disgust (index 1) is the rarest class and must receive the highest weight
        disgust_idx = self.class_mapping["disgust"]
        happy_idx = self.class_mapping["happy"]
        self.assertGreater(weights[disgust_idx].item(), weights[happy_idx].item())

        sampler = create_weighted_sampler(labels, num_classes=7)
        self.assertEqual(len(sampler), len(labels))

    def test_build_dataloaders_pipeline(self):
        """Verify end-to-end building of DataLoaders and metadata extraction."""
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
        train_loader, val_loader, test_loader, meta = build_dataloaders(config_path)

        self.assertEqual(meta["train_count"], 24402)
        self.assertEqual(meta["val_count"], 4307)
        self.assertEqual(meta["test_count"], 7178)
        self.assertEqual(meta["num_classes"], 7)

        # Fetch one batch from train loader to check shape
        images, labels = next(iter(train_loader))
        self.assertEqual(images.shape[1:], (1, 48, 48))
        self.assertEqual(images.shape[0], self.config["dataloader"]["batch_size"])
        self.assertEqual(labels.shape[0], self.config["dataloader"]["batch_size"])


if __name__ == "__main__":
    unittest.main()
