"""
Unit and Integration Tests for FaceSense AI Models (Baseline CNN & Residual CNN V2)
Verifies model instantiation, summary calculations, input/output tensor shapes,
forward pass execution, backward pass gradients, residual connections, and builder selection.
"""

import sys
from pathlib import Path
import unittest
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.dataloader import load_yaml_config
from ml.models.baseline_cnn import (
    BaselineEmotionCNN,
    ConvBlock,
)
from ml.models.residual_cnn import (
    ResidualEmotionCNN,
    ResidualBlock,
)
from ml.models.builder import (
    build_model_from_config,
    get_model_summary,
    print_model_summary,
)


class TestBaselineEmotionCNN(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_yaml_config(PROJECT_ROOT / "configs" / "config.yaml")
        cls.num_classes = cls.config["classes"]["num_classes"]
        cls.in_channels = cls.config["dataset"]["channels"]

    def test_model_instantiation_default(self):
        """Verify model instantiates with default parameters."""
        model = BaselineEmotionCNN()
        self.assertIsInstance(model, BaselineEmotionCNN)
        self.assertEqual(model.in_channels, 1)
        self.assertEqual(model.num_classes, 7)

    def test_model_instantiation_from_config(self):
        """Verify model builds directly from config.yaml and matches the configured architecture."""
        from ml.models.residual_cnn import ResidualEmotionCNN
        arch = str(self.config.get("model", {}).get("architecture", "BaselineEmotionCNN")).lower()
        expected_cls = ResidualEmotionCNN if ("res" in arch or "v2" in arch) else BaselineEmotionCNN
        model = build_model_from_config(self.config)
        self.assertIsInstance(model, expected_cls)
        self.assertEqual(model.in_channels, self.in_channels)
        self.assertEqual(model.num_classes, self.num_classes)

    def test_forward_pass_batch_shapes(self):
        """Verify forward pass on multiple batch sizes with input [B, 1, 48, 48] -> [B, 7]."""
        model = BaselineEmotionCNN(in_channels=1, num_classes=7)
        model.eval()

        for batch_size in [1, 4, 16, 64]:
            x = torch.randn(batch_size, 1, 48, 48)
            with torch.no_grad():
                out = model(x)
            
            self.assertEqual(out.shape, (batch_size, 7))
            self.assertFalse(torch.isnan(out).any().item(), f"Output contains NaN for batch {batch_size}")
            self.assertFalse(torch.isinf(out).any().item(), f"Output contains Inf for batch {batch_size}")

    def test_conv_block(self):
        """Verify intermediate ConvBlock operations and downsampling."""
        block = ConvBlock(in_channels=1, out_channels=32, dropout_rate=0.25, pool=True)
        block.eval()
        x = torch.randn(2, 1, 48, 48)
        with torch.no_grad():
            out = block(x)
        self.assertEqual(out.shape, (2, 32, 24, 24))

    def test_gradient_flow_backward(self):
        """Verify backward pass computes valid gradients without NaN or zero-grad stagnation."""
        model = BaselineEmotionCNN(in_channels=1, num_classes=7)
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        
        inputs = torch.randn(4, 1, 48, 48)
        targets = torch.tensor([0, 2, 4, 6], dtype=torch.long)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        self.assertFalse(torch.isnan(loss).item())
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing for {name}")
                self.assertFalse(torch.isnan(param.grad).any().item(), f"Gradient NaN in {name}")

    def test_model_summary(self):
        """Verify model summary utility outputs correct dictionary fields and positive parameter counts."""
        model = BaselineEmotionCNN(in_channels=1, num_classes=7)
        summary = get_model_summary(model, input_size=(1, 1, 48, 48))

        self.assertEqual(summary["model_class"], "BaselineEmotionCNN")
        self.assertEqual(summary["input_shape"], [1, 1, 48, 48])
        self.assertEqual(summary["output_shape"], [1, 7])
        self.assertGreater(summary["total_params"], 100_000)
        self.assertEqual(summary["total_params"], summary["trainable_params"] + summary["non_trainable_params"])


class TestResidualEmotionCNN(unittest.TestCase):

    def setUp(self):
        self.model = ResidualEmotionCNN(in_channels=1, num_classes=7)

    def test_model_instantiation(self):
        """Verify ResidualEmotionCNN instantiates with valid parameters."""
        self.assertIsInstance(self.model, ResidualEmotionCNN)
        self.assertEqual(self.model.in_channels, 1)
        self.assertEqual(self.model.num_classes, 7)

    def test_forward_pass_batch_shapes(self):
        """Verify forward pass produces [B, 7] shape without NaN/Inf across different batch sizes."""
        self.model.eval()
        for batch_size in [1, 4, 16, 64]:
            x = torch.randn(batch_size, 1, 48, 48)
            with torch.no_grad():
                out = self.model(x)
            self.assertEqual(out.shape, (batch_size, 7))
            self.assertFalse(torch.isnan(out).any().item(), f"NaN in batch size {batch_size}")
            self.assertFalse(torch.isinf(out).any().item(), f"Inf in batch size {batch_size}")

    def test_residual_block_identity(self):
        """Verify ResidualBlock with matching in/out channels performs identity addition."""
        block = ResidualBlock(in_channels=32, out_channels=32, stride=1, dropout_rate=0.0)
        block.eval()
        x = torch.randn(2, 32, 24, 24)
        with torch.no_grad():
            out = block(x)
        self.assertEqual(out.shape, (2, 32, 24, 24))
        self.assertIsInstance(block.shortcut, torch.nn.Identity)

    def test_residual_block_projection(self):
        """Verify ResidualBlock with dimension changes uses 1x1 Conv shortcut projection."""
        block = ResidualBlock(in_channels=32, out_channels=64, stride=1, dropout_rate=0.1)
        block.eval()
        x = torch.randn(2, 32, 24, 24)
        with torch.no_grad():
            out = block(x)
        self.assertEqual(out.shape, (2, 64, 24, 24))
        self.assertIsInstance(block.shortcut, torch.nn.Sequential)

    def test_backward_pass_gradients(self):
        """Verify backward pass computes valid non-NaN gradients across all residual layers."""
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()
        inputs = torch.randn(4, 1, 48, 48)
        targets = torch.tensor([0, 1, 5, 6], dtype=torch.long)

        outputs = self.model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        self.assertFalse(torch.isnan(loss).item())
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Missing gradient for {name}")
                self.assertFalse(torch.isnan(param.grad).any().item(), f"NaN gradient in {name}")

    def test_builder_selection(self):
        """Verify build_model_from_config selects both architectures seamlessly."""
        config_base = {
            "dataset": {"channels": 1},
            "classes": {"num_classes": 7},
            "model": {"architecture": "BaselineEmotionCNN"}
        }
        config_res = {
            "dataset": {"channels": 1},
            "classes": {"num_classes": 7},
            "model": {"architecture": "ResidualEmotionCNN"}
        }
        
        m1 = build_model_from_config(config_base)
        m2 = build_model_from_config(config_res)

        self.assertIsInstance(m1, BaselineEmotionCNN)
        self.assertIsInstance(m2, ResidualEmotionCNN)

    def test_model_summary_residual(self):
        """Verify parameter counting and size calculation for ResidualEmotionCNN."""
        summary = get_model_summary(self.model, input_size=(1, 1, 48, 48))
        self.assertEqual(summary["model_class"], "ResidualEmotionCNN")
        self.assertEqual(summary["input_shape"], [1, 1, 48, 48])
        self.assertEqual(summary["output_shape"], [1, 7])
        self.assertGreater(summary["total_params"], 100_000)
        self.assertLess(summary["total_params"], 10_000_000)  # Lightweight CPU suitable


if __name__ == "__main__":
    unittest.main()
