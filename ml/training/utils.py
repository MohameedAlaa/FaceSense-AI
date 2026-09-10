"""
FaceSense AI - Training Utilities
Provides seed management, device detection, metric tracking, and early stopping.
"""

import os
import random
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> Tuple[torch.device, Dict[str, Any]]:
    """
    Detects and returns the compute device along with hardware metadata.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        info = {
            "device": "cuda",
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
        }
    else:
        device = torch.device("cpu")
        info = {
            "device": "cpu",
            "cpu_threads": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
        }
    return device, info


class EarlyStopping:
    """
    Early stops the training if validation score doesn't improve after a given patience.
    Saves the best model weights.
    """

    def __init__(
        self,
        patience: int = 10,
        mode: str = "max",
        delta: float = 1e-4,
    ):
        """
        Args:
            patience: Number of epochs to wait without improvement before stopping.
            mode: 'max' for metrics like Macro F1 / Accuracy, 'min' for Loss.
            delta: Minimum change in monitored metric to qualify as improvement.
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_epoch = 0

    def step(self, score: float, epoch: int) -> bool:
        """
        Updates early stopping state.
        Returns:
            True if score is the new best, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return True

        if self.mode == "max":
            improved = score > (self.best_score + self.delta)
        elif self.mode == "min":
            improved = score < (self.best_score - self.delta)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class MetricTracker:
    """Computes and stores running averages of loss and accuracy during an epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.correct = 0
        self.total_samples = 0

    def update(self, loss: float, preds: torch.Tensor, targets: torch.Tensor):
        batch_size = targets.size(0)
        self.total_loss += loss * batch_size
        self.correct += (preds == targets).sum().item()
        self.total_samples += batch_size

    @property
    def avg_loss(self) -> float:
        return self.total_loss / max(1, self.total_samples)

    @property
    def accuracy(self) -> float:
        return (self.correct / max(1, self.total_samples)) * 100.0
