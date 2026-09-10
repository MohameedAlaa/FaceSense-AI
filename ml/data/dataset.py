"""
FaceSense AI - Dataset Implementation
Custom PyTorch Dataset supporting grayscale 48x48 emotion recognition,
reproducible stratified train/val splits, and metadata queries.
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class FER2013Dataset(Dataset):
    """
    PyTorch Dataset for FER2013 facial expression recognition.
    Loads 48x48 single-channel grayscale images.
    """

    def __init__(
        self,
        samples: List[Tuple[Union[str, Path], int]],
        class_names: List[str],
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            samples: List of (image_path, class_index) tuples.
            class_names: List of emotion class names ordered by index.
            transform: Optional torchvision transforms / callable to apply.
        """
        self.samples = samples
        self.class_names = class_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, target = self.samples[idx]
        
        # Load image in grayscale ('L')
        with Image.open(img_path) as img:
            image = img.convert("L")

        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default fallback: convert PIL to Tensor
            import torchvision.transforms.functional as F
            image = F.to_tensor(image)

        return image, target

    def get_labels(self) -> List[int]:
        """Returns all sample target labels (useful for Sampler weight calculation)."""
        return [target for _, target in self.samples]

    def get_class_counts(self) -> Dict[str, int]:
        """Returns dictionary mapping class names to sample counts in this dataset."""
        counts = {name: 0 for name in self.class_names}
        for _, target in self.samples:
            name = self.class_names[target]
            counts[name] += 1
        return counts


def get_image_paths_and_targets(
    data_dir: Union[str, Path],
    class_mapping: Dict[str, int],
) -> List[Tuple[Path, int]]:
    """
    Scans a directory structured as <data_dir>/<class_name>/*.jpg
    and returns a list of (Path, target_index) tuples.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory '{root}' does not exist.")

    samples: List[Tuple[Path, int]] = []
    for class_name, target_idx in class_mapping.items():
        class_folder = root / class_name
        if not class_folder.exists() or not class_folder.is_dir():
            continue
        
        for file_path in sorted(class_folder.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                samples.append((file_path, target_idx))

    return samples


def create_train_val_split(
    train_dir: Union[str, Path],
    class_mapping: Dict[str, int],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Splits the training directory samples into train and validation sets
    using reproducible stratified sampling.

    Args:
        train_dir: Path to raw train directory (e.g. Data(FER2013)/train).
        class_mapping: Dict mapping class names to integer labels.
        val_ratio: Proportion of training data for validation (default 0.15).
        seed: Random seed for reproducibility.

    Returns:
        (train_samples, val_samples)
    """
    all_train_samples = get_image_paths_and_targets(train_dir, class_mapping)
    if not all_train_samples:
        raise ValueError(f"No samples found in '{train_dir}'.")

    targets = [sample[1] for sample in all_train_samples]

    train_samples, val_samples = train_test_split(
        all_train_samples,
        test_size=val_ratio,
        random_state=seed,
        stratify=targets,
        shuffle=True,
    )

    return train_samples, val_samples
