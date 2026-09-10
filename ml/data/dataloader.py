"""
FaceSense AI - DataLoader and Transforms Pipeline
Provides augmentation pipelines, class-imbalance mitigations (class weights & WeightedRandomSampler),
and PyTorch DataLoader builders driven by config.yaml.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as T

from ml.data.dataset import (
    FER2013Dataset,
    create_train_val_split,
    get_image_paths_and_targets,
)


def load_yaml_config(config_path: Union[str, Path] = "configs/config.yaml") -> Dict[str, Any]:
    """Loads YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_transforms(
    img_size: Tuple[int, int] = (48, 48),
    is_train: bool = True,
    hflip_prob: float = 0.5,
    rotation_deg: float = 10.0,
    crop_scale: Tuple[float, float] = (0.85, 1.0),
) -> T.Compose:
    """
    Constructs PyTorch image transformations.
    
    Training:
      - RandomHorizontalFlip
      - Small RandomRotation (e.g. +-10 deg)
      - RandomResizedCrop (controlled scale e.g. 0.85-1.0 to preserve facial landmarks)
      - ToTensor (scales [0, 255] uint8 -> [0.0, 1.0] float tensor)
      - Grayscale normalization (standardized mean=0.5, std=0.5 or [0, 1] range)

    Validation / Testing:
      - Deterministic Resize to 48x48
      - ToTensor
      - Same normalization
    """
    if is_train:
        return T.Compose([
            T.RandomResizedCrop(
                size=img_size,
                scale=crop_scale,
                ratio=(0.9, 1.1),
                antialias=True,
            ),
            T.RandomHorizontalFlip(p=hflip_prob),
            T.RandomRotation(degrees=rotation_deg),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])
    else:
        return T.Compose([
            T.Resize(size=img_size, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])


def compute_class_weights(
    labels: List[int],
    num_classes: int = 7,
    mode: str = "balanced",
) -> torch.Tensor:
    """
    Computes class weights for CrossEntropyLoss to counteract severe class imbalance (e.g., disgust).
    
    Balanced mode: weight[c] = total_samples / (num_classes * count[c])
    """
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    class_counts = torch.bincount(labels_tensor, minlength=num_classes).float()
    
    # Avoid division by zero if a class has 0 samples
    class_counts = torch.clamp(class_counts, min=1.0)
    total_samples = float(len(labels))

    if mode == "balanced":
        weights = total_samples / (num_classes * class_counts)
    elif mode == "inverse":
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * num_classes
    else:
        raise ValueError(f"Unknown weight mode: {mode}")

    return weights


def create_weighted_sampler(
    labels: List[int],
    num_classes: int = 7,
) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler that samples each example inversely
    proportional to its class frequency, giving underrepresented classes
    (like 'disgust') equal representation during training mini-batches.
    """
    class_weights = compute_class_weights(labels, num_classes=num_classes, mode="balanced")
    sample_weights = [class_weights[label].item() for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def build_dataloaders(
    config_or_path: Union[str, Path, Dict[str, Any]] = "configs/config.yaml",
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Builds (train_loader, val_loader, test_loader, metadata) using the configuration.

    Returns:
        train_loader: PyTorch DataLoader for training
        val_loader: PyTorch DataLoader for validation
        test_loader: PyTorch DataLoader for testing
        metadata: Dict containing class weights, sample counts, class mappings, etc.
    """
    if isinstance(config_or_path, (str, Path)):
        config = load_yaml_config(config_or_path)
    else:
        config = config_or_path

    # Dataset configs
    train_dir_path = Path(config["dataset"]["train_split_dir"])
    test_dir_path = Path(config["dataset"]["test_split_dir"])
    if not train_dir_path.exists() and (Path.cwd().parent / train_dir_path).exists():
        train_dir = str(Path.cwd().parent / train_dir_path)
        test_dir = str(Path.cwd().parent / test_dir_path)
    else:
        train_dir = config["dataset"]["train_split_dir"]
        test_dir = config["dataset"]["test_split_dir"]

    img_size = tuple(config["dataset"]["image_size"])
    val_ratio = float(config["dataset"].get("val_split_ratio", 0.15))
    seed = int(config["dataset"].get("seed", 42))

    # Classes
    class_names = config["classes"]["names"]
    class_mapping = config["classes"]["mapping"]
    num_classes = config["classes"]["num_classes"]

    # Augmentation configs
    aug_cfg = config.get("augmentation", {})
    hflip_prob = float(aug_cfg.get("horizontal_flip_prob", 0.5))
    rotation_deg = float(aug_cfg.get("rotation_degrees", 10))
    crop_scale = tuple(aug_cfg.get("crop_scale", [0.85, 1.0]))

    # DataLoader configs
    dl_cfg = config.get("dataloader", {})
    batch_size = int(dl_cfg.get("batch_size", config.get("training", {}).get("batch_size", 64)))
    num_workers = int(dl_cfg.get("num_workers", 0))
    pin_memory = bool(dl_cfg.get("pin_memory", False))  # False on CPU-only machines
    # persistent_workers requires num_workers > 0
    persistent_workers = bool(dl_cfg.get("persistent_workers", False)) and num_workers > 0
    # prefetch_factor only valid when num_workers > 0; yaml null becomes Python None
    _pf_raw = dl_cfg.get("prefetch_factor", 2)
    prefetch_factor = int(_pf_raw) if (num_workers > 0 and _pf_raw is not None) else None
    use_weighted_sampler = bool(dl_cfg.get("use_weighted_sampler", True))

    # Create stratified train/val split
    train_samples, val_samples = create_train_val_split(
        train_dir=train_dir,
        class_mapping=class_mapping,
        val_ratio=val_ratio,
        seed=seed,
    )

    # Load test samples
    test_samples = get_image_paths_and_targets(
        data_dir=test_dir,
        class_mapping=class_mapping,
    )

    # Prepare Transforms
    train_transform = get_transforms(
        img_size=img_size,
        is_train=True,
        hflip_prob=hflip_prob,
        rotation_deg=rotation_deg,
        crop_scale=crop_scale,
    )
    eval_transform = get_transforms(
        img_size=img_size,
        is_train=False,
    )

    # Construct Datasets
    train_dataset = FER2013Dataset(train_samples, class_names=class_names, transform=train_transform)
    val_dataset = FER2013Dataset(val_samples, class_names=class_names, transform=eval_transform)
    test_dataset = FER2013Dataset(test_samples, class_names=class_names, transform=eval_transform)

    # Class Imbalance Handling
    train_labels = train_dataset.get_labels()
    class_weights = compute_class_weights(train_labels, num_classes=num_classes, mode="balanced")

    # Shared DataLoader kwargs
    shared_dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    # Training Sampler
    if use_weighted_sampler:
        train_sampler = create_weighted_sampler(train_labels, num_classes=num_classes)
        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            **shared_dl_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **shared_dl_kwargs,
        )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **shared_dl_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **shared_dl_kwargs,
    )

    metadata = {
        "num_classes": num_classes,
        "class_names": class_names,
        "class_mapping": class_mapping,
        "class_weights": class_weights,
        "train_count": len(train_dataset),
        "val_count": len(val_dataset),
        "test_count": len(test_dataset),
        "train_class_counts": train_dataset.get_class_counts(),
        "val_class_counts": val_dataset.get_class_counts(),
        "test_class_counts": test_dataset.get_class_counts(),
    }

    return train_loader, val_loader, test_loader, metadata
