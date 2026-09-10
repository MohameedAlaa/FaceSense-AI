"""
FaceSense AI - Continuous Retraining Dataset Pipeline
Combines the base FER2013 dataset with validated feedback samples according to
a controlled mixing ratio to prevent feedback data from overwhelming base distributions.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from ml.data.dataset import (
    FER2013Dataset,
    create_train_val_split,
    get_image_paths_and_targets,
)
from ml.data.dataloader import (
    get_transforms,
    compute_class_weights,
    create_weighted_sampler,
    load_yaml_config,
)
from ml.feedback.dataset_builder import FeedbackDatasetBuilder


class MixedEmotionDataset(Dataset):
    """
    Blends base dataset samples (e.g. FER2013) with collected user feedback samples.
    
    Supports:
      - Direct combination of samples
      - Controlled feedback weight / oversampling ratio
      - Source tracking (base vs feedback)
    """

    def __init__(
        self,
        base_samples: List[Tuple[Union[str, Path], int]],
        feedback_samples: Optional[List[Tuple[Union[str, Path], int]]] = None,
        class_names: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        feedback_weight: float = 1.0,
        seed: int = 42,
    ):
        """
        Args:
            base_samples: List of (image_path, target_idx) from FER2013.
            feedback_samples: List of (image_path, target_idx) from feedback store.
            class_names: List of class names.
            transform: Transform pipeline.
            feedback_weight: Multiplier/weight for feedback samples relative to base.
            seed: Random seed for sampling/replication.
        """
        self.class_names = class_names or [
            "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
        ]
        self.transform = transform
        self.base_samples = [(Path(p), int(t)) for p, t in base_samples]
        self.feedback_samples = (
            [(Path(p), int(t)) for p, t in feedback_samples] if feedback_samples else []
        )

        self.combined_samples: List[Tuple[Path, int, str]] = []  # (path, target, source)

        # 1. Add base samples
        for p, t in self.base_samples:
            self.combined_samples.append((p, t, "base"))

        # 2. Add feedback samples with weight/replication control
        if self.feedback_samples and feedback_weight > 0:
            if feedback_weight == 1.0:
                for p, t in self.feedback_samples:
                    self.combined_samples.append((p, t, "feedback"))
            elif feedback_weight > 1.0:
                # Oversample feedback items
                repeat_count = int(np.floor(feedback_weight))
                remainder = feedback_weight - repeat_count
                for _ in range(repeat_count):
                    for p, t in self.feedback_samples:
                        self.combined_samples.append((p, t, "feedback"))
                if remainder > 0:
                    rng = np.random.default_rng(seed)
                    k = int(round(len(self.feedback_samples) * remainder))
                    if k > 0:
                        chosen_indices = rng.choice(len(self.feedback_samples), size=k, replace=False)
                        for idx in chosen_indices:
                            p, t = self.feedback_samples[idx]
                            self.combined_samples.append((p, t, "feedback"))
            else:
                # Subsample feedback items (feedback_weight < 1.0)
                rng = np.random.default_rng(seed)
                k = max(1, int(round(len(self.feedback_samples) * feedback_weight)))
                chosen_indices = rng.choice(len(self.feedback_samples), size=k, replace=False)
                for idx in chosen_indices:
                    p, t = self.feedback_samples[idx]
                    self.combined_samples.append((p, t, "feedback"))

    def __len__(self) -> int:
        return len(self.combined_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, target, _ = self.combined_samples[idx]
        with Image.open(img_path) as img:
            image = img.convert("L")

        if self.transform is not None:
            image = self.transform(image)
        else:
            import torchvision.transforms.functional as F
            image = F.to_tensor(image)

        return image, target

    def get_labels(self) -> List[int]:
        return [target for _, target, _ in self.combined_samples]

    def get_class_counts(self) -> Dict[str, int]:
        counts = {name: 0 for name in self.class_names}
        for _, target, _ in self.combined_samples:
            name = self.class_names[target]
            counts[name] += 1
        return counts

    def get_source_distribution(self) -> Dict[str, int]:
        counts = {"base": 0, "feedback": 0}
        for _, _, src in self.combined_samples:
            counts[src] = counts.get(src, 0) + 1
        return counts


def build_retraining_dataloaders(
    config: Dict[str, Any],
    feedback_dir: Optional[Union[str, Path]] = None,
    feedback_sampling_ratio: Optional[float] = None,
    feedback_weight: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Constructs DataLoaders blending the FER2013 training set with feedback dataset samples
    while strictly evaluating on the untouched FER2013 test set and validation set.
    """
    root_dir = Path(__file__).resolve().parent.parent.parent
    ds_cfg = config["dataset"]
    tr_cfg = config.get("training", {})
    retrain_cfg = config.get("retraining", {})
    dl_cfg = config.get("dataloader", {})
    aug_cfg = config.get("augmentation", {})
    class_names = config["classes"]["names"]
    class_mapping = config["classes"]["mapping"]

    train_dir = root_dir / ds_cfg["train_split_dir"]
    test_dir = root_dir / ds_cfg["test_split_dir"]
    img_size = tuple(ds_cfg.get("image_size", [48, 48]))
    val_ratio = float(ds_cfg.get("val_split_ratio", 0.15))
    seed = int(ds_cfg.get("seed", 42))

    # 1. Standard base train/val split from FER2013
    base_train_samples, val_samples = create_train_val_split(
        train_dir=train_dir,
        class_mapping=class_mapping,
        val_ratio=val_ratio,
        seed=seed,
    )

    # 2. Extract feedback samples if available
    fb_samples: List[Tuple[Path, int]] = []
    fb_root = Path(feedback_dir) if feedback_dir else root_dir / config.get("paths", {}).get("outputs_feedback_dir", "outputs/feedback")
    fb_dataset_dir = fb_root / "feedback_dataset"

    # Check if feedback dataset is already built or if we should parse raw feedback records
    if (fb_dataset_dir / "train").exists():
        fb_samples = get_image_paths_and_targets(fb_dataset_dir / "train", class_mapping)
    elif fb_root.exists():
        builder = FeedbackDatasetBuilder(feedback_dir=fb_root, img_size=img_size)
        raw_labeled = builder.extract_labeled_samples()
        for item in raw_labeled:
            lbl = item["label"]
            if lbl in class_mapping:
                fb_samples.append((Path(item["image_path"]), class_mapping[lbl]))

    # Controlled feedback ratio / weight
    effective_fb_weight = (
        feedback_weight
        if feedback_weight is not None
        else float(retrain_cfg.get("feedback_weight", 1.0))
    )

    # Max feedback ratio cap to ensure feedback doesn't overwhelm FER2013
    max_fb_ratio = (
        feedback_sampling_ratio
        if feedback_sampling_ratio is not None
        else float(retrain_cfg.get("max_feedback_ratio", 0.20))
    )

    # If feedback exceeds max_feedback_ratio of base, downsample
    if fb_samples and len(base_train_samples) > 0:
        max_allowed_fb = int(round(len(base_train_samples) * max_fb_ratio))
        if len(fb_samples) > max_allowed_fb:
            rng = np.random.default_rng(seed)
            chosen_idx = rng.choice(len(fb_samples), size=max_allowed_fb, replace=False)
            fb_samples = [fb_samples[i] for i in chosen_idx]

    # 3. Build Transforms
    train_transform = get_transforms(
        img_size=img_size,
        is_train=True,
        hflip_prob=float(aug_cfg.get("horizontal_flip_prob", 0.5)),
        rotation_deg=float(aug_cfg.get("rotation_degrees", 10)),
        crop_scale=tuple(aug_cfg.get("crop_scale", [0.85, 1.0])),
    )
    eval_transform = get_transforms(img_size=img_size, is_train=False)

    # 4. Construct Datasets
    mixed_train_dataset = MixedEmotionDataset(
        base_samples=base_train_samples,
        feedback_samples=fb_samples,
        class_names=class_names,
        transform=train_transform,
        feedback_weight=effective_fb_weight,
        seed=seed,
    )

    val_dataset = FER2013Dataset(
        samples=val_samples,
        class_names=class_names,
        transform=eval_transform,
    )

    test_samples = get_image_paths_and_targets(test_dir, class_mapping)
    test_dataset = FER2013Dataset(
        samples=test_samples,
        class_names=class_names,
        transform=eval_transform,
    )

    # 5. Build Sampler for training
    train_labels = mixed_train_dataset.get_labels()
    use_weighted_sampler = dl_cfg.get("use_weighted_sampler", True)
    if use_weighted_sampler:
        sampler = create_weighted_sampler(train_labels, num_classes=len(class_names))
        shuffle = False
    else:
        sampler = None
        shuffle = True

    batch_size = int(dl_cfg.get("batch_size", 64))
    num_workers = int(dl_cfg.get("num_workers", 0))
    pin_memory = bool(dl_cfg.get("pin_memory", False))

    train_loader = DataLoader(
        mixed_train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    metadata = {
        "base_train_count": len(base_train_samples),
        "feedback_samples_used": len(fb_samples),
        "total_train_count": len(mixed_train_dataset),
        "val_count": len(val_dataset),
        "test_count": len(test_dataset),
        "class_names": class_names,
        "source_distribution": mixed_train_dataset.get_source_distribution(),
        "class_counts_train": mixed_train_dataset.get_class_counts(),
        "effective_feedback_weight": effective_fb_weight,
    }

    return train_loader, val_loader, test_loader, metadata
