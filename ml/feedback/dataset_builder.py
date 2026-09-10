"""
FaceSense AI - Feedback Dataset Builder
Constructs future retraining datasets from verified feedback records while strictly preserving
the original FER2013 dataset. Preprocesses and formats collected face crops into standard 48x48
grayscale images organized in PyTorch/ImageFolder compatible directory structures or manifest files.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from ml.feedback.collector import FeedbackCollector, FeedbackState, DEFAULT_VALID_EMOTIONS


class FeedbackDatasetBuilder:
    """
    Builds structured training/validation dataset partitions from logged feedback samples.
    Features:
      - Filters valid feedback (extracts ground truth from 'correct' or 'incorrect' with corrected_emotion)
      - Resolves and standardizes images (crops face bounding box if full frame, resizes to 48x48 grayscale)
      - Exports to standard directory tree (dataset_dir/<split>/<class_name>/<filename>.jpg)
      - Preserves original FER2013 by generating merged manifests or standalone feedback datasets
      - Produces detailed dataset manifest metadata
    """

    def __init__(
        self,
        feedback_collector: Optional[FeedbackCollector] = None,
        feedback_dir: Union[str, Path] = "outputs/feedback",
        img_size: Tuple[int, int] = (48, 48),
        valid_emotions: Optional[Tuple[str, ...]] = None,
    ):
        self.feedback_dir = Path(feedback_dir)
        self.collector = feedback_collector or FeedbackCollector(feedback_dir=self.feedback_dir)
        self.img_size = tuple(img_size)
        self.valid_emotions = valid_emotions or DEFAULT_VALID_EMOTIONS

    def extract_labeled_samples(
        self,
        include_correct: bool = True,
        include_incorrect: bool = True,
        include_uncertain_with_correction: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Filters raw feedback records into validated training candidates with determined ground truth labels.
        """
        records = self.collector.load_feedback_records()
        usable_samples = []

        for r in records:
            state = r.get("state")
            pred = r.get("predicted_emotion")
            corr = r.get("corrected_emotion")
            img_p = Path(r.get("image_path", ""))

            if not img_p.exists():
                continue

            ground_truth_label: Optional[str] = None

            if state == FeedbackState.CORRECT.value and include_correct:
                # Prediction was confirmed correct
                if pred in self.valid_emotions:
                    ground_truth_label = pred
            elif state == FeedbackState.INCORRECT.value and include_incorrect:
                # User supplied a corrected ground truth label
                if corr and corr in self.valid_emotions:
                    ground_truth_label = corr
            elif state == FeedbackState.UNCERTAIN.value and include_uncertain_with_correction:
                # Uncertain prediction where user provided the correct label
                if corr and corr in self.valid_emotions:
                    ground_truth_label = corr

            if ground_truth_label is not None:
                usable_samples.append({
                    "feedback_id": r.get("feedback_id"),
                    "image_path": str(img_p.resolve()),
                    "label": ground_truth_label,
                    "bounding_box": r.get("bounding_box"),
                    "confidence": r.get("confidence"),
                    "model_version": r.get("model_version"),
                    "state": state,
                    "timestamp": r.get("timestamp"),
                })

        return usable_samples

    def process_and_save_face(
        self,
        image_path: Union[str, Path],
        dest_path: Union[str, Path],
        bounding_box: Optional[List[int]] = None,
    ) -> None:
        """
        Loads the image, crops the bounding box if provided, converts to 48x48 grayscale, and saves.
        """
        img_p = Path(image_path)
        dest_p = Path(dest_path)
        dest_p.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(img_p) as img:
            gray_img = img.convert("L")

            if bounding_box is not None and len(bounding_box) == 4:
                x, y, w, h = bounding_box
                # Validate coordinates against image bounds
                im_w, im_h = gray_img.size
                x1 = max(0, min(x, im_w - 1))
                y1 = max(0, min(y, im_h - 1))
                x2 = max(x1 + 1, min(x + w, im_w))
                y2 = max(y1 + 1, min(y + h, im_h))
                cropped = gray_img.crop((x1, y1, x2, y2))
            else:
                cropped = gray_img

            final_face = cropped.resize(self.img_size, Image.Resampling.BILINEAR)
            final_face.save(dest_p, format="JPEG", quality=95)

    def build_feedback_dataset(
        self,
        output_dataset_dir: Union[str, Path] = "outputs/feedback/feedback_dataset",
        val_split_ratio: float = 0.15,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Compiles collected feedback records into a train/val ImageFolder compatible dataset.

        Directory Structure:
          <output_dataset_dir>/
            train/
              angry/
              happy/
              ...
            val/
              angry/
              ...
            manifest.json

        Returns:
            Dictionary with build statistics and dataset manifest.
        """
        out_dir = Path(output_dataset_dir)
        train_dir = out_dir / "train"
        val_dir = out_dir / "val"

        # Clean/create target dirs
        for emo in self.valid_emotions:
            (train_dir / emo).mkdir(parents=True, exist_ok=True)
            (val_dir / emo).mkdir(parents=True, exist_ok=True)

        samples = self.extract_labeled_samples()
        if not samples:
            manifest = {
                "total_samples": 0,
                "train_samples": 0,
                "val_samples": 0,
                "class_counts_train": {emo: 0 for emo in self.valid_emotions},
                "class_counts_val": {emo: 0 for emo in self.valid_emotions},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "output_dir": str(out_dir.resolve()),
            }
            with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            return manifest

        # Group samples by class for stratified partitioning
        by_class: Dict[str, List[Dict[str, Any]]] = {emo: [] for emo in self.valid_emotions}
        for s in samples:
            by_class[s["label"]].append(s)

        rng = np.random.default_rng(seed)
        train_count = 0
        val_count = 0
        class_counts_train = {emo: 0 for emo in self.valid_emotions}
        class_counts_val = {emo: 0 for emo in self.valid_emotions}

        for emo, class_samples in by_class.items():
            if not class_samples:
                continue
            indices = np.arange(len(class_samples))
            rng.shuffle(indices)

            n_val = int(round(len(class_samples) * val_split_ratio))
            val_idx = set(indices[:n_val])

            for idx, item in enumerate(class_samples):
                is_val = idx in val_idx
                target_split = "val" if is_val else "train"
                dest_filename = f"{item['feedback_id']}_{emo}.jpg"
                dest_file = (val_dir if is_val else train_dir) / emo / dest_filename

                self.process_and_save_face(
                    image_path=item["image_path"],
                    dest_path=dest_file,
                    bounding_box=item.get("bounding_box"),
                )

                if is_val:
                    val_count += 1
                    class_counts_val[emo] += 1
                else:
                    train_count += 1
                    class_counts_train[emo] += 1

        manifest = {
            "total_samples": len(samples),
            "train_samples": train_count,
            "val_samples": val_count,
            "class_counts_train": class_counts_train,
            "class_counts_val": class_counts_val,
            "image_size": list(self.img_size),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(out_dir.resolve()),
        }

        with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest
