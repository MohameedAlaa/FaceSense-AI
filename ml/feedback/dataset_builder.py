"""
FaceSense AI - Feedback Dataset Builder
Constructs future retraining datasets from verified feedback records while strictly preserving
the original FER2013 dataset. Preprocesses and formats collected face crops into standard 48x48
grayscale images organized in PyTorch/ImageFolder compatible directory structures or manifest files.
Features structured diagnostic auditing and explicit exclusion tracking.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from ml.feedback.collector import FeedbackCollector, FeedbackState, DEFAULT_VALID_EMOTIONS


class FeedbackDatasetBuilder:
    """
    Builds structured training/validation dataset partitions from logged feedback samples.
    Features:
      - Filters valid feedback (extracts ground truth from 'correct' or 'incorrect' with corrected_emotion)
      - Resolves and standardizes images (crops face bounding box if full frame, resizes to 48x48 grayscale)
      - Diagnostic auditing: tracks exactly why records are included or excluded with explicit reason codes
      - Robust path resolution: supports absolute paths, project-relative paths, and feedback-relative paths
      - Exports to standard directory tree (dataset_dir/<split>/<class_name>/<filename>.jpg)
      - Preserves original FER2013 by generating merged manifests or standalone feedback datasets
      - Produces detailed dataset manifest metadata with full audit breakdown
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
        self.last_audit: Optional[Dict[str, Any]] = None

    def resolve_image_path(self, raw_path: Union[str, Path]) -> Optional[Path]:
        """
        Resolves an image path, testing absolute path, project-relative, and feedback-relative locations.
        """
        if not raw_path:
            return None

        p = Path(raw_path)
        # 1. Direct path
        if p.exists() and p.is_file():
            return p.resolve()

        # 2. Relative to PROJECT_ROOT
        p_root = PROJECT_ROOT / p
        if p_root.exists() and p_root.is_file():
            return p_root.resolve()

        # 3. Relative to feedback_dir
        p_fb = self.feedback_dir / p
        if p_fb.exists() and p_fb.is_file():
            return p_fb.resolve()

        # 4. Check in feedback_dir/images/<filename>
        p_fb_img = self.feedback_dir / "images" / p.name
        if p_fb_img.exists() and p_fb_img.is_file():
            return p_fb_img.resolve()

        return None

    def audit_feedback_records(
        self,
        include_correct: bool = True,
        include_incorrect: bool = True,
        include_uncertain_with_correction: bool = True,
        deduplicate_by_id: bool = True,
    ) -> Dict[str, Any]:
        """
        Thoroughly audits all logged feedback records, classifying each as usable or excluded
        with an explicit, standardized reason code.

        Returns:
            Dictionary containing:
              - total_records: Total count of records evaluated
              - usable_count: Count of usable records
              - excluded_count: Count of excluded records
              - usable_samples: List of usable sample dictionaries
              - excluded_records: List of excluded record dictionaries with reason
              - exclusion_reasons: Dict mapping reason codes to occurrence counts
              - diagnostic_report: Formatted list of table rows for display
        """
        records = self.collector.load_feedback_records()
        usable_samples: List[Dict[str, Any]] = []
        excluded_records: List[Dict[str, Any]] = []
        exclusion_reasons: Dict[str, int] = {}
        diagnostic_report: List[Dict[str, Any]] = []

        seen_feedback_ids: Set[str] = set()

        for idx, r in enumerate(records):
            f_id = r.get("feedback_id", f"record_{idx+1}")
            state = r.get("state")
            pred = r.get("predicted_emotion")
            corr = r.get("corrected_emotion")
            raw_img_p = r.get("image_path", "")

            # 1. Check duplicate ID
            if deduplicate_by_id and f_id in seen_feedback_ids:
                reason = "duplicate_feedback_id"
                exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
                excluded_records.append({
                    "feedback_id": f_id,
                    "state": state,
                    "predicted_emotion": pred,
                    "corrected_emotion": corr,
                    "reason": reason,
                    "image_path": raw_img_p,
                })
                diagnostic_report.append({
                    "index": idx + 1,
                    "feedback_id": f_id,
                    "state": state,
                    "predicted": pred,
                    "corrected": corr,
                    "included": False,
                    "reason": "Duplicate record identifier",
                })
                continue

            seen_feedback_ids.add(f_id)

            # 2. Resolve image path
            resolved_p = self.resolve_image_path(raw_img_p)
            if resolved_p is None:
                reason = "missing_image_file"
                exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
                excluded_records.append({
                    "feedback_id": f_id,
                    "state": state,
                    "predicted_emotion": pred,
                    "corrected_emotion": corr,
                    "reason": reason,
                    "image_path": raw_img_p,
                })
                diagnostic_report.append({
                    "index": idx + 1,
                    "feedback_id": f_id,
                    "state": state,
                    "predicted": pred,
                    "corrected": corr,
                    "included": False,
                    "reason": f"Missing image file on disk: {raw_img_p}",
                })
                continue

            # 3. Validate image integrity
            try:
                with Image.open(resolved_p) as img_test:
                    img_test.verify()
            except Exception as e:
                reason = "corrupted_image"
                exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
                excluded_records.append({
                    "feedback_id": f_id,
                    "state": state,
                    "predicted_emotion": pred,
                    "corrected_emotion": corr,
                    "reason": reason,
                    "image_path": str(resolved_p),
                })
                diagnostic_report.append({
                    "index": idx + 1,
                    "feedback_id": f_id,
                    "state": state,
                    "predicted": pred,
                    "corrected": corr,
                    "included": False,
                    "reason": f"Corrupted image file: {e}",
                })
                continue

            # 4. State & Ground Truth Label Validation
            ground_truth_label: Optional[str] = None
            exclusion_reason: Optional[str] = None
            display_reason: Optional[str] = None

            if state == FeedbackState.CORRECT.value:
                if not include_correct:
                    exclusion_reason = "correct_state_disabled"
                    display_reason = "Correct state excluded by configuration"
                elif pred in self.valid_emotions:
                    ground_truth_label = pred
                else:
                    exclusion_reason = "invalid_predicted_emotion"
                    display_reason = f"Predicted emotion '{pred}' not in valid classes"

            elif state == FeedbackState.INCORRECT.value:
                if not include_incorrect:
                    exclusion_reason = "incorrect_state_disabled"
                    display_reason = "Incorrect state excluded by configuration"
                elif not corr:
                    exclusion_reason = "missing_corrected_label"
                    display_reason = "Incorrect state requires verified corrected_emotion"
                elif corr in self.valid_emotions:
                    ground_truth_label = corr
                else:
                    exclusion_reason = "invalid_corrected_emotion"
                    display_reason = f"Corrected emotion '{corr}' not in valid classes"

            elif state == FeedbackState.UNCERTAIN.value:
                if corr and corr in self.valid_emotions and include_uncertain_with_correction:
                    ground_truth_label = corr
                else:
                    exclusion_reason = "uncertain_without_correction"
                    display_reason = "Uncertain prediction without verified ground-truth correction"

            else:
                exclusion_reason = "unsupported_state"
                display_reason = f"Unsupported feedback state '{state}'"

            # 5. Record classification
            if ground_truth_label is not None:
                usable_samples.append({
                    "feedback_id": f_id,
                    "image_path": str(resolved_p),
                    "label": ground_truth_label,
                    "bounding_box": r.get("bounding_box"),
                    "confidence": r.get("confidence"),
                    "model_version": r.get("model_version"),
                    "state": state,
                    "timestamp": r.get("timestamp"),
                })
                diagnostic_report.append({
                    "index": idx + 1,
                    "feedback_id": f_id,
                    "state": state,
                    "predicted": pred,
                    "corrected": corr,
                    "included": True,
                    "reason": f"Valid ground truth label: {ground_truth_label}",
                })
            else:
                code = exclusion_reason or "unknown_exclusion"
                exclusion_reasons[code] = exclusion_reasons.get(code, 0) + 1
                excluded_records.append({
                    "feedback_id": f_id,
                    "state": state,
                    "predicted_emotion": pred,
                    "corrected_emotion": corr,
                    "reason": code,
                    "image_path": str(resolved_p),
                })
                diagnostic_report.append({
                    "index": idx + 1,
                    "feedback_id": f_id,
                    "state": state,
                    "predicted": pred,
                    "corrected": corr,
                    "included": False,
                    "reason": display_reason or code,
                })

        audit_result = {
            "total_records": len(records),
            "usable_count": len(usable_samples),
            "excluded_count": len(excluded_records),
            "usable_samples": usable_samples,
            "excluded_records": excluded_records,
            "exclusion_reasons": exclusion_reasons,
            "diagnostic_report": diagnostic_report,
        }
        self.last_audit = audit_result
        return audit_result

    def extract_labeled_samples(
        self,
        include_correct: bool = True,
        include_incorrect: bool = True,
        include_uncertain_with_correction: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Filters raw feedback records into validated training candidates with determined ground truth labels.
        Uses audit_feedback_records under the hood for rigorous validation.
        """
        audit = self.audit_feedback_records(
            include_correct=include_correct,
            include_incorrect=include_incorrect,
            include_uncertain_with_correction=include_uncertain_with_correction,
        )
        return audit["usable_samples"]

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
        Saves a comprehensive manifest with training/validation splits, class counts,
        total records evaluated, excluded records, and detailed exclusion reasons.

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
            Dictionary with build statistics, dataset manifest, and audit details.
        """
        out_dir = Path(output_dataset_dir)
        train_dir = out_dir / "train"
        val_dir = out_dir / "val"

        # Clean/create target dirs
        for emo in self.valid_emotions:
            (train_dir / emo).mkdir(parents=True, exist_ok=True)
            (val_dir / emo).mkdir(parents=True, exist_ok=True)

        audit = self.audit_feedback_records()
        samples = audit["usable_samples"]

        if not samples:
            manifest = {
                "total_records_evaluated": audit["total_records"],
                "total_samples": 0,
                "usable_samples": 0,
                "excluded_samples": audit["excluded_count"],
                "exclusion_reasons": audit["exclusion_reasons"],
                "excluded_records_detail": audit["excluded_records"],
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
            "total_records_evaluated": audit["total_records"],
            "total_samples": len(samples),
            "usable_samples": len(samples),
            "excluded_samples": audit["excluded_count"],
            "exclusion_reasons": audit["exclusion_reasons"],
            "excluded_records_detail": audit["excluded_records"],
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
