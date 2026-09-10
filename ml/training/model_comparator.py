"""
FaceSense AI - Model Comparator and Safe Promotion Gate
Evaluates baseline and candidate models side-by-side on the untouched FER2013 test set,
computes metric diffs, verifies promotion criteria on Macro F1 and per-class thresholds,
and safely promotes winning checkpoints with complete version history preservation.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.data.dataloader import load_yaml_config, get_transforms
from ml.data.dataset import FER2013Dataset, get_image_paths_and_targets
from ml.models.builder import build_model_from_config
from ml.training.metrics import compute_evaluation_metrics, format_confusion_matrix_ascii


def evaluate_model_on_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Tuple[float, Dict[str, Any]]:
    """
    Runs model evaluation across a DataLoader and returns (loss, metrics_dict).
    """
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += float(loss.item()) * len(targets)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    avg_loss = total_loss / max(1, len(all_targets))
    metrics = compute_evaluation_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        class_names=class_names,
    )
    return avg_loss, metrics


def load_checkpoint_into_model(
    checkpoint_path: Union[str, Path],
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Constructs model from config and loads saved state dict.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path.resolve()}")

    checkpoint = torch.load(path, map_location=device)
    saved_cfg = checkpoint.get("config", config)

    model = build_model_from_config(saved_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


class ModelComparator:
    """
    Evaluates baseline vs candidate models and executes the safe promotion gate.
    
    Gate Rules:
      1. Candidate Macro F1 must be >= baseline Macro F1 + min_macro_f1_delta (default 0.0%).
      2. Candidate accuracy must not drop significantly (max allowed drop: max_accuracy_drop, default 1.0%).
      3. No individual class F1 score can degrade by more than max_class_f1_degradation (default 3.0%).
    """

    def __init__(
        self,
        config_path: Union[str, Path] = "configs/config.yaml",
        min_macro_f1_delta: float = 0.0,
        max_accuracy_drop: float = 1.0,
        max_class_f1_degradation: float = 3.0,
        device: Optional[torch.device] = None,
    ):
        self.config_path = Path(config_path)
        self.config = load_yaml_config(self.config_path)
        self.min_macro_f1_delta = float(min_macro_f1_delta)
        self.max_accuracy_drop = float(max_accuracy_drop)
        self.max_class_f1_degradation = float(max_class_f1_degradation)
        self.device = device or torch.device("cpu")
        self.class_names = self.config["classes"]["names"]

    def build_test_loader(self) -> DataLoader:
        """Constructs clean DataLoader for untouched test split."""
        root_dir = Path(__file__).resolve().parent.parent.parent
        ds_cfg = self.config["dataset"]
        test_dir = root_dir / ds_cfg["test_split_dir"]
        img_size = tuple(ds_cfg.get("image_size", [48, 48]))
        class_mapping = self.config["classes"]["mapping"]

        test_samples = get_image_paths_and_targets(test_dir, class_mapping)
        eval_transform = get_transforms(img_size=img_size, is_train=False)

        test_dataset = FER2013Dataset(
            samples=test_samples,
            class_names=self.class_names,
            transform=eval_transform,
        )

        return DataLoader(
            test_dataset,
            batch_size=int(self.config.get("dataloader", {}).get("batch_size", 64)),
            shuffle=False,
            num_workers=int(self.config.get("dataloader", {}).get("num_workers", 0)),
            pin_memory=False,
        )

    def compare_models(
        self,
        baseline_checkpoint_path: Union[str, Path],
        candidate_checkpoint_path: Union[str, Path],
        baseline_version_name: Optional[str] = None,
        candidate_version_name: Optional[str] = None,
        test_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """
        Runs rigorous test evaluation on both models and calculates diffs.
        """
        loader = test_loader or self.build_test_loader()

        # 1. Load Baseline
        base_model, base_ckpt = load_checkpoint_into_model(
            baseline_checkpoint_path, self.config, self.device
        )
        base_loss, base_metrics = evaluate_model_on_dataloader(
            base_model, loader, self.device, self.class_names
        )

        # 2. Load Candidate
        cand_model, cand_ckpt = load_checkpoint_into_model(
            candidate_checkpoint_path, self.config, self.device
        )
        cand_loss, cand_metrics = evaluate_model_on_dataloader(
            cand_model, loader, self.device, self.class_names
        )

        # 3. Model Version strings
        b_epoch = base_ckpt.get("epoch", "final")
        c_epoch = cand_ckpt.get("epoch", "candidate")
        b_name = baseline_version_name or f"ResidualEmotionCNN-epoch{b_epoch}"
        c_name = candidate_version_name or f"ResidualEmotionCNN-Candidate-epoch{c_epoch}"

        # 4. Compute differences (Candidate - Baseline)
        acc_diff = cand_metrics["accuracy"] - base_metrics["accuracy"]
        macro_f1_diff = cand_metrics["macro_f1"] - base_metrics["macro_f1"]
        weighted_f1_diff = cand_metrics["weighted_f1"] - base_metrics["weighted_f1"]
        loss_diff = cand_loss - base_loss

        per_class_diff: Dict[str, Dict[str, float]] = {}
        max_degraded_class = None
        max_degradation_val = 0.0

        for c_name_idx in self.class_names:
            b_f1 = base_metrics["per_class"][c_name_idx]["f1_score"]
            c_f1 = cand_metrics["per_class"][c_name_idx]["f1_score"]
            diff = c_f1 - b_f1
            per_class_diff[c_name_idx] = {
                "baseline_f1": b_f1,
                "candidate_f1": c_f1,
                "f1_diff": diff,
            }
            if diff < 0 and abs(diff) > max_degradation_val:
                max_degradation_val = abs(diff)
                max_degraded_class = c_name_idx

        # 5. Evaluate Promotion Gate Rules
        decision = "PROMOTE"
        rejection_reasons: List[str] = []

        if macro_f1_diff < self.min_macro_f1_delta:
            decision = "REJECT"
            rejection_reasons.append(
                f"Candidate Macro F1 ({cand_metrics['macro_f1']:.2f}%) did not exceed baseline "
                f"({base_metrics['macro_f1']:.2f}%) by required delta {self.min_macro_f1_delta:.2f}% (diff: {macro_f1_diff:+.2f}%)."
            )

        if acc_diff < -self.max_accuracy_drop:
            decision = "REJECT"
            rejection_reasons.append(
                f"Candidate overall accuracy dropped by {abs(acc_diff):.2f}%, exceeding max allowable drop of {self.max_accuracy_drop:.2f}%."
            )

        if max_degradation_val > self.max_class_f1_degradation:
            decision = "REJECT"
            rejection_reasons.append(
                f"Class '{max_degraded_class}' F1 degraded by {max_degradation_val:.2f}%, exceeding safety limit of {self.max_class_f1_degradation:.2f}%."
            )

        comparison_report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "baseline": {
                "version": b_name,
                "checkpoint_path": str(Path(baseline_checkpoint_path).resolve()),
                "test_loss": base_loss,
                "accuracy": base_metrics["accuracy"],
                "macro_f1": base_metrics["macro_f1"],
                "weighted_f1": base_metrics["weighted_f1"],
                "per_class": base_metrics["per_class"],
                "confusion_matrix": base_metrics["confusion_matrix"],
            },
            "candidate": {
                "version": c_name,
                "checkpoint_path": str(Path(candidate_checkpoint_path).resolve()),
                "test_loss": cand_loss,
                "accuracy": cand_metrics["accuracy"],
                "macro_f1": cand_metrics["macro_f1"],
                "weighted_f1": cand_metrics["weighted_f1"],
                "per_class": cand_metrics["per_class"],
                "confusion_matrix": cand_metrics["confusion_matrix"],
            },
            "differences": {
                "accuracy_diff": acc_diff,
                "macro_f1_diff": macro_f1_diff,
                "weighted_f1_diff": weighted_f1_diff,
                "loss_diff": loss_diff,
                "per_class_diff": per_class_diff,
            },
            "gate_criteria": {
                "min_macro_f1_delta": self.min_macro_f1_delta,
                "max_accuracy_drop": self.max_accuracy_drop,
                "max_class_f1_degradation": self.max_class_f1_degradation,
            },
            "decision": decision,
            "rejection_reasons": rejection_reasons,
        }

        return comparison_report

    def apply_promotion(
        self,
        comparison_report: Dict[str, Any],
        production_dir: Union[str, Path] = "ml/models/checkpoints/final",
    ) -> Dict[str, Any]:
        """
        Applies promotion if decision is PROMOTE.
        Preserves previous production checkpoint by backing it up to versioned filename before copying.
        Updates model_metadata.json with complete lineage.
        """
        prod_dir = Path(production_dir)
        prod_dir.mkdir(parents=True, exist_ok=True)
        best_model_pt = prod_dir / "best_model.pt"
        meta_file = prod_dir / "model_metadata.json"

        decision = comparison_report["decision"]
        if decision != "PROMOTE":
            return {
                "promoted": False,
                "status": "REJECTED",
                "message": "Candidate model rejected by promotion gate. Production checkpoint left untouched.",
                "production_checkpoint": str(best_model_pt.resolve()) if best_model_pt.exists() else None,
            }

        cand_ckpt_path = Path(comparison_report["candidate"]["checkpoint_path"])
        if not cand_ckpt_path.exists():
            raise FileNotFoundError(f"Candidate checkpoint to promote not found at: {cand_ckpt_path}")

        # 1. Version and preserve previous checkpoint if exists
        prev_backup_path = None
        if best_model_pt.exists():
            # Load old metadata to get old version name
            old_version_tag = "prev"
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        old_meta = json.load(f)
                        old_version_tag = f"epoch{old_meta.get('best_epoch', 'old')}_{old_meta.get('experiment_name', 'v1')}"
                except Exception:
                    pass
            ts_suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            prev_backup_path = prod_dir / f"model_backup_{old_version_tag}_{ts_suffix}.pt"
            shutil.copy2(best_model_pt, prev_backup_path)

        # 2. Copy candidate checkpoint as new best_model.pt
        shutil.copy2(cand_ckpt_path, best_model_pt)

        # 3. Create updated versioned metadata
        new_meta = {
            "model_architecture": self.config.get("model", {}).get("architecture", "ResidualEmotionCNN"),
            "dataset": "FER2013 + Feedback Dataset",
            "model_version": comparison_report["candidate"]["version"],
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "previous_backup": str(prev_backup_path.resolve()) if prev_backup_path else None,
            "test_accuracy": comparison_report["candidate"]["accuracy"],
            "test_macro_f1": comparison_report["candidate"]["macro_f1"],
            "test_weighted_f1": comparison_report["candidate"]["weighted_f1"],
            "test_loss": comparison_report["candidate"]["test_loss"],
            "checkpoint_source_path": str(cand_ckpt_path.resolve()),
            "improvement_over_baseline": {
                "macro_f1_delta": comparison_report["differences"]["macro_f1_diff"],
                "accuracy_delta": comparison_report["differences"]["accuracy_diff"],
            },
        }

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(new_meta, f, indent=2)

        return {
            "promoted": True,
            "status": "PROMOTED",
            "message": f"Successfully promoted {comparison_report['candidate']['version']} to production.",
            "production_checkpoint": str(best_model_pt.resolve()),
            "backup_checkpoint": str(prev_backup_path.resolve()) if prev_backup_path else None,
            "metadata_file": str(meta_file.resolve()),
        }
