"""
FaceSense AI - Retraining Orchestrator
Executes the continuous retraining pipeline:
  1. Compiles and mixes FER2013 base train set with feedback dataset.
  2. Trains a candidate model in an isolated timestamped experiment directory under outputs/experiments/.
  3. Evaluates both the production baseline model and the candidate on the untouched FER2013 test set.
  4. Runs the promotion gate comparison.
  5. If promoted: updates production checkpoint safely with backup preservation.
     If rejected: leaves production checkpoint intact and outputs the rejection audit.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple, Union
import torch

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.dataloader import load_yaml_config
from ml.models.builder import build_model_from_config
from ml.training.retrain_dataset import build_retraining_dataloaders
from ml.training.trainer import EmotionTrainer
from ml.training.model_comparator import ModelComparator
from ml.training.utils import set_seed, get_device
from ml.training.metrics import format_confusion_matrix_ascii


def run_retraining_pipeline(
    config_path: Union[str, Path] = "configs/config.yaml",
    max_epochs: Optional[int] = None,
    feedback_dir: Optional[Union[str, Path]] = None,
    feedback_weight: Optional[float] = None,
    feedback_ratio: Optional[float] = None,
    min_macro_f1_delta: Optional[float] = None,
    auto_promote: bool = True,
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Orchestrates candidate retraining, model comparison, and safe promotion.
    """
    config = load_yaml_config(config_path)
    retrain_cfg = config.get("retraining", {})
    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)

    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    exp_name = experiment_name or f"retrain_{ts_str}"

    # Isolated experiment directory under outputs/experiments/
    exp_dir = PROJECT_ROOT / "outputs" / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_checkpoints_dir = exp_dir / "checkpoints"
    exp_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(f"FaceSense AI - Continuous Retraining & Promotion Pipeline: {exp_name}")
    print("=" * 75)

    # 1. Build Mixed DataLoaders
    print("\n[1/5] Building Mixed DataLoaders (Base FER2013 + Feedback)...")
    train_loader, val_loader, test_loader, meta = build_retraining_dataloaders(
        config=config,
        feedback_dir=feedback_dir,
        feedback_sampling_ratio=feedback_ratio,
        feedback_weight=feedback_weight,
    )
    print(f"  Base Training Samples:     {meta['base_train_count']}")
    print(f"  Feedback Samples Included: {meta['feedback_samples_used']}")
    print(f"  Total Training Samples:    {meta['total_train_count']}")
    print(f"  Validation Samples:        {meta['val_count']}")
    print(f"  Test Samples (Untouched):  {meta['test_count']}")

    # 2. Build Candidate Model
    print("\n[2/5] Initializing Candidate Model...")
    model = build_model_from_config(config)

    # Override checkpoint saving paths inside trainer for this experiment
    cand_best_ckpt = exp_checkpoints_dir / "candidate_best_model.pt"

    trainer = EmotionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        class_names=meta["class_names"],
    )
    trainer.checkpoints_dir = exp_checkpoints_dir
    trainer.best_checkpoint_path = cand_best_ckpt
    trainer.metrics_dir = exp_dir / "metrics"
    trainer.metrics_dir.mkdir(parents=True, exist_ok=True)

    # Determine epochs to run
    epochs_to_run = (
        max_epochs
        if max_epochs is not None
        else int(retrain_cfg.get("candidate_epochs", config.get("training", {}).get("epochs", 50)))
    )

    print(f"\n[3/5] Training Candidate Model for {epochs_to_run} epoch(s)...")
    train_results = trainer.train(max_epochs=epochs_to_run)

    # 3. Model Comparison
    print("\n[4/5] Evaluating Baseline vs Candidate on Untouched FER2013 Test Set...")
    baseline_ckpt_path = PROJECT_ROOT / "ml" / "models" / "checkpoints" / "final" / "best_model.pt"
    if not baseline_ckpt_path.exists():
        # Fallback to main checkpoints/best_model.pt if final isn't found
        baseline_ckpt_path = PROJECT_ROOT / "ml" / "models" / "checkpoints" / "best_model.pt"

    gate_delta = (
        min_macro_f1_delta
        if min_macro_f1_delta is not None
        else float(retrain_cfg.get("min_macro_f1_delta", 0.0))
    )
    max_acc_drop = float(retrain_cfg.get("max_accuracy_drop", 1.0))
    max_cls_drop = float(retrain_cfg.get("max_class_f1_degradation", 3.0))

    comparator = ModelComparator(
        config_path=config_path,
        min_macro_f1_delta=gate_delta,
        max_accuracy_drop=max_acc_drop,
        max_class_f1_degradation=max_cls_drop,
    )

    cand_version_tag = f"ResidualEmotionCNN-Candidate-{exp_name}"
    comparison_report = comparator.compare_models(
        baseline_checkpoint_path=baseline_ckpt_path,
        candidate_checkpoint_path=cand_best_ckpt,
        baseline_version_name="ResidualEmotionCNN-epoch24",
        candidate_version_name=cand_version_tag,
        test_loader=test_loader,
    )

    # Save comparison report
    report_file = exp_dir / "comparison_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)

    # Print Comparison Summary
    print("\n" + "=" * 75)
    print("MODEL COMPARISON REPORT")
    print("=" * 75)
    b_res = comparison_report["baseline"]
    c_res = comparison_report["candidate"]
    diffs = comparison_report["differences"]

    print(f"{'Metric':<25} {'Baseline (' + b_res['version'] + ')':<30} {'Candidate (' + c_res['version'] + ')':<32} {'Delta':<10}")
    print("-" * 97)
    print(f"{'Accuracy':<25} {b_res['accuracy']:<30.2f}% {c_res['accuracy']:<32.2f}% {diffs['accuracy_diff']:+.2f}%")
    print(f"{'Macro F1':<25} {b_res['macro_f1']:<30.2f}% {c_res['macro_f1']:<32.2f}% {diffs['macro_f1_diff']:+.2f}%")
    print(f"{'Weighted F1':<25} {b_res['weighted_f1']:<30.2f}% {c_res['weighted_f1']:<32.2f}% {diffs['weighted_f1_diff']:+.2f}%")
    print(f"{'Test Loss':<25} {b_res['test_loss']:<30.4f} {c_res['test_loss']:<32.4f} {diffs['loss_diff']:+.4f}")

    print("\nPer-Class F1 Score Comparison:")
    for cls_name in meta["class_names"]:
        b_f1 = diffs["per_class_diff"][cls_name]["baseline_f1"]
        c_f1 = diffs["per_class_diff"][cls_name]["candidate_f1"]
        c_diff = diffs["per_class_diff"][cls_name]["f1_diff"]
        print(f"  {cls_name:<12} Baseline: {b_f1:5.2f}% | Candidate: {c_f1:5.2f}% | Diff: {c_diff:+6.2f}%")

    print("\n" + "=" * 75)
    print(f"PROMOTION DECISION: {comparison_report['decision']}")
    if comparison_report["decision"] == "REJECT":
        print("Reasons:")
        for r in comparison_report["rejection_reasons"]:
            print(f"  - {r}")
    print("=" * 75)

    # 4. Safe Promotion Execution
    promotion_result = {"promoted": False, "status": "SKIPPED"}
    if auto_promote:
        print("\n[5/5] Executing Promotion Gate Handler...")
        promotion_result = comparator.apply_promotion(
            comparison_report=comparison_report,
            production_dir=PROJECT_ROOT / "ml" / "models" / "checkpoints" / "final",
        )
        print(f"  Promotion Status:  {promotion_result['status']}")
        print(f"  Message:           {promotion_result['message']}")
        if promotion_result.get("backup_checkpoint"):
            print(f"  Previous Backup:   {promotion_result['backup_checkpoint']}")
    else:
        print("\n[5/5] Auto-promotion disabled; candidate evaluation preserved.")

    # Save complete run summary
    summary = {
        "experiment_name": exp_name,
        "experiment_dir": str(exp_dir.resolve()),
        "dataset_metadata": meta,
        "training_epochs": epochs_to_run,
        "comparison": comparison_report,
        "promotion": promotion_result,
    }
    with open(exp_dir / "retraining_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="FaceSense AI - Feedback-Based Retraining & Safe Promotion CLI"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override candidate training epochs (e.g. 1 for testing)",
    )
    parser.add_argument(
        "--feedback-dir",
        type=str,
        default=None,
        help="Override feedback directory",
    )
    parser.add_argument(
        "--feedback-weight",
        type=float,
        default=None,
        help="Weight multiplier for feedback samples",
    )
    parser.add_argument(
        "--min-f1-delta",
        type=float,
        default=None,
        help="Minimum required Macro F1 improvement in percent",
    )
    parser.add_argument(
        "--no-auto-promote",
        action="store_true",
        help="Disable automatic promotion even if candidate is better",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Custom experiment run identifier",
    )

    args = parser.parse_args()

    try:
        run_retraining_pipeline(
            config_path=args.config,
            max_epochs=args.epochs,
            feedback_dir=args.feedback_dir,
            feedback_weight=args.feedback_weight,
            min_macro_f1_delta=args.min_f1_delta,
            auto_promote=not args.no_auto_promote,
            experiment_name=args.experiment_name,
        )
    except Exception as e:
        print(f"\n[Error] Retraining pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
