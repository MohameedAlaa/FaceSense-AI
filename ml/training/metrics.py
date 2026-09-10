"""
FaceSense AI - Metrics and Evaluation Utilities
Calculates overall accuracy, macro F1, weighted F1, per-class precision/recall/F1,
confusion matrix, and saves JSON/CSV metrics and text reports.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_evaluation_metrics(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_pred: Union[List[int], np.ndarray, torch.Tensor],
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Computes comprehensive multi-class classification metrics.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    num_classes = len(class_names)
    labels = list(range(num_classes))

    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0)
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0)

    # Per-class metrics
    precisions = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    per_class_metrics = {}
    for idx, name in enumerate(class_names):
        per_class_metrics[name] = {
            "precision": float(precisions[idx] * 100.0),
            "recall": float(recalls[idx] * 100.0),
            "f1_score": float(f1s[idx] * 100.0),
            "support": int((y_true == idx).sum()),
        }

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    # Classification Report
    clf_report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        digits=4,
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class_metrics,
        "confusion_matrix": cm,
        "classification_report": clf_report,
    }


def save_metrics(
    metrics_dict: Dict[str, Any],
    output_dir: Union[str, Path] = "outputs/metrics",
    prefix: str = "test_metrics",
) -> Path:
    """
    Saves metrics as formatted JSON and text summary.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_file = out_path / f"{prefix}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    # Text report
    if "classification_report" in metrics_dict:
        report_file = out_path / f"{prefix}_classification_report.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"FaceSense AI Evaluation Report ({prefix})\n")
            f.write("=" * 60 + "\n\n")
            f.write(metrics_dict["classification_report"])
            f.write("\n\nOverall Summary:\n")
            f.write(f"  Accuracy:    {metrics_dict['accuracy']:.2f}%\n")
            f.write(f"  Macro F1:    {metrics_dict['macro_f1']:.2f}%\n")
            f.write(f"  Weighted F1: {metrics_dict['weighted_f1']:.2f}%\n")

    return json_file


def format_confusion_matrix_ascii(cm: List[List[int]], class_names: List[str]) -> str:
    """Formats confusion matrix into an ASCII table string."""
    title_label = "True \\ Pred"
    header = f"{title_label:<12} | " + " | ".join(f"{name[:6]:>6}" for name in class_names)
    separator = "-" * len(header)
    lines = [header, separator]
    for idx, row in enumerate(cm):
        row_str = f"{class_names[idx]:<12} | " + " | ".join(f"{val:>6}" for val in row)
        lines.append(row_str)
    return "\n".join(lines)
