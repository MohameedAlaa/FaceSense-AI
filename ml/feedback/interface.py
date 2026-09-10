"""
FaceSense AI - Feedback Interface & CLI
Provides a command-line interface and programmatic helper for submitting, listing,
exporting, and building datasets from user feedback on predictions.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.feedback.collector import FeedbackCollector, FeedbackState, DEFAULT_VALID_EMOTIONS
from ml.feedback.dataset_builder import FeedbackDatasetBuilder
from ml.inference.predictor import EmotionPredictor


def create_feedback_cli_parser() -> argparse.ArgumentParser:
    """Creates the command line argument parser for FaceSense AI feedback operations."""
    parser = argparse.ArgumentParser(
        description="FaceSense AI - Feedback Collection and Management CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Feedback commands")

    # Command 1: submit
    submit_p = subparsers.add_parser("submit", help="Submit a feedback record for an image prediction")
    submit_p.add_argument("--image", type=str, required=True, help="Path to face/frame image file")
    submit_p.add_argument(
        "--state",
        type=str,
        choices=["correct", "incorrect", "uncertain"],
        default="correct",
        help="Feedback state/outcome",
    )
    submit_p.add_argument(
        "--predicted",
        type=str,
        default=None,
        help="Predicted emotion label (if omitted, model will predict automatically)",
    )
    submit_p.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Prediction confidence score [0.0 - 1.0]",
    )
    submit_p.add_argument(
        "--corrected",
        type=str,
        default=None,
        choices=list(DEFAULT_VALID_EMOTIONS),
        help="User-verified/corrected emotion label (required for incorrect predictions)",
    )
    submit_p.add_argument(
        "--bbox",
        type=int,
        nargs=4,
        default=None,
        metavar=("X", "Y", "W", "H"),
        help="Face bounding box coordinates: x y w h",
    )
    submit_p.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Model version string producing prediction",
    )
    submit_p.add_argument(
        "--feedback-dir",
        type=str,
        default="outputs/feedback",
        help="Directory to store feedback records",
    )
    submit_p.add_argument(
        "--copy-image",
        action="store_true",
        help="Copy image to feedback image store for permanent archiving",
    )

    # Command 2: list
    list_p = subparsers.add_parser("list", help="List collected feedback records")
    list_p.add_argument(
        "--state",
        type=str,
        choices=["correct", "incorrect", "uncertain"],
        default=None,
        help="Filter records by state",
    )
    list_p.add_argument(
        "--feedback-dir",
        type=str,
        default="outputs/feedback",
        help="Directory where feedback records are stored",
    )
    list_p.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum records to display",
    )

    # Command 3: stats
    stats_p = subparsers.add_parser("stats", help="Display summary statistics of feedback data")
    stats_p.add_argument(
        "--feedback-dir",
        type=str,
        default="outputs/feedback",
        help="Directory where feedback records are stored",
    )

    # Command 4: export
    export_p = subparsers.add_parser("export", help="Export feedback records to CSV")
    export_p.add_argument(
        "--output",
        type=str,
        default="outputs/feedback/feedback_records.csv",
        help="Path for output CSV file",
    )
    export_p.add_argument(
        "--feedback-dir",
        type=str,
        default="outputs/feedback",
        help="Directory where feedback records are stored",
    )

    # Command 5: build-dataset
    build_p = subparsers.add_parser("build-dataset", help="Build retraining dataset from feedback samples")
    build_p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/feedback/feedback_dataset",
        help="Target directory for structured dataset",
    )
    build_p.add_argument(
        "--feedback-dir",
        type=str,
        default="outputs/feedback",
        help="Directory where feedback records are stored",
    )
    build_p.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation split ratio (0.0 to 1.0)",
    )

    return parser


def handle_submit(args: argparse.Namespace) -> None:
    collector = FeedbackCollector(feedback_dir=args.feedback_dir)

    predicted_emotion = args.predicted
    confidence = args.confidence
    model_version = args.model_version

    # If predicted/confidence not provided, run predictor on the image
    if predicted_emotion is None or confidence is None:
        checkpoint_path = PROJECT_ROOT / "ml" / "models" / "checkpoints" / "final" / "best_model.pt"
        if checkpoint_path.exists():
            predictor = EmotionPredictor(checkpoint_path=checkpoint_path)
            res = predictor.predict(args.image)
            predicted_emotion = predicted_emotion or res["predicted_emotion"]
            confidence = confidence if confidence is not None else res["confidence"]
            model_version = model_version or f"{res['model_name']}-epoch{predictor.best_epoch or 'final'}"
        else:
            if predicted_emotion is None:
                raise ValueError("Must provide --predicted emotion when model checkpoint is not loaded.")
            if confidence is None:
                confidence = 1.0

    record = collector.add_feedback(
        image_path=args.image,
        predicted_emotion=predicted_emotion,
        confidence=confidence,
        corrected_emotion=args.corrected,
        bounding_box=args.bbox,
        state=args.state,
        model_version=model_version,
        copy_image=args.copy_image,
    )

    print("\n[FaceSense AI Feedback] Feedback record successfully recorded:")
    print(json.dumps(record, indent=2))


def handle_list(args: argparse.Namespace) -> None:
    collector = FeedbackCollector(feedback_dir=args.feedback_dir)
    records = collector.load_feedback_records(filter_state=args.state)

    print(f"\n[FaceSense AI Feedback] Found {len(records)} records (showing up to {args.limit}):")
    print(f"{'ID':<28} {'State':<12} {'Predicted':<10} {'Conf':<6} {'Corrected':<10} {'Model Version':<25}")
    print("-" * 95)
    for rec in records[-args.limit:]:
        fid = rec.get("feedback_id", "N/A")
        state = rec.get("state", "N/A")
        pred = rec.get("predicted_emotion", "N/A")
        conf = f"{rec.get('confidence', 0.0):.2f}"
        corr = rec.get("corrected_emotion") or "-"
        mv = rec.get("model_version", "N/A")
        print(f"{fid:<28} {state:<12} {pred:<10} {conf:<6} {corr:<10} {mv:<25}")


def handle_stats(args: argparse.Namespace) -> None:
    collector = FeedbackCollector(feedback_dir=args.feedback_dir)
    stats = collector.get_summary_statistics()
    print("\n[FaceSense AI Feedback] Summary Statistics:")
    print(json.dumps(stats, indent=2))


def handle_export(args: argparse.Namespace) -> None:
    collector = FeedbackCollector(feedback_dir=args.feedback_dir)
    csv_path = collector.export_to_csv(args.output)
    print(f"\n[FaceSense AI Feedback] Successfully exported records to CSV: {csv_path.resolve()}")


def handle_build_dataset(args: argparse.Namespace) -> None:
    builder = FeedbackDatasetBuilder(feedback_dir=args.feedback_dir)
    manifest = builder.build_feedback_dataset(
        output_dataset_dir=args.output_dir,
        val_split_ratio=args.val_split,
    )
    print(f"\n[FaceSense AI Feedback] Successfully built feedback dataset:")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = create_feedback_cli_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == "submit":
            handle_submit(args)
        elif args.command == "list":
            handle_list(args)
        elif args.command == "stats":
            handle_stats(args)
        elif args.command == "export":
            handle_export(args)
        elif args.command == "build-dataset":
            handle_build_dataset(args)
    except Exception as e:
        print(f"\n[Error] Feedback operation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
