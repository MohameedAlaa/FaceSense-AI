"""
FaceSense AI - Inference CLI
Command-line interface for running single image facial expression predictions.

Usage:
    .\\.venv\\Scripts\\python.exe ml/inference/predict.py --image <path_to_image> [--checkpoint <path>] [--threshold <0.0-1.0>] [--save-preprocessed <output_path>]
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.predictor import EmotionPredictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="FaceSense AI - Facial Expression Prediction CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        required=True,
        help="Path to the input face image file.",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default="ml/models/checkpoints/final/best_model.pt",
        help="Path to trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.0,
        help="Confidence threshold (0.0 to 1.0). If max probability is lower, predicted emotion is 'uncertain'.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted text.",
    )
    parser.add_argument(
        "--save-preprocessed",
        type=str,
        default=None,
        help="Optional path to save the 48x48 preprocessed grayscale face image.",
    )
    return parser.parse_args()


def print_formatted_result(result: dict, image_path: str):
    print("=" * 65)
    print("FaceSense AI - Facial Expression Prediction")
    print("=" * 65)
    print(f"Input Image:       {image_path}")
    print(f"Model:             {result['model_name']}")
    print(f"Checkpoint:        {Path(result['checkpoint_path']).name}")
    print("-" * 65)
    print(f"Predicted Emotion: {result['predicted_emotion'].upper()}")
    print(f"Confidence:        {result['confidence'] * 100:.2f}%")
    if result["is_uncertain"]:
        print(f"Note:              Confidence is below threshold ({result['confidence_threshold'] * 100:.1f}%) -> uncertain")
    print("-" * 65)
    print("Emotion Probabilities:")
    # Sort probabilities descending
    sorted_probs = sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True)
    for emotion, prob in sorted_probs:
        bar_len = int(prob * 30)
        bar = "#" * bar_len + "-" * (30 - bar_len)
        print(f"  {emotion:<10} : {prob * 100:6.2f}%  [{bar}]")
    print("=" * 65)
    print("Notice: Predictions reflect facial expression classification,")
    print("not the person's true internal emotional state.")
    print("=" * 65)


def main():
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[Error] Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[Error] Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    try:
        predictor = EmotionPredictor(
            checkpoint_path=checkpoint_path,
            confidence_threshold=args.threshold,
        )
    except Exception as e:
        print(f"[Error] Failed to initialize predictor: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.save_preprocessed:
            result, visual_img = predictor.predict_with_visual(image_path)
            out_p = Path(args.save_preprocessed)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            visual_img.save(out_p)
            result["preprocessed_image_saved"] = str(out_p.resolve())
        else:
            result = predictor.predict(image_path)
    except Exception as e:
        print(f"[Error] Prediction failed: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_formatted_result(result, str(image_path))


if __name__ == "__main__":
    main()
