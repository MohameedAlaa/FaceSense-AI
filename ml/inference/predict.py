"""
FaceSense AI - Inference CLI
Command-line interface for running single-image facial expression predictions,
batch directory processing, and interactive image feedback collection.

Usage:
    # 1. Standard single face inference (unchanged)
    .\\.venv\\Scripts\\python.exe ml/inference/predict.py --image <path_to_image>

    # 2. Interactive image inference with user feedback
    .\\.venv\\Scripts\\python.exe ml/inference/predict.py --image <path_to_image> --feedback [--save-output <annotated_path>]

    # 3. Batch directory inference (non-interactive)
    .\\.venv\\Scripts\\python.exe ml/inference/predict.py --input-dir <path_to_dir> [--save-output <output_dir>]
"""

import argparse
import json
import sys
from pathlib import Path
import cv2

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.detection.detector import FaceDetector
from ml.inference.predictor import EmotionPredictor
from ml.inference.image_pipeline import ImageInferencePipeline
from ml.feedback.image_feedback import ImageFeedbackHandler


def parse_args():
    parser = argparse.ArgumentParser(
        description="FaceSense AI - Facial Expression Prediction & Feedback CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default=None,
        help="Path to input image file for inference or feedback.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Path to folder for batch inference across multiple images.",
    )
    parser.add_argument(
        "--feedback",
        action="store_true",
        help="Enable interactive feedback collection on detected faces.",
    )
    parser.add_argument(
        "--save-output",
        type=str,
        default=None,
        help="Optional path to save annotated image (or directory in batch mode).",
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
        "--feedback-dir",
        type=str,
        default="outputs/feedback",
        help="Directory to store feedback records and face crops.",
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


def print_batch_summary(results: list):
    print("\n" + "=" * 75)
    print("FaceSense AI - Batch Image Inference Summary")
    print("=" * 75)
    total_images = len(results)
    total_faces = sum(r.get("faces_detected", 0) for r in results)
    print(f"Total Images Processed : {total_images}")
    print(f"Total Faces Detected   : {total_faces}")
    print("-" * 75)
    print(f" {'Image File':<25} {'Faces':<8} {'Predictions (Emotion, Conf)':<40}")
    print("-" * 75)
    for r in results:
        fname = r.get("image_file", "unknown")
        faces = r.get("faces_detected", 0)
        preds = r.get("predictions", [])
        pred_strs = [
            f"F{p.get('face_idx')}:{p.get('predicted_emotion')}({p.get('confidence',0)*100:.0f}%)"
            for p in preds
        ]
        pred_summary = ", ".join(pred_strs) if pred_strs else "(None)"
        print(f" {fname:<25} {faces:<8} {pred_summary:<40}")
    print("=" * 75)


def main():
    args = parse_args()

    if not args.image and not args.input_dir:
        print("[Error] Must specify either --image <path> or --input-dir <path>.", file=sys.stderr)
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

    # ── CASE 1: Batch Directory Processing ────────────────────────
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            print(f"[Error] Directory not found: {input_dir}", file=sys.stderr)
            sys.exit(1)

        try:
            pipeline = ImageInferencePipeline(
                detector=FaceDetector(),
                predictor=predictor,
                confidence_threshold=args.threshold,
            )
            save_annotated = bool(args.save_output)
            results = pipeline.process_directory(
                input_dir=input_dir,
                output_dir=args.save_output,
                save_annotated=save_annotated,
            )
        except Exception as e:
            print(f"[Error] Batch processing failed: {e}", file=sys.stderr)
            sys.exit(1)

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print_batch_summary(results)
            if args.save_output:
                print(f"[FaceSense AI] Annotated images saved to: {args.save_output}")
        return

    # ── Validate Image Path for Cases 2 & 3 ────────────────────────
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[Error] Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # ── CASE 2: Interactive Feedback Mode ─────────────────────────
    if args.feedback:
        try:
            # For face detection pipeline, default threshold to 0.35 if not overridden
            det_thresh = args.threshold if args.threshold > 0.0 else 0.35
            pipeline = ImageInferencePipeline(
                detector=FaceDetector(),
                predictor=predictor,
                confidence_threshold=det_thresh,
            )
            frame_bgr, predictions = pipeline.process_image(image_path)
        except Exception as e:
            print(f"[Error] Failed processing image for feedback: {e}", file=sys.stderr)
            sys.exit(1)

        feedback_handler = ImageFeedbackHandler(
            feedback_dir=args.feedback_dir,
            model_version=predictor.model_version,
        )

        if not predictions:
            print(f"\n[FaceSense AI] No faces detected in '{image_path}'.")
            print("Interactive feedback requires at least one detected face.")
        else:
            if predictions[0].get("is_fallback"):
                print(f"[FaceSense AI] Note: No frontal face detected by Haar cascade. Treating full image as Face 1 {predictions[0]['bbox']}.")
            feedback_handler.run_interactive_feedback(
                predictions=predictions,
                original_frame=frame_bgr,
                image_source=image_path,
            )

        if args.save_output and predictions:
            try:
                out_p = Path(args.save_output)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                annotated = pipeline.annotate_image(frame_bgr, predictions)
                cv2.imwrite(str(out_p), annotated)
                print(f"[FaceSense AI] Annotated output image saved to: {out_p.resolve()}")
            except Exception as e:
                print(f"[Warning] Failed to save annotated image: {e}", file=sys.stderr)
        return

    # ── CASE 3: Standard Single-Image Inference (Unchanged) ───────
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

    if args.save_output:
        try:
            out_p = Path(args.save_output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            # Run annotation with detected face box or single full-image box
            pipeline = ImageInferencePipeline(
                detector=FaceDetector(),
                predictor=predictor,
                confidence_threshold=args.threshold,
            )
            frame_bgr, preds = pipeline.process_image(image_path)
            if preds:
                annotated = pipeline.annotate_image(frame_bgr, preds)
            else:
                # Fallback: annotate whole frame with prediction
                h, w = frame_bgr.shape[:2]
                preds = [{
                    "face_idx": 1,
                    "bbox": (0, 0, w, h),
                    "predicted_emotion": result["predicted_emotion"],
                    "confidence": result["confidence"],
                    "is_uncertain": result["is_uncertain"],
                }]
                annotated = pipeline.annotate_image(frame_bgr, preds)
            cv2.imwrite(str(out_p), annotated)
            result["annotated_image_saved"] = str(out_p.resolve())
        except Exception as e:
            print(f"[Warning] Failed to save annotated image: {e}", file=sys.stderr)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_formatted_result(result, str(image_path))


if __name__ == "__main__":
    main()
