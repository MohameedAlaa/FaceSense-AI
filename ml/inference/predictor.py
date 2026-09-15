"""
FaceSense AI - Single Image Emotion Prediction Pipeline
Provides EmotionPredictor class for loading the final ResidualEmotionCNN checkpoint
and running inference on single face images with exact validation/test preprocessing.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
import torchvision.transforms as T

from ml.models.residual_cnn import ResidualEmotionCNN
from ml.models.builder import build_model_from_config


def get_inference_transforms(img_size: Tuple[int, int] = (48, 48)) -> T.Compose:
    """
    Constructs the exact evaluation/test preprocessing transform pipeline:
      1. Deterministic Resize to 48x48
      2. ToTensor (scales uint8 [0, 255] -> float [0.0, 1.0])
      3. Grayscale normalization (mean=[0.5], std=[0.5])
    """
    return T.Compose([
        T.Resize(size=img_size, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])


def load_and_preprocess_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    img_size: Tuple[int, int] = (48, 48),
    transform: Optional[T.Compose] = None,
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Loads, validates, and preprocesses an input image to a tensor matching model inputs.

    Args:
        image_input: File path, PIL Image, or numpy array.
        img_size: Target image resolution tuple (default 48x48).
        transform: Optional torchvision transform (defaults to get_inference_transforms).

    Returns:
        tensor: Normalized float tensor of shape [1, 1, H, W]
        pil_image: Preprocessed 48x48 grayscale PIL Image for inspection/visualization.
    """
    if transform is None:
        transform = get_inference_transforms(img_size)

    # 1. Resolve PIL Image
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path.resolve()}")
        if not path.is_file():
            raise ValueError(f"Path is not a regular file: {path.resolve()}")
        try:
            with Image.open(path) as img:
                pil_img = img.convert("L")
        except UnidentifiedImageError as e:
            raise ValueError(f"Cannot identify or read image file '{path}': {e}")
        except Exception as e:
            raise ValueError(f"Failed opening image '{path}': {e}")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("L")
    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            raise ValueError("Input numpy array is empty.")
        if image_input.ndim == 3:
            # If 3-channel RGB/BGR, convert using PIL
            pil_img = Image.fromarray(image_input).convert("L")
        elif image_input.ndim == 2:
            pil_img = Image.fromarray(image_input, mode="L")
        else:
            raise ValueError(f"Unsupported numpy array dimensions: {image_input.shape}")
    else:
        raise TypeError(
            f"Unsupported image input type '{type(image_input)}'. Expected str, Path, PIL.Image, or numpy.ndarray."
        )

    # Resize PIL image for visualization
    vis_pil = pil_img.resize(img_size, Image.Resampling.BILINEAR)

    # Apply transform: PIL (L) -> [1, 48, 48] -> Add batch dim -> [1, 1, 48, 48]
    tensor = transform(vis_pil)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    return tensor, vis_pil


class EmotionPredictor:
    """
    Production-ready Emotion Predictor for Facial Expression Recognition.
    Loads and runs the verified ResidualEmotionCNN checkpoint on single images.

    Important notice on model interpretation:
    This model performs facial expression classification based on visual patterns.
    Predictions represent categorized facial expressions, not a person's true internal emotional state.
    """

    DEFAULT_CHECKPOINT = "ml/models/checkpoints/final/best_model.pt"

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT,
        device: Optional[Union[str, torch.device]] = None,
        confidence_threshold: float = 0.0,
    ):
        """
        Args:
            checkpoint_path: Path to best_model.pt checkpoint.
            device: Computing device ('cpu' by default).
            confidence_threshold: Minimum confidence score (0.0 to 1.0) required.
                                  If top probability is below this threshold, predicted emotion is "uncertain".
        """
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found at: {self.checkpoint_path.resolve()}")

        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.confidence_threshold = float(confidence_threshold)

        # Load checkpoint & reconstruct architecture
        self._load_checkpoint_and_model()

        # Build inference transform
        self.img_size = tuple(self.config.get("dataset", {}).get("image_size", [48, 48]))
        self.transform = get_inference_transforms(self.img_size)

    def _load_checkpoint_and_model(self) -> None:
        """Loads weights and builds model from saved checkpoint configuration."""
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        except Exception as e:
            raise ValueError(f"Failed loading checkpoint from '{self.checkpoint_path}': {e}")

        if "model_state_dict" not in checkpoint:
            raise KeyError(f"Invalid checkpoint format in '{self.checkpoint_path}': 'model_state_dict' missing.")

        self.config = checkpoint.get("config", {})
        self.class_names = checkpoint.get(
            "class_names",
            self.config.get("classes", {}).get("names", [
                "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
            ]),
        )
        self.num_classes = len(self.class_names)
        self.best_epoch = checkpoint.get("epoch", None)

        # Build model from checkpoint config or instantiate ResidualEmotionCNN default
        model_cfg = self.config.get("model", {})
        arch_name = model_cfg.get("architecture", "ResidualEmotionCNN")

        if "res" in str(arch_name).lower() or "v2" in str(arch_name).lower():
            self.model = ResidualEmotionCNN(
                in_channels=int(self.config.get("dataset", {}).get("channels", 1)),
                num_classes=self.num_classes,
                channel_list=model_cfg.get("channel_list", [32, 64, 128, 256]),
                fc_dim=int(model_cfg.get("fc_dim", 128)),
                conv_dropout=float(model_cfg.get("conv_dropout", 0.1)),
                fc_dropout=float(model_cfg.get("fc_dropout", 0.4)),
            )
        else:
            self.model = build_model_from_config(self.config)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.model_name = self.model.__class__.__name__

        # Resolve model_version string from metadata or checkpoint
        self.model_version = None
        meta_path = self.checkpoint_path.parent / "model_metadata.json"
        if meta_path.exists():
            try:
                import json
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                    self.model_version = meta_data.get("model_version")
            except Exception:
                pass

        if not self.model_version:
            self.model_version = (
                f"{self.model_name}-epoch{self.best_epoch}"
                if self.best_epoch is not None
                else self.model_name
            )

    def preprocess_image(
        self, image_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocesses an image and returns both the tensor and the PIL visualization image.
        """
        return load_and_preprocess_image(
            image_input=image_input,
            img_size=self.img_size,
            transform=self.transform,
        )

    @torch.no_grad()
    def predict_batch(
        self,
        image_inputs: Sequence[Union[str, Path, Image.Image, np.ndarray]],
        confidence_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs batched inference on multiple face images in a single model forward pass.

        Args:
            image_inputs: Sequence of image paths, PIL Images, or numpy arrays.
            confidence_threshold: Optional threshold override.

        Returns:
            List of prediction dictionaries corresponding to input images in original order.
        """
        if not image_inputs:
            return []

        threshold = (
            self.confidence_threshold
            if confidence_threshold is None
            else float(confidence_threshold)
        )

        # 1. Preprocess each image crop to shape [1, 1, 48, 48]
        tensors = []
        for img_input in image_inputs:
            tensor, _ = self.preprocess_image(img_input)
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            tensors.append(tensor)

        # 2. Concatenate into single batch tensor [N, 1, 48, 48]
        batch_tensor = torch.cat(tensors, dim=0).to(self.device)

        # 3. Single forward pass -> logits [N, num_classes]
        logits = self.model(batch_tensor)

        # 4. Softmax probabilities -> [N, num_classes]
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        # 5. Map rows back to individual prediction dicts in original order
        results: List[Dict[str, Any]] = []
        resolved_checkpoint = str(self.checkpoint_path.resolve())

        for idx in range(len(image_inputs)):
            row_probs = probs[idx]
            prob_dict = {
                class_name: float(row_probs[c_idx])
                for c_idx, class_name in enumerate(self.class_names)
            }

            top_idx = int(np.argmax(row_probs))
            top_class = self.class_names[top_idx]
            top_confidence = float(row_probs[top_idx])

            is_uncertain = top_confidence < threshold
            predicted_emotion = "uncertain" if is_uncertain else top_class

            results.append({
                "predicted_emotion": predicted_emotion,
                "confidence": top_confidence,
                "probabilities": prob_dict,
                "raw_predicted_emotion": top_class,
                "is_uncertain": is_uncertain,
                "confidence_threshold": threshold,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "checkpoint_path": resolved_checkpoint,
            })

        return results

    @torch.no_grad()
    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        confidence_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Runs inference on a single image.

        Args:
            image_input: Path to image file, PIL Image, or numpy array.
            confidence_threshold: Optional threshold override.

        Returns:
            Dictionary containing prediction results.
        """
        results = self.predict_batch([image_input], confidence_threshold=confidence_threshold)
        return results[0]

    def predict_with_visual(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        confidence_threshold: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], Image.Image]:
        """
        Convenience method that returns both prediction result dict and the preprocessed 48x48 PIL face image.
        """
        tensor, pil_face = self.preprocess_image(image_input)
        result = self.predict(pil_face, confidence_threshold=confidence_threshold)
        return result, pil_face
