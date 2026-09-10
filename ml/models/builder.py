"""
FaceSense AI - Unified Model Factory and Summary Utilities
Instantiates baseline CNN or lightweight residual CNN models driven by configuration.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from ml.models.baseline_cnn import BaselineEmotionCNN, ConvBlock
from ml.models.residual_cnn import ResidualEmotionCNN, ResidualBlock


def get_model_summary(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 1, 48, 48),
) -> Dict[str, Any]:
    """
    Computes parameter counts and performs a dummy dry-run to verify tensor flow.

    Returns:
        Dict with total_params, trainable_params, non_trainable_params, input_shape, and output_shape.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    device = next(model.parameters()).device
    dummy_input = torch.zeros(*input_size, device=device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    if was_training:
        model.train()

    return {
        "model_class": model.__class__.__name__,
        "input_shape": list(dummy_input.shape),
        "output_shape": list(output.shape),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_size_mb": (total_params * 4) / (1024 * 1024),  # FP32 size
    }


def print_model_summary(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 1, 48, 48),
):
    """Prints a formatted human-readable summary of the model."""
    summary = get_model_summary(model, input_size)
    print("=" * 65)
    print(f"Model Summary: {summary['model_class']}")
    print("=" * 65)
    print(f"Input Shape:            {summary['input_shape']}")
    print(f"Output Shape:           {summary['output_shape']}")
    print(f"Trainable Parameters:   {summary['trainable_params']:,}")
    print(f"Non-Trainable Params:   {summary['non_trainable_params']:,}")
    print(f"Total Parameters:       {summary['total_params']:,}")
    print(f"Model Size (FP32):      {summary['param_size_mb']:.2f} MB")
    print("=" * 65)


def build_model_from_config(config: Dict[str, Any]) -> nn.Module:
    """
    Instantiates either BaselineEmotionCNN or ResidualEmotionCNN based on project config.
    Supports architecture names:
      - 'BaselineEmotionCNN', 'baseline_cnn', 'baseline'
      - 'ResidualEmotionCNN', 'residual_cnn', 'residual', 'model_v2'
    """
    in_channels = int(config.get("dataset", {}).get("channels", 1))
    num_classes = int(config.get("classes", {}).get("num_classes", 7))
    model_cfg = config.get("model", {})

    arch = str(model_cfg.get("architecture", "BaselineEmotionCNN")).lower()
    channel_list = model_cfg.get("channel_list", [32, 64, 128, 256])
    fc_dim = int(model_cfg.get("fc_dim", 128))
    conv_dropout = float(model_cfg.get("conv_dropout", 0.25 if "baseline" in arch else 0.1))
    fc_dropout = float(model_cfg.get("fc_dropout", 0.5 if "baseline" in arch else 0.4))

    if "res" in arch or "v2" in arch:
        return ResidualEmotionCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            channel_list=channel_list,
            fc_dim=fc_dim,
            conv_dropout=conv_dropout,
            fc_dropout=fc_dropout,
        )
    elif "base" in arch or "cnn" in arch:
        return BaselineEmotionCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            channel_list=channel_list,
            fc_dim=fc_dim,
            conv_dropout=conv_dropout,
            fc_dropout=fc_dropout,
        )
    else:
        raise ValueError(
            f"Unsupported architecture '{model_cfg.get('architecture')}'. "
            f"Expected one of ['BaselineEmotionCNN', 'baseline_cnn', 'ResidualEmotionCNN', 'residual_cnn']."
        )
