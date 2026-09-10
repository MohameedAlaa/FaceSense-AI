"""
FaceSense AI - Baseline Emotion Classification CNN Model
Clean, modular PyTorch CNN architecture designed for 48x48 single-channel grayscale facial expression recognition.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Standard convolutional block with:
    Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> Dropout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.25,
        pool: bool = True,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(p=dropout_rate))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineEmotionCNN(nn.Module):
    """
    Baseline CNN for 48x48 Grayscale Facial Expression Recognition.

    Architecture summary:
      1. Stage 1: ConvBlock(1 -> 32)   [48x48 -> 24x24]
      2. Stage 2: ConvBlock(32 -> 64)  [24x24 -> 12x12]
      3. Stage 3: ConvBlock(64 -> 128) [12x12 -> 6x6]
      4. Stage 4: ConvBlock(128 -> 256)[6x6 -> 3x3]
      5. Global Average Pooling (GAP)  [256 x 3 x 3 -> 256 x 1 x 1]
      6. Flatten & Dropout             [256]
      7. Dense FC Layer                [256 -> 128] + BatchNorm1d + ReLU + Dropout
      8. Linear Classifier Head        [128 -> num_classes (7)]
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 7,
        channel_list: Optional[List[int]] = None,
        fc_dim: int = 128,
        conv_dropout: float = 0.25,
        fc_dropout: float = 0.5,
    ):
        super().__init__()
        if channel_list is None:
            channel_list = [32, 64, 128, 256]

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Feature extractor backbone
        blocks = []
        current_channels = in_channels
        for out_c in channel_list:
            blocks.append(
                ConvBlock(
                    in_channels=current_channels,
                    out_channels=out_c,
                    dropout_rate=conv_dropout,
                    pool=True,
                )
            )
            current_channels = out_c
        self.features = nn.Sequential(*blocks)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=fc_dropout),
            nn.Linear(current_channels, fc_dim, bias=False),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=fc_dropout),
            nn.Linear(fc_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming (He) normal weight initialization for Conv and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, in_channels, 48, 48)
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        features = self.features(x)
        pooled = self.global_pool(features)
        logits = self.classifier(pooled)
        return logits


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


def print_model_summary(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 1, 48, 48)):
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


def build_model_from_config(config: Dict[str, Any]) -> BaselineEmotionCNN:
    """
    Instantiates BaselineEmotionCNN based on project config dict.
    """
    in_channels = int(config.get("dataset", {}).get("channels", 1))
    num_classes = int(config.get("classes", {}).get("num_classes", 7))
    model_cfg = config.get("model", {})

    channel_list = model_cfg.get("channel_list", [32, 64, 128, 256])
    fc_dim = int(model_cfg.get("fc_dim", 128))
    conv_dropout = float(model_cfg.get("conv_dropout", 0.25))
    fc_dropout = float(model_cfg.get("fc_dropout", 0.5))

    return BaselineEmotionCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        channel_list=channel_list,
        fc_dim=fc_dim,
        conv_dropout=conv_dropout,
        fc_dropout=fc_dropout,
    )
