"""
FaceSense AI - Lightweight Residual Emotion Classification CNN (Model V2)
Custom residual convolutional neural network designed for 48x48 single-channel grayscale
facial expression recognition on CPU/GPU.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Lightweight Residual Block with shortcut connection:
    Path: Conv2d(3x3) -> BatchNorm2d -> ReLU -> Conv2d(3x3) -> BatchNorm2d -> Dropout (optional)
    Shortcut: Identity if in_channels == out_channels and stride == 1,
              otherwise Conv2d(1x1, stride) -> BatchNorm2d.
    Output: ReLU(Path(x) + Shortcut(x))
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()

        # Shortcut projection if channel dimension changes or spatial downsampling occurs
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        out = out + identity
        out = self.relu(out)
        return out


class ResidualEmotionCNN(nn.Module):
    """
    Lightweight Residual Emotion Classification CNN (Model V2).

    Architecture overview:
      1. Initial Stem: Conv2d(1 -> channel_list[0], 3x3) + BatchNorm2d + ReLU [48x48]
      2. Stage 1: ResidualBlock(32 -> 32, stride=1) + MaxPool2d(2)           [48x48 -> 24x24]
      3. Stage 2: ResidualBlock(32 -> 64, stride=1) + MaxPool2d(2)           [24x24 -> 12x12]
      4. Stage 3: ResidualBlock(64 -> 128, stride=1) + MaxPool2d(2)          [12x12 -> 6x6]
      5. Stage 4: ResidualBlock(128 -> 256, stride=1) + MaxPool2d(2)         [6x6 -> 3x3]
      6. Global Average Pooling (GAP)                                        [256 x 3 x 3 -> 256 x 1 x 1]
      7. Flatten & Dropout
      8. Dense FC classification head: Linear(256 -> fc_dim) + BatchNorm1d + ReLU + Dropout
      9. Output Logits: Linear(fc_dim -> num_classes (7))
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 7,
        channel_list: Optional[List[int]] = None,
        fc_dim: int = 128,
        conv_dropout: float = 0.1,
        fc_dropout: float = 0.4,
    ):
        super().__init__()
        if channel_list is None:
            channel_list = [32, 64, 128, 256]

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channel_list = channel_list

        # Initial stem
        first_channels = channel_list[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, first_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(first_channels),
            nn.ReLU(inplace=True),
        )

        # Residual stages with spatial downsampling
        stages = []
        curr_c = first_channels
        for out_c in channel_list:
            stages.append(
                ResidualBlock(
                    in_channels=curr_c,
                    out_channels=out_c,
                    stride=1,
                    dropout_rate=conv_dropout,
                )
            )
            stages.append(nn.MaxPool2d(kernel_size=2, stride=2))
            curr_c = out_c

        self.stages = nn.Sequential(*stages)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=fc_dropout),
            nn.Linear(curr_c, fc_dim, bias=False),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=fc_dropout),
            nn.Linear(fc_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming (He) normal initialization for Conv and Linear layers."""
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
            x: Input tensor [batch_size, in_channels, 48, 48]
        Returns:
            Logits tensor [batch_size, num_classes]
        """
        out = self.stem(x)
        out = self.stages(out)
        out = self.global_pool(out)
        logits = self.classifier(out)
        return logits
