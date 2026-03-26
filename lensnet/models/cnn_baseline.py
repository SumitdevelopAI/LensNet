"""
Standard CNN baseline for gravitational lens classification (Common Test I).

This model mimics the architecture style used in Common Test I of the
Strong Gravitational Lens Finding Challenge: a ResNet-18 backbone with a
custom classification head suited for 3-class lens classification.

Classes:
    0 – no substructure
    1 – CDM subhalo substructure
    2 – WDM vortex substructure
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class CNNBaseline(nn.Module):
    """ResNet-18 baseline classifier for gravitational lens images.

    The model uses a pretrained (or randomly initialised) ResNet-18 as the
    convolutional feature extractor and replaces the final fully-connected
    layer with a task-specific head.

    Args:
        num_classes:   Number of output classes (default 3).
        in_channels:   Number of input image channels.  ResNet-18 expects 3
                       channels; pass 1 for single-band images and the first
                       conv layer will be replaced automatically.
        pretrained:    Load ImageNet pretrained weights (requires internet on
                       the first call; defaults to ``False`` for offline use).
        dropout_rate:  Dropout probability applied before the final linear
                       layer (regularisation).
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 1,
        pretrained: bool = False,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)

        # Adapt the first convolution for single-channel (or multi-channel)
        # inputs while preserving the remaining layer weights.
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        # Remove the original classification head; keep the feature extractor.
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Custom classification head.
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Image tensor of shape ``(B, C, H, W)``.

        Returns:
            Logit tensor of shape ``(B, num_classes)``.
        """
        features = self.backbone(x)
        return self.head(features)
