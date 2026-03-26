"""
Fourier Neural Operator (FNO) classifier for gravitational lens images.

Architecture overview
---------------------
Input image (B, C_in, H, W)
    │
    ▼
Lifting layer  – pointwise Conv2d maps C_in → hidden_channels
    │
    ▼
N × FNO blocks  – each block = SpectralConv2d + bypass Conv1×1 + GELU
    │
    ▼
Projection layer  – Conv1×1 maps hidden_channels → projection_channels
    │
    ▼
Global average pooling  → (B, projection_channels)
    │
    ▼
MLP head  → (B, num_classes)

Why FNO differs from a standard CNN
-------------------------------------
A CNN learns spatially local filters (k×k receptive field).  A spectral
convolution (the core FNO operation) effectively has a *global* receptive
field: it operates in the Fourier domain and can therefore model long-range
correlations in the image without needing many stacked layers.  The learnable
weights in frequency space also act as a natural regulariser: by retaining
only the ``modes`` lowest frequencies, the network is explicitly biased
towards smooth, slowly-varying features – well suited for the diffuse,
arc-shaped structures in gravitational lens images.

For classification the FNO blocks replace (or augment) the convolutional
feature extractor, while a lightweight MLP head produces the final logits.
"""

import torch
import torch.nn as nn

from .spectral_layers import FNOBlock2d


class FNOClassifier(nn.Module):
    """FNO-based image classifier for gravitational lens images.

    Args:
        num_classes:        Number of output classes (default 3).
        in_channels:        Number of input image channels (default 1).
        hidden_channels:    Width of the internal feature representation.
        modes1:             Fourier modes along the height dimension.
        modes2:             Fourier modes along the width dimension.
        n_fno_blocks:       Number of stacked FNO residual blocks.
        projection_channels: Channels after the final 1×1 projection.
        dropout_rate:       Dropout probability in the MLP head.
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 1,
        hidden_channels: int = 64,
        modes1: int = 12,
        modes2: int = 12,
        n_fno_blocks: int = 4,
        projection_channels: int = 128,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # 1. Lifting: map input channels → hidden_channels
        # ------------------------------------------------------------------
        self.lifting = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

        # ------------------------------------------------------------------
        # 2. FNO blocks
        # ------------------------------------------------------------------
        self.fno_blocks = nn.Sequential(
            *[
                FNOBlock2d(hidden_channels, modes1, modes2)
                for _ in range(n_fno_blocks)
            ]
        )

        # ------------------------------------------------------------------
        # 3. Projection: hidden_channels → projection_channels
        # ------------------------------------------------------------------
        self.projection = nn.Sequential(
            nn.Conv2d(hidden_channels, projection_channels, kernel_size=1),
            nn.GELU(),
        )

        # ------------------------------------------------------------------
        # 4. Global average pooling + MLP head
        # ------------------------------------------------------------------
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(projection_channels, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Image tensor of shape ``(B, C_in, H, W)``.

        Returns:
            Logit tensor of shape ``(B, num_classes)``.
        """
        x = self.lifting(x)       # (B, hidden, H, W)
        x = self.fno_blocks(x)    # (B, hidden, H, W)
        x = self.projection(x)    # (B, proj, H, W)
        x = self.pool(x)          # (B, proj, 1, 1)
        return self.head(x)       # (B, num_classes)
