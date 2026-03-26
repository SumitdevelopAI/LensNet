"""
Spectral convolution layers for the Fourier Neural Operator (FNO).

Reference:
    Li et al., "Fourier Neural Operator for Parametric Partial Differential
    Equations", ICLR 2021 (arXiv:2010.08895).

Key idea:
    Instead of learning filters in the spatial domain (like a CNN), an FNO
    layer applies a *learnable linear transform* in the Fourier frequency
    domain and then maps back to the spatial domain via the inverse FFT.
    This allows the layer to capture *global* dependencies across the entire
    input at the cost of a single FFT pass – O(N log N) vs O(N·k²) for a
    k×k CNN kernel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """2-D spectral convolution layer used inside FNO blocks.

    The layer keeps the ``modes1 × modes2`` lowest-frequency Fourier modes
    and applies a learnable complex weight matrix to them.  All higher
    frequencies are zeroed out (implicit low-pass regularisation).

    Args:
        in_channels:  Number of input feature channels.
        out_channels: Number of output feature channels.
        modes1:       Number of Fourier modes to keep along the first
                      spatial dimension (height).
        modes2:       Number of Fourier modes to keep along the second
                      spatial dimension (width).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # Scaling factor from the original FNO paper.
        scale = 1.0 / (in_channels * out_channels)

        # The complex weights are stored as pairs of real tensors to remain
        # compatible with older PyTorch versions that do not support complex
        # parameter gradients out of the box.
        self.weights1 = nn.Parameter(
            scale
            * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale
            * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _complex_mul2d(
        x: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Batched complex matrix–vector multiply over the channel axis.

        Args:
            x:       Complex tensor of shape ``(B, C_in, M1, M2)``.
            weights: Complex tensor of shape ``(C_in, C_out, M1, M2)``.

        Returns:
            Complex tensor of shape ``(B, C_out, M1, M2)``.
        """
        # Einstein summation: batch (b), input channel (i), modes (x, y),
        # output channel (o).
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the spectral convolution.

        Args:
            x: Real tensor of shape ``(B, C_in, H, W)``.

        Returns:
            Real tensor of shape ``(B, C_out, H, W)``.
        """
        batchsize = x.shape[0]
        h, w = x.shape[-2], x.shape[-1]

        # 1. Forward 2-D real FFT → shape (B, C_in, H, W//2+1) complex.
        x_ft = torch.fft.rfft2(x, norm="ortho")

        # 2. Allocate output spectrum (all zeros).
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            h,
            w // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # 3. Multiply the *kept* frequency modes by the learned weights.
        #    Two patches are needed because rfft2 stores frequencies for
        #    positive and negative wavenumbers differently.
        #    Clamp the effective modes to the actual spectrum size so the
        #    layer gracefully handles inputs smaller than the configured modes.
        m1 = min(self.modes1, h // 2)
        m2 = min(self.modes2, w // 2 + 1)

        out_ft[:, :, :m1, :m2] = self._complex_mul2d(
            x_ft[:, :, :m1, :m2],
            self.weights1[:, :, :m1, :m2],
        )
        out_ft[:, :, -m1:, :m2] = self._complex_mul2d(
            x_ft[:, :, -m1:, :m2],
            self.weights2[:, :, :m1, :m2],
        )

        # 4. Inverse 2-D real FFT → back to spatial domain.
        x_out = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
        return x_out


class FNOBlock2d(nn.Module):
    """A single FNO residual block for 2-D inputs.

    Each block combines:
    * A spectral (Fourier) convolution that captures global frequency patterns.
    * A pointwise 1×1 convolution that captures local channel interactions.

    The outputs of both paths are summed before applying a non-linearity,
    matching the original FNO architecture (Fig. 2 in the paper).

    Args:
        channels:  Number of feature channels (kept constant through the block).
        modes1:    Number of Fourier modes along height.
        modes2:    Number of Fourier modes along width.
        activation: Activation function applied after the residual sum.
    """

    def __init__(
        self,
        channels: int,
        modes1: int,
        modes2: int,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.spectral_conv = SpectralConv2d(channels, channels, modes1, modes2)
        self.bypass_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation = activation if activation is not None else nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.spectral_conv(x) + self.bypass_conv(x))
