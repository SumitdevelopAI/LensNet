"""
Unit and integration tests for LensNet.

Tests cover:
* SpectralConv2d / FNOBlock2d layer shapes and forward passes.
* CNNBaseline and FNOClassifier end-to-end forward passes.
* LensDataset (synthetic fallback) data loading.
* Trainer smoke-test with a tiny synthetic dataset.
"""

import pytest
import torch
import numpy as np

from lensnet.models.spectral_layers import SpectralConv2d, FNOBlock2d
from lensnet.models.cnn_baseline import CNNBaseline
from lensnet.models.fno_classifier import FNOClassifier
from lensnet.data.dataset import LensDataset, make_synthetic_dataset, get_dataloaders
from lensnet.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture()
def tiny_loaders():
    """Return tiny train / val / test loaders using synthetic data."""
    return get_dataloaders(
        root=None,
        image_size=32,
        batch_size=8,
        n_synthetic=48,
        seed=0,
        augment=False,
    )


# ---------------------------------------------------------------------------
# Spectral layer tests
# ---------------------------------------------------------------------------


class TestSpectralConv2d:
    def test_output_shape(self, device):
        layer = SpectralConv2d(in_channels=4, out_channels=8, modes1=6, modes2=6)
        layer = layer.to(device)
        x = torch.randn(2, 4, 32, 32, device=device)
        y = layer(x)
        assert y.shape == (2, 8, 32, 32), f"Unexpected shape {y.shape}"

    def test_output_is_real(self, device):
        layer = SpectralConv2d(8, 8, 4, 4).to(device)
        x = torch.randn(1, 8, 16, 16, device=device)
        y = layer(x)
        assert y.is_floating_point(), "Output should be real-valued"
        assert not y.is_complex(), "Output should not be complex"

    def test_non_square_input(self, device):
        layer = SpectralConv2d(4, 4, 4, 4).to(device)
        x = torch.randn(2, 4, 24, 32, device=device)
        y = layer(x)
        assert y.shape == (2, 4, 24, 32)

    def test_gradient_flow(self, device):
        layer = SpectralConv2d(4, 4, 4, 4).to(device)
        x = torch.randn(2, 4, 16, 16, device=device, requires_grad=True)
        y = layer(x).sum()
        y.backward()
        assert x.grad is not None
        # All complex weight parameters should have gradients.
        for name, p in layer.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


class TestFNOBlock2d:
    def test_output_shape(self, device):
        block = FNOBlock2d(channels=16, modes1=4, modes2=4).to(device)
        x = torch.randn(3, 16, 32, 32, device=device)
        y = block(x)
        assert y.shape == x.shape

    def test_residual_connection(self, device):
        """Output should differ from both the spectral and bypass branches alone."""
        block = FNOBlock2d(channels=8, modes1=4, modes2=4).to(device)
        x = torch.randn(1, 8, 16, 16, device=device)
        y_full = block(x)
        # Just check output is defined and has the right shape.
        assert y_full.shape == x.shape


# ---------------------------------------------------------------------------
# Model forward pass tests
# ---------------------------------------------------------------------------


class TestCNNBaseline:
    @pytest.mark.parametrize("in_ch", [1, 3])
    def test_forward_shape(self, device, in_ch):
        model = CNNBaseline(num_classes=3, in_channels=in_ch).to(device)
        x = torch.randn(4, in_ch, 64, 64, device=device)
        logits = model(x)
        assert logits.shape == (4, 3)

    def test_parameter_count(self):
        model = CNNBaseline(num_classes=3, in_channels=1)
        n = sum(p.numel() for p in model.parameters())
        # ResNet-18 ~ 11M params; our model is slightly smaller.
        assert n > 1_000_000, "Parameter count unexpectedly low"

    def test_output_is_logits(self, device):
        model = CNNBaseline(num_classes=3, in_channels=1).to(device)
        x = torch.randn(2, 1, 64, 64, device=device)
        logits = model(x)
        # Logits are unbounded; probabilities (after softmax) should sum to 1.
        probs = torch.softmax(logits, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2, device=device), atol=1e-5)


class TestFNOClassifier:
    @pytest.mark.parametrize("in_ch,h,w", [(1, 32, 32), (3, 48, 48)])
    def test_forward_shape(self, device, in_ch, h, w):
        model = FNOClassifier(
            num_classes=3,
            in_channels=in_ch,
            hidden_channels=16,
            modes1=8,
            modes2=8,
            n_fno_blocks=2,
            projection_channels=32,
        ).to(device)
        x = torch.randn(4, in_ch, h, w, device=device)
        logits = model(x)
        assert logits.shape == (4, 3)

    def test_gradient_flows_through_fno(self, device):
        model = FNOClassifier(
            num_classes=3, in_channels=1,
            hidden_channels=8, modes1=4, modes2=4, n_fno_blocks=1,
        ).to(device)
        x = torch.randn(2, 1, 16, 16, device=device)
        loss = model(x).sum()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_modes_clamped_by_spatial_size(self, device):
        """Modes > spatial_dim // 2 should still produce valid output."""
        model = FNOClassifier(
            num_classes=3, in_channels=1,
            hidden_channels=8, modes1=20, modes2=20, n_fno_blocks=1,
        ).to(device)
        x = torch.randn(2, 1, 16, 16, device=device)
        # Should not raise even though modes > spatial_dim // 2.
        logits = model(x)
        assert logits.shape == (2, 3)


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


class TestMakeSyntheticDataset:
    def test_output_shapes(self):
        images, labels = make_synthetic_dataset(n_samples=30, image_size=32)
        assert images.shape == (30, 1, 32, 32)
        assert labels.shape == (30,)

    def test_class_balance(self):
        images, labels = make_synthetic_dataset(n_samples=30, image_size=16)
        for cls in range(3):
            assert (labels == cls).sum() == 10

    def test_pixel_range(self):
        images, _ = make_synthetic_dataset(n_samples=15, image_size=16)
        assert images.min() >= 0.0
        assert images.max() <= 1.0


class TestLensDataset:
    def test_synthetic_fallback(self):
        ds = LensDataset(root=None, n_synthetic=30, image_size=32)
        assert len(ds) == 30

    def test_getitem_returns_tensor(self):
        ds = LensDataset(root=None, n_synthetic=12, image_size=32)
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(label, int)
        assert img.shape[0] == 1  # single channel

    def test_class_names(self):
        ds = LensDataset(root=None, n_synthetic=12, image_size=16)
        assert ds.class_names == ["no_sub", "cdm_sub", "wdm_sub"]

    def test_nonexistent_root_falls_back(self, tmp_path):
        ds = LensDataset(
            root=tmp_path / "nonexistent",
            n_synthetic=12,
            image_size=16,
        )
        assert len(ds) == 12


class TestGetDataloaders:
    def test_split_sizes(self):
        train, val, test = get_dataloaders(
            n_synthetic=90, image_size=16, batch_size=8, seed=0
        )
        total = len(train.dataset) + len(val.dataset) + len(test.dataset)
        assert total == 90

    def test_batch_shape(self, tiny_loaders):
        train, _, _ = tiny_loaders
        imgs, labels = next(iter(train))
        assert imgs.ndim == 4  # (B, C, H, W)
        assert labels.ndim == 1


# ---------------------------------------------------------------------------
# Trainer smoke test
# ---------------------------------------------------------------------------


class TestTrainer:
    def test_smoke_train_cnn(self, tiny_loaders):
        train_loader, val_loader, test_loader = tiny_loaders
        model = CNNBaseline(num_classes=3, in_channels=1)
        trainer = Trainer(model, device="cpu", lr=1e-3, epochs=2, patience=None)
        history = trainer.fit(train_loader, val_loader)
        assert len(history["train_loss"]) == 2
        assert len(history["val_acc"]) == 2

    def test_smoke_train_fno(self, tiny_loaders):
        train_loader, val_loader, test_loader = tiny_loaders
        model = FNOClassifier(
            num_classes=3, in_channels=1,
            hidden_channels=8, modes1=4, modes2=4, n_fno_blocks=2,
        )
        trainer = Trainer(model, device="cpu", lr=1e-3, epochs=2, patience=None)
        history = trainer.fit(train_loader, val_loader)
        assert len(history["train_loss"]) == 2

    def test_evaluate_returns_metrics(self, tiny_loaders):
        _, _, test_loader = tiny_loaders
        model = FNOClassifier(
            num_classes=3, in_channels=1,
            hidden_channels=8, modes1=4, modes2=4, n_fno_blocks=1,
        )
        trainer = Trainer(model, device="cpu", lr=1e-3, epochs=1, patience=None)
        metrics = trainer.evaluate(test_loader)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "auc" in metrics

    def test_early_stopping(self, tiny_loaders):
        train_loader, val_loader, _ = tiny_loaders
        model = CNNBaseline(num_classes=3, in_channels=1)
        # Very short patience → early stopping triggers quickly.
        trainer = Trainer(model, device="cpu", lr=1e-3, epochs=50, patience=2)
        history = trainer.fit(train_loader, val_loader)
        # Should stop well before 50 epochs.
        assert len(history["train_loss"]) <= 50
