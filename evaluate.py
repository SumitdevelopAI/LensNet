"""
evaluate.py – Compare CNN baseline vs. FNO classifier on the test set.

Usage
-----
Compare both models after loading their checkpoints:
    python evaluate.py \\
        --cnn-checkpoint  checkpoints/cnn_best.pt \\
        --fno-checkpoint  checkpoints/fno_best.pt

Run a quick comparison on synthetic data (no checkpoints needed):
    python evaluate.py --synthetic --epochs 5

The script prints a side-by-side comparison table and optionally saves a
matplotlib figure with learning curves and a ROC-AUC bar chart.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from lensnet.data import get_dataloaders
from lensnet.models import CNNBaseline, FNOClassifier
from lensnet.training import Trainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare CNN baseline and FNO classifier."
    )
    p.add_argument("--data", default=None, metavar="DIR")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--in-channels", type=int, default=1)
    p.add_argument("--cnn-checkpoint", default=None, metavar="PATH",
                   help="Load CNN model from checkpoint instead of training.")
    p.add_argument("--fno-checkpoint", default=None, metavar="PATH",
                   help="Load FNO model from checkpoint instead of training.")
    p.add_argument("--device", default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--n-synthetic", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results",
                   help="Directory to save figures and JSON results.")
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_predictions(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (labels, predictions, probabilities) for *loader*."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.append(probs)
    return (
        np.array(all_labels),
        np.array(all_preds),
        np.concatenate(all_probs, axis=0),
    )


def _compute_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    n_classes = probs.shape[1]
    try:
        if n_classes == 2:
            return float(roc_auc_score(labels, probs[:, 1]))
        return float(
            roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        )
    except ValueError:
        return float("nan")


def _train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: str,
    lr: float,
    epochs: int,
    patience: int,
) -> tuple[nn.Module, dict]:
    trainer = Trainer(
        model=model,
        device=device,
        lr=lr,
        epochs=epochs,
        patience=patience,
    )
    history = trainer.fit(train_loader, val_loader)
    return model, history


def _load_checkpoint(
    model: nn.Module,
    path: str,
    device: torch.device,
) -> nn.Module:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_learning_curves(
    cnn_history: dict,
    fno_history: dict,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, metric, title in zip(
        axes,
        ["val_loss", "val_acc", "val_auc"],
        ["Validation Loss", "Validation Accuracy", "Validation AUC"],
    ):
        ax.plot(cnn_history[metric], label="CNN (baseline)", marker="o", ms=3)
        ax.plot(fno_history[metric], label="FNO", marker="s", ms=3)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Learning-curve plot saved to {out_path}")


def _plot_confusion_matrices(
    cnn_labels: np.ndarray,
    cnn_preds: np.ndarray,
    fno_labels: np.ndarray,
    fno_preds: np.ndarray,
    class_names: list[str],
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, labels, preds, title in zip(
        axes,
        [cnn_labels, fno_labels],
        [cnn_preds, fno_preds],
        ["CNN Baseline (Common Test I)", "FNO Classifier"],
    ):
        cm = confusion_matrix(labels, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Confusion-matrix plot saved to {out_path}")


def _plot_auc_comparison(
    cnn_auc: float,
    fno_auc: float,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    models = ["CNN Baseline\n(Common Test I)", "FNO Classifier"]
    aucs = [cnn_auc, fno_auc]
    colors = ["steelblue", "darkorange"]
    bars = ax.bar(models, aucs, color=colors, width=0.4)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Macro AUC (OvR)")
    ax.set_title("Test AUC: CNN vs. FNO")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"AUC comparison plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    # Device.
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)

    # Data.
    data_root = None if args.synthetic else args.data
    train_loader, val_loader, test_loader = get_dataloaders(
        root=data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        n_synthetic=args.n_synthetic,
        seed=args.seed,
        augment=True,
    )
    class_names = train_loader.dataset.subset.dataset.class_names

    print(
        f"Data splits: train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} "
        f"test={len(test_loader.dataset)}"
    )

    # ------------------------------------------------------------------
    # CNN Baseline
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CNN Baseline (Common Test I)")
    print("=" * 60)
    cnn_model = CNNBaseline(
        num_classes=args.num_classes, in_channels=args.in_channels
    )
    if args.cnn_checkpoint:
        cnn_model = _load_checkpoint(
            cnn_model, args.cnn_checkpoint, torch.device(device)
        )
        cnn_history: dict = {"val_loss": [], "val_acc": [], "val_auc": []}
    else:
        cnn_model, cnn_history = _train_model(
            cnn_model, train_loader, val_loader,
            device, args.lr, args.epochs, args.patience,
        )

    cnn_labels, cnn_preds, cnn_probs = _collect_predictions(
        cnn_model.to(device), test_loader, torch.device(device)
    )
    cnn_auc = _compute_auc(cnn_labels, cnn_probs)
    cnn_acc = float((cnn_preds == cnn_labels).mean())
    print(f"\nCNN Test – Accuracy: {cnn_acc:.4f} | AUC: {cnn_auc:.4f}")
    print(classification_report(cnn_labels, cnn_preds, target_names=class_names))

    # ------------------------------------------------------------------
    # FNO Classifier
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FNO Classifier")
    print("=" * 60)
    fno_model = FNOClassifier(
        num_classes=args.num_classes, in_channels=args.in_channels
    )
    if args.fno_checkpoint:
        fno_model = _load_checkpoint(
            fno_model, args.fno_checkpoint, torch.device(device)
        )
        fno_history: dict = {"val_loss": [], "val_acc": [], "val_auc": []}
    else:
        fno_model, fno_history = _train_model(
            fno_model, train_loader, val_loader,
            device, args.lr, args.epochs, args.patience,
        )

    fno_labels, fno_preds, fno_probs = _collect_predictions(
        fno_model.to(device), test_loader, torch.device(device)
    )
    fno_auc = _compute_auc(fno_labels, fno_probs)
    fno_acc = float((fno_preds == fno_labels).mean())
    print(f"\nFNO Test – Accuracy: {fno_acc:.4f} | AUC: {fno_auc:.4f}")
    print(classification_report(fno_labels, fno_preds, target_names=class_names))

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"{'Model':<25} {'Accuracy':>10} {'AUC':>10}")
    print("-" * 45)
    print(f"{'CNN Baseline (Test I)':<25} {cnn_acc:>10.4f} {cnn_auc:>10.4f}")
    print(f"{'FNO Classifier':<25} {fno_acc:>10.4f} {fno_auc:>10.4f}")
    diff_acc = fno_acc - cnn_acc
    diff_auc = fno_auc - cnn_auc
    sign_acc = "+" if diff_acc >= 0 else ""
    sign_auc = "+" if diff_auc >= 0 else ""
    print(f"{'FNO vs. CNN (delta)':<25} {sign_acc}{diff_acc:>9.4f} {sign_auc}{diff_auc:>9.4f}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    results = {
        "cnn": {"accuracy": cnn_acc, "auc": cnn_auc},
        "fno": {"accuracy": fno_acc, "auc": fno_auc},
    }
    results_path = os.path.join(args.output_dir, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if cnn_history["val_loss"] and fno_history["val_loss"]:
        _plot_learning_curves(
            cnn_history,
            fno_history,
            os.path.join(args.output_dir, "learning_curves.png"),
        )

    _plot_confusion_matrices(
        cnn_labels, cnn_preds,
        fno_labels, fno_preds,
        class_names,
        os.path.join(args.output_dir, "confusion_matrices.png"),
    )

    _plot_auc_comparison(
        cnn_auc,
        fno_auc,
        os.path.join(args.output_dir, "auc_comparison.png"),
    )


if __name__ == "__main__":
    main()
