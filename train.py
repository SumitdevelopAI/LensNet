"""
train.py – Train either the CNN baseline or the FNO classifier.

Usage examples
--------------
Train FNO model with synthetic data (quick smoke-test):
    python train.py --model fno --epochs 5 --synthetic

Train CNN baseline on a real dataset:
    python train.py --model cnn --data /path/to/data --epochs 30

Train FNO on a real dataset, save checkpoint:
    python train.py --model fno --data /path/to/data --epochs 30 \\
        --save checkpoints/fno_best.pt
"""

import argparse
import json
import os
import sys

import torch

from lensnet.data import get_dataloaders
from lensnet.models import CNNBaseline, FNOClassifier
from lensnet.training import Trainer


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a LensNet model.")
    p.add_argument(
        "--model",
        choices=["cnn", "fno"],
        default="fno",
        help="Model architecture to train (default: fno).",
    )
    p.add_argument(
        "--data",
        default=None,
        metavar="DIR",
        help="Path to the dataset root directory.  If omitted, synthetic data "
        "is used.",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=64,
        metavar="N",
        help="Spatial size to resize images to (default: 64).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="Mini-batch size (default: 32).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=30,
        metavar="N",
        help="Maximum number of training epochs (default: 30).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="F",
        help="Initial learning rate (default: 1e-3).",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=10,
        metavar="N",
        help="Early-stopping patience in epochs (default: 10).",
    )
    p.add_argument(
        "--num-classes",
        type=int,
        default=3,
        metavar="N",
        help="Number of output classes (default: 3).",
    )
    p.add_argument(
        "--in-channels",
        type=int,
        default=1,
        metavar="N",
        help="Number of image channels (default: 1).",
    )
    p.add_argument(
        "--hidden-channels",
        type=int,
        default=64,
        metavar="N",
        help="[FNO only] Hidden channel width (default: 64).",
    )
    p.add_argument(
        "--modes",
        type=int,
        default=12,
        metavar="N",
        help="[FNO only] Fourier modes to keep along each dimension "
        "(default: 12).",
    )
    p.add_argument(
        "--n-fno-blocks",
        type=int,
        default=4,
        metavar="N",
        help="[FNO only] Number of FNO residual blocks (default: 4).",
    )
    p.add_argument(
        "--device",
        default=None,
        help='Compute device, e.g. "cpu" or "cuda:0" (auto-detected if '
        "omitted).",
    )
    p.add_argument(
        "--save",
        default=None,
        metavar="PATH",
        help="Save the trained model checkpoint to this path.",
    )
    p.add_argument(
        "--history",
        default=None,
        metavar="PATH",
        help="Save training history as JSON to this path.",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Force use of synthetic data even if --data is provided.",
    )
    p.add_argument(
        "--n-synthetic",
        type=int,
        default=600,
        metavar="N",
        help="Number of synthetic samples (default: 600).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="Random seed (default: 42).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_root = None if args.synthetic else args.data
    train_loader, val_loader, test_loader = get_dataloaders(
        root=data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        n_synthetic=args.n_synthetic,
        seed=args.seed,
        augment=True,
    )
    print(
        f"Data splits: train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} "
        f"test={len(test_loader.dataset)}"
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if args.model == "cnn":
        model = CNNBaseline(
            num_classes=args.num_classes,
            in_channels=args.in_channels,
        )
        print(f"Model: CNNBaseline (ResNet-18 backbone)")
    else:
        model = FNOClassifier(
            num_classes=args.num_classes,
            in_channels=args.in_channels,
            hidden_channels=args.hidden_channels,
            modes1=args.modes,
            modes2=args.modes,
            n_fno_blocks=args.n_fno_blocks,
        )
        print(
            f"Model: FNOClassifier "
            f"(hidden={args.hidden_channels}, "
            f"modes={args.modes}, "
            f"blocks={args.n_fno_blocks})"
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        device=device,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
    )
    history = trainer.fit(train_loader, val_loader)

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    test_metrics = trainer.evaluate(test_loader)
    print(
        f"\nTest results: "
        f"loss={test_metrics['loss']:.4f}  "
        f"accuracy={test_metrics['accuracy']:.4f}  "
        f"AUC={test_metrics['auc']:.4f}"
    )

    # ------------------------------------------------------------------
    # Persist outputs
    # ------------------------------------------------------------------
    if args.save is not None:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "test_metrics": test_metrics,
            },
            args.save,
        )
        print(f"Checkpoint saved to {args.save}")

    if args.history is not None:
        os.makedirs(os.path.dirname(args.history) or ".", exist_ok=True)
        with open(args.history, "w") as f:
            json.dump({"history": history, "test_metrics": test_metrics}, f, indent=2)
        print(f"History saved to {args.history}")


if __name__ == "__main__":
    main()
