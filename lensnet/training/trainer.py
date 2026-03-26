"""
Training and evaluation loop for LensNet models.

The ``Trainer`` class provides a self-contained training loop with:
* Cross-entropy loss.
* Configurable optimiser (Adam by default).
* Cosine-annealing learning-rate scheduler.
* Per-epoch metric tracking (loss, accuracy, AUC).
* Optional early stopping.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score


class Trainer:
    """Generic trainer for lens classification models.

    Args:
        model:        PyTorch model to train.
        device:       Device string or ``torch.device`` (default ``"cpu"``).
        lr:           Initial learning rate (default ``1e-3``).
        weight_decay: L2 regularisation coefficient (default ``1e-4``).
        epochs:       Maximum number of training epochs (default ``30``).
        patience:     Early-stopping patience in epochs.  ``None`` disables
                      early stopping (default ``10``).
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        patience: Optional[int] = 10,
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.epochs = epochs
        self.patience = patience

        self.criterion = nn.CrossEntropyLoss()
        self.optimiser = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimiser, T_max=epochs
        )

        # History stored as lists of per-epoch scalars.
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_auc": [],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, list[float]]:
        """Train the model.

        Args:
            train_loader: DataLoader for training data.
            val_loader:   DataLoader for validation data.

        Returns:
            ``self.history`` dictionary.
        """
        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            t0 = time.perf_counter()
            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc, val_auc = self._eval_epoch(val_loader)
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_auc"].append(val_auc)

            elapsed = time.perf_counter() - t0
            print(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} auc {val_auc:.4f} | "
                f"{elapsed:.1f}s"
            )

            # Early stopping.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
                if self.patience is not None and no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """Evaluate the model on a data loader and return metrics.

        Args:
            loader: DataLoader to evaluate on.

        Returns:
            Dictionary with keys ``loss``, ``accuracy``, ``auc``.
        """
        loss, acc, auc = self._eval_epoch(loader)
        return {"loss": loss, "accuracy": acc, "auc": auc}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self, loader: DataLoader
    ) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in loader:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            self.optimiser.zero_grad()
            logits = self.model(imgs)
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimiser.step()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        return total_loss / total, correct / total

    def _eval_epoch(
        self, loader: DataLoader
    ) -> tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_labels: list[int] = []
        all_probs: list[np.ndarray] = []

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(imgs)
                loss = self.criterion(logits, labels)

                total_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / total
        accuracy = correct / total

        # Compute macro-averaged ROC-AUC (one-vs-rest).
        all_probs_arr = np.concatenate(all_probs, axis=0)
        all_labels_arr = np.array(all_labels)
        n_classes = all_probs_arr.shape[1]
        try:
            if n_classes == 2:
                auc = roc_auc_score(all_labels_arr, all_probs_arr[:, 1])
            else:
                auc = roc_auc_score(
                    all_labels_arr,
                    all_probs_arr,
                    multi_class="ovr",
                    average="macro",
                )
        except ValueError:
            auc = float("nan")

        return avg_loss, accuracy, auc
