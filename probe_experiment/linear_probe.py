"""
Linear probe: single linear layer + CrossEntropyLoss trained with AdamW.

Also provides a sklearn LogisticRegression baseline.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Optional
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════
# PyTorch linear probe
# ══════════════════════════════════════════════════════════════════════

class LinearProbe(nn.Module):
    """Single linear layer for classification."""

    def __init__(self, input_dim: int = 200, num_classes: int = 200):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train_probe(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    input_dim: int = 200,
    num_classes: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 200,
    batch_size: int = 128,
    early_stop_patience: int = 20,
    device: torch.device = None,
    num_workers: int = 4,
    verbose: bool = True,
) -> Tuple[LinearProbe, dict]:
    """
    Train a linear probe with AdamW and early stopping.

    Returns:
        model: best checkpoint (on validation accuracy)
        history: dict with keys 'train_loss', 'val_acc', 'best_epoch'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalise features (z-score per dimension, fitted on train)
    mean = train_X.mean(axis=0, keepdims=True)
    std = train_X.std(axis=0, keepdims=True) + 1e-8
    train_X_norm = (train_X - mean) / std
    val_X_norm = (val_X - mean) / std

    # DataLoaders
    train_dataset = TensorDataset(
        torch.from_numpy(train_X_norm).float(),
        torch.from_numpy(train_y).long(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_X_norm).float(),
        torch.from_numpy(val_y).long(),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    model = LinearProbe(input_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_acc": [], "best_epoch": 0}

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(train_dataset)
        history["train_loss"].append(avg_loss)

        # ── Validate ──
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / total * 100
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            history["best_epoch"] = epoch
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  "
                  f"loss={avg_loss:.4f}  val_acc={val_acc:.2f}%  "
                  f"best={best_val_acc:.2f}%")

        if patience_counter >= early_stop_patience:
            if verbose:
                print(f"  Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model, {"mean": mean, "std": std}, history


@torch.no_grad()
def evaluate_probe_pytorch(
    model: LinearProbe,
    test_X: np.ndarray,
    test_y: np.ndarray,
    norm_stats: dict,
    batch_size: int = 128,
    device: torch.device = None,
    num_workers: int = 4,
) -> dict:
    """Evaluate the PyTorch probe on test data. Returns Top-1, Top-5, Balanced Acc."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_X_norm = (test_X - norm_stats["mean"]) / (norm_stats["std"] + 1e-8)
    test_dataset = TensorDataset(
        torch.from_numpy(test_X_norm).float(),
        torch.from_numpy(test_y).long(),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    model.eval()
    all_logits = []
    all_labels = []
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(y.numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = all_logits.argmax(axis=1)

    top1 = (all_preds == all_labels).mean() * 100
    n_classes = all_logits.shape[1]
    top5 = top_k_accuracy_score_np(all_labels, all_logits, k=min(5, n_classes)) * 100
    balanced_acc = balanced_accuracy_score_np(all_labels, all_preds) * 100

    return {
        "top1_acc": float(top1),
        "top5_acc": float(top5),
        "balanced_acc": float(balanced_acc),
        "predictions": all_preds,
        "labels": all_labels,
    }


# ══════════════════════════════════════════════════════════════════════
# sklearn baseline (fast, no GPU needed for fitting)
# ══════════════════════════════════════════════════════════════════════

def train_probe_sklearn(
    train_X: np.ndarray,
    train_y: np.ndarray,
    C: float = 1.0,
    seed: int = 42,
) -> object:
    """Train sklearn LogisticRegression (multinomial)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            C=C,
            max_iter=5000,
            multi_class='multinomial',
            solver='lbfgs',
            random_state=seed,
        )),
    ])
    pipe.fit(train_X, train_y)
    return pipe


def evaluate_probe_sklearn(
    pipe,
    test_X: np.ndarray,
    test_y: np.ndarray,
) -> dict:
    """Evaluate sklearn probe."""
    preds = pipe.predict(test_X)
    probs = pipe.predict_proba(test_X)

    top1 = (preds == test_y).mean() * 100
    top5 = top_k_accuracy_score_np(test_y, probs, k=min(5, probs.shape[1])) * 100
    balanced_acc = balanced_accuracy_score_np(test_y, preds) * 100

    return {
        "top1_acc": float(top1),
        "top5_acc": float(top5),
        "balanced_acc": float(balanced_acc),
        "predictions": preds,
        "labels": test_y,
    }


# ══════════════════════════════════════════════════════════════════════
# NumPy helpers (avoid sklearn dependency for simple metrics if desired)
# ══════════════════════════════════════════════════════════════════════

def top_k_accuracy_score_np(y_true: np.ndarray, y_score: np.ndarray, k: int = 5) -> float:
    """Top-k accuracy.  y_score: (N, C) with higher = more likely."""
    top_k_preds = np.argpartition(y_score, -k, axis=1)[:, -k:]
    return np.any(top_k_preds == y_true[:, None], axis=1).mean()


def balanced_accuracy_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced accuracy (average recall across classes)."""
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        mask = y_true == c
        if mask.sum() > 0:
            recalls.append((y_pred[mask] == c).mean())
    return np.mean(recalls) if recalls else 0.0
