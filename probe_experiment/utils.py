"""
Utility functions: channel mapping, permutation test, set_seed.
"""

import numpy as np
import torch
from typing import List, Tuple

# ──────────────────────────────────────────────────────────────────────
# LaBraM's canonical 10-20 channel list (from utils.py standard_1020).
# The 128 learnable spatial position embeddings follow this ordering.
# ──────────────────────────────────────────────────────────────────────
STANDARD_1020 = [
    'FP1', 'FPZ', 'FP2',
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2',
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h',
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]

NUM_SPATIAL_POSITIONS = 128  # pos_embed has 1 CLS + 128 spatial


def build_input_chans(ch_names: List[str]) -> List[int]:
    """
    Map THINGS-EEG channel names to LaBraM position-embedding indices.

    Args:
        ch_names: list of 63 channel names, e.g. ['Fp1', 'Fp2', ...]

    Returns:
        List of 64 integers.  Index 0 is the CLS token; indices 1..63
        are the spatial position-embedding slots (1-based within pos_embed).
    """
    indices = [0]  # CLS token always at pos 0
    for ch in ch_names:
        try:
            pos = STANDARD_1020.index(ch.upper())
        except ValueError:
            raise ValueError(
                f"Channel '{ch}' not found in STANDARD_1020 list. "
                f"Cannot build position embedding index."
            )
        if pos >= NUM_SPATIAL_POSITIONS:
            raise ValueError(
                f"Channel '{ch}' maps to STANDARD_1020 index {pos}, "
                f"but pos_embed only has {NUM_SPATIAL_POSITIONS} spatial slots."
            )
        indices.append(pos + 1)  # +1: CLS occupies index 0 in pos_embed
    return indices


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def permutation_test(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
    observed_acc: float = None,
) -> dict:
    """
    Permutation test: shuffle labels to build null distribution of Top-1 accuracy.

    Uses sklearn LogisticRegression (fast) for the null distribution.
    If `observed_acc` is provided, use it directly; otherwise fit on real labels.

    Returns dict with:
        p_value: fraction of permuted accuracies >= observed accuracy
        observed_acc: accuracy on real labels (fraction, not %)
        null_mean: mean of null distribution
        null_std:  std of null distribution
        null_distribution: list of n_permutations accuracy values
        n_permutations: number of permutations
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    rng = np.random.RandomState(seed)
    n_classes = len(np.unique(train_y))

    # Observed accuracy
    if observed_acc is None:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                C=1.0, max_iter=5000, multi_class='multinomial',
                solver='lbfgs', random_state=seed,
            )),
        ])
        pipe.fit(train_X, train_y)
        observed_acc = (pipe.predict(test_X) == test_y).mean()
    else:
        # Use the caller-supplied accuracy (e.g., from PyTorch probe)
        pass

    # Null distribution — use sklearn for speed
    null_accs = []
    for _ in range(n_permutations):
        y_shuffled = rng.permutation(train_y)
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                C=1.0, max_iter=5000, multi_class='multinomial',
                solver='lbfgs', random_state=seed,
            )),
        ])
        pipe.fit(train_X, y_shuffled)
        null_accs.append((pipe.predict(test_X) == test_y).mean())

    null_accs = np.array(null_accs)
    p_value = (null_accs >= observed_acc).mean()

    return {
        "p_value": float(p_value),
        "observed_acc": float(observed_acc),
        "null_mean": float(null_accs.mean()),
        "null_std": float(null_accs.std()),
        "null_distribution": null_accs.tolist(),
        "n_permutations": n_permutations,
        "n_classes": n_classes,
    }
