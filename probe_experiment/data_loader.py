"""
Data loading, zero-padding, and stratified trial-level splitting.
Supports both single-subject and multi-subject loading.
"""

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, List, Optional, Union


def _load_npy_dict(path: str) -> dict:
    """Load a pickle-serialised .npy file (as saved by NICE-EEG preprocessing)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return np.load(path, allow_pickle=True)


def load_test_data(
    data_dir: str,
    subs: Union[int, List[int]],
    target_tp: int = 200,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load the THINGS-EEG 2 *test* set for one or more subjects.

    All subjects share the same 200 image conditions (labels are aligned),
    so data is concatenated along the trial (repetition) dimension.

    Args:
        data_dir: root path, e.g. /root/.../Preprocessed_data_200Hz/
        subs: single subject index or list of indices (1-based, e.g. [1,2,3,5])
        target_tp: desired time points (zero-pad if needed)

    Returns:
        X:  float32  (200, total_reps, 63, target_tp)
        y:  int64    (200,)  — condition ids 0..199
        ch_names: list of 63 str
    """
    if isinstance(subs, int):
        subs = [subs]

    X_list = []
    ch_names = None

    for sub in subs:
        path = os.path.join(data_dir, f"sub-{sub:02d}", "preprocessed_eeg_test.npy")
        data_dict = _load_npy_dict(path)

        X_sub = data_dict["preprocessed_eeg_data"]        # (200, 80, 63, 190)
        ch_names_sub = list(data_dict["ch_names"])

        if ch_names is None:
            ch_names = ch_names_sub
        else:
            assert ch_names == ch_names_sub, \
                f"Channel mismatch for sub-{sub:02d}"

        # Zero-pad time dimension
        pad_width = target_tp - X_sub.shape[-1]
        if pad_width > 0:
            X_sub = np.pad(
                X_sub,
                ((0, 0), (0, 0), (0, 0), (0, pad_width)),
                mode='constant',
            )
        elif pad_width < 0:
            X_sub = X_sub[..., :target_tp]

        X_list.append(X_sub.astype(np.float32))

    # Concatenate along trial axis: (200, 80*N, 63, Tp)
    X = np.concatenate(X_list, axis=1)
    y = np.arange(X.shape[0], dtype=np.int64)  # 0..199

    print(f"   Loaded {len(subs)} subject(s): {subs}")
    print(f"   X: {X.shape}  y: {y.shape}")
    return X, y, ch_names


def load_training_data(
    data_dir: str,
    subs: Union[int, List[int]],
    target_tp: int = 200,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load the THINGS-EEG 2 *training* set (16540 conditions × few reps).
    Supports multi-subject concatenation.
    """
    if isinstance(subs, int):
        subs = [subs]

    X_list = []
    ch_names = None

    for sub in subs:
        path = os.path.join(
            data_dir, f"sub-{sub:02d}", "preprocessed_eeg_training.npy"
        )
        data_dict = _load_npy_dict(path)

        X_sub = data_dict["preprocessed_eeg_data"]        # (16540, 4, 63, 190)
        ch_names_sub = list(data_dict["ch_names"])

        if ch_names is None:
            ch_names = ch_names_sub
        else:
            assert ch_names == ch_names_sub

        pad_width = target_tp - X_sub.shape[-1]
        if pad_width > 0:
            X_sub = np.pad(
                X_sub,
                ((0, 0), (0, 0), (0, 0), (0, pad_width)),
                mode='constant',
            )
        elif pad_width < 0:
            X_sub = X_sub[..., :target_tp]

        X_list.append(X_sub.astype(np.float32))

    X = np.concatenate(X_list, axis=1)
    y = np.arange(X.shape[0], dtype=np.int64)
    return X, y, ch_names


def split_by_trials(
    X: np.ndarray,
    y: np.ndarray,
    n_train: int = 50,
    n_val: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified per-class trial split.

    Args:
        X: (n_classes, n_reps, n_channels, n_tp)
        y: (n_classes,)  — class labels (0..C-1)
        n_train, n_val: per-class trial counts; remainder goes to test.

    Returns:
        train_X, train_y  — (total_train_samples, n_channels, n_tp), ...
        val_X,   val_y
        test_X,  test_y
    """
    n_classes, n_reps = X.shape[0], X.shape[1]
    assert n_train + n_val <= n_reps, \
        f"n_train ({n_train}) + n_val ({n_val}) > n_reps ({n_reps})"

    rng = np.random.RandomState(seed)
    train_list, val_list, test_list = [], [], []

    for c in range(n_classes):
        idx = rng.permutation(n_reps)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        train_list.append(X[c, train_idx])
        val_list.append(X[c, val_idx])
        test_list.append(X[c, test_idx])

    train_X = np.concatenate(train_list, axis=0)  # (total, Ch, Tp)
    val_X   = np.concatenate(val_list,   axis=0)
    test_X  = np.concatenate(test_list,  axis=0)

    train_y = np.repeat(y, n_train)
    val_y   = np.repeat(y, n_val)
    test_y  = np.repeat(y, n_reps - n_train - n_val)

    return train_X, train_y, val_X, val_y, test_X, test_y


def make_dataloaders(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders from numpy arrays."""

    train_dataset = TensorDataset(
        torch.from_numpy(train_X), torch.from_numpy(train_y).long()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_X), torch.from_numpy(val_y).long()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_X), torch.from_numpy(test_y).long()
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader
