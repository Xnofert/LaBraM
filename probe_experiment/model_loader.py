"""
Load the frozen LaBraM student encoder and extract CLS-token features.
"""

import sys
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


def _ensure_labram_on_path():
    """Add the LaBraM root to sys.path so we can import modeling_pretrain."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


def load_labram_student(
    checkpoint_path: str,
    device: torch.device,
    random_init: bool = False,
) -> nn.Module:
    """
    Load the LaBraM base model and return its *student* encoder (frozen).

    Args:
        checkpoint_path: path to labram-base.pth
        device: torch device
        random_init: if True, skip loading weights (ablation baseline)

    Returns:
        student: NeuralTransformerForMaskedEEGModeling, eval mode, frozen.
    """
    _ensure_labram_on_path()
    from modeling_pretrain import labram_base_patch200_1600_8k_vocab

    # Load checkpoint manually to control weights_only (needed for PyTorch >= 2.6)
    if not random_init:
        print(f"   Loading checkpoint from {checkpoint_path} ...")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False,
        )

    # init_values must match the pretrained checkpoint (0.1 for labram-base).
    # The model code has a bug where None causes TypeError in Block.__init__.
    model = labram_base_patch200_1600_8k_vocab(
        pretrained=False, init_values=0.1,
    )

    if not random_init:
        # The checkpoint may contain extra keys (e.g. "logit_scale") not
        # present in the current model definition.  Use strict=False.
        model.load_state_dict(checkpoint["model"], strict=False)
        print("   Checkpoint loaded successfully.")

    student = model.student
    student.eval()
    for p in student.parameters():
        p.requires_grad = False

    return student.to(device)


@torch.no_grad()
def extract_features(
    student: nn.Module,
    loader: DataLoader,
    channel_indices: List[int],
    device: torch.device,
    extract_layer: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract LaBraM features for all samples in `loader`.

    Args:
        student: frozen LaBraM student encoder
        loader: DataLoader yielding (X, y) where X = (B, Ch, Tp) — raw EEG
        channel_indices: list of 64 ints for pos_embed indexing
        device: torch device
        extract_layer: if set (0-11), extract output after that transformer block
                       instead of the final CLS token.

    Returns:
        features: (N, embed_dim) float32
        labels:   (N,) int64
    """
    if extract_layer is not None:
        return _extract_layer_features(
            student, loader, channel_indices, device, extract_layer
        )
    else:
        return _extract_cls_features(
            student, loader, channel_indices, device
        )


@torch.no_grad()
def _extract_cls_features(
    student: nn.Module,
    loader: DataLoader,
    channel_indices: List[int],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the final CLS token from the last transformer layer."""

    all_features = []
    all_labels = []

    # Move channel_indices to device once
    ch_idx = torch.tensor(channel_indices, dtype=torch.long, device=device)

    for x, y in tqdm(loader, desc="Extracting features"):
        x = x.to(device)                      # (B, Ch, Tp)
        x = x.unsqueeze(2)                    # (B, Ch, 1, Tp)

        # student.forward with return_all_patch_tokens=True returns
        # the full (CLS + patches) after layer norm.
        output = student.forward(
            x,
            input_chans=ch_idx,
            return_all_patch_tokens=True,
        )  # (B, 1 + n_tokens, embed_dim)

        cls_feat = output[:, 0, :]            # (B, embed_dim)
        all_features.append(cls_feat.cpu().numpy())
        all_labels.append(y.numpy())

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.int64)
    return features, labels


@torch.no_grad()
def _extract_layer_features(
    student: nn.Module,
    loader: DataLoader,
    channel_indices: List[int],
    device: torch.device,
    layer_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from an intermediate transformer layer.

    We manually run the forward pass up to layer `layer_id`, then
    take the CLS token from that layer's output.
    """
    all_features = []
    all_labels = []

    ch_idx = torch.tensor(channel_indices, dtype=torch.long, device=device)

    for x, y in tqdm(loader, desc=f"Extracting layer {layer_id}"):
        x = x.to(device)
        x = x.unsqueeze(2)  # (B, Ch, 1, Tp)

        # --- Manual forward: replicate forward_features up to layer_id ---
        batch_size, c, time_window, _ = x.shape

        # Patch embedding
        x = student.patch_embed(x)
        batch_size, seq_len, _ = x.shape

        # CLS token
        cls_tokens = student.cls_token.expand(batch_size, -1, -1)
        mask_token = student.mask_token.expand(batch_size, seq_len, -1)

        # All-zero mask → keep all tokens
        bool_masked_pos = torch.zeros(
            (batch_size, seq_len), dtype=torch.bool, device=device
        )
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)

        # Position & time embeddings
        if student.pos_embed is not None:
            pos_embed_used = student.pos_embed[:, ch_idx]  # (1, 64, D)
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2) \
                .expand(batch_size, -1, time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((
                pos_embed_used[:, 0:1, :].expand(batch_size, -1, -1),
                pos_embed,
            ), dim=1)
            x = x + pos_embed

        if student.time_embed is not None:
            time_embed = student.time_embed[:, 0:time_window, :] \
                .unsqueeze(1).expand(batch_size, c, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed

        x = student.pos_drop(x)

        # Run transformer blocks up to layer_id
        rel_pos_bias = student.rel_pos_bias() if student.rel_pos_bias is not None else None
        for i, blk in enumerate(student.blocks):
            x = blk(x, rel_pos_bias=rel_pos_bias)
            if i == layer_id:
                break

        # CLS token from this layer (no final norm for intermediate)
        cls_feat = x[:, 0, :]  # (B, D)
        all_features.append(cls_feat.cpu().numpy())
        all_labels.append(y.numpy())

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.int64)
    return features, labels
