"""
Configuration for LaBraM linear probe experiment.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────────────
    # Server data root
    data_dir: str = "/root/autodl-tmp/BCI/things-eeg2-data/Preprocessed_data_200Hz"
    subs: list = None                # None → all 10 subjects [1..10]; or e.g. [1,2,3]
    n_ses: int = 4                   # number of sessions (per subject)

    # Preprocessing parameters (must match preprocessing.py invocation)
    sfreq: int = 200                 # downsampling frequency
    original_tp: int = 190           # time points after preprocessing (slice [50:])
    target_tp: int = 200             # time points for LaBraM (zero-pad 10)
    n_channels: int = 63             # THINGS-EEG channels
    n_classes: int = 200             # test-set image conditions

    # Use test set (200 classes × 80 reps × N_subs) as the probe dataset.
    use_training_data: bool = False   # set True to also load the 16540-class training set

    # Train / val / test split (per-class trial counts; scaled by n_subs)
    # For example, 10 subs = 800 trials/class → recommended: 500/100/200
    train_trials_per_class: int = 500
    val_trials_per_class: int = 100
    # remaining go to held-out test

    # ── LaBraM ────────────────────────────────────────────────────────
    checkpoint_path: str = "./checkpoints/labram-base.pth"
    embed_dim: int = 200
    patch_size: int = 200

    # ── Linear probe ──────────────────────────────────────────────────
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 128
    weight_decay: float = 1e-4
    early_stop_patience: int = 20

    # sklearn fallback
    sklearn_C: float = 1.0           # inverse L2 strength for LogisticRegressionCV

    # ── Output ────────────────────────────────────────────────────────
    output_dir: str = "./probe_experiment/outputs"
    cache_features: bool = True      # save extracted features to disk

    # ── Classifier ────────────────────────────────────────────────────
    classifier: str = "pytorch"      # "pytorch" or "sklearn"

    # ── Hardware ──────────────────────────────────────────────────────
    device: str = "cuda"
    num_workers: int = 4

    # ── Reproducibility ───────────────────────────────────────────────
    seed: int = 42

    # ── Ablation flags ────────────────────────────────────────────────
    use_random_init: bool = False    # skip pretrained weights → random LaBraM
    extract_layer: Optional[int] = None  # if set, extract from this transformer layer
    # (None = use final CLS token)
