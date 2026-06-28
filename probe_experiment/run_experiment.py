#!/usr/bin/env python3
"""
LaBraM Linear Probe Experiment — Main Entry Point.

Usage:
    # Main experiment (frozen LaBraM, CLS token features)
    python run_experiment.py

    # Ablation: random-init LaBraM
    python run_experiment.py --random_init

    # Ablation: intermediate layer features
    python run_experiment.py --extract_layer 6

    # Use sklearn instead of PyTorch for the classifier
    python run_experiment.py --classifier sklearn

    # All-layer sweep (writes per-layer results)
    python run_experiment.py --sweep_layers
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

from config import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description="LaBraM linear probe on THINGS-EEG 2"
    )
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data directory")
    parser.add_argument("--subs", type=str, default="all",
                        help="Subject indices: 'all' for 1-10, or '1,3,5' for specific")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to labram-base.pth")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)

    # Experiment variants
    parser.add_argument("--random_init", action="store_true",
                        help="Use randomly initialised LaBraM (ablation)")
    parser.add_argument("--extract_layer", type=int, default=None,
                        help="Extract features from this transformer layer (0-11)")
    parser.add_argument("--sweep_layers", action="store_true",
                        help="Run probe on all 12 layers sequentially")
    parser.add_argument("--classifier", type=str, default="pytorch",
                        choices=["pytorch", "sklearn"])
    parser.add_argument("--skip_permutation", action="store_true",
                        help="Skip permutation test (faster)")

    # Hyperparameter overrides
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)

    return parser.parse_args()


def build_config(args) -> Config:
    """Merge CLI args into Config."""
    cfg = Config()
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.subs is not None:
        if args.subs == "all":
            cfg.subs = list(range(1, 11))  # subjects 1..10
        else:
            cfg.subs = [int(s.strip()) for s in args.subs.split(",")]
    if args.checkpoint is not None:
        cfg.checkpoint_path = args.checkpoint
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.device is not None:
        cfg.device = args.device
    if args.lr is not None:
        cfg.lr = args.lr
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    cfg.use_random_init = args.random_init
    cfg.extract_layer = args.extract_layer
    cfg.classifier = args.classifier
    return cfg


def run_single_experiment(cfg: Config, layer: int = None) -> dict:
    """
    Run one probe experiment.

    Args:
        cfg: configuration
        layer: if not None, override cfg.extract_layer for this run

    Returns:
        results dict
    """
    from data_loader import (
        load_test_data, split_by_trials, to_la_bram_format, make_dataloaders,
    )
    from model_loader import load_labram_student, extract_features
    from linear_probe import (
        train_probe, evaluate_probe_pytorch,
        train_probe_sklearn, evaluate_probe_sklearn,
    )
    from utils import build_input_chans, set_seed, permutation_test

    set_seed(cfg.seed)

    extract_layer = layer if layer is not None else cfg.extract_layer
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("LaBraM Linear Probe Experiment")
    print("=" * 60)
    subs_list = cfg.subs if cfg.subs else list(range(1, 11))
    print(f"  Subjects:         {subs_list} ({len(subs_list)} total)")
    print(f"  Data dir:         {cfg.data_dir}")
    print(f"  Checkpoint:       {cfg.checkpoint_path}")
    print(f"  Random init:      {cfg.use_random_init}")
    print(f"  Extract layer:    {extract_layer}")
    print(f"  Classifier:       {cfg.classifier}")
    print(f"  Device:           {device}")
    print(f"  Output dir:       {cfg.output_dir}")
    print()

    # ── 1. Load data ──────────────────────────────────────────────
    print("1. Loading test data...")
    X, y_cond, ch_names = load_test_data(
        cfg.data_dir, subs_list, target_tp=cfg.target_tp,
    )
    print(f"   X: {X.shape}  y: {y_cond.shape}  channels: {len(ch_names)}")

    # ── 2. Split ──────────────────────────────────────────────────
    print("2. Splitting data...")
    train_X_raw, train_y, val_X_raw, val_y, test_X_raw, test_y = split_by_trials(
        X, y_cond,
        n_train=cfg.train_trials_per_class,
        n_val=cfg.val_trials_per_class,
        seed=cfg.seed,
    )
    print(f"   Train: {train_X_raw.shape[0]} samples")
    print(f"   Val:   {val_X_raw.shape[0]} samples")
    print(f"   Test:  {test_X_raw.shape[0]} samples")

    # ── 3. Channel mapping ────────────────────────────────────────
    channel_indices = build_input_chans(ch_names)
    print(f"   Channel indices: {len(channel_indices)} (1 CLS + 63 channels)")

    # ── 4. Load LaBraM ────────────────────────────────────────────
    print("3. Loading LaBraM student...")
    student = load_labram_student(
        cfg.checkpoint_path, device, random_init=cfg.use_random_init,
    )
    n_params = sum(p.numel() for p in student.parameters())
    print(f"   Student parameters: {n_params:,} (all frozen)")

    # ── 5. Feature extraction ─────────────────────────────────────
    # Build a compact identifier for the subject set
    if len(subs_list) == 10:
        sub_tag = "all10"
    elif len(subs_list) == 1:
        sub_tag = f"sub-{subs_list[0]:02d}"
    else:
        sub_tag = f"subs-{'_'.join(str(s) for s in subs_list)}"

    feature_cache_dir = os.path.join(
        cfg.output_dir, sub_tag, "features"
    )
    os.makedirs(feature_cache_dir, exist_ok=True)

    layer_suffix = f"_layer{extract_layer}" if extract_layer is not None else ""
    init_suffix = "_randinit" if cfg.use_random_init else ""

    feat_train_path = os.path.join(
        feature_cache_dir, f"train_features{layer_suffix}{init_suffix}.npy"
    )
    feat_val_path = os.path.join(
        feature_cache_dir, f"val_features{layer_suffix}{init_suffix}.npy"
    )
    feat_test_path = os.path.join(
        feature_cache_dir, f"test_features{layer_suffix}{init_suffix}.npy"
    )

    if cfg.cache_features and os.path.exists(feat_train_path):
        print("4. Loading cached features...")
        train_feat = np.load(feat_train_path)
        val_feat = np.load(feat_val_path)
        test_feat = np.load(feat_test_path)
        # Labels are invariant
        np.save(os.path.join(feature_cache_dir, "train_labels.npy"), train_y)
        np.save(os.path.join(feature_cache_dir, "val_labels.npy"), val_y)
        np.save(os.path.join(feature_cache_dir, "test_labels.npy"), test_y)
    else:
        print("4. Extracting features...")

        # Build DataLoaders for the *raw* EEG (un-tokenized), then extract
        train_loader_raw, val_loader_raw, test_loader_raw = make_dataloaders(
            train_X_raw, train_y, val_X_raw, val_y, test_X_raw, test_y,
            batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        )

        t0 = time.time()
        train_feat, train_y2 = extract_features(
            student, train_loader_raw, channel_indices, device,
            extract_layer=extract_layer,
        )
        val_feat, val_y2 = extract_features(
            student, val_loader_raw, channel_indices, device,
            extract_layer=extract_layer,
        )
        test_feat, test_y2 = extract_features(
            student, test_loader_raw, channel_indices, device,
            extract_layer=extract_layer,
        )
        elapsed = time.time() - t0
        print(f"   Feature extraction took {elapsed:.1f}s")
        print(f"   Train features: {train_feat.shape}")
        print(f"   Val features:   {val_feat.shape}")
        print(f"   Test features:  {test_feat.shape}")

        if cfg.cache_features:
            np.save(feat_train_path, train_feat)
            np.save(feat_val_path, val_feat)
            np.save(feat_test_path, test_feat)
            np.save(os.path.join(feature_cache_dir, "train_labels.npy"), train_y)
            np.save(os.path.join(feature_cache_dir, "val_labels.npy"), val_y)
            np.save(os.path.join(feature_cache_dir, "test_labels.npy"), test_y)
            print(f"   Features cached to {feature_cache_dir}")

    # ── 6. Train linear probe ─────────────────────────────────────
    print("5. Training linear probe...")

    if cfg.classifier == "sklearn":
        model = train_probe_sklearn(
            train_feat, train_y, C=cfg.sklearn_C, seed=cfg.seed,
        )
        results = evaluate_probe_sklearn(model, test_feat, test_y)
        val_results = evaluate_probe_sklearn(model, val_feat, val_y)
    else:
        model, norm_stats, history = train_probe(
            train_feat, train_y, val_feat, val_y,
            input_dim=cfg.embed_dim,
            num_classes=cfg.n_classes,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            early_stop_patience=cfg.early_stop_patience,
            device=device,
            num_workers=cfg.num_workers,
        )
        results = evaluate_probe_pytorch(
            model, test_feat, test_y, norm_stats,
            batch_size=cfg.batch_size, device=device,
        )
        val_results = evaluate_probe_pytorch(
            model, val_feat, val_y, norm_stats,
            batch_size=cfg.batch_size, device=device,
        )
        results["history"] = history

    # ── 7. Permutation test ───────────────────────────────────────
    if not args.skip_permutation:
        print("6. Permutation test (1000 shuffles)...")
        t0 = time.time()
        # Use the actual probe's test accuracy for the observed value
        probe_acc_fraction = results["top1_acc"] / 100.0
        perm_results = permutation_test(
            train_feat, train_y, test_feat, test_y,
            n_permutations=1000, seed=cfg.seed,
            observed_acc=probe_acc_fraction,
        )
        results["permutation_test"] = perm_results
        print(f"   p = {perm_results['p_value']:.4f}  "
              f"(observed = {perm_results['observed_acc']*100:.2f}%, "
              f"null = {perm_results['null_mean']*100:.2f}% "
              f"± {perm_results['null_std']*100:.2f}%)")
        print(f"   Permutation test took {time.time() - t0:.1f}s")

    # ── 8. Confusion matrix data ──────────────────────────────────
    # Save per-class accuracy breakdown (useful for analysis)
    preds = results.get("predictions")
    labels = results.get("labels")
    if preds is not None and labels is not None:
        n_cls = cfg.n_classes
        per_class_acc = {}
        for c in range(n_cls):
            mask = labels == c
            if mask.sum() > 0:
                per_class_acc[int(c)] = float((preds[mask] == c).mean() * 100)
        results["per_class_acc"] = per_class_acc

    # ── 9. Report ─────────────────────────────────────────────────
    chance_level = 1.0 / cfg.n_classes * 100
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Chance level:     {chance_level:.2f}%")
    print(f"  Test  Top-1:      {results['top1_acc']:.2f}%")
    print(f"  Test  Top-5:      {results.get('top5_acc', float('nan')):.2f}%")
    print(f"  Test  Balanced:   {results['balanced_acc']:.2f}%")
    print(f"  Val   Top-1:      {val_results['top1_acc']:.2f}%")

    results["config"] = {
        "subs": subs_list,
        "n_subs": len(subs_list),
        "n_classes": cfg.n_classes,
        "train_trials_per_class": cfg.train_trials_per_class,
        "val_trials_per_class": cfg.val_trials_per_class,
        "random_init": cfg.use_random_init,
        "extract_layer": extract_layer,
        "classifier": cfg.classifier,
        "n_train": int(train_X_raw.shape[0]),
        "n_val": int(val_X_raw.shape[0]),
        "n_test": int(test_X_raw.shape[0]),
        "feature_dim": cfg.embed_dim,
        "chance_level_pct": chance_level,
    }
    results["val_results"] = val_results

    return results


def sweep_layers(cfg: Config) -> dict:
    """Run probe on all 12 transformer layers."""
    layer_results = {}
    for layer in range(12):
        print(f"\n{'='*60}")
        print(f"LAYER {layer}")
        print(f"{'='*60}")
        r = run_single_experiment(cfg, layer=layer)
        layer_results[f"layer_{layer}"] = {
            "top1_acc": r["top1_acc"],
            "top5_acc": r.get("top5_acc"),
            "balanced_acc": r["balanced_acc"],
        }
    return layer_results


def save_results(results: dict, cfg: Config, suffix: str = ""):
    """Write results JSON."""
    # Use the subject tag from the results config
    subs_list = results.get("config", {}).get("subs", [1])
    if len(subs_list) == 10:
        sub_tag = "all10"
    elif len(subs_list) == 1:
        sub_tag = f"sub-{subs_list[0]:02d}"
    else:
        sub_tag = f"subs-{'_'.join(str(s) for s in subs_list)}"

    results_dir = os.path.join(cfg.output_dir, sub_tag)
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"results{suffix}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_json_serializable)
    print(f"\nResults saved to {path}")


def _json_serializable(obj):
    """Fallback for JSON encoding."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return str(obj)


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = parse_args()
    cfg = build_config(args)

    if args.sweep_layers:
        layer_results = sweep_layers(cfg)
        save_results(layer_results, cfg, suffix="_layer_sweep")
    else:
        results = run_single_experiment(cfg)
        save_results(results, cfg)
