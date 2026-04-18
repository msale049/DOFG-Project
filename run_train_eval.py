#!/usr/bin/env python3
"""
run_train_eval.py
=================
End-to-end training, evaluation, and stress test with persistent results.

Saves to results/run_YYYYMMDD_HHMMSS/:
  - config.json
  - training_history.json
  - training_curves.png, training_dynamics.png
  - eval_metrics.json
  - confusion_matrix.png, per_class_metrics.png
  - gate_distributions.png, gate_response_curves.png
  - attention_heatmap.png, per_subject_accuracy.png
  - stress_test_details.csv, stress_test_summary.csv
  - stress_test_heatmap.png, gating_on_vs_off.png
  - opacity_analysis.png, gates_vs_opacity.png
  - occlusion_visualization.png
  - model_best.pt

Usage
-----
    python run_train_eval.py --samples 20 --epochs 3 --stress-frames 15
    python run_train_eval.py --samples 0 --epochs 20
    python run_train_eval.py --mode fixed --num-test 3
    python run_train_eval.py --mode kfold --k 5
"""

import argparse
import gc
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('ORT_LOG_LEVEL', '3')  # suppress ONNX Runtime thread affinity warnings
os.environ.setdefault('ONNXRUNTIME_SESSION_THREAD_POOL_SIZE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from collections import Counter

from config import CONFIG, SEED
from data_loading import load_csv_video_data
from split_generator import create_splits, get_subject_ids
from pipeline import extract_features_stratified, extract_features_for_clips
from datasets import DriverStateDataset
from transformer_enhanced import EnhancedOcclusionAwareTransformer
from mlp_baseline import RegionFeatureMLP
from resnet_baseline import ResNet34Baseline
from trainer_enhanced import TinyTransformerTrainer
from evaluation import compute_metrics_on_loader, collect_eval_with_occlusion
from metrics_utils import (
    compute_classification_metrics,
    compute_classification_uncertainty,
    compute_paired_binary_statistics,
)
from stress_test import run_stress_test
from visualize_occlusion import (
    generate_occlusion_grid_png,
    generate_clean_vs_synthetic_opacity_grid,
)


# ─── GPU memory helpers ──────────────────────────────────────────────────────

def _gpu_memory_snapshot() -> str:
    """Return a compact snapshot of CUDA allocator state."""
    if not torch.cuda.is_available():
        return 'cuda unavailable'
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        return (
            f'free={free_bytes / 1e9:.2f} GB / total={total_bytes / 1e9:.2f} GB | '
            f'allocated={allocated / 1e6:.0f} MB | reserved={reserved / 1e6:.0f} MB'
        )
    except Exception as e:
        return f'cuda snapshot unavailable ({e})'


def _require_cuda(context: str) -> None:
    """Fail fast instead of silently falling back to CPU."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            f'CUDA is required for {context}, but torch.cuda.is_available() is False. '
            'This run would otherwise fall back to CPU.'
        )


def _gpu_cleanup(msg: str = ''):
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if msg:
        print(f'  [GPU cleanup] {msg} — {_gpu_memory_snapshot()}')


def _delete_models(*models):
    """Delete model objects and run GPU cleanup."""
    for m in models:
        if m is not None:
            del m
    _gpu_cleanup('models freed')


def _set_global_seed(seed: int):
    """Seed Python, NumPy, and PyTorch for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f'Global seed set to {seed} (deterministic mode enabled)')


def _seed_worker(worker_id: int):
    """Seed DataLoader workers from PyTorch's per-worker initial seed."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _parse_float_list(spec: str | None) -> list[float]:
    """Parse a comma-separated float list."""
    if not spec:
        return []
    values = []
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def _build_train_opacity_sampler(args) -> Dict:
    """
    Build the train-time opacity sampling config.

    Precedence:
      1. --train-opacity-values => discrete
      2. --train-opacity-range  => uniform
      3. otherwise              => selected mode (default legacy)
    """
    values = _parse_float_list(args.train_opacity_values)
    weights = _parse_float_list(args.train_opacity_weights)
    range_vals = _parse_float_list(args.train_opacity_range)

    for v in values:
        if not (0.0 < v <= 1.0):
            raise ValueError(f'train opacity values must lie in (0, 1]; got {v}')
    for v in range_vals:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f'train opacity range values must lie in [0, 1]; got {v}')

    if weights:
        if not values:
            raise ValueError('--train-opacity-weights requires --train-opacity-values')
        if len(weights) != len(values):
            raise ValueError('--train-opacity-weights must match --train-opacity-values length')
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError('--train-opacity-weights must be non-negative and sum to a positive value')

    if values:
        return {
            'mode': 'discrete',
            'values': values,
            'weights': weights or None,
        }

    if range_vals:
        if len(range_vals) != 2:
            raise ValueError('--train-opacity-range must be exactly "LOW,HIGH"')
        low, high = range_vals
        if low > high:
            raise ValueError('--train-opacity-range requires LOW <= HIGH')
        return {
            'mode': 'uniform',
            'low': low,
            'high': high,
        }

    return {'mode': args.train_opacity_mode}


# ─── Split / results helpers ─────────────────────────────────────────────────

def _splits_from_fixed_or_fold(csv_data, mode='fixed', k=5, num_test=5, seed=SEED, fold=0):
    """Build pipeline splits dict from create_splits output."""
    subjects = get_subject_ids(csv_data)
    split_list = create_splits(subjects, mode=mode, k=k, num_test=num_test, seed=seed)
    fold_idx = min(fold, len(split_list) - 1)
    cfg = split_list[fold_idx]
    train_subjects = set(cfg['train_subjects'])
    test_subjects = set(cfg['test_subjects'])

    train_keys = [k for k, v in csv_data.items() if v['subject'] in train_subjects]
    test_keys = [k for k, v in csv_data.items() if v['subject'] in test_subjects]

    return {
        'train': train_keys,
        'val': train_keys,
        'test': test_keys,
        'split_config': cfg,
    }


def _ensure_results_dir(results_root: str = 'results', run_name: str | None = None) -> str:
    """Create results/run_YYYYMMDD_HHMMSS (or custom name) and return path."""
    os.makedirs(results_root, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(results_root, run_name or f'run_{stamp}')
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _audit_subjects(csv_data: dict):
    """Print which subjects were found and their annotation counts."""
    subjects = {}
    for vk, meta in csv_data.items():
        subj = meta['subject']
        n_ann = len(meta.get('annotations', []))
        if subj not in subjects:
            subjects[subj] = {'videos': [], 'total_annotations': 0}
        subjects[subj]['videos'].append(vk)
        subjects[subj]['total_annotations'] += n_ann

    print(f'\n--- Subject audit: {len(subjects)} subjects found ---')
    for subj in sorted(subjects.keys()):
        info = subjects[subj]
        print(f'  {subj}: {info["total_annotations"]} annotations, '
              f'{len(info["videos"])} video(s)')
    return subjects


# ─── Visualization helpers ───────────────────────────────────────────────────

def _import_plt():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style('whitegrid')
        return plt, sns
    except ImportError:
        return None, None


def _save_training_curves(history: dict, run_dir: str):
    """Train vs Val: loss and accuracy on same axes."""
    plt, _ = _import_plt()
    if plt is None:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = np.arange(1, len(history.get('epoch_losses', [])) + 1)
    has_val = bool(history.get('val_epoch_losses'))
    if epochs.size > 0:
        ax1.plot(epochs, history['epoch_losses'], 'o-', lw=2, color='#2196F3', label='Train')
        if has_val:
            ax1.plot(epochs, history['val_epoch_losses'], 's--', lw=2, color='#FF5722', label='Val')
        ax1.set(xlabel='Epoch', ylabel='Total Loss', title='Train vs Val — Total Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs, history['accuracies'], 'o-', lw=2, color='#2196F3', label='Train')
        if has_val:
            ax2.plot(epochs, history['val_accuracies'], 's--', lw=2, color='#FF5722', label='Val')
        ax2.set(xlabel='Epoch', ylabel='Accuracy (%)', title='Train vs Val — Accuracy')
        ax2.legend()
        ax2.grid(True)
    fig.suptitle('Training Curves', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'training_curves.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


def _save_training_dynamics(history: dict, run_dir: str):
    """Train-only loss components, accuracy, and gate statistics."""
    plt, _ = _import_plt()
    if plt is None:
        return
    epochs = np.arange(1, len(history.get('epoch_losses', [])) + 1)
    if epochs.size == 0:
        return

    has_gates = bool(history.get('mean_eye_gate'))
    n_rows = 3 if has_gates else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4.5 * n_rows), sharex=True)

    axes[0].plot(epochs, history['epoch_losses'], 'o-', lw=2, color='#2196F3', label='Total Loss')
    axes[0].plot(epochs, history['classification_losses'], 's--', lw=1.5, color='#4CAF50', label='Classification Loss')
    axes[0].plot(epochs, history['gate_occ_losses'], '^--', lw=1.5, color='#9C27B0', label='Gate Alignment Loss')
    axes[0].set(ylabel='Loss', title='Training Dynamics — Loss Components')
    axes[0].legend(fontsize=9)
    axes[0].grid(True)

    axes[1].plot(epochs, history['accuracies'], 'o-', lw=2, color='#4CAF50')
    axes[1].set(ylabel='Accuracy (%)', title='Training Dynamics — Accuracy')
    axes[1].grid(True)

    if has_gates:
        axes[2].plot(epochs, history['mean_eye_gate'], 'o-', lw=2, color='#2196F3', label='Mean Eye Gate')
        axes[2].plot(epochs, history['mean_mouth_gate'], 's-', lw=2, color='#4CAF50', label='Mean Mouth Gate')
        axes[2].axhline(y=1.0, color='gray', ls=':', lw=1, label='Max (clean target)')
        axes[2].axhline(y=0.05, color='red', ls=':', lw=1, label='Floor (0.05)')
        axes[2].set(xlabel='Epoch', ylabel='Gate Value',
                    title='Gate Statistics Across Training')
        axes[2].legend(fontsize=9)
        axes[2].grid(True)
    else:
        axes[-1].set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'training_dynamics.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


def _save_val_loss_components(history: dict, run_dir: str):
    """Separate plot: validation loss components (classification + gate)."""
    plt, _ = _import_plt()
    if plt is None or not history.get('val_classification_losses'):
        return
    epochs = np.arange(1, len(history['val_classification_losses']) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history['val_epoch_losses'], 'o-', lw=2, color='#FF5722', label='Val Total Loss')
    ax.plot(epochs, history['val_classification_losses'], 's--', lw=2, color='#FF9800', label='Val Classification Loss')
    ax.plot(epochs, history['val_gate_occ_losses'], '^--', lw=2, color='#E91E63', label='Val Gate Alignment Loss')
    ax.set(xlabel='Epoch', ylabel='Loss', title='Validation Loss Components Across Training')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'val_loss_components.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


def _save_val_gate_statistics(history: dict, run_dir: str):
    """Separate plot: validation gate statistics across epochs."""
    plt, _ = _import_plt()
    if plt is None or not history.get('val_mean_eye_gate'):
        return
    epochs = np.arange(1, len(history['val_mean_eye_gate']) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history['val_mean_eye_gate'], 'o-', lw=2, color='#2196F3', label='Val Mean Eye Gate')
    ax.plot(epochs, history['val_mean_mouth_gate'], 's-', lw=2, color='#4CAF50', label='Val Mean Mouth Gate')
    ax.axhline(y=1.0, color='gray', ls=':', lw=1, label='Max (clean target)')
    ax.axhline(y=0.05, color='red', ls=':', lw=1, label='Floor (0.05)')
    ax.set(xlabel='Epoch', ylabel='Mean Gate Value',
           title='Validation Gate Statistics Across Training')
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'val_gate_statistics.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


def _save_per_class_stress_tables(details_csv: str, run_dir: str):
    """Generate per-class accuracy and gating benefit tables from stress details."""
    if not os.path.exists(details_csv):
        return
    df = pd.read_csv(details_csv)
    required = {'condition', 'regime', 'opacity', 'class_label',
                'correct_gating_on', 'correct_gating_off'}
    if not required.issubset(df.columns):
        return

    classes = sorted(df['class_label'].unique())
    rows_on, rows_off, rows_delta = [], [], []

    for (cond, regime, opacity), g in df.groupby(['condition', 'regime', 'opacity']):
        on_row = {'condition': cond, 'regime': regime, 'opacity': opacity}
        off_row = {'condition': cond, 'regime': regime, 'opacity': opacity}
        delta_row = {'condition': cond, 'regime': regime, 'opacity': opacity}
        for cls in classes:
            sub = g[g['class_label'] == cls]
            if len(sub) == 0:
                on_row[f'{cls}_acc'] = None
                off_row[f'{cls}_acc'] = None
                delta_row[f'{cls}_delta'] = None
            else:
                on  = sub['correct_gating_on'].mean() * 100
                off = sub['correct_gating_off'].mean() * 100
                on_row[f'{cls}_acc'] = round(on, 2)
                off_row[f'{cls}_acc'] = round(off, 2)
                delta_row[f'{cls}_delta'] = round(on - off, 2)
        rows_on.append(on_row)
        rows_off.append(off_row)
        rows_delta.append(delta_row)

    cond_order = ['clean'] + [c for c in df['condition'].unique() if c != 'clean']
    on_df = pd.DataFrame(rows_on)
    off_df = pd.DataFrame(rows_off)
    delta_df = pd.DataFrame(rows_delta)
    if 'condition' in on_df.columns:
        cat = pd.CategoricalDtype(categories=cond_order, ordered=True)
        for tdf in (on_df, off_df, delta_df):
            tdf['condition'] = tdf['condition'].astype(cat)
        on_df = on_df.sort_values('condition').reset_index(drop=True)
        off_df = off_df.sort_values('condition').reset_index(drop=True)
        delta_df = delta_df.sort_values('condition').reset_index(drop=True)

    on_df.to_csv(os.path.join(run_dir, 'per_class_accuracy.csv'), index=False)
    off_df.to_csv(os.path.join(run_dir, 'per_class_accuracy_gating_off.csv'), index=False)
    delta_df.to_csv(os.path.join(run_dir, 'per_class_gating_benefit.csv'), index=False)
    print(f'  Saved per_class_accuracy.csv, per_class_accuracy_gating_off.csv, '
          f'and per_class_gating_benefit.csv')


def _save_confusion_matrix(preds, labels, label_names, run_dir: str):
    """Save confusion matrix heatmap."""
    plt, sns = _import_plt()
    if plt is None:
        return
    try:
        from sklearn.metrics import confusion_matrix
    except ImportError:
        return
    cm = confusion_matrix(labels, preds, labels=list(range(len(label_names))))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set(xlabel='Predicted', ylabel='True', title='Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'confusion_matrix.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


def _save_per_class_metrics(preds, labels, label_names, run_dir: str):
    """Save per-class precision/recall/F1 bar chart."""
    plt, _ = _import_plt()
    if plt is None:
        return
    try:
        from sklearn.metrics import precision_recall_fscore_support
    except ImportError:
        return
    prec, rec, f1, sup = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(label_names))),
        average=None, zero_division=0)
    x = np.arange(len(label_names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, prec, w, label='Precision')
    ax.bar(x, rec, w, label='Recall')
    ax.bar(x + w, f1, w, label='F1')
    ax.set_xticks(x)
    ax.set_xticklabels(label_names)
    ax.set_ylim(0, 1.05)
    ax.set(ylabel='Score', title='Per-Class Metrics')
    ax.legend()
    ax.grid(True, axis='y')
    for i in range(len(label_names)):
        ax.text(x[i] + w, f1[i] + 0.02, f'n={sup[i]}', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'per_class_metrics.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


def _save_gate_distributions(eval_df: pd.DataFrame, run_dir: str):
    """Save gate distribution box plots by class."""
    plt, sns = _import_plt()
    if plt is None or len(eval_df) == 0:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    gate_cols = [('gate_face', 'Face Gate'), ('gate_eye', 'Eye Gate'), ('gate_mouth', 'Mouth Gate')]
    for ax, (col, title) in zip(axes, gate_cols):
        if col not in eval_df.columns:
            continue
        sns.boxplot(data=eval_df, x='class_name', y=col, hue='class_name',
                    ax=ax, palette='Set2', legend=False)
        ax.set(title=title, xlabel='Class', ylabel='Gate Value')
        ax.set_ylim(0, 1.05)
    plt.suptitle('Gate Distributions by Class', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'gate_distributions.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


def _save_gate_response_curves(eval_df: pd.DataFrame, run_dir: str):
    """Save gate response vs occlusion probability scatter + trend."""
    plt, _ = _import_plt()
    if plt is None or len(eval_df) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, occ_col, gate_col, title in [
        (axes[0], 'eye_occ', 'gate_eye', 'Eye Gate vs P(eye_occ)'),
        (axes[1], 'mouth_occ', 'gate_mouth', 'Mouth Gate vs P(mouth_occ)'),
    ]:
        if occ_col not in eval_df.columns or gate_col not in eval_df.columns:
            continue
        ax.scatter(eval_df[occ_col], eval_df[gate_col], alpha=0.3, s=10)
        bins = np.linspace(0, 1, 11)
        eval_df['_bin'] = pd.cut(eval_df[occ_col], bins=bins)
        trend = eval_df.groupby('_bin', observed=True)[gate_col].mean()
        bin_centers = [b.mid for b in trend.index]
        ax.plot(bin_centers, trend.values, 'r-o', lw=2, label='Mean trend')
        eval_df.drop(columns='_bin', inplace=True)
        ax.set(xlabel='Occlusion Probability', ylabel='Gate Value', title=title)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'gate_response_curves.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


def _save_attention_heatmap(eval_df: pd.DataFrame, run_dir: str):
    """Save attention weight heatmap across regions by class (from gate values)."""
    plt, sns = _import_plt()
    if plt is None or len(eval_df) == 0:
        return
    gate_cols = ['gate_face', 'gate_eye', 'gate_mouth']
    if not all(c in eval_df.columns for c in gate_cols):
        return
    classes = eval_df['class_name'].unique()
    data = []
    for cls in sorted(classes):
        sub = eval_df[eval_df['class_name'] == cls]
        data.append([sub[c].mean() for c in gate_cols])
    hm_df = pd.DataFrame(data, index=sorted(classes), columns=['Face', 'Eye', 'Mouth'])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(hm_df, annot=True, fmt='.3f', cmap='YlOrRd', vmin=0, vmax=1, ax=ax)
    ax.set(title='Mean Gate Values by Class')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'attention_heatmap.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


def _save_per_subject_accuracy(eval_df: pd.DataFrame, run_dir: str):
    """Save per-subject accuracy bar chart."""
    plt, _ = _import_plt()
    if plt is None or len(eval_df) == 0 or 'subject' not in eval_df.columns:
        return
    subj_acc = eval_df.groupby('subject')['is_correct'].mean() * 100
    subj_acc = subj_acc.sort_index()
    fig, ax = plt.subplots(figsize=(max(8, len(subj_acc) * 0.8), 5))
    bars = ax.bar(range(len(subj_acc)), subj_acc.values, color='steelblue')
    ax.set_xticks(range(len(subj_acc)))
    ax.set_xticklabels(subj_acc.index, rotation=45, ha='right')
    ax.axhline(y=subj_acc.mean(), color='r', linestyle='--',
               label=f'Mean: {subj_acc.mean():.1f}%')
    ax.set(xlabel='Subject', ylabel='Accuracy (%)',
           title='Per-Subject Accuracy (LOSO-relevant)')
    ax.legend()
    ax.grid(True, axis='y')
    for bar, val in zip(bars, subj_acc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                f'{val:.0f}%', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'per_subject_accuracy.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


def _save_stress_heatmap(summary_df: pd.DataFrame, run_dir: str,
                         details_df: pd.DataFrame = None):
    """Save stress test heatmap with overall and macro-averaged deltas."""
    plt, sns = _import_plt()
    if plt is None or len(summary_df) == 0:
        return
    col = 'condition' if 'condition' in summary_df.columns else 'occlusion_type'
    hm_data = summary_df[[col, 'opacity', 'acc_gating_on', 'acc_gating_off', 'delta_pp']].copy()
    hm_data['label'] = hm_data.apply(lambda r: f"{r[col]}\n(op={r['opacity']:.1f})", axis=1)

    has_macro = details_df is not None and len(details_df) > 0
    n_cols = 2 if has_macro else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, max(4, len(hm_data) * 0.6)))
    if n_cols == 1:
        axes = [axes]

    vals = np.column_stack([hm_data['acc_gating_on'].values,
                            hm_data['acc_gating_off'].values,
                            hm_data['delta_pp'].values])
    df_hm = pd.DataFrame(vals, index=hm_data['label'].values,
                          columns=['Gating ON', 'Gating OFF', 'Delta (pp)'])
    sns.heatmap(df_hm, annot=True, fmt='.1f', cmap='RdYlGn', center=50, ax=axes[0])
    axes[0].set(title='Overall Accuracy by Condition')

    if has_macro:
        label_names = ['EyeClosed', 'Yawn', 'Neutral']
        det_col = 'condition' if 'condition' in details_df.columns else 'occlusion_type'
        macro_rows = []
        for _, row in summary_df.iterrows():
            cond = row[col]
            sub = details_df[details_df[det_col] == cond]
            class_deltas = []
            for cls in label_names:
                sc = sub[sub['class_label'] == cls]
                if len(sc) > 0:
                    class_deltas.append(
                        (sc['correct_gating_on'].mean() - sc['correct_gating_off'].mean()) * 100)
            macro_d = float(np.mean(class_deltas)) if class_deltas else 0
            nn_vals = []
            for cls in ['EyeClosed', 'Yawn']:
                sc = sub[sub['class_label'] == cls]
                if len(sc) > 0:
                    nn_vals.append(
                        (sc['correct_gating_on'].mean() - sc['correct_gating_off'].mean()) * 100)
            nn_d = float(np.mean(nn_vals)) if nn_vals else 0
            macro_rows.append({'overall': row['delta_pp'], 'macro': macro_d, 'non_neutral': nn_d})

        df_m = pd.DataFrame(macro_rows, index=hm_data['label'].values,
                             columns=['Overall Delta', 'Macro Delta', 'Non-Neutral Delta'])
        sns.heatmap(df_m, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                    ax=axes[1], vmin=-5, vmax=10, linewidths=0.5)
        axes[1].set(title='Macro & Non-Neutral Mean Deltas (pp)')

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'stress_test_heatmap.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


def _plot_gating_comparison(summary_df: pd.DataFrame, run_dir: str):
    """Plot gating ON vs OFF comparison."""
    plt, _ = _import_plt()
    if plt is None or len(summary_df) == 0:
        return
    col = 'condition' if 'condition' in summary_df.columns else 'occlusion_type'
    labels = [f"{r[col]}_{r['opacity']:.1f}" for _, r in summary_df.iterrows()]
    x = np.arange(len(labels))
    w = 0.35
    on_vals = summary_df['acc_gating_on'].values
    off_vals = summary_df['acc_gating_off'].values
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 5))
    ax.bar(x - w / 2, on_vals, w, label='Gating ON', color='#2196F3')
    ax.bar(x + w / 2, off_vals, w, label='Gating OFF', color='#FF5722')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Gating ON vs OFF on Synthetic Occlusion')
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'gating_on_vs_off.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


def _plot_opacity_analysis(summary_df: pd.DataFrame, run_dir: str):
    """Plot accuracy & delta vs opacity level for each regime."""
    plt, _ = _import_plt()
    if plt is None or len(summary_df) == 0:
        return
    regime_col = 'regime' if 'regime' in summary_df.columns else (
        'condition' if 'condition' in summary_df.columns else 'occlusion_type')
    occluded = summary_df[summary_df['opacity'] > 0].copy()
    if len(occluded) == 0:
        return

    regimes = sorted(occluded[regime_col].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    markers = ['o', 's', 'D', '^', 'v']
    for i, regime in enumerate(regimes):
        sub = occluded[occluded[regime_col] == regime].sort_values('opacity')
        mk = markers[i % len(markers)]
        axes[0].plot(sub['opacity'], sub['acc_gating_on'], f'{mk}-', label=f'{regime} ON')
        axes[0].plot(sub['opacity'], sub['acc_gating_off'], f'{mk}--', alpha=0.5, label=f'{regime} OFF')
        axes[1].plot(sub['opacity'], sub['delta_pp'], f'{mk}-', lw=2, label=regime)
    axes[0].set(xlabel='Opacity', ylabel='Accuracy (%)',
                title='Accuracy vs Opacity (Gating ON vs OFF)')
    axes[0].legend(fontsize=7)
    axes[0].grid(True)
    axes[1].axhline(y=0, color='black', ls='-', lw=0.8)
    axes[1].set(xlabel='Opacity', ylabel='Delta (pp)',
                title='Gating Benefit (ON - OFF) vs Opacity')
    axes[1].legend(fontsize=8)
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'opacity_analysis.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


def _plot_gates_vs_opacity(details_df: pd.DataFrame, run_dir: str):
    """Plot gate values and p_eye/p_mouth vs opacity by regime group."""
    plt, _ = _import_plt()
    if plt is None or len(details_df) == 0:
        return
    regime_col = 'regime' if 'regime' in details_df.columns else (
        'condition' if 'condition' in details_df.columns else 'occlusion_type')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    eye_conds = ['eye_only', 'persistent_eye', 'transient_eye']
    mouth_conds = ['mouth_only', 'persistent_mouth', 'transient_mouth']
    both_conds = ['both', 'persistent_both']
    for ax, conds, title in zip(axes,
                                [eye_conds, mouth_conds, both_conds],
                                ['Eye occlusion', 'Mouth occlusion', 'Both']):
        sub = details_df[details_df[regime_col].isin(conds)]
        if len(sub) == 0:
            ax.set_title(title)
            continue
        s = sub.groupby('opacity').agg(
            gate_eye=('gate_eye', 'mean'), gate_mouth=('gate_mouth', 'mean'),
            p_eye=('p_eye', 'mean'), p_mouth=('p_mouth', 'mean')).reset_index()
        ax.plot(s['opacity'], s['gate_eye'], 'o-', color='#9C27B0', lw=2, label='Gate(eye)')
        ax.plot(s['opacity'], s['gate_mouth'], 's-', color='#FF9800', lw=2, label='Gate(mouth)')
        ax.plot(s['opacity'], s['p_eye'], 'o--', color='#9C27B0', alpha=0.5, lw=1.5, label='p_eye')
        ax.plot(s['opacity'], s['p_mouth'], 's--', color='#FF9800', alpha=0.5, lw=1.5, label='p_mouth')
        ax.set_xlabel('Opacity')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True)
        ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'gates_vs_opacity.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


def _save_per_class_stress_deltas(details_df: pd.DataFrame, run_dir: str):
    """Compute per-class, macro-averaged, and non-neutral mean gating deltas."""
    os.makedirs(run_dir, exist_ok=True)
    plt, sns = _import_plt()
    if details_df is None or len(details_df) == 0:
        return
    label_names = ['EyeClosed', 'Yawn', 'Neutral']
    cond_col = 'condition' if 'condition' in details_df.columns else 'occlusion_type'
    conditions = details_df[cond_col].unique()

    rows = []
    for cond in conditions:
        sub = details_df[details_df[cond_col] == cond]
        opacity = sub['opacity'].iloc[0] if 'opacity' in sub.columns else 0.0
        regime = sub['regime'].iloc[0] if 'regime' in sub.columns else cond
        class_deltas = {}
        for cls in label_names:
            sc = sub[sub['class_label'] == cls]
            if len(sc) > 0:
                d = (sc['correct_gating_on'].mean() - sc['correct_gating_off'].mean()) * 100
                class_deltas[cls] = d
            else:
                class_deltas[cls] = float('nan')

        overall = (sub['correct_gating_on'].mean() - sub['correct_gating_off'].mean()) * 100
        valid = [v for v in class_deltas.values() if not np.isnan(v)]
        macro = float(np.mean(valid)) if valid else float('nan')
        event_vals = [class_deltas.get(c, float('nan')) for c in ['EyeClosed', 'Yawn']]
        event_vals = [v for v in event_vals if not np.isnan(v)]
        event_mean = float(np.mean(event_vals)) if event_vals else float('nan')

        rows.append({
            'condition': cond, 'regime': regime, 'opacity': opacity,
            'delta_EyeClosed': class_deltas.get('EyeClosed', float('nan')),
            'delta_Yawn': class_deltas.get('Yawn', float('nan')),
            'delta_Neutral': class_deltas.get('Neutral', float('nan')),
            'delta_overall': overall,
            'delta_macro': macro,
            'delta_non_neutral': event_mean,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run_dir, 'stress_test_per_class_deltas.csv'), index=False)
    print('\nPer-class gating delta (ON - OFF):')
    print(df.to_string(index=False, float_format='%.1f'))

    if plt is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, max(5, len(df) * 0.45)))

    labels = [f"{r['condition']}" for _, r in df.iterrows()]
    hm_vals = df[['delta_EyeClosed', 'delta_Yawn', 'delta_Neutral', 'delta_overall',
                   'delta_macro', 'delta_non_neutral']].values
    hm_df = pd.DataFrame(hm_vals, index=labels,
                          columns=['EyeClosed', 'Yawn', 'Neutral', 'Overall',
                                   'Macro', 'Non-Neutral'])

    sns.heatmap(hm_df[['EyeClosed', 'Yawn', 'Neutral']], annot=True, fmt='.1f',
                cmap='RdYlGn', center=0, ax=axes[0], vmin=-5, vmax=25,
                linewidths=0.5, cbar_kws={'label': 'Delta (pp)'})
    axes[0].set_title('Per-Class Gating Delta (ON - OFF)')
    axes[0].set_ylabel('')

    sns.heatmap(hm_df[['Overall', 'Macro', 'Non-Neutral']], annot=True, fmt='.1f',
                cmap='RdYlGn', center=0, ax=axes[1], vmin=-3, vmax=10,
                linewidths=0.5, cbar_kws={'label': 'Delta (pp)'})
    axes[1].set_title('Aggregated Deltas')
    axes[1].set_ylabel('')

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'per_class_stress_deltas.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


def _add_paired_stats_to_stress_summary(summary_df: pd.DataFrame,
                                        details_df: pd.DataFrame,
                                        seed: int) -> pd.DataFrame:
    """Augment stress summary with paired delta CIs and exact p-values."""
    if summary_df is None or len(summary_df) == 0 or details_df is None or len(details_df) == 0:
        return summary_df

    cond_col = 'condition' if 'condition' in details_df.columns else 'occlusion_type'
    summary_cond_col = 'condition' if 'condition' in summary_df.columns else 'occlusion_type'
    rows = []
    for _, row in summary_df.iterrows():
        cond = row[summary_cond_col]
        sub = details_df[details_df[cond_col] == cond]
        if len(sub) == 0:
            rows.append(row.to_dict())
            continue
        stats = compute_paired_binary_statistics(
            sub['correct_gating_on'].to_numpy(),
            sub['correct_gating_off'].to_numpy(),
            seed=seed,
        )
        merged = row.to_dict()
        merged.update(stats)
        merged['delta_significant_05'] = bool(
            not np.isnan(stats['p_value_mcnemar']) and stats['p_value_mcnemar'] < 0.05)
        rows.append(merged)
    return pd.DataFrame(rows)


def _extract_clean_metrics_from_stress_details(stress_details: pd.DataFrame) -> Dict:
    """Build clean test metrics from the clean stress-test condition."""
    if stress_details is None or len(stress_details) == 0 or 'condition' not in stress_details.columns:
        return {}

    clean_rows = stress_details[stress_details['condition'] == 'clean']
    if len(clean_rows) == 0:
        return {}

    labels = clean_rows['gt_label'].to_numpy()
    preds_on = clean_rows['pred_gating_on'].to_numpy()
    preds_off = clean_rows['pred_gating_off'].to_numpy()

    metrics = compute_classification_metrics(
        labels, preds_on, label_names=['EyeClosed', 'Yawn', 'Neutral'])
    uncertainty = compute_classification_uncertainty(labels, preds_on)
    paired = compute_paired_binary_statistics(
        clean_rows['correct_gating_on'].to_numpy(),
        clean_rows['correct_gating_off'].to_numpy(),
    )
    off_metrics = compute_classification_metrics(
        labels, preds_off, label_names=['EyeClosed', 'Yawn', 'Neutral'])

    result = {
        'accuracy': metrics.get('accuracy', 0.0),
        'balanced_accuracy': metrics.get('balanced_accuracy', float('nan')),
        'precision': metrics.get('precision', float('nan')),
        'recall': metrics.get('recall', float('nan')),
        'f1': metrics.get('f1', float('nan')),
        'macro_precision': metrics.get('macro_precision', float('nan')),
        'macro_recall': metrics.get('macro_recall', float('nan')),
        'macro_f1': metrics.get('macro_f1', float('nan')),
        'per_class': metrics.get('per_class', {}),
        'confusion_matrix': metrics.get('confusion_matrix', []),
        'uncertainty': uncertainty,
        'gating_off_accuracy': off_metrics.get('accuracy', float('nan')),
        'clean_delta_pp': paired.get('delta_pp', float('nan')),
        'clean_delta_ci_low': paired.get('delta_ci_low', float('nan')),
        'clean_delta_ci_high': paired.get('delta_ci_high', float('nan')),
        'clean_delta_p_value': paired.get('p_value_mcnemar', float('nan')),
        'n_samples': int(len(clean_rows)),
        'source': 'stress_test_clean_condition',
    }
    return result


def _save_comprehensive_analysis(model, val_loader, test_loader, val_samples,
                                 test_samples, history, stress_summary,
                                 stress_details, run_dir, device):
    """Generate all analysis plots and save summary JSON."""
    label_names = ['EyeClosed', 'Yawn', 'Neutral']

    eval_loader = test_loader if test_loader is not None else val_loader
    eval_samples = test_samples if test_samples else val_samples
    eval_label = 'test' if test_samples else 'val'

    model.eval()
    eval_df = collect_eval_with_occlusion(model, eval_loader)

    if 'subject' not in eval_df.columns and eval_samples:
        subjects = [s.get('subject', 'unknown') for s in eval_samples]
        if len(subjects) == len(eval_df):
            eval_df['subject'] = subjects

    if len(eval_df) > 0:
        preds = eval_df['pred'].values
        labels = eval_df['true'].values
        _save_confusion_matrix(preds, labels, label_names, run_dir)
        _save_per_class_metrics(preds, labels, label_names, run_dir)
        _save_gate_distributions(eval_df, run_dir)
        _save_gate_response_curves(eval_df, run_dir)
        _save_attention_heatmap(eval_df, run_dir)
        _save_per_subject_accuracy(eval_df, run_dir)
        eval_df.to_csv(os.path.join(run_dir, f'{eval_label}_predictions.csv'), index=False)
        print(f'  Saved {len(eval_df)} {eval_label} predictions and analysis plots')

    _save_training_dynamics(history, run_dir)
    _save_val_loss_components(history, run_dir)
    _save_val_gate_statistics(history, run_dir)

    if len(stress_summary) > 0:
        _save_stress_heatmap(stress_summary, run_dir, details_df=stress_details)

    summary = {
        'n_train': len(val_samples) if not test_samples else 0,
        'n_eval': len(eval_df),
        'eval_set': eval_label,
        'overall_accuracy': float(eval_df['is_correct'].mean() * 100) if len(eval_df) else 0,
    }
    if len(eval_df) > 0:
        metrics = compute_classification_metrics(
            eval_df['true'].to_numpy(),
            eval_df['pred'].to_numpy(),
            label_names=label_names,
        )
        summary['balanced_accuracy'] = metrics.get('balanced_accuracy', float('nan'))
        summary['macro_f1'] = metrics.get('macro_f1', float('nan'))
        summary['weighted_f1'] = metrics.get('f1', float('nan'))
        for cls in label_names:
            sub = eval_df[eval_df['class_name'] == cls]
            if len(sub) > 0:
                summary[f'acc_{cls}'] = float(sub['is_correct'].mean() * 100)
                summary[f'n_{cls}'] = len(sub)

    with open(os.path.join(run_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


# ─── Model loading helpers ───────────────────────────────────────────────────

def _load_extraction_models(device, args):
    """Load feature extractor, occlusion model, and face detector."""
    from feature_extraction import ResNet34FeatureExtractor
    from occlusion_estimator import ResNet34OcclusionModel

    print('Loading feature extraction models...')
    if device.type == 'cuda':
        print(f'  [GPU] before feature extractor: {_gpu_memory_snapshot()}')
    try:
        feat_extractor = ResNet34FeatureExtractor(
            CONFIG['RESNET34_MODEL_PATH'], device=str(device))
    except Exception:
        if device.type == 'cuda':
            print(f'  [GPU] feature extractor load failed: {_gpu_memory_snapshot()}')
        raise
    _gpu_cleanup('feature extractor loaded')

    if device.type == 'cuda':
        print(f'  [GPU] before occlusion model: {_gpu_memory_snapshot()}')
    try:
        occ_model = ResNet34OcclusionModel(
            CONFIG['RESNET34_OCCLUSION_MODEL_PATH'], device=str(device))
    except Exception:
        if device.type == 'cuda':
            print(f'  [GPU] occlusion model load failed: {_gpu_memory_snapshot()}')
        raise

    det_size = (args.det_size, args.det_size) if args.face == 'retina' else None
    if device.type == 'cuda':
        print(f'  [GPU] before face detector: {_gpu_memory_snapshot()}')
    try:
        if args.face == 'retina':
            from face_detection_retinaface import FaceDetector
            face_detector = FaceDetector(
                shape_model_path=CONFIG['DLIB_MODEL_PATH'],
                det_size=det_size, det_thresh=0.35)
        else:
            from face_detection_dlib import FaceDetector
            face_detector = FaceDetector(shape_model_path=CONFIG['DLIB_MODEL_PATH'])
    except ImportError as e:
        print(f'Face detector import failed: {e}')
        if args.face == 'retina':
            print('Falling back to dlib.')
            from face_detection_dlib import FaceDetector
            face_detector = FaceDetector(shape_model_path=CONFIG['DLIB_MODEL_PATH'])
        else:
            raise

    _gpu_cleanup('all extraction models loaded')
    return feat_extractor, occ_model, face_detector


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Train, evaluate, and stress-test DOFG pipeline')
    ap.add_argument('--data', default='Data', help='Path to Data folder')
    ap.add_argument('--results-root', default='results',
                    help='Directory under which the run folder will be created')
    ap.add_argument('--run-name', default=None,
                    help='Optional explicit run directory name')
    ap.add_argument('--samples', type=int, default=30,
                    help='Samples per video (0=all). Use 20-30 for quick CPU test')
    ap.add_argument('--epochs', type=int, default=5, help='Training epochs')
    ap.add_argument('--batch', type=int, default=16, help='Batch size')
    ap.add_argument('--mode', choices=['fixed', 'kfold', 'loso'], default='fixed')
    ap.add_argument('--k', type=int, default=5, help='k for k-fold')
    ap.add_argument('--num-test', type=int, default=3, help='Test subjects for fixed split')
    ap.add_argument('--face', choices=['dlib', 'retina'], default='retina',
                    help='Face detector (retina=RetinaFace, dlib=fallback)')
    ap.add_argument('--det-size', type=int, default=640,
                    help='RetinaFace input size (320/480/640)')
    ap.add_argument('--strategy', choices=['clip', 'legacy'], default='clip',
                    help='clip=STRATEGY_DESIGN; legacy=frame sampling')
    ap.add_argument('--max-train-clips', type=int, default=None)
    ap.add_argument('--max-val-clips', type=int, default=None)
    ap.add_argument('--max-test-clips', type=int, default=None)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--stress', action='store_true', default=True)
    ap.add_argument('--no-stress', action='store_false', dest='stress')
    ap.add_argument('--stress-frames', type=int, default=20)
    ap.add_argument('--stress-opacities', type=str, default='0.4,0.6,0.8,1.0')
    ap.add_argument('--train-opacity-mode',
                    choices=['legacy', 'discrete', 'uniform'],
                    default='legacy',
                    help='Train-time opacity sampler: legacy=V4 hard/medium policy')
    ap.add_argument('--train-opacity-values', type=str, default=None,
                    help='Comma-separated train opacities for the discrete ablation, e.g. 0.4,0.6,0.8,1.0')
    ap.add_argument('--train-opacity-weights', type=str, default=None,
                    help='Optional comma-separated weights matching --train-opacity-values')
    ap.add_argument('--train-opacity-range', type=str, default=None,
                    help='Uniform train opacity range LOW,HIGH for the ablation, e.g. 0.4,1.0')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--defer-test', action='store_true', default=True)
    ap.add_argument('--no-defer-test', action='store_false', dest='defer_test')
    ap.add_argument('--benchmark', action='store_true')
    ap.add_argument('--face-cpu', action='store_true')
    ap.add_argument('--model', choices=['transformer', 'mlp_baseline', 'resnet_baseline'],
                    default='transformer',
                    help='Model architecture: transformer (gated), mlp_baseline (concat+MLP), '
                         'or resnet_baseline (end-to-end ResNet-34 on face crop)')
    ap.add_argument('--class-weighted', action='store_true',
                    help='Use class-weighted cross-entropy loss')
    ap.add_argument('--gate-supervision', choices=['gt', 'estimator'], default='gt',
                    help='Gate alignment target: gt=ground-truth, estimator=proxy')
    ap.add_argument('--gate-weight', type=float, default=0.5,
                    help='Weight for gate alignment loss (w_gate_occ)')
    ap.add_argument('--gate-floor', type=float, default=0.05,
                    help='Minimum gate value (0.0-1.0), used for all regions unless --asymmetric-floor')
    ap.add_argument('--asymmetric-floor', action='store_true',
                    help='Use different floor values for eye and mouth gates')
    ap.add_argument('--eye-floor', type=float, default=None,
                    help='Eye gate floor (only with --asymmetric-floor)')
    ap.add_argument('--mouth-floor', type=float, default=None,
                    help='Mouth gate floor (only with --asymmetric-floor)')
    ap.add_argument('--diversity-reg', type=float, default=0.0,
                    help='Gate diversity regularisation weight (0=off)')
    ap.add_argument('--gating-mode', choices=['attention', 'legacy', 'both', 'none'],
                    default='attention',
                    help='Gating mechanism in the transformer. "attention" = additive '
                         'log-gate bias on attention scores (V2, default, recommended); '
                         '"legacy" = V4 multiplicative token gating (largely neutralised '
                         'by pre-norm LayerNorm); "both" = apply both; "none" = gates '
                         'computed for logging but not applied.')
    # V7 Phase-1 defaults: only attention-bias is ON. Auxiliary heads opt-in.
    ap.add_argument('--use-logit-bias', dest='use_logit_bias', action='store_true',
                    help='Enable gate-conditioned per-class logit bias head '
                         '(OFF by default — see docs/GATING_V7_MINIMAL.md).')
    ap.add_argument('--no-logit-bias', dest='use_logit_bias', action='store_false',
                    help='Disable the gate-conditioned per-class logit bias head.')
    ap.add_argument('--use-estimator-calibration', dest='use_estimator_calibration',
                    action='store_true',
                    help='Enable joint eye+mouth estimator calibration head '
                         '(OFF by default).')
    ap.add_argument('--no-estimator-calibration', dest='use_estimator_calibration',
                    action='store_false',
                    help='Disable the joint eye+mouth estimator calibration head.')
    ap.add_argument('--gate-dropout', type=float, default=0.0,
                    help='Probability per sample/region of forcing gate=1.0 during '
                         'training (default 0.0 = disabled).')
    ap.add_argument('--clean-invariance-weight', type=float, default=0.0,
                    help='Weight on MSE(logits_on, logits_off) for clean samples. '
                         'Enforces gating≈no-op on clean frames. 0 = off (default).')
    ap.add_argument('--clean-invariance-thresh', type=float, default=0.1,
                    help='GT occlusion threshold for "clean" in the invariance reg.')
    ap.add_argument('--checkpoint-metric', choices=['macro_f1', 'accuracy'],
                    default='macro_f1',
                    help='Validation metric used for best-checkpoint selection '
                         '(default: macro_f1).')
    ap.set_defaults(use_logit_bias=False, use_estimator_calibration=False)
    ap.add_argument('--clip-length', type=int, default=32,
                    help='Frames per clip (T)')
    ap.add_argument('--natural-occlusion-eval', action='store_true',
                    help='Run evaluation on natural-occlusion subset')
    args = ap.parse_args()

    # Resolve per-region gate floors
    if args.asymmetric_floor:
        if args.eye_floor is None or args.mouth_floor is None:
            ap.error('--asymmetric-floor requires both --eye-floor and --mouth-floor')
        args._eye_floor = args.eye_floor
        args._mouth_floor = args.mouth_floor
        print(f'\n  Asymmetric gate floor ENABLED: eye={args._eye_floor}, mouth={args._mouth_floor}')
    else:
        args._eye_floor = args.gate_floor
        args._mouth_floor = args.gate_floor

    try:
        args._train_opacity_sampler = _build_train_opacity_sampler(args)
    except ValueError as e:
        ap.error(str(e))

    if args.model == 'resnet_baseline' and args.strategy != 'clip':
        print('WARNING: resnet_baseline requires --strategy clip. Overriding.')
        args.strategy = 'clip'

    if args.face_cpu:
        os.environ['DOFG_FACE_CPU'] = '1'

    _require_cuda('run_train_eval.py')
    _set_global_seed(args.seed)
    device = torch.device('cuda')
    run_dir = _ensure_results_dir(results_root=args.results_root, run_name=args.run_name)
    print(f'Results dir: {run_dir}')
    print(f'Device: {device}')
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f'GPU: {torch.cuda.get_device_name(0)} '
              f'({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)')
    print(f'Mode: {args.mode}, samples/video: {args.samples or "ALL"}, epochs: {args.epochs}')
    print(f'Train opacity sampler: {args._train_opacity_sampler}')

    # Override CLIP_CONFIG['T'] if --clip-length differs from default
    if args.clip_length != 32:
        from config import CLIP_CONFIG
        CLIP_CONFIG['T'] = args.clip_length
        print(f'  Clip length overridden: T={args.clip_length}')

    config_dict = {
        'data': args.data, 'samples_per_video': args.samples or 'all',
        'epochs': args.epochs, 'batch_size': args.batch, 'mode': args.mode,
        'k': args.k, 'num_test': args.num_test, 'fold': args.fold,
        'face_detector': args.face, 'det_size': args.det_size if args.face == 'retina' else None,
        'face_cpu': args.face_cpu, 'seed': args.seed,
        'stress_test': args.stress, 'stress_frames': args.stress_frames,
        'train_opacity_sampler': args._train_opacity_sampler,
        'train_opacity_mode': args._train_opacity_sampler.get('mode'),
        'train_opacity_values': args._train_opacity_sampler.get('values'),
        'train_opacity_weights': args._train_opacity_sampler.get('weights'),
        'train_opacity_range': [
            args._train_opacity_sampler.get('low'),
            args._train_opacity_sampler.get('high'),
        ] if args._train_opacity_sampler.get('mode') == 'uniform' else None,
        'defer_test': args.defer_test, 'strategy': args.strategy,
        'model': args.model,
        'class_weighted': args.class_weighted,
        'gate_supervision': args.gate_supervision,
        'gate_weight': args.gate_weight,
        'gate_floor': args.gate_floor,
        'asymmetric_floor': args.asymmetric_floor,
        'eye_floor': args._eye_floor,
        'mouth_floor': args._mouth_floor,
        'diversity_reg': args.diversity_reg,
        'gating_mode': args.gating_mode,
        'use_logit_bias': bool(args.use_logit_bias),
        'use_estimator_calibration': bool(args.use_estimator_calibration),
        'gate_dropout': float(args.gate_dropout),
        'clean_invariance_weight': float(args.clean_invariance_weight),
        'clean_invariance_thresh': float(args.clean_invariance_thresh),
        'checkpoint_metric': args.checkpoint_metric,
        'learning_rate': 1e-3 if args.model == 'resnet_baseline' else 3e-5,
        'clip_length': args.clip_length,
        'max_train_clips': args.max_train_clips,
        'max_val_clips': args.max_val_clips,
        'max_test_clips': args.max_test_clips,
        'natural_occlusion_eval': args.natural_occlusion_eval,
        'validation_protocol': 'dual (clean primary + stress secondary)',
    }
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

    t_pipeline_start = time.time()
    stage_times = {}

    # ── Stage 1: Load data ────────────────────────────────────────────────────
    t0 = time.time()
    print('\n=== Stage 1: Loading data ===')
    csv_data = load_csv_video_data(args.data, filter_eye_states=True)
    if not csv_data:
        print('No data found. Check --data path.')
        sys.exit(1)

    _audit_subjects(csv_data)

    splits = _splits_from_fixed_or_fold(
        csv_data, mode=args.mode, k=args.k,
        num_test=args.num_test, seed=args.seed, fold=args.fold)
    split_config = splits.get('split_config', {})
    split_info = {
        'mode': args.mode,
        'fold': args.fold,
        'k': args.k,
        'seed': args.seed,
        'train_subjects': split_config.get('train_subjects', []),
        'test_subjects': split_config.get('test_subjects', []),
        'train_videos': splits.get('train', []),
        'test_videos': splits.get('test', []),
    }
    with open(os.path.join(run_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f'Train videos: {len(splits["train"])}, Test videos: {len(splits["test"])}')
    print(f'Strategy: {args.strategy}')
    stage_times['stage1_data_loading'] = time.time() - t0
    print(f'  [Stage 1 completed in {stage_times["stage1_data_loading"]:.1f}s]')

    # ── Stage 2: Feature extraction ───────────────────────────────────────────
    t0 = time.time()
    print('\n=== Stage 2: Feature extraction ===')
    feat_extractor, occ_model, face_detector = _load_extraction_models(device, args)

    try:
        if args.strategy == 'clip':
            print('  Using clip-based pipeline (T=32, regime aug, temporal val)')
            _save_crops = (args.model == 'resnet_baseline')
            train_samples, val_samples, test_samples, test_clips, val_clips = extract_features_for_clips(
                csv_data, split_config,
                face_detector=face_detector, feat_extractor=feat_extractor,
                occ_model=occ_model, val_ratio=0.20, seed=args.seed,
                max_train_clips=args.max_train_clips,
                max_val_clips=args.max_val_clips,
                max_test_clips=args.max_test_clips,
                skip_test=args.defer_test,
                train_opacity_sampler=args._train_opacity_sampler,
                save_face_crops=_save_crops)
        else:
            num_samples = args.samples if args.samples > 0 else None
            print(f'  Using legacy frame sampling (num_samples_per_video={num_samples or "ALL"})')
            train_samples, val_samples, test_samples = extract_features_stratified(
                csv_data, splits,
                face_detector=face_detector, feat_extractor=feat_extractor,
                occ_model=occ_model, num_samples_per_video=num_samples,
                val_ratio=0.20, random_state=args.seed)
            test_clips = []
            val_clips = []
    finally:
        # Free extraction models — training uses only the transformer
        if not args.stress and not args.benchmark:
            _delete_models(feat_extractor, occ_model, face_detector)
            feat_extractor = occ_model = face_detector = None

    _gpu_cleanup('feature extraction complete')
    stage_times['stage2_feature_extraction'] = time.time() - t0
    print(f'  [Stage 2 completed in {stage_times["stage2_feature_extraction"]:.1f}s]')

    print(f'\nTrain: {len(train_samples)}  Val: {len(val_samples)}  Test: {len(test_samples)}')
    for name, ss in [('Train', train_samples), ('Val', val_samples), ('Test', test_samples)]:
        if ss:
            print(f'  {name}: {Counter(s["class_name"] for s in ss)}')

    # ── Stage 3: Training ─────────────────────────────────────────────────────
    t0 = time.time()
    print('\n=== Stage 3: Training ===')
    train_ds = DriverStateDataset(train_samples, device=str(device),
                                   gate_supervision=args.gate_supervision)
    val_ds = DriverStateDataset(val_samples, device=str(device),
                                gate_supervision=args.gate_supervision)
    test_loader = None
    if test_samples:
        test_ds = DriverStateDataset(test_samples, device=str(device),
                                     gate_supervision=args.gate_supervision)
        test_generator = torch.Generator()
        test_generator.manual_seed(args.seed + 2)
        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                                 collate_fn=test_ds.collate_samples,
                                 worker_init_fn=_seed_worker,
                                 generator=test_generator)

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=train_ds.collate_samples, drop_last=True,
                              worker_init_fn=_seed_worker,
                              generator=train_generator)
    val_generator = torch.Generator()
    val_generator.manual_seed(args.seed + 1)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            collate_fn=val_ds.collate_samples,
                            worker_init_fn=_seed_worker,
                            generator=val_generator)

    if args.model == 'mlp_baseline':
        model = RegionFeatureMLP(
            feature_dim=512, hidden_dim=512, num_classes=3, dropout=0.3,
        ).to(device)
        effective_gate_weight = 0.0
        print(f'Model: RegionFeatureMLP (no gating)')
    elif args.model == 'resnet_baseline':
        model = ResNet34Baseline(num_classes=3, dropout=0.3).to(device)
        effective_gate_weight = 0.0
        print(f'Model: ResNet34Baseline (end-to-end face-crop CNN)')
    else:
        model = EnhancedOcclusionAwareTransformer(
            feature_dim=512, hidden_dim=128, num_heads=4,
            num_classes=3, num_layers=3, use_relative_pos=True,
            gate_floor=args.gate_floor,
            eye_floor=args._eye_floor,
            mouth_floor=args._mouth_floor,
            gating_mode=args.gating_mode,
            use_logit_bias=args.use_logit_bias,
            use_estimator_calibration=args.use_estimator_calibration,
            gate_dropout=args.gate_dropout,
        ).to(device)
        effective_gate_weight = args.gate_weight
        print(f'Model: EnhancedOcclusionAwareTransformer '
              f'(gating_mode={args.gating_mode}, logit_bias={args.use_logit_bias}, '
              f'estimator_calibration={args.use_estimator_calibration}, '
              f'gate_dropout={args.gate_dropout})')
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Model params: {total:,} total, {trainable:,} trainable')

    class_weights = None
    if args.class_weighted and train_samples:
        label_counts = Counter(s['label'] for s in train_samples)
        total = sum(label_counts.values())
        num_classes = 3
        present_classes = [i for i in range(num_classes) if label_counts.get(i, 0) > 0]
        n_present = max(1, len(present_classes))
        missing_classes = [i for i in range(num_classes) if i not in present_classes]
        if missing_classes:
            print(f'  WARNING: train subset is missing classes {missing_classes}; '
                  'setting their class weights to 0.0 for this run.')
        weights = [
            (total / (n_present * label_counts[i])) if label_counts.get(i, 0) > 0 else 0.0
            for i in range(num_classes)
        ]
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f'  Class weights: {[f"{w:.2f}" for w in weights]}')

    lr = 1e-3 if args.model == 'resnet_baseline' else 3e-5
    trainer = TinyTransformerTrainer(model, device=str(device), learning_rate=lr,
                                    class_weights=class_weights,
                                    gate_weight=effective_gate_weight,
                                    gate_floor=args.gate_floor,
                                    eye_floor=args._eye_floor,
                                    mouth_floor=args._mouth_floor,
                                    diversity_reg=args.diversity_reg,
                                    clean_invariance_weight=args.clean_invariance_weight,
                                    clean_invariance_thresh=args.clean_invariance_thresh)

    ckpt_metric = args.checkpoint_metric
    best_ckpt_value = float('-inf')
    best_val_acc = float('-inf')
    best_val_f1  = 0.0
    best_epoch = None
    for epoch in range(args.epochs):
        # Per-epoch re-seed so stochastic components (dropout, gate_dropout,
        # shuffling) are reproducible from the epoch number alone. This keeps
        # the training loop byte-identical to run_gate_floor_sweep.py.
        _set_global_seed(args.seed + epoch)
        train_metrics = trainer.train_epoch(train_loader, epoch=epoch)
        val_metrics = trainer.validate_epoch(val_loader, epoch=epoch)
        val_acc = val_metrics.get('val_accuracy', 0.0)
        val_f1  = val_metrics.get('val_macro_f1', 0.0)
        metric_value = val_f1 if ckpt_metric == 'macro_f1' else val_acc

        if best_epoch is None or metric_value > best_ckpt_value:
            best_ckpt_value = metric_value
            best_val_acc = val_acc
            best_val_f1  = val_f1
            best_epoch = epoch + 1
            ckpt_path = os.path.join(run_dir, 'model_best.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f'  ** Best val ({ckpt_metric}={metric_value:.3f}) -> saved')
        print(f'  Ep {epoch + 1}/{args.epochs}: '
              f'train_acc={train_metrics.get("accuracy", 0):.1f}% '
              f'val_acc={val_acc:.1f}% val_f1={val_f1:.3f}')

    history = trainer.history
    with open(os.path.join(run_dir, 'training_history.json'), 'w') as f:
        json.dump({k: (v if isinstance(v, list) else str(v))
                   for k, v in history.items()}, f, indent=2)
    _save_training_curves(history, run_dir)

    # Free optimizer/scheduler state before stress test
    del trainer.optimizer, trainer.scheduler
    _gpu_cleanup('training complete, optimizer freed')
    stage_times['stage3_training'] = time.time() - t0
    print(f'  [Stage 3 completed in {stage_times["stage3_training"]:.1f}s]')

    # Load best checkpoint
    ckpt = torch.load(os.path.join(run_dir, 'model_best.pt'),
                      map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        print(f'  [ckpt] missing keys (new heads, randomly init): {missing[:4]}'
              f'{" ..." if len(missing) > 4 else ""}')
    if unexpected:
        print(f'  [ckpt] unexpected keys (ignored): {unexpected[:4]}'
              f'{" ..." if len(unexpected) > 4 else ""}')
    del ckpt
    _gpu_cleanup('best checkpoint loaded')

    # ── Stage 4a: Val-Stress evaluation (secondary robustness diagnostic) ───
    t0_stress = time.time()
    val_stress_details = pd.DataFrame()
    val_stress_summary = pd.DataFrame()
    if args.stress and val_clips and args.strategy == 'clip':
        print('\n=== Stage 4a: Val-Stress evaluation (secondary) ===')
        if feat_extractor is None:
            feat_extractor, occ_model, face_detector = _load_extraction_models(device, args)

        opacity_levels = [float(x.strip()) for x in args.stress_opacities.split(',') if x.strip()]
        val_stress_details, val_stress_summary = run_stress_test(
            csv_data=csv_data, test_keys=splits['train'][:0], model=model,
            face_detector=face_detector, feat_extractor=feat_extractor,
            occ_model=occ_model, trainer=trainer, device=str(device),
            opacity_levels=opacity_levels,
            max_frames_per_video=args.stress_frames, batch_size=args.batch,
            seed=args.seed,
            test_clips=val_clips,
            max_frames_per_clip=min(8, args.stress_frames // 2))
        val_stress_summary = _add_paired_stats_to_stress_summary(
            val_stress_summary, val_stress_details, seed=args.seed)

        if len(val_stress_details) > 0:
            val_stress_details.to_csv(os.path.join(run_dir, 'val_stress_details.csv'), index=False)
            val_stress_summary.to_csv(os.path.join(run_dir, 'val_stress_summary.csv'), index=False)
            print(f'\nVal-Stress: {len(val_stress_details)} detail rows')
            print(val_stress_summary.to_string(index=False))
            _save_per_class_stress_deltas(val_stress_details,
                                          os.path.join(run_dir, 'val_stress'))
        else:
            print('  No Val-Stress results.')

    stage_times['stage4a_val_stress'] = time.time() - t0_stress
    if args.stress and val_clips and args.strategy == 'clip':
        print(f'  [Stage 4a completed in {stage_times["stage4a_val_stress"]:.1f}s]')

    # ── Stage 4b: Test-Stress evaluation ─────────────────────────────────────
    t0 = time.time()
    stress_details = pd.DataFrame()
    stress_summary = pd.DataFrame()
    if args.stress and splits['test']:
        print('\n=== Stage 4b: Test-Stress evaluation ===')
        if feat_extractor is None:
            feat_extractor, occ_model, face_detector = _load_extraction_models(device, args)

        opacity_levels = [float(x.strip()) for x in args.stress_opacities.split(',') if x.strip()]
        stress_details, stress_summary = run_stress_test(
            csv_data=csv_data, test_keys=splits['test'], model=model,
            face_detector=face_detector, feat_extractor=feat_extractor,
            occ_model=occ_model, trainer=trainer, device=str(device),
            opacity_levels=opacity_levels,
            max_frames_per_video=args.stress_frames, batch_size=args.batch,
            seed=args.seed,
            test_clips=test_clips if args.strategy == 'clip' else None,
            max_frames_per_clip=min(8, args.stress_frames // 2) if args.strategy == 'clip' else None)
        stress_summary = _add_paired_stats_to_stress_summary(
            stress_summary, stress_details, seed=args.seed)

        if len(stress_details) > 0:
            stress_details.to_csv(os.path.join(run_dir, 'stress_test_details.csv'), index=False)
            stress_summary.to_csv(os.path.join(run_dir, 'stress_test_summary.csv'), index=False)
            print(f'\nTest-Stress: {len(stress_details)} detail rows')
            print(stress_summary.to_string(index=False))
            _plot_gating_comparison(stress_summary, run_dir)
            _plot_opacity_analysis(stress_summary, run_dir)
            _plot_gates_vs_opacity(stress_details, run_dir)
            _save_per_class_stress_deltas(stress_details, run_dir)
            _save_per_class_stress_tables(
                os.path.join(run_dir, 'stress_test_details.csv'), run_dir)
        else:
            print('  No stress test results.')

        # Free extraction models after stress test
        _delete_models(feat_extractor, occ_model, face_detector)
        feat_extractor = occ_model = face_detector = None

    stage_times['stage4b_test_stress'] = time.time() - t0
    if args.stress and splits['test']:
        print(f'  [Stage 4b completed in {stage_times["stage4b_test_stress"]:.1f}s]')

    # ── Stage 5: Evaluation and analysis ──────────────────────────────────────
    t0 = time.time()
    print('\n=== Stage 5: Evaluation and analysis ===')

    if args.defer_test and not args.stress:
        eval_dict = {'accuracy': 0, 'n_samples': 0, 'source': 'defer_test_no_stress'}
    elif args.defer_test and len(stress_summary) > 0:
        eval_dict = _extract_clean_metrics_from_stress_details(stress_details)
        if not eval_dict:
            eval_dict = {'accuracy': 0, 'n_samples': 0, 'source': 'stress_test_clean_condition'}
    elif test_loader is not None:
        test_metrics = compute_metrics_on_loader(trainer, test_loader, compute_loss=True)
        eval_dict = {
            'accuracy': test_metrics.get('accuracy', 0),
            'balanced_accuracy': test_metrics.get('balanced_accuracy'),
            'loss': test_metrics.get('loss'),
            'precision': test_metrics.get('precision'),
            'recall': test_metrics.get('recall'),
            'f1': test_metrics.get('f1'),
            'macro_precision': test_metrics.get('macro_precision'),
            'macro_recall': test_metrics.get('macro_recall'),
            'macro_f1': test_metrics.get('macro_f1'),
            'per_class': test_metrics.get('per_class', {}),
            'confusion_matrix': test_metrics.get('confusion_matrix', []),
            'uncertainty': test_metrics.get('uncertainty', {}),
            'n_samples': len(test_samples), 'source': 'test_loader',
        }
    else:
        eval_dict = {'accuracy': 0, 'n_samples': 0, 'source': 'none'}

    eval_dict['val_clean_accuracy'] = float(best_val_acc) if best_epoch is not None else 0.0
    eval_dict['val_macro_f1']       = float(best_val_f1) if best_epoch is not None else 0.0
    eval_dict['best_epoch'] = best_epoch
    if len(val_stress_summary) > 0:
        clean_vs = val_stress_summary[val_stress_summary['condition'] == 'clean'] \
            if 'condition' in val_stress_summary.columns else pd.DataFrame()
        if len(clean_vs) > 0:
            eval_dict['val_stress_clean_acc'] = float(clean_vs['acc_gating_on'].iloc[0])
        eval_dict['val_stress_mean_delta'] = float(val_stress_summary['delta_pp'].mean())

    with open(os.path.join(run_dir, 'eval_metrics.json'), 'w') as f:
        json.dump(eval_dict, f, indent=2)

    # Compact, human-readable reports.
    try:
        from reporting import write_all_reports
        write_all_reports(eval_dict, stress_summary, run_dir)
    except Exception as e:
        print(f'  [report] failed: {e}')

    print(f'  Test accuracy: {eval_dict["accuracy"]:.1f}%')
    print(f'  Val-Clean accuracy: {best_val_acc:.1f}%')
    if 'val_stress_mean_delta' in eval_dict:
        print(f'  Val-Stress mean delta: {eval_dict["val_stress_mean_delta"]:.2f} pp')
    for k in ['loss', 'precision', 'recall', 'f1', 'macro_f1', 'balanced_accuracy']:
        if eval_dict.get(k) is not None:
            print(f'  {k.capitalize()}: {eval_dict[k]:.4f}')

    _save_comprehensive_analysis(
        model, val_loader, test_loader, val_samples, test_samples,
        history, stress_summary, stress_details, run_dir, device)

    # Natural occlusion evaluation (DMD annotations: eyes_occluded, mouth_occluded)
    if args.natural_occlusion_eval and test_samples:
        print('\n--- Natural Occlusion Evaluation ---')
        nat_occ_samples = [
            s for s in test_samples
            if s.get('ground_truth', {}).get('eyes_occluded')
            or s.get('ground_truth', {}).get('mouth_occluded')
        ]
        if nat_occ_samples:
            nat_ds = DriverStateDataset(nat_occ_samples, device=str(device),
                                        gate_supervision=args.gate_supervision)
            nat_loader = DataLoader(nat_ds, batch_size=args.batch, shuffle=False,
                                    collate_fn=nat_ds.collate_samples)
            nat_metrics = compute_metrics_on_loader(trainer, nat_loader, compute_loss=False)
            nat_result = {
                'accuracy': nat_metrics.get('accuracy', 0),
                'n_samples': len(nat_occ_samples),
                'n_eyes_occluded': sum(
                    1 for s in nat_occ_samples
                    if s.get('ground_truth', {}).get('eyes_occluded')),
                'n_mouth_occluded': sum(
                    1 for s in nat_occ_samples
                    if s.get('ground_truth', {}).get('mouth_occluded')),
            }
            with open(os.path.join(run_dir, 'natural_occlusion_eval.json'), 'w') as f:
                json.dump(nat_result, f, indent=2)
            print(f'  Natural occ samples: {nat_result["n_samples"]} '
                  f'(eyes={nat_result["n_eyes_occluded"]}, '
                  f'mouth={nat_result["n_mouth_occluded"]})')
            print(f'  Natural occ accuracy: {nat_result["accuracy"]:.1f}%')
            eval_dict['natural_occ_accuracy'] = nat_result['accuracy']
            eval_dict['natural_occ_n'] = nat_result['n_samples']
        else:
            print('  No natural occlusion samples found in test set.')

    # Occlusion visualization
    occ_png = os.path.join(run_dir, 'occlusion_visualization.png')
    if generate_occlusion_grid_png(csv_data, occ_png,
                                   face_detector=face_detector, face_type=args.face):
        print(f'Saved occlusion grid to {occ_png}')

    opacity_png = os.path.join(run_dir, 'clean_vs_synthetic_opacity_grid.png')
    if generate_clean_vs_synthetic_opacity_grid(
            csv_data, opacity_png, face_detector=face_detector, face_type=args.face):
        print(f'Saved clean-vs-synthetic opacity grid to {opacity_png}')

    # Latency benchmark
    if args.benchmark:
        if feat_extractor is None:
            feat_extractor, occ_model, face_detector = _load_extraction_models(device, args)
        from stress_test import run_latency_benchmark
        test_keys_bench = splits['test'] if splits['test'] else list(csv_data.keys())[:1]
        latency = run_latency_benchmark(
            model, face_detector, feat_extractor, occ_model,
            csv_data, test_keys_bench,
            device=str(device), num_warmup=20, num_iter=50)
        with open(os.path.join(run_dir, 'latency_report.json'), 'w') as f:
            json.dump(latency, f, indent=2)
        print('\n--- Latency benchmark ---')
        for k, v in latency.items():
            if isinstance(v, dict) and 'mean_ms' in v:
                print(f'  {k}: {v["mean_ms"]:.2f} +/- {v.get("std_ms", 0):.2f} ms')
        _delete_models(feat_extractor, occ_model, face_detector)

    stage_times['stage5_evaluation'] = time.time() - t0
    print(f'  [Stage 5 completed in {stage_times["stage5_evaluation"]:.1f}s]')

    _gpu_cleanup('pipeline complete')

    total_elapsed = time.time() - t_pipeline_start
    stage_times['total_elapsed'] = total_elapsed
    with open(os.path.join(run_dir, 'timing.json'), 'w') as f:
        json.dump({k: round(v, 2) for k, v in stage_times.items()}, f, indent=2)

    print(f'\n{"="*50}')
    print('TIMING SUMMARY')
    print(f'{"="*50}')
    for k, v in stage_times.items():
        if k == 'total_elapsed':
            continue
        m, s = divmod(v, 60)
        h, m = divmod(m, 60)
        label = k.replace('_', ' ').title()
        if h > 0:
            print(f'  {label:.<40s} {int(h)}h {int(m)}m {s:.0f}s')
        elif m > 0:
            print(f'  {label:.<40s} {int(m)}m {s:.0f}s')
        else:
            print(f'  {label:.<40s} {s:.1f}s')
    m, s = divmod(total_elapsed, 60)
    h, m = divmod(m, 60)
    print(f'  {"─"*40}')
    if h > 0:
        print(f'  {"Total":.<40s} {int(h)}h {int(m)}m {s:.0f}s')
    else:
        print(f'  {"Total":.<40s} {int(m)}m {s:.0f}s')
    print(f'{"="*50}')

    print(f'\nDone. All results saved to {run_dir}')


if __name__ == '__main__':
    main()
