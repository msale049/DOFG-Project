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
import sys
import time
from datetime import datetime

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('ORT_LOG_LEVEL', '3')  # suppress ONNX Runtime thread affinity warnings
os.environ.setdefault('ONNXRUNTIME_SESSION_THREAD_POOL_SIZE', '1')

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
from trainer_enhanced import TinyTransformerTrainer
from evaluation import compute_metrics_on_loader, collect_eval_with_occlusion
from stress_test import run_stress_test
from visualize_occlusion import generate_occlusion_grid_png


# ─── GPU memory helpers ──────────────────────────────────────────────────────

def _gpu_cleanup(msg: str = ''):
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if msg:
        allocated = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        print(f'  [GPU cleanup] {msg} — {allocated:.0f} MB allocated')


def _delete_models(*models):
    """Delete model objects and run GPU cleanup."""
    for m in models:
        if m is not None:
            del m
    _gpu_cleanup('models freed')


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


def _ensure_results_dir() -> str:
    """Create results/run_YYYYMMDD_HHMMSS and return path."""
    os.makedirs('results', exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('results', f'run_{stamp}')
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
    """Plot and save training loss and accuracy curves."""
    plt, _ = _import_plt()
    if plt is None:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    epochs = np.arange(1, len(history.get('epoch_losses', [])) + 1)
    if epochs.size > 0:
        ax1.plot(epochs, history.get('epoch_losses', []), 'o-')
        ax1.set(xlabel='Epoch', ylabel='Loss', title='Training Loss')
        ax1.grid(True)
        ax2.plot(epochs, history.get('accuracies', []), 's-', label='Train')
        ax2.set(xlabel='Epoch', ylabel='Accuracy (%)', title='Training Accuracy')
        ax2.legend()
        ax2.grid(True)
    fig.suptitle('Training Curves', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'training_curves.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


def _save_training_dynamics(history: dict, run_dir: str):
    """Plot loss components and gate statistics across epochs."""
    plt, _ = _import_plt()
    if plt is None:
        return
    epochs = np.arange(1, len(history.get('epoch_losses', [])) + 1)
    if epochs.size == 0:
        return

    has_gates = bool(history.get('mean_eye_gate'))
    n_rows = 3 if has_gates else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), sharex=True)

    axes[0].plot(epochs, history.get('epoch_losses', []), 'o-', lw=2, label='Total Loss')
    if 'classification_losses' in history:
        axes[0].plot(epochs, history['classification_losses'], 's--', lw=1.5, label='Classification')
    if 'gate_occ_losses' in history:
        axes[0].plot(epochs, history['gate_occ_losses'], '^--', lw=1.5, label='Gate Alignment')
    axes[0].set(ylabel='Loss', title='Training Dynamics — Loss Components')
    axes[0].legend()
    axes[0].grid(True)

    if 'accuracies' in history:
        axes[1].plot(epochs, history['accuracies'], 'D-', lw=2, color='green')
        axes[1].set(ylabel='Accuracy (%)', title='Training Accuracy')
        axes[1].grid(True)

    if has_gates:
        axes[2].plot(epochs, history['mean_eye_gate'], 'o-', lw=2, label='Mean Eye Gate')
        axes[2].plot(epochs, history['mean_mouth_gate'], 's-', lw=2, label='Mean Mouth Gate')
        axes[2].axhline(y=1.0, color='gray', ls=':', lw=1, label='Max (clean target)')
        axes[2].axhline(y=0.05, color='red', ls=':', lw=1, label='Floor (0.05)')
        axes[2].set(xlabel='Epoch', ylabel='Gate Value', title='Gate Statistics Across Training')
        axes[2].legend()
        axes[2].grid(True)
    else:
        axes[-1].set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'training_dynamics.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


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

    if len(stress_summary) > 0:
        _save_stress_heatmap(stress_summary, run_dir, details_df=stress_details)

    summary = {
        'n_train': len(val_samples) if not test_samples else 0,
        'n_eval': len(eval_df),
        'eval_set': eval_label,
        'overall_accuracy': float(eval_df['is_correct'].mean() * 100) if len(eval_df) else 0,
    }
    if len(eval_df) > 0:
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
    feat_extractor = ResNet34FeatureExtractor(
        CONFIG['RESNET34_MODEL_PATH'], device=str(device))
    _gpu_cleanup('feature extractor loaded')

    occ_model = ResNet34OcclusionModel(
        CONFIG['RESNET34_OCCLUSION_MODEL_PATH'], device=str(device))

    det_size = (args.det_size, args.det_size) if args.face == 'retina' else None
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
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--stress', action='store_true', default=True)
    ap.add_argument('--no-stress', action='store_false', dest='stress')
    ap.add_argument('--stress-frames', type=int, default=20)
    ap.add_argument('--stress-opacities', type=str, default='0.4,0.6,0.8,1.0')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--defer-test', action='store_true', default=True)
    ap.add_argument('--no-defer-test', action='store_false', dest='defer_test')
    ap.add_argument('--benchmark', action='store_true')
    ap.add_argument('--face-cpu', action='store_true')
    ap.add_argument('--class-weighted', action='store_true',
                    help='Use class-weighted cross-entropy loss')
    args = ap.parse_args()

    if args.face_cpu:
        os.environ['DOFG_FACE_CPU'] = '1'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = _ensure_results_dir()
    print(f'Results dir: {run_dir}')
    print(f'Device: {device}')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.cuda.empty_cache()
        print(f'GPU: {torch.cuda.get_device_name(0)} '
              f'({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)')
    print(f'Mode: {args.mode}, samples/video: {args.samples or "ALL"}, epochs: {args.epochs}')

    config_dict = {
        'data': args.data, 'samples_per_video': args.samples or 'all',
        'epochs': args.epochs, 'batch_size': args.batch, 'mode': args.mode,
        'k': args.k, 'num_test': args.num_test, 'fold': args.fold,
        'face_detector': args.face, 'det_size': args.det_size if args.face == 'retina' else None,
        'face_cpu': args.face_cpu, 'seed': args.seed,
        'stress_test': args.stress, 'stress_frames': args.stress_frames,
        'defer_test': args.defer_test, 'strategy': args.strategy,
        'class_weighted': args.class_weighted,
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
            train_samples, val_samples, test_samples, test_clips, val_clips = extract_features_for_clips(
                csv_data, split_config,
                face_detector=face_detector, feat_extractor=feat_extractor,
                occ_model=occ_model, val_ratio=0.20, seed=args.seed,
                max_train_clips=args.max_train_clips,
                max_val_clips=args.max_val_clips,
                skip_test=args.defer_test)
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
    train_ds = DriverStateDataset(train_samples, device=str(device))
    val_ds = DriverStateDataset(val_samples, device=str(device))
    test_loader = None
    if test_samples:
        test_ds = DriverStateDataset(test_samples, device=str(device))
        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                                 collate_fn=test_ds.collate_samples)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=train_ds.collate_samples, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            collate_fn=val_ds.collate_samples)

    model = EnhancedOcclusionAwareTransformer(
        feature_dim=512, hidden_dim=128, num_heads=4,
        num_classes=3, num_layers=3, use_relative_pos=True,
    ).to(device)
    print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    class_weights = None
    if args.class_weighted and train_samples:
        label_counts = Counter(s['label'] for s in train_samples)
        total = sum(label_counts.values())
        n_classes = len(label_counts)
        weights = [total / (n_classes * label_counts.get(i, 1)) for i in range(n_classes)]
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f'  Class weights: {[f"{w:.2f}" for w in weights]}')

    trainer = TinyTransformerTrainer(model, device=str(device), learning_rate=3e-5,
                                    class_weights=class_weights)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch=epoch)
        val_metrics = trainer.evaluate(val_loader, name='VAL')
        val_acc = val_metrics.get('accuracy', 0.0)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(run_dir, 'model_best.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f'  ** Best val: {best_val_acc:.1f}% -> saved')
        print(f'  Ep {epoch + 1}/{args.epochs}: train_acc={train_metrics.get("accuracy", 0):.1f}% '
              f'val_acc={val_acc:.1f}%')

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
    model.load_state_dict(ckpt)
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

        if len(stress_details) > 0:
            stress_details.to_csv(os.path.join(run_dir, 'stress_test_details.csv'), index=False)
            stress_summary.to_csv(os.path.join(run_dir, 'stress_test_summary.csv'), index=False)
            print(f'\nTest-Stress: {len(stress_details)} detail rows')
            print(stress_summary.to_string(index=False))
            _plot_gating_comparison(stress_summary, run_dir)
            _plot_opacity_analysis(stress_summary, run_dir)
            _plot_gates_vs_opacity(stress_details, run_dir)
            _save_per_class_stress_deltas(stress_details, run_dir)
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
        clean_row = stress_summary[stress_summary.get('condition', stress_summary.columns[0]) == 'clean'] \
            if 'condition' in stress_summary.columns else pd.DataFrame()
        if len(clean_row) > 0:
            eval_dict = {
                'accuracy': float(clean_row['acc_gating_on'].iloc[0]),
                'n_samples': int(clean_row['n'].iloc[0]),
                'source': 'stress_test_clean_condition',
            }
        else:
            eval_dict = {'accuracy': 0, 'n_samples': 0, 'source': 'stress_test_clean_condition'}
    elif test_loader is not None:
        test_metrics = compute_metrics_on_loader(trainer, test_loader, compute_loss=True)
        eval_dict = {
            'accuracy': test_metrics.get('accuracy', 0),
            'loss': test_metrics.get('loss'),
            'precision': test_metrics.get('precision'),
            'recall': test_metrics.get('recall'),
            'f1': test_metrics.get('f1'),
            'n_samples': len(test_samples), 'source': 'test_loader',
        }
    else:
        eval_dict = {'accuracy': 0, 'n_samples': 0, 'source': 'none'}

    eval_dict['val_clean_accuracy'] = best_val_acc
    if len(val_stress_summary) > 0:
        clean_vs = val_stress_summary[val_stress_summary['condition'] == 'clean'] \
            if 'condition' in val_stress_summary.columns else pd.DataFrame()
        if len(clean_vs) > 0:
            eval_dict['val_stress_clean_acc'] = float(clean_vs['acc_gating_on'].iloc[0])
        eval_dict['val_stress_mean_delta'] = float(val_stress_summary['delta_pp'].mean())

    with open(os.path.join(run_dir, 'eval_metrics.json'), 'w') as f:
        json.dump(eval_dict, f, indent=2)
    print(f'  Test accuracy: {eval_dict["accuracy"]:.1f}%')
    print(f'  Val-Clean accuracy: {best_val_acc:.1f}%')
    if 'val_stress_mean_delta' in eval_dict:
        print(f'  Val-Stress mean delta: {eval_dict["val_stress_mean_delta"]:.2f} pp')
    for k in ['loss', 'precision', 'recall', 'f1']:
        if eval_dict.get(k) is not None:
            print(f'  {k.capitalize()}: {eval_dict[k]:.4f}')

    _save_comprehensive_analysis(
        model, val_loader, test_loader, val_samples, test_samples,
        history, stress_summary, stress_details, run_dir, device)

    # Occlusion visualization
    if face_detector is not None:
        occ_png = os.path.join(run_dir, 'occlusion_visualization.png')
        if generate_occlusion_grid_png(csv_data, occ_png,
                                       face_detector=face_detector, face_type=args.face):
            print(f'Saved occlusion grid to {occ_png}')

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
