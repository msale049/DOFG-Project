#!/usr/bin/env python3
"""
aggregate_gate_floor_sweep.py
=============================
Aggregate results from a gate_floor parametric sweep and produce
publication-quality plots.

Supports two directory layouts:
  - Fixed split:  sweep_dir/floor_X.XX/{eval_metrics.json, stress_test_details.csv, ...}
  - K-fold:       sweep_dir/floor_X.XX/fold_NN/{eval_metrics.json, stress_test_details.csv, ...}

Outputs:
  - gate_floor_sweep_summary.csv     (one row per gate_floor; mean across folds if kfold)
  - gate_floor_sweep_plot.png        (4-panel: accuracy, delta, stress acc, per-class F1)
  - gate_floor_non_neutral_plot.png  (4-panel: non-neutral F1 metrics)

Usage:
    python aggregate_gate_floor_sweep.py --sweep-dir results/gate_floor_sweep
    python aggregate_gate_floor_sweep.py --sweep-dir results/gate_floor_sweep_kfold
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from sklearn.metrics import f1_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

sns.set_style('whitegrid')

LABEL_MAP = {'EyeClosed': 0, 'Yawn': 1, 'Neutral': 2}
NON_NEUTRAL_CLASSES = ['EyeClosed', 'Yawn']


def _save_confusion_matrix_from_eval(eval_metrics: Dict, run_dir: str) -> None:
    cm = eval_metrics.get('confusion_matrix')
    if not cm:
        return
    label_names = ['EyeClosed', 'Yawn', 'Neutral']
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(np.asarray(cm), annot=True, fmt='g', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set(xlabel='Predicted', ylabel='True', title='Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'confusion_matrix.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


def _save_per_class_metrics_from_eval(eval_metrics: Dict, run_dir: str) -> None:
    per_class = eval_metrics.get('per_class') or {}
    label_names = ['EyeClosed', 'Yawn', 'Neutral']
    rows = []
    for name in label_names:
        row = per_class.get(name)
        if row is None:
            return
        rows.append(row)

    prec = [float(r.get('precision', 0.0)) for r in rows]
    rec = [float(r.get('recall', 0.0)) for r in rows]
    f1 = [float(r.get('f1', 0.0)) for r in rows]
    sup = [int(r.get('support', 0)) for r in rows]

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
    for i, support in enumerate(sup):
        ax.text(x[i] + w, min(1.02, f1[i] + 0.02), f'n={support}', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'per_class_metrics.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


def generate_run_level_plots(run_dir: str) -> None:
    """Backfill fold/run-level plots from saved artifacts only."""
    eval_path = os.path.join(run_dir, 'eval_metrics.json')
    history_path = os.path.join(run_dir, 'training_history.json')
    stress_summary_path = os.path.join(run_dir, 'stress_test_summary.csv')
    stress_details_path = os.path.join(run_dir, 'stress_test_details.csv')

    if not os.path.exists(eval_path):
        return

    with open(eval_path) as f:
        eval_metrics = json.load(f)

    history = {}
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)

    stress_summary = pd.read_csv(stress_summary_path) if os.path.exists(stress_summary_path) else pd.DataFrame()
    stress_details = pd.read_csv(stress_details_path) if os.path.exists(stress_details_path) else pd.DataFrame()

    from run_train_eval import (
        _plot_gating_comparison,
        _plot_gates_vs_opacity,
        _plot_opacity_analysis,
        _save_per_class_stress_deltas,
        _save_per_class_stress_tables,
        _save_stress_heatmap,
        _save_training_curves,
        _save_training_dynamics,
        _save_val_gate_statistics,
        _save_val_loss_components,
    )

    if history:
        _save_training_curves(history, run_dir)
        _save_training_dynamics(history, run_dir)
        _save_val_loss_components(history, run_dir)
        _save_val_gate_statistics(history, run_dir)

    _save_confusion_matrix_from_eval(eval_metrics, run_dir)
    _save_per_class_metrics_from_eval(eval_metrics, run_dir)

    if len(stress_summary) > 0:
        _save_stress_heatmap(stress_summary, run_dir, details_df=stress_details)
        _plot_gating_comparison(stress_summary, run_dir)
        _plot_opacity_analysis(stress_summary, run_dir)

    if len(stress_details) > 0:
        _plot_gates_vs_opacity(stress_details, run_dir)
        _save_per_class_stress_deltas(stress_details, run_dir)
        _save_per_class_stress_tables(stress_details_path, run_dir)


# ─── Per-condition F1 from stress_test_details.csv ───────────────────────────

def _compute_non_neutral_f1_from_details(details_df: pd.DataFrame) -> Dict:
    """
    Compute non-neutral F1 metrics from a stress_test_details DataFrame.

    Returns dict with keys:
      nn_f1_all_stress_on    - mean non-neutral F1 (gating ON) across all stress conditions
      nn_f1_all_stress_off   - mean non-neutral F1 (gating OFF) across all stress conditions
      nn_f1_full_occ_on      - mean non-neutral F1 (gating ON) for opacity==1.0 only
      nn_f1_full_occ_off     - mean non-neutral F1 (gating OFF) for opacity==1.0 only
      nn_df1_all_stress      - mean non-neutral delta-F1 (ON - OFF) all stress
      nn_df1_full_occ        - mean non-neutral delta-F1 (ON - OFF) opacity==1.0
    """
    result = {k: float('nan') for k in [
        'nn_f1_all_stress_on', 'nn_f1_all_stress_off',
        'nn_f1_full_occ_on', 'nn_f1_full_occ_off',
        'nn_df1_all_stress', 'nn_df1_full_occ',
    ]}
    if not _HAS_SKLEARN or details_df is None or len(details_df) == 0:
        return result

    cond_col = 'condition' if 'condition' in details_df.columns else 'occlusion_type'
    occluded = details_df[details_df[cond_col] != 'clean'].copy()
    if len(occluded) == 0:
        return result

    nn_mask = occluded['class_label'].isin(NON_NEUTRAL_CLASSES)
    occluded_nn = occluded[nn_mask]

    def _per_condition_nn_f1(df_subset: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Return (list_of_f1_on, list_of_f1_off) per condition."""
        f1_on_list, f1_off_list = [], []
        for cond, grp in df_subset.groupby(cond_col):
            if len(grp) < 2:
                continue
            y_true = grp['gt_label'].values
            pred_on = grp['pred_gating_on'].values
            pred_off = grp['pred_gating_off'].values
            labels = sorted(set(y_true))
            if len(labels) < 2:
                f1_on = float(f1_score(y_true, pred_on, labels=labels, average='macro', zero_division=0))
                f1_off = float(f1_score(y_true, pred_off, labels=labels, average='macro', zero_division=0))
            else:
                f1_on = float(f1_score(y_true, pred_on, labels=labels, average='macro', zero_division=0))
                f1_off = float(f1_score(y_true, pred_off, labels=labels, average='macro', zero_division=0))
            f1_on_list.append(f1_on)
            f1_off_list.append(f1_off)
        return f1_on_list, f1_off_list

    # All stress conditions (non-neutral samples only)
    f1_on_all, f1_off_all = _per_condition_nn_f1(occluded_nn)
    if f1_on_all:
        result['nn_f1_all_stress_on'] = float(np.mean(f1_on_all))
        result['nn_f1_all_stress_off'] = float(np.mean(f1_off_all))
        deltas = [a - b for a, b in zip(f1_on_all, f1_off_all)]
        result['nn_df1_all_stress'] = float(np.mean(deltas))

    # Full occlusion only (opacity == 1.0)
    full_occ = occluded_nn[occluded_nn['opacity'] == 1.0]
    if len(full_occ) > 0:
        f1_on_fo, f1_off_fo = _per_condition_nn_f1(full_occ)
        if f1_on_fo:
            result['nn_f1_full_occ_on'] = float(np.mean(f1_on_fo))
            result['nn_f1_full_occ_off'] = float(np.mean(f1_off_fo))
            deltas_fo = [a - b for a, b in zip(f1_on_fo, f1_off_fo)]
            result['nn_df1_full_occ'] = float(np.mean(deltas_fo))

    return result


# ─── Single-run loader ───────────────────────────────────────────────────────

def _load_run(run_dir: str) -> Dict:
    """Load config, eval metrics, stress summary, and stress details from a run."""
    config_path = os.path.join(run_dir, 'config.json')
    eval_path = os.path.join(run_dir, 'eval_metrics.json')
    stress_summary_path = os.path.join(run_dir, 'stress_test_summary.csv')
    stress_details_path = os.path.join(run_dir, 'stress_test_details.csv')

    if not os.path.exists(eval_path):
        return {}

    with open(config_path) as f:
        config = json.load(f)
    with open(eval_path) as f:
        eval_metrics = json.load(f)

    stress_summary = pd.read_csv(stress_summary_path) if os.path.exists(stress_summary_path) else pd.DataFrame()
    stress_details = pd.read_csv(stress_details_path) if os.path.exists(stress_details_path) else pd.DataFrame()

    return {
        'config': config,
        'eval': eval_metrics,
        'stress_summary': stress_summary,
        'stress_details': stress_details,
        'run_dir': run_dir,
    }


def _extract_row(run_data: Dict) -> Dict:
    """Extract a flat summary row from one run's data."""
    cfg = run_data['config']
    ev = run_data['eval']
    stress = run_data['stress_summary']
    details = run_data['stress_details']

    gate_floor = cfg.get('gate_floor', cfg.get('eye_floor'))

    row = {
        'gate_floor': gate_floor,
        'run_dir': run_data['run_dir'],
        'clean_accuracy': ev.get('accuracy', float('nan')),
        'balanced_accuracy': ev.get('balanced_accuracy', float('nan')),
        'macro_f1': ev.get('macro_f1', float('nan')),
        'weighted_f1': ev.get('f1', float('nan')),
        'gating_off_accuracy': ev.get('gating_off_accuracy', float('nan')),
        'clean_delta_pp': ev.get('clean_delta_pp', float('nan')),
        'val_clean_accuracy': ev.get('val_clean_accuracy', float('nan')),
        'n_samples': ev.get('n_samples', 0),
    }

    per_class = ev.get('per_class', {})
    for cls in ['EyeClosed', 'Yawn', 'Neutral']:
        cls_data = per_class.get(cls, {})
        row[f'{cls}_f1'] = cls_data.get('f1', float('nan'))
        row[f'{cls}_precision'] = cls_data.get('precision', float('nan'))
        row[f'{cls}_recall'] = cls_data.get('recall', float('nan'))

    uncertainty = ev.get('uncertainty', {})
    for metric_name in ['accuracy', 'balanced_accuracy', 'macro_f1']:
        u = uncertainty.get(metric_name, {})
        row[f'{metric_name}_ci_low'] = u.get('ci_low', float('nan'))
        row[f'{metric_name}_ci_high'] = u.get('ci_high', float('nan'))

    if len(stress) > 0:
        cond_col = 'condition' if 'condition' in stress.columns else 'occlusion_type'
        occluded = stress[stress[cond_col] != 'clean'] if 'condition' in stress.columns else stress[stress['opacity'] > 0]
        if len(occluded) > 0:
            row['mean_stress_delta_pp'] = occluded['delta_pp'].mean()
            row['mean_stress_acc_on'] = occluded['acc_gating_on'].mean()
            row['mean_stress_acc_off'] = occluded['acc_gating_off'].mean()
        else:
            row['mean_stress_delta_pp'] = float('nan')
            row['mean_stress_acc_on'] = float('nan')
            row['mean_stress_acc_off'] = float('nan')

        for regime, opacity in [('persistent_eye', 1.0), ('persistent_mouth', 1.0), ('persistent_both', 1.0)]:
            cond_name = f'{regime}@{opacity}'
            match = stress[stress[cond_col] == cond_name]
            if len(match) > 0:
                row[f'{regime}_{opacity}_acc_on'] = match['acc_gating_on'].iloc[0]
                row[f'{regime}_{opacity}_delta'] = match['delta_pp'].iloc[0]
            else:
                row[f'{regime}_{opacity}_acc_on'] = float('nan')
                row[f'{regime}_{opacity}_delta'] = float('nan')

        for opacity_val in [0.4, 0.6, 0.8, 1.0]:
            sub = occluded[occluded['opacity'] == opacity_val] if len(occluded) > 0 else pd.DataFrame()
            if len(sub) > 0:
                row[f'stress_delta_op{opacity_val}'] = sub['delta_pp'].mean()
                row[f'stress_acc_on_op{opacity_val}'] = sub['acc_gating_on'].mean()
            else:
                row[f'stress_delta_op{opacity_val}'] = float('nan')
                row[f'stress_acc_on_op{opacity_val}'] = float('nan')
    else:
        for k in ['mean_stress_delta_pp', 'mean_stress_acc_on', 'mean_stress_acc_off']:
            row[k] = float('nan')

    # Non-neutral F1 metrics from stress details
    nn_metrics = _compute_non_neutral_f1_from_details(details)
    row.update(nn_metrics)

    return row


# ─── Directory scanning (supports fixed + kfold layouts) ─────────────────────

def _detect_layout(sweep_dir: str) -> str:
    """Detect whether sweep_dir contains fixed runs or kfold runs."""
    for name in os.listdir(sweep_dir):
        floor_dir = os.path.join(sweep_dir, name)
        if not os.path.isdir(floor_dir) or not name.startswith('floor_'):
            continue
        if os.path.exists(os.path.join(floor_dir, 'eval_metrics.json')):
            return 'fixed'
        for sub in os.listdir(floor_dir):
            if sub.startswith('fold_') and os.path.isdir(os.path.join(floor_dir, sub)):
                return 'kfold'
    return 'fixed'


def collect_sweep_fixed(sweep_dir: str) -> pd.DataFrame:
    """Scan sweep directory with fixed-split layout."""
    rows = []
    for name in sorted(os.listdir(sweep_dir)):
        run_dir = os.path.join(sweep_dir, name)
        if not os.path.isdir(run_dir) or not name.startswith('floor_'):
            continue
        run_data = _load_run(run_dir)
        if not run_data:
            print(f'  Skipping {name}: no eval_metrics.json')
            continue
        row = _extract_row(run_data)
        rows.append(row)
        print(f'  Loaded floor={row["gate_floor"]:.2f}: '
              f'acc={row["clean_accuracy"]:.1f}%, '
              f'macro_f1={row["macro_f1"]:.4f}, '
              f'stress_delta={row.get("mean_stress_delta_pp", float("nan")):.2f} pp')

    if not rows:
        print(f'No completed runs found in {sweep_dir}')
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('gate_floor').reset_index(drop=True)


def collect_sweep_kfold(sweep_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scan sweep directory with k-fold layout.

    Returns:
      per_fold_df  - one row per (gate_floor, fold)
      summary_df   - one row per gate_floor with mean/std/ci across folds
    """
    per_fold_rows = []
    for name in sorted(os.listdir(sweep_dir)):
        floor_dir = os.path.join(sweep_dir, name)
        if not os.path.isdir(floor_dir) or not name.startswith('floor_'):
            continue

        for fold_name in sorted(os.listdir(floor_dir)):
            fold_dir = os.path.join(floor_dir, fold_name)
            if not os.path.isdir(fold_dir) or not fold_name.startswith('fold_'):
                continue
            run_data = _load_run(fold_dir)
            if not run_data:
                continue
            row = _extract_row(run_data)
            fold_idx = int(fold_name.split('_')[-1]) if fold_name.split('_')[-1].isdigit() else -1
            row['fold'] = fold_idx
            per_fold_rows.append(row)

        n_folds = sum(1 for r in per_fold_rows if abs(r['gate_floor'] - float(name.replace('floor_', ''))) < 1e-6)
        floor_val = float(name.replace('floor_', ''))
        print(f'  Loaded floor={floor_val:.2f}: {n_folds} folds')

    if not per_fold_rows:
        print(f'No completed fold runs found in {sweep_dir}')
        return pd.DataFrame(), pd.DataFrame()

    per_fold_df = pd.DataFrame(per_fold_rows).sort_values(['gate_floor', 'fold']).reset_index(drop=True)

    # Aggregate across folds
    agg_cols = [c for c in per_fold_df.columns
                if c not in ('gate_floor', 'fold', 'run_dir')
                and per_fold_df[c].dtype in (np.float64, np.int64, float, int)]
    summary_rows = []
    for gf, grp in per_fold_df.groupby('gate_floor'):
        row = {'gate_floor': gf, 'n_folds': len(grp)}
        for col in agg_cols:
            vals = grp[col].dropna().values
            if len(vals) == 0:
                row[col] = float('nan')
                row[f'{col}_std'] = float('nan')
                row[f'{col}_ci_low'] = float('nan')
                row[f'{col}_ci_high'] = float('nan')
                continue
            mean = float(np.mean(vals))
            row[col] = mean
            if len(vals) > 1:
                std = float(np.std(vals, ddof=1))
                sem = std / math.sqrt(len(vals))
                from scipy.stats import t as student_t
                ci_low, ci_high = student_t.interval(0.95, df=len(vals)-1, loc=mean, scale=sem)
                row[f'{col}_std'] = std
                row[f'{col}_ci_low'] = float(ci_low)
                row[f'{col}_ci_high'] = float(ci_high)
            else:
                row[f'{col}_std'] = 0.0
                row[f'{col}_ci_low'] = mean
                row[f'{col}_ci_high'] = mean
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values('gate_floor').reset_index(drop=True)
    return per_fold_df, summary_df


# ─── Plotting helpers ────────────────────────────────────────────────────────

COLORS = {
    'primary': '#1976D2',
    'secondary': '#D32F2F',
    'accent1': '#388E3C',
    'accent2': '#F57C00',
    'accent3': '#7B1FA2',
    'neutral': '#455A64',
}
MARKER_KW = dict(markersize=7, linewidth=2.2, markeredgecolor='white', markeredgewidth=1)


def _plot_line(ax, floors, values, style, color, label, annotate=True, fmt='.1f', **kw):
    """Plot a line with optional per-point annotations."""
    merged_kw = {**MARKER_KW, **kw}
    ax.plot(floors, values, style, color=color, label=label, **merged_kw)
    if annotate:
        for x, y in zip(floors, values):
            if not np.isnan(y):
                text = f'{y:{fmt}}'
                ax.annotate(text, (x, y), textcoords='offset points',
                            xytext=(0, 9), fontsize=7, ha='center', color=color)


def _plot_line_with_ci(ax, floors, values, ci_low, ci_high, style, color, label,
                       annotate=True, fmt='.1f', **kw):
    """Plot a line with shaded CI band."""
    _plot_line(ax, floors, values, style, color, label, annotate=annotate, fmt=fmt, **kw)
    valid = ~(np.isnan(ci_low) | np.isnan(ci_high))
    if valid.any():
        ax.fill_between(np.array(floors)[valid], np.array(ci_low)[valid],
                        np.array(ci_high)[valid], alpha=0.15, color=color)


def _has_ci(df: pd.DataFrame, col: str) -> bool:
    lo, hi = f'{col}_ci_low', f'{col}_ci_high'
    return lo in df.columns and hi in df.columns and df[lo].notna().any()


def _format_floor_labels(floors: np.ndarray) -> List[str]:
    return [f'{float(f):.2f}' for f in floors]


def _set_floor_axis(ax, x_pos: np.ndarray, floors: np.ndarray):
    """Use evenly spaced discrete x positions for sweep values."""
    ax.set_xticks(x_pos)
    ax.set_xticklabels(_format_floor_labels(floors))
    ax.set_xlim(x_pos[0] - 0.2, x_pos[-1] + 0.2)


def _title_suffix(df: pd.DataFrame, is_kfold: bool) -> str:
    if not is_kfold:
        return ''
    if 'n_folds' in df.columns and df['n_folds'].notna().any():
        max_folds = int(df['n_folds'].max())
        if max_folds <= 1:
            return ' (single fold)'
    return ' (mean ± CI across folds)'


def plot_sweep(df: pd.DataFrame, output_path: str, is_kfold: bool = False):
    """Generate the 4-panel gate_floor sweep figure (accuracy / delta / stress / per-class F1)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    floors = df['gate_floor'].values
    x_pos = np.arange(len(floors), dtype=float)
    title_suffix = _title_suffix(df, is_kfold)

    # ── Plot A: Clean Accuracy and Macro F1 ─────────────────────────────────
    ax = axes[0, 0]
    if _has_ci(df, 'clean_accuracy'):
        _plot_line_with_ci(ax, x_pos, df['clean_accuracy'].values,
                           df['clean_accuracy_ci_low'].values, df['clean_accuracy_ci_high'].values,
                           'o-', COLORS['primary'], 'Clean Test Accuracy (%)')
    else:
        _plot_line(ax, x_pos, df['clean_accuracy'].values, 'o-', COLORS['primary'],
                   'Clean Test Accuracy (%)')

    macro_f1_pct = df['macro_f1'].values * 100
    if _has_ci(df, 'macro_f1'):
        _plot_line_with_ci(ax, x_pos, macro_f1_pct,
                           df['macro_f1_ci_low'].values * 100, df['macro_f1_ci_high'].values * 100,
                           's-', COLORS['secondary'], 'Macro F1 (%)')
    else:
        _plot_line(ax, x_pos, macro_f1_pct, 's-', COLORS['secondary'], 'Macro F1 (%)')

    ax.set_xlabel('Gate Floor', fontsize=11)
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_title(f'(a) Clean-Test Accuracy & Macro F1{title_suffix}', fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    _set_floor_axis(ax, x_pos, floors)
    ax.grid(True, alpha=0.3)
    y_vals = [v for v in list(df['clean_accuracy']) + list(macro_f1_pct) if not np.isnan(v)]
    if y_vals:
        margin = max(3, (max(y_vals) - min(y_vals)) * 0.25)
        ax.set_ylim(max(0, min(y_vals) - margin), min(100, max(y_vals) + margin))

    # ── Plot B: Mean Stress Delta ────────────────────────────────────────────
    ax = axes[0, 1]
    if 'mean_stress_delta_pp' in df.columns and df['mean_stress_delta_pp'].notna().any():
        if _has_ci(df, 'mean_stress_delta_pp'):
            _plot_line_with_ci(ax, x_pos, df['mean_stress_delta_pp'].values,
                               df['mean_stress_delta_pp_ci_low'].values,
                               df['mean_stress_delta_pp_ci_high'].values,
                               'D-', COLORS['accent1'], 'Mean Delta (all occluded)', fmt='+.2f')
        else:
            _plot_line(ax, x_pos, df['mean_stress_delta_pp'].values,
                       'D-', COLORS['accent1'], 'Mean Delta (all occluded)', fmt='+.2f')

    for opacity_val, style, lbl_color in [(1.0, '^--', COLORS['secondary']), (0.8, 'v--', COLORS['accent2'])]:
        col = f'stress_delta_op{opacity_val}'
        if col in df.columns and df[col].notna().any():
            ax.plot(x_pos, df[col].values, style, color=lbl_color,
                    label=f'Delta @ opacity={opacity_val}', markersize=5, linewidth=1.5, alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Gate Floor', fontsize=11)
    ax.set_ylabel('ΔAccuracy (pp)', fontsize=11)
    ax.set_title(f'(b) Stress ΔAccuracy (ON−OFF){title_suffix}', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    _set_floor_axis(ax, x_pos, floors)
    ax.grid(True, alpha=0.3)

    # ── Plot C: Stress Accuracy under Heavy Occlusion ────────────────────────
    ax = axes[1, 0]
    for col, label, color, style in [
        ('persistent_eye_1.0_acc_on', 'Persistent Eye @1.0', COLORS['primary'], 'o-'),
        ('persistent_mouth_1.0_acc_on', 'Persistent Mouth @1.0', COLORS['secondary'], 's-'),
        ('persistent_both_1.0_acc_on', 'Persistent Both @1.0', COLORS['accent3'], 'D-'),
    ]:
        if col in df.columns and df[col].notna().any():
            if _has_ci(df, col):
                _plot_line_with_ci(ax, x_pos, df[col].values,
                                   df[f'{col}_ci_low'].values, df[f'{col}_ci_high'].values,
                                   style, color, label)
            else:
                _plot_line(ax, x_pos, df[col].values, style, color, label)

    ax.set_xlabel('Gate Floor', fontsize=11)
    ax.set_ylabel('Accuracy (%) — Gating ON', fontsize=11)
    ax.set_title(f'(c) Stress Accuracy at Full Occlusion{title_suffix}', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    _set_floor_axis(ax, x_pos, floors)
    ax.grid(True, alpha=0.3)

    # ── Plot D: Per-class F1 on Clean Test ───────────────────────────────────
    ax = axes[1, 1]
    for col, label, color, style in [
        ('EyeClosed_f1', 'EyeClosed', COLORS['primary'], 'o-'),
        ('Yawn_f1', 'Yawn', COLORS['secondary'], 's-'),
        ('Neutral_f1', 'Neutral', COLORS['neutral'], 'D-'),
    ]:
        if col in df.columns and df[col].notna().any():
            vals = df[col].values * 100
            if _has_ci(df, col):
                _plot_line_with_ci(ax, x_pos, vals,
                                   df[f'{col}_ci_low'].values * 100, df[f'{col}_ci_high'].values * 100,
                                   style, color, label)
            else:
                _plot_line(ax, x_pos, vals, style, color, label)

    ax.set_xlabel('Gate Floor', fontsize=11)
    ax.set_ylabel('F1 (%)', fontsize=11)
    ax.set_title(f'(d) Per-Class Clean F1{title_suffix}', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    _set_floor_axis(ax, x_pos, floors)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Gate Floor Parametric Study', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'  Saved plot to {output_path}')


def plot_non_neutral(df: pd.DataFrame, output_path: str, is_kfold: bool = False):
    """Generate the 4-panel non-neutral F1 figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    floors = df['gate_floor'].values
    x_pos = np.arange(len(floors), dtype=float)
    title_suffix = _title_suffix(df, is_kfold)

    configs = [
        {
            'ax': axes[0, 0],
            'col': 'nn_f1_all_stress_on',
            'title': '(a) Mean Non-Neutral F1 Across All Stress Conditions',
            'ylabel': 'F1',
            'style': 'o-',
            'color': COLORS['primary'],
            'label': 'Gating ON',
            'fmt': '.3f',
            'extra_col': 'nn_f1_all_stress_off',
            'extra_label': 'Gating OFF',
            'extra_color': COLORS['secondary'],
            'extra_style': 's--',
        },
        {
            'ax': axes[0, 1],
            'col': 'nn_f1_full_occ_on',
            'title': '(b) Mean Non-Neutral F1 Under Full Occlusion',
            'ylabel': 'F1',
            'style': 'o-',
            'color': COLORS['primary'],
            'label': 'Gating ON',
            'fmt': '.3f',
            'extra_col': 'nn_f1_full_occ_off',
            'extra_label': 'Gating OFF',
            'extra_color': COLORS['secondary'],
            'extra_style': 's--',
        },
        {
            'ax': axes[1, 0],
            'col': 'nn_df1_all_stress',
            'title': '(c) Mean Non-Neutral ΔF1 Across All Stress Conditions',
            'ylabel': 'ΔF1 (ON − OFF)',
            'style': 'D-',
            'color': COLORS['accent1'],
            'label': 'ΔF1',
            'fmt': '+.4f',
            'hline': True,
        },
        {
            'ax': axes[1, 1],
            'col': 'nn_df1_full_occ',
            'title': '(d) Mean Non-Neutral ΔF1 Under Full Occlusion',
            'ylabel': 'ΔF1 (ON − OFF)',
            'style': 'D-',
            'color': COLORS['accent1'],
            'label': 'ΔF1',
            'fmt': '+.4f',
            'hline': True,
        },
    ]

    for cfg in configs:
        ax = cfg['ax']
        col = cfg['col']
        if col not in df.columns or df[col].isna().all():
            ax.set_title(cfg['title'], fontsize=11, fontweight='bold')
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, color='gray')
            continue

        if _has_ci(df, col):
            _plot_line_with_ci(ax, x_pos, df[col].values,
                               df[f'{col}_ci_low'].values, df[f'{col}_ci_high'].values,
                               cfg['style'], cfg['color'], cfg['label'], fmt=cfg['fmt'])
        else:
            _plot_line(ax, x_pos, df[col].values, cfg['style'], cfg['color'],
                       cfg['label'], fmt=cfg['fmt'])

        extra_col = cfg.get('extra_col')
        if extra_col and extra_col in df.columns and df[extra_col].notna().any():
            if _has_ci(df, extra_col):
                _plot_line_with_ci(ax, x_pos, df[extra_col].values,
                                   df[f'{extra_col}_ci_low'].values, df[f'{extra_col}_ci_high'].values,
                                   cfg['extra_style'], cfg['extra_color'], cfg['extra_label'],
                                   fmt=cfg['fmt'])
            else:
                _plot_line(ax, x_pos, df[extra_col].values,
                           cfg['extra_style'], cfg['extra_color'], cfg['extra_label'],
                           fmt=cfg['fmt'])

        if cfg.get('hline'):
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

        ax.set_xlabel('Gate Floor', fontsize=11)
        ax.set_ylabel(cfg['ylabel'], fontsize=11)
        ax.set_title(f'{cfg["title"]}{title_suffix}', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        _set_floor_axis(ax, x_pos, floors)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Non-Neutral Metrics vs Gate Floor', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'  Saved non-neutral plot to {output_path}')


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Aggregate gate_floor sweep results')
    ap.add_argument('--sweep-dir', required=True,
                    help='Path to the sweep directory (e.g. results/gate_floor_sweep)')
    ap.add_argument('--output-csv', default=None,
                    help='Output CSV path (default: <sweep-dir>/gate_floor_sweep_summary.csv)')
    ap.add_argument('--output-plot', default=None,
                    help='Output plot path (default: <sweep-dir>/gate_floor_sweep_plot.png)')
    args = ap.parse_args()

    sweep_dir = args.sweep_dir
    if not os.path.isdir(sweep_dir):
        print(f'Sweep directory not found: {sweep_dir}')
        sys.exit(1)

    csv_path = args.output_csv or os.path.join(sweep_dir, 'gate_floor_sweep_summary.csv')
    plot_path = args.output_plot or os.path.join(sweep_dir, 'gate_floor_sweep_plot.png')
    nn_plot_path = os.path.join(sweep_dir, 'gate_floor_non_neutral_plot.png')

    layout = _detect_layout(sweep_dir)
    print(f'Scanning {sweep_dir} (layout: {layout})...')

    if layout == 'kfold':
        per_fold_df, df = collect_sweep_kfold(sweep_dir)
        if len(df) == 0:
            sys.exit(1)
        per_fold_csv = os.path.join(sweep_dir, 'gate_floor_sweep_per_fold.csv')
        per_fold_df.to_csv(per_fold_csv, index=False)
        print(f'\nPer-fold data saved to {per_fold_csv}')
        is_kfold = True
        run_dirs = per_fold_df['run_dir'].dropna().unique().tolist() if 'run_dir' in per_fold_df.columns else []
    else:
        df = collect_sweep_fixed(sweep_dir)
        if len(df) == 0:
            sys.exit(1)
        is_kfold = False
        run_dirs = df['run_dir'].dropna().unique().tolist() if 'run_dir' in df.columns else []

    if run_dirs:
        print(f'\nGenerating per-run plots for {len(run_dirs)} run(s)...')
        for run_dir in run_dirs:
            try:
                generate_run_level_plots(run_dir)
            except Exception as e:
                print(f'  Warning: per-run plot generation failed for {run_dir}: {e}')

    df.to_csv(csv_path, index=False)
    print(f'\nSummary saved to {csv_path}')

    display_cols = ['gate_floor', 'clean_accuracy', 'macro_f1', 'mean_stress_delta_pp',
                    'nn_f1_all_stress_on', 'nn_df1_all_stress',
                    'nn_f1_full_occ_on', 'nn_df1_full_occ']
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].to_string(index=False))

    if len(df) >= 2:
        plot_sweep(df, plot_path, is_kfold=is_kfold)
        plot_non_neutral(df, nn_plot_path, is_kfold=is_kfold)
    else:
        print('  Need at least 2 completed runs for plotting.')

    print('\nDone.')


if __name__ == '__main__':
    main()
