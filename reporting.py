"""
reporting.py
============
Compact, readable reporting utilities for stress-test runs.

Produces three small artifacts per run that are easier to eyeball than the raw
``stress_test_details.csv`` / ``stress_test_summary.csv`` dump:

- ``stress_summary_compact.csv`` — one row per regime, averaged over opacity.
- ``overall_deltas.csv`` — one row, the headline numbers:
    clean_on, clean_off, stress_mean_on, stress_mean_off,
    delta_clean_pp, delta_stress_pp, n_stress_conditions
- ``run_report.txt`` — plain-text headline for quick ``cat`` inspection.

The helpers are pure-Python (pandas only) and safe to call with empty inputs.
"""

from __future__ import annotations

import json
import os
from typing import Dict

import pandas as pd


def _safe_mean(series: pd.Series) -> float:
    try:
        return float(series.mean()) if len(series) else float('nan')
    except Exception:
        return float('nan')


def write_compact_stress_summary(
    stress_summary: pd.DataFrame,
    run_dir: str,
) -> Dict[str, str]:
    """Group stress_summary by regime and emit a compact CSV.

    Input columns used (if present): regime, opacity, acc_gating_on,
    acc_gating_off, delta_pp, n, p_value_mcnemar.

    Output columns (one row per regime):
      regime, n_conditions, opacities, mean_acc_on, mean_acc_off,
      mean_delta_pp, min_delta_pp, max_delta_pp, mean_n_samples.

    Returns a dict of written paths.
    """
    paths: Dict[str, str] = {}
    if stress_summary is None or len(stress_summary) == 0:
        return paths

    df = stress_summary.copy()
    for col in ('acc_gating_on', 'acc_gating_off', 'delta_pp', 'n', 'opacity'):
        if col not in df.columns:
            df[col] = float('nan')

    rows = []
    for regime, sub in df.groupby('regime', sort=False):
        opacities = sorted(set(float(x) for x in sub['opacity'].tolist()
                               if x == x))
        rows.append({
            'regime':         regime,
            'n_conditions':   int(len(sub)),
            'opacities':      ','.join(f'{o:g}' for o in opacities),
            'mean_acc_on':    round(_safe_mean(sub['acc_gating_on']), 2),
            'mean_acc_off':   round(_safe_mean(sub['acc_gating_off']), 2),
            'mean_delta_pp':  round(_safe_mean(sub['delta_pp']), 2),
            'min_delta_pp':   round(float(sub['delta_pp'].min()), 2)
                              if len(sub) else float('nan'),
            'max_delta_pp':   round(float(sub['delta_pp'].max()), 2)
                              if len(sub) else float('nan'),
            'mean_n_samples': int(sub['n'].mean()) if len(sub) else 0,
        })
    out = pd.DataFrame(rows)

    # Order: clean first, then other regimes alphabetically.
    out['__sort_key'] = out['regime'].apply(lambda r: (r != 'clean', r))
    out = out.sort_values('__sort_key').drop(columns='__sort_key').reset_index(drop=True)

    out_path = os.path.join(run_dir, 'stress_summary_compact.csv')
    out.to_csv(out_path, index=False)
    paths['compact'] = out_path
    return paths


def write_overall_deltas(
    eval_dict: Dict,
    stress_summary: pd.DataFrame,
    run_dir: str,
) -> Dict[str, str]:
    """Emit one-row ``overall_deltas.csv`` + plain-text ``run_report.txt``.

    The headline numbers the reader actually wants:
      clean_on, clean_off, delta_clean_pp,
      stress_mean_on, stress_mean_off, delta_stress_pp,
      val_clean_accuracy, best_epoch, macro_f1.
    """
    paths: Dict[str, str] = {}

    clean_on = float('nan')
    clean_off = float('nan')
    stress_on = float('nan')
    stress_off = float('nan')

    if stress_summary is not None and len(stress_summary):
        clean_rows = stress_summary[stress_summary.get('regime', '') == 'clean']
        if len(clean_rows):
            clean_on  = float(clean_rows['acc_gating_on'].iloc[0])
            clean_off = float(clean_rows['acc_gating_off'].iloc[0])
        stress_rows = stress_summary[stress_summary.get('regime', '') != 'clean']
        if len(stress_rows):
            stress_on  = float(stress_rows['acc_gating_on'].mean())
            stress_off = float(stress_rows['acc_gating_off'].mean())

    delta_clean  = (clean_on  - clean_off)  if clean_on  == clean_on  and clean_off == clean_off else float('nan')
    delta_stress = (stress_on - stress_off) if stress_on == stress_on and stress_off == stress_off else float('nan')

    row = {
        'val_clean_accuracy': round(float(eval_dict.get('val_clean_accuracy', float('nan'))), 2),
        'best_epoch':         eval_dict.get('best_epoch'),
        'test_accuracy':      round(float(eval_dict.get('accuracy', float('nan'))), 2),
        'test_macro_f1':      round(float(eval_dict.get('macro_f1', float('nan'))), 4)
                              if eval_dict.get('macro_f1') is not None else float('nan'),
        'clean_on_pp':        round(clean_on, 2) if clean_on == clean_on else float('nan'),
        'clean_off_pp':       round(clean_off, 2) if clean_off == clean_off else float('nan'),
        'delta_clean_pp':     round(delta_clean, 2) if delta_clean == delta_clean else float('nan'),
        'stress_mean_on_pp':  round(stress_on, 2) if stress_on == stress_on else float('nan'),
        'stress_mean_off_pp': round(stress_off, 2) if stress_off == stress_off else float('nan'),
        'delta_stress_pp':    round(delta_stress, 2) if delta_stress == delta_stress else float('nan'),
        'n_stress_conditions': int(len(stress_summary) - 1) if (stress_summary is not None
                                                                and len(stress_summary) > 0) else 0,
    }
    out_path = os.path.join(run_dir, 'overall_deltas.csv')
    pd.DataFrame([row]).to_csv(out_path, index=False)
    paths['overall'] = out_path

    # Readable text report.
    lines = [
        '=' * 58,
        f' Run report: {os.path.basename(run_dir.rstrip("/"))}',
        '=' * 58,
        f'  Val-clean accuracy : {row["val_clean_accuracy"]} %   '
        f'(best epoch {row["best_epoch"]})',
        f'  Test accuracy       : {row["test_accuracy"]} %',
        f'  Test macro F1       : {row["test_macro_f1"]}',
        '',
        '  -- Clean stress-test condition --',
        f'    gating ON         : {row["clean_on_pp"]} %',
        f'    gating OFF        : {row["clean_off_pp"]} %',
        f'    delta ON-OFF      : {row["delta_clean_pp"]} pp   '
        '(should be ≈ 0 — this is the invariance check)',
        '',
        '  -- Mean across occluded stress conditions --',
        f'    gating ON         : {row["stress_mean_on_pp"]} %',
        f'    gating OFF        : {row["stress_mean_off_pp"]} %',
        f'    delta ON-OFF      : {row["delta_stress_pp"]} pp   '
        '(positive = gating helps)',
        f'    n conditions      : {row["n_stress_conditions"]}',
        '=' * 58,
    ]
    txt_path = os.path.join(run_dir, 'run_report.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    paths['report'] = txt_path
    return paths


def write_all_reports(
    eval_dict: Dict,
    stress_summary: pd.DataFrame,
    run_dir: str,
) -> Dict[str, str]:
    """Convenience wrapper: write both compact summary and overall deltas."""
    paths: Dict[str, str] = {}
    try:
        paths.update(write_compact_stress_summary(stress_summary, run_dir))
    except Exception as e:
        print(f'  [report] compact summary failed: {e}')
    try:
        paths.update(write_overall_deltas(eval_dict, stress_summary, run_dir))
    except Exception as e:
        print(f'  [report] overall deltas failed: {e}')
    return paths
