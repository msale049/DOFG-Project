#!/usr/bin/env python3
"""
run_crossval.py
===============
Sequential subject-wise cross-validation runner and aggregator for the DOFG
pipeline. It launches per-fold runs via ``run_train_eval.py`` and then writes
mean/std/CI summaries suitable for paper reporting.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from data_loading import load_csv_video_data
from metrics_utils import summarize_fold_metric
from split_generator import get_subject_ids


def _timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _parse_fold_list(spec: str | None, max_folds: int) -> List[int]:
    if not spec:
        return list(range(max_folds))
    folds = []
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        fold = int(token)
        if fold < 0 or fold >= max_folds:
            raise ValueError(f'fold {fold} outside valid range [0, {max_folds - 1}]')
        folds.append(fold)
    return sorted(set(folds))


def _flatten_eval_metrics(eval_metrics: Dict, split_info: Dict, fold_dir: str) -> Dict:
    row = {
        'fold': split_info.get('fold'),
        'run_dir': fold_dir,
        'test_subjects': ','.join(split_info.get('test_subjects', [])),
        'train_subjects': ','.join(split_info.get('train_subjects', [])),
    }
    scalar_keys = [
        'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1',
        'macro_precision', 'macro_recall', 'macro_f1',
        'val_clean_accuracy', 'val_stress_clean_acc', 'val_stress_mean_delta',
        'gating_off_accuracy', 'clean_delta_pp', 'clean_delta_p_value',
        'n_samples',
    ]
    for key in scalar_keys:
        if key in eval_metrics:
            row[key] = eval_metrics[key]

    per_class = eval_metrics.get('per_class', {})
    for cls_name, cls_metrics in per_class.items():
        for metric_name, value in cls_metrics.items():
            row[f'per_class.{cls_name}.{metric_name}'] = value
    return row


def _aggregate_scalar_table(df: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        summary = summarize_fold_metric(df[metric].tolist())
        summary['metric'] = metric
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    cols = ['metric', 'n', 'mean', 'std', 'ci_low', 'ci_high', 'min', 'max']
    return pd.DataFrame(rows)[cols]


def _aggregate_grouped_metrics(df: pd.DataFrame,
                               group_cols: List[str],
                               metric_cols: List[str]) -> pd.DataFrame:
    rows = []
    for group_vals, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        base = dict(zip(group_cols, group_vals))
        base['n_folds'] = int(sub['fold'].nunique()) if 'fold' in sub.columns else len(sub)
        for metric in metric_cols:
            if metric not in sub.columns:
                continue
            summary = summarize_fold_metric(sub[metric].tolist())
            base[f'{metric}_mean'] = summary['mean']
            base[f'{metric}_std'] = summary['std']
            base[f'{metric}_ci_low'] = summary['ci_low']
            base[f'{metric}_ci_high'] = summary['ci_high']
        rows.append(base)
    return pd.DataFrame(rows)


def _collect_fold_outputs(parent_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_metric_rows = []
    stress_rows = []
    per_class_rows = []

    for name in sorted(os.listdir(parent_dir)):
        fold_dir = os.path.join(parent_dir, name)
        if not os.path.isdir(fold_dir) or not name.startswith('fold_'):
            continue

        eval_path = os.path.join(fold_dir, 'eval_metrics.json')
        split_path = os.path.join(fold_dir, 'split_info.json')
        if not os.path.exists(eval_path):
            continue

        with open(eval_path) as f:
            eval_metrics = json.load(f)
        if os.path.exists(split_path):
            with open(split_path) as f:
                split_info = json.load(f)
        else:
            split_info = {
                'fold': int(name.split('_')[-1]) if name.split('_')[-1].isdigit() else None,
                'train_subjects': [],
                'test_subjects': [],
            }

        fold_metric_rows.append(_flatten_eval_metrics(eval_metrics, split_info, fold_dir))

        stress_path = os.path.join(fold_dir, 'stress_test_summary.csv')
        if os.path.exists(stress_path):
            sdf = pd.read_csv(stress_path)
            sdf['fold'] = split_info.get('fold')
            stress_rows.append(sdf)

        per_class_path = os.path.join(fold_dir, 'stress_test_per_class_deltas.csv')
        if os.path.exists(per_class_path):
            pdf = pd.read_csv(per_class_path)
            pdf['fold'] = split_info.get('fold')
            per_class_rows.append(pdf)

    return (
        pd.DataFrame(fold_metric_rows),
        pd.concat(stress_rows, ignore_index=True) if stress_rows else pd.DataFrame(),
        pd.concat(per_class_rows, ignore_index=True) if per_class_rows else pd.DataFrame(),
    )


def aggregate_crossval_results(parent_dir: str, mode: str, k: int) -> Dict:
    fold_metrics_df, stress_df, per_class_df = _collect_fold_outputs(parent_dir)
    if len(fold_metrics_df) == 0:
        raise RuntimeError(f'No completed fold outputs found in {parent_dir}')

    fold_metrics_path = os.path.join(parent_dir, 'crossval_fold_metrics.csv')
    fold_metrics_df.to_csv(fold_metrics_path, index=False)

    scalar_metrics = [
        'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1',
        'macro_precision', 'macro_recall', 'macro_f1',
        'val_clean_accuracy', 'val_stress_clean_acc', 'val_stress_mean_delta',
        'gating_off_accuracy', 'clean_delta_pp',
    ]
    scalar_summary_df = _aggregate_scalar_table(fold_metrics_df, scalar_metrics)
    scalar_summary_path = os.path.join(parent_dir, 'crossval_metric_summary.csv')
    scalar_summary_df.to_csv(scalar_summary_path, index=False)

    clean_per_class_cols = [c for c in fold_metrics_df.columns if c.startswith('per_class.')]
    if clean_per_class_cols:
        clean_per_class_summary_df = _aggregate_scalar_table(fold_metrics_df, clean_per_class_cols)
        clean_per_class_summary_df.to_csv(
            os.path.join(parent_dir, 'crossval_clean_per_class_summary.csv'), index=False)

    if len(stress_df) > 0:
        stress_df.to_csv(os.path.join(parent_dir, 'crossval_stress_per_fold.csv'), index=False)
        metric_cols = [c for c in [
            'acc_gating_on', 'acc_gating_off', 'delta_pp',
            'delta_ci_low', 'delta_ci_high', 'p_value_mcnemar',
        ] if c in stress_df.columns]
        stress_summary_df = _aggregate_grouped_metrics(
            stress_df, ['condition', 'regime', 'opacity'], metric_cols)
        if 'p_value_mcnemar' in stress_df.columns:
            sig = stress_df.groupby(['condition', 'regime', 'opacity'])['p_value_mcnemar'] \
                .apply(lambda s: float((s < 0.05).mean()))
            stress_summary_df = stress_summary_df.merge(
                sig.rename('fraction_folds_significant').reset_index(),
                on=['condition', 'regime', 'opacity'],
                how='left',
            )
        stress_summary_df.to_csv(
            os.path.join(parent_dir, 'crossval_stress_summary.csv'), index=False)

    if len(per_class_df) > 0:
        per_class_df.to_csv(os.path.join(parent_dir, 'crossval_per_class_stress_per_fold.csv'), index=False)
        metric_cols = [
            'delta_EyeClosed', 'delta_Yawn', 'delta_Neutral',
            'delta_overall', 'delta_macro', 'delta_non_neutral',
        ]
        metric_cols = [c for c in metric_cols if c in per_class_df.columns]
        per_class_summary_df = _aggregate_grouped_metrics(
            per_class_df, ['condition', 'regime', 'opacity'], metric_cols)
        per_class_summary_df.to_csv(
            os.path.join(parent_dir, 'crossval_per_class_stress_summary.csv'), index=False)

    summary_json = {
        'mode': mode,
        'k': k,
        'parent_dir': parent_dir,
        'n_completed_folds': int(fold_metrics_df['fold'].nunique()),
        'metric_summary': (
            scalar_summary_df.set_index('metric').to_dict(orient='index')
            if len(scalar_summary_df) > 0 else {}
        ),
    }
    with open(os.path.join(parent_dir, 'crossval_summary.json'), 'w') as f:
        json.dump(summary_json, f, indent=2)
    return summary_json


def main():
    ap = argparse.ArgumentParser(description='Sequential k-fold / LOSO runner for DOFG')
    ap.add_argument('--data', default='Data', help='Path to dataset root')
    ap.add_argument('--mode', choices=['kfold', 'loso'], default='kfold')
    ap.add_argument('--k', type=int, default=5, help='Number of folds for k-fold')
    ap.add_argument('--results-root', default='results', help='Root directory for cross-val outputs')
    ap.add_argument('--run-name', default=None, help='Optional parent directory name')
    ap.add_argument('--folds', default=None, help='Comma-separated fold indices to run (default: all)')
    ap.add_argument('--skip-existing', action='store_true',
                    help='Skip folds that already have eval_metrics.json')
    ap.add_argument('--continue-on-error', action='store_true',
                    help='Continue running remaining folds if one fold fails')
    ap.add_argument('--aggregate-only', action='store_true',
                    help='Skip execution and aggregate an existing parent directory')
    ap.add_argument('--parent-dir', default=None,
                    help='Existing parent directory to aggregate (used with --aggregate-only)')
    args, passthrough = ap.parse_known_args()

    if args.aggregate_only:
        if not args.parent_dir:
            ap.error('--aggregate-only requires --parent-dir')
        summary = aggregate_crossval_results(args.parent_dir, args.mode, args.k)
        print(json.dumps(summary, indent=2))
        return

    os.makedirs(args.results_root, exist_ok=True)
    parent_dir = os.path.join(
        args.results_root,
        args.run_name or f'{args.mode}_{args.k if args.mode == "kfold" else "loso"}_{_timestamp()}',
    )
    os.makedirs(parent_dir, exist_ok=True)

    csv_data = load_csv_video_data(args.data, filter_eye_states=True)
    subject_ids = get_subject_ids(csv_data)
    fold_count = args.k if args.mode == 'kfold' else len(subject_ids)
    fold_ids = _parse_fold_list(args.folds, fold_count)

    runner_meta = {
        'mode': args.mode,
        'k': args.k,
        'fold_ids': fold_ids,
        'results_root': args.results_root,
        'parent_dir': parent_dir,
        'passthrough_args': passthrough,
        'python': sys.executable,
    }
    with open(os.path.join(parent_dir, 'crossval_runner_config.json'), 'w') as f:
        json.dump(runner_meta, f, indent=2)

    script_path = os.path.join(os.path.dirname(__file__), 'run_train_eval.py')
    failures = []
    for fold in fold_ids:
        fold_name = f'fold_{fold:02d}'
        fold_dir = os.path.join(parent_dir, fold_name)
        if args.skip_existing and os.path.exists(os.path.join(fold_dir, 'eval_metrics.json')):
            print(f'Skipping existing {fold_name}')
            continue

        cmd = [
            sys.executable,
            script_path,
        ]
        cmd.extend(passthrough)
        cmd.extend([
            '--data', args.data,
            '--mode', args.mode,
            '--results-root', parent_dir,
            '--run-name', fold_name,
            '--fold', str(fold),
        ])
        if args.mode == 'kfold':
            cmd.extend(['--k', str(args.k)])

        print('\n' + '=' * 80)
        print(f'Running {fold_name}: {" ".join(cmd)}')
        print('=' * 80)
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            failures.append({'fold': fold, 'returncode': proc.returncode})
            print(f'Fold {fold} failed with exit code {proc.returncode}')
            if not args.continue_on_error:
                break

    summary = aggregate_crossval_results(parent_dir, args.mode, args.k)
    if failures:
        summary['failures'] = failures
        with open(os.path.join(parent_dir, 'crossval_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    print('\nCross-validation summary:')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
