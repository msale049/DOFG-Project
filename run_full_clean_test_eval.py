#!/usr/bin/env python3
"""
run_full_clean_test_eval.py
===========================
Evaluate one or more saved checkpoints on the full clean test split using the
same data loading, clip formation, feature extraction, and model inference path
as the main training pipeline.

For each checkpoint, results are written to:
    <run_dir>/clean-test-results/

The evaluator is intentionally post-hoc and read-only with respect to the
original run directory contents outside the new output folder.
"""

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Tuple

_mpl_cache_dir = '/tmp/codex-mplconfig'
os.makedirs(_mpl_cache_dir, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', _mpl_cache_dir)

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ablation_utils import disable_gates_at_inference
from config import CLIP_CONFIG
from data_loading import load_csv_video_data
from datasets import DriverStateDataset
from evaluation import collect_eval_with_occlusion
from metrics_utils import (
    compute_classification_metrics,
    compute_paired_binary_statistics,
    compute_classification_uncertainty,
)
from mlp_baseline import RegionFeatureMLP
from pipeline import (
    extract_clean_test_features_for_clips,
    extract_features_stratified,
)
from resnet_baseline import ResNet34Baseline
from run_train_eval import (
    _delete_models,
    _gpu_cleanup,
    _load_extraction_models,
    _require_cuda,
    _save_attention_heatmap,
    _save_confusion_matrix,
    _save_gate_distributions,
    _save_gate_response_curves,
    _save_per_class_metrics,
    _save_per_subject_accuracy,
    _seed_worker,
    _set_global_seed,
    _splits_from_fixed_or_fold,
)
from transformer_enhanced import EnhancedOcclusionAwareTransformer


LABEL_NAMES = ['EyeClosed', 'Yawn', 'Neutral']
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LAUNCH_CWD = os.getcwd()


@contextmanager
def _nullcontext():
    yield


def _resolve_path(path: str, base_dir: str = PROJECT_ROOT) -> str:
    """Resolve *path* against *base_dir* when it is relative."""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def _load_json(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def _resolve_checkpoint_and_run_dir(path: str) -> Tuple[str, str]:
    """
    Accept either a checkpoint path or a run directory path and return
    (checkpoint_path, run_dir).
    """
    abs_path = path if os.path.isabs(path) else os.path.abspath(os.path.join(LAUNCH_CWD, path))
    if os.path.isdir(abs_path):
        checkpoint_path = os.path.join(abs_path, 'model_best.pt')
        run_dir = abs_path
    else:
        checkpoint_path = abs_path
        run_dir = os.path.dirname(abs_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    config_path = os.path.join(run_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Run config not found beside checkpoint: {config_path}')

    return checkpoint_path, run_dir


def _load_run_spec(path: str) -> Dict:
    """Load one model checkpoint spec from a run directory/checkpoint path."""
    checkpoint_path, run_dir = _resolve_checkpoint_and_run_dir(path)
    config = _load_json(os.path.join(run_dir, 'config.json'))

    data_path = _resolve_path(config.get('data', 'Data'))
    model_type = config.get('model', 'transformer')
    strategy = config.get('strategy', 'clip')
    face_detector = config.get('face_detector', 'retina')
    det_size = config.get('det_size', 640 if face_detector == 'retina' else None)
    seed = int(config.get('seed', 42))
    mode = config.get('mode', 'fixed')
    fold = int(config.get('fold', 0))
    k = int(config.get('k', 5))
    num_test = int(config.get('num_test', 3))
    clip_length = int(config.get('clip_length', 32))
    batch_size = int(config.get('batch_size', 16))

    return {
        'checkpoint_path': checkpoint_path,
        'run_dir': run_dir,
        'config': config,
        'data_path': data_path,
        'model_type': model_type,
        'strategy': strategy,
        'face_detector': face_detector,
        'det_size': det_size,
        'face_cpu': bool(config.get('face_cpu', False)),
        'seed': seed,
        'mode': mode,
        'fold': fold,
        'k': k,
        'num_test': num_test,
        'clip_length': clip_length,
        'batch_size': batch_size,
        'needs_face_crop': model_type == 'resnet_baseline',
        'gate_supervision': config.get('gate_supervision', 'gt'),
    }


def _build_model(spec: Dict, device: torch.device):
    """Rebuild the saved model architecture from the run config."""
    cfg = spec['config']
    model_type = spec['model_type']

    if model_type == 'mlp_baseline':
        model = RegionFeatureMLP(
            feature_dim=512,
            hidden_dim=512,
            num_classes=3,
            dropout=0.3,
        )
    elif model_type == 'resnet_baseline':
        model = ResNet34Baseline(num_classes=3, dropout=0.3)
    else:
        model = EnhancedOcclusionAwareTransformer(
            feature_dim=512,
            hidden_dim=128,
            num_heads=4,
            num_classes=3,
            num_layers=3,
            use_relative_pos=True,
            gate_floor=float(cfg.get('gate_floor', 0.05)),
            eye_floor=float(cfg.get('eye_floor', cfg.get('gate_floor', 0.05))),
            mouth_floor=float(cfg.get('mouth_floor', cfg.get('gate_floor', 0.05))),
        )

    state = torch.load(spec['checkpoint_path'], map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _make_test_loader(test_samples: List[Dict],
                      batch_size: int,
                      device: torch.device,
                      gate_supervision: str,
                      seed: int):
    """Create the clean test DataLoader for one evaluated model."""
    test_ds = DriverStateDataset(
        test_samples,
        device=str(device),
        gate_supervision=gate_supervision,
    )
    test_generator = torch.Generator()
    test_generator.manual_seed(seed + 2)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_ds.collate_samples,
        worker_init_fn=_seed_worker,
        generator=test_generator,
    )
    return test_ds, test_loader


def _split_info_from_rebuild(csv_data: Dict, spec: Dict) -> Dict:
    """Rebuild the split metadata exactly as the training script does."""
    splits = _splits_from_fixed_or_fold(
        csv_data,
        mode=spec['mode'],
        k=spec['k'],
        num_test=spec['num_test'],
        seed=spec['seed'],
        fold=spec['fold'],
    )
    split_config = splits.get('split_config', {})
    return {
        'mode': spec['mode'],
        'fold': spec['fold'],
        'k': spec['k'],
        'seed': spec['seed'],
        'train_subjects': split_config.get('train_subjects', []),
        'test_subjects': split_config.get('test_subjects', []),
        'train_videos': splits.get('train', []),
        'test_videos': splits.get('test', []),
        'split_config': split_config,
    }


def _extract_full_clean_test_samples(csv_data: Dict,
                                     spec: Dict,
                                     face_detector,
                                     feat_extractor,
                                     occ_model,
                                     max_test_clips=None) -> Tuple[List[Dict], Dict]:
    """
    Extract the full clean test set for one run config using the same split and
    pipeline definitions as the main training script.
    """
    split_info = _split_info_from_rebuild(csv_data, spec)
    split_config = split_info['split_config']

    original_clip_length = CLIP_CONFIG.get('T', 32)
    CLIP_CONFIG['T'] = spec['clip_length']
    try:
        if spec['strategy'] == 'clip':
            test_samples, test_clips = extract_clean_test_features_for_clips(
                csv_data=csv_data,
                split_config=split_config,
                face_detector=face_detector,
                feat_extractor=feat_extractor,
                occ_model=occ_model,
                val_ratio=0.20,
                seed=spec['seed'],
                max_test_clips=max_test_clips,
                save_face_crops=spec['needs_face_crop'],
            )
            split_info['n_test_clips'] = len(test_clips)
        else:
            # Fallback for legacy runs. This path reuses the existing extraction
            # function even though it also rebuilds train/val internally.
            splits = {
                'train': split_info['train_videos'],
                'val': split_info['train_videos'],
                'test': split_info['test_videos'],
            }
            _, _, test_samples = extract_features_stratified(
                csv_data=csv_data,
                splits=splits,
                face_detector=face_detector,
                feat_extractor=feat_extractor,
                occ_model=occ_model,
                num_samples_per_video=None,
                val_ratio=0.20,
                random_state=spec['seed'],
            )
            split_info['n_test_clips'] = None
    finally:
        CLIP_CONFIG['T'] = original_clip_length

    split_info['n_test_samples'] = len(test_samples)
    return test_samples, split_info


def _build_eval_dataframe(model,
                          test_loader,
                          test_samples: List[Dict],
                          gating_mode: str = 'on') -> pd.DataFrame:
    """Run inference and enrich the resulting per-sample dataframe."""
    if gating_mode not in {'on', 'off'}:
        raise ValueError(f'Unsupported gating_mode: {gating_mode}')

    ctx = disable_gates_at_inference(model) if gating_mode == 'off' else _nullcontext()
    with ctx:
        eval_df = collect_eval_with_occlusion(model, test_loader)
    if len(eval_df) == 0:
        return eval_df

    if len(test_samples) == len(eval_df):
        eval_df['subject'] = [s.get('subject', 'unknown') for s in test_samples]
        eval_df['video_key'] = [s.get('video_key', 'unknown') for s in test_samples]
        eval_df['frame_id'] = [s.get('frame_id', -1) for s in test_samples]

    return eval_df


def _merge_eval_dataframes(eval_df_on: pd.DataFrame,
                           eval_df_off: pd.DataFrame) -> pd.DataFrame:
    """Merge gating-ON and gating-OFF results into one side-by-side table."""
    if len(eval_df_on) != len(eval_df_off):
        raise ValueError('Cannot merge ON/OFF eval dataframes with different lengths')

    base_cols = []
    preferred_cols = [
        'true', 'class_name', 'eye_occ', 'mouth_occ',
        'subject', 'video_key', 'frame_id',
    ]
    for col in preferred_cols:
        if col in eval_df_on.columns and col not in base_cols:
            base_cols.append(col)

    dynamic_cols = {
        'pred', 'is_correct', 'conf',
        'gate_face', 'gate_eye', 'gate_mouth',
    }
    for col in eval_df_on.columns:
        if col not in dynamic_cols and col not in base_cols:
            base_cols.append(col)

    for col in base_cols:
        if col in eval_df_off.columns and not eval_df_on[col].equals(eval_df_off[col]):
            raise ValueError(f'Cannot merge ON/OFF eval dataframes: column mismatch for {col}')

    merged = eval_df_on[base_cols].reset_index(drop=True).copy()
    merged['pred_gating_on'] = eval_df_on['pred'].reset_index(drop=True)
    merged['pred_gating_off'] = eval_df_off['pred'].reset_index(drop=True)
    merged['correct_gating_on'] = eval_df_on['is_correct'].reset_index(drop=True)
    merged['correct_gating_off'] = eval_df_off['is_correct'].reset_index(drop=True)
    if 'conf' in eval_df_on.columns:
        merged['conf_gating_on'] = eval_df_on['conf'].reset_index(drop=True)
    if 'conf' in eval_df_off.columns:
        merged['conf_gating_off'] = eval_df_off['conf'].reset_index(drop=True)
    for gate_col in ['gate_face', 'gate_eye', 'gate_mouth']:
        if gate_col in eval_df_on.columns:
            merged[gate_col] = eval_df_on[gate_col].reset_index(drop=True)
    return merged


def _build_analysis_summary(eval_df: pd.DataFrame,
                            pred_col: str = 'pred',
                            correct_col: str = 'is_correct') -> Dict:
    """Build the compact summary JSON saved beside the prediction CSV."""
    if len(eval_df) == 0:
        return {'n_eval': 0, 'eval_set': 'test', 'overall_accuracy': 0.0}

    preds = eval_df[pred_col].to_numpy()
    labels = eval_df['true'].to_numpy()
    metrics = compute_classification_metrics(labels, preds, label_names=LABEL_NAMES)
    summary = {
        'n_eval': int(len(eval_df)),
        'eval_set': 'test',
        'overall_accuracy': float(eval_df[correct_col].mean() * 100.0),
        'balanced_accuracy': metrics.get('balanced_accuracy', float('nan')),
        'macro_f1': metrics.get('macro_f1', float('nan')),
        'weighted_f1': metrics.get('f1', float('nan')),
    }

    for cls in LABEL_NAMES:
        sub = eval_df[eval_df['class_name'] == cls]
        if len(sub) > 0:
            summary[f'acc_{cls}'] = float(sub[correct_col].mean() * 100.0)
            summary[f'n_{cls}'] = int(len(sub))
    return summary


def _build_dual_analysis_summary(eval_df_on: pd.DataFrame,
                                 eval_df_off: pd.DataFrame) -> Dict:
    """Build a combined summary for side-by-side gating ON/OFF evaluation."""
    on_summary = _build_analysis_summary(eval_df_on)
    off_summary = _build_analysis_summary(eval_df_off)
    paired = compute_paired_binary_statistics(
        eval_df_on['is_correct'].astype(int).to_numpy(),
        eval_df_off['is_correct'].astype(int).to_numpy(),
    )

    summary = {
        'n_eval': int(len(eval_df_on)),
        'eval_set': 'test',
        'gating_mode': 'both',
        'overall_accuracy_gating_on': on_summary.get('overall_accuracy', 0.0),
        'overall_accuracy_gating_off': off_summary.get('overall_accuracy', 0.0),
        'balanced_accuracy_gating_on': on_summary.get('balanced_accuracy', float('nan')),
        'balanced_accuracy_gating_off': off_summary.get('balanced_accuracy', float('nan')),
        'macro_f1_gating_on': on_summary.get('macro_f1', float('nan')),
        'macro_f1_gating_off': off_summary.get('macro_f1', float('nan')),
        'clean_delta_pp': paired.get('delta_pp', float('nan')),
        'clean_delta_ci_low': paired.get('delta_ci_low', float('nan')),
        'clean_delta_ci_high': paired.get('delta_ci_high', float('nan')),
        'clean_delta_p_value': paired.get('p_value_mcnemar', float('nan')),
    }

    for cls in LABEL_NAMES:
        if f'acc_{cls}' in on_summary:
            summary[f'acc_{cls}_gating_on'] = on_summary[f'acc_{cls}']
        if f'acc_{cls}' in off_summary:
            summary[f'acc_{cls}_gating_off'] = off_summary[f'acc_{cls}']
        if f'n_{cls}' in on_summary:
            summary[f'n_{cls}'] = on_summary[f'n_{cls}']
    return summary


def _build_eval_metrics(eval_df: pd.DataFrame,
                        pred_col: str = 'pred',
                        source: str = 'full_clean_test_loader') -> Dict:
    """Compute classification metrics from the clean test predictions."""
    if len(eval_df) == 0:
        return {
            'accuracy': 0.0,
            'balanced_accuracy': None,
            'loss': None,
            'precision': None,
            'recall': None,
            'f1': None,
            'macro_precision': None,
            'macro_recall': None,
            'macro_f1': None,
            'per_class': {},
            'confusion_matrix': [],
            'uncertainty': {},
            'n_samples': 0,
            'source': source,
        }

    labels = eval_df['true'].to_numpy()
    preds = eval_df[pred_col].to_numpy()
    metrics = compute_classification_metrics(labels, preds, label_names=LABEL_NAMES)
    uncertainty = compute_classification_uncertainty(labels, preds)

    return {
        'accuracy': metrics.get('accuracy', 0.0),
        'balanced_accuracy': metrics.get('balanced_accuracy', float('nan')),
        'loss': None,
        'precision': metrics.get('precision', float('nan')),
        'recall': metrics.get('recall', float('nan')),
        'f1': metrics.get('f1', float('nan')),
        'macro_precision': metrics.get('macro_precision', float('nan')),
        'macro_recall': metrics.get('macro_recall', float('nan')),
        'macro_f1': metrics.get('macro_f1', float('nan')),
        'per_class': metrics.get('per_class', {}),
        'confusion_matrix': metrics.get('confusion_matrix', []),
        'uncertainty': uncertainty,
        'n_samples': int(len(eval_df)),
        'source': source,
    }


def _build_dual_eval_metrics(eval_df_on: pd.DataFrame,
                             eval_df_off: pd.DataFrame) -> Dict:
    """Compute combined metrics for gating ON/OFF clean-test evaluation."""
    on_metrics = _build_eval_metrics(
        eval_df_on,
        source='full_clean_test_loader_gating_on',
    )
    off_metrics = _build_eval_metrics(
        eval_df_off,
        source='full_clean_test_loader_gating_off',
    )
    on_metrics['gating_mode'] = 'on'
    off_metrics['gating_mode'] = 'off'

    paired = compute_paired_binary_statistics(
        eval_df_on['is_correct'].astype(int).to_numpy(),
        eval_df_off['is_correct'].astype(int).to_numpy(),
    )

    result = dict(on_metrics)
    result.update({
        'gating_mode': 'both',
        'source': 'full_clean_test_loader_gating_both',
        'gating_off_accuracy': off_metrics.get('accuracy', float('nan')),
        'gating_off_balanced_accuracy': off_metrics.get('balanced_accuracy', float('nan')),
        'gating_off_macro_f1': off_metrics.get('macro_f1', float('nan')),
        'clean_delta_pp': paired.get('delta_pp', float('nan')),
        'clean_delta_ci_low': paired.get('delta_ci_low', float('nan')),
        'clean_delta_ci_high': paired.get('delta_ci_high', float('nan')),
        'clean_delta_p_value': paired.get('p_value_mcnemar', float('nan')),
        'paired': paired,
        'gating_on': on_metrics,
        'gating_off': off_metrics,
    })
    return result


def _save_analysis_outputs(eval_df: pd.DataFrame,
                           out_dir: str,
                           gating_mode: str = 'on') -> Dict:
    """Save prediction CSV and the same clean-eval plots used by the main run."""
    os.makedirs(out_dir, exist_ok=True)

    if len(eval_df) == 0:
        summary = {
            'n_eval': 0,
            'eval_set': 'test',
            'overall_accuracy': 0.0,
            'gating_mode': gating_mode,
        }
        with open(os.path.join(out_dir, 'analysis_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        return summary

    preds = eval_df['pred'].to_numpy()
    labels = eval_df['true'].to_numpy()

    _save_confusion_matrix(preds, labels, LABEL_NAMES, out_dir)
    _save_per_class_metrics(preds, labels, LABEL_NAMES, out_dir)
    _save_gate_distributions(eval_df, out_dir)
    _save_gate_response_curves(eval_df, out_dir)
    _save_attention_heatmap(eval_df, out_dir)
    _save_per_subject_accuracy(eval_df, out_dir)
    eval_df.to_csv(os.path.join(out_dir, 'test_predictions.csv'), index=False)

    summary = _build_analysis_summary(eval_df)
    summary['gating_mode'] = gating_mode
    with open(os.path.join(out_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def _maybe_load_original_eval(run_dir: str) -> Dict:
    """Load the original run's eval_metrics.json when available."""
    path = os.path.join(run_dir, 'eval_metrics.json')
    if not os.path.exists(path):
        return {}
    try:
        return _load_json(path)
    except Exception:
        return {}


def _write_clean_test_metadata(out_dir: str,
                               spec: Dict,
                               split_info: Dict,
                               timing: Dict,
                               eval_metrics: Dict,
                               gating_mode: str):
    """Write evaluator metadata/config files into the output directory."""
    os.makedirs(out_dir, exist_ok=True)

    cfg = {
        'evaluated_at': datetime.now().isoformat(),
        'checkpoint_path': spec['checkpoint_path'],
        'run_dir': spec['run_dir'],
        'data_path': spec['data_path'],
        'mode': spec['mode'],
        'fold': spec['fold'],
        'k': spec['k'],
        'num_test': spec['num_test'],
        'strategy': spec['strategy'],
        'clip_length': spec['clip_length'],
        'batch_size': spec['batch_size'],
        'model': spec['model_type'],
        'face_detector': spec['face_detector'],
        'det_size': spec['det_size'],
        'gate_supervision': spec['gate_supervision'],
        'gating_mode': gating_mode,
        'color_handling': (
            'Frames are read and kept in OpenCV BGR order for face detection, '
            'feature extraction, and face-crop handling. RGB conversion is used '
            'only where the existing occlusion estimator / synthetic-occlusion '
            'pipeline already expects it.'
        ),
        'n_test_samples': split_info.get('n_test_samples', 0),
        'n_test_clips': split_info.get('n_test_clips'),
        'metrics_source': eval_metrics.get('source'),
    }
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    split_meta = {
        'mode': split_info.get('mode'),
        'fold': split_info.get('fold'),
        'k': split_info.get('k'),
        'seed': split_info.get('seed'),
        'train_subjects': split_info.get('train_subjects', []),
        'test_subjects': split_info.get('test_subjects', []),
        'train_videos': split_info.get('train_videos', []),
        'test_videos': split_info.get('test_videos', []),
        'n_test_samples': split_info.get('n_test_samples', 0),
        'n_test_clips': split_info.get('n_test_clips'),
    }
    with open(os.path.join(out_dir, 'split_info.json'), 'w') as f:
        json.dump(split_meta, f, indent=2)

    with open(os.path.join(out_dir, 'timing.json'), 'w') as f:
        json.dump({k: round(float(v), 3) for k, v in timing.items()}, f, indent=2)


def main():
    ap = argparse.ArgumentParser(
        description='Evaluate saved checkpoints on the full clean test split')
    ap.add_argument(
        '--model-paths',
        nargs='+',
        required=True,
        help='Checkpoint paths or run directories. Quote paths that contain spaces.',
    )
    ap.add_argument(
        '--output-subdir',
        default='clean-test-results',
        help='Per-run output folder name created inside each run directory.',
    )
    ap.add_argument(
        '--batch',
        type=int,
        default=None,
        help='Optional batch-size override for evaluation.',
    )
    ap.add_argument(
        '--max-test-clips',
        type=int,
        default=None,
        help='Optional clip cap for debugging. Leave unset for the full clean test.',
    )
    ap.add_argument(
        '--gating-mode',
        choices=['on', 'off', 'both'],
        default='on',
        help=(
            'Evaluate with normal gating, with gating disabled, or run both. '
            'The "both" mode writes side-by-side predictions in the main output '
            'folder and per-mode plots under gating_on/ and gating_off/.'
        ),
    )
    args = ap.parse_args()

    specs = [_load_run_spec(path) for path in args.model_paths]
    os.chdir(PROJECT_ROOT)
    _set_global_seed(42)

    _require_cuda('run_full_clean_test_eval.py')
    device = torch.device('cuda')
    print(f'Device: {device}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

    csv_data_cache: Dict[str, Dict] = {}
    sample_cache: Dict[Tuple, Tuple[List[Dict], Dict]] = {}
    summary_rows = []

    grouped_specs = defaultdict(list)
    for spec in specs:
        extraction_key = (
            spec['data_path'],
            spec['face_detector'],
            spec['det_size'],
            spec['face_cpu'],
        )
        grouped_specs[extraction_key].append(spec)

    overall_start = time.time()

    for extraction_key, group_specs in grouped_specs.items():
        data_path, face_type, det_size, face_cpu = extraction_key
        print('\n' + '=' * 80)
        print(f'Extraction group: data={data_path} | face={face_type} | det_size={det_size} | face_cpu={face_cpu}')
        print('=' * 80)

        if data_path not in csv_data_cache:
            csv_data_cache[data_path] = load_csv_video_data(data_path, filter_eye_states=True)
        csv_data = csv_data_cache[data_path]

        if not csv_data:
            raise RuntimeError(f'No data found for {data_path}')

        if face_cpu:
            os.environ['DOFG_FACE_CPU'] = '1'
        else:
            os.environ.pop('DOFG_FACE_CPU', None)

        extractor_args = SimpleNamespace(face=face_type, det_size=det_size, face_cpu=face_cpu)
        feat_extractor, occ_model, face_detector = _load_extraction_models(device, extractor_args)

        try:
            for spec in group_specs:
                run_start = time.time()
                out_dir = os.path.join(spec['run_dir'], args.output_subdir)
                os.makedirs(out_dir, exist_ok=True)
                print(f'\n--- Evaluating {spec["checkpoint_path"]} ---')

                sample_cache_key = (
                    extraction_key,
                    spec['mode'],
                    spec['fold'],
                    spec['k'],
                    spec['num_test'],
                    spec['seed'],
                    spec['strategy'],
                    spec['clip_length'],
                    spec['needs_face_crop'],
                    args.max_test_clips,
                )

                if sample_cache_key not in sample_cache:
                    print('  Building full clean test samples...')
                    extract_start = time.time()
                    test_samples, split_info = _extract_full_clean_test_samples(
                        csv_data=csv_data,
                        spec=spec,
                        face_detector=face_detector,
                        feat_extractor=feat_extractor,
                        occ_model=occ_model,
                        max_test_clips=args.max_test_clips,
                    )
                    split_info['extraction_seconds'] = time.time() - extract_start
                    sample_cache[sample_cache_key] = (test_samples, split_info)
                else:
                    print('  Reusing cached full clean test samples.')

                test_samples, split_info = sample_cache[sample_cache_key]
                batch_size = args.batch or spec['batch_size']
                print(f'  Test samples: {len(test_samples)}  {Counter(s["class_name"] for s in test_samples)}')
                print(f'  Batch size: {batch_size}')

                model = _build_model(spec, device)
                try:
                    _, test_loader = _make_test_loader(
                        test_samples=test_samples,
                        batch_size=batch_size,
                        device=device,
                        gate_supervision=spec['gate_supervision'],
                        seed=spec['seed'],
                    )

                    infer_start = time.time()
                    original_eval = _maybe_load_original_eval(spec['run_dir'])

                    if args.gating_mode == 'both':
                        eval_df_on = _build_eval_dataframe(
                            model, test_loader, test_samples, gating_mode='on')
                        eval_df_off = _build_eval_dataframe(
                            model, test_loader, test_samples, gating_mode='off')
                        infer_seconds = time.time() - infer_start

                        eval_metrics = _build_dual_eval_metrics(eval_df_on, eval_df_off)
                        if original_eval:
                            eval_metrics['original_reported_accuracy'] = original_eval.get('accuracy')
                            eval_metrics['original_reported_n_samples'] = original_eval.get('n_samples')
                            eval_metrics['original_reported_source'] = original_eval.get('source')
                            if original_eval.get('accuracy') is not None:
                                eval_metrics['delta_vs_original_pp'] = (
                                    eval_metrics['accuracy'] - float(original_eval['accuracy'])
                                )

                        gating_on_dir = os.path.join(out_dir, 'gating_on')
                        gating_off_dir = os.path.join(out_dir, 'gating_off')
                        _save_analysis_outputs(eval_df_on, gating_on_dir, gating_mode='on')
                        _save_analysis_outputs(eval_df_off, gating_off_dir, gating_mode='off')
                        with open(os.path.join(gating_on_dir, 'eval_metrics.json'), 'w') as f:
                            json.dump(eval_metrics['gating_on'], f, indent=2)
                        with open(os.path.join(gating_off_dir, 'eval_metrics.json'), 'w') as f:
                            json.dump(eval_metrics['gating_off'], f, indent=2)

                        merged_eval_df = _merge_eval_dataframes(eval_df_on, eval_df_off)
                        merged_eval_df.to_csv(os.path.join(out_dir, 'test_predictions.csv'), index=False)
                        analysis_summary = _build_dual_analysis_summary(eval_df_on, eval_df_off)
                        with open(os.path.join(out_dir, 'analysis_summary.json'), 'w') as f:
                            json.dump(analysis_summary, f, indent=2)
                    else:
                        eval_df = _build_eval_dataframe(
                            model, test_loader, test_samples, gating_mode=args.gating_mode)
                        infer_seconds = time.time() - infer_start

                        eval_metrics = _build_eval_metrics(
                            eval_df,
                            source=f'full_clean_test_loader_gating_{args.gating_mode}',
                        )
                        eval_metrics['gating_mode'] = args.gating_mode
                        if args.gating_mode == 'on' and original_eval:
                            eval_metrics['original_reported_accuracy'] = original_eval.get('accuracy')
                            eval_metrics['original_reported_n_samples'] = original_eval.get('n_samples')
                            eval_metrics['original_reported_source'] = original_eval.get('source')
                            if original_eval.get('accuracy') is not None:
                                eval_metrics['delta_vs_original_pp'] = (
                                    eval_metrics['accuracy'] - float(original_eval['accuracy'])
                                )
                        elif args.gating_mode == 'off' and original_eval:
                            eval_metrics['reference_original_gating_on_accuracy'] = original_eval.get('accuracy')
                            eval_metrics['reference_original_gating_on_n_samples'] = original_eval.get('n_samples')
                            eval_metrics['reference_original_gating_on_source'] = original_eval.get('source')
                            if original_eval.get('accuracy') is not None:
                                eval_metrics['delta_vs_original_gating_on_pp'] = (
                                    eval_metrics['accuracy'] - float(original_eval['accuracy'])
                                )

                        analysis_summary = _save_analysis_outputs(
                            eval_df, out_dir, gating_mode=args.gating_mode)

                    with open(os.path.join(out_dir, 'eval_metrics.json'), 'w') as f:
                        json.dump(eval_metrics, f, indent=2)

                    timing = {
                        'cached_extraction_seconds': split_info.get('extraction_seconds', 0.0),
                        'inference_seconds': infer_seconds,
                        'total_model_seconds': time.time() - run_start,
                    }
                    _write_clean_test_metadata(
                        out_dir=out_dir,
                        spec=spec,
                        split_info=split_info,
                        timing=timing,
                        eval_metrics=eval_metrics,
                        gating_mode=args.gating_mode,
                    )

                    accuracy_gating_on = (
                        eval_metrics.get('accuracy')
                        if args.gating_mode == 'on'
                        else eval_metrics.get('gating_on', {}).get('accuracy')
                    )
                    accuracy_gating_off = (
                        eval_metrics.get('accuracy')
                        if args.gating_mode == 'off'
                        else eval_metrics.get('gating_off_accuracy')
                    )
                    summary_rows.append({
                        'run_dir': spec['run_dir'],
                        'checkpoint_path': spec['checkpoint_path'],
                        'mode': spec['mode'],
                        'fold': spec['fold'],
                        'model': spec['model_type'],
                        'gating_mode': args.gating_mode,
                        'n_test_samples': eval_metrics.get('n_samples', 0),
                        'accuracy': eval_metrics.get('accuracy'),
                        'accuracy_gating_on': accuracy_gating_on,
                        'accuracy_gating_off': accuracy_gating_off,
                        'clean_delta_pp': eval_metrics.get('clean_delta_pp'),
                        'macro_f1': eval_metrics.get('macro_f1'),
                        'balanced_accuracy': eval_metrics.get('balanced_accuracy'),
                        'original_reported_accuracy': eval_metrics.get('original_reported_accuracy'),
                        'delta_vs_original_pp': eval_metrics.get('delta_vs_original_pp'),
                    })

                    if args.gating_mode == 'both':
                        print(f'  Full clean test accuracy (gating ON): {eval_metrics.get("accuracy", 0.0):.2f}%')
                        print(f'  Full clean test accuracy (gating OFF): {eval_metrics.get("gating_off_accuracy", 0.0):.2f}%')
                        print(f'  Gating benefit (ON - OFF): {eval_metrics.get("clean_delta_pp", 0.0):.2f} pp')
                        if eval_metrics.get('original_reported_accuracy') is not None:
                            print(f'  Original reported accuracy: {eval_metrics["original_reported_accuracy"]:.2f}%')
                            print(f'  Delta vs original: {eval_metrics.get("delta_vs_original_pp", 0.0):.2f} pp')
                        print(f'  Saved combined clean test outputs to {out_dir}')
                        print(f'  Saved gating-ON analysis to {os.path.join(out_dir, "gating_on")}')
                        print(f'  Saved gating-OFF analysis to {os.path.join(out_dir, "gating_off")}')
                    else:
                        print(f'  Full clean test accuracy (gating {args.gating_mode.upper()}): {eval_metrics.get("accuracy", 0.0):.2f}%')
                        if args.gating_mode == 'on' and eval_metrics.get('original_reported_accuracy') is not None:
                            print(f'  Original reported accuracy: {eval_metrics["original_reported_accuracy"]:.2f}%')
                            print(f'  Delta vs original: {eval_metrics.get("delta_vs_original_pp", 0.0):.2f} pp')
                        if args.gating_mode == 'off' and eval_metrics.get('reference_original_gating_on_accuracy') is not None:
                            print(f'  Reference original gating-ON accuracy: {eval_metrics["reference_original_gating_on_accuracy"]:.2f}%')
                            print(f'  Delta vs original gating-ON: {eval_metrics.get("delta_vs_original_gating_on_pp", 0.0):.2f} pp')
                        print(f'  Saved clean test outputs to {out_dir}')
                    if analysis_summary.get('n_eval', 0):
                        if args.gating_mode == 'both':
                            print(
                                f'  Saved {analysis_summary["n_eval"]} side-by-side test predictions '
                                'plus per-mode plots'
                            )
                        else:
                            print(f'  Saved {analysis_summary["n_eval"]} test predictions and plots')
                finally:
                    del model
                    _gpu_cleanup('per-model evaluation complete')
        finally:
            _delete_models(feat_extractor, occ_model, face_detector)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.abspath('clean_test_eval_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print('\nSummary:')
        print(summary_df.to_string(index=False))
        print(f'\nSaved aggregate summary to {summary_path}')

    total_seconds = time.time() - overall_start
    print(f'\nDone in {total_seconds / 60.0:.1f} minutes.')


if __name__ == '__main__':
    main()
