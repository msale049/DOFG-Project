#!/usr/bin/env python3
"""
run_gate_floor_sweep.py
=======================
Optimized gate_floor parametric sweep: extracts features **once** per fold,
caches stress-test frame features, then trains + evaluates a separate
transformer for each gate_floor value.

Time savings (per fold)
-----------------------
                               Naive (×N)    Optimized
  Feature extraction  (~50m)    ×N            ×1
  Training            (~11m)    ×N            ×N
  Stress feat cache   (~60m)    ×N            ×1
  Stress inference    (~1m)     —             ×N
  ─────────────────────────────────────────────────
  Total (7 floors)             ~17.5h         ~2.4h

Usage
-----
    # Fixed split (fold 0), all 7 gate_floor values
    python run_gate_floor_sweep.py

    # K-fold (5 folds × 7 floors)
    python run_gate_floor_sweep.py --mode kfold --k 5

    # Subset of floors, skip completed
    python run_gate_floor_sweep.py --floors 0.50,0.70,0.90,1.00 --skip-existing
"""

import argparse
import gc
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('ORT_LOG_LEVEL', '3')
os.environ.setdefault('ONNXRUNTIME_SESSION_THREAD_POOL_SIZE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import CONFIG, SEED, CLIP_CONFIG
from data_loading import load_csv_video_data
from split_generator import create_splits, get_subject_ids
from pipeline import extract_features_for_clips
from datasets import DriverStateDataset
from transformer_enhanced import EnhancedOcclusionAwareTransformer
from trainer_enhanced import TinyTransformerTrainer
from evaluation import compute_metrics_on_loader
from metrics_utils import (
    compute_classification_metrics,
    compute_classification_uncertainty,
    compute_paired_binary_statistics,
)
from stress_test import (
    STRESS_REGIMES,
    _build_stress_conditions,
    _apply_stress_condition,
    run_with_gating_disabled,
)
from ablation_utils import disable_gates_at_inference

# ─── GPU helpers ─────────────────────────────────────────────────────────────

def _gpu_cleanup(msg: str = ''):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if msg:
        try:
            free, total = torch.cuda.mem_get_info()
            alloc = torch.cuda.memory_allocated()
            print(f'  [GPU] {msg} — free={free/1e9:.2f}GB alloc={alloc/1e6:.0f}MB')
        except Exception:
            print(f'  [GPU] {msg}')


def _set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _parse_fold_list(spec: Optional[str], max_folds: int) -> List[int]:
    """Parse a comma-separated list of fold ids."""
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


def _save_fold_level_analysis(
    model: torch.nn.Module,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader],
    val_samples: List[Dict],
    test_samples: List[Dict],
    history: Dict,
    stress_summary: pd.DataFrame,
    stress_details: pd.DataFrame,
    run_dir: str,
    device: torch.device,
) -> None:
    """Generate per-run plots and analysis artifacts without affecting training success."""
    try:
        from run_train_eval import (
            _plot_gating_comparison,
            _plot_gates_vs_opacity,
            _plot_opacity_analysis,
            _save_comprehensive_analysis,
            _save_per_class_stress_deltas,
            _save_per_class_stress_tables,
            _save_training_curves,
        )

        _save_training_curves(history, run_dir)
        _save_comprehensive_analysis(
            model=model,
            val_loader=val_loader,
            test_loader=test_loader,
            val_samples=val_samples,
            test_samples=test_samples,
            history=history,
            stress_summary=stress_summary,
            stress_details=stress_details,
            run_dir=run_dir,
            device=device,
        )

        if len(stress_details) > 0 and len(stress_summary) > 0:
            _plot_gating_comparison(stress_summary, run_dir)
            _plot_opacity_analysis(stress_summary, run_dir)
            _plot_gates_vs_opacity(stress_details, run_dir)
            _save_per_class_stress_deltas(stress_details, run_dir)
            _save_per_class_stress_tables(
                os.path.join(run_dir, 'stress_test_details.csv'),
                run_dir,
            )
    except Exception as e:
        print(f'WARNING: fold-level analysis plotting failed for {run_dir}: {e}')


# ─── Extraction model loading ───────────────────────────────────────────────

def _load_extraction_models(device, face_type='retina', det_size=640):
    from feature_extraction import ResNet34FeatureExtractor
    from occlusion_estimator import ResNet34OcclusionModel

    print('Loading feature extraction models...')
    feat_extractor = ResNet34FeatureExtractor(CONFIG['RESNET34_MODEL_PATH'], device=str(device))
    occ_model = ResNet34OcclusionModel(CONFIG['RESNET34_OCCLUSION_MODEL_PATH'], device=str(device))

    det_sz = (det_size, det_size) if face_type == 'retina' else None
    try:
        if face_type == 'retina':
            from face_detection_retinaface import FaceDetector
            face_detector = FaceDetector(shape_model_path=CONFIG['DLIB_MODEL_PATH'],
                                         det_size=det_sz, det_thresh=0.35)
        else:
            from face_detection_dlib import FaceDetector
            face_detector = FaceDetector(shape_model_path=CONFIG['DLIB_MODEL_PATH'])
    except ImportError:
        from face_detection_dlib import FaceDetector
        face_detector = FaceDetector(shape_model_path=CONFIG['DLIB_MODEL_PATH'])

    _gpu_cleanup('extraction models loaded')
    return feat_extractor, occ_model, face_detector


# ─── Stress-test feature cache ──────────────────────────────────────────────

@dataclass
class CachedStressFrame:
    """Pre-computed features for one (frame × condition) pair."""
    subject: str
    video_key: str
    clip_start: int
    frame_num: int
    class_label: str
    gt_label: int
    condition: str
    regime: str
    opacity: float
    face_feat: np.ndarray          # (512,)
    left_eye_feat: np.ndarray      # (512,)
    right_eye_feat: np.ndarray     # (512,)
    mouth_feat: np.ndarray         # (512,)
    p_eye: float
    p_mouth: float


def precompute_stress_features(
    csv_data: Dict,
    test_clips: List,
    face_detector,
    feat_extractor,
    occ_model,
    stress_opacities: List[float],
    max_frames_per_clip: int = 8,
    seed: int = SEED,
    label_map: Optional[Dict[str, int]] = None,
) -> List[CachedStressFrame]:
    """
    Read test frames, apply each stress condition, extract features + occ probs.
    Returns a list of CachedStressFrame, one per (frame × condition).
    Model-independent -- can be replayed for any gate_floor transformer.
    """
    if label_map is None:
        label_map = {'EyeClosed': 0, 'Yawn': 1, 'Neutral': 2}
    if stress_opacities is None:
        stress_opacities = [0.4, 0.6, 0.8, 1.0]

    conditions = _build_stress_conditions(stress_opacities)
    cache: List[CachedStressFrame] = []
    processed, skipped = 0, 0
    t0 = time.time()

    for clip in test_clips:
        meta = csv_data.get(clip.video_key)
        if not meta:
            continue
        cap = cv2.VideoCapture(meta['video_path'])
        if not cap.isOpened():
            continue

        ann_by_frame = {a.frame: a for a in meta['annotations']}
        frame_indices = list(range(len(clip.frame_numbers)))
        if max_frames_per_clip and len(frame_indices) > max_frames_per_clip:
            step = len(frame_indices) / max_frames_per_clip
            frame_indices = [int(i * step) for i in range(max_frames_per_clip)]

        for fi in frame_indices:
            if fi >= len(clip.frame_numbers):
                continue
            frame_num = clip.frame_numbers[fi]
            ann = ann_by_frame.get(frame_num)
            if ann is None or ann.class_label not in label_map:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, bgr = cap.read()
            if not ret:
                continue

            det = face_detector.detect_face_and_landmarks(bgr)
            if not det['is_valid'] or det.get('landmarks') is None:
                skipped += 1
                continue

            lm = det['landmarks']
            fb = det['face_bbox']
            er = det['eye_regions']
            mr = det['mouth_region']
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gt = label_map[ann.class_label]

            for cond_name, regime, opacity in conditions:
                aug_rgb = _apply_stress_condition(rgb, lm, regime, opacity, fi, clip.T)
                aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)

                feat = feat_extractor.extract_region_features(aug_bgr, fb, er, mr)
                rkeys = ['face_features', 'left_eye_features',
                         'right_eye_features', 'mouth_features']
                if any(feat[k] is None for k in rkeys):
                    continue

                probs = occ_model.predict_probs(
                    aug_rgb, face_bbox=fb, image_bgr=False, face_margin=0.15)

                cache.append(CachedStressFrame(
                    subject=clip.subject,
                    video_key=clip.video_key,
                    clip_start=clip.clip_start,
                    frame_num=frame_num,
                    class_label=ann.class_label,
                    gt_label=gt,
                    condition=cond_name,
                    regime=regime,
                    opacity=opacity,
                    face_feat=np.asarray(feat['face_features'], dtype=np.float32),
                    left_eye_feat=np.asarray(feat['left_eye_features'], dtype=np.float32),
                    right_eye_feat=np.asarray(feat['right_eye_features'], dtype=np.float32),
                    mouth_feat=np.asarray(feat['mouth_features'], dtype=np.float32),
                    p_eye=float(probs[0]),
                    p_mouth=float(probs[1]),
                ))
                del aug_rgb, aug_bgr, feat

            processed += 1
            if processed % 100 == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                print(f'    Stress cache: [{processed}] {rate:.1f} fr/s, '
                      f'{len(cache)} entries', flush=True)

        cap.release()

    elapsed = time.time() - t0
    mem_mb = sum(c.face_feat.nbytes + c.left_eye_feat.nbytes +
                 c.right_eye_feat.nbytes + c.mouth_feat.nbytes
                 for c in cache) / 1e6
    print(f'  Stress feature cache: {len(cache)} entries from {processed} frames '
          f'({skipped} skipped), {elapsed:.0f}s, ~{mem_mb:.0f} MB')
    return cache


def run_stress_from_cache(
    model: torch.nn.Module,
    cache: List[CachedStressFrame],
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run transformer inference (gating ON + OFF) on pre-computed stress features.
    Returns (details_df, summary_df) matching run_stress_test output format.
    """
    model.eval()
    rows = []
    n = len(cache)
    t0 = time.time()

    for start in range(0, n, batch_size):
        batch = cache[start:start + batch_size]
        bs = len(batch)

        fdict = {
            'face': torch.zeros(bs, 512, device=device),
            'left_eye': torch.zeros(bs, 512, device=device),
            'right_eye': torch.zeros(bs, 512, device=device),
            'mouth': torch.zeros(bs, 512, device=device),
        }
        occ_info = {
            'eye_occlusion_prob': torch.zeros(bs, device=device),
            'mouth_occlusion_prob': torch.zeros(bs, device=device),
        }

        for i, c in enumerate(batch):
            fdict['face'][i] = torch.from_numpy(c.face_feat)
            fdict['left_eye'][i] = torch.from_numpy(c.left_eye_feat)
            fdict['right_eye'][i] = torch.from_numpy(c.right_eye_feat)
            fdict['mouth'][i] = torch.from_numpy(c.mouth_feat)
            occ_info['eye_occlusion_prob'][i] = c.p_eye
            occ_info['mouth_occlusion_prob'][i] = c.p_mouth

        with torch.no_grad():
            out_on = model(fdict, occ_info, return_attention=True)
            preds_on = out_on['predicted_class'].cpu().numpy()
            gates = out_on['gate_factors'].cpu().numpy()

        with disable_gates_at_inference(model):
            with torch.no_grad():
                out_off = model(fdict, occ_info, return_attention=False)
                preds_off = out_off['predicted_class'].cpu().numpy()

        for i, c in enumerate(batch):
            rows.append({
                'subject': c.subject,
                'video_key': c.video_key,
                'clip_start': c.clip_start,
                'frame': c.frame_num,
                'class_label': c.class_label,
                'gt_label': c.gt_label,
                'condition': c.condition,
                'regime': c.regime,
                'opacity': c.opacity,
                'p_eye': c.p_eye,
                'p_mouth': c.p_mouth,
                'pred_gating_on': int(preds_on[i]),
                'pred_gating_off': int(preds_off[i]),
                'correct_gating_on': int(preds_on[i] == c.gt_label),
                'correct_gating_off': int(preds_off[i] == c.gt_label),
                'gate_face': float(gates[i, 0]),
                'gate_eye': float(gates[i, 1]),
                'gate_mouth': float(gates[i, 3]),
            })

        del fdict, occ_info, out_on, out_off

    details = pd.DataFrame(rows)
    elapsed = time.time() - t0
    print(f'    Stress inference: {n} entries in {elapsed:.1f}s')

    if len(details) == 0:
        return details, pd.DataFrame()

    conditions = sorted(details['condition'].unique(),
                        key=lambda c: (c != 'clean', c))
    summary_rows = []
    for cond in conditions:
        sub = details[details['condition'] == cond]
        if len(sub) == 0:
            continue
        parts = cond.split('@')
        regime = parts[0] if len(parts) >= 1 else cond
        opacity = float(parts[1]) if len(parts) == 2 else 0.0
        if cond == 'clean':
            regime, opacity = 'clean', 0.0
        summary_rows.append({
            'condition': cond,
            'regime': regime,
            'opacity': opacity,
            'acc_gating_on': sub['correct_gating_on'].mean() * 100,
            'acc_gating_off': sub['correct_gating_off'].mean() * 100,
            'delta_pp': (sub['correct_gating_on'].mean() - sub['correct_gating_off'].mean()) * 100,
            'n': len(sub),
        })
    summary = pd.DataFrame(summary_rows)
    return details, summary


# ─── Paired stats helper ────────────────────────────────────────────────────

def _add_paired_stats(summary_df, details_df, seed):
    if summary_df is None or len(summary_df) == 0:
        return summary_df
    if details_df is None or len(details_df) == 0:
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
            sub['correct_gating_off'].to_numpy(), seed=seed)
        merged = row.to_dict()
        merged.update(stats)
        rows.append(merged)
    return pd.DataFrame(rows)


# ─── Train + evaluate one gate_floor value ───────────────────────────────────

def _train_and_eval_one_floor(
    gate_floor: float,
    train_samples: List[Dict],
    val_samples: List[Dict],
    test_samples: List[Dict],
    stress_cache: List[CachedStressFrame],
    device: torch.device,
    run_dir: str,
    args,
    fold: int,
    splits: Dict,
) -> Dict:
    """Train a transformer with the given gate_floor, evaluate, stress-test from cache."""
    os.makedirs(run_dir, exist_ok=True)
    print(f'\n{"="*60}')
    print(f'  gate_floor = {gate_floor:.2f}  |  fold = {fold}  |  {run_dir}')
    print(f'{"="*60}')

    _set_global_seed(args.seed)
    t0_total = time.time()
    stage_times = {}
    ckpt_path = os.path.join(run_dir, 'model_best.pt')

    # Save config
    config_dict = {
        'data': args.data, 'samples_per_video': 'all',
        'epochs': args.epochs, 'batch_size': args.batch, 'mode': args.mode,
        'k': args.k, 'num_test': args.num_test, 'fold': fold,
        'face_detector': args.face, 'det_size': args.det_size,
        'seed': args.seed, 'stress_test': True, 'stress_frames': args.stress_frames,
        'max_train_clips': args.max_train_clips,
        'max_val_clips': args.max_val_clips,
        'max_test_clips': args.max_test_clips,
        'train_opacity_sampler': {'mode': 'discrete', 'values': [0.4, 0.6, 0.8, 1.0], 'weights': None},
        'train_opacity_mode': 'discrete',
        'train_opacity_values': [0.4, 0.6, 0.8, 1.0],
        'strategy': 'clip', 'model': 'transformer',
        'class_weighted': True, 'gate_supervision': 'gt',
        'gate_weight': args.gate_weight, 'gate_floor': gate_floor,
        'asymmetric_floor': False, 'eye_floor': gate_floor, 'mouth_floor': gate_floor,
        'diversity_reg': 0.0, 'learning_rate': 3e-5, 'clip_length': 32,
        'gating_mode': args.gating_mode,
        'use_logit_bias': bool(args.use_logit_bias),
        'use_estimator_calibration': bool(args.use_estimator_calibration),
        'gate_dropout': float(args.gate_dropout),
        'validation_protocol': 'clean primary',
    }
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

    split_config = splits.get('split_config', {})
    with open(os.path.join(run_dir, 'split_info.json'), 'w') as f:
        json.dump({
            'mode': args.mode, 'fold': fold, 'k': args.k, 'seed': args.seed,
            'train_subjects': split_config.get('train_subjects', []),
            'test_subjects': split_config.get('test_subjects', []),
        }, f, indent=2)

    # ── Build data loaders ───────────────────────────────────────────────────
    train_ds = DriverStateDataset(train_samples, device=str(device), gate_supervision='gt')
    val_ds = DriverStateDataset(val_samples, device=str(device), gate_supervision='gt')

    train_gen = torch.Generator(); train_gen.manual_seed(args.seed)
    val_gen = torch.Generator(); val_gen.manual_seed(args.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=train_ds.collate_samples, drop_last=True,
                              worker_init_fn=_seed_worker, generator=train_gen)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            collate_fn=val_ds.collate_samples,
                            worker_init_fn=_seed_worker, generator=val_gen)

    test_loader = None
    if test_samples:
        test_ds = DriverStateDataset(test_samples, device=str(device), gate_supervision='gt')
        test_gen = torch.Generator(); test_gen.manual_seed(args.seed + 2)
        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                                 collate_fn=test_ds.collate_samples,
                                 worker_init_fn=_seed_worker, generator=test_gen)

    # ── Build model ──────────────────────────────────────────────────────────
    model = EnhancedOcclusionAwareTransformer(
        feature_dim=512, hidden_dim=128, num_heads=4,
        num_classes=3, num_layers=3, use_relative_pos=True,
        gate_floor=gate_floor, eye_floor=gate_floor, mouth_floor=gate_floor,
        gating_mode=args.gating_mode,
        use_logit_bias=args.use_logit_bias,
        use_estimator_calibration=args.use_estimator_calibration,
        gate_dropout=args.gate_dropout,
    ).to(device)
    print(f'  Model: gating_mode={args.gating_mode}, '
          f'logit_bias={args.use_logit_bias}, '
          f'estimator_calibration={args.use_estimator_calibration}, '
          f'gate_dropout={args.gate_dropout}')

    label_counts = Counter(s['label'] for s in train_samples)
    total_c = sum(label_counts.values())
    present = [i for i in range(3) if label_counts.get(i, 0) > 0]
    weights = [(total_c / (len(present) * label_counts[i])) if label_counts.get(i, 0) > 0 else 0.0
               for i in range(3)]
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    trainer = TinyTransformerTrainer(
        model, device=str(device), learning_rate=3e-5,
        class_weights=class_weights, gate_weight=args.gate_weight,
        gate_floor=gate_floor, eye_floor=gate_floor, mouth_floor=gate_floor,
        diversity_reg=0.0)

    # ── Training ─────────────────────────────────────────────────────────────
    t0 = time.time()
    best_val_acc = float('-inf')
    best_epoch = None
    for epoch in range(args.epochs):
        _set_global_seed(args.seed + epoch)
        train_metrics = trainer.train_epoch(train_loader, epoch=epoch)
        val_metrics = trainer.validate_epoch(val_loader, epoch=epoch)
        val_acc = val_metrics.get('val_accuracy', 0.0)
        if best_epoch is None or val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), ckpt_path)
        print(f'  Ep {epoch+1}/{args.epochs}: train_acc={train_metrics.get("accuracy",0):.1f}% '
              f'val_acc={val_acc:.1f}%')

    with open(os.path.join(run_dir, 'training_history.json'), 'w') as f:
        json.dump({k: (v if isinstance(v, list) else str(v))
                   for k, v in trainer.history.items()}, f, indent=2)

    del trainer.optimizer, trainer.scheduler
    _gpu_cleanup('training complete')
    stage_times['training'] = time.time() - t0

    # Load best checkpoint
    ckpt = torch.load(ckpt_path,
                      map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing or unexpected:
        print(f'  [ckpt] missing={len(missing)} unexpected={len(unexpected)} '
              f'(architecture likely changed)')
    del ckpt
    _gpu_cleanup('best ckpt loaded')

    # ── Clean test evaluation ────────────────────────────────────────────────
    t0 = time.time()
    eval_dict = {}
    if test_loader is not None:
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
    eval_dict['best_epoch'] = best_epoch
    stage_times['clean_eval'] = time.time() - t0

    # ── Stress test from cache (fast!) ───────────────────────────────────────
    t0 = time.time()
    stress_details = pd.DataFrame()
    stress_summary = pd.DataFrame()
    if stress_cache:
        stress_details, stress_summary = run_stress_from_cache(
            model, stress_cache, device, batch_size=256)
        stress_summary = _add_paired_stats(stress_summary, stress_details, seed=args.seed)
        if len(stress_details) > 0:
            stress_details.to_csv(os.path.join(run_dir, 'stress_test_details.csv'), index=False)
            stress_summary.to_csv(os.path.join(run_dir, 'stress_test_summary.csv'), index=False)

    stage_times['stress_test'] = time.time() - t0

    # If no direct test eval, extract from stress clean condition
    if eval_dict.get('source') == 'none' and len(stress_details) > 0:
        clean = stress_details[stress_details['condition'] == 'clean']
        if len(clean) > 0:
            labels = clean['gt_label'].to_numpy()
            preds = clean['pred_gating_on'].to_numpy()
            m = compute_classification_metrics(
                labels, preds, label_names=['EyeClosed', 'Yawn', 'Neutral'])
            eval_dict.update({
                'accuracy': m.get('accuracy', 0), 'macro_f1': m.get('macro_f1'),
                'per_class': m.get('per_class', {}), 'n_samples': len(clean),
                'source': 'stress_test_clean_condition',
            })
    eval_dict['val_clean_accuracy'] = float(best_val_acc) if best_epoch is not None else 0.0
    eval_dict['best_epoch'] = best_epoch

    with open(os.path.join(run_dir, 'eval_metrics.json'), 'w') as f:
        json.dump(eval_dict, f, indent=2)

    _save_fold_level_analysis(
        model=model,
        val_loader=val_loader,
        test_loader=test_loader,
        val_samples=val_samples,
        test_samples=test_samples,
        history=trainer.history,
        stress_summary=stress_summary,
        stress_details=stress_details,
        run_dir=run_dir,
        device=device,
    )

    stage_times['total'] = time.time() - t0_total
    with open(os.path.join(run_dir, 'timing.json'), 'w') as f:
        json.dump({k: round(v, 2) for k, v in stage_times.items()}, f, indent=2)

    print(f'  floor={gate_floor:.2f}: acc={eval_dict.get("accuracy",0):.1f}%, '
          f'macro_f1={eval_dict.get("macro_f1","N/A")}, '
          f'train={stage_times["training"]:.0f}s, stress={stage_times["stress_test"]:.0f}s, '
          f'total={stage_times["total"]:.0f}s')

    del model, trainer
    _gpu_cleanup(f'floor={gate_floor:.2f} freed')
    return eval_dict


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Optimized gate_floor parametric sweep')
    ap.add_argument('--data', default='Data')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--sweep-name', default=None,
                    help='Override sweep directory name')
    ap.add_argument('--mode', choices=['fixed', 'kfold'], default='fixed')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--folds', type=str, default=None,
                    help='Optional comma-separated fold ids for k-fold mode, e.g. 0,1,2')
    ap.add_argument('--num-test', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--face', choices=['dlib', 'retina'], default='retina')
    ap.add_argument('--det-size', type=int, default=640)
    ap.add_argument('--stress-frames', type=int, default=20)
    ap.add_argument('--max-train-clips', type=int, default=None,
                    help='Optional cap on training clips for quick smoke tests')
    ap.add_argument('--max-val-clips', type=int, default=None,
                    help='Optional cap on validation clips for quick smoke tests')
    ap.add_argument('--max-test-clips', type=int, default=None,
                    help='Optional cap on clean/stress test clips for quick smoke tests')
    ap.add_argument('--gate-weight', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--floors', type=str, default='0.05,0.10,0.30,0.50,0.70,0.90,1.00',
                    help='Comma-separated gate_floor values')
    ap.add_argument('--skip-existing', action='store_true',
                    help='Skip floors that already have eval_metrics.json')
    ap.add_argument('--aggregate', action='store_true', default=True)
    ap.add_argument('--no-aggregate', action='store_false', dest='aggregate')
    ap.add_argument('--stress-opacities', type=str, default='0.4,0.6,0.8,1.0',
                    help='Comma-separated stress test opacity levels')
    ap.add_argument('--gating-mode', choices=['attention', 'legacy', 'both', 'none'],
                    default='attention',
                    help='Gating mechanism in the transformer (default: attention = '
                         'new log-gate attention bias; survives LayerNorm).')
    ap.add_argument('--no-logit-bias', action='store_false', dest='use_logit_bias',
                    help='Disable the gate-conditioned per-class logit bias head.')
    ap.add_argument('--no-estimator-calibration', action='store_false',
                    dest='use_estimator_calibration',
                    help='Disable the joint eye+mouth estimator calibration head.')
    ap.add_argument('--gate-dropout', type=float, default=0.1,
                    help='Probability per sample/region of forcing gate=1.0 during '
                         'training (default 0.1). Set to 0 to disable.')
    ap.set_defaults(use_logit_bias=True, use_estimator_calibration=True)
    args = ap.parse_args()

    gate_floors = sorted(set(float(x.strip()) for x in args.floors.split(',')))
    stress_opacities = [float(x.strip()) for x in args.stress_opacities.split(',')]
    is_kfold = args.mode == 'kfold'
    sweep_name = args.sweep_name or ('gate_floor_sweep_kfold' if is_kfold else 'gate_floor_sweep')
    sweep_dir = os.path.join(args.results_root, sweep_name)
    os.makedirs(sweep_dir, exist_ok=True)

    if not torch.cuda.is_available():
        print('CUDA required.')
        sys.exit(1)

    device = torch.device('cuda')
    print(f'\n{"="*70}')
    print(f'  Gate Floor Sweep — Optimized')
    print(f'  Device : {torch.cuda.get_device_name(0)}')
    print(f'  Mode   : {args.mode}')
    print(f'  Floors : {gate_floors}')
    print(f'  Epochs : {args.epochs}')
    if any(v is not None for v in (args.max_train_clips, args.max_val_clips, args.max_test_clips)):
        print('  Clip caps : '
              f'train={args.max_train_clips or "all"}, '
              f'val={args.max_val_clips or "all"}, '
              f'test={args.max_test_clips or "all"}')
    print(f'  Sweep  : {sweep_dir}')
    print(f'{"="*70}\n')

    # ── Load CSV data (once) ─────────────────────────────────────────────────
    print('=== Loading CSV data ===')
    csv_data = load_csv_video_data(args.data, filter_eye_states=True)
    if not csv_data:
        print('No data. Check --data.')
        sys.exit(1)
    print(f'  Loaded {len(csv_data)} videos')

    subjects = get_subject_ids(csv_data)
    if is_kfold:
        all_splits = create_splits(subjects, mode='kfold', k=args.k, seed=args.seed)
        fold_ids = _parse_fold_list(args.folds, len(all_splits))
        print(f'  Folds  : {fold_ids}')
    else:
        all_splits = create_splits(subjects, mode='fixed', num_test=args.num_test, seed=args.seed)
        fold_ids = [0]

    sweep_manifest = {
        'data': args.data,
        'results_root': args.results_root,
        'sweep_name': sweep_name,
        'mode': args.mode,
        'k': args.k,
        'folds': fold_ids,
        'num_test': args.num_test,
        'epochs': args.epochs,
        'batch': args.batch,
        'face': args.face,
        'det_size': args.det_size,
        'stress_frames': args.stress_frames,
        'max_train_clips': args.max_train_clips,
        'max_val_clips': args.max_val_clips,
        'max_test_clips': args.max_test_clips,
        'gate_weight': args.gate_weight,
        'seed': args.seed,
        'floors': gate_floors,
        'stress_opacities': stress_opacities,
        'skip_existing': args.skip_existing,
        'aggregate': args.aggregate,
        'gating_mode': args.gating_mode,
        'use_logit_bias': bool(args.use_logit_bias),
        'use_estimator_calibration': bool(args.use_estimator_calibration),
        'gate_dropout': float(args.gate_dropout),
    }
    with open(os.path.join(sweep_dir, 'sweep_config.json'), 'w') as f:
        json.dump(sweep_manifest, f, indent=2)

    # ── Load extraction models (once) ────────────────────────────────────────
    feat_extractor, occ_model, face_detector = _load_extraction_models(
        device, face_type=args.face, det_size=args.det_size)

    t_sweep_start = time.time()

    for fold_idx in fold_ids:
        fold_cfg = all_splits[min(fold_idx, len(all_splits) - 1)]
        train_subjects = set(fold_cfg['train_subjects'])
        test_subjects = set(fold_cfg['test_subjects'])
        train_keys = [k for k, v in csv_data.items() if v['subject'] in train_subjects]
        test_keys = [k for k, v in csv_data.items() if v['subject'] in test_subjects]
        splits = {
            'train': train_keys, 'val': train_keys, 'test': test_keys,
            'split_config': fold_cfg,
        }

        # Check if all floors for this fold are done already
        if args.skip_existing:
            all_done = True
            for gf in gate_floors:
                if is_kfold:
                    rd = os.path.join(sweep_dir, f'floor_{gf:.2f}', f'fold_{fold_idx:02d}')
                else:
                    rd = os.path.join(sweep_dir, f'floor_{gf:.2f}')
                if not os.path.exists(os.path.join(rd, 'eval_metrics.json')):
                    all_done = False
                    break
            if all_done:
                print(f'\n  >>> Fold {fold_idx}: all floors complete, skipping')
                continue

        print(f'\n{"#"*70}')
        print(f'  FOLD {fold_idx}: train={fold_cfg["train_subjects"]}, '
              f'test={fold_cfg["test_subjects"]}')
        print(f'{"#"*70}')

        # ── Feature extraction (once per fold) ───────────────────────────────
        t0 = time.time()
        _set_global_seed(args.seed)
        opacity_sampler = {'mode': 'discrete', 'values': [0.4, 0.6, 0.8, 1.0], 'weights': None}

        print('\n=== Stage A: Feature extraction (shared across all floors) ===')
        train_samples, val_samples, test_samples, test_clips, val_clips = \
            extract_features_for_clips(
                csv_data, fold_cfg,
                face_detector=face_detector, feat_extractor=feat_extractor,
                occ_model=occ_model, val_ratio=0.20, seed=args.seed,
                max_train_clips=args.max_train_clips,
                max_val_clips=args.max_val_clips,
                max_test_clips=args.max_test_clips,
                skip_test=False, train_opacity_sampler=opacity_sampler)

        feat_time = time.time() - t0
        print(f'  Feature extraction: {feat_time:.0f}s')
        print(f'  Train: {len(train_samples)}  Val: {len(val_samples)}  Test: {len(test_samples)}')
        for name, ss in [('Train', train_samples), ('Val', val_samples), ('Test', test_samples)]:
            if ss:
                print(f'    {name}: {Counter(s["class_name"] for s in ss)}')

        # ── Stress feature cache (once per fold) ────────────────────────────
        t0 = time.time()
        print('\n=== Stage B: Stress-test feature cache (shared across all floors) ===')
        stress_cache = precompute_stress_features(
            csv_data=csv_data,
            test_clips=test_clips,
            face_detector=face_detector,
            feat_extractor=feat_extractor,
            occ_model=occ_model,
            stress_opacities=stress_opacities,
            max_frames_per_clip=max(1, min(8, args.stress_frames // 2)),
            seed=args.seed,
        )
        cache_time = time.time() - t0
        print(f'  Stress cache: {cache_time:.0f}s')

        # ── Loop over gate_floor values ──────────────────────────────────────
        print(f'\n=== Stage C: Training + evaluation (per floor) ===')
        for gate_floor in gate_floors:
            if is_kfold:
                run_dir = os.path.join(sweep_dir, f'floor_{gate_floor:.2f}', f'fold_{fold_idx:02d}')
            else:
                run_dir = os.path.join(sweep_dir, f'floor_{gate_floor:.2f}')

            if args.skip_existing and os.path.exists(os.path.join(run_dir, 'eval_metrics.json')):
                print(f'\n  >>> Skipping floor={gate_floor:.2f} fold={fold_idx} (exists)')
                continue

            _train_and_eval_one_floor(
                gate_floor=gate_floor,
                train_samples=train_samples,
                val_samples=val_samples,
                test_samples=test_samples,
                stress_cache=stress_cache,
                device=device,
                run_dir=run_dir,
                args=args,
                fold=fold_idx,
                splits=splits,
            )

        del train_samples, val_samples, test_samples, test_clips, val_clips, stress_cache
        _gpu_cleanup(f'fold {fold_idx} complete')

    del feat_extractor, occ_model, face_detector
    _gpu_cleanup('extraction models freed')

    total_time = time.time() - t_sweep_start
    h, rem = divmod(total_time, 3600)
    m, s = divmod(rem, 60)
    print(f'\n{"="*70}')
    print(f'  Sweep complete: {int(h)}h {int(m)}m {s:.0f}s')
    print(f'{"="*70}')

    # ── Aggregation ──────────────────────────────────────────────────────────
    if args.aggregate:
        print('\nRunning aggregation...')
        import subprocess
        result = subprocess.run([sys.executable, 'aggregate_gate_floor_sweep.py',
                                 '--sweep-dir', sweep_dir], check=False)
        if result.returncode != 0:
            print(f'WARNING: aggregation failed with exit code {result.returncode}. '
                  f'You can rerun it manually on {sweep_dir}.')

    print('\nDone.')


if __name__ == '__main__':
    main()
