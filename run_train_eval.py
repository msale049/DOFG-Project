#!/usr/bin/env python3
"""
run_train_eval.py
=================
End-to-end training, evaluation, and stress test with persistent results.

Saves to results/run_YYYYMMDD_HHMMSS/:
  - config.json
  - training_history.json
  - training_curves.png
  - eval_metrics.json
  - stress_test_results.csv
  - gating_on_vs_off.png
  - opacity_analysis.png
  - occlusion_visualization.png
  - model_best.pt

Usage
-----
    # Small subset (CPU, ~5–15 min with stress test)
    python run_train_eval.py --samples 20 --epochs 3 --stress-frames 15

    # Full run (GPU)
    python run_train_eval.py --samples 0 --epochs 20

    # With fixed split (default)
    python run_train_eval.py --mode fixed --num-test 3

    # With k-fold
    python run_train_eval.py --mode kfold --k 5
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Avoid ONNX/OpenMP issues on HPC and sandbox
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

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
from evaluation import compute_metrics_on_loader
from stress_test import run_stress_test
from visualize_occlusion import generate_occlusion_grid_png


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
        'split_config': cfg,  # for clip-based pipeline: train_subjects, test_subjects
    }


def _ensure_results_dir() -> str:
    """Create results/run_YYYYMMDD_HHMMSS and return path."""
    os.makedirs('results', exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('results', f'run_{stamp}')
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _save_training_curves(history: dict, run_dir: str):
    """Plot and save training loss and accuracy curves."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
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
    plt.savefig(os.path.join(run_dir, 'training_curves.png'), bbox_inches='tight')
    plt.close()


def _plot_gating_comparison(summary_df: pd.DataFrame, run_dir: str):
    """Plot gating ON vs OFF comparison by (occlusion_type/condition, opacity)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if len(summary_df) == 0:
        return
    col = 'condition' if 'condition' in summary_df.columns else 'occlusion_type'
    labels = [f"{r[col]}_{r['opacity']:.1f}" for _, r in summary_df.iterrows()]
    x = np.arange(len(labels))
    w = 0.35
    on_vals = summary_df['acc_gating_on'].values
    off_vals = summary_df['acc_gating_off'].values
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 5))
    ax.bar(x - w/2, on_vals, w, label='Gating ON')
    ax.bar(x + w/2, off_vals, w, label='Gating OFF')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Gating ON vs OFF on Synthetic Occlusion')
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'gating_on_vs_off.png'), bbox_inches='tight')
    plt.close()


def _plot_opacity_analysis(summary_df: pd.DataFrame, run_dir: str):
    """Plot accuracy vs opacity level for eye/mouth/both conditions."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if len(summary_df) == 0:
        return
    col = 'condition' if 'condition' in summary_df.columns else 'occlusion_type'
    fig, ax = plt.subplots(figsize=(8, 5))
    # Match both legacy (eye_only) and strategy (persistent_eye) names
    for ot, label in [('eye_only', 'Eye'), ('mouth_only', 'Mouth'), ('both', 'Both'),
                      ('persistent_eye', 'Eye'), ('persistent_mouth', 'Mouth'), ('persistent_both', 'Both')]:
        sub = summary_df[summary_df[col] == ot]
        if len(sub) == 0:
            continue
        if len(sub) > 0:
            ax.plot(sub['opacity'], sub['acc_gating_on'], 'o-', label=label)
    ax.set_xlabel('Opacity')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy vs Opacity Level (Gating ON)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'opacity_analysis.png'), bbox_inches='tight')
    plt.close()


def _plot_gates_vs_opacity(details_df: pd.DataFrame, run_dir: str):
    """Plot gate values and p_eye/p_mouth vs opacity (from notebook)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if len(details_df) == 0:
        return
    col = 'condition' if 'condition' in details_df.columns else 'occlusion_type'
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    eye_conds = ['eye_only', 'persistent_eye', 'transient_eye']
    mouth_conds = ['mouth_only', 'persistent_mouth', 'transient_mouth']
    both_conds = ['both', 'persistent_both']
    for ax, conds, title in zip(axes,
                               [eye_conds, mouth_conds, both_conds],
                               ['Eye occlusion', 'Mouth occlusion', 'Both']):
        sub = details_df[details_df[col].isin(conds)]
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
        ax.legend()
        ax.grid(True)
        ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'gates_vs_opacity.png'), bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser(description='Train, evaluate, and stress-test DOFG pipeline')
    ap.add_argument('--data', default='Data', help='Path to Data folder')
    ap.add_argument('--samples', type=int, default=30,
                    help='Samples per video (0=all). Use 20–30 for quick CPU test')
    ap.add_argument('--epochs', type=int, default=5, help='Training epochs')
    ap.add_argument('--batch', type=int, default=16, help='Batch size')
    ap.add_argument('--mode', choices=['fixed', 'kfold', 'loso'], default='fixed')
    ap.add_argument('--k', type=int, default=5, help='k for k-fold')
    ap.add_argument('--num-test', type=int, default=3, help='Test subjects for fixed split')
    ap.add_argument('--face', choices=['dlib', 'retina'], default='retina',
                    help='Face detector (retina=RetinaFace, dlib=fallback)')
    ap.add_argument('--det-size', type=int, default=640,
                    help='RetinaFace input size (320/480/640). Smaller=less GPU memory, use 320 if OOM')
    ap.add_argument('--strategy', choices=['clip', 'legacy'], default='clip',
                    help='clip=STRATEGY_DESIGN (clips, regime aug, temporal val); legacy=frame sampling')
    ap.add_argument('--max-train-clips', type=int, default=None,
                    help='Max train clips for clip strategy (for quick test; default=all)')
    ap.add_argument('--max-val-clips', type=int, default=None,
                    help='Max val clips for clip strategy')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--stress', action='store_true', default=True,
                    help='Run stress test (default: True)')
    ap.add_argument('--no-stress', action='store_false', dest='stress')
    ap.add_argument('--stress-frames', type=int, default=20,
                    help='Max frames per video for stress test (smaller = faster)')
    ap.add_argument('--stress-opacities', type=str, default='0,0.5,1',
                    help='Comma-separated opacity levels for stress test')
    ap.add_argument('--fold', type=int, default=0,
                    help='Fold index for kfold/loso (0-based); fixed uses fold 0 only')
    ap.add_argument('--defer-test', action='store_true', default=True,
                    help='[clip] Defer test load until stress test (saves time); default on')
    ap.add_argument('--no-defer-test', action='store_false', dest='defer_test')
    ap.add_argument('--benchmark', action='store_true',
                    help='Run latency benchmark (GPU warm-up + per-phase ms) for paper')
    ap.add_argument('--face-cpu', action='store_true',
                    help='Force RetinaFace to CPU (avoids GPU OOM; PyTorch keeps GPU)')
    args = ap.parse_args()

    if args.face_cpu:
        os.environ['DOFG_FACE_CPU'] = '1'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = _ensure_results_dir()
    print(f'Results dir: {run_dir}')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')
    print(f'Mode: {args.mode}, samples/video: {args.samples or "ALL"}, epochs: {args.epochs}')

    # Save config
    config_dict = {
        'data': args.data,
        'samples_per_video': args.samples or 'all',
        'epochs': args.epochs,
        'batch_size': args.batch,
        'mode': args.mode,
        'k': args.k,
        'num_test': args.num_test,
        'fold': args.fold,
        'face_detector': args.face,
        'det_size': args.det_size if args.face == 'retina' else None,
        'face_cpu': args.face_cpu,
        'seed': args.seed,
        'stress_test': args.stress,
        'stress_frames': args.stress_frames,
        'defer_test': args.defer_test,
    }
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

    # 1. Load data
    csv_data = load_csv_video_data(args.data, filter_eye_states=True)
    if not csv_data:
        print('No data found. Check --data path.')
        sys.exit(1)

    splits = _splits_from_fixed_or_fold(
        csv_data, mode=args.mode, k=args.k,
        num_test=args.num_test, seed=args.seed, fold=args.fold,
    )
    split_config = splits.get('split_config', {})
    print(f'Train videos: {len(splits["train"])}, Test videos: {len(splits["test"])}')
    print(f'Strategy: {args.strategy}')

    # 2. Load models — PyTorch FIRST (ONNXRuntime needs torch imported first for CUDA; cudnn.benchmark can OOM on H100)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False  # avoids large alloc on first run
        torch.cuda.empty_cache()
    import gc
    from feature_extraction import ResNet34FeatureExtractor
    from occlusion_estimator import ResNet34OcclusionModel

    print('Loading PyTorch models (feature extractor, occlusion) onto GPU...')
    feat_extractor = ResNet34FeatureExtractor(
        CONFIG['RESNET34_MODEL_PATH'], device=str(device),
    )
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    occ_model = ResNet34OcclusionModel(
        CONFIG['RESNET34_OCCLUSION_MODEL_PATH'], device=str(device),
    )

    # Now load face detector (RetinaFace can use remaining GPU or fall back to CPU)
    det_size = (args.det_size, args.det_size) if args.face == 'retina' else None
    try:
        if args.face == 'retina':
            from face_detection_retinaface import FaceDetector
            face_detector = FaceDetector(
                shape_model_path=CONFIG['DLIB_MODEL_PATH'],
                det_size=det_size, det_thresh=0.35,
            )
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

    # 3. Extract features
    print('\nExtracting features ...')
    if args.strategy == 'clip':
        print('  Using clip-based pipeline (STRATEGY_DESIGN: T=32, regime aug, temporal val)')
        train_samples, val_samples, test_samples, test_clips = extract_features_for_clips(
            csv_data, split_config,
            face_detector=face_detector,
            feat_extractor=feat_extractor,
            occ_model=occ_model,
            val_ratio=0.20,
            seed=args.seed,
            max_train_clips=args.max_train_clips,
            max_val_clips=args.max_val_clips,
            skip_test=args.defer_test,
        )
    else:
        num_samples = args.samples if args.samples > 0 else None
        print(f'  Using legacy frame sampling (num_samples_per_video={num_samples or "ALL"})')
        train_samples, val_samples, test_samples = extract_features_stratified(
            csv_data, splits,
            face_detector=face_detector,
            feat_extractor=feat_extractor,
            occ_model=occ_model,
            num_samples_per_video=num_samples,
            val_ratio=0.20,
            random_state=args.seed,
        )
        test_clips = []  # legacy uses frame-based stress test

    print(f'\nTrain: {len(train_samples)}  Val: {len(val_samples)}  Test: {len(test_samples)}')
    for name, ss in [('Train', train_samples), ('Val', val_samples), ('Test', test_samples)]:
        if ss:
            print(f'  {name}: {Counter(s["class_name"] for s in ss)}')

    # 4. DataLoaders (test_loader only when test_samples non-empty)
    train_ds = DriverStateDataset(train_samples, device=str(device))
    val_ds = DriverStateDataset(val_samples, device=str(device))
    test_loader = None
    if test_samples:
        test_ds = DriverStateDataset(test_samples, device=str(device))
        test_loader = DataLoader(
            test_ds, batch_size=args.batch, shuffle=False,
            collate_fn=test_ds.collate_samples,
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        collate_fn=train_ds.collate_samples, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        collate_fn=val_ds.collate_samples,
    )

    # 5. Model and trainer
    model = EnhancedOcclusionAwareTransformer(
        feature_dim=512, hidden_dim=128, num_heads=4,
        num_classes=3, num_layers=3, use_relative_pos=True,
    ).to(device)
    print(f'\nModel params: {sum(p.numel() for p in model.parameters()):,}')

    trainer = TinyTransformerTrainer(model, device=str(device), learning_rate=3e-5)

    # 6. Train
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch=epoch)
        val_metrics = trainer.evaluate(val_loader, name='VAL')
        val_acc = val_metrics.get('accuracy', 0.0)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(run_dir, 'model_best.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f'  ** Best val: {best_val_acc:.1f}% → saved to {ckpt_path}')
        print(f'  Ep {epoch+1}/{args.epochs}: train_acc={train_metrics.get("accuracy", 0):.1f}% '
              f'val_acc={val_acc:.1f}%')

    # Save training history
    history = trainer.history
    with open(os.path.join(run_dir, 'training_history.json'), 'w') as f:
        json.dump({k: (v if isinstance(v, list) else str(v))
                   for k, v in history.items()}, f, indent=2)
    _save_training_curves(history, run_dir)

    # 7. Load best checkpoint
    ckpt = torch.load(os.path.join(run_dir, 'model_best.pt'), map_location=device)
    model.load_state_dict(ckpt)

    # 8. Stress test (loads test frames when defer_test; clean condition = test eval)
    stress_details = pd.DataFrame()
    stress_summary = pd.DataFrame()
    if args.stress and splits['test']:
        print('\n--- Stress test (loads test frames; clean condition = test eval) ---')
        opacity_levels = [float(x.strip()) for x in args.stress_opacities.split(',') if x.strip()]
        stress_details, stress_summary = run_stress_test(
            csv_data=csv_data,
            test_keys=splits['test'],
            model=model,
            face_detector=face_detector,
            feat_extractor=feat_extractor,
            occ_model=occ_model,
            trainer=trainer,
            device=str(device),
            opacity_levels=opacity_levels,
            max_frames_per_video=args.stress_frames,
            batch_size=args.batch,
            seed=args.seed,
            test_clips=test_clips if args.strategy == 'clip' else None,
            max_frames_per_clip=min(8, args.stress_frames // 2) if args.strategy == 'clip' else None,
        )
        if len(stress_details) > 0:
            details_path = os.path.join(run_dir, 'stress_test_details.csv')
            summary_path = os.path.join(run_dir, 'stress_test_summary.csv')
            stress_details.to_csv(details_path, index=False)
            stress_summary.to_csv(summary_path, index=False)
            print(f'\nSaved stress test details ({len(stress_details)} rows) to {details_path}')
            print(f'Saved stress test summary to {summary_path}')
            print('\nSummary:')
            print(stress_summary.to_string(index=False))
            _plot_gating_comparison(stress_summary, run_dir)
            _plot_opacity_analysis(stress_summary, run_dir)
            _plot_gates_vs_opacity(stress_details, run_dir)
        else:
            print('  No stress test results (check test videos and face detection).')

    # 9. Test evaluation: from stress clean condition (defer_test) or test_loader
    if args.defer_test and not args.stress:
        eval_dict = {'accuracy': 0, 'n_samples': 0, 'source': 'defer_test_no_stress', 'note': 'Run with --stress for eval when using --defer-test'}
    elif args.defer_test and len(stress_summary) > 0:
        clean_row = stress_summary[stress_summary['condition'] == 'clean']
        if len(clean_row) > 0:
            eval_dict = {
                'accuracy': float(clean_row['acc_gating_on'].iloc[0]),
                'loss': None,
                'precision': None, 'recall': None, 'f1': None,
                'n_samples': int(clean_row['n'].iloc[0]),
                'source': 'stress_test_clean_condition',
            }
        else:
            eval_dict = {'accuracy': 0, 'n_samples': 0, 'source': 'stress_test_clean_condition'}
    elif test_loader is not None:
        print('\n--- Test evaluation (clean, from pre-extracted) ---')
        test_metrics = compute_metrics_on_loader(trainer, test_loader, compute_loss=True)
        eval_dict = {
            'accuracy': test_metrics.get('accuracy', 0),
            'loss': test_metrics.get('loss'),
            'precision': test_metrics.get('precision'),
            'recall': test_metrics.get('recall'),
            'f1': test_metrics.get('f1'),
            'n_samples': len(test_samples),
            'source': 'test_loader',
        }
    else:
        eval_dict = {'accuracy': 0, 'n_samples': 0, 'source': 'none'}

    with open(os.path.join(run_dir, 'eval_metrics.json'), 'w') as f:
        json.dump(eval_dict, f, indent=2)
    print(f'\n--- Test evaluation (clean) ---')
    print(f'  Test accuracy: {eval_dict["accuracy"]:.1f}%')
    if eval_dict.get('loss') is not None:
        print(f'  Test loss: {eval_dict["loss"]:.4f}')
    if eval_dict.get('precision') is not None:
        print(f'  Precision: {eval_dict["precision"]:.4f}')
    if eval_dict.get('recall') is not None:
        print(f'  Recall: {eval_dict["recall"]:.4f}')
    if eval_dict.get('f1') is not None:
        print(f'  F1: {eval_dict["f1"]:.4f}')

    # 10. Occlusion visualization (single PNG grid for train + stress + legacy)
    occ_png = os.path.join(run_dir, 'occlusion_visualization.png')
    if generate_occlusion_grid_png(csv_data, occ_png, face_detector=face_detector, face_type=args.face):
        print(f'\nSaved occlusion grid to {occ_png}')
    else:
        print('\nOcclusion visualization skipped (no valid frame or matplotlib missing).')

    # 11. Latency benchmark (for paper)
    if args.benchmark:
        from stress_test import run_latency_benchmark
        test_keys_bench = splits['test'] if splits['test'] else list(csv_data.keys())[:1]
        latency = run_latency_benchmark(
            model, face_detector, feat_extractor, occ_model,
            csv_data, test_keys_bench,
            device=str(device), num_warmup=20, num_iter=50,
        )
        with open(os.path.join(run_dir, 'latency_report.json'), 'w') as f:
            json.dump(latency, f, indent=2)
        print('\n--- Latency benchmark (for paper) ---')
        for k, v in latency.items():
            if isinstance(v, dict) and 'mean_ms' in v:
                print(f'  {k}: {v["mean_ms"]:.2f} ± {v.get("std_ms", 0):.2f} ms')
            elif not isinstance(v, dict):
                print(f'  {k}: {v}')

    print(f'\nDone. All results saved to {run_dir}')


if __name__ == '__main__':
    main()
