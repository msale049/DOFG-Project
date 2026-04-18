#!/usr/bin/env python3
"""
visualize_occlusion.py
======================
Generate sample images showing all synthetic occlusion types for visualization.

Covers:
- Training regimes (clip strategy): clean, persistent_eye, persistent_mouth,
  persistent_both, transient_eye, transient_mouth at opacity 0.5, 0.8, 1.0
- Stress conditions (clip): clean + 5 stress at 0.8
- Legacy: eye_only, mouth_only, both at opacities 0.3, 0.5, 0.7, 0.9, 1.0

Usage
-----
    python visualize_occlusion.py --data Data --out occlusion_samples
    python visualize_occlusion.py --data Data --out occlusion_samples --face dlib
"""

import argparse
import os

import cv2
import numpy as np

from config import CONFIG
from data_loading import load_csv_video_data
from synthetic_occlusion import (
    apply_synthetic_occlusion,
    apply_eye_band,
    apply_mouth_rect,
    OPACITY_LEVELS,
)
from occlusion_augmentation import apply_occlusion_to_frame, get_transient_segment


def _load_face_detector(face_type: str = 'dlib'):
    """Load the requested face detector with dlib fallback."""
    try:
        if face_type == 'retina':
            from face_detection_retinaface import FaceDetector
            return FaceDetector(
                shape_model_path=CONFIG['DLIB_MODEL_PATH'],
                det_size=(640, 640),
                det_thresh=0.35,
            )
        from face_detection_dlib import FaceDetector
        return FaceDetector(shape_model_path=CONFIG['DLIB_MODEL_PATH'])
    except ImportError:
        from face_detection_dlib import FaceDetector
        return FaceDetector(shape_model_path=CONFIG['DLIB_MODEL_PATH'])


def _collect_valid_face_samples(
    csv_data: dict,
    face_detector,
    num_samples: int = 4,
    frames_to_try_per_video: int = 24,
):
    """Collect a small set of clean frames with valid landmarks."""
    samples = []

    for video_key, meta in csv_data.items():
        anns = meta.get('annotations', [])
        if not anns:
            continue

        cap = cv2.VideoCapture(meta['video_path'])
        if not cap.isOpened():
            continue

        try_indices = np.linspace(
            0, len(anns) - 1, min(frames_to_try_per_video, len(anns)), dtype=int)

        found_for_video = False
        for idx in try_indices:
            ann = anns[int(idx)]
            cap.set(cv2.CAP_PROP_POS_FRAMES, ann.frame)
            ret, bgr = cap.read()
            if not ret:
                continue

            det = face_detector.detect_face_and_landmarks(bgr)
            if not det['is_valid'] or det.get('landmarks') is None:
                continue

            samples.append({
                'rgb': cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                'landmarks': det['landmarks'],
                'subject': meta.get('subject', 'unknown'),
                'video_key': video_key,
                'frame': ann.frame,
                'class_label': ann.class_label,
            })
            found_for_video = True
            break

        cap.release()

        if found_for_video and len(samples) >= num_samples:
            break

    return samples


def generate_clean_vs_synthetic_opacity_grid(
    csv_data: dict,
    output_path: str,
    face_detector=None,
    face_type: str = 'dlib',
    opacities=None,
    n_clean_samples: int = 4,
) -> bool:
    """
    Generate a figure with:
    - a strip of clean frames from the dataset
    - an opacity sweep for eye, mouth, and combined synthetic occlusions

    Returns True if saved successfully.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
    except ImportError:
        return False

    if not csv_data:
        return False

    if opacities is None:
        opacities = OPACITY_LEVELS
    opacities = list(opacities)

    if face_detector is None:
        face_detector = _load_face_detector(face_type)

    clean_samples = _collect_valid_face_samples(
        csv_data, face_detector, num_samples=max(1, n_clean_samples))
    if not clean_samples:
        return False

    base = clean_samples[0]
    base_rgb = base['rgb']
    base_lm = base['landmarks']

    fig = plt.figure(figsize=(2.9 * len(opacities), 9.5))
    outer = gridspec.GridSpec(2, 1, height_ratios=[1.05, 3.1], hspace=0.22)

    clean_grid = gridspec.GridSpecFromSubplotSpec(
        1, len(clean_samples), subplot_spec=outer[0], wspace=0.05)
    for idx, sample in enumerate(clean_samples):
        ax = fig.add_subplot(clean_grid[0, idx])
        ax.imshow(sample['rgb'])
        ax.set_title(
            f'Clean {idx + 1}\n{sample["class_label"]} | {sample["subject"]}',
            fontsize=9,
        )
        ax.axis('off')

    sweep_grid = gridspec.GridSpecFromSubplotSpec(
        3, len(opacities), subplot_spec=outer[1], wspace=0.04, hspace=0.08)
    row_specs = [
        ('Eye occlusion', lambda img, lm, op: apply_eye_band(img, lm, opacity=op)),
        ('Mouth occlusion', lambda img, lm, op: apply_mouth_rect(img, lm, opacity=op)),
        ('Both', lambda img, lm, op: apply_synthetic_occlusion(
            img, lm, eye_opacity=op, mouth_opacity=op)),
    ]

    for row_idx, (row_label, apply_fn) in enumerate(row_specs):
        for col_idx, opacity in enumerate(opacities):
            ax = fig.add_subplot(sweep_grid[row_idx, col_idx])
            if opacity <= 0:
                aug = base_rgb.copy()
            else:
                aug = apply_fn(base_rgb.copy(), base_lm, float(opacity))
            ax.imshow(aug)
            if row_idx == 0:
                col_title = 'Clean' if opacity <= 0 else f'Opacity {opacity:.1f}'
                ax.set_title(col_title, fontsize=10)
            if col_idx == 0:
                ax.text(
                    -0.08, 0.5, row_label,
                    transform=ax.transAxes,
                    ha='right',
                    va='center',
                    fontsize=10,
                    fontweight='bold',
                )
            ax.axis('off')

    fig.suptitle(
        'Clean vs Synthetic Occlusion Examples',
        fontsize=14,
        fontweight='bold',
        y=0.98,
    )
    fig.text(
        0.5, 0.015,
        'Bottom sweep uses one representative clean frame rendered with the same '
        'eye-band and mouth-rectangle synthetic occlusion utilities used in training/stress code.',
        ha='center',
        va='bottom',
        fontsize=9,
    )
    plt.savefig(output_path, bbox_inches='tight', dpi=160)
    plt.close(fig)
    return True


def generate_occlusion_grid_png(
    csv_data: dict,
    output_path: str,
    face_detector=None,
    face_type: str = 'dlib',
) -> bool:
    """
    Generate a single PNG with grid of representative occlusion samples
    (train regimes + stress/test conditions + legacy), like Jupyter style.
    Returns True if saved successfully.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    if not csv_data:
        return False

    # Load face detector if not provided
    if face_detector is None:
        face_detector = _load_face_detector(face_type)

    video_key = list(csv_data.keys())[0]
    meta = csv_data[video_key]
    cap = cv2.VideoCapture(meta['video_path'])
    anns = meta['annotations']
    if not anns:
        return False

    bgr = None
    det = None
    for i in np.linspace(0, len(anns) - 1, min(50, len(anns)), dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, anns[i].frame)
        ret, bgr = cap.read()
        if not ret:
            continue
        det = face_detector.detect_face_and_landmarks(bgr)
        if det['is_valid'] and det.get('landmarks') is not None:
            break
    cap.release()

    if bgr is None or not det['is_valid'] or det.get('landmarks') is None:
        return False

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lm = det['landmarks']
    T = 32
    clip_seed = 42
    seg_len = max(4, T // 4)
    seg_start = (T - seg_len) // 2

    actual_seg_start, actual_seg_end = get_transient_segment(T, clip_seed)
    actual_seg_mid = (actual_seg_start + actual_seg_end) // 2

    def _apply_regime(regime, opacity, use_middle_seg=False):
        if regime == 'clean' or opacity <= 0:
            return rgb.copy()
        if regime.startswith('transient') and use_middle_seg:
            if 'eye' in regime:
                return apply_eye_band(rgb, lm, opacity=opacity)
            return apply_mouth_rect(rgb, lm, opacity=opacity)
        fi = actual_seg_mid if regime.startswith('transient') else T // 2
        return apply_occlusion_to_frame(rgb, lm, regime, opacity, fi, T, clip_seed)

    # Build grid: 5 rows x 3 cols
    # Row 1-2: Training regimes (6)
    # Row 3-4: Stress/test conditions (6)
    # Row 5: Legacy samples (eye_only, mouth_only, both)
    regimes = [
        ('clean', 0.0), ('persistent_eye', 0.8), ('persistent_mouth', 0.8),
        ('persistent_both', 0.8), ('transient_eye', 0.8), ('transient_mouth', 0.8),
    ]
    legacy_samples = [
        ('eye_only', 0.5), ('mouth_only', 0.5), ('both', 0.5),
    ]

    fig, axes = plt.subplots(5, 3, figsize=(10, 14))
    for idx, (regime, opacity) in enumerate(regimes):
        ax = axes[idx // 3, idx % 3]
        aug = _apply_regime(regime, opacity, use_middle_seg=(regime.startswith('transient')))
        ax.imshow(aug)
        ax.set_title(f'Train: {regime} op={opacity}', fontsize=9)
        ax.axis('off')

    for idx, (regime, opacity) in enumerate(regimes):
        ax = axes[2 + idx // 3, idx % 3]
        aug = _apply_regime(regime, opacity, use_middle_seg=(regime.startswith('transient')))
        ax.imshow(aug)
        ax.set_title(f'Stress: {regime} op={opacity}', fontsize=9)
        ax.axis('off')

    for idx, (occ_name, op) in enumerate(legacy_samples):
        ax = axes[4, idx]
        eo = op if occ_name in ('eye_only', 'both') else 0.0
        mo = op if occ_name in ('mouth_only', 'both') else 0.0
        aug = apply_synthetic_occlusion(rgb, lm, eye_opacity=eo, mouth_opacity=mo)
        ax.imshow(aug)
        ax.set_title(f'Legacy: {occ_name} op={op}', fontsize=9)
        ax.axis('off')

    fig.suptitle('Synthetic Occlusion: Train Regimes, Stress Test, Legacy', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=120)
    plt.close()
    return True


def main():
    ap = argparse.ArgumentParser(description='Visualize synthetic occlusion samples')
    ap.add_argument('--data', default='Data', help='Path to Data folder')
    ap.add_argument('--out', default='occlusion_samples', help='Output directory')
    ap.add_argument('--face', choices=['dlib', 'retina'], default='retina')
    ap.add_argument('--num-frames', type=int, default=2, help='Frames to sample per video')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    detector = _load_face_detector(args.face)

    csv_data = load_csv_video_data(args.data, filter_eye_states=True)
    if not csv_data:
        print('No data found.')
        return

    # Get one frame with valid face from first video
    video_key = list(csv_data.keys())[0]
    meta = csv_data[video_key]
    cap = cv2.VideoCapture(meta['video_path'])
    anns = meta['annotations']
    if not anns:
        print('No annotations.')
        return

    frames_tried = 0
    bgr = None
    det = None
    for i in np.linspace(0, len(anns) - 1, min(args.num_frames * 10, len(anns)), dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, anns[i].frame)
        ret, bgr = cap.read()
        if not ret:
            continue
        det = detector.detect_face_and_landmarks(bgr)
        if det['is_valid'] and det.get('landmarks') is not None:
            break
        frames_tried += 1
    cap.release()

    if bgr is None or not det['is_valid'] or det.get('landmarks') is None:
        print('Could not find a frame with valid face.')
        return

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lm = det['landmarks']
    T = 32  # clip length for transient
    clip_seed = 42

    # 1. Training regimes (persistent + transient at frame 0 and middle)
    # Transient: random segment from clip_seed; for fi=0 often outside, fi=T//2 may be in
    print('Saving training regime samples...')
    regimes = [
        ('clean', 0.0),
        ('persistent_eye', 0.8),
        ('persistent_mouth', 0.8),
        ('persistent_both', 0.8),
        ('transient_eye', 0.8),
        ('transient_mouth', 0.8),
    ]
    seg_start, seg_end = get_transient_segment(T, clip_seed)
    seg_mid = (seg_start + seg_end) // 2
    for regime, opacity in regimes:
        if regime.startswith('transient'):
            frame_indices = [0, seg_mid]
        else:
            frame_indices = [0, T // 2]
        for fi, frame_idx in enumerate(frame_indices):
            aug = apply_occlusion_to_frame(rgb, lm, regime, opacity, frame_idx, T, clip_seed)
            out_bgr = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)
            fname = f'train_{regime}_op{opacity}_fi{frame_idx}.jpg'
            cv2.imwrite(os.path.join(args.out, fname), out_bgr)

    # 2. Stress conditions (transient uses middle T/4 like stress test)
    print('Saving stress condition samples...')
    seg_len = max(4, T // 4)
    seg_start = (T - seg_len) // 2
    for cond, regime, opacity in [
        ('clean', 'clean', 0.0),
        ('persistent_eye', 'persistent_eye', 0.8),
        ('persistent_mouth', 'persistent_mouth', 0.8),
        ('persistent_both', 'persistent_both', 0.8),
        ('transient_eye', 'transient_eye', 0.8),
        ('transient_mouth', 'transient_mouth', 0.8),
    ]:
        if regime == 'clean' or opacity <= 0:
            aug = rgb.copy()
        elif regime.startswith('transient'):
            in_seg = seg_start <= (T // 2) < seg_start + seg_len
            if in_seg:
                if 'eye' in regime:
                    aug = apply_eye_band(rgb, lm, opacity=opacity)
                else:
                    aug = apply_mouth_rect(rgb, lm, opacity=opacity)
            else:
                aug = rgb.copy()
        else:
            aug = apply_occlusion_to_frame(rgb, lm, regime, opacity, T // 2, T, clip_seed)
        out_bgr = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.out, f'stress_{cond}_op{opacity}.jpg'), out_bgr)

    # 3. Legacy: eye_only, mouth_only, both × opacities
    print('Saving legacy occlusion samples...')
    for occ_name, eflag, mflag in [('none', 0.0, 0.0), ('eye_only', 1.0, 0.0), ('mouth_only', 0.0, 1.0), ('both', 1.0, 1.0)]:
        for op in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
            if occ_name == 'none' and op > 0:
                continue
            if occ_name != 'none' and op == 0:
                continue
            eo = op if eflag else 0.0
            mo = op if mflag else 0.0
            aug = apply_synthetic_occlusion(rgb, lm, eye_opacity=eo, mouth_opacity=mo)
            out_bgr = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args.out, f'legacy_{occ_name}_op{op}.jpg'), out_bgr)

    # 4. Grid summary image (training regimes)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(9, 6))
        for idx, (regime, opacity) in enumerate(regimes):
            aug = apply_occlusion_to_frame(rgb, lm, regime, opacity, T // 2, T, clip_seed)
            ax = axes[idx // 3, idx % 3]
            ax.imshow(aug)
            ax.set_title(f'{regime} op={opacity}')
            ax.axis('off')
        plt.suptitle('Training Regimes (sample)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'summary_training.jpg'), bbox_inches='tight')
        plt.close()
    except ImportError:
        pass

    opacity_grid_path = os.path.join(args.out, 'clean_vs_synthetic_opacity_grid.png')
    if generate_clean_vs_synthetic_opacity_grid(
        csv_data, opacity_grid_path, face_detector=detector, face_type=args.face):
        print(f'Saved clean vs synthetic opacity grid to {opacity_grid_path}')

    print(f'Saved occlusion samples to {args.out}/')


if __name__ == '__main__':
    main()
