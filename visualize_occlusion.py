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
        try:
            if face_type == 'retina':
                from face_detection_retinaface import FaceDetector
                face_detector = FaceDetector(
                    shape_model_path=CONFIG['DLIB_MODEL_PATH'],
                    det_size=(640, 640), det_thresh=0.35,
                )
            else:
                from face_detection_dlib import FaceDetector
                face_detector = FaceDetector(shape_model_path=CONFIG['DLIB_MODEL_PATH'])
        except ImportError:
            from face_detection_dlib import FaceDetector
            face_detector = FaceDetector(shape_model_path=CONFIG['DLIB_MODEL_PATH'])

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

    # Load face detector
    try:
        if args.face == 'retina':
            from face_detection_retinaface import FaceDetector
            detector = FaceDetector(shape_model_path=CONFIG['DLIB_MODEL_PATH'], det_size=(640, 640), det_thresh=0.35)
        else:
            from face_detection_dlib import FaceDetector
            detector = FaceDetector(shape_model_path=CONFIG['DLIB_MODEL_PATH'])
    except ImportError as e:
        print(f'Face detector failed: {e}. Falling back to dlib.')
        from face_detection_dlib import FaceDetector
        detector = FaceDetector(shape_model_path=CONFIG['DLIB_MODEL_PATH'])

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

    print(f'Saved occlusion samples to {args.out}/')


if __name__ == '__main__':
    main()
