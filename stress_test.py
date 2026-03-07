"""
stress_test.py
==============
Synthetic occlusion stress test: per-frame analysis with occlusion probabilities,
gating values, and gating ON vs OFF predictions.

Two modes:
- Legacy: frame-based, OCCLUSION_CONFIGS (none, eye_only, mouth_only, both) × opacities
- Strategy (STRATEGY_DESIGN): clip-based, 6 conditions (clean + 5 stress at 0.8)

Functions
---------
run_with_gating_disabled   — Run model with gates forced to 1.0.
run_stress_test_detailed  — Per-frame DataFrame (legacy).
run_stress_test_clips     — Clip-based with 6 STRATEGY_DESIGN conditions.
run_stress_test           — Wrapper: clips if test_clips else legacy.
"""

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

from synthetic_occlusion import (
    apply_synthetic_occlusion,
    OPACITY_LEVELS,
)
from ablation_utils import disable_gates_at_inference
from data_loading import sample_frames_for_audit
from synthetic_occlusion import apply_eye_band, apply_mouth_rect

# STRATEGY_DESIGN E.2: 6 conditions — clean + 5 stress (opacity 0.8)
# Transient uses middle T/4 of clip
STRESS_CONDITIONS_STRATEGY = [
    ('clean', 'clean', 0.0),
    ('persistent_eye', 'persistent_eye', 0.8),
    ('persistent_mouth', 'persistent_mouth', 0.8),
    ('persistent_both', 'persistent_both', 0.8),
    ('transient_eye', 'transient_eye', 0.8),
    ('transient_mouth', 'transient_mouth', 0.8),
]

# Occlusion configs: (name, eye_flag, mouth_flag) — which region gets the opacity (legacy)
OCCLUSION_CONFIGS = [
    ('none', 0.0, 0.0),
    ('eye_only', 1.0, 0.0),
    ('mouth_only', 0.0, 1.0),
    ('both', 1.0, 1.0),
]


def run_with_gating_disabled(model: torch.nn.Module, features: Dict, occlusion_info: Dict,
                             device: str = 'cpu'):
    """
    Run model with gates forced to 1.0 (no suppression).
    Uses ablation_utils.disable_gates_at_inference.
    """
    with disable_gates_at_inference(model):
        with torch.no_grad():
            # Ensure tensors on device
            fdev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in features.items()}
            oi = {k: (v.to(device) if isinstance(v, torch.Tensor)
                       else torch.tensor([v], device=device).float()
                       if isinstance(v, (int, float)) else v)
                  for k, v in occlusion_info.items()}
            out = model(fdev, oi, return_attention=True)
    return out


def run_stress_test_detailed(
    csv_data: Dict,
    test_keys: List[str],
    model: torch.nn.Module,
    face_detector,
    feat_extractor,
    occ_model,
    device: str = 'cpu',
    opacity_levels: Optional[List[float]] = None,
    max_frames_per_video: Optional[int] = 30,
    seed: int = 42,
    label_map: Optional[Dict[str, int]] = None,
    include_timing: bool = True,
) -> pd.DataFrame:
    """
    Run stress test with per-frame, per-condition detail.

    For each test frame, for each (opacity × occlusion_type):
      - Apply synthetic occlusion
      - Extract features, run occlusion estimator
      - Run transformer (gating ON and OFF)
      - Store: subject, frame, occlusion_type, opacity, p_eye, p_mouth,
        pred_gating_on, pred_gating_off, correct_gating_on, correct_gating_off,
        gate_face, gate_eye, gate_mouth

    Returns
    -------
    DataFrame with one row per (frame, occlusion_type, opacity).
    """
    if label_map is None:
        label_map = {'EyeClosed': 0, 'Yawn': 1, 'Neutral': 2}
    if opacity_levels is None:
        opacity_levels = OPACITY_LEVELS

    test_csv = {k: v for k, v in csv_data.items() if k in test_keys}
    sampled = sample_frames_for_audit(
        test_csv, max_per_subject=max_frames_per_video,
        include_all_occluded=True, seed=seed,
    )
    total_frames = sum(len(v) for v in sampled.values())
    print(f'  Stress-test: {total_frames} frames from {len(sampled)} videos')

    use_cuda = (device == 'cuda')
    if use_cuda:
        print('  Warming up GPU...')
        _w_feat = {r: torch.randn(1, 512, device=device) for r in ['face', 'left_eye', 'right_eye', 'mouth']}
        _w_occ = {'eye_occlusion_prob': torch.tensor([0.0], device=device),
                  'mouth_occlusion_prob': torch.tensor([0.0], device=device)}
        for _ in range(10):
            with torch.no_grad():
                _ = model(_w_feat, _w_occ)
        if use_cuda:
            torch.cuda.synchronize()
        del _w_feat, _w_occ
        print('  Warm-up done.')

    rows = []
    processed, skipped = 0, 0
    t0 = time.time()

    for video_key, ann_list in sampled.items():
        meta = csv_data[video_key]
        cap = cv2.VideoCapture(meta['video_path'])
        if not cap.isOpened():
            continue
        subject = meta['subject']
        print(f'  {video_key} ({len(ann_list)} frames) ...', flush=True)

        for ann in ann_list:
            cap.set(cv2.CAP_PROP_POS_FRAMES, ann.frame)
            ret, bgr = cap.read()
            if not ret:
                continue

            _t = time.perf_counter()
            det = face_detector.detect_face_and_landmarks(bgr)
            t_det = (time.perf_counter() - _t) * 1000

            if not det['is_valid'] or 'landmarks' not in det:
                skipped += 1
                processed += 1
                continue

            lm = det['landmarks']
            fb = det['face_bbox']
            er = det['eye_regions']
            mr = det['mouth_region']
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            if ann.class_label not in label_map:
                processed += 1
                continue
            gt = label_map[ann.class_label]

            for opacity in opacity_levels:
                for occ_name, eflag, mflag in OCCLUSION_CONFIGS:
                    if occ_name == 'none' and opacity > 0:
                        continue
                    if occ_name != 'none' and opacity == 0:
                        continue
                    eo = opacity if eflag else 0.0
                    mo = opacity if mflag else 0.0

                    aug_rgb = (apply_synthetic_occlusion(rgb, lm, eye_opacity=eo, mouth_opacity=mo)
                               if occ_name != 'none' else rgb)
                    aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR) if occ_name != 'none' else bgr

                    if use_cuda:
                        torch.cuda.synchronize()
                    _t = time.perf_counter()
                    feat = feat_extractor.extract_region_features(aug_bgr, fb, er, mr)
                    if use_cuda:
                        torch.cuda.synchronize()
                    t_feat = (time.perf_counter() - _t) * 1000

                    rkeys = ['face_features', 'left_eye_features', 'right_eye_features', 'mouth_features']
                    if any(feat[k] is None for k in rkeys):
                        continue

                    fdict = {
                        'face': torch.tensor(feat['face_features'], dtype=torch.float32).unsqueeze(0).to(device),
                        'left_eye': torch.tensor(feat['left_eye_features'], dtype=torch.float32).unsqueeze(0).to(device),
                        'right_eye': torch.tensor(feat['right_eye_features'], dtype=torch.float32).unsqueeze(0).to(device),
                        'mouth': torch.tensor(feat['mouth_features'], dtype=torch.float32).unsqueeze(0).to(device),
                    }

                    _t = time.perf_counter()
                    probs = occ_model.predict_probs(
                        aug_rgb, face_bbox=fb, image_bgr=False, face_margin=0.15)
                    t_occ = (time.perf_counter() - _t) * 1000

                    occ_info = {
                        'eye_occlusion_prob': torch.tensor([float(probs[0])], device=device, dtype=torch.float32),
                        'mouth_occlusion_prob': torch.tensor([float(probs[1])], device=device, dtype=torch.float32),
                    }

                    if use_cuda:
                        torch.cuda.synchronize()
                    _t = time.perf_counter()
                    model.eval()
                    with torch.no_grad():
                        out_on = model(fdict, occ_info, return_attention=True)
                        out_off = run_with_gating_disabled(model, fdict, occ_info, device=device)
                    if use_cuda:
                        torch.cuda.synchronize()
                    t_trans = (time.perf_counter() - _t) * 1000

                    gates = out_on['gate_factors'].squeeze().cpu().numpy()
                    pred_on = int(out_on['predicted_class'].item())
                    pred_off = int(out_off['predicted_class'].item())

                    row = {
                        'subject': subject,
                        'video_key': video_key,
                        'frame': ann.frame,
                        'class_label': ann.class_label,
                        'gt_label': gt,
                        'occlusion_type': occ_name,
                        'opacity': opacity if occ_name != 'none' else 0.0,
                        'p_eye': float(probs[0]),
                        'p_mouth': float(probs[1]),
                        'pred_gating_on': pred_on,
                        'pred_gating_off': pred_off,
                        'correct_gating_on': int(pred_on == gt),
                        'correct_gating_off': int(pred_off == gt),
                        'gate_face': float(gates[0]),
                        'gate_eye': float(gates[1]),
                        'gate_mouth': float(gates[3]),
                    }
                    if include_timing:
                        row['t_det_ms'] = round(t_det, 3)
                        row['t_feat_ms'] = round(t_feat, 3)
                        row['t_occ_ms'] = round(t_occ, 3)
                        row['t_trans_ms'] = round(t_trans, 3)
                        row['t_total_ms'] = round(t_det + t_feat + t_occ + t_trans, 3)
                    rows.append(row)

            processed += 1
            if processed % 25 == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                print(f'    [{processed}/{total_frames}] {rate:.1f} fr/s', flush=True)

        cap.release()

    print(f'  Done: {processed} frames, {skipped} skipped, {time.time() - t0:.0f}s')
    return pd.DataFrame(rows)


def _apply_stress_condition(
    rgb: np.ndarray,
    landmarks: np.ndarray,
    condition_name: str,
    opacity: float,
    frame_idx_in_clip: int,
    clip_len: int,
) -> np.ndarray:
    """Apply one STRATEGY_DESIGN stress condition to a frame."""
    if condition_name == 'clean' or opacity <= 0:
        return rgb.copy()
    # Transient: middle T/4 of clip (STRATEGY_DESIGN E.2)
    seg_len = max(4, clip_len // 4)
    seg_start = (clip_len - seg_len) // 2
    if condition_name.startswith('transient'):
        in_seg = seg_start <= frame_idx_in_clip < seg_start + seg_len
        if not in_seg:
            return rgb.copy()
    # Apply occlusion (persistent or transient-in-segment)
    result = rgb.copy()
    if condition_name in ('persistent_eye', 'transient_eye'):
        result = apply_eye_band(result, landmarks, opacity=opacity)
    elif condition_name in ('persistent_mouth', 'transient_mouth'):
        result = apply_mouth_rect(result, landmarks, opacity=opacity)
    elif condition_name == 'persistent_both':
        result = apply_eye_band(result, landmarks, opacity=opacity)
        result = apply_mouth_rect(result, landmarks, opacity=opacity)
    return result


def run_stress_test_clips(
    csv_data: Dict,
    test_clips: List,
    model: torch.nn.Module,
    face_detector,
    feat_extractor,
    occ_model,
    device: str = 'cpu',
    max_frames_per_clip: Optional[int] = 8,
    seed: int = 42,
    label_map: Optional[Dict[str, int]] = None,
    include_timing: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clip-based stress test with 6 STRATEGY_DESIGN conditions.

    Each test clip is evaluated in: clean + persistent_eye/mouth/both + transient_eye/mouth (opacity 0.8).
    Returns (details_df, summary_df) with p_eye, p_mouth, gates, predictions per frame/condition.
    """
    if label_map is None:
        label_map = {'EyeClosed': 0, 'Yawn': 1, 'Neutral': 2}

    if not test_clips:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    processed, skipped = 0, 0
    t0 = time.time()
    use_cuda = (device == 'cuda')

    for clip in test_clips:
        meta = csv_data.get(clip.video_key)
        if not meta:
            continue
        cap = cv2.VideoCapture(meta['video_path'])
        if not cap.isOpened():
            continue

        ann_by_frame = {a.frame: a for a in meta['annotations']}
        # Subsample frames for speed
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

            _t = time.perf_counter()
            det = face_detector.detect_face_and_landmarks(bgr)
            t_det = (time.perf_counter() - _t) * 1000
            if not det['is_valid'] or det.get('landmarks') is None:
                skipped += 1
                continue

            lm = det['landmarks']
            fb = det['face_bbox']
            er = det['eye_regions']
            mr = det['mouth_region']
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gt = label_map[ann.class_label]

            for cond_name, regime, opacity in STRESS_CONDITIONS_STRATEGY:
                aug_rgb = _apply_stress_condition(
                    rgb, lm, regime, opacity, fi, clip.T,
                )
                aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)

                _t = time.perf_counter()
                feat = feat_extractor.extract_region_features(aug_bgr, fb, er, mr)
                t_feat = (time.perf_counter() - _t) * 1000

                rkeys = ['face_features', 'left_eye_features', 'right_eye_features', 'mouth_features']
                if any(feat[k] is None for k in rkeys):
                    continue

                fdict = {
                    'face': torch.tensor(feat['face_features'], dtype=torch.float32).unsqueeze(0).to(device),
                    'left_eye': torch.tensor(feat['left_eye_features'], dtype=torch.float32).unsqueeze(0).to(device),
                    'right_eye': torch.tensor(feat['right_eye_features'], dtype=torch.float32).unsqueeze(0).to(device),
                    'mouth': torch.tensor(feat['mouth_features'], dtype=torch.float32).unsqueeze(0).to(device),
                }
                probs = occ_model.predict_probs(
                    aug_rgb, face_bbox=fb, image_bgr=False, face_margin=0.15)
                t_occ = 0.0
                occ_info = {
                    'eye_occlusion_prob': torch.tensor([float(probs[0])], device=device, dtype=torch.float32),
                    'mouth_occlusion_prob': torch.tensor([float(probs[1])], device=device, dtype=torch.float32),
                }

                _t = time.perf_counter()
                model.eval()
                with torch.no_grad():
                    out_on = model(fdict, occ_info, return_attention=True)
                    out_off = run_with_gating_disabled(model, fdict, occ_info, device=device)
                t_trans = (time.perf_counter() - _t) * 1000

                gates = out_on['gate_factors'].squeeze().cpu().numpy()
                pred_on = int(out_on['predicted_class'].item())
                pred_off = int(out_off['predicted_class'].item())

                row = {
                    'subject': clip.subject,
                    'video_key': clip.video_key,
                    'clip_start': clip.clip_start,
                    'frame': frame_num,
                    'class_label': ann.class_label,
                    'gt_label': gt,
                    'condition': cond_name,
                    'opacity': opacity,
                    'p_eye': float(probs[0]),
                    'p_mouth': float(probs[1]),
                    'pred_gating_on': pred_on,
                    'pred_gating_off': pred_off,
                    'correct_gating_on': int(pred_on == gt),
                    'correct_gating_off': int(pred_off == gt),
                    'gate_face': float(gates[0]),
                    'gate_eye': float(gates[1]),
                    'gate_mouth': float(gates[3]),
                }
                if include_timing:
                    row['t_det_ms'] = round(t_det, 3)
                    row['t_feat_ms'] = round(t_feat, 3)
                    row['t_trans_ms'] = round(t_trans, 3)
                rows.append(row)

            processed += 1

        cap.release()

    details = pd.DataFrame(rows)
    if len(details) == 0:
        return details, pd.DataFrame()

    # Summary: accuracy by condition
    summary_rows = []
    for cond_name, _, opacity in STRESS_CONDITIONS_STRATEGY:
        sub = details[details['condition'] == cond_name]
        if len(sub) > 0:
            summary_rows.append({
                'condition': cond_name,
                'opacity': opacity,
                'acc_gating_on': sub['correct_gating_on'].mean() * 100,
                'acc_gating_off': sub['correct_gating_off'].mean() * 100,
                'delta_pp': sub['correct_gating_on'].mean() * 100 - sub['correct_gating_off'].mean() * 100,
                'n': len(sub),
            })
    summary = pd.DataFrame(summary_rows)
    print(f'  Stress-test (clips): {processed} frames, {len(details)} rows, {time.time()-t0:.0f}s')
    return details, summary


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-frame results to accuracy by (occlusion_type, opacity)."""
    summary_rows = []
    b = df[df['occlusion_type'] == 'none']
    if len(b) > 0:
        bl_on = b['correct_gating_on'].mean() * 100
        bl_off = b['correct_gating_off'].mean() * 100
        summary_rows.append({
            'occlusion_type': 'none', 'opacity': 0.0,
            'acc_gating_on': bl_on, 'acc_gating_off': bl_off,
            'delta_pp': bl_on - bl_off, 'n': len(b),
        })

    for ot in ['eye_only', 'mouth_only', 'both']:
        sub = df[df['occlusion_type'] == ot]
        for op in sorted(sub['opacity'].unique()):
            ss = sub[sub['opacity'] == op]
            a_on = ss['correct_gating_on'].mean() * 100
            a_off = ss['correct_gating_off'].mean() * 100
            summary_rows.append({
                'occlusion_type': ot, 'opacity': op,
                'acc_gating_on': a_on, 'acc_gating_off': a_off,
                'delta_pp': a_on - a_off, 'n': len(ss),
            })

    return pd.DataFrame(summary_rows)


def run_latency_benchmark(
    model: torch.nn.Module,
    face_detector,
    feat_extractor,
    occ_model,
    csv_data: Dict,
    test_keys: List[str],
    device: str = 'cpu',
    num_warmup: int = 20,
    num_iter: int = 50,
) -> Dict:
    """
    Run latency benchmark with GPU warm-up for paper reporting.

    Returns dict with ms per phase: face_detection, feature_extraction,
    occlusion_estimator, transformer_inference, total_per_frame.
    """
    import time
    test_keys = [k for k in test_keys if k in csv_data]
    if not test_keys:
        return {}
    meta = csv_data[test_keys[0]]
    cap = cv2.VideoCapture(meta['video_path'])
    if not cap.isOpened():
        return {}
    anns = meta['annotations']
    frame_num = anns[len(anns) // 2].frame if anns else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, bgr = cap.read()
    cap.release()
    if not ret:
        return {}

    use_cuda = (device == 'cuda')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Warm-up
    for _ in range(num_warmup):
        det = face_detector.detect_face_and_landmarks(bgr)
        if det['is_valid']:
            feat = feat_extractor.extract_region_features(
                bgr, det['face_bbox'], det['eye_regions'], det['mouth_region'])
            probs = occ_model.predict_probs(
                rgb, face_bbox=det['face_bbox'], image_bgr=False, face_margin=0.15)
            fdict = {
                'face': torch.tensor(feat['face_features'], dtype=torch.float32).unsqueeze(0).to(device),
                'left_eye': torch.tensor(feat['left_eye_features'], dtype=torch.float32).unsqueeze(0).to(device),
                'right_eye': torch.tensor(feat['right_eye_features'], dtype=torch.float32).unsqueeze(0).to(device),
                'mouth': torch.tensor(feat['mouth_features'], dtype=torch.float32).unsqueeze(0).to(device),
            }
            occ_info = {
                'eye_occlusion_prob': torch.tensor([float(probs[0])], device=device, dtype=torch.float32),
                'mouth_occlusion_prob': torch.tensor([float(probs[1])], device=device, dtype=torch.float32),
            }
            with torch.no_grad():
                _ = model(fdict, occ_info, return_attention=True)
    if use_cuda:
        torch.cuda.synchronize()

    # Benchmark
    t_det, t_feat, t_occ, t_trans = [], [], [], []
    for _ in range(num_iter):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        det = face_detector.detect_face_and_landmarks(bgr)
        if use_cuda:
            torch.cuda.synchronize()
        t_det.append((time.perf_counter() - t0) * 1000)

        if not det['is_valid']:
            continue
        t0 = time.perf_counter()
        feat = feat_extractor.extract_region_features(
            bgr, det['face_bbox'], det['eye_regions'], det['mouth_region'])
        if use_cuda:
            torch.cuda.synchronize()
        t_feat.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        probs = occ_model.predict_probs(
            rgb, face_bbox=det['face_bbox'], image_bgr=False, face_margin=0.15)
        if use_cuda:
            torch.cuda.synchronize()
        t_occ.append((time.perf_counter() - t0) * 1000)

        fdict = {
            'face': torch.tensor(feat['face_features'], dtype=torch.float32).unsqueeze(0).to(device),
            'left_eye': torch.tensor(feat['left_eye_features'], dtype=torch.float32).unsqueeze(0).to(device),
            'right_eye': torch.tensor(feat['right_eye_features'], dtype=torch.float32).unsqueeze(0).to(device),
            'mouth': torch.tensor(feat['mouth_features'], dtype=torch.float32).unsqueeze(0).to(device),
        }
        occ_info = {
            'eye_occlusion_prob': torch.tensor([float(probs[0])], device=device, dtype=torch.float32),
            'mouth_occlusion_prob': torch.tensor([float(probs[1])], device=device, dtype=torch.float32),
        }
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(fdict, occ_info, return_attention=True)
        if use_cuda:
            torch.cuda.synchronize()
        t_trans.append((time.perf_counter() - t0) * 1000)

    def _stats(x):
        return {'mean_ms': float(np.mean(x)), 'std_ms': float(np.std(x)), 'median_ms': float(np.median(x))} if x else {}

    total = [t_det[i] + t_feat[i] + t_occ[i] + t_trans[i] for i in range(min(len(t_det), len(t_feat), len(t_occ), len(t_trans)))]
    return {
        'face_detection_ms': _stats(t_det),
        'feature_extraction_ms': _stats(t_feat),
        'occlusion_estimator_ms': _stats(t_occ),
        'transformer_inference_ms': _stats(t_trans),
        'total_per_frame_ms': _stats(total),
        'num_warmup': num_warmup,
        'num_iter': num_iter,
        'device': device,
    }


def run_stress_test(
    csv_data: Dict,
    test_keys: List[str],
    model: torch.nn.Module,
    face_detector,
    feat_extractor,
    occ_model,
    trainer,  # unused; kept for API compatibility
    device: str = 'cpu',
    opacity_levels: Optional[List[float]] = None,
    max_frames_per_video: Optional[int] = 30,
    batch_size: int = 16,  # unused
    seed: int = 42,
    test_clips: Optional[List] = None,
    max_frames_per_clip: Optional[int] = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run full stress test: per-frame details + summary.

    If test_clips is provided (clip strategy), uses 6 STRATEGY_DESIGN conditions.
    Otherwise uses legacy frame-based test with occlusion_type × opacity.

    Returns
    -------
    (details_df, summary_df)
    - details_df: per-frame, per-condition with p_eye, p_mouth, gates, etc.
    - summary_df: aggregated accuracy by condition.
    """
    if test_clips:
        return run_stress_test_clips(
            csv_data, test_clips, model, face_detector, feat_extractor, occ_model,
            device=device,
            max_frames_per_clip=max_frames_per_clip,
            seed=seed,
            include_timing=True,
        )
    details = run_stress_test_detailed(
        csv_data, test_keys, model, face_detector, feat_extractor, occ_model,
        device=device,
        opacity_levels=opacity_levels,
        max_frames_per_video=max_frames_per_video,
        seed=seed,
        include_timing=True,
    )
    if len(details) == 0:
        return details, pd.DataFrame()

    summary = _build_summary(details)
    return details, summary
