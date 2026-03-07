"""
occlusion_augmentation.py
=========================
Clip-consistent and transient synthetic occlusion for training.
Extends synthetic_occlusion.py with regime assignment and temporal consistency.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from config import AUGMENTATION_CONFIG, SEED
from synthetic_occlusion import apply_eye_band, apply_mouth_rect


# Regime names
OCCLUSION_REGIMES = [
    'clean',
    'persistent_eye',
    'persistent_mouth',
    'persistent_both',
    'transient_eye',
    'transient_mouth',
]


def _sample_opacity(opacity_band: str, rng: np.random.Generator) -> float:
    """Sample opacity from band."""
    bands = AUGMENTATION_CONFIG.get('OPACITY_BANDS', {
        'hard': 1.0,
        'medium': (0.7, 0.9),
        'light': (0.4, 0.7),
    })
    b = bands.get(opacity_band, 0.8)
    if isinstance(b, (int, float)):
        return float(b)
    return float(rng.uniform(b[0], b[1]))


def assign_regime_to_clip(
    clip_info,
    regime_weights: Optional[Dict[str, float]] = None,
    class_caps: Optional[Dict[str, float]] = None,
    seed: int = SEED,
) -> Tuple[str, float]:
    """
    Assign occlusion regime to a clip with optional class-aware caps.

    Parameters
    ----------
    clip_info : ClipInfo with majority_class
    regime_weights : dict regime -> weight (default from AUGMENTATION_CONFIG)
    class_caps : dict for safety caps (e.g. yawn_persistent_mouth: 0.3)
    seed : for reproducibility

    Returns
    -------
    (regime, opacity)
    """
    if regime_weights is None:
        regime_weights = AUGMENTATION_CONFIG.get('REGIME_WEIGHTS', {
            'clean': 0.55,
            'persistent_eye': 0.15,
            'persistent_mouth': 0.15,
            'persistent_both': 0.05,
            'transient_eye': 0.05,
            'transient_mouth': 0.05,
        })
    if class_caps is None:
        class_caps = AUGMENTATION_CONFIG.get('CLASS_CAPS', {})

    rng = np.random.default_rng(seed)
    h = hash((clip_info.video_key, clip_info.clip_start, clip_info.T, seed)) % (2**32)
    rng = np.random.default_rng(int(h))

    regimes = list(regime_weights.keys())
    weights = [regime_weights[r] for r in regimes]

    regime = str(rng.choice(regimes, p=np.array(weights) / sum(weights)))

    if regime == 'clean':
        return 'clean', 0.0

    majority = clip_info.majority_class

    if majority == 'Yawn' and regime == 'persistent_mouth':
        cap = class_caps.get('yawn_persistent_mouth', 1.0)
        if rng.random() > cap:
            regime = 'clean'
            return 'clean', 0.0

    if majority == 'EyeClosed' and regime == 'persistent_eye':
        cap = class_caps.get('eyeclosed_persistent_eye', 1.0)
        if rng.random() > cap:
            regime = 'clean'
            return 'clean', 0.0

    if regime == 'persistent_both' and majority in ('Yawn', 'EyeClosed'):
        cap = class_caps.get('positive_persistent_both', 1.0)
        if rng.random() > cap:
            regime = 'clean'
            return 'clean', 0.0

    opacity_band = 'hard' if rng.random() < 0.5 else 'medium'
    opacity = _sample_opacity(opacity_band, rng)
    return regime, opacity


def get_transient_segment(
    clip_len: int,
    clip_seed: int,
    seg_len: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Get (seg_start, seg_end) for transient occlusion subsegment.
    Deterministic given clip_seed.
    """
    if seg_len is None:
        seg_len = max(4, clip_len // 4)
    seg_len = min(seg_len, clip_len)
    rng = np.random.default_rng(clip_seed)
    max_start = clip_len - seg_len
    seg_start = int(rng.integers(0, max(1, max_start + 1)))
    return seg_start, seg_start + seg_len


def is_frame_in_transient_segment(
    frame_idx_in_clip: int,
    clip_len: int,
    seg_len: Optional[int] = None,
    seg_start: Optional[int] = None,
    clip_seed: int = 0,
) -> bool:
    """
    Determine if frame is in the transient occlusion subsegment.

    Parameters
    ----------
    frame_idx_in_clip : 0..clip_len-1
    clip_len : T
    seg_len : length of transient segment (default T//4)
    seg_start : start index (if None, computed from clip_seed)
    clip_seed : for deterministic segment boundaries (e.g. hash(video_key, clip_start))

    Returns
    -------
    True if frame should be occluded
    """
    if seg_start is None:
        seg_start, seg_end = get_transient_segment(clip_len, clip_seed, seg_len)
    else:
        if seg_len is None:
            seg_len = max(4, clip_len // 4)
        seg_end = seg_start + min(seg_len, clip_len - seg_start)
    return seg_start <= frame_idx_in_clip < seg_end


def apply_occlusion_to_frame(
    image: np.ndarray,
    landmarks: np.ndarray,
    regime: str,
    opacity: float,
    frame_idx_in_clip: int,
    clip_len: int,
    clip_seed: int = 0,
) -> np.ndarray:
    """
    Apply occlusion to a single frame based on regime and position in clip.

    - For persistent_*: apply to all frames
    - For transient_*: apply only if frame in transient subsegment

    Parameters
    ----------
    image : HxWx3 BGR
    landmarks : (68, 2)
    regime : str
    opacity : float in [0, 1]
    frame_idx_in_clip : 0..clip_len-1
    clip_len : T
    clip_seed : hash(video_key, clip_start, seed) for deterministic transient segment

    Returns
    -------
    Augmented image (copy)
    """
    if regime == 'clean' or opacity <= 0:
        return image.copy()

    if regime.startswith('transient'):
        in_seg = is_frame_in_transient_segment(
            frame_idx_in_clip, clip_len,
            seg_len=clip_len // 4, seg_start=None, clip_seed=clip_seed,
        )
        if not in_seg:
            return image.copy()

    result = image.copy()
    if regime in ('persistent_eye', 'transient_eye'):
        result = apply_eye_band(result, landmarks, opacity=opacity)
    elif regime in ('persistent_mouth', 'transient_mouth'):
        result = apply_mouth_rect(result, landmarks, opacity=opacity)
    elif regime == 'persistent_both':
        result = apply_eye_band(result, landmarks, opacity=opacity)
        result = apply_mouth_rect(result, landmarks, opacity=opacity)
    return result
