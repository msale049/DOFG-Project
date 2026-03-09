"""
config.py
=========
Central configuration, constants, and data-structure definitions for the
DOFG-DMS pipeline.  All other modules import from here.
"""

import random
import numpy as np
from dataclasses import dataclass, field

# ─── Global seed ─────────────────────────────────────────────────────────────

SEED = 42

def seed_everything(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch (if available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ─── Project-level configuration ─────────────────────────────────────────────

CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────────
    'CSV_DATASET_PATH': 'Data',
    'RESNET34_MODEL_PATH': 'models/resnet34_portable.state_dict.pt',
    'RESNET34_OCCLUSION_MODEL_PATH': 'models/resnet34_occlusion.pt',
    'DLIB_MODEL_PATH': 'models/shape_predictor_68_face_landmarks.dat',
    'MMOD_MODEL_PATH': 'models/mmod_human_face_detector.dat',

    # ── Image processing ────────────────────────────────────────────────────
    'IMG_SIZE': (224, 224),
    'TARGET_FACE_SIZE': (224, 224),
    'IMAGENET_MEAN': [0.485, 0.456, 0.406],
    'IMAGENET_STD':  [0.229, 0.224, 0.225],

    # ── Thresholds ──────────────────────────────────────────────────────────
    'EAR_THRESHOLD': 0.25,       # Eye Aspect Ratio → closed eyes
    'MAR_THRESHOLD': 0.65,       # Mouth Aspect Ratio → yawning
    'MIN_FACE_CONFIDENCE': 0.5,

    # ── Batch / data ────────────────────────────────────────────────────────
    'BATCH': 32,

    # ── Label map ────────────────────────────────────────────────────────────
    'CSV_LABELS': {'EyeClosed': 0, 'Yawn': 1, 'Neutral': 2},
    'NUM_CLASSES': 3,
}

# ─── Clip-based sampling (Phase 2 redesign) ───────────────────────────────────

CLIP_CONFIG = {
    'FPS_TARGET': 15,
    'FPS_SOURCE': 29.76,
    'T': 32,
    'T_ABLATION': 16,
    'TRAIN_STRIDE': 16,
    'EVAL_STRIDE': 32,
}

# ─── Occlusion augmentation regime weights ────────────────────────────────────
# clean, persistent_eye, persistent_mouth, persistent_both, transient_eye, transient_mouth

AUGMENTATION_CONFIG = {
    'REGIME_WEIGHTS': {
        'clean': 0.55,
        'persistent_eye': 0.15,
        'persistent_mouth': 0.15,
        'persistent_both': 0.05,
        'transient_eye': 0.05,
        'transient_mouth': 0.05,
    },
    'OPACITY_BANDS': {
        'hard': 1.0,
        'medium': (0.7, 0.9),
        'light': (0.4, 0.7),
    },
    'CLASS_CAPS': {
        'yawn_persistent_mouth': 0.30,
        'eyeclosed_persistent_eye': 0.30,
        'positive_persistent_both': 0.10,
    },
}


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class CSVAnnotation:
    """One row from a DMD subject CSV annotation file."""
    frame: int
    timestamp: float
    class_label: str
    variant: str
    eyes_state: str
    yawn_with_hand: bool
    yawn_without_hand: bool
    eyes_occluded_prior: bool
    mouth_occluded_prior: bool
    glasses: bool


@dataclass
class PhaseResult:
    """Container for the output of one pipeline phase (detection / extraction / …)."""
    phase_name: str
    success: bool
    processing_time: float
    data: dict = field(default_factory=dict)
    confidence: float = 0.0
