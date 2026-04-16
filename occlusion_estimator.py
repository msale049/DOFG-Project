"""
occlusion_estimator.py
======================
ROF-trained ResNet-34 occlusion estimator, providing:
  - P(eyes occluded)
  - P(mouth occluded)

Classes
-------
ResNet34OcclusionModel     — Inference-only wrapper around the 2-logit checkpoint.
TrainedOcclusionDetector   — Pipeline-compatible phase-3 wrapper.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as M

from utils import _safe_crop, _to_pil_rgb, _eval_transform


# ─── Inference-only Occlusion Model ──────────────────────────────────────────

class ResNet34OcclusionModel:
    """
    Inference-only wrapper around a ResNet-34 trained to output
    [P(eyes occluded), P(mouth occluded)].

    The model weights are **frozen** — this estimator is never retrained
    inside the DOFG experiments.
    """

    def __init__(self, ckpt_path: str, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # Use weights=None because the checkpoint already contains the trained
        # backbone and the cluster environment may not allow network downloads.
        self.model = M.resnet34(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
        # Load to CPU first to avoid GPU OOM when RetinaFace/ONNX already used GPU
        state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False  # inference only
        self.model.to(self.device)
        self.tf = _eval_transform()

    @torch.no_grad()
    def predict_probs(self, image_np: np.ndarray, face_bbox=None,
                      image_bgr: bool = True,
                      face_margin: float = 0.15) -> np.ndarray:
        """
        Return np.array([p_eyes, p_mouth]) in [0, 1].

        Parameters
        ----------
        image_np : HxWxC NumPy array.
        face_bbox : (x, y, w, h) dict or tuple, or None for full image.
        image_bgr : whether image_np is BGR (True) or RGB (False).
        face_margin : fractional margin to expand the face crop.
        """
        crop = (_safe_crop(image_np, face_bbox, margin=face_margin)
                if face_bbox is not None else image_np)
        pil = _to_pil_rgb(crop, bgr=image_bgr)
        x = self.tf(pil).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        return probs


# ─── Pipeline-compatible Phase-3 Wrapper ─────────────────────────────────────

class TrainedOcclusionDetector:
    """
    High-level wrapper that matches the pipeline's phase-3 interface.

    Wraps ``ResNet34OcclusionModel`` and returns a dict that the downstream
    pipeline expects from ``analyze_occlusion_and_states``.
    """

    def __init__(self, ckpt_path: str, device=None,
                 image_bgr: bool = True, face_margin: float = 0.15,
                 thr_eyes: Optional[float] = None,
                 thr_mouth: Optional[float] = None):
        self.model = ResNet34OcclusionModel(ckpt_path, device=device)
        self.image_bgr = image_bgr
        self.face_margin = face_margin
        self.thr_eyes = thr_eyes
        self.thr_mouth = thr_mouth

    @torch.no_grad()
    def analyze_occlusion_and_states(self, image_np, landmarks, eye_regions,
                                     mouth_region, face_bbox,
                                     feature_result) -> dict:
        """
        Run the occlusion estimator on one frame.

        Returns
        -------
        dict with 'is_valid' (bool) and 'occlusion_analysis' sub-dict containing
        'eye_occlusion_prob' and 'mouth_occlusion_prob' (floats in [0,1]).
        Optionally also 'eyes_occluded'/'mouth_occluded' booleans if thresholds
        were provided at init time.
        """
        try:
            p_eyes, p_mouth = self.model.predict_probs(
                image_np, face_bbox=face_bbox,
                image_bgr=self.image_bgr, face_margin=self.face_margin,
            )
            result: dict = {
                'is_valid': True,
                'occlusion_analysis': {
                    'eye_occlusion_prob': float(p_eyes),
                    'mouth_occlusion_prob': float(p_mouth),
                },
            }
            if self.thr_eyes is not None and self.thr_mouth is not None:
                result['occlusion_analysis'].update({
                    'eyes_occluded': bool(p_eyes >= self.thr_eyes),
                    'mouth_occluded': bool(p_mouth >= self.thr_mouth),
                })
            return result
        except Exception as e:
            return {'is_valid': False, 'error': str(e)}
