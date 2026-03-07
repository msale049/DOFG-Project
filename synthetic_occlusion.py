"""
Synthetic Occlusion Utilities
==============================
Functions to apply controlled synthetic occlusions (sunglasses, masks, hand
overlays) to facial images for evaluating the DOFG-DMS gating pipeline.

Occlusion types
---------------
- **eye_band**: A dark band across both eyes (simulates sunglasses).
- **mouth_rect**: A light-coloured rectangle over the lower face (simulates
  a surgical mask or hand-over-mouth).

Each function accepts an *opacity* parameter in [0, 1]:
    0.0 → no visible occlusion  |  1.0 → fully opaque overlay.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List


# ─── Low-level overlay helpers ────────────────────────────────────────────────

def _blend_overlay(image: np.ndarray, overlay: np.ndarray,
                   mask: np.ndarray, opacity: float) -> np.ndarray:
    """Alpha-blend *overlay* onto *image* using *mask* and *opacity*.

    Parameters
    ----------
    image : HxWx3 uint8
    overlay : HxWx3 uint8  (same shape as *image*)
    mask : HxW float32 in [0, 1]  (per-pixel weight for the overlay)
    opacity : float in [0, 1]  (global opacity multiplier)
    """
    alpha = (mask * opacity)[..., np.newaxis].astype(np.float32)
    blended = image.astype(np.float32) * (1 - alpha) + overlay.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _bbox_from_landmark_indices(landmarks: np.ndarray,
                                indices: List[int],
                                pad_x: int = 10,
                                pad_y: int = 8,
                                img_shape: Optional[Tuple[int, int]] = None
                                ) -> Tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) bounding box around selected landmarks."""
    pts = landmarks[indices]
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    x1 = int(x1 - pad_x)
    y1 = int(y1 - pad_y)
    x2 = int(x2 + pad_x)
    y2 = int(y2 + pad_y)
    if img_shape is not None:
        H, W = img_shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
    return x1, y1, x2, y2


# ─── Eye band (sunglasses) ───────────────────────────────────────────────────

def apply_eye_band(image: np.ndarray,
                   landmarks: np.ndarray,
                   opacity: float = 0.8,
                   color: Tuple[int, int, int] = (30, 30, 30),
                   vertical_expand: float = 1.6) -> np.ndarray:
    """Place sunglasses-style occlusion across both eyes using elliptical
    lenses connected by a nose bridge.

    Uses per-eye ellipses rather than a flat rectangle to produce a shape
    closer to real sunglasses, which keeps the occlusion estimator's
    response monotonic across the full opacity range.

    Parameters
    ----------
    image : HxWx3 uint8.
    landmarks : (68, 2) int array from dlib.
    opacity : 0.0–1.0.
    color : RGB tuple for the lenses (default dark grey).
    vertical_expand : vertical growth factor for lens ellipses.
    """
    if opacity <= 0:
        return image.copy()

    H, W = image.shape[:2]
    overlay = image.copy()
    mask = np.zeros((H, W), dtype=np.float32)

    left_eye_pts = landmarks[36:42]
    right_eye_pts = landmarks[42:48]

    for eye_pts in [left_eye_pts, right_eye_pts]:
        cx = int(eye_pts[:, 0].mean())
        cy = int(eye_pts[:, 1].mean())
        half_w = int((eye_pts[:, 0].max() - eye_pts[:, 0].min()) * 0.85 + 10)
        half_h = int((eye_pts[:, 1].max() - eye_pts[:, 1].min()) * vertical_expand + 6)
        half_h = max(half_h, int(half_w * 0.55))

        cv2.ellipse(overlay, (cx, cy), (half_w, half_h), 0, 0, 360, color, -1)
        cv2.ellipse(mask, (cx, cy), (half_w, half_h), 0, 0, 360, 1.0, -1)

    # Nose bridge connecting the two lenses
    bridge_left_x = int(left_eye_pts[:, 0].max())
    bridge_right_x = int(right_eye_pts[:, 0].min())
    bridge_y = int((landmarks[39, 1] + landmarks[42, 1]) / 2)
    bridge_half_h = max(3, int((left_eye_pts[:, 1].max() - left_eye_pts[:, 1].min()) * 0.35))
    cv2.rectangle(overlay,
                  (bridge_left_x, bridge_y - bridge_half_h),
                  (bridge_right_x, bridge_y + bridge_half_h),
                  color, -1)
    mask[bridge_y - bridge_half_h:bridge_y + bridge_half_h,
         bridge_left_x:bridge_right_x] = 1.0

    kernel_size = max(3, int(min(W, H) * 0.02) | 1)
    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    return _blend_overlay(image, overlay, mask, opacity)


# ─── Mouth rectangle (mask / hand) ───────────────────────────────────────────

def apply_mouth_rect(image: np.ndarray,
                     landmarks: np.ndarray,
                     opacity: float = 0.8,
                     color: Tuple[int, int, int] = (180, 200, 210),
                     vertical_expand: float = 1.8) -> np.ndarray:
    """Place a light-coloured rectangle over the mouth/chin area
    (surgical-mask / hand-over-mouth style occlusion).

    Parameters
    ----------
    image : HxWx3 uint8.
    landmarks : (68, 2) int array from dlib.
    opacity : 0.0–1.0.
    color : RGB tuple for the rectangle (default light blue ≈ surgical mask).
    vertical_expand : vertical growth factor beyond the mouth bbox.
    """
    if opacity <= 0:
        return image.copy()

    mouth_indices = list(range(48, 68))
    chin_indices = list(range(2, 15))
    all_indices = mouth_indices + chin_indices
    x1, y1, x2, y2 = _bbox_from_landmark_indices(
        landmarks, all_indices, pad_x=12, pad_y=8, img_shape=image.shape)

    y_top_mouth = int(landmarks[48:68, 1].min())
    y1 = max(0, y_top_mouth - 10)

    overlay = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.float32)

    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    mask[y1:y2, x1:x2] = 1.0

    kernel_size = max(3, int(min(x2 - x1, y2 - y1) * 0.15) | 1)
    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    return _blend_overlay(image, overlay, mask, opacity)


# ─── Combined / convenience ──────────────────────────────────────────────────

OCCLUSION_TYPES = {
    'none':  {'eye': 0.0, 'mouth': 0.0},
    'eye_only': {'eye': 0.8, 'mouth': 0.0},
    'mouth_only': {'eye': 0.0, 'mouth': 0.8},
    'both': {'eye': 0.8, 'mouth': 0.8},
}

OPACITY_LEVELS = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]


def apply_synthetic_occlusion(image: np.ndarray,
                              landmarks: np.ndarray,
                              eye_opacity: float = 0.0,
                              mouth_opacity: float = 0.0,
                              eye_color: Tuple[int, int, int] = (30, 30, 30),
                              mouth_color: Tuple[int, int, int] = (180, 200, 210),
                              ) -> np.ndarray:
    """Apply both eye and mouth synthetic occlusions in one call.

    Returns the augmented image (HxWx3 uint8).
    """
    result = image.copy()
    if eye_opacity > 0:
        result = apply_eye_band(result, landmarks, opacity=eye_opacity,
                                color=eye_color)
    if mouth_opacity > 0:
        result = apply_mouth_rect(result, landmarks, opacity=mouth_opacity,
                                  color=mouth_color)
    return result
