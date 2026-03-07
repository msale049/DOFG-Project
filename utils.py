"""
utils.py
========
Shared pure-utility functions used across the DOFG-DMS pipeline.

Categories
----------
- BBox utilities  : bbox_from_points, expand_bbox, pad_mouth_xyxy,
                    xyxy_to_xywh, rect_to_tuple
- Landmark helpers: approx_yaw_from_landmarks
- Image helpers   : _safe_crop, _to_pil_rgb, _eval_transform
- Tensor helpers  : move_batch_to_device
"""

import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# ─── ImageNet normalisation constants ─────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─── BBox Utilities ───────────────────────────────────────────────────────────

def bbox_from_points(pts: np.ndarray) -> np.ndarray:
    """Return (x0, y0, x1, y1) tight bounding box around *pts* (N×2)."""
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def expand_bbox(b: np.ndarray, scale: float = 1.25, bias: Tuple[float, float] = (0.0, 0.0),
                img_wh: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Expand a (x0, y0, x1, y1) bbox by *scale* around its centre."""
    x0, y0, x1, y1 = b.astype(np.float32)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    w, h = x1 - x0, y1 - y0
    cx += bias[0] * w
    cy += bias[1] * h
    w2, h2 = w * scale / 2.0, h * scale / 2.0
    out = np.array([cx - w2, cy - h2, cx + w2, cy + h2], dtype=np.float32)
    if img_wh is not None:
        W, H = img_wh
        out[0::2] = np.clip(out[0::2], 0, W - 1)
        out[1::2] = np.clip(out[1::2], 0, H - 1)
    return out


def pad_mouth_xyxy(x0: int, y0: int, x1: int, y1: int,
                   pad_x: int = 15, pad_y: int = 10,
                   img_wh: Optional[Tuple[int, int]] = None) -> Tuple[int, int, int, int]:
    """Add padding to a mouth xyxy bbox, clipping to image bounds if supplied."""
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    if img_wh is not None:
        W, H = img_wh
        x1 = min(W - 1, x1 + pad_x)
        y1 = min(H - 1, y1 + pad_y)
    else:
        x1 += pad_x
        y1 += pad_y
    return x0, y0, x1, y1


def xyxy_to_xywh(x0: int, y0: int, x1: int, y1: int) -> Tuple[int, int, int, int]:
    """Convert (x0, y0, x1, y1) to (x, y, w, h)."""
    return x0, y0, x1 - x0, y1 - y0


def rect_to_tuple(rect) -> Tuple[int, int, int, int]:
    """Convert a dlib rectangle to (x, y, w, h)."""
    return (int(rect.left()), int(rect.top()),
            int(rect.width()), int(rect.height()))


# ─── Landmark Helpers ─────────────────────────────────────────────────────────

def approx_yaw_from_landmarks(landmarks: np.ndarray) -> float:
    """Estimate yaw angle (degrees) from dlib 68-pt landmarks."""
    le = landmarks[36:42].mean(axis=0)
    re = landmarks[42:48].mean(axis=0)
    nose = landmarks[30]
    dl = np.linalg.norm(nose - le)
    dr = np.linalg.norm(nose - re)
    if (dl + dr) == 0:
        return 0.0
    asym = (dr - dl) / (dl + dr)
    return float(asym * 90.0)


# ─── Image Helpers ────────────────────────────────────────────────────────────

def _safe_crop(img_np: np.ndarray, bbox, margin: float = 0.15) -> np.ndarray:
    """Crop *img_np* around *bbox* (dict or (x,y,w,h)) with a fractional margin."""
    if isinstance(bbox, dict):
        x = bbox.get('x', bbox.get('left', 0))
        y = bbox.get('y', bbox.get('top', 0))
        w = bbox.get('w', bbox.get('width', 0))
        h = bbox.get('h', bbox.get('height', 0))
    else:
        x, y, w, h = bbox

    H, W = img_np.shape[:2]
    x1 = int(max(0, math.floor(x - margin * w)))
    y1 = int(max(0, math.floor(y - margin * h)))
    x2 = int(min(W, math.ceil(x + w + margin * w)))
    y2 = int(min(H, math.ceil(y + h + margin * h)))
    crop = img_np[y1:y2, x1:x2]
    if crop.size == 0:
        crop = img_np
    return crop


def _to_pil_rgb(img_np: np.ndarray, bgr: bool = True) -> Image.Image:
    """Convert a NumPy HxWxC array to a PIL RGB image."""
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    if img_np.shape[2] == 4:
        img_np = img_np[..., :3]
    if bgr:
        img_np = img_np[..., ::-1]
    return Image.fromarray(np.ascontiguousarray(img_np))


def _eval_transform() -> T.Compose:
    """Standard ImageNet eval transform (256→CenterCrop224→Normalize)."""
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ─── Tensor Helpers ───────────────────────────────────────────────────────────

def move_batch_to_device(batch: Dict, device) -> Dict:
    """
    Recursively move all tensors in *batch* to *device*.

    Handles nested dicts, lists, tuples, and scalar numpy numbers.
    Coerces float64 tensors to float32.
    """
    def to_dev(x):
        if isinstance(x, torch.Tensor):
            if x.dtype == torch.float64:
                x = x.float()
            return x.to(device)
        if isinstance(x, dict):
            return {k: to_dev(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(to_dev(v) for v in x)
        if isinstance(x, (float, int, np.number)):
            return torch.tensor(x, dtype=torch.float32, device=device)
        return x

    out = {}
    for k, v in batch.items():
        if k == 'features' and isinstance(v, dict):
            out[k] = {r: to_dev(t) for r, t in v.items()}
        else:
            out[k] = to_dev(v)
    return out
