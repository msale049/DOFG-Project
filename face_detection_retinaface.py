"""
face_detection_retinaface.py
============================
Single-stage RetinaFace (InsightFace buffalo_sc) + dlib 68-point landmark
detector — the **journal-version** face detector for the DOFG-DMS pipeline.

Compared with the two-stage dlib HOG+CNN detector this version is:
  - Faster (single-stage deep network)
  - More robust under varied poses and lighting
  - Produces a detection confidence score

Usage
-----
    from face_detection_retinaface import FaceDetector as RetinaFaceDetector
    detector = RetinaFaceDetector(shape_model_path='shape_predictor_68_face_landmarks.dat')
    result = detector.detect_face_and_landmarks(bgr_frame)
"""

import os
import time
from typing import Dict, Optional, Tuple

os.environ.setdefault('ORT_LOG_LEVEL', '3')
os.environ.setdefault('ONNXRUNTIME_SESSION_THREAD_POOL_SIZE', '1')

import cv2
import dlib
import numpy as np
import torch

try:
    import onnxruntime
    onnxruntime.set_default_logger_severity(3)
except Exception:
    pass

from insightface.app import FaceAnalysis

from utils import bbox_from_points, approx_yaw_from_landmarks


class FaceDetector:
    """
    RetinaFace (InsightFace buffalo_sc model) + dlib 68-pt shape predictor.

    Parameters
    ----------
    shape_model_path : str
        Path to dlib ``shape_predictor_68_face_landmarks.dat``.
    det_size : tuple of int
        Input resolution for the RetinaFace detector (height, width).
    det_thresh : float
        Minimum detection confidence to accept a face.
    use_pose_filter : bool
        If True, reject detections with |yaw| > yaw_thresh_deg.
    yaw_thresh_deg : float
        Maximum acceptable absolute yaw (degrees).
    """

    def __init__(self,
                 shape_model_path: str,
                 det_size: Tuple[int, int] = (768, 768),
                 det_thresh: float = 0.35,
                 use_pose_filter: bool = False,
                 yaw_thresh_deg: float = 15.0):

        print('Initializing RetinaFace detector...')
        force_cpu = os.environ.get('DOFG_FACE_CPU', '0') in ('1', 'true', 'yes')
        if torch.cuda.is_available() and not force_cpu:
            available = onnxruntime.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                ctx_id = 0
            else:
                print('  WARNING: CUDAExecutionProvider not available, falling back to CPU')
                providers = ['CPUExecutionProvider']
                ctx_id = -1
        else:
            providers = ['CPUExecutionProvider']
            ctx_id = -1

        stderr_fd = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        try:
            self.face_app = FaceAnalysis(name='buffalo_sc', providers=providers)
            self.face_app.prepare(ctx_id=ctx_id, det_size=det_size,
                                  det_thresh=det_thresh)
        finally:
            os.dup2(stderr_fd, 2)
            os.close(stderr_fd)
            os.close(devnull)

        self.predictor = dlib.shape_predictor(shape_model_path)
        self.use_pose_filter = bool(use_pose_filter)
        self.yaw_thresh_deg = float(yaw_thresh_deg)
        ep_used = providers[0].replace('ExecutionProvider', '')
        print(f'  RetinaFace (buffalo_sc) + dlib 68pt ready  [thresh={det_thresh}, backend={ep_used}]')

    # ── Private helpers ────────────────────────────────────────────────────────

    def _predict_landmarks(self, rgb: np.ndarray,
                           rect: dlib.rectangle) -> np.ndarray:
        shape = self.predictor(rgb, rect)
        return np.array([[p.x, p.y] for p in shape.parts()], dtype=np.int32)

    def _rois_from_landmarks(self, landmarks: np.ndarray,
                              img_wh: Tuple[int, int]):
        """Return (left_eye, right_eye, mouth) ROI tuples in (x, y, w, h)."""
        W, H = img_wh

        def eye_box(points, pad: int = 12):
            x0, y0 = points.min(axis=0)
            x1, y1 = points.max(axis=0)
            x0 = max(0, int(x0 - pad))
            y0 = max(0, int(y0 - pad))
            w  = int((x1 - x0) + 2 * pad)
            h  = int((y1 - y0) + 2 * pad)
            return (x0, y0, w, h)

        left_eye  = eye_box(landmarks[36:42], pad=12)
        right_eye = eye_box(landmarks[42:48], pad=12)

        mouth_tight = bbox_from_points(landmarks[48:68])
        x0, y0, x1, y1 = mouth_tight.astype(int)
        pad_x, pad_y = 15, 10
        x0 = max(0, x0 - pad_x)
        y0 = max(0, y0 - pad_y)
        x1 = min(W - 1, x1 + pad_x)
        y1 = min(H - 1, y1 + pad_y)
        mouth = (x0, y0, x1 - x0, y1 - y0)

        return left_eye, right_eye, mouth

    # ── Public interface ───────────────────────────────────────────────────────

    def detect_face_and_landmarks(self, image_bgr: np.ndarray) -> Dict:
        """
        Detect the largest face and predict 68 landmarks using RetinaFace +
        dlib shape predictor.

        Returns
        -------
        dict with keys:
            is_valid, face_bbox (x,y,w,h), landmarks (68×2),
            eye_regions [(l,r) tuples], mouth_region tuple,
            processing_time, detector ('retinaface'),
            detection_score (float, only when is_valid=True)
        """
        t0 = time.time()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        H, W = image_bgr.shape[:2]

        faces = self.face_app.get(image_bgr)
        if len(faces) == 0:
            return {
                'is_valid': False, 'face_bbox': None, 'landmarks': None,
                'eye_regions': None, 'mouth_region': None,
                'processing_time': time.time() - t0, 'detector': 'retinaface',
            }

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = face.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        rect = dlib.rectangle(x1, y1, x2, y2)

        landmarks = self._predict_landmarks(rgb, rect)

        if self.use_pose_filter:
            yaw = approx_yaw_from_landmarks(landmarks)
            if abs(yaw) > self.yaw_thresh_deg:
                return {
                    'is_valid': False, 'face_bbox': None, 'landmarks': None,
                    'eye_regions': None, 'mouth_region': None,
                    'processing_time': time.time() - t0, 'detector': 'retinaface',
                }

        face_bbox = (x1, y1, x2 - x1, y2 - y1)
        eL, eR, mB = self._rois_from_landmarks(landmarks, (W, H))

        return {
            'is_valid': True,
            'face_bbox': face_bbox,
            'landmarks': landmarks,
            'eye_regions': [tuple(eL), tuple(eR)],
            'mouth_region': tuple(mB),
            'processing_time': time.time() - t0,
            'detector': 'retinaface',
            'detection_score': float(face.det_score),
        }
