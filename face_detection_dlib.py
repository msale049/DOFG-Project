"""
face_detection_dlib.py
======================
Two-stage dlib HOG + CNN face detector with 68-point landmark prediction.

This is the **original** detector used in the DOFG conference paper.
For the journal extension (faster, more robust) use face_detection_retinaface.py.

Usage
-----
    from face_detection_dlib import FaceDetector as DlibFaceDetector
    detector = DlibFaceDetector(shape_model_path='shape_predictor_68_face_landmarks.dat',
                                 cnn_model_path='mmod_human_face_detector.dat')
    result = detector.detect_face_and_landmarks(bgr_frame)
"""

import time
from typing import Dict, Optional, Tuple

import cv2
import dlib
import numpy as np

from utils import bbox_from_points, approx_yaw_from_landmarks, rect_to_tuple


class FaceDetector:
    """
    Two-stage face detector: dlib HOG detector first, CNN fallback second.

    Stage 1 — dlib HOG (fast, works well for frontal faces).
    Stage 2 — dlib CNN (slower, more robust to unusual poses/lighting).

    Parameters
    ----------
    shape_model_path : str
        Path to dlib 68-point shape predictor .dat file.
    cnn_model_path : str
        Path to dlib MMOD CNN face detector .dat file.
    hog_upsample : int
        How many times to upsample the image for HOG detection (0 = none).
    cnn_upsample : int
        How many times to upsample the image for CNN detection (0 = none).
    use_pose_filter : bool
        If True, discard detections with |yaw| > yaw_thresh_deg.
    yaw_thresh_deg : float
        Maximum acceptable yaw angle (degrees).
    """

    def __init__(self,
                 shape_model_path: str,
                 cnn_model_path: str = 'mmod_human_face_detector.dat',
                 hog_upsample: int = 0,
                 cnn_upsample: int = 0,
                 use_pose_filter: bool = False,
                 yaw_thresh_deg: float = 15.0):
        self.hog = dlib.get_frontal_face_detector()
        self.cnn = dlib.cnn_face_detection_model_v1(cnn_model_path)
        self.predictor = dlib.shape_predictor(shape_model_path)
        self.hog_upsample = int(hog_upsample)
        self.cnn_upsample = int(cnn_upsample)
        self.use_pose_filter = bool(use_pose_filter)
        self.yaw_thresh_deg = float(yaw_thresh_deg)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _predict_landmarks(self, rgb: np.ndarray,
                           rect: dlib.rectangle) -> np.ndarray:
        shape = self.predictor(rgb, rect)
        return np.array([[p.x, p.y] for p in shape.parts()], dtype=np.int32)

    def _rois_from_landmarks(self, landmarks: np.ndarray,
                              img_wh: Tuple[int, int],
                              face_rect: Optional[dlib.rectangle] = None,
                              pad_mouth: bool = False):
        """Return (left_eye, right_eye, mouth) ROI tuples in (x, y, w, h) format."""
        W, H = img_wh

        def eye_box(points, pad: int = 10):
            x0, y0 = points.min(axis=0)
            x1, y1 = points.max(axis=0)
            x0 = max(0, int(x0 - pad))
            y0 = max(0, int(y0 - pad))
            w  = int((x1 - x0) + 2 * pad)
            h  = int((y1 - y0) + 2 * pad)
            x1 = min(W - 1, x0 + w)
            y1 = min(H - 1, y0 + h)
            return (x0, y0, w, h)

        left_eye  = eye_box(landmarks[36:42], pad=12)
        right_eye = eye_box(landmarks[42:48], pad=12)

        mouth_tight = bbox_from_points(landmarks[48:68])

        if pad_mouth:
            if face_rect is not None:
                face_w = face_rect.width()
                face_h = face_rect.height()
                pad_x     = int(face_w * 0.20)
                pad_y_up  = int(face_h * 0.05)
                pad_y_down= int(face_h * 0.25)
            else:
                pad_x, pad_y_up, pad_y_down = 30, 10, 40
            x0, y0, x1, y1 = mouth_tight.astype(int)
            x0 = max(0, x0 - pad_x)
            y0 = max(0, y0 - pad_y_up)
            x1 = min(W - 1, x1 + pad_x)
            y1 = min(H - 1, y1 + pad_y_down)
        else:
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
        Detect the largest face and predict 68 landmarks.

        Returns
        -------
        dict with keys:
            is_valid, face_bbox (x,y,w,h), landmarks (68×2),
            eye_regions [(l,r) tuples], mouth_region tuple,
            processing_time, detector ('dlib_hog' or 'dlib_cnn')
        """
        t0 = time.time()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        W, H = image_bgr.shape[1], image_bgr.shape[0]

        def make_output(rect: dlib.rectangle, pad_mouth: bool = False,
                        detector: str = 'dlib_hog') -> Optional[Dict]:
            lm = self._predict_landmarks(rgb, rect)
            if self.use_pose_filter:
                yaw = approx_yaw_from_landmarks(lm)
                if abs(yaw) > self.yaw_thresh_deg:
                    return None
            x, y, w, h = rect_to_tuple(rect)
            eL, eR, mB = self._rois_from_landmarks(lm, (W, H), face_rect=rect,
                                                    pad_mouth=pad_mouth)
            return {
                'is_valid': True,
                'face_bbox': (x, y, w, h),
                'landmarks': lm,
                'eye_regions': [tuple(eL), tuple(eR)],
                'mouth_region': tuple(mB),
                'processing_time': time.time() - t0,
                'detector': detector,
            }

        # Stage 1: HOG
        faces_hog = self.hog(rgb, self.hog_upsample)
        if len(faces_hog) > 0:
            rect = max(faces_hog, key=lambda r: r.width() * r.height())
            out = make_output(rect, pad_mouth=False, detector='dlib_hog')
            if out is not None:
                return out

        # Stage 2: CNN fallback
        dets = self.cnn(rgb, self.cnn_upsample)
        if len(dets) == 0:
            return {
                'is_valid': False, 'face_bbox': None, 'landmarks': None,
                'eye_regions': None, 'mouth_region': None,
                'processing_time': time.time() - t0, 'detector': 'dlib_cnn',
            }
        rects = [d.rect for d in dets]
        rect = max(rects, key=lambda r: r.width() * r.height())
        out = make_output(rect, pad_mouth=True, detector='dlib_cnn')
        if out is not None:
            return out

        return {
            'is_valid': False, 'face_bbox': None, 'landmarks': None,
            'eye_regions': None, 'mouth_region': None,
            'processing_time': time.time() - t0, 'detector': 'dlib_cnn',
        }
