"""
Occlusion Estimator Audit Utilities
====================================
Standalone utility module extracted from IV_notebook_journal_extension.ipynb
for auditing the ROF-trained occlusion estimator on DMD data.
"""

import os
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import dlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as M
import torchvision.transforms as T
from PIL import Image
from insightface.app import FaceAnalysis

from config import CONFIG, CSVAnnotation


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_csv_video_data(dataset_path: str, filter_eye_states: bool = True) -> Dict:
    """Load CSV video annotations and metadata for all subjects."""
    print(f"Loading CSV video data from: {dataset_path}")

    csv_metadata = {}
    excluded_states = {"opening", "closing", "undefined"}

    for subject_folder in sorted(os.listdir(dataset_path)):
        subject_path = os.path.join(dataset_path, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        video_files = [f for f in os.listdir(subject_path) if f.endswith('.mp4')]
        csv_files = [f for f in os.listdir(subject_path) if f.endswith('.csv')]

        for video_file in video_files:
            video_name = os.path.splitext(video_file)[0]
            csv_file = f"{video_name}.csv"

            if csv_file not in csv_files:
                continue

            video_path = os.path.join(subject_path, video_file)
            csv_path = os.path.join(subject_path, csv_file)

            try:
                df = pd.read_csv(csv_path)

                if filter_eye_states:
                    before = len(df)
                    df = df[~df['eyes_state'].astype(str).str.lower().isin(excluded_states)]
                    filtered = before - len(df)
                else:
                    filtered = 0

                annotations = []
                for _, row in df.iterrows():
                    ann = CSVAnnotation(
                        frame=int(row['frame']),
                        timestamp=float(row['timestamp_s']),
                        class_label=row['class'],
                        variant=row['variant'],
                        eyes_state=row['eyes_state'],
                        yawn_with_hand=bool(row['yawn_with_hand']),
                        yawn_without_hand=bool(row['yawn_without_hand']),
                        eyes_occluded_prior=bool(row['eyes_occluded_prior']),
                        mouth_occluded_prior=bool(row['mouth_occluded_prior']),
                        glasses=bool(row['glasses']),
                    )
                    annotations.append(ann)

                key = f"{subject_folder}_{video_name}"
                csv_metadata[key] = {
                    'video_path': video_path,
                    'csv_path': csv_path,
                    'annotations': annotations,
                    'total_frames': len(annotations),
                    'subject': subject_folder,
                }
                print(f"  {key}: {len(annotations)} frames"
                      + (f" (filtered {filtered})" if filtered else ""))

            except Exception as e:
                print(f"  Error loading {csv_file}: {e}")

    return csv_metadata


# ─── Geometry Helpers ─────────────────────────────────────────────────────────

def bbox_from_points(pts: np.ndarray) -> np.ndarray:
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def expand_bbox(b: np.ndarray, scale: float = 1.25, bias=(0.0, 0.0),
                img_wh: Optional[Tuple[int, int]] = None) -> np.ndarray:
    x0, y0, x1, y1 = b.astype(np.float32)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    w, h = (x1 - x0), (y1 - y0)
    cx += bias[0] * w
    cy += bias[1] * h
    w2, h2 = (w * scale) / 2.0, (h * scale) / 2.0
    out = np.array([cx - w2, cy - h2, cx + w2, cy + h2], dtype=np.float32)
    if img_wh is not None:
        W, H = img_wh
        out[0::2] = np.clip(out[0::2], 0, W - 1)
        out[1::2] = np.clip(out[1::2], 0, H - 1)
    return out


def approx_yaw_from_landmarks(landmarks: np.ndarray) -> float:
    le = landmarks[36:42].mean(axis=0)
    re = landmarks[42:48].mean(axis=0)
    nose = landmarks[30]
    dl = np.linalg.norm(nose - le)
    dr = np.linalg.norm(nose - re)
    if (dl + dr) == 0:
        return 0.0
    asym = (dr - dl) / (dl + dr)
    return float(asym * 90.0)


# ─── Occlusion Model Helpers ─────────────────────────────────────────────────

def _safe_crop(img_np, bbox, margin=0.15):
    """Crop image around bbox (x, y, w, h) with margin."""
    if isinstance(bbox, dict):
        x = bbox.get("x", bbox.get("left", 0))
        y = bbox.get("y", bbox.get("top", 0))
        w = bbox.get("w", bbox.get("width", 0))
        h = bbox.get("h", bbox.get("height", 0))
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


def _to_pil_rgb(img_np, bgr=True):
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    if img_np.shape[2] == 4:
        img_np = img_np[..., :3]
    if bgr:
        img_np = img_np[..., ::-1]
    return Image.fromarray(np.ascontiguousarray(img_np))


def _eval_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(CONFIG['IMAGENET_MEAN'], CONFIG['IMAGENET_STD']),
    ])


# ─── Face Detector (RetinaFace + dlib landmarks) ─────────────────────────────

class FaceDetector:
    """RetinaFace detection + dlib 68-point landmarks."""

    def __init__(self,
                 shape_model_path: str,
                 det_size: Tuple[int, int] = (768, 768),
                 det_thresh: float = 0.35,
                 use_pose_filter: bool = False,
                 yaw_thresh_deg: float = 15.0):

        print("Initializing RetinaFace detector...")
        if torch.cuda.is_available():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            ctx_id = 0
        else:
            providers = ['CPUExecutionProvider']
            ctx_id = -1
        self.face_app = FaceAnalysis(
            name='buffalo_sc',
            providers=providers,
        )
        self.face_app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)
        self.predictor = dlib.shape_predictor(shape_model_path)
        self.use_pose_filter = bool(use_pose_filter)
        self.yaw_thresh_deg = float(yaw_thresh_deg)
        print(f"  RetinaFace (buffalo_sc) + dlib 68pt ready  [thresh={det_thresh}]")

    def _predict_landmarks(self, rgb: np.ndarray, rect: dlib.rectangle) -> np.ndarray:
        shape = self.predictor(rgb, rect)
        return np.array([[p.x, p.y] for p in shape.parts()], dtype=np.int32)

    def _rois_from_landmarks(self, landmarks: np.ndarray, img_wh: Tuple[int, int]):
        W, H = img_wh

        def eye_box(points, pad=12):
            x0, y0 = points.min(axis=0)
            x1, y1 = points.max(axis=0)
            x0 = max(0, int(x0 - pad))
            y0 = max(0, int(y0 - pad))
            w = int((x1 - x0) + 2 * pad)
            h = int((y1 - y0) + 2 * pad)
            return (x0, y0, w, h)

        left_eye = eye_box(landmarks[36:42], pad=12)
        right_eye = eye_box(landmarks[42:48], pad=12)

        mouth_tight = bbox_from_points(landmarks[48:68])
        x0, y0, x1, y1 = mouth_tight.astype(int)
        pad_x, pad_y = 15, 10
        x0 = max(0, x0 - pad_x)
        y0 = max(0, y0 - pad_y)
        x1 = min(W - 1, x1 + pad_x)
        y1 = min(H - 1, y1 + pad_y)
        mouth = (x0, y0, x1 - x0, y1 - y0)

        return (left_eye, right_eye, mouth)

    def detect_face_and_landmarks(self, image_bgr: np.ndarray) -> Dict:
        t0 = time.time()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        H, W = image_bgr.shape[:2]

        faces = self.face_app.get(image_bgr)

        if len(faces) == 0:
            return {'is_valid': False, 'face_bbox': None, 'landmarks': None,
                    'eye_regions': None, 'mouth_region': None,
                    'processing_time': time.time() - t0, 'detector': 'retinaface'}

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = face.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        rect = dlib.rectangle(x1, y1, x2, y2)

        landmarks = self._predict_landmarks(rgb, rect)

        if self.use_pose_filter:
            yaw = approx_yaw_from_landmarks(landmarks)
            if abs(yaw) > self.yaw_thresh_deg:
                return {'is_valid': False, 'face_bbox': None, 'landmarks': None,
                        'eye_regions': None, 'mouth_region': None,
                        'processing_time': time.time() - t0, 'detector': 'retinaface'}

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


# ─── Occlusion Estimator (ROF-trained ResNet-34) ─────────────────────────────

class ResNet34OcclusionModel:
    """
    Inference-only wrapper around the 2-logit ROF-trained model.
    Outputs: [P(eyes occluded), P(mouth occluded)]
    """

    def __init__(self, ckpt_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights_enum = getattr(M, "ResNet34_Weights", None)
        weights = weights_enum.IMAGENET1K_V1 if weights_enum else None
        self.model = M.resnet34(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))
        self.model.to(self.device).eval()
        self.tf = _eval_transform()

    @torch.no_grad()
    def predict_probs(self, image_np, face_bbox=None, image_bgr=True, face_margin=0.15):
        """
        Returns np.array([p_eyes, p_mouth]) in [0, 1].
        image_np: HxWxC numpy array.
        face_bbox: (x, y, w, h) or None for full image.
        """
        crop = _safe_crop(image_np, face_bbox, margin=face_margin) if face_bbox is not None else image_np
        pil = _to_pil_rgb(crop, bgr=image_bgr)
        x = self.tf(pil).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        return probs


class TrainedOcclusionDetector:
    """High-level wrapper matching the pipeline's phase-3 interface."""

    def __init__(self, ckpt_path: str, device=None,
                 image_bgr: bool = True, face_margin: float = 0.15,
                 thr_eyes=None, thr_mouth=None):
        self.model = ResNet34OcclusionModel(ckpt_path, device=device)
        self.image_bgr = image_bgr
        self.face_margin = face_margin
        self.thr_eyes = thr_eyes
        self.thr_mouth = thr_mouth

    @torch.no_grad()
    def analyze_occlusion_and_states(self, image_np, landmarks, eye_regions,
                                     mouth_region, face_bbox, feature_result):
        try:
            p_eyes, p_mouth = self.model.predict_probs(
                image_np, face_bbox=face_bbox,
                image_bgr=self.image_bgr, face_margin=self.face_margin,
            )
            result = {
                "is_valid": True,
                "occlusion_analysis": {
                    "eye_occlusion_prob": float(p_eyes),
                    "mouth_occlusion_prob": float(p_mouth),
                },
            }
            if self.thr_eyes is not None and self.thr_mouth is not None:
                result["occlusion_analysis"].update({
                    "eyes_occluded": bool(p_eyes >= self.thr_eyes),
                    "mouth_occluded": bool(p_mouth >= self.thr_mouth),
                })
            return result
        except Exception as e:
            return {"is_valid": False, "error": str(e)}


# ─── Frame Sampling ──────────────────────────────────────────────────────────

def sample_frames_for_audit(
    csv_data: Dict,
    max_per_subject: Optional[int] = None,
    include_all_occluded: bool = True,
    seed: int = 42,
) -> Dict[str, List[CSVAnnotation]]:
    """
    Frame selection for audit.
    - max_per_subject=None  -> use ALL frames (no sampling).
    - max_per_subject=300   -> keep all occluded + sample non-occluded up to budget.
    Returns {video_key: [selected CSVAnnotation objects]}.
    """
    rng = np.random.RandomState(seed)
    selected = {}

    for video_key, meta in csv_data.items():
        anns = meta['annotations']

        if max_per_subject is None:
            selected[video_key] = sorted(anns, key=lambda a: a.frame)
            print(f"  {video_key}: ALL {len(anns)} frames")
            continue

        occluded = [a for a in anns if a.eyes_occluded_prior or a.mouth_occluded_prior]
        non_occluded = [a for a in anns if not a.eyes_occluded_prior and not a.mouth_occluded_prior]

        if include_all_occluded and len(occluded) <= max_per_subject:
            # Occluded fit within budget; fill remainder with non-occluded
            budget = max_per_subject - len(occluded)
            if budget < len(non_occluded):
                idx = rng.choice(len(non_occluded), size=budget, replace=False)
                sampled_non_occ = [non_occluded[i] for i in sorted(idx)]
            else:
                sampled_non_occ = non_occluded
            combined = occluded + sampled_non_occ
        else:
            # Either not prioritising occluded, or too many to fit.
            # Stratified sample: keep the same occluded/non-occluded ratio.
            n_occ_target = max(1, int(max_per_subject * len(occluded) / len(anns))) if occluded else 0
            n_non_target = max_per_subject - n_occ_target

            if n_occ_target < len(occluded):
                idx = rng.choice(len(occluded), size=n_occ_target, replace=False)
                sampled_occ = [occluded[i] for i in sorted(idx)]
            else:
                sampled_occ = occluded

            if n_non_target < len(non_occluded):
                idx = rng.choice(len(non_occluded), size=n_non_target, replace=False)
                sampled_non_occ = [non_occluded[i] for i in sorted(idx)]
            else:
                sampled_non_occ = non_occluded

            combined = sampled_occ + sampled_non_occ

        combined.sort(key=lambda a: a.frame)
        selected[video_key] = combined
        n_occ = sum(1 for a in combined if a.eyes_occluded_prior or a.mouth_occluded_prior)
        n_non = len(combined) - n_occ
        print(f"  {video_key}: {n_occ} occluded + {n_non} non-occluded = {len(combined)} frames")

    return selected


def extract_and_predict(
    csv_data: Dict,
    sampled: Dict[str, List[CSVAnnotation]],
    face_detector: FaceDetector,
    occlusion_model: ResNet34OcclusionModel,
    image_bgr_for_model: bool = False,
) -> pd.DataFrame:
    """
    For each sampled frame: read from video, detect face, run occlusion estimator.
    Returns a DataFrame with columns:
        subject, video_key, frame, class_label, variant, eyes_state,
        yawn_with_hand, yawn_without_hand, glasses,
        eyes_occluded_prior, mouth_occluded_prior,
        p_eye, p_mouth, face_detected, det_score
    """
    rows = []
    total = sum(len(v) for v in sampled.values())
    processed = 0

    for video_key, ann_list in sampled.items():
        meta = csv_data[video_key]
        cap = cv2.VideoCapture(meta['video_path'])
        if not cap.isOpened():
            print(f"  [SKIP] Cannot open {meta['video_path']}")
            continue

        subject = meta['subject']
        print(f"\n  Processing {video_key} ({len(ann_list)} frames) ...")

        for ann in ann_list:
            cap.set(cv2.CAP_PROP_POS_FRAMES, ann.frame)
            ret, frame_bgr = cap.read()
            if not ret:
                continue

            det = face_detector.detect_face_and_landmarks(frame_bgr)

            row = {
                'subject': subject,
                'video_key': video_key,
                'frame': ann.frame,
                'class_label': ann.class_label,
                'variant': ann.variant,
                'eyes_state': ann.eyes_state,
                'yawn_with_hand': ann.yawn_with_hand,
                'yawn_without_hand': ann.yawn_without_hand,
                'glasses': ann.glasses,
                'eyes_occluded_prior': ann.eyes_occluded_prior,
                'mouth_occluded_prior': ann.mouth_occluded_prior,
                'face_detected': det['is_valid'],
                'det_score': det.get('detection_score', np.nan),
                'p_eye': np.nan,
                'p_mouth': np.nan,
            }

            if det['is_valid']:
                frame_for_model = frame_bgr if image_bgr_for_model else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                probs = occlusion_model.predict_probs(
                    frame_for_model,
                    face_bbox=det['face_bbox'],
                    image_bgr=image_bgr_for_model,
                    face_margin=0.15,
                )
                row['p_eye'] = float(probs[0])
                row['p_mouth'] = float(probs[1])

            rows.append(row)
            processed += 1
            if processed % 200 == 0:
                print(f"    [{processed}/{total}] frames processed ...")

        cap.release()

    print(f"\nDone: {processed}/{total} frames processed, {len(rows)} rows collected.")
    return pd.DataFrame(rows)
