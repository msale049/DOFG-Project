"""
DOFG-DMS Pipeline Components
=============================
Reusable model classes for the Dynamic Occlusion-Aware Feature Gating
pipeline, extracted from IV_notebook_journal_extension.ipynb.

Contains:
- ResNet34Classifier: backbone architecture
- ResNet34FeatureExtractor: region-wise 512D feature extraction
- EnhancedOcclusionAwareTransformer: gating + classification head
- split_video_ids / extract_features_stratified: data pipeline helpers
"""

import math
import random
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ─── ResNet34 Backbone ──────────────────────────────────────────────────────

class ResNet34Classifier(nn.Module):
    def __init__(self, num_classes=3, dropout_prob=0.3):
        super().__init__()
        self.model = tv_models.resnet34(weights=tv_models.ResNet34_Weights.DEFAULT)

        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.layer4.parameters():
            p.requires_grad = True
        for p in self.model.fc.parameters():
            p.requires_grad = True

        self.model.layer4 = nn.Sequential(self.model.layer4, nn.Dropout(dropout_prob))
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)


# ─── Region-wise Feature Extractor ─────────────────────────────────────────

class ResNet34FeatureExtractor:
    """Extracts 512-D features for face / left_eye / right_eye / mouth."""

    def __init__(self, model_path: str, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = ResNet34Classifier(num_classes=3, dropout_prob=0.3)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()

        self.feature_extractor = self._create_feature_extractor()

        self.transform_val = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _create_feature_extractor(self):
        backbone = self.model.model if hasattr(self.model, "model") else self.model
        feat_ext = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            backbone.avgpool,
        )
        feat_ext.to(self.device)
        feat_ext.eval()
        return feat_ext

    def extract_region_features(
        self, image: np.ndarray, face_bbox: Tuple,
        eye_regions: List[Tuple], mouth_region: Tuple,
    ) -> Dict:
        results = {
            "face_features": None,
            "left_eye_features": None,
            "right_eye_features": None,
            "mouth_features": None,
            "successful_regions": [],
        }
        regions = [
            ("face_features", "face", face_bbox),
            ("left_eye_features", "left_eye", eye_regions[0] if eye_regions and len(eye_regions) >= 1 else None),
            ("right_eye_features", "right_eye", eye_regions[1] if eye_regions and len(eye_regions) >= 2 else None),
            ("mouth_features", "mouth", mouth_region),
        ]
        for key, name, bbox in regions:
            if bbox is None:
                continue
            feat = self._extract_single_region(image, bbox, name)
            if feat is not None:
                results[key] = feat
                results["successful_regions"].append(name)
        return results

    def _extract_single_region(
        self, image: np.ndarray, region_bbox: Tuple, region_name: str,
    ) -> Optional[np.ndarray]:
        try:
            x, y, w, h = [int(v) for v in region_bbox]
            x = max(0, x)
            y = max(0, y)
            w = min(image.shape[1] - x, w)
            h = min(image.shape[0] - y, h)
            if w <= 0 or h <= 0:
                return None
            region = image[y : y + h, x : x + w]
            if region.size == 0:
                return None
            region_tensor = self.transform_val(region).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.feature_extractor(region_tensor)
                features = features.view(features.size(0), -1).cpu().numpy().squeeze()
            return features.astype(np.float32)
        except Exception as e:
            print(f"  [warn] {region_name}: {e}")
            return None


# ─── Enhanced Occlusion-Aware Transformer ───────────────────────────────────

class EnhancedOcclusionAwareTransformer(nn.Module):

    def __init__(self, feature_dim=512, hidden_dim=128, num_heads=4,
                 num_classes=3, num_layers=2, use_relative_pos=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.num_regions = 4
        self.num_layers = num_layers
        self.use_relative_pos = use_relative_pos

        self.feature_projectors = nn.ModuleDict({
            r: nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            for r in ["face", "left_eye", "right_eye", "mouth"]
        })

        self.occlusion_gates = nn.ModuleDict({
            "eye_gate": nn.Sequential(
                nn.Linear(1, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 1), nn.Sigmoid(),
            ),
            "mouth_gate": nn.Sequential(
                nn.Linear(1, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 1), nn.Sigmoid(),
            ),
        })

        if use_relative_pos:
            self.pos_encoding = self._create_sinusoidal_embeddings(self.num_regions, hidden_dim)
        else:
            self.pos_embedding = nn.Embedding(self.num_regions, hidden_dim)

        self.region_type_embedding = nn.Embedding(3, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_dim),
        )

        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        nn.init.xavier_uniform_(self.query_token)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=0.1, batch_first=True,
        )
        self.pool_norm = nn.LayerNorm(hidden_dim)
        self.attention_temperature = nn.Parameter(
            torch.ones(1) * math.sqrt(hidden_dim // num_heads)
        )

        self.state_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self._init_weights()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _create_sinusoidal_embeddings(self, n_pos, d_model):
        pe = torch.zeros(n_pos, d_model)
        pos = torch.arange(0, n_pos, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(pos * div)
        else:
            pe[:, 1::2] = torch.cos(pos * div[:-1])
        return nn.Parameter(pe, requires_grad=False)

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(self, features, occlusion_info,
                return_attention=False, disable_gating=False):
        """
        Parameters
        ----------
        features : dict with keys face, left_eye, right_eye, mouth
        occlusion_info : dict with eye_occlusion_prob, mouth_occlusion_prob
        disable_gating : if True, all gate factors are forced to 1.0
        """
        device = next(self.parameters()).device
        region_names = ["face", "left_eye", "right_eye", "mouth"]
        region_types = [0, 1, 1, 2]

        def to_B512(x):
            if isinstance(x, (list, tuple)):
                parts = []
                for t in x:
                    parts.append(t if torch.is_tensor(t) else torch.tensor(t, dtype=torch.float32))
                x = torch.stack(parts, dim=0)
            elif not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return x

        batch_size = None
        region_feats = {}
        for region in region_names:
            feat = to_B512(features[region]).to(device)
            if batch_size is None:
                batch_size = feat.shape[0]
            region_feats[region] = feat

        projected = [self.feature_projectors[r](region_feats[r]) for r in region_names]
        token_sequence = torch.stack(projected, dim=1)

        if self.use_relative_pos:
            pos_emb = self.pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            positions = torch.arange(self.num_regions, device=device)
            pos_emb = self.pos_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)

        type_ids = torch.tensor(region_types, device=device)
        type_emb = self.region_type_embedding(type_ids).unsqueeze(0).expand(batch_size, -1, -1)
        token_sequence = token_sequence + pos_emb + type_emb

        # --- Gating ---
        if disable_gating:
            face_gates = torch.ones(batch_size, device=device)
            left_eye_gates = torch.ones(batch_size, device=device)
            right_eye_gates = torch.ones(batch_size, device=device)
            mouth_gates_final = torch.ones(batch_size, device=device)
        else:
            eye_occ, mouth_occ = self._extract_occ_probs(occlusion_info, batch_size, device)
            eye_g = self.occlusion_gates["eye_gate"](eye_occ.unsqueeze(1)).squeeze(1)
            mouth_g = self.occlusion_gates["mouth_gate"](mouth_occ.unsqueeze(1)).squeeze(1)
            face_gates = torch.ones(batch_size, device=device)
            left_eye_gates = 0.3 + 0.7 * eye_g
            right_eye_gates = 0.3 + 0.7 * eye_g
            mouth_gates_final = 0.3 + 0.7 * mouth_g

        gated_tokens = token_sequence.clone()
        gated_tokens[:, 0, :] *= face_gates.unsqueeze(1)
        gated_tokens[:, 1, :] *= left_eye_gates.unsqueeze(1)
        gated_tokens[:, 2, :] *= right_eye_gates.unsqueeze(1)
        gated_tokens[:, 3, :] *= mouth_gates_final.unsqueeze(1)

        hidden_states = self.transformer_encoder(gated_tokens)

        query = self.query_token.expand(batch_size, -1, -1)
        pooled, attn_w = self.cross_attention(query, hidden_states, hidden_states, need_weights=True)
        pooled = self.pool_norm(pooled.squeeze(1))
        attn_w = attn_w.squeeze(1)

        logits = self.state_classifier(pooled)
        gate_tensor = torch.stack([face_gates, left_eye_gates, right_eye_gates, mouth_gates_final], dim=1)

        outputs = {
            "class_logits": logits,
            "class_probs": F.softmax(logits, dim=-1),
            "predicted_class": torch.argmax(logits, dim=-1),
            "attention_weights": attn_w,
            "gate_factors": gate_tensor,
            "hidden_states": hidden_states,
            "pooled_state": pooled,
        }
        return outputs

    @staticmethod
    def _extract_occ_probs(oi, batch_size, device):
        if isinstance(oi, dict):
            eye_prob = oi.get("eye_occlusion_prob", 0.0)
            mouth_prob = oi.get("mouth_occlusion_prob", 0.0)
            for name, val in [("eye", eye_prob), ("mouth", mouth_prob)]:
                pass  # validated below
            def _to_tensor(v, bs, dev):
                if torch.is_tensor(v):
                    v = v.to(dev).view(-1)
                    if v.size(0) == 1 and bs > 1:
                        v = v.expand(bs)
                    return v
                return torch.full((bs,), float(v), device=dev)
            return _to_tensor(eye_prob, batch_size, device), _to_tensor(mouth_prob, batch_size, device)
        return torch.zeros(batch_size, device=device), torch.zeros(batch_size, device=device)


# ─── Data Pipeline Helpers ──────────────────────────────────────────────────

def split_video_ids(csv_data: Dict, num_test: int = 3,
                    num_val: int = 0, seed: int = 42) -> Dict:
    """Deterministic video-level split (matches IV_notebook strategy)."""
    vids = sorted(csv_data.keys())
    rng = random.Random(seed)
    rng.shuffle(vids)
    test = vids[:num_test]
    val = vids[num_test:num_test + num_val]
    train = vids[num_test + num_val:]
    return {"train": train, "val": train, "test": test}


def extract_features_with_augmentation(
    csv_data: Dict,
    splits: Dict,
    face_detector,
    feat_extractor: "ResNet34FeatureExtractor",
    occ_model,
    apply_occ_fn,
    opacity_levels: List[float],
    aug_clean_fraction: float = 0.60,
    num_samples_per_video: Optional[int] = None,
    val_ratio: float = 0.20,
    random_state: int = 42,
    label_map: Optional[Dict] = None,
) -> Tuple[List, List, List]:
    """Extract features from DMD videos with per-subject synthetic occlusion
    augmentation for training frames.

    Augmentation strategy (training subjects only):
    - For each subject, frame indices are divided into buckets:
        60 % clean (no overlay)
        ~13 % eye_only  (sunglasses band, random opacity)
        ~13 % mouth_only (mask rect, random opacity)
        ~14 % both      (eye + mouth, random opacity)
    - Each augmented frame receives ONE randomly chosen opacity from
      the non-zero entries of *opacity_levels*.
    - Test-subject frames are always left clean (they are occluded later
      during the stress-test phase, exactly as in Experiment A).

    Parameters
    ----------
    apply_occ_fn : callable
        ``apply_synthetic_occlusion(rgb, landmarks, eye_opacity, mouth_opacity)``
        from synthetic_occlusion_utils.
    opacity_levels : list of float
        Full OPACITY_LEVELS list; non-zero entries are used for augmentation.
    aug_clean_fraction : float
        Fraction of each subject's frames kept clean (default 0.60).
    """
    if label_map is None:
        label_map = {"EyeClosed": 0, "Yawn": 1, "Neutral": 2}

    aug_opacities = [op for op in opacity_levels if op > 0]
    rng = np.random.default_rng(random_state)

    train_subject_samples: List[Dict] = []
    test_subject_samples: List[Dict] = []
    global_id = 0

    for video_key, meta in csv_data.items():
        cap = cv2.VideoCapture(meta["video_path"])
        if not cap.isOpened():
            print(f"  [SKIP] {video_key}")
            continue

        anns = meta["annotations"]
        n = len(anns)
        if num_samples_per_video is None or num_samples_per_video >= n:
            indices = list(range(n))
        else:
            indices = np.linspace(0, n - 1, num_samples_per_video, dtype=int).tolist()

        subject = meta["subject"]
        is_test = video_key in splits["test"]
        ok, bad = 0, 0

        # Pre-assign per-subject augmentation buckets for training subjects.
        # We shuffle frame positions, then split them into four buckets.
        bucket_map: Dict[int, Tuple[str, float]] = {}
        if not is_test:
            nf = len(indices)
            n_clean = int(nf * aug_clean_fraction)
            n_aug = nf - n_clean
            n_eye   = n_aug // 3
            n_mouth = n_aug // 3
            n_both  = n_aug - n_eye - n_mouth          # absorbs remainder
            bucket_labels = (
                ["none"] * n_clean
                + ["eye_only"] * n_eye
                + ["mouth_only"] * n_mouth
                + ["both"] * n_both
            )
            perm = rng.permutation(nf)
            for pos, bucket_pos in enumerate(perm):
                btype = bucket_labels[bucket_pos]
                if btype == "none":
                    bucket_map[pos] = ("none", 0.0)
                else:
                    op = float(rng.choice(aug_opacities))
                    bucket_map[pos] = (btype, op)

        for frame_pos, idx in enumerate(indices):
            ann = anns[idx]
            if ann.class_label not in label_map:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, ann.frame)
            ret, bgr = cap.read()
            if not ret:
                continue

            det = face_detector.detect_face_and_landmarks(bgr)
            if not det["is_valid"]:
                bad += 1
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            aug_type, aug_opacity = "none", 0.0
            if not is_test:
                aug_type, aug_opacity = bucket_map.get(frame_pos, ("none", 0.0))
                if aug_type != "none":
                    lm = det.get("landmarks")
                    if lm is not None:
                        eo = aug_opacity if aug_type in ("eye_only", "both") else 0.0
                        mo = aug_opacity if aug_type in ("mouth_only", "both") else 0.0
                        rgb = apply_occ_fn(rgb, lm, eye_opacity=eo, mouth_opacity=mo)
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            feat = feat_extractor.extract_region_features(
                bgr, det["face_bbox"], det["eye_regions"], det["mouth_region"],
            )
            rkeys = ["face_features", "left_eye_features",
                     "right_eye_features", "mouth_features"]
            if any(feat[k] is None for k in rkeys):
                bad += 1
                continue

            probs = occ_model.predict_probs(
                rgb, face_bbox=det["face_bbox"], image_bgr=False, face_margin=0.15,
            )

            sample = {
                "frame_id": global_id,
                "subject": subject,
                "video_key": video_key,
                "features": {
                    "face": feat["face_features"],
                    "left_eye": feat["left_eye_features"],
                    "right_eye": feat["right_eye_features"],
                    "mouth": feat["mouth_features"],
                },
                "occlusion_info": {
                    "eye_occlusion_prob": float(probs[0]),
                    "mouth_occlusion_prob": float(probs[1]),
                },
                "label": label_map[ann.class_label],
                "class_name": ann.class_label,
                "ground_truth": {
                    "eyes_occluded": ann.eyes_occluded_prior,
                    "mouth_occluded": ann.mouth_occluded_prior,
                    "eyes_state": ann.eyes_state,
                },
                "aug_type": aug_type,
                "aug_opacity": aug_opacity,
            }
            global_id += 1

            if is_test:
                test_subject_samples.append(sample)
            else:
                train_subject_samples.append(sample)
            ok += 1

            if ok % 500 == 0:
                print(f"    ... {ok}/{len(indices)} frames processed")

        cap.release()
        dest = "TEST" if is_test else "TRAIN"
        print(f"  {video_key} [{dest}]: {ok} ok, {bad} skipped")

    if train_subject_samples:
        labels = [s["class_name"] for s in train_subject_samples]
        train_idx, val_idx = train_test_split(
            list(range(len(train_subject_samples))),
            test_size=val_ratio,
            stratify=labels,
            random_state=random_state,
        )
        train_samples = [train_subject_samples[i] for i in train_idx]
        val_samples   = [train_subject_samples[i] for i in val_idx]
    else:
        train_samples, val_samples = [], []

    for name, ss in [("Train", train_samples), ("Val", val_samples),
                     ("Test", test_subject_samples)]:
        dist = {}
        for s in ss:
            dist[s["class_name"]] = dist.get(s["class_name"], 0) + 1
        print(f"  {name}: {len(ss)} samples  {dist}")

    if train_samples:
        aug_dist: Dict[str, int] = {}
        for s in train_samples:
            aug_dist[s["aug_type"]] = aug_dist.get(s["aug_type"], 0) + 1
        print(f"  Train aug distribution: {aug_dist}")

    return train_samples, val_samples, test_subject_samples


def extract_features_stratified(
    csv_data: Dict,
    splits: Dict,
    face_detector,
    feat_extractor: "ResNet34FeatureExtractor",
    occ_model,
    num_samples_per_video: Optional[int] = None,
    val_ratio: float = 0.20,
    random_state: int = 42,
    label_map: Optional[Dict] = None,
) -> Tuple[List, List, List]:
    """Extract features from DMD videos and return stratified train/val/test.

    Replicates the IV_notebook split logic:
    - Videos assigned to train-subjects or test-subjects via *splits*
    - From each video, *num_samples_per_video* frames are evenly sampled
      (None = all frames)
    - Train-subject frames are stratified 80/20 into train/val by class
    - Test-subject frames go to test

    Each returned sample is a dict with keys:
        frame_id, subject, video_key, features, occlusion_info,
        label, class_name, ground_truth
    """
    if label_map is None:
        label_map = {"EyeClosed": 0, "Yawn": 1, "Neutral": 2}

    train_subject_samples: List[Dict] = []
    test_subject_samples: List[Dict] = []
    global_id = 0

    for video_key, meta in csv_data.items():
        cap = cv2.VideoCapture(meta["video_path"])
        if not cap.isOpened():
            print(f"  [SKIP] {video_key}")
            continue

        anns = meta["annotations"]
        n = len(anns)
        if num_samples_per_video is None or num_samples_per_video >= n:
            indices = list(range(n))
        else:
            indices = np.linspace(0, n - 1, num_samples_per_video, dtype=int).tolist()

        subject = meta["subject"]
        is_test = video_key in splits["test"]
        ok, bad = 0, 0

        for idx in indices:
            ann = anns[idx]
            if ann.class_label not in label_map:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, ann.frame)
            ret, bgr = cap.read()
            if not ret:
                continue

            det = face_detector.detect_face_and_landmarks(bgr)
            if not det["is_valid"]:
                bad += 1
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            feat = feat_extractor.extract_region_features(
                bgr, det["face_bbox"], det["eye_regions"], det["mouth_region"],
            )
            rkeys = ["face_features", "left_eye_features",
                     "right_eye_features", "mouth_features"]
            if any(feat[k] is None for k in rkeys):
                bad += 1
                continue

            probs = occ_model.predict_probs(
                rgb, face_bbox=det["face_bbox"], image_bgr=False, face_margin=0.15,
            )

            sample = {
                "frame_id": global_id,
                "subject": subject,
                "video_key": video_key,
                "features": {
                    "face": feat["face_features"],
                    "left_eye": feat["left_eye_features"],
                    "right_eye": feat["right_eye_features"],
                    "mouth": feat["mouth_features"],
                },
                "occlusion_info": {
                    "eye_occlusion_prob": float(probs[0]),
                    "mouth_occlusion_prob": float(probs[1]),
                },
                "label": label_map[ann.class_label],
                "class_name": ann.class_label,
                "ground_truth": {
                    "eyes_occluded": ann.eyes_occluded_prior,
                    "mouth_occluded": ann.mouth_occluded_prior,
                    "eyes_state": ann.eyes_state,
                },
            }
            global_id += 1

            if is_test:
                test_subject_samples.append(sample)
            else:
                train_subject_samples.append(sample)
            ok += 1

            if ok % 500 == 0:
                print(f"    ... {ok}/{len(indices)} frames processed")

        cap.release()
        dest = "TEST" if is_test else "TRAIN"
        print(f"  {video_key} [{dest}]: {ok} ok, {bad} skipped")

    # Stratified train / val split (80/20 by class)
    if train_subject_samples:
        labels = [s["class_name"] for s in train_subject_samples]
        train_idx, val_idx = train_test_split(
            list(range(len(train_subject_samples))),
            test_size=val_ratio,
            stratify=labels,
            random_state=random_state,
        )
        train_samples = [train_subject_samples[i] for i in train_idx]
        val_samples = [train_subject_samples[i] for i in val_idx]
    else:
        train_samples, val_samples = [], []

    for name, ss in [("Train", train_samples), ("Val", val_samples),
                     ("Test", test_subject_samples)]:
        dist = {}
        for s in ss:
            dist[s["class_name"]] = dist.get(s["class_name"], 0) + 1
        print(f"  {name}: {len(ss)} samples  {dist}")

    return train_samples, val_samples, test_subject_samples
