"""
feature_extraction.py
=====================
ResNet-34 backbone and region-wise 512-D feature extractor.

Classes
-------
ResNet34Classifier     — Fine-tunable backbone (layer4 + fc unfrozen).
ResNet34FeatureExtractor — Extracts 512-D avgpool features per facial region.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─── ResNet-34 Backbone ──────────────────────────────────────────────────────

class ResNet34Classifier(nn.Module):
    """
    ResNet-34 with the last block (layer4) and FC head unfrozen for fine-tuning.
    All earlier layers remain frozen.
    """

    def __init__(self, num_classes: int = 3, dropout_prob: float = 0.3, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet34_Weights.DEFAULT if pretrained else None
        self.model = tv_models.resnet34(weights=weights)

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


# ─── Region-wise Feature Extractor ──────────────────────────────────────────

class ResNet34FeatureExtractor:
    """
    Loads a trained ResNet34Classifier checkpoint and exposes 512-D
    average-pooled feature vectors for each facial region.

    Regions: face, left_eye, right_eye, mouth.
    """

    def __init__(self, model_path: str, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Use weights=None to avoid loading ImageNet into memory (we load our checkpoint)
        self.model = ResNet34Classifier(num_classes=3, dropout_prob=0.3, pretrained=False)
        # Load to CPU first to avoid GPU OOM when RetinaFace/ONNX already used GPU
        state = torch.load(model_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False  # inference only, no grad storage on GPU
        self.model.to(self.device)

        self.feature_extractor = self._create_feature_extractor()

        self.transform_val = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _create_feature_extractor(self) -> nn.Sequential:
        backbone = self.model.model if hasattr(self.model, 'model') else self.model
        feat_ext = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            backbone.avgpool,
        )
        feat_ext.to(self.device)
        feat_ext.eval()
        return feat_ext

    def extract_region_features(
        self,
        image: np.ndarray,
        face_bbox: Tuple,
        eye_regions: List[Tuple],
        mouth_region: Tuple,
    ) -> Dict:
        """
        Extract 512-D features for all four facial regions.

        Parameters
        ----------
        image : HxWx3 BGR uint8 NumPy array.
        face_bbox : (x, y, w, h) for the whole face crop.
        eye_regions : [(left_eye), (right_eye)] in (x, y, w, h).
        mouth_region : (x, y, w, h).

        Returns
        -------
        dict with keys:
            face_features, left_eye_features, right_eye_features,
            mouth_features (each 512-D float32 or None), successful_regions.
        """
        results = {
            'face_features': None,
            'left_eye_features': None,
            'right_eye_features': None,
            'mouth_features': None,
            'successful_regions': [],
        }
        regions = [
            ('face_features', 'face', face_bbox),
            ('left_eye_features',  'left_eye',
             eye_regions[0] if eye_regions and len(eye_regions) >= 1 else None),
            ('right_eye_features', 'right_eye',
             eye_regions[1] if eye_regions and len(eye_regions) >= 2 else None),
            ('mouth_features', 'mouth', mouth_region),
        ]
        for key, name, bbox in regions:
            if bbox is None:
                continue
            feat = self._extract_single_region(image, bbox, name)
            if feat is not None:
                results[key] = feat
                results['successful_regions'].append(name)
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
            region = image[y: y + h, x: x + w]
            if region.size == 0:
                return None
            region_tensor = self.transform_val(region).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.feature_extractor(region_tensor)
                features = features.view(features.size(0), -1).cpu().numpy().squeeze()
            return features.astype(np.float32)
        except Exception as e:
            print(f'  [warn] {region_name}: {e}')
            return None
