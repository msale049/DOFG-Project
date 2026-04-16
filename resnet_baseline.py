"""
resnet_baseline.py
==================
End-to-end ResNet-34 baseline for the DOFG-DMS pipeline.

Uses only the full-face crop (224×224) as input and trains a standard
ResNet-34 classifier for the 3 driver-state classes.  No region-level
features, no occlusion estimation, no gating.

The forward signature and output dict match EnhancedOcclusionAwareTransformer
so the same trainer, evaluation, and stress-test code work unchanged.
"""

from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

FACE_CROP_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def crop_and_preprocess_face(bgr_image: np.ndarray, face_bbox, size: int = 224) -> np.ndarray:
    """Crop face from BGR image, resize to *size*×*size*, return uint8 HWC array."""
    x, y, w, h = [int(v) for v in face_bbox]
    ih, iw = bgr_image.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(iw - x, w)
    h = min(ih - y, h)
    if w <= 0 or h <= 0:
        return None
    crop = bgr_image[y:y + h, x:x + w]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size))


def face_crop_to_tensor(crop_uint8: np.ndarray) -> torch.Tensor:
    """Convert uint8 HWC BGR crop to normalised CHW float tensor."""
    return FACE_CROP_TRANSFORM(crop_uint8)


class ResNet34Baseline(nn.Module):
    """
    Full-face ResNet-34 end-to-end baseline.

    Architecture (matching reference notebook):
      - Pretrained ResNet-34 backbone
      - Frozen layers conv1 → layer3
      - Trainable layer4 + FC head
      - Dropout before final linear

    Parameters
    ----------
    num_classes  : int   Output classes (default 3).
    dropout      : float Dropout rate (default 0.3).
    """

    needs_face_crop = True

    def __init__(self, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, features: Dict, occlusion_info: Dict,
                return_attention: bool = False,
                disable_gating: bool = False) -> Dict:
        """
        Interface-compatible with EnhancedOcclusionAwareTransformer.

        Parameters
        ----------
        features       : dict; must contain 'face_crop' key with [B, 3, 224, 224] tensor.
                         Other region keys are ignored.
        occlusion_info : dict (ignored — no gating).
        return_attention, disable_gating : accepted but ignored.
        """
        device = next(self.parameters()).device
        face_crop = features['face_crop']
        if not torch.is_tensor(face_crop):
            face_crop = torch.tensor(face_crop, dtype=torch.float32)
        if face_crop.dim() == 3:
            face_crop = face_crop.unsqueeze(0)
        face_crop = face_crop.to(device)
        batch_size = face_crop.shape[0]

        logits = self.backbone(face_crop)

        gate_factors = torch.ones(batch_size, 4, device=device)
        attn_weights = torch.full((batch_size, 4), 0.25, device=device)

        return {
            'class_logits': logits,
            'class_probs': F.softmax(logits, dim=-1),
            'predicted_class': torch.argmax(logits, dim=-1),
            'attention_weights': attn_weights,
            'gate_factors': gate_factors,
            'hidden_states': face_crop.view(batch_size, -1).unsqueeze(1),
            'pooled_state': face_crop.view(batch_size, -1),
        }
