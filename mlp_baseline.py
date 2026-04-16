"""
mlp_baseline.py
===============
Region-feature MLP baseline for the DOFG-DMS pipeline.

Concatenates the 4 frozen ResNet-34 region features (face, left_eye,
right_eye, mouth — each 512-D) and passes the 2048-D vector through a
simple MLP classifier to predict 3 driver-state classes.

No gating, no attention, no positional encoding — a pure feature-fusion
baseline for comparison with the gated transformer.

The forward signature and output dict match EnhancedOcclusionAwareTransformer
so the same trainer, evaluation, and stress-test code work unchanged.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionFeatureMLP(nn.Module):
    """
    Concat(face, left_eye, right_eye, mouth) → MLP → 3 classes.

    Parameters
    ----------
    feature_dim : int   Per-region feature dimension (default 512).
    hidden_dim  : int   First hidden layer size (default 256).
    num_classes : int   Output classes (default 3).
    dropout     : float Dropout rate (default 0.3).
    """

    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256,
                 num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        input_dim = feature_dim * 4  # 4 regions

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, features: Dict, occlusion_info: Dict,
                return_attention: bool = False,
                disable_gating: bool = False) -> Dict:
        """
        Interface-compatible with EnhancedOcclusionAwareTransformer.

        Parameters
        ----------
        features       : dict with keys face, left_eye, right_eye, mouth (each [B, 512]).
        occlusion_info : dict (ignored — MLP has no gating).
        return_attention, disable_gating : accepted but ignored.
        """
        device = next(self.parameters()).device
        region_names = ['face', 'left_eye', 'right_eye', 'mouth']

        def to_B512(x):
            if isinstance(x, (list, tuple)):
                parts = [t if torch.is_tensor(t)
                         else torch.tensor(t, dtype=torch.float32) for t in x]
                x = torch.stack(parts, dim=0)
            elif not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return x

        region_tensors = []
        for r in region_names:
            region_tensors.append(to_B512(features[r]).to(device))

        concat = torch.cat(region_tensors, dim=-1)  # [B, 2048]
        batch_size = concat.shape[0]

        logits = self.mlp(concat)

        gate_factors = torch.ones(batch_size, 4, device=device)
        attn_weights = torch.full((batch_size, 4), 0.25, device=device)

        return {
            'class_logits':      logits,
            'class_probs':       F.softmax(logits, dim=-1),
            'predicted_class':   torch.argmax(logits, dim=-1),
            'attention_weights': attn_weights,
            'gate_factors':      gate_factors,
            'hidden_states':     concat.unsqueeze(1),
            'pooled_state':      concat,
        }
