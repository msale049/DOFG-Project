"""
transformer_enhanced.py
=======================
EnhancedOcclusionAwareTransformer — the primary gated transformer model used in
the DOFG-DMS journal experiments.

Architecture highlights
-----------------------
- 4 region-wise feature projectors (face, left_eye, right_eye, mouth)
- MLP occlusion gates (eye_gate, mouth_gate) mapping P(occ) → gate ∈ [0.3, 1.0]
- Sinusoidal positional encodings + region-type embeddings
- 2-layer Transformer encoder with pre-norm
- Learnable query-token cross-attention pooling
- 3-class classification head

Usage
-----
    from transformer_enhanced import EnhancedOcclusionAwareTransformer
    model = EnhancedOcclusionAwareTransformer()
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedOcclusionAwareTransformer(nn.Module):
    """
    Gated multi-region transformer for driver-state classification.

    Parameters
    ----------
    feature_dim : int   Input feature dimension per region (default 512).
    hidden_dim  : int   Transformer hidden size (default 128).
    num_heads   : int   Number of attention heads (default 4).
    num_classes : int   Number of output classes (default 3).
    num_layers  : int   Number of transformer encoder layers (default 2).
    use_relative_pos : bool  Use sinusoidal PE instead of learnable embeddings.
    """

    def __init__(self, feature_dim: int = 512, hidden_dim: int = 128,
                 num_heads: int = 4, num_classes: int = 3,
                 num_layers: int = 2, use_relative_pos: bool = True):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim  = hidden_dim
        self.num_heads   = num_heads
        self.num_classes = num_classes
        self.num_regions = 4
        self.num_layers  = num_layers
        self.use_relative_pos = use_relative_pos

        # ── Feature projectors ────────────────────────────────────────────────
        self.feature_projectors = nn.ModuleDict({
            r: nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            for r in ['face', 'left_eye', 'right_eye', 'mouth']
        })

        # ── Occlusion gates (MLP: p → gate ∈ [0,1]) ─────────────────────────
        self.occlusion_gates = nn.ModuleDict({
            'eye_gate': nn.Sequential(
                nn.Linear(1, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 1), nn.Sigmoid(),
            ),
            'mouth_gate': nn.Sequential(
                nn.Linear(1, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 1), nn.Sigmoid(),
            ),
        })

        # ── Positional encodings ──────────────────────────────────────────────
        if use_relative_pos:
            self.pos_encoding = self._create_sinusoidal_embeddings(
                self.num_regions, hidden_dim)
        else:
            self.pos_embedding = nn.Embedding(self.num_regions, hidden_dim)

        # ── Region-type embeddings (0=face, 1=eye, 2=mouth) ─────────────────
        self.region_type_embedding = nn.Embedding(3, hidden_dim)

        # ── Transformer encoder ────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        # ── Query-token cross-attention pooling ───────────────────────────────
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        nn.init.xavier_uniform_(self.query_token)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=0.1, batch_first=True,
        )
        self.pool_norm = nn.LayerNorm(hidden_dim)
        self.attention_temperature = nn.Parameter(
            torch.ones(1) * math.sqrt(hidden_dim // num_heads)
        )

        # ── Classification head ───────────────────────────────────────────────
        self.state_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self._init_weights()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _create_sinusoidal_embeddings(self, n_pos: int, d_model: int):
        pe  = torch.zeros(n_pos, d_model)
        pos = torch.arange(0, n_pos, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(pos * div)
        else:
            pe[:, 1::2] = torch.cos(pos * div[:-1])
        return nn.Parameter(pe, requires_grad=False)

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    @staticmethod
    def _extract_occ_probs(oi, batch_size: int, device):
        if isinstance(oi, dict):
            eye_prob   = oi.get('eye_occlusion_prob', 0.0)
            mouth_prob = oi.get('mouth_occlusion_prob', 0.0)

            def _to_tensor(v, bs, dev):
                if torch.is_tensor(v):
                    v = v.to(dev).view(-1)
                    if v.size(0) == 1 and bs > 1:
                        v = v.expand(bs)
                    return v
                return torch.full((bs,), float(v), device=dev)

            return (_to_tensor(eye_prob, batch_size, device),
                    _to_tensor(mouth_prob, batch_size, device))
        return (torch.zeros(batch_size, device=device),
                torch.zeros(batch_size, device=device))

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, features: Dict, occlusion_info: Dict,
                return_attention: bool = False,
                disable_gating: bool = False) -> Dict:
        """
        Parameters
        ----------
        features : dict[str, Tensor]
            Keys: face, left_eye, right_eye, mouth.  Shape: (B, 512).
        occlusion_info : dict
            Keys: eye_occlusion_prob, mouth_occlusion_prob.
        disable_gating : bool
            If True, all gate factors are forced to 1.0 (ablation mode).

        Returns
        -------
        dict with keys:
            class_logits, class_probs, predicted_class,
            attention_weights, gate_factors, hidden_states, pooled_state.
        """
        device = next(self.parameters()).device
        region_names = ['face', 'left_eye', 'right_eye', 'mouth']
        region_types = [0, 1, 1, 2]

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

        batch_size   = None
        region_feats = {}
        for region in region_names:
            feat = to_B512(features[region]).to(device)
            if batch_size is None:
                batch_size = feat.shape[0]
            region_feats[region] = feat

        projected      = [self.feature_projectors[r](region_feats[r]) for r in region_names]
        token_sequence = torch.stack(projected, dim=1)   # [B, 4, H]

        if self.use_relative_pos:
            pos_emb = self.pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            positions = torch.arange(self.num_regions, device=device)
            pos_emb   = self.pos_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)

        type_ids = torch.tensor(region_types, device=device)
        type_emb = self.region_type_embedding(type_ids).unsqueeze(0).expand(batch_size, -1, -1)
        token_sequence = token_sequence + pos_emb + type_emb

        # ── Gating ─────────────────────────────────────────────────────────
        if disable_gating:
            face_gates        = torch.ones(batch_size, device=device)
            left_eye_gates    = torch.ones(batch_size, device=device)
            right_eye_gates   = torch.ones(batch_size, device=device)
            mouth_gates_final = torch.ones(batch_size, device=device)
        else:
            eye_occ, mouth_occ = self._extract_occ_probs(occlusion_info, batch_size, device)
            eye_g   = self.occlusion_gates['eye_gate'](eye_occ.unsqueeze(1)).squeeze(1)
            mouth_g = self.occlusion_gates['mouth_gate'](mouth_occ.unsqueeze(1)).squeeze(1)
            face_gates        = torch.ones(batch_size, device=device)
            left_eye_gates    = 0.3 + 0.7 * eye_g
            right_eye_gates   = 0.3 + 0.7 * eye_g
            mouth_gates_final = 0.3 + 0.7 * mouth_g

        gated = token_sequence.clone()
        gated[:, 0, :] *= face_gates.unsqueeze(1)
        gated[:, 1, :] *= left_eye_gates.unsqueeze(1)
        gated[:, 2, :] *= right_eye_gates.unsqueeze(1)
        gated[:, 3, :] *= mouth_gates_final.unsqueeze(1)

        hidden_states = self.transformer_encoder(gated)   # [B, 4, H]

        query = self.query_token.expand(batch_size, -1, -1)
        pooled, attn_w = self.cross_attention(query, hidden_states, hidden_states,
                                              need_weights=True)
        pooled = self.pool_norm(pooled.squeeze(1))
        attn_w = attn_w.squeeze(1)

        logits = self.state_classifier(pooled)
        gate_tensor = torch.stack([face_gates, left_eye_gates,
                                   right_eye_gates, mouth_gates_final], dim=1)

        return {
            'class_logits':      logits,
            'class_probs':       F.softmax(logits, dim=-1),
            'predicted_class':   torch.argmax(logits, dim=-1),
            'attention_weights': attn_w,
            'gate_factors':      gate_tensor,
            'hidden_states':     hidden_states,
            'pooled_state':      pooled,
        }
