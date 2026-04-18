"""
transformer_enhanced.py
=======================
EnhancedOcclusionAwareTransformer — the primary gated transformer model used in
the DOFG-DMS journal experiments.

Architecture highlights
-----------------------
- 4 region-wise feature projectors (face, left_eye, right_eye, mouth)
- MLP occlusion gates (eye_gate, mouth_gate) mapping P(occ) -> gate in [floor, 1]
- Sinusoidal positional encodings + region-type embeddings
- 2/3-layer pre-norm Transformer encoder
- Learnable query-token cross-attention pooling
- 3-class classification head

V7 (Phase-1) — minimum-viable attention-bias gating (**default**)
-----------------------------------------------------------------
The legacy (V4/V5) behavior multiplied each region token by its gate before
feeding the encoder. Because the encoder is pre-norm (``norm_first=True``) and
ends with a LayerNorm, per-token multiplicative scaling is neutralised inside
the attention computation (``LN(g*x) == LN(x)`` when g scales all features of a
token). The gate factor therefore only lived on the residual path and was
largely washed out by the time pooling happened, which matches the tiny
observed stress-test gains.

V6 replaced that with **additive log-space attention-bias gating** (inspired by
ORFormer WACV 2025 and Ma et al. CVPR 2023) AND bundled three auxiliary heads
on top — logit-bias, 2-D estimator calibration, and gate-dropout 0.1. That
bundle regressed clean-test accuracy by ~30 pp on fold 0 because the logit-bias
head activated whenever ``gate < 1`` (which is true even on clean frames where
``p_eye, p_mouth ≈ 0.1-0.25``) and learned a pathological "boost Yawn" prior
under class-weighted loss.

V7 keeps the *mechanically correct* part (attention-bias gating) and makes the
three auxiliary knobs strictly opt-in, off by default:

1. **Attention-bias gating** (ON by default). Convert the gate in
   ``[floor, 1]`` into an additive log-space bias that is added to the **key**
   columns of every attention operation (encoder self-attention and the final
   query-token cross-attention). Lower gate ⇒ more negative bias ⇒ softmax
   down-weights that region as a key. This is *invariant* to the per-token
   LayerNorm and therefore actually changes the classifier's view of which
   regions to trust. **Critical invariant**: when ``gate == 1`` the log-bias is
   exactly 0, so clean-frame behaviour is identical to the ungated model.
2. **Gate-conditioned logit bias head** (OFF by default). A tiny MLP takes the
   two gates and emits a per-class logit shift. Useful in theory, but in
   practice learns a class-prior shortcut under class-weighted loss. Only
   enable with ``use_logit_bias=True`` after validating on a dev fold.
3. **Estimator calibration** (OFF by default). Residual 2-D MLP re-mapping
   ``[p_eye, p_mouth]`` before the gate MLPs. Adds capacity that de-synchs the
   ``gate_alignment`` loss from the gate-MLP input distribution. Enable only
   when the raw estimator is the measured bottleneck.
4. **Gate dropout** (0.0 by default). Forces a gate to 1.0 with probability
   ``gate_dropout`` during training. V6 used 0.1 which made training
   unstable for this 2-layer model; V7 defaults to 0 and only re-enables it
   as an ablation.

All additions are backward-compatible: legacy checkpoints load with
``strict=False`` and the model still returns ``gate_factors`` in ``[B, 4]``,
so the existing trainer, ablation utilities, stress-test, and analysis plots
work unchanged.

Usage
-----
    from transformer_enhanced import EnhancedOcclusionAwareTransformer
    model = EnhancedOcclusionAwareTransformer(gating_mode='attention')
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedOcclusionAwareTransformer(nn.Module):
    """Gated multi-region transformer for driver-state classification.

    Parameters
    ----------
    feature_dim : int
        Input feature dimension per region (default 512).
    hidden_dim : int
        Transformer hidden size (default 128).
    num_heads : int
        Number of attention heads (default 4).
    num_classes : int
        Number of output classes (default 3).
    num_layers : int
        Number of transformer encoder layers (default 2).
    use_relative_pos : bool
        Use sinusoidal positional encodings instead of learnable embeddings.
    gate_floor, eye_floor, mouth_floor : float
        Minimum gate value per region group. ``eye_floor``/``mouth_floor``
        default to ``gate_floor`` if None.
    gating_mode : {'attention', 'legacy', 'both', 'none'}
        - ``attention`` (default): additive log-gate bias on attention scores.
        - ``legacy``: multiplicative gating on input tokens (historical V4).
        - ``both``: apply both mechanisms (mostly for ablation).
        - ``none``: gates are computed for logging but not applied.
    use_logit_bias : bool
        If True, add a gate-conditioned logit bias head that lets the
        classifier learn to adjust class priors when regions are gated out.
    use_estimator_calibration : bool
        If True, add a per-channel 1-D calibration before the gate MLPs.
    gate_dropout : float
        Probability during training that a gate is forced to 1.0 (per sample,
        per region group). Improves robustness to estimator noise.
    """

    _VALID_GATING_MODES = ('attention', 'legacy', 'both', 'none')

    def __init__(self,
                 feature_dim: int = 512,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 num_classes: int = 3,
                 num_layers: int = 2,
                 use_relative_pos: bool = True,
                 gate_floor: float = 0.05,
                 eye_floor: Optional[float] = None,
                 mouth_floor: Optional[float] = None,
                 gating_mode: str = 'attention',
                 use_logit_bias: bool = False,
                 use_estimator_calibration: bool = False,
                 gate_dropout: float = 0.0):
        super().__init__()
        if gating_mode not in self._VALID_GATING_MODES:
            raise ValueError(
                f'gating_mode={gating_mode!r} must be one of {self._VALID_GATING_MODES}')

        self.feature_dim = feature_dim
        self.hidden_dim  = hidden_dim
        self.num_heads   = num_heads
        self.num_classes = num_classes
        self.num_regions = 4
        self.num_layers  = num_layers
        self.use_relative_pos = use_relative_pos
        self.gate_floor  = float(gate_floor)
        self.eye_floor   = float(eye_floor)   if eye_floor   is not None else self.gate_floor
        self.mouth_floor = float(mouth_floor) if mouth_floor is not None else self.gate_floor
        self.gating_mode = gating_mode
        self.use_logit_bias = bool(use_logit_bias)
        self.use_estimator_calibration = bool(use_estimator_calibration)
        self.gate_dropout = float(gate_dropout)
        # Set by ablation_utils.disable_gates_at_inference to bypass ALL
        # gating paths (attention bias + logit bias) in one go.
        self._force_gating_disabled = False

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

        # ── Estimator calibration (joint, 2-D input / 2-D output) ────────────
        # Residual calibration head that re-maps raw [p_eye, p_mouth] to a
        # better-behaved severity signal *before* the gate MLP sees them.
        # A joint (2-D) head was chosen so the module can repair the known
        # cross-region interference failure mode where p_mouth collapses
        # under simultaneous eye+mouth occlusion. The head is initialised
        # to ≈identity (delta ≈ 0) so behavior at epoch 0 is equivalent to
        # the no-calibration path.
        if self.use_estimator_calibration:
            self.estimator_calibration = nn.Sequential(
                nn.Linear(2, 16), nn.GELU(),
                nn.Linear(16, 2),
            )
            # Small-init so the residual delta starts near zero and the
            # training dynamics are stable.
            for p in self.estimator_calibration.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=0.01)
                else:
                    nn.init.zeros_(p)
            self._calib_alpha = nn.Parameter(torch.tensor(0.5))
        else:
            self.estimator_calibration = None
            self._calib_alpha = None

        # ── Occlusion gates (MLP: p -> gate in [0,1]) ────────────────────────
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

        # ── Positional + region-type embeddings ───────────────────────────────
        if use_relative_pos:
            self.pos_encoding = self._create_sinusoidal_embeddings(
                self.num_regions, hidden_dim)
        else:
            self.pos_embedding = nn.Embedding(self.num_regions, hidden_dim)
        self.region_type_embedding = nn.Embedding(3, hidden_dim)  # face/eye/mouth

        # ── Transformer encoder ───────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        # ── Query-token cross-attention pooling ──────────────────────────────
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

        # ── Gate-conditioned logit bias head ─────────────────────────────────
        # Input is [eye_gate, mouth_gate] in [0,1]. Output is a per-class shift
        # that gets added to the logits. Helps the model keep a sensible prior
        # over (EyeClosed, Yawn, Neutral) when a region is missing.
        if self.use_logit_bias:
            self.gate_logit_bias = nn.Sequential(
                nn.Linear(2, 16), nn.GELU(),
                nn.Linear(16, num_classes),
            )
            # Start with near-zero impact so behavior ≈ legacy at init.
            for p in self.gate_logit_bias.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=0.01)
                else:
                    nn.init.zeros_(p)
        else:
            self.gate_logit_bias = None

        self._init_weights()

    # ── Helpers ──────────────────────────────────────────────────────────────

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
            # Skip already-initialised heads.
            if name.startswith('gate_logit_bias'):
                continue
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
                    v = v.to(dev).view(-1).float()
                    if v.size(0) == 1 and bs > 1:
                        v = v.expand(bs)
                    return v
                return torch.full((bs,), float(v), device=dev)

            return (_to_tensor(eye_prob, batch_size, device),
                    _to_tensor(mouth_prob, batch_size, device))
        return (torch.zeros(batch_size, device=device),
                torch.zeros(batch_size, device=device))

    def _compute_gates(self, eye_occ: torch.Tensor, mouth_occ: torch.Tensor
                       ) -> Dict[str, torch.Tensor]:
        """Return a dict with per-region gate values in [floor, 1].

        Shape of each gate: ``[B]``. The ``face_gate`` is always 1.0.
        """
        if self.use_estimator_calibration and self.estimator_calibration is not None:
            # Joint residual calibration: p_cal = clamp(p_raw + alpha * tanh(f(p_raw)))
            # Uses both channels as input so the head can correct the
            # cross-interference failure mode (p_mouth collapsing under
            # combined occlusion) noted in the estimator audit.
            p_raw = torch.stack([eye_occ, mouth_occ], dim=1)  # [B, 2]
            delta = torch.tanh(self.estimator_calibration(p_raw))
            alpha = self._calib_alpha
            p_cal = (p_raw + alpha * delta).clamp(0.0, 1.0)
            eye_cal, mouth_cal = p_cal[:, 0], p_cal[:, 1]
        else:
            eye_cal, mouth_cal = eye_occ, mouth_occ

        eye_g   = self.occlusion_gates['eye_gate'](eye_cal.unsqueeze(1)).squeeze(1)
        mouth_g = self.occlusion_gates['mouth_gate'](mouth_cal.unsqueeze(1)).squeeze(1)

        fe, fm = self.eye_floor, self.mouth_floor
        left_eye  = fe + (1.0 - fe) * eye_g
        right_eye = fe + (1.0 - fe) * eye_g
        mouth     = fm + (1.0 - fm) * mouth_g
        face      = torch.ones_like(eye_g)

        # Optional gate dropout during training.
        if self.training and self.gate_dropout > 0.0:
            b = eye_g.shape[0]
            dev = eye_g.device
            keep_eye   = (torch.rand(b, device=dev) >= self.gate_dropout).float()
            keep_mouth = (torch.rand(b, device=dev) >= self.gate_dropout).float()
            # Where dropped, gate becomes 1.0.
            left_eye  = left_eye  * keep_eye   + (1.0 - keep_eye)
            right_eye = right_eye * keep_eye   + (1.0 - keep_eye)
            mouth     = mouth     * keep_mouth + (1.0 - keep_mouth)

        return {
            'face':      face,
            'left_eye':  left_eye,
            'right_eye': right_eye,
            'mouth':     mouth,
        }

    @staticmethod
    def _build_attention_bias(log_gate: torch.Tensor,
                              num_heads: int,
                              q_len: int) -> torch.Tensor:
        """Build an additive attention mask of shape ``[B*H, q_len, L]``.

        ``log_gate[:, j]`` is added uniformly to column j of the attention
        score matrix, i.e. every query attending to key position j gets a
        bias of ``log(gate_j)``. Lower gate ⇒ more negative bias ⇒ softmax
        suppresses that key.
        """
        B, L = log_gate.shape
        # [B, 1, 1, L] -> [B, H, q_len, L] -> [B*H, q_len, L]
        bias = log_gate.view(B, 1, 1, L).expand(B, num_heads, q_len, L)
        return bias.reshape(B * num_heads, q_len, L).contiguous()

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self,
                features: Dict,
                occlusion_info: Dict,
                return_attention: bool = False,
                disable_gating: bool = False) -> Dict:
        """See class docstring for the new attention-bias gating mechanism."""
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
        tokens = token_sequence + pos_emb + type_emb

        # ── Gate computation ─────────────────────────────────────────────────
        # Allow external ablation context managers to disable gating globally
        # (see ablation_utils.disable_gates_at_inference). This is important
        # because the attention-bias and logit-bias heads are not controlled by
        # the legacy trick of replacing the gate MLPs with OnesGate.
        force_disabled = bool(disable_gating) or bool(
            getattr(self, '_force_gating_disabled', False))
        if force_disabled:
            gates_bf = {
                'face':      torch.ones(batch_size, device=device),
                'left_eye':  torch.ones(batch_size, device=device),
                'right_eye': torch.ones(batch_size, device=device),
                'mouth':     torch.ones(batch_size, device=device),
            }
            effective_mode = 'none'
        else:
            eye_occ, mouth_occ = self._extract_occ_probs(
                occlusion_info, batch_size, device)
            gates_bf = self._compute_gates(eye_occ, mouth_occ)
            effective_mode = self.gating_mode

        gate_tensor = torch.stack(
            [gates_bf['face'], gates_bf['left_eye'],
             gates_bf['right_eye'], gates_bf['mouth']],
            dim=1,
        )  # [B, 4]

        # ── Apply gating ─────────────────────────────────────────────────────
        # 1. Optional multiplicative pre-gate on input tokens (legacy behavior).
        if effective_mode in ('legacy', 'both'):
            tokens = tokens * gate_tensor.unsqueeze(2)  # [B, 4, H]

        # 2. Attention-score gating: this is the mechanism that actually
        #    survives LayerNorm and meaningfully changes the classifier view.
        if effective_mode in ('attention', 'both'):
            log_gate = torch.log(gate_tensor.clamp(min=1e-6))  # [B, 4]
            self_mask  = self._build_attention_bias(
                log_gate, self.num_heads, q_len=self.num_regions)
            cross_mask = self._build_attention_bias(
                log_gate, self.num_heads, q_len=1)
        else:
            self_mask, cross_mask = None, None

        hidden_states = self.transformer_encoder(tokens, mask=self_mask)  # [B, 4, H]

        query = self.query_token.expand(batch_size, -1, -1)
        pooled, attn_w = self.cross_attention(
            query, hidden_states, hidden_states,
            attn_mask=cross_mask, need_weights=True,
        )
        pooled = self.pool_norm(pooled.squeeze(1))
        attn_w = attn_w.squeeze(1)

        logits = self.state_classifier(pooled)

        # 3. Gate-conditioned logit bias head (only when gating is active).
        if self.gate_logit_bias is not None and effective_mode != 'none':
            # Use the raw eye/mouth gate (not face) so the head sees exactly
            # the suppression the attention bias is applying.
            gate_summary = torch.stack(
                [gates_bf['left_eye'], gates_bf['mouth']], dim=1)  # [B, 2]
            logits = logits + self.gate_logit_bias(gate_summary)

        return {
            'class_logits':      logits,
            'class_probs':       F.softmax(logits, dim=-1),
            'predicted_class':   torch.argmax(logits, dim=-1),
            'attention_weights': attn_w,
            'gate_factors':      gate_tensor,
            'hidden_states':     hidden_states,
            'pooled_state':      pooled,
        }
