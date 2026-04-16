"""
ablation_utils.py
=================
Context managers and helpers for gate-ablation studies.

The utilities in this module let you:
  - Temporarily replace the gate MLPs with constant-output modules.
  - Mask input features to zero to measure per-region importance.
  - Sweep forced gate values and compare accuracy.

All functions require a model that has an ``occlusion_gates`` ModuleDict
with keys ``'eye_gate'`` and ``'mouth_gate'`` (i.e.
``EnhancedOcclusionAwareTransformer``).

Functions / Classes
-------------------
_OnesGate              — Gate replacement that always outputs 1.0.
_ConstGate             — Gate replacement that always outputs a constant value.
disable_gates_at_inference — Context manager: force all gates to 1.0.
force_gate_values          — Context manager: force eye/mouth gates to chosen values.
sweep_forced_gates         — Evaluate accuracy across a grid of forced gate values.
eval_with_feature_mask     — Zero-out specified regions and measure accuracy.
eval_with_mask_and_occinfo — Same but also zeroes the occlusion info for masked regions.
evaluate_bins_with_gates_disabled — Accuracy by occlusion bin with gates disabled.
"""

from contextlib import contextmanager, nullcontext
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from evaluation import collect_eval_with_occlusion
from utils import move_batch_to_device


# ─── Replacement Gate Modules ────────────────────────────────────────────────

class _OnesGate(nn.Module):
    """Gate replacement that outputs a column of ones (gate = 1.0 → no suppression)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)


class _ConstGate(nn.Module):
    """Gate replacement that outputs a fixed constant *val*."""
    def __init__(self, val: float):
        super().__init__()
        self.val = float(val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 1), self.val,
                          device=x.device, dtype=x.dtype)


# ─── Context Managers ─────────────────────────────────────────────────────────

@contextmanager
def disable_gates_at_inference(model: nn.Module):
    """
    Temporarily force eye and mouth gates to 1.0 (equivalent to no gating).

    The original gate modules are restored after the context exits.

    Parameters
    ----------
    model : EnhancedOcclusionAwareTransformer with ``occlusion_gates`` attribute.

    Example
    -------
    >>> with disable_gates_at_inference(my_model):
    ...     df_no_gate = collect_eval_with_occlusion(my_model, test_loader)
    """
    if not hasattr(model, 'occlusion_gates'):
        yield
        return

    gates = model.occlusion_gates
    try:
        eye_orig   = gates['eye_gate']
        mouth_orig = gates['mouth_gate']
    except Exception as e:
        raise AttributeError(
            "Could not access 'eye_gate'/'mouth_gate' in model.occlusion_gates"
        ) from e

    # Legacy mechanism: force the gate MLPs to output 1.0. Covers old
    # checkpoints that rely on multiplicative token gating.
    gates['eye_gate']   = _OnesGate()
    gates['mouth_gate'] = _OnesGate()

    # New mechanism (transformer_enhanced.py V2): signal the model to skip
    # attention-bias gating *and* the gate-conditioned logit bias head.
    # Without this, setting gates to 1.0 still leaves log(1)=0 attention bias
    # (harmless) but the logit bias head would still be evaluated at (1,1)
    # which produces a constant per-class shift that biases predictions.
    prev_flag = getattr(model, '_force_gating_disabled', False)
    try:
        model._force_gating_disabled = True
        yield
    finally:
        gates['eye_gate']   = eye_orig
        gates['mouth_gate'] = mouth_orig
        model._force_gating_disabled = prev_flag


@contextmanager
def force_gate_values(model: nn.Module,
                      eye: Optional[float] = None,
                      mouth: Optional[float] = None):
    """
    Temporarily force eye and/or mouth gate MLP outputs to a constant *val*.

    Because the transformer applies ``final_gate = 0.3 + 0.7 * gate_mlp_output``,
    forcing ``gate_mlp_output = val`` produces ``final_gate = 0.3 + 0.7 * val``.

    Parameters
    ----------
    model  : model with ``occlusion_gates``.
    eye    : constant value for the eye gate MLP output (or None to leave unchanged).
    mouth  : constant value for the mouth gate MLP output (or None to leave unchanged).

    Example
    -------
    >>> with force_gate_values(model, eye=0.0):  # final eye gate ≈ 0.3
    ...     df = collect_eval_with_occlusion(model, test_loader)
    """
    gates    = model.occlusion_gates
    old_eye  = gates['eye_gate']
    old_mouth = gates['mouth_gate']

    if eye   is not None: gates['eye_gate']   = _ConstGate(eye)
    if mouth is not None: gates['mouth_gate'] = _ConstGate(mouth)
    try:
        yield
    finally:
        gates['eye_gate']   = old_eye
        gates['mouth_gate'] = old_mouth


# ─── Sweep ────────────────────────────────────────────────────────────────────

def sweep_forced_gates(model: nn.Module, loader,
                        eye_vals: Tuple[float, ...] = (1.0, 0.6, 0.0),
                        mouth_vals: Tuple[float, ...] = (1.0, 0.6, 0.0)
                        ) -> pd.DataFrame:
    """
    Evaluate accuracy for each forced gate value in *eye_vals* and *mouth_vals*.

    Returns a DataFrame with columns:
        where (eye|mouth), forced_val, final_gate (= 0.3 + 0.7 * forced_val),
        acc_overall_%.
    """
    model.eval()
    rows = []

    for e in eye_vals:
        with force_gate_values(model, eye=e, mouth=None):
            df  = collect_eval_with_occlusion(model, loader)
            acc = float(df['is_correct'].mean() * 100.0) if len(df) else float('nan')
        rows.append({'where': 'eye', 'forced_val': e,
                     'final_gate': 0.3 + 0.7 * e, 'acc_overall_%': acc})

    for m in mouth_vals:
        with force_gate_values(model, eye=None, mouth=m):
            df  = collect_eval_with_occlusion(model, loader)
            acc = float(df['is_correct'].mean() * 100.0) if len(df) else float('nan')
        rows.append({'where': 'mouth', 'forced_val': m,
                     'final_gate': 0.3 + 0.7 * m, 'acc_overall_%': acc})

    return pd.DataFrame(rows)


# ─── Feature Masking ──────────────────────────────────────────────────────────

@torch.no_grad()
def eval_with_feature_mask(model: nn.Module, loader,
                            mask: str = 'eyes',
                            no_gates: bool = False) -> float:
    """
    Zero-out specified feature regions and return overall accuracy (%).

    Parameters
    ----------
    model    : gated or no-gate transformer.
    loader   : DataLoader.
    mask     : one of 'eyes', 'mouth', 'all' — which regions to zero out.
    no_gates : if True, also disable gate MLPs while masking.

    Returns
    -------
    accuracy (float, 0–100).
    """
    model.eval()
    device  = next(model.parameters()).device
    ctx     = disable_gates_at_inference(model) if no_gates else nullcontext()

    correct = total = 0
    with ctx:
        for batch in loader:
            b = move_batch_to_device(batch, device)
            if mask in ('eyes', 'all'):
                for r in ['left_eye', 'right_eye']:
                    b['features'][r] = torch.zeros_like(b['features'][r])
            if mask in ('mouth', 'all'):
                b['features']['mouth'] = torch.zeros_like(b['features']['mouth'])
            out  = model(b['features'], b['occlusion_info'])
            pred = torch.argmax(out['class_logits'], dim=-1)
            lab  = b['label'].view(-1)
            correct += (pred == lab).sum().item()
            total   += lab.numel()

    return correct / max(total, 1) * 100.0


@torch.no_grad()
def eval_with_mask_and_occinfo(model: nn.Module, loader,
                                mask: str = 'eyes',
                                no_gates: bool = False) -> float:
    """
    Zero-out features **and** set the corresponding occlusion probabilities to 1.0
    (fully occluded), then evaluate accuracy.

    Useful for testing whether the model actually benefits from occlusion
    information when a region is absent.
    """
    model.eval()
    device  = next(model.parameters()).device
    ctx     = disable_gates_at_inference(model) if no_gates else nullcontext()

    correct = total = 0
    with ctx:
        for batch in loader:
            b = move_batch_to_device(batch, device)
            if mask in ('eyes', 'all'):
                for r in ['left_eye', 'right_eye']:
                    b['features'][r] = torch.zeros_like(b['features'][r])
                b['occlusion_info']['eye_occlusion_prob'] = torch.ones_like(
                    b['occlusion_info']['eye_occlusion_prob'])
            if mask in ('mouth', 'all'):
                b['features']['mouth'] = torch.zeros_like(b['features']['mouth'])
                b['occlusion_info']['mouth_occlusion_prob'] = torch.ones_like(
                    b['occlusion_info']['mouth_occlusion_prob'])
            out  = model(b['features'], b['occlusion_info'])
            pred = torch.argmax(out['class_logits'], dim=-1)
            lab  = b['label'].view(-1)
            correct += (pred == lab).sum().item()
            total   += lab.numel()

    return correct / max(total, 1) * 100.0


# ─── Binned Evaluation with Gates Disabled ────────────────────────────────────

@torch.no_grad()
def evaluate_bins_with_gates_disabled(model: nn.Module, loader,
                                       bin_edges=None) -> dict:
    """
    Run collect_eval_with_occlusion with gates disabled and report accuracy
    broken down by occlusion bins.

    Parameters
    ----------
    model      : gated transformer.
    loader     : DataLoader.
    bin_edges  : list of edges for pd.cut; defaults to [-0.01, 0.3, 0.7, 1.01].

    Returns
    -------
    dict with keys: df, overall_acc, acc_eye, acc_mouth.
    """
    if bin_edges is None:
        bin_edges = [-0.01, 0.3, 0.7, 1.01]
    bin_labels = ['low', 'med', 'high']

    with disable_gates_at_inference(model):
        df = collect_eval_with_occlusion(model, loader)

    df = df.copy()
    df['eye_bin']   = pd.cut(df['eye_occ'],   bins=bin_edges, labels=bin_labels)
    df['mouth_bin'] = pd.cut(df['mouth_occ'], bins=bin_edges, labels=bin_labels)

    overall   = df['is_correct'].mean() * 100.0 if len(df) else float('nan')
    acc_eye   = df.groupby('eye_bin')['is_correct'].mean().reindex(bin_labels) * 100.0
    acc_mouth = df.groupby('mouth_bin')['is_correct'].mean().reindex(bin_labels) * 100.0
    cnt_eye   = df['eye_bin'].value_counts().reindex(bin_labels)
    cnt_mouth = df['mouth_bin'].value_counts().reindex(bin_labels)

    print(f'\n[No-Gate Ablation] Overall accuracy: {overall:.2f}% (N={len(df)})')
    print('Accuracy by eye occlusion bin (%):\n' + acc_eye.round(2).to_string())
    print('Counts:\n' + cnt_eye.to_string())
    print('\nAccuracy by mouth occlusion bin (%):\n' + acc_mouth.round(2).to_string())
    print('Counts:\n' + cnt_mouth.to_string())

    return {'df': df, 'overall_acc': overall,
            'acc_eye': acc_eye, 'acc_mouth': acc_mouth}
