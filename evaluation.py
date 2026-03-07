"""
evaluation.py
=============
Evaluation utilities for the DOFG-DMS pipeline.

Functions
---------
compute_metrics_on_loader     — Loss, accuracy, precision, recall, F1 on a loader.
plot_curves                   — Training/validation loss and accuracy curves.
collect_eval_with_occlusion   — Per-sample DataFrame with gates, probs, occlusion.
occlusion_accuracy_report     — Accuracy and F1 broken down by occlusion bins.
save_test_predictions         — Incrementally save predictions to CSV.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

# Optional heavy imports — only used when called
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
    )
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

from utils import move_batch_to_device


# ─── Metrics on Loader ───────────────────────────────────────────────────────

def compute_metrics_on_loader(trainer, loader,
                               compute_loss: bool = True) -> Dict:
    """
    Compute loss, accuracy, precision, recall, F1 on *loader* using
    *trainer.model*.

    Returns
    -------
    dict with keys: loss, accuracy, precision, recall, f1, preds, labels.
    """
    model     = trainer.model
    device    = trainer.device
    criterion = trainer.classification_criterion
    model.eval()

    all_preds, all_labels = [], []
    total_loss, n = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            batch   = trainer._move_batch_to_device(batch)
            outputs = model(batch['features'], batch['occlusion_info'])
            logits  = outputs['class_logits']
            preds   = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            labels  = batch['label'].view(-1).detach().cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

            if compute_loss:
                loss = criterion(logits, batch['label'].view(-1))
                total_loss += float(loss.item()) * labels.shape[0]
                n += labels.shape[0]

    if _HAS_SKLEARN:
        acc = accuracy_score(all_labels, all_preds) * 100 if all_labels else 0.0
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0)
    else:
        acc  = float(np.mean(np.array(all_preds) == np.array(all_labels)) * 100)
        prec = rec = f1 = float('nan')

    val_loss = (total_loss / max(n, 1)) if compute_loss else None

    return {
        'loss':      val_loss,
        'accuracy':  acc,
        'precision': float(prec),
        'recall':    float(rec),
        'f1':        float(f1),
        'preds':     all_preds,
        'labels':    all_labels,
    }


# ─── Training Curves ─────────────────────────────────────────────────────────

def plot_curves(train_loss: List[float], val_loss: Optional[List[float]],
                val_acc: List[float],
                val_p: Optional[List[float]] = None,
                val_r: Optional[List[float]] = None,
                val_f1: Optional[List[float]] = None,
                title_suffix: str = '') -> None:
    """Plot training/validation loss, accuracy and (optionally) P/R/F1 curves."""
    if not _HAS_PLT:
        print('matplotlib not available, skipping plot_curves.')
        return

    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label='Train loss')
    if val_loss is not None:
        plt.plot(epochs, val_loss, label='Val loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(f'Loss vs Epoch {title_suffix}')
    plt.legend(); plt.grid(True); plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, val_acc, label='Val accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
    plt.title(f'Val Accuracy vs Epoch {title_suffix}')
    plt.legend(); plt.grid(True); plt.show()

    if val_p is not None and val_r is not None and val_f1 is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, val_p,  label='Precision (weighted)')
        plt.plot(epochs, val_r,  label='Recall (weighted)')
        plt.plot(epochs, val_f1, label='F1 (weighted)')
        plt.xlabel('Epoch'); plt.ylabel('Score'); plt.ylim(0, 1.0)
        plt.title(f'Val Precision/Recall/F1 vs Epoch {title_suffix}')
        plt.legend(); plt.grid(True); plt.show()


# ─── Per-sample Evaluation DataFrame ─────────────────────────────────────────

def _safe_move_batch_to_device(batch: Dict, device) -> Dict:
    """Move batch to device using the utils helper."""
    return move_batch_to_device(batch, device)


@torch.no_grad()
def collect_eval_with_occlusion(model: torch.nn.Module,
                                loader) -> pd.DataFrame:
    """
    Iterate *loader* and collect per-sample predictions, labels, occlusion
    probabilities, gate values, and attention weights.

    Parameters
    ----------
    model  : the gated (or no-gate) transformer, already on its device.
    loader : DataLoader backed by DriverStateDataset.

    Returns
    -------
    DataFrame with columns:
        true, pred, is_correct, class_name,
        eye_occ, mouth_occ,
        gate_face, gate_eye, gate_mouth, conf.
    """
    device = next(model.parameters()).device
    model.eval()
    rows = []

    for batch in loader:
        batch_dev = _safe_move_batch_to_device(batch, device)
        outputs   = model(batch_dev['features'], batch_dev['occlusion_info'],
                          return_attention=True)

        preds  = outputs['predicted_class'].view(-1).detach().cpu().numpy()
        labels = batch_dev['label'].view(-1).detach().cpu().numpy()
        probs  = (outputs['class_probs'].detach().cpu().numpy()
                  if 'class_probs' in outputs else None)
        gates  = (outputs['gate_factors'].detach().cpu().numpy()
                  if 'gate_factors' in outputs else None)

        eye_occ   = batch_dev['occlusion_info']['eye_occlusion_prob'].view(-1).detach().cpu().numpy()
        mouth_occ = batch_dev['occlusion_info']['mouth_occlusion_prob'].view(-1).detach().cpu().numpy()

        names = batch.get('class_name')
        if isinstance(names, list):
            class_names = [str(x) for x in names]
        else:
            class_names = [str(int(y)) for y in labels]

        bsz = len(labels)
        for i in range(bsz):
            gate_face  = float(gates[i, 0]) if gates is not None else float('nan')
            gate_eye   = float(np.mean(gates[i, 1:3])) if gates is not None else float('nan')
            gate_mouth = float(gates[i, 3]) if gates is not None else float('nan')
            conf       = float(probs[i, int(preds[i])]) if probs is not None else float('nan')

            rows.append({
                'true':       int(labels[i]),
                'pred':       int(preds[i]),
                'is_correct': bool(int(preds[i]) == int(labels[i])),
                'class_name': class_names[i],
                'eye_occ':    float(eye_occ[i]),
                'mouth_occ':  float(mouth_occ[i]),
                'gate_face':  gate_face,
                'gate_eye':   gate_eye,
                'gate_mouth': gate_mouth,
                'conf':       conf,
            })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print('No samples collected. Check that the loader has data and the '
              'model is on the correct device.')
    return df


# ─── Occlusion-binned Accuracy Report ────────────────────────────────────────

def occlusion_accuracy_report(df: pd.DataFrame,
                               n_bins: int = 3,
                               bin_labels: Optional[List[str]] = None) -> Dict:
    """
    Compute accuracy and weighted F1 by eye/mouth occlusion bins.

    Parameters
    ----------
    df : output of collect_eval_with_occlusion.
    n_bins : number of quantile-based bins (default 3 → low/med/high).
    bin_labels : custom labels for bins; defaults to ['low', 'med', 'high'].

    Returns
    -------
    dict with keys: overall_acc, acc_eye, acc_mouth, f1_eye, f1_mouth.
    """
    if bin_labels is None:
        bin_labels = ['low', 'med', 'high']

    overall_acc = df['is_correct'].mean() * 100.0 if len(df) else float('nan')
    print(f'Overall accuracy: {overall_acc:.2f}%  (N={len(df)})')

    df = df.copy()
    df['eye_bin']   = pd.qcut(df['eye_occ'],   q=n_bins, labels=bin_labels, duplicates='drop')
    df['mouth_bin'] = pd.qcut(df['mouth_occ'], q=n_bins, labels=bin_labels, duplicates='drop')

    acc_eye   = df.groupby('eye_bin')['is_correct'].mean().reindex(bin_labels) * 100.0
    acc_mouth = df.groupby('mouth_bin')['is_correct'].mean().reindex(bin_labels) * 100.0

    print('\nAccuracy by eye occlusion bin (%):')
    print(acc_eye.round(2).to_string())
    print('\nAccuracy by mouth occlusion bin (%):')
    print(acc_mouth.round(2).to_string())

    def _weighted_f1(sub: pd.DataFrame) -> float:
        if len(sub) == 0 or not _HAS_SKLEARN:
            return float('nan')
        _, _, f1, _ = precision_recall_fscore_support(
            sub['true'].to_numpy(), sub['pred'].to_numpy(),
            average='weighted', zero_division=0)
        return float(f1)

    f1_eye   = df.groupby('eye_bin').apply(_weighted_f1).reindex(bin_labels)
    f1_mouth = df.groupby('mouth_bin').apply(_weighted_f1).reindex(bin_labels)

    if _HAS_PLT:
        sns.set_style('whitegrid')
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(bin_labels, acc_eye.fillna(0.0))
        axes[0].set_title('Accuracy vs Eye Occlusion')
        axes[0].set_ylim(0, 100); axes[0].set_ylabel('% correct')
        axes[1].bar(bin_labels, acc_mouth.fillna(0.0))
        axes[1].set_title('Accuracy vs Mouth Occlusion')
        axes[1].set_ylim(0, 100)
        plt.tight_layout(); plt.show()

    return {
        'overall_acc': overall_acc,
        'acc_eye':     acc_eye,
        'acc_mouth':   acc_mouth,
        'f1_eye':      f1_eye,
        'f1_mouth':    f1_mouth,
        'df_binned':   df,
    }


# ─── Save Predictions ─────────────────────────────────────────────────────────

def save_test_predictions(df: pd.DataFrame, path: str,
                          mode: str = 'w') -> None:
    """
    Save a predictions DataFrame to CSV.

    Parameters
    ----------
    df   : output of collect_eval_with_occlusion.
    path : output CSV file path.
    mode : 'w' to overwrite, 'a' to append.
    """
    header = (mode == 'w') or (not os.path.exists(path))
    df.to_csv(path, index=False, mode=mode, header=header)
    print(f'Saved {len(df)} rows to {path}')
