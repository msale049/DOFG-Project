"""
metrics_utils.py
================
Shared metric, uncertainty, and cross-validation aggregation helpers.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

try:
    from scipy.stats import binomtest, t as student_t
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_SCIPY = False

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_SKLEARN = False


def _to_numpy_int(values: Sequence[int]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.int64)
    return arr.reshape(-1)


def _safe_float(value) -> float:
    if value is None:
        return float('nan')
    return float(value)


def _bootstrap_ci(
    values_a: np.ndarray,
    values_b: Optional[np.ndarray],
    metric_fn: Callable[[np.ndarray, Optional[np.ndarray]], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Generic paired/non-paired bootstrap CI.

    Parameters
    ----------
    values_a : np.ndarray
        Base array. For paired metrics this defines the bootstrap index base.
    values_b : np.ndarray or None
        Optional paired companion array.
    metric_fn : callable
        Called as metric_fn(sample_a, sample_b).
    """
    n = len(values_a)
    if n == 0:
        return {
            'mean': float('nan'),
            'std': float('nan'),
            'ci_low': float('nan'),
            'ci_high': float('nan'),
        }

    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample_a = values_a[idx]
        sample_b = values_b[idx] if values_b is not None else None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            stats[i] = metric_fn(sample_a, sample_b)

    low = np.percentile(stats, 100 * (alpha / 2))
    high = np.percentile(stats, 100 * (1 - alpha / 2))
    return {
        'mean': float(np.mean(stats)),
        'std': float(np.std(stats, ddof=1)) if len(stats) > 1 else 0.0,
        'ci_low': float(low),
        'ci_high': float(high),
    }


def _metric_accuracy_pct(labels: np.ndarray, preds: np.ndarray) -> float:
    if len(labels) == 0:
        return float('nan')
    if _HAS_SKLEARN:
        return float(accuracy_score(labels, preds) * 100.0)
    return float(np.mean(labels == preds) * 100.0)


def _metric_balanced_accuracy_pct(labels: np.ndarray, preds: np.ndarray) -> float:
    if len(labels) == 0:
        return float('nan')
    if _HAS_SKLEARN:
        return float(balanced_accuracy_score(labels, preds) * 100.0)

    recalls = []
    for cls in np.unique(labels):
        mask = labels == cls
        recalls.append(float(np.mean(preds[mask] == labels[mask])))
    return float(np.mean(recalls) * 100.0) if recalls else float('nan')


def _metric_macro_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    if len(labels) == 0:
        return float('nan')
    if _HAS_SKLEARN:
        _, _, f1, _ = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0)
        return float(f1)
    return float('nan')


def _metric_weighted_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    if len(labels) == 0:
        return float('nan')
    if _HAS_SKLEARN:
        _, _, f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0)
        return float(f1)
    return float('nan')


def compute_classification_metrics(
    labels: Sequence[int],
    preds: Sequence[int],
    label_names: Optional[Sequence[str]] = None,
) -> Dict:
    """
    Compute clean classification metrics and per-class summaries.

    Returns accuracy / balanced_accuracy in percent and precision / recall /
    F1 metrics on the [0, 1] scale for consistency with sklearn.
    """
    labels_np = _to_numpy_int(labels)
    preds_np = _to_numpy_int(preds)

    if label_names is None:
        label_ids = sorted(set(labels_np.tolist()) | set(preds_np.tolist()))
        label_names = [str(i) for i in label_ids]
    else:
        label_ids = list(range(len(label_names)))

    result = {
        'n_samples': int(len(labels_np)),
        'accuracy': _metric_accuracy_pct(labels_np, preds_np),
        'balanced_accuracy': _metric_balanced_accuracy_pct(labels_np, preds_np),
        'precision': float('nan'),
        'recall': float('nan'),
        'f1': float('nan'),
        'macro_precision': float('nan'),
        'macro_recall': float('nan'),
        'macro_f1': float('nan'),
        'per_class': {},
        'confusion_matrix': [],
        'label_names': list(label_names),
    }

    if len(labels_np) == 0:
        return result

    if _HAS_SKLEARN:
        w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(
            labels_np, preds_np, average='weighted', zero_division=0)
        m_prec, m_rec, m_f1, _ = precision_recall_fscore_support(
            labels_np, preds_np, average='macro', zero_division=0)
        per_prec, per_rec, per_f1, per_sup = precision_recall_fscore_support(
            labels_np, preds_np, labels=label_ids, average=None, zero_division=0)
        cm = confusion_matrix(labels_np, preds_np, labels=label_ids)

        result.update({
            'precision': float(w_prec),
            'recall': float(w_rec),
            'f1': float(w_f1),
            'macro_precision': float(m_prec),
            'macro_recall': float(m_rec),
            'macro_f1': float(m_f1),
            'confusion_matrix': cm.tolist(),
        })

        for idx, name in enumerate(label_names):
            result['per_class'][name] = {
                'precision': float(per_prec[idx]),
                'recall': float(per_rec[idx]),
                'f1': float(per_f1[idx]),
                'support': int(per_sup[idx]),
            }
    return result


def compute_classification_uncertainty(
    labels: Sequence[int],
    preds: Sequence[int],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Bootstrap confidence intervals for clean classification metrics."""
    labels_np = _to_numpy_int(labels)
    preds_np = _to_numpy_int(preds)

    metrics = {
        'accuracy': lambda a, b: _metric_accuracy_pct(a, b),
        'balanced_accuracy': lambda a, b: _metric_balanced_accuracy_pct(a, b),
        'macro_f1': lambda a, b: _metric_macro_f1(a, b),
        'weighted_f1': lambda a, b: _metric_weighted_f1(a, b),
    }

    out = {}
    for name, fn in metrics.items():
        out[name] = _bootstrap_ci(
            labels_np, preds_np, fn, n_boot=n_boot, alpha=alpha, seed=seed)
    return out


def compute_paired_binary_statistics(
    correct_on: Sequence[int],
    correct_off: Sequence[int],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute paired ON/OFF delta statistics for stress-test comparisons.

    Returns delta in percentage points, bootstrap CI, and exact McNemar-style
    binomial p-value over discordant pairs.
    """
    on_np = _to_numpy_int(correct_on)
    off_np = _to_numpy_int(correct_off)
    if len(on_np) != len(off_np):
        raise ValueError('correct_on and correct_off must have the same length')

    delta_pp = float((np.mean(on_np) - np.mean(off_np)) * 100.0) if len(on_np) else float('nan')
    wins_on = int(np.sum((on_np == 1) & (off_np == 0)))
    wins_off = int(np.sum((on_np == 0) & (off_np == 1)))
    n_discordant = wins_on + wins_off

    if n_discordant == 0:
        p_value = 1.0
    elif _HAS_SCIPY:
        p_value = float(binomtest(wins_on, n=n_discordant, p=0.5, alternative='two-sided').pvalue)
    else:  # pragma: no cover - scipy available in project env
        p_value = float('nan')

    boot = _bootstrap_ci(
        on_np,
        off_np,
        lambda a, b: (np.mean(a) - np.mean(b)) * 100.0,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
    )

    return {
        'delta_pp': delta_pp,
        'delta_ci_low': boot['ci_low'],
        'delta_ci_high': boot['ci_high'],
        'delta_boot_mean': boot['mean'],
        'delta_boot_std': boot['std'],
        'wins_on': wins_on,
        'wins_off': wins_off,
        'n_discordant': n_discordant,
        'p_value_mcnemar': p_value,
    }


def summarize_fold_metric(values: Iterable[float], confidence: float = 0.95) -> Dict[str, float]:
    """Summarize a list of fold-level metrics with mean/std and t-based CI."""
    arr = np.asarray([float(v) for v in values if v is not None and not math.isnan(float(v))], dtype=np.float64)
    if len(arr) == 0:
        return {
            'n': 0,
            'mean': float('nan'),
            'std': float('nan'),
            'ci_low': float('nan'),
            'ci_high': float('nan'),
            'min': float('nan'),
            'max': float('nan'),
        }

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    if len(arr) > 1 and _HAS_SCIPY and std > 0:
        sem = std / math.sqrt(len(arr))
        alpha = 1.0 - confidence
        ci_low, ci_high = student_t.interval(confidence, df=len(arr) - 1, loc=mean, scale=sem)
        ci_low = float(ci_low)
        ci_high = float(ci_high)
    else:
        ci_low = ci_high = mean

    return {
        'n': int(len(arr)),
        'mean': mean,
        'std': std,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
    }
