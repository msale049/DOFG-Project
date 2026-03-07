"""
split_generator.py
==================
Leakage-safe train/val/test split at clip level.
Supports: fixed split, k-fold, and LOSO (Leave-One-Subject-Out).
"""

import random
from typing import Dict, List, Optional, Tuple

from config import SEED
from clip_sampler import ClipInfo


def get_subject_ids(csv_data: Dict) -> List[str]:
    """Extract unique subject IDs from csv_data."""
    subjects = set()
    for meta in csv_data.values():
        subjects.add(meta['subject'])
    return sorted(subjects)


def create_splits(
    subject_ids: List[str],
    mode: str = 'kfold',
    k: int = 5,
    num_test: int = 5,
    seed: int = SEED,
) -> List[Dict]:
    """
    Create train/test splits. Single entry point for all modes.

    Parameters
    ----------
    subject_ids : sorted list of subject IDs
    mode : 'fixed' | 'kfold' | 'loso'
    k : number of folds (for mode='kfold')
    num_test : subjects held out for test (for mode='fixed')
    seed : random seed

    Returns
    -------
    List of split configs. For 'fixed': length 1. For 'kfold'/'loso': length k or 15.
    Each config: {'train_subjects': [...], 'test_subjects': [...]}
    """
    if mode == 'fixed':
        return [create_fixed_split(subject_ids, num_test=num_test, seed=seed)]
    if mode == 'kfold':
        return create_kfold_folds(subject_ids, k=k, seed=seed)
    if mode == 'loso':
        return create_loso_folds(subject_ids, seed=seed)
    raise ValueError(f"mode must be 'fixed', 'kfold', or 'loso'; got {mode!r}")


def create_loso_folds(subject_ids: List[str], seed: int = SEED) -> List[Dict]:
    """
    Create LOSO folds. Each fold: one subject for test, rest for train.

    Returns
    -------
    List of dicts: {'train_subjects': [...], 'test_subjects': [str]}
    """
    folds = []
    for test_subject in subject_ids:
        train_subjects = [s for s in subject_ids if s != test_subject]
        folds.append({
            'train_subjects': train_subjects,
            'test_subjects': [test_subject],
        })
    return folds


def create_kfold_folds(
    subject_ids: List[str],
    k: int = 5,
    seed: int = SEED,
) -> List[Dict]:
    """
    Create k-fold splits. Subjects are shuffled and divided into k folds.
    Each fold: one fold for test, rest for train.

    Returns
    -------
    List of dicts: {'train_subjects': [...], 'test_subjects': [...]}
    """
    rng = random.Random(seed)
    shuffled = subject_ids.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    if k < 2 or k > n:
        raise ValueError(f"k must be in [2, {n}]; got {k}")

    folds: List[List[str]] = []
    base_size = n // k
    remainder = n % k
    start = 0
    for i in range(k):
        size = base_size + (1 if i < remainder else 0)
        folds.append(shuffled[start : start + size])
        start += size

    result = []
    for i in range(k):
        test_subjects = folds[i]
        train_subjects = [s for j, f in enumerate(folds) if j != i for s in f]
        result.append({
            'train_subjects': train_subjects,
            'test_subjects': test_subjects,
        })
    return result


def create_fixed_split(
    subject_ids: List[str],
    num_test: int = 5,
    seed: int = SEED,
) -> Dict[str, List[str]]:
    """
    Fixed video-level split.

    Returns
    -------
    {'train': [video_keys], 'val': [...], 'test': [video_keys]}
    """
    rng = random.Random(seed)
    shuffled = subject_ids.copy()
    rng.shuffle(shuffled)
    test_subjects = shuffled[:num_test]
    train_subjects = shuffled[num_test:]

    def video_keys_for_subjects(subs: List[str]):
        return [s for s in subs]

    return {
        'train_subjects': train_subjects,
        'test_subjects': test_subjects,
    }


def split_clips_temporal(
    clips: List[ClipInfo],
    val_ratio: float = 0.20,
    seed: int = SEED,
) -> Tuple[List[ClipInfo], List[ClipInfo]]:
    """
    Split clips temporally: first (1-val_ratio) for train, last val_ratio for val.
    No overlap between train and val clips.

    Parameters
    ----------
    clips : list of ClipInfo (must be sorted by clip_start)
    val_ratio : fraction of clips for validation
    seed : unused (deterministic split)

    Returns
    -------
    (train_clips, val_clips)
    """
    if not clips:
        return [], []
    n = len(clips)
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val
    return clips[:n_train], clips[-n_val:]


def get_train_val_test_clips(
    clips_per_video: Dict[str, List[ClipInfo]],
    split_config: Dict,
    val_ratio: float = 0.20,
    seed: int = SEED,
) -> Tuple[List[ClipInfo], List[ClipInfo], List[ClipInfo]]:
    """
    Produce train, val, test clip lists from split config.

    split_config must have:
    - 'train_subjects': list of subject IDs for train
    - 'test_subjects': list of subject IDs for test (or 'test_subject' for single)

    Returns
    -------
    (train_clips, val_clips, test_clips)
    """
    train_subjects = set(split_config['train_subjects'])
    if 'test_subjects' in split_config:
        test_subjects = set(split_config['test_subjects'])
    elif 'test_subject' in split_config:
        test_subjects = {split_config['test_subject']}
    else:
        test_subjects = set()

    train_clips: List[ClipInfo] = []
    val_clips: List[ClipInfo] = []
    test_clips: List[ClipInfo] = []

    for video_key, clips in clips_per_video.items():
        if not clips:
            continue
        subject = clips[0].subject
        if subject in test_subjects:
            test_clips.extend(clips)
        elif subject in train_subjects:
            tr, va = split_clips_temporal(clips, val_ratio, seed)
            train_clips.extend(tr)
            val_clips.extend(va)

    return train_clips, val_clips, test_clips
