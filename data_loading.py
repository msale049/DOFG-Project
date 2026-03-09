"""
data_loading.py
===============
Utilities for loading DMD CSV annotations and splitting video IDs into
train / validation / test subsets.

Functions
---------
load_csv_video_data   — Parse all subject CSVs into CSVAnnotation dicts.
split_video_ids        — Deterministic video-level train/val/test split.
sample_frames_for_audit — Stratified frame sampling with occluded-priority.
"""

import os
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import CSVAnnotation


# ─── CSV Loading ─────────────────────────────────────────────────────────────

def load_csv_video_data(dataset_path: str,
                        filter_eye_states: bool = True) -> Dict:
    """
    Walk *dataset_path*, find every ``(subject_folder/*.mp4, *.csv)`` pair,
    and return a dict keyed by ``"{subject}_{video_name}"``.

    Parameters
    ----------
    dataset_path : str
        Root folder containing one sub-folder per subject.
    filter_eye_states : bool
        If True, rows with eye_state in {"opening", "closing", "undefined"}
        are dropped before creating annotations.

    Returns
    -------
    dict mapping video_key → {
        video_path, csv_path, annotations, total_frames, subject
    }
    """
    print(f'Loading CSV video data from: {dataset_path}')
    csv_metadata: Dict = {}
    excluded_states = {'opening', 'closing', 'undefined'}

    for subject_folder in sorted(os.listdir(dataset_path)):
        subject_path = os.path.join(dataset_path, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        video_files = [f for f in os.listdir(subject_path) if f.endswith('.mp4')]
        csv_files   = [f for f in os.listdir(subject_path) if f.endswith('.csv')]

        for video_file in video_files:
            video_name = os.path.splitext(video_file)[0]
            csv_file = f'{video_name}.csv'
            if csv_file not in csv_files:
                # Try alternate naming: Sub11_video.mp4 -> Sub11_csv.csv
                alt_csv = video_name.rsplit('_', 1)[0] + '_csv.csv' if '_' in video_name else None
                if alt_csv and alt_csv in csv_files:
                    csv_file = alt_csv
                elif len(csv_files) == 1:
                    csv_file = csv_files[0]
                else:
                    continue

            video_path = os.path.join(subject_path, video_file)
            csv_path   = os.path.join(subject_path, csv_file)

            try:
                df = pd.read_csv(csv_path)
                if filter_eye_states:
                    before = len(df)
                    df = df[~df['eyes_state'].astype(str).str.lower().isin(excluded_states)]
                    filtered = before - len(df)
                else:
                    filtered = 0

                annotations = []
                for _, row in df.iterrows():
                    ann = CSVAnnotation(
                        frame=int(row['frame']),
                        timestamp=float(row['timestamp_s']),
                        class_label=row['class'],
                        variant=row['variant'],
                        eyes_state=row['eyes_state'],
                        yawn_with_hand=bool(row['yawn_with_hand']),
                        yawn_without_hand=bool(row['yawn_without_hand']),
                        eyes_occluded_prior=bool(row['eyes_occluded_prior']),
                        mouth_occluded_prior=bool(row['mouth_occluded_prior']),
                        glasses=bool(row['glasses']),
                    )
                    annotations.append(ann)

                key = f'{subject_folder}_{video_name}'
                csv_metadata[key] = {
                    'video_path': video_path,
                    'csv_path': csv_path,
                    'annotations': annotations,
                    'total_frames': len(annotations),
                    'subject': subject_folder,
                }
                print(f'  {key}: {len(annotations)} frames'
                      + (f' (filtered {filtered})' if filtered else ''))

            except Exception as e:
                print(f'  Error loading {csv_file}: {e}')

    return csv_metadata


# ─── Video-level Splitting ────────────────────────────────────────────────────

def split_video_ids(csv_data: Dict, num_test: int = 3,
                    num_val: int = 0, seed: int = 42) -> Dict:
    """
    Deterministic video-level split matching the IV_notebook strategy.

    Returns
    -------
    dict with keys 'train', 'val', 'test' → each a list of video_key strings.
    Note: 'val' defaults to the same videos as 'train' when num_val=0.
    """
    vids = sorted(csv_data.keys())
    rng  = random.Random(seed)
    rng.shuffle(vids)
    test  = vids[:num_test]
    val   = vids[num_test: num_test + num_val]
    train = vids[num_test + num_val:]
    return {'train': train, 'val': train if num_val == 0 else val, 'test': test}


# ─── Frame Sampling ───────────────────────────────────────────────────────────

def sample_frames_for_audit(
    csv_data: Dict,
    max_per_subject: Optional[int] = None,
    include_all_occluded: bool = True,
    seed: int = 42,
) -> Dict[str, List[CSVAnnotation]]:
    """
    Select a subset of frames per video for audit / stress-testing.

    Parameters
    ----------
    csv_data : output of load_csv_video_data.
    max_per_subject : int or None
        Budget per video.  None → use ALL frames (no sampling).
    include_all_occluded : bool
        When True and the number of occluded frames fits within the budget,
        all occluded frames are kept and the remaining budget is filled with
        randomly sampled non-occluded frames.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict mapping video_key → [CSVAnnotation, …]
    """
    rng = np.random.RandomState(seed)
    selected: Dict[str, List[CSVAnnotation]] = {}

    for video_key, meta in csv_data.items():
        anns = meta['annotations']

        if max_per_subject is None:
            selected[video_key] = sorted(anns, key=lambda a: a.frame)
            print(f'  {video_key}: ALL {len(anns)} frames')
            continue

        occluded     = [a for a in anns if a.eyes_occluded_prior or a.mouth_occluded_prior]
        non_occluded = [a for a in anns
                        if not a.eyes_occluded_prior and not a.mouth_occluded_prior]

        if include_all_occluded and len(occluded) <= max_per_subject:
            budget = max_per_subject - len(occluded)
            if budget < len(non_occluded):
                idx = rng.choice(len(non_occluded), size=budget, replace=False)
                sampled_non_occ = [non_occluded[i] for i in sorted(idx)]
            else:
                sampled_non_occ = non_occluded
            combined = occluded + sampled_non_occ
        else:
            n_occ_target = (max(1, int(max_per_subject * len(occluded) / len(anns)))
                            if occluded else 0)
            n_non_target = max_per_subject - n_occ_target

            if n_occ_target < len(occluded):
                idx = rng.choice(len(occluded), size=n_occ_target, replace=False)
                sampled_occ = [occluded[i] for i in sorted(idx)]
            else:
                sampled_occ = occluded

            if n_non_target < len(non_occluded):
                idx = rng.choice(len(non_occluded), size=n_non_target, replace=False)
                sampled_non_occ = [non_occluded[i] for i in sorted(idx)]
            else:
                sampled_non_occ = non_occluded

            combined = sampled_occ + sampled_non_occ

        combined.sort(key=lambda a: a.frame)
        selected[video_key] = combined

        n_occ = sum(1 for a in combined if a.eyes_occluded_prior or a.mouth_occluded_prior)
        n_non = len(combined) - n_occ
        print(f'  {video_key}: {n_occ} occluded + {n_non} non-occluded = {len(combined)} frames')

    return selected
