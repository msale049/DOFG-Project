"""
clip_sampler.py
===============
Deterministic clip extraction from video annotations.
Supports FPS downsampling, stride-based sampling, and reproducible clip boundaries.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import CSVAnnotation, CLIP_CONFIG, SEED


@dataclass
class ClipInfo:
    """One clip: contiguous frame indices from a video."""
    video_key: str
    subject: str
    clip_start: int  # index in downsampled annotation list
    frame_indices: List[int]  # indices in original annotation list
    frame_numbers: List[int]  # actual frame numbers for VideoCapture (from ann.frame)
    majority_class: str
    fps: float
    T: int
    stride: int


def downsample_annotations(
    annotations: List[CSVAnnotation],
    fps_src: float,
    fps_target: float,
) -> List[Tuple[int, CSVAnnotation]]:
    """
    Downsample annotations to target FPS.
    Returns list of (original_index, annotation) for frames to keep.
    """
    if fps_target >= fps_src:
        return [(i, a) for i, a in enumerate(annotations)]

    step = max(1, int(round(fps_src / fps_target)))
    return [(i, a) for i, a in enumerate(annotations) if i % step == 0]


def extract_clips(
    annotations: List[CSVAnnotation],
    video_key: str,
    subject: str,
    T: int,
    stride: int,
    fps_src: float,
    fps_target: float,
    label_map: Optional[Dict[str, int]] = None,
) -> List[ClipInfo]:
    """
    Extract non-overlapping or overlapping clips from annotations.

    Parameters
    ----------
    annotations : list of CSVAnnotation (frame-ordered)
    video_key : e.g. "Sub1_sub1_video"
    subject : e.g. "Sub1"
    T : clip length (number of frames at target FPS)
    stride : step between clip starts
    fps_src : original video FPS
    fps_target : target FPS for clip boundaries
    label_map : for filtering; if None, all frames kept

    Returns
    -------
    List of ClipInfo
    """
    downsampled = downsample_annotations(annotations, fps_src, fps_target)
    if len(downsampled) < T:
        return []

    clips: List[ClipInfo] = []
    for start in range(0, len(downsampled) - T + 1, stride):
        window = downsampled[start : start + T]
        orig_indices = [downsampled[i][0] for i in range(start, start + T)]
        anns = [w[1] for w in window]

        if label_map is not None:
            valid = [a for a in anns if a.class_label in label_map]
            if len(valid) < T // 2:
                continue

        # Majority class
        classes = [a.class_label for a in anns if label_map is None or a.class_label in label_map]
        if not classes:
            continue
        majority = max(set(classes), key=classes.count)

        frame_numbers = [a.frame for a in anns]

        clips.append(ClipInfo(
            video_key=video_key,
            subject=subject,
            clip_start=start,
            frame_indices=orig_indices,
            frame_numbers=frame_numbers,
            majority_class=majority,
            fps=fps_target,
            T=T,
            stride=stride,
        ))
    return clips


def extract_clips_for_videos(
    csv_data: Dict,
    T: int = 32,
    stride: int = 16,
    fps_src: float = 29.76,
    fps_target: float = 15,
    label_map: Optional[Dict[str, int]] = None,
    video_keys: Optional[List[str]] = None,
    stride_per_video: Optional[Dict[str, int]] = None,
) -> Dict[str, List[ClipInfo]]:
    """
    Extract clips for videos in csv_data.

    Use stride=16 for train/val (overlapping), stride=32 for test (non-overlapping).
    stride_per_video: optional dict video_key -> stride for per-split stride.

    Returns
    -------
    dict mapping video_key -> list of ClipInfo
    """
    if label_map is None:
        label_map = {'EyeClosed': 0, 'Yawn': 1, 'Neutral': 2}

    result: Dict[str, List[ClipInfo]] = {}
    keys = video_keys if video_keys is not None else list(csv_data.keys())
    for video_key in keys:
        if video_key not in csv_data:
            continue
        meta = csv_data[video_key]
        anns = meta['annotations']
        subject = meta['subject']
        s = stride_per_video.get(video_key, stride) if stride_per_video else stride
        clips = extract_clips(
            anns, video_key, subject, T, s,
            fps_src, fps_target, label_map,
        )
        result[video_key] = clips
    return result
