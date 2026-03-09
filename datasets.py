"""
datasets.py
===========
PyTorch Dataset and DataLoader utilities for the DOFG-DMS pipeline.

Classes / Functions
-------------------
DriverStateDataset            — Dataset backed by pre-extracted feature dicts.
prepare_tiny_transformer_training_data — Convert pipeline output to sample dicts.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ─── Data Preparation ────────────────────────────────────────────────────────

def prepare_tiny_transformer_training_data(
    processed_results,
    label_map: Optional[Dict[str, int]] = None,
    require_success: bool = True,
    skip_unknown: bool = True,
    expect_dim: int = 512,
) -> List[Dict]:
    """
    Convert the list returned by ``process_sample_frames_complete_pipeline_batched``
    into sample dicts suitable for DriverStateDataset.

    Parameters
    ----------
    processed_results : list of frame-result dicts from the pipeline.
    label_map : class-name → integer label map.
    require_success : skip frames where any pipeline phase failed.
    skip_unknown : skip frames whose class_label is not in label_map.
    expect_dim : expected feature vector length (512 for ResNet-34).

    Returns
    -------
    List of sample dicts, each with keys:
        frame_id, features, occlusion_info, label, class_name,
        ground_truth, meta.
    """
    print('PREPARING TINY TRANSFORMER DATA')

    if label_map is None:
        label_map = {'EyeClosed': 0, 'Yawn': 1, 'Neutral': 2}

    training_samples: List[Dict] = []
    unknown_labels: set = set()

    for result in processed_results:
        if require_success and not (
            result.get('phase1_success') and
            result.get('phase2_success') and
            result.get('phase3_success')
        ):
            continue

        frame_data = result.get('frame_data', {})
        features   = result.get('features', {})
        oc_raw     = result.get('occlusion_analysis', {})

        if isinstance(oc_raw, dict) and 'occlusion_analysis' in oc_raw:
            oc_raw = oc_raw['occlusion_analysis']

        annotation = frame_data.get('annotation')
        if annotation is None:
            continue

        class_label = getattr(annotation, 'class_label', None)
        if class_label not in label_map:
            unknown_labels.add(class_label)
            if skip_unknown:
                continue
        label_idx = label_map.get(class_label, 0)

        def to_vec(x):
            if x is None:
                return None
            return np.asarray(x, dtype=np.float32)

        feature_dict = {
            'face':      to_vec(features.get('face_features')),
            'left_eye':  to_vec(features.get('left_eye_features')),
            'right_eye': to_vec(features.get('right_eye_features')),
            'mouth':     to_vec(features.get('mouth_features')),
        }

        if expect_dim is not None:
            bad = [k for k, v in feature_dict.items()
                   if v is None or v.ndim != 1 or v.shape[0] != expect_dim]
            if bad:
                continue

        eye_prob   = float(oc_raw.get('eye_occlusion_prob', 0.0))
        mouth_prob = float(oc_raw.get('mouth_occlusion_prob', 0.0))
        occlusion_info = {
            'eye_occlusion_prob':   eye_prob,
            'mouth_occlusion_prob': mouth_prob,
        }

        training_samples.append({
            'frame_id':      result.get('frame_id'),
            'features':      feature_dict,
            'occlusion_info': occlusion_info,
            'label':         label_idx,
            'class_name':    class_label,
            'ground_truth': {
                'eyes_occluded': getattr(annotation, 'eyes_occluded_prior', False),
                'mouth_occluded': getattr(annotation, 'mouth_occluded_prior', False),
                'eyes_state':    getattr(annotation, 'eyes_state', 'unknown'),
            },
            'meta': {
                'video_key':    frame_data.get('video_key'),
                'frame_number': getattr(annotation, 'frame', None),
            },
        })

    class_dist: Dict[str, int] = {}
    for s in training_samples:
        class_dist[s['class_name']] = class_dist.get(s['class_name'], 0) + 1
    print(f'  Class distribution: {class_dist}')
    if unknown_labels:
        print(f'  Unknown labels skipped: {sorted(str(u) for u in unknown_labels)}')

    return training_samples


# ─── Dataset ─────────────────────────────────────────────────────────────────

class DriverStateDataset(Dataset):
    """
    Dataset for driver-state classification with occlusion awareness.

    Accepts samples produced either by
    ``prepare_tiny_transformer_training_data`` (from the phase-based pipeline)
    or by ``extract_features_stratified`` / ``extract_features_with_augmentation``
    (from the direct feature extraction pipeline).

    Parameters
    ----------
    training_samples : list of sample dicts.
    device : target device string (informational only; tensors are created on CPU
             and moved to the device by the trainer's collate / batch move).
    """

    CLASS_NAMES = ['EyeClosed', 'Yawn', 'Neutral']

    def __init__(self, training_samples: List[Dict], device: str = 'cpu'):
        self.samples    = training_samples
        self.device     = device

        print(f'DriverStateDataset initialized')
        print(f'  Samples : {len(training_samples)}')
        print(f'  Classes : {self.CLASS_NAMES}')
        print(f'  Device  : {device}')

        if training_samples:
            self._compute_statistics()

    def _compute_statistics(self):
        class_counts: Dict[str, int] = {}
        eye_sum = mouth_sum = 0.0
        for s in self.samples:
            class_counts[s['class_name']] = class_counts.get(s['class_name'], 0) + 1
            eye_sum   += s['occlusion_info']['eye_occlusion_prob']
            mouth_sum += s['occlusion_info']['mouth_occlusion_prob']
        n = len(self.samples)
        print(f'  Class distribution: {class_counts}')
        print(f'  Avg occlusion — Eyes: {eye_sum/n:.3f}, Mouth: {mouth_sum/n:.3f}')
        self.class_counts = class_counts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        features_dict = {
            region: (torch.tensor(arr, dtype=torch.float32)
                     if not isinstance(arr, torch.Tensor) else arr.float())
            for region, arr in sample['features'].items()
        }

        occlusion_info = dict(sample['occlusion_info'])

        label = torch.tensor(sample['label'], dtype=torch.long)
        eye_occ = sample.get('gt_eye_occ', occlusion_info['eye_occlusion_prob'])
        mouth_occ = sample.get('gt_mouth_occ', occlusion_info['mouth_occlusion_prob'])
        occ_targets = torch.tensor([eye_occ, mouth_occ], dtype=torch.float32)

        return {
            'features':          features_dict,
            'occlusion_info':    occlusion_info,
            'label':             label,
            'occlusion_targets': occ_targets,
            'class_name':        sample['class_name'],
            'frame_id':          sample.get('frame_id', -1),
            'ground_truth':      sample.get('ground_truth', {}),
        }

    # ── Collate ───────────────────────────────────────────────────────────────

    def _safe_stack(self, tensors):
        shapes = [tuple(t.shape) for t in tensors]
        return torch.stack(tensors, 0) if len(set(shapes)) == 1 else tensors

    def collate_samples(self, batch: List[Dict]) -> Dict:
        """Custom collate function — use as DataLoader's collate_fn."""
        regions = batch[0]['features'].keys()
        out: Dict = {}

        out['features'] = {
            r: self._safe_stack([b['features'][r] for b in batch])
            for r in regions
        }
        out['occlusion_info'] = {
            'eye_occlusion_prob': torch.tensor(
                [b['occlusion_info']['eye_occlusion_prob'] for b in batch],
                dtype=torch.float32),
            'mouth_occlusion_prob': torch.tensor(
                [b['occlusion_info']['mouth_occlusion_prob'] for b in batch],
                dtype=torch.float32),
        }
        out['label']             = torch.stack([b['label'] for b in batch], 0)
        out['occlusion_targets'] = torch.stack([b['occlusion_targets'] for b in batch], 0)
        out['class_name']        = [b['class_name'] for b in batch]
        out['frame_id']          = torch.tensor([b['frame_id'] for b in batch],
                                                dtype=torch.long)
        out['ground_truth']      = [b['ground_truth'] for b in batch]
        return out

    # ── DataLoader factory ────────────────────────────────────────────────────

    def create_dataloader(self, batch_size: int = 32,
                          shuffle: bool = True,
                          drop_last: bool = False) -> DataLoader:
        """Create a DataLoader backed by this dataset."""
        is_cuda   = torch.cuda.is_available()
        num_workers       = 2 if is_cuda else 0
        pin_memory        = bool(is_cuda)
        persistent_workers= bool(is_cuda and num_workers > 0)

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=self.collate_samples,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    def get_sample_by_class(self, class_name: str) -> Optional[Dict]:
        """Return the first sample with the given class name, or None."""
        for i, sample in enumerate(self.samples):
            if sample['class_name'] == class_name:
                return self[i]
        return None
