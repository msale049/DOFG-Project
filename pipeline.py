"""
pipeline.py
===========
Full-frame processing pipeline and feature-extraction helpers.

Functions
---------
process_sample_frames_complete_pipeline_batched
    Run face detection → feature extraction → occlusion estimation on a
    list of pre-loaded frame dicts.

extract_features_stratified
    Iterate DMD videos (clean), detect faces, extract 512-D features, run
    the occlusion estimator, and return stratified train/val/test sample lists.

extract_features_with_augmentation
    Same as above but applies per-subject synthetic occlusion to training
    frames (Experiment A+).

extract_features_for_clips
    Clip-based extraction with regime-based occlusion (STRATEGY_DESIGN.md).
    Uses clip_sampler, split_generator, occlusion_augmentation.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from config import CSVAnnotation, CLIP_CONFIG, SEED
from clip_sampler import ClipInfo, extract_clips_for_videos
from split_generator import get_train_val_test_clips
from occlusion_augmentation import assign_regime_to_clip, apply_occlusion_to_frame


# ─── Phase-based pipeline (used with process_sample_frames approach) ──────────

def process_sample_frames_complete_pipeline_batched(
    sample_frames: List[Dict],
    face_detector,
    feature_extractor,
    occlusion_detector,
    batch_size: int = 32,
    is_training: bool = False,
) -> List[Dict]:
    """
    Process a list of ``{frame_id, image, annotation, …}`` dicts through all
    three pipeline phases in batches.

    Parameters
    ----------
    sample_frames     : list of frame dicts, each containing at minimum
                        ``frame_id`` and ``image`` (HxWx3 NumPy BGR array).
    face_detector     : FaceDetector instance (dlib or RetinaFace).
    feature_extractor : ResNet34FeatureExtractor instance.
    occlusion_detector: TrainedOcclusionDetector instance.
    batch_size        : number of frames per processing batch.
    is_training       : informational flag (passed to feature extractor when
                        the extractor supports an ``is_training`` argument).

    Returns
    -------
    List of result dicts, one per input frame, with keys:
        frame_id, frame_data, phase1_success, phase2_success,
        phase3_success, features, occlusion_analysis, processing_errors.
    """
    mode_str = 'TRAINING (Augmentation ON)' if is_training else 'INFERENCE'
    print(f"\n{'='*80}")
    print(f"PROCESSING PIPELINE — {mode_str}")
    print(f"   Frames to process: {len(sample_frames)}")
    print(f"   Batch size: {batch_size}")
    print(f"{'='*80}\n")

    processed_results: List[Dict] = []
    total_batches = max(1, len(sample_frames) // batch_size)
    batch_count   = 0

    for batch_start in range(0, len(sample_frames), batch_size):
        batch_end = min(batch_start + batch_size, len(sample_frames))
        batch     = sample_frames[batch_start:batch_end]

        batch_images = [frame['image'] for frame in batch]
        face_results = [face_detector.detect_face_and_landmarks(img)
                        for img in batch_images]

        for frame_data, face_result in zip(batch, face_results):
            frame_result: Dict = {
                'frame_id':        frame_data['frame_id'],
                'frame_data':      frame_data,
                'phase1_success':  False,
                'phase2_success':  False,
                'phase3_success':  False,
                'features':        None,
                'occlusion_analysis': None,
                'processing_errors':  [],
            }

            if face_result['is_valid']:
                frame_result['phase1_success'] = True
                try:
                    extract_kwargs = {
                        'image':        frame_data['image'],
                        'face_bbox':    face_result['face_bbox'],
                        'eye_regions':  face_result['eye_regions'],
                        'mouth_region': face_result['mouth_region'],
                    }
                    feature_result = feature_extractor.extract_region_features(
                        **extract_kwargs)

                    if len(feature_result['successful_regions']) >= 3:
                        frame_result['phase2_success'] = True
                        frame_result['features']       = feature_result

                        occlusion_result = occlusion_detector.analyze_occlusion_and_states(
                            frame_data['image'],
                            face_result['landmarks'],
                            face_result['eye_regions'],
                            face_result['mouth_region'],
                            face_result['face_bbox'],
                            feature_result,
                        )
                        if occlusion_result['is_valid']:
                            frame_result['phase3_success'] = True
                            frame_result['occlusion_analysis'] = occlusion_result

                except Exception as e:
                    frame_result['processing_errors'].append(f'Processing error: {e}')

            processed_results.append(frame_result)

        batch_count += 1
        if batch_count % 100 == 0:
            print(f'   Progress: {batch_count}/{total_batches} batches completed')

    successful = sum(1 for r in processed_results
                     if r['phase1_success'] and
                        r['phase2_success'] and
                        r['phase3_success'])
    print(f"\nPIPELINE COMPLETE: {successful}/{len(processed_results)} frames processed\n")
    return processed_results


# ─── Direct feature extraction (no per-frame image loading overhead) ──────────

def extract_features_stratified(
    csv_data: Dict,
    splits: Dict,
    face_detector,
    feat_extractor,
    occ_model,
    num_samples_per_video: Optional[int] = None,
    val_ratio: float = 0.20,
    random_state: int = 42,
    label_map: Optional[Dict] = None,
) -> Tuple[List, List, List]:
    """
    Extract features from DMD videos (clean — no synthetic occlusion).

    Videos in ``splits['test']`` go to the test set; all others are
    split 80/20 train/val by class.

    Returns
    -------
    (train_samples, val_samples, test_samples)
    """
    if label_map is None:
        label_map = {'EyeClosed': 0, 'Yawn': 1, 'Neutral': 2}

    train_subject_samples: List[Dict] = []
    test_subject_samples:  List[Dict] = []
    global_id = 0

    for video_key, meta in csv_data.items():
        cap = cv2.VideoCapture(meta['video_path'])
        if not cap.isOpened():
            print(f'  [SKIP] {video_key}')
            continue

        anns = meta['annotations']
        n    = len(anns)
        indices = (list(range(n))
                   if num_samples_per_video is None or num_samples_per_video >= n
                   else np.linspace(0, n - 1, num_samples_per_video, dtype=int).tolist())

        subject = meta['subject']
        is_test = video_key in splits['test']
        ok, bad = 0, 0

        for idx in indices:
            ann = anns[idx]
            if ann.class_label not in label_map:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, ann.frame)
            ret, bgr = cap.read()
            if not ret:
                continue

            det = face_detector.detect_face_and_landmarks(bgr)
            if not det['is_valid']:
                bad += 1
                continue

            rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            feat = feat_extractor.extract_region_features(
                bgr, det['face_bbox'], det['eye_regions'], det['mouth_region'])

            rkeys = ['face_features', 'left_eye_features',
                     'right_eye_features', 'mouth_features']
            if any(feat[k] is None for k in rkeys):
                bad += 1
                continue

            probs = occ_model.predict_probs(
                rgb, face_bbox=det['face_bbox'], image_bgr=False, face_margin=0.15)

            sample = {
                'frame_id':  global_id,
                'subject':   subject,
                'video_key': video_key,
                'features': {
                    'face':      feat['face_features'],
                    'left_eye':  feat['left_eye_features'],
                    'right_eye': feat['right_eye_features'],
                    'mouth':     feat['mouth_features'],
                },
                'occlusion_info': {
                    'eye_occlusion_prob':   float(probs[0]),
                    'mouth_occlusion_prob': float(probs[1]),
                },
                'label':      label_map[ann.class_label],
                'class_name': ann.class_label,
                'ground_truth': {
                    'eyes_occluded': ann.eyes_occluded_prior,
                    'mouth_occluded': ann.mouth_occluded_prior,
                    'eyes_state':    ann.eyes_state,
                },
            }
            global_id += 1

            if is_test:
                test_subject_samples.append(sample)
            else:
                train_subject_samples.append(sample)
            ok += 1

            if ok % 500 == 0:
                print(f'    ... {ok}/{len(indices)} frames processed')

        cap.release()
        dest = 'TEST' if is_test else 'TRAIN'
        print(f'  {video_key} [{dest}]: {ok} ok, {bad} skipped')

    if train_subject_samples:
        labels = [s['class_name'] for s in train_subject_samples]
        train_idx, val_idx = train_test_split(
            list(range(len(train_subject_samples))),
            test_size=val_ratio, stratify=labels, random_state=random_state)
        train_samples = [train_subject_samples[i] for i in train_idx]
        val_samples   = [train_subject_samples[i] for i in val_idx]
    else:
        train_samples, val_samples = [], []

    for name, ss in [('Train', train_samples), ('Val', val_samples),
                     ('Test', test_subject_samples)]:
        dist: Dict = {}
        for s in ss:
            dist[s['class_name']] = dist.get(s['class_name'], 0) + 1
        print(f'  {name}: {len(ss)} samples  {dist}')

    return train_samples, val_samples, test_subject_samples


def extract_features_with_augmentation(
    csv_data: Dict,
    splits: Dict,
    face_detector,
    feat_extractor,
    occ_model,
    apply_occ_fn,
    opacity_levels: List[float],
    aug_clean_fraction: float = 0.60,
    num_samples_per_video: Optional[int] = None,
    val_ratio: float = 0.20,
    random_state: int = 42,
    label_map: Optional[Dict] = None,
) -> Tuple[List, List, List]:
    """
    Extract features with per-subject synthetic occlusion augmentation
    (Experiment A+).

    Augmentation strategy (training subjects only)
    ----------------------------------------------
    For each subject's frames:
      - aug_clean_fraction (default 60%) stay clean
      - Remaining frames are split ~evenly into eye_only, mouth_only, both
      - Each augmented frame receives ONE randomly chosen non-zero opacity

    Test-subject frames are always kept clean (stress-test applies occlusion
    separately).

    Parameters
    ----------
    apply_occ_fn   : callable with signature
                     ``apply_synthetic_occlusion(rgb, landmarks, eye_opacity, mouth_opacity)``
                     (from synthetic_occlusion.py).
    opacity_levels : full OPACITY_LEVELS list; non-zero entries used for
                     augmentation.
    """
    if label_map is None:
        label_map = {'EyeClosed': 0, 'Yawn': 1, 'Neutral': 2}

    aug_opacities = [op for op in opacity_levels if op > 0]
    rng = np.random.default_rng(random_state)

    train_subject_samples: List[Dict] = []
    test_subject_samples:  List[Dict] = []
    global_id = 0

    for video_key, meta in csv_data.items():
        cap = cv2.VideoCapture(meta['video_path'])
        if not cap.isOpened():
            print(f'  [SKIP] {video_key}')
            continue

        anns = meta['annotations']
        n    = len(anns)
        indices = (list(range(n))
                   if num_samples_per_video is None or num_samples_per_video >= n
                   else np.linspace(0, n - 1, num_samples_per_video, dtype=int).tolist())

        subject = meta['subject']
        is_test = video_key in splits['test']
        ok, bad = 0, 0

        # Pre-assign per-subject augmentation buckets for training subjects
        bucket_map: Dict[int, Tuple[str, float]] = {}
        if not is_test:
            nf      = len(indices)
            n_clean = int(nf * aug_clean_fraction)
            n_aug   = nf - n_clean
            n_eye   = n_aug // 3
            n_mouth = n_aug // 3
            n_both  = n_aug - n_eye - n_mouth
            bucket_labels = (['none'] * n_clean + ['eye_only'] * n_eye
                             + ['mouth_only'] * n_mouth + ['both'] * n_both)
            perm = rng.permutation(nf)
            for pos, bpos in enumerate(perm):
                btype = bucket_labels[bpos]
                op    = 0.0 if btype == 'none' else float(rng.choice(aug_opacities))
                bucket_map[pos] = (btype, op)

        for frame_pos, idx in enumerate(indices):
            ann = anns[idx]
            if ann.class_label not in label_map:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, ann.frame)
            ret, bgr = cap.read()
            if not ret:
                continue

            det = face_detector.detect_face_and_landmarks(bgr)
            if not det['is_valid']:
                bad += 1
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            aug_type, aug_opacity = 'none', 0.0
            if not is_test:
                aug_type, aug_opacity = bucket_map.get(frame_pos, ('none', 0.0))
                if aug_type != 'none':
                    lm = det.get('landmarks')
                    if lm is not None:
                        eo = aug_opacity if aug_type in ('eye_only', 'both') else 0.0
                        mo = aug_opacity if aug_type in ('mouth_only', 'both') else 0.0
                        rgb = apply_occ_fn(rgb, lm, eye_opacity=eo, mouth_opacity=mo)
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            feat = feat_extractor.extract_region_features(
                bgr, det['face_bbox'], det['eye_regions'], det['mouth_region'])

            rkeys = ['face_features', 'left_eye_features',
                     'right_eye_features', 'mouth_features']
            if any(feat[k] is None for k in rkeys):
                bad += 1
                continue

            probs = occ_model.predict_probs(
                rgb, face_bbox=det['face_bbox'], image_bgr=False, face_margin=0.15)

            sample = {
                'frame_id':  global_id,
                'subject':   subject,
                'video_key': video_key,
                'features': {
                    'face':      feat['face_features'],
                    'left_eye':  feat['left_eye_features'],
                    'right_eye': feat['right_eye_features'],
                    'mouth':     feat['mouth_features'],
                },
                'occlusion_info': {
                    'eye_occlusion_prob':   float(probs[0]),
                    'mouth_occlusion_prob': float(probs[1]),
                },
                'label':      label_map[ann.class_label],
                'class_name': ann.class_label,
                'ground_truth': {
                    'eyes_occluded': ann.eyes_occluded_prior,
                    'mouth_occluded': ann.mouth_occluded_prior,
                    'eyes_state':    ann.eyes_state,
                },
                'aug_type':    aug_type,
                'aug_opacity': aug_opacity,
            }
            global_id += 1

            if is_test:
                test_subject_samples.append(sample)
            else:
                train_subject_samples.append(sample)
            ok += 1

            if ok % 500 == 0:
                print(f'    ... {ok}/{len(indices)} frames processed')

        cap.release()
        dest = 'TEST' if is_test else 'TRAIN'
        print(f'  {video_key} [{dest}]: {ok} ok, {bad} skipped')

    if train_subject_samples:
        labels = [s['class_name'] for s in train_subject_samples]
        train_idx, val_idx = train_test_split(
            list(range(len(train_subject_samples))),
            test_size=val_ratio, stratify=labels, random_state=random_state)
        train_samples = [train_subject_samples[i] for i in train_idx]
        val_samples   = [train_subject_samples[i] for i in val_idx]
    else:
        train_samples, val_samples = [], []

    for name, ss in [('Train', train_samples), ('Val', val_samples),
                     ('Test', test_subject_samples)]:
        dist: Dict = {}
        for s in ss:
            dist[s['class_name']] = dist.get(s['class_name'], 0) + 1
        print(f'  {name}: {len(ss)} samples  {dist}')

    if train_samples:
        aug_dist: Dict[str, int] = {}
        for s in train_samples:
            aug_dist[s['aug_type']] = aug_dist.get(s['aug_type'], 0) + 1
        print(f'  Train aug distribution: {aug_dist}')

    return train_samples, val_samples, test_subject_samples


# ─── Clip-based pipeline (STRATEGY_DESIGN.md) ─────────────────────────────────

def extract_features_for_clips(
    csv_data: Dict,
    split_config: Dict,
    face_detector,
    feat_extractor,
    occ_model,
    val_ratio: float = 0.20,
    seed: int = SEED,
    label_map: Optional[Dict[str, int]] = None,
    max_train_clips: Optional[int] = None,
    max_val_clips: Optional[int] = None,
    max_test_clips: Optional[int] = None,
    skip_test: bool = False,
) -> Tuple[List[Dict], List[Dict], List[Dict], List]:
    """
    Extract features from clips with regime-based occlusion (STRATEGY_DESIGN.md).

    - Clip-level sampling: T=32, stride 16 train/val, stride 32 test
    - FPS downsampling to 15
    - Temporal val split (last 20% of train subjects)
    - Training: 55% clean, 45% augmented (persistent/transient with label-aware caps)
    - Val/test: clean only

    skip_test : if True, do not extract test (saves time); stress test loads test frames later.

    Returns
    -------
    (train_samples, val_samples, test_samples, test_clips) — frame-level samples + test clips for stress test
    """
    if label_map is None:
        label_map = {'EyeClosed': 0, 'Yawn': 1, 'Neutral': 2}

    train_subjects = set(split_config['train_subjects'])
    ts = split_config.get('test_subjects') or split_config.get('test_subject')
    test_subjects = {ts} if isinstance(ts, str) else set(ts or [])

    T = CLIP_CONFIG.get('T', 32)
    fps_src = CLIP_CONFIG.get('FPS_SOURCE', 29.76)
    fps_target = CLIP_CONFIG.get('FPS_TARGET', 15)
    train_stride = CLIP_CONFIG.get('TRAIN_STRIDE', 16)
    eval_stride = CLIP_CONFIG.get('EVAL_STRIDE', 32)

    # Build clips per video with appropriate stride
    stride_per_video: Dict[str, int] = {}
    for video_key, meta in csv_data.items():
        subj = meta['subject']
        stride_per_video[video_key] = eval_stride if subj in test_subjects else train_stride

    clips_per_video = extract_clips_for_videos(
        csv_data, T=T, stride=train_stride,
        fps_src=fps_src, fps_target=fps_target,
        label_map=label_map, stride_per_video=stride_per_video,
    )

    train_clips, val_clips, test_clips = get_train_val_test_clips(
        clips_per_video, split_config, val_ratio=val_ratio, seed=seed,
    )
    if max_train_clips is not None:
        train_clips = train_clips[:max_train_clips]
    if max_val_clips is not None:
        val_clips = val_clips[:max_val_clips]
    if max_test_clips is not None:
        test_clips = test_clips[:max_test_clips]

    train_samples: List[Dict] = []
    val_samples: List[Dict] = []
    test_samples: List[Dict] = []

    def _annotations_by_frame(meta):
        return {a.frame: a for a in meta['annotations']}

    for clip in train_clips:
        meta = csv_data[clip.video_key]
        cap = cv2.VideoCapture(meta['video_path'])
        if not cap.isOpened():
            continue
        regime, opacity = assign_regime_to_clip(clip, seed=seed)
        clip_seed = hash((clip.video_key, clip.clip_start, seed)) % (2**32)
        ann_by_frame = _annotations_by_frame(meta)
        ok = 0
        for fi, frame_num in enumerate(clip.frame_numbers):
            ann = ann_by_frame.get(frame_num)
            if ann is None or ann.class_label not in label_map:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, bgr = cap.read()
            if not ret:
                continue
            det = face_detector.detect_face_and_landmarks(bgr)
            if not det['is_valid']:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if regime != 'clean' and opacity > 0 and det.get('landmarks') is not None:
                aug_img = apply_occlusion_to_frame(
                    rgb, det['landmarks'], regime, opacity,
                    frame_idx_in_clip=fi, clip_len=clip.T, clip_seed=clip_seed,
                )
                bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                rgb = aug_img
            feat = feat_extractor.extract_region_features(
                bgr, det['face_bbox'], det['eye_regions'], det['mouth_region'])
            rkeys = ['face_features', 'left_eye_features', 'right_eye_features', 'mouth_features']
            if any(feat[k] is None for k in rkeys):
                continue
            probs = occ_model.predict_probs(
                rgb, face_bbox=det['face_bbox'], image_bgr=False, face_margin=0.15)
            sample = {
                'frame_id': len(train_samples) + ok,
                'subject': clip.subject,
                'video_key': clip.video_key,
                'features': {
                    'face': feat['face_features'],
                    'left_eye': feat['left_eye_features'],
                    'right_eye': feat['right_eye_features'],
                    'mouth': feat['mouth_features'],
                },
                'occlusion_info': {
                    'eye_occlusion_prob': float(probs[0]),
                    'mouth_occlusion_prob': float(probs[1]),
                },
                'label': label_map[ann.class_label],
                'class_name': ann.class_label,
                'ground_truth': {
                    'eyes_occluded': ann.eyes_occluded_prior,
                    'mouth_occluded': ann.mouth_occluded_prior,
                    'eyes_state': ann.eyes_state,
                },
            }
            train_samples.append(sample)
            ok += 1
        cap.release()
        if ok > 0:
            print(f'  {clip.video_key} clip@{clip.clip_start}: {ok} frames (regime={regime})')

    for clip in val_clips:
        meta = csv_data[clip.video_key]
        cap = cv2.VideoCapture(meta['video_path'])
        if not cap.isOpened():
            continue
        ann_by_frame = _annotations_by_frame(meta)
        ok = 0
        for fi, frame_num in enumerate(clip.frame_numbers):
            ann = ann_by_frame.get(frame_num)
            if ann is None or ann.class_label not in label_map:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, bgr = cap.read()
            if not ret:
                continue
            det = face_detector.detect_face_and_landmarks(bgr)
            if not det['is_valid']:
                continue
            feat = feat_extractor.extract_region_features(
                bgr, det['face_bbox'], det['eye_regions'], det['mouth_region'])
            rkeys = ['face_features', 'left_eye_features', 'right_eye_features', 'mouth_features']
            if any(feat[k] is None for k in rkeys):
                continue
            probs = occ_model.predict_probs(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), face_bbox=det['face_bbox'],
                image_bgr=False, face_margin=0.15)
            sample = {
                'frame_id': len(val_samples) + ok,
                'subject': clip.subject,
                'video_key': clip.video_key,
                'features': {
                    'face': feat['face_features'],
                    'left_eye': feat['left_eye_features'],
                    'right_eye': feat['right_eye_features'],
                    'mouth': feat['mouth_features'],
                },
                'occlusion_info': {
                    'eye_occlusion_prob': float(probs[0]),
                    'mouth_occlusion_prob': float(probs[1]),
                },
                'label': label_map[ann.class_label],
                'class_name': ann.class_label,
                'ground_truth': {
                    'eyes_occluded': ann.eyes_occluded_prior,
                    'mouth_occluded': ann.mouth_occluded_prior,
                    'eyes_state': ann.eyes_state,
                },
            }
            val_samples.append(sample)
            ok += 1
        cap.release()

    if not skip_test:
        for clip in test_clips:
            meta = csv_data[clip.video_key]
            cap = cv2.VideoCapture(meta['video_path'])
            if not cap.isOpened():
                continue
            ann_by_frame = _annotations_by_frame(meta)
            ok = 0
            for fi, frame_num in enumerate(clip.frame_numbers):
                ann = ann_by_frame.get(frame_num)
                if ann is None or ann.class_label not in label_map:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, bgr = cap.read()
                if not ret:
                    continue
                det = face_detector.detect_face_and_landmarks(bgr)
                if not det['is_valid']:
                    continue
                feat = feat_extractor.extract_region_features(
                    bgr, det['face_bbox'], det['eye_regions'], det['mouth_region'])
                rkeys = ['face_features', 'left_eye_features', 'right_eye_features', 'mouth_features']
                if any(feat[k] is None for k in rkeys):
                    continue
                probs = occ_model.predict_probs(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), face_bbox=det['face_bbox'],
                    image_bgr=False, face_margin=0.15)
                sample = {
                    'frame_id': len(test_samples) + ok,
                    'subject': clip.subject,
                    'video_key': clip.video_key,
                    'features': {
                        'face': feat['face_features'],
                        'left_eye': feat['left_eye_features'],
                        'right_eye': feat['right_eye_features'],
                        'mouth': feat['mouth_features'],
                    },
                    'occlusion_info': {
                        'eye_occlusion_prob': float(probs[0]),
                        'mouth_occlusion_prob': float(probs[1]),
                    },
                    'label': label_map[ann.class_label],
                    'class_name': ann.class_label,
                    'ground_truth': {
                        'eyes_occluded': ann.eyes_occluded_prior,
                        'mouth_occluded': ann.mouth_occluded_prior,
                        'eyes_state': ann.eyes_state,
                    },
                }
                test_samples.append(sample)
                ok += 1
            cap.release()
            if ok > 0:
                print(f'  {clip.video_key} clip@{clip.clip_start}: {ok} frames [TEST]')
    else:
        print('  [Skipped test extraction — will load during stress test]')

    for name, ss in [('Train', train_samples), ('Val', val_samples), ('Test', test_samples)]:
        if ss:
            dist = {}
            for s in ss:
                dist[s['class_name']] = dist.get(s['class_name'], 0) + 1
            print(f'  {name}: {len(ss)} samples  {dist}')

    return train_samples, val_samples, test_samples, test_clips
