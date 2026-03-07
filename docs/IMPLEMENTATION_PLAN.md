# Phase 3 — File-by-File Implementation Plan

## New Modules to Create

### 1. `clip_sampler.py` (NEW)

**Purpose:** Deterministic clip extraction from video annotations.

**Contents:**
- `downsample_frames_to_fps(annotations, fps_src, fps_target)` → filtered annotation indices
- `extract_clips(annotations, T, stride, fps, seed)` → list of `(video_key, clip_start, frame_indices)`
- `ClipInfo` dataclass: `video_key`, `subject`, `clip_start`, `frame_indices`, `majority_class`, `fps`, `T`
- Deterministic seeding via `(video_key, clip_start, T, seed)`

**Dependencies:** `config`, `data_loading`

---

### 2. `split_generator.py` (NEW)

**Purpose:** Leakage-safe train/val/test split at clip level.

**Contents:**
- `create_loso_splits(subject_ids, seed)` → list of 15 folds, each `{train_subjects, test_subject}`
- `create_fixed_split(subject_ids, num_test, seed)` → `{train_subjects, test_subjects}` (current behaviour)
- `split_clips_temporal(clips_per_subject, val_ratio, seed)` → `{train_clips, val_clips}` per subject
- `get_train_val_test_clips(csv_data, split_config, clip_config)` → `train_clips`, `val_clips`, `test_clips`

**Dependencies:** `config`, `data_loading`, `clip_sampler`

---

### 3. `occlusion_augmentation.py` (NEW, extends `synthetic_occlusion.py`)

**Purpose:** Clip-consistent and transient occlusion.

**Contents:**
- `OCCLUSION_REGIMES`: enum or constants for persistent_eye, persistent_mouth, persistent_both, transient_eye, transient_mouth
- `OPACITY_BANDS`: `{"hard": 1.0, "medium": (0.7, 0.9), "light": (0.4, 0.7)}`
- `assign_regime_to_clip(clip_info, regime_weights, class_caps, seed)` → regime, opacity
- `apply_persistent_occlusion(image, landmarks, regime, opacity)` → uses existing `apply_eye_band`, `apply_mouth_rect`
- `apply_transient_occlusion(image, landmarks, regime, opacity, frame_idx_in_clip, clip_len)` → occluded only if frame in subsegment
- `get_transient_mask(frame_idx, clip_len, seg_len, seg_start)` → bool

**Dependencies:** `synthetic_occlusion`, `config`

---

### 4. `stress_test_evaluator.py` (NEW)

**Purpose:** Deterministic stress-test evaluation.

**Contents:**
- `STRESS_CONDITIONS`: list of `(name, regime, opacity)` for test
- `run_stress_test(model, test_clips, conditions, pipeline_components)` → dict of per-condition metrics
- `aggregate_stress_results(results)` → summary table

**Dependencies:** `synthetic_occlusion`, `occlusion_augmentation`, `evaluation`, `pipeline`

---

## Files to Modify

### 1. `config.py`

**Changes:**
- Add `CLIP_CONFIG`: `fps=15`, `T=32`, `T_ablation=16`, `train_stride=16`, `eval_stride=32`
- Add `AUGMENTATION_CONFIG`: regime weights, opacity bands, class caps

---

### 2. `data_loading.py`

**Changes:**
- Add optional `filter_eye_states=True` behaviour (unchanged)
- Add `load_annotations_for_clips(csv_data, clip_infos)` → annotations for each clip
- Optional: `get_frame_indices_at_fps(annotations, fps_src, fps_target)` for downsampling

---

### 3. `pipeline.py`

**Changes:**
- Add `extract_features_for_clips(clips, splits, face_detector, ...)` — clip-aware version
- Use `split_generator.get_train_val_test_clips` instead of inline split logic
- Replace `train_test_split` with temporal split from `split_generator`
- Add `apply_augmentation_to_clip(clip_frames, clip_info, regime, ...)` for training

---

### 4. `datasets.py`

**Changes:**
- Add `ClipDriverStateDataset` that yields `(clip_features, clip_labels)` — one sample per clip
- Clip label: majority vote or last-frame label (configurable)
- Support `T` in sample shape

---

### 5. `trainer_enhanced.py` (if clip-based)

**Changes:**
- Accept clip-level batches (batch of clips, each clip = T frames)
- Model may need to accept sequences; current transformer is per-frame — may need aggregation (e.g. mean over T, or temporal encoder)

**Note:** Current model is per-frame. Clip-level training requires either:
- (a) Aggregate clip to single prediction (e.g. mean over T frames, then classify), or
- (b) Use temporal model (e.g. 1D conv over T, or transformer over T).  
- (c) Keep frame-level for now; clip-level sampling only affects which frames are selected and how augmentation is applied.

**Recommendation:** Start with (a) — clip-level sampling + augmentation, but still frame-level prediction per frame. Aggregate to clip-level only for evaluation (majority vote). This minimises model changes.

---

## What to Keep Unchanged

- `synthetic_occlusion.py`: low-level `apply_eye_band`, `apply_mouth_rect`, `apply_synthetic_occlusion` — keep as-is
- `face_detection_*.py`, `feature_extraction.py`, `occlusion_estimator.py` — no changes
- `transformer_enhanced.py`, `transformer_enhanced_no_gates.py` — no structural changes
- `evaluation.py` — extend for stress conditions, not replace

---

## Implementation Order

1. `clip_sampler.py` — core clip extraction
2. `split_generator.py` — split logic
3. `config.py` — add clip config
4. `occlusion_augmentation.py` — regime assignment and application
5. `pipeline.py` — integrate clip-aware extraction
6. `stress_test_evaluator.py` — test evaluation
7. `datasets.py` — extend for clips
8. `trainer_enhanced.py` — optional clip aggregation

---

## Reproducibility

- All randomness seeded via `config.SEED`
- Clip IDs: `(subject_id, video_key, clip_start, T, stride)` — deterministic
- Regime assignment: `hash(subject_id, clip_start, seed) % 100` for regime choice
- Log split config in `run_config.json` or similar
