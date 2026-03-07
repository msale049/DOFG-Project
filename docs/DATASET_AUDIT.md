# Phase 1 — DMD Dataset Audit Report

**Date:** 2026-02-23  
**Scope:** 15 subjects from DMD (Driver Monitoring Dataset), drowsiness subset  
**Target classes:** Neutral, EyeClosed, Yawn

---

## 1. Dataset Structure

### 1.1 Raw Data Layout

| Location | Content |
|----------|---------|
| `Data/Sub{N}/` | Per-subject folder (N = 1..15) |
| `Data/Sub{N}/sub{N}_video.mp4` | Face-camera video (1280×720, ~29.76 fps) |
| `Data/Sub{N}/sub{N}_video.csv` | Frame-level annotations (derived from JSON) |
| `Data/Sub{N}/labels.json` | OpenLABEL JSON (Sub11–15 only; others may be elsewhere) |

**Videos:** All 15 subjects have `sub{N}_video.mp4` and `sub{N}_video.csv`. No session IDs or camera IDs in the current layout — one video per subject.

### 1.2 Annotation Sources

- **Primary:** CSV files (`sub{N}_video.csv`) created from OpenLABEL JSON via `DMD_Face_Detection_and_Visualization.ipynb`
- **JSON format:** ASAM OpenLABEL with `actions` (frame_intervals) and `frames` (per-frame references)
- **Label derivation (notebook):**
  - Eyes state: `opened`, `opening`, `closed`, `closing`, `undefined`
  - Yawning: `Yawning with hand`, `Yawning without hand`
  - Class: Yawn > EyeClosed > Neutral (priority order)
  - Occlusion priors: `eyes_occluded_prior` = glasses; `mouth_occluded_prior` = yawn_with_hand

### 1.3 CSV Schema

| Column | Type | Description |
|--------|------|--------------|
| frame | int | Frame index (0-based) |
| timestamp_s | float | Seconds from start |
| class | str | Neutral \| EyeClosed \| Yawn |
| variant | str | e.g. Neutral_Occluded |
| eyes_state | str | opened \| opening \| closed \| closing \| undefined |
| yawn_with_hand | bool | Hand-over-mouth during yawn |
| yawn_without_hand | bool | Open-mouth yawn |
| eyes_occluded_prior | bool | Glasses (persistent) |
| mouth_occluded_prior | bool | Hand over mouth (yawn_with_hand) |
| glasses | bool | Subject wears glasses |

---

## 2. Usable Data Summary

### 2.1 Frame Counts (Raw CSV, No Filtering)

| Subject | Frames | Duration (s) | FPS |
|---------|--------|--------------|-----|
| Sub1 | 5,480 | 184.1 | ~29.8 |
| Sub2 | 5,416 | 182.0 | ~29.8 |
| Sub3 | 5,527 | 185.7 | ~29.8 |
| Sub4 | 5,229 | 175.7 | ~29.8 |
| Sub5 | 5,287 | 177.6 | ~29.8 |
| Sub6 | 5,410 | 181.8 | ~29.8 |
| Sub7 | 5,405 | 181.6 | ~29.8 |
| Sub8 | 5,407 | 181.7 | ~29.8 |
| Sub9 | 5,292 | 177.8 | ~29.8 |
| Sub10 | 6,055 | 203.4 | ~29.8 |
| Sub11 | 5,377 | 180.6 | ~29.8 |
| Sub12 | 5,576 | 187.3 | ~29.8 |
| Sub13 | 5,380 | 180.7 | ~29.8 |
| Sub14 | 5,514 | 185.2 | ~29.8 |
| Sub15 | 5,294 | 177.9 | ~29.8 |
| **Total** | **81,649** | ~2,725 s | ~29.8 |

### 2.2 With `filter_eye_states=True` (Current Default)

`data_loading.load_csv_video_data` drops frames with `eyes_state` in `{opening, closing, undefined}`.

| Metric | Value |
|--------|-------|
| Total frames (filtered) | **61,308** |
| Dropped | 20,341 (~25%) |
| Per-subject range | 3,463 (Sub3) – 4,585 (Sub14) |

### 2.3 Class Distribution (Raw CSV)

| Class | Frames | % |
|-------|--------|---|
| Neutral | 56,734 | **69.5%** |
| EyeClosed | 15,024 | **18.4%** |
| Yawn | 9,891 | **12.1%** |

**Imbalance:** Neutral dominates; EyeClosed and Yawn are minority classes.

### 2.4 Clips at FPS=15, T=32

| Parameter | Value |
|-----------|-------|
| FPS (target) | 15 |
| Clip length T | 32 |
| Train stride | 16 |
| Eval stride | 32 |
| Avg frames/subject @ 15 fps | ~2,700 |
| Eval clips/subject | ~83 |
| Train clips/subject | ~167 |
| **Total eval clips (15 subj)** | **~1,245** |
| **Total train clips (12 subj)** | **~2,004** |

---

## 3. Label Granularity and Representation

### 3.1 Granularity

- **Frame-level:** One label per frame
- **Segment-level:** JSON has `frame_intervals` per action; CSV is flattened to frame-level
- **Event-level:** Yawn and eye-close are temporal events; CSV encodes presence per frame

### 3.2 Subject / Session / Camera IDs

- **Subject ID:** Folder name (Sub1–Sub15)
- **Session ID:** Not explicit; DMD drowsiness uses s5 only (per README)
- **Camera ID:** Not in current layout; face camera only
- **Video key:** `{subject_folder}_{video_name}` (e.g. `Sub1_sub1_video`)

### 3.3 Preprocessing Artifacts

- **Extracted features:** None stored on disk; computed on-the-fly in pipeline
- **Crops / landmarks:** Not precomputed; face detection + landmarks at runtime
- **Filtered segments:** `Sub11/filtered_segments/segment_0_info.json` exists (different processing path; not used by main pipeline)

---

## 4. Existing Splits and Pipeline

### 4.1 Train/Val/Test Split

- **`data_loading.split_video_ids`:** Video-level split
  - `num_test=3` → 3 videos held out for test
  - `num_val=0` → validation uses same videos as train
  - Shuffle by `seed`, then: test = first 3, train = rest

### 4.2 Feature Extraction

- **`pipeline.extract_features_stratified`** (clean) and **`extract_features_with_augmentation`** (Experiment A+)
- Frame-level: face detection → feature extraction → occlusion estimator
- Sampling: `num_samples_per_video` or `np.linspace` over all frames
- **Val split:** `train_test_split` with `stratify=labels` on **frame indices** within train videos → random frame-level split

### 4.3 Augmentation

- **`synthetic_occlusion.apply_eye_band`**, **`apply_mouth_rect`**, **`apply_synthetic_occlusion`**
- Applied per-frame with landmarks
- Opacity in [0, 1]; `OPACITY_LEVELS = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]`
- **Experiment A+:** 60% clean, 40% augmented (eye_only, mouth_only, both) with random opacity per frame

---

## 5. Leakage Risks (Current Code)

### 5.1 Confirmed Leakage Risks

| Risk | Location | Description |
|------|----------|-------------|
| **Train/val temporal overlap** | `pipeline.py` L261–266 | Val is a random 20% of frames from train videos. Train and val can share temporally adjacent frames from the same video. |
| **Frame-level random split** | `pipeline.py` | `train_test_split` on frame indices; no clip boundaries or temporal separation. |
| **Augmentation on val** | `extract_features_with_augmentation` | Augmentation applied only to train; val is clean. (Correct.) |
| **Stress-test in training** | N/A | Stress-test conditions (synthetic occlusion) are applied only at test time. (Correct.) |

### 5.2 Potential Issues

| Issue | Description |
|-------|-------------|
| **No clip-level sampling** | Pipeline is frame-based; no T=32 clips, no stride. |
| **No FPS downsampling** | Uses all frames at ~30 fps; no explicit 15 fps. |
| **filter_eye_states** | Dropping opening/closing may remove useful transition frames; consider keeping for clip-level aggregation. |
| **Single session per subject** | No session-level split; only subject-level possible. |

### 5.3 What Is Correct

- Test set is subject-disjoint (video-level)
- Augmentation is train-only
- Stress tests are test-only
- Split is deterministic (seed)

---

## 6. Issues and Limitations

### 6.1 Class Imbalance

- Neutral ~70%; EyeClosed ~18%; Yawn ~12%
- Recommendation: weighted loss or class-balanced sampling

### 6.2 Occlusion in DMD

- **Eyes:** Mostly glasses (prescription); no sunglasses
- **Mouth:** Hand-over-mouth during yawn only
- **Synthetic occlusion:** Used to stress-test gating; occlusion estimator trained on ROF (masks, sunglasses)

### 6.3 Small Subject Count

- 15 subjects total; 3 for test → 12 for train
- LOSO (Leave-One-Subject-Out) over 15 folds is feasible for evaluation
- Per-fold: 14 train, 1 test

### 6.4 Missing / Inconsistent Data

- `labels.json` present for Sub11–15; Sub1–10 may have been processed from a different path (e.g. More_Data)
- CSV is the canonical source for the current pipeline

---

## 7. Recommendations for Redesign

1. **Clip-level sampling:** Introduce T=32 (and T=16 ablation) with stride 16 (train) and 32 (eval).
2. **FPS downsampling:** Resample to 15 fps before clip extraction.
3. **Leakage-safe val:** Use temporal split (e.g. last 20% of each train video) or subject-held-out validation within LOSO.
4. **Persistent vs transient occlusion:** Extend augmentation to support clip-consistent (persistent) and subsegment (transient) occlusion.
5. **Deterministic seeding:** Use `(subject_id, video_id, clip_start, fps, T)` for reproducibility.
6. **LOSO evaluation:** Prefer subject-wise evaluation given 15 subjects.

---

## 8. Label Logic Verification (JSON → CSV)

From `DMD_Face_Detection_and_Visualization.ipynb`:

| Rule | Implementation | Notes |
|------|----------------|-------|
| Class priority | Yawn > EyeClosed > Neutral | Correct |
| EyeClosed | closed OR closing | Transition frames (closing) counted as EyeClosed |
| Neutral | opened OR opening | Transition frames (opening) counted as Neutral |
| eyes_occluded_prior | glasses (subject-level) | Prescription glasses flagged; may overstate occlusion |
| mouth_occluded_prior | yawn_with_hand | Hand-over-mouth during yawn |

**Potential issues:**
- **Glasses:** `OCCLUDE_ON_GLASSES=True` treats prescription glasses as eye occlusion. Real occlusion (sunglasses) is rare in DMD.
- **filter_eye_states:** Dropping opening/closing/undefined removes ~25% of frames. For clip-level, consider keeping them and using majority vote.

---

## 9. Uncertainty / Open Questions

- Exact mapping of `Data/Sub*` to DMD group/session names (gA, gB, s5, etc.) if needed for citation
- Whether `filter_eye_states` should be toggled for clip-level (e.g. majority vote over T frames)
- Whether `filtered_segments` in Sub11 is used anywhere or is legacy
