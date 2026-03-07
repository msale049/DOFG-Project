# Leakage Risks and Fixes

## Identified Risks

### 1. Train/Val Temporal Overlap (HIGH)

**Location:** `pipeline.py` L261–266, `dofg_pipeline.py` (if used)

**Problem:** Validation is a random 20% of frames from the same videos as training. Consecutive frames in a video are highly correlated. A frame in val can be immediately adjacent to a frame in train.

**Fix:** Use temporal split. For each train subject video:
- Sort frames/clips by start index
- First 80% of clip starts → train
- Last 20% of clip starts → val
- Ensures no temporal overlap between train and val clips

---

### 2. Frame-Level Split Instead of Clip-Level (HIGH)

**Problem:** Current pipeline samples individual frames. With clip-based training (T=32), we need clip-level splits. Random frame split would put some frames from the same 32-frame window in train and others in val.

**Fix:** Split at clip granularity. Each clip belongs entirely to train or val. Clip boundaries defined by `(video_key, clip_start, T, stride)`.

---

### 3. Augmentation on Validation (LOW — Currently Correct)

**Status:** Augmentation is applied only to train. Val and test are clean in training pipeline. **No fix needed.**

---

### 4. Stress-Test Contamination (LOW — Currently Correct)

**Status:** Stress tests (synthetic occlusion at test time) are separate from training. Test frames are not used for training. **No fix needed.**

---

### 5. Subject Leakage in Train/Val (MEDIUM)

**Problem:** With video-level split, train and test are subject-disjoint. But train and val share subjects. If we use random frame split, val performance may be optimistic because val frames are from same subjects as train.

**Fix:** Temporal split (Fix 1) reduces this — val is from temporally distinct regions. For stricter evaluation, use subject-held-out val (1 of 14 train subjects for val).

---

## Implementation Checklist

- [ ] Replace `train_test_split` on frame indices with temporal clip split
- [ ] Ensure clip boundaries never span train/val boundary
- [ ] Add `split_clips_temporal(cfg)` returning `{train_clips, val_clips}` per subject
- [ ] Update `extract_features_stratified` and `extract_features_with_augmentation` to consume clip lists
- [ ] Add unit test: no train clip overlaps val clip (by frame range)
