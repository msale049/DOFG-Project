# Phase 2 — Strategy Design

**Based on:** Dataset audit (Phase 1) and user requirements

---

## A. Split Design

### A.1 Subject-Wise Evaluation

- **LOSO (Leave-One-Subject-Out):** 15 folds; each fold: 14 subjects train, 1 subject test
- **Why:** Only 15 subjects; subject-wise evaluation is essential for generalisation
- **Alternative:** Fixed 3-subject test (current) for faster iteration; report both LOSO and fixed-split

### A.2 Validation Within Each LOSO Fold

- **Problem:** With 1 session per subject, we cannot hold out a full subject for validation
- **Options:**
  1. **Temporal split:** Last 20% of each train subject's video for validation (no overlap with train clips)
  2. **Subject-held-out val:** Use 1 of the 14 train subjects as val (14 train, 1 val, 1 test)
  3. **Cross-subject val:** Use 2 subjects for val, 12 for train, 1 for test

**Recommendation:** Option 1 (temporal split) — simple, no subject leakage, clip boundaries respected.

- For each train subject video: clips with `clip_start` in last 20% of valid range → val
- Clips with `clip_start` in first 80% → train
- Deterministic by `(subject_id, clip_start, T, stride)`

### A.3 Split Summary

| Split | Source | Leakage-safe |
|-------|--------|--------------|
| Test | 1 held-out subject (LOSO) or fixed 3 | Yes |
| Val | Temporal last 20% of train subjects | Yes |
| Train | Temporal first 80% of train subjects | Yes |

---

## B. Sampling

### B.1 Default Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| FPS | 15 | Downsample from ~30 |
| T (clip length) | 32 | ~2.1 s per clip |
| Train stride | 16 | 50% overlap |
| Eval stride | 32 | No overlap |
| Ablation T | 16 | ~1.1 s per clip |

### B.2 Clip Counts (Approximate)

- **Per subject:** ~2,700 frames @ 15 fps → ~83 eval clips, ~167 train clips
- **12 train subjects:** ~2,004 train clips, ~500 val clips
- **1 test subject:** ~83 test clips (eval stride)

### B.3 Frame Index Mapping

- Original: frame index at 29.76 fps
- Downsample: take every `round(29.76/15) ≈ 2` frames → effective 14.88 fps, or use `frame % 2 == 0` for simplicity
- **Exact:** `frame_15fps = frame_30fps // 2` (or more precise ratio)

---

## C. Synthetic Occlusion Design

### C.1 Regime Family (Refined)

| Regime | % of augmented | Description | Opacity |
|--------|----------------|-------------|---------|
| Clean | 55% | No occlusion | — |
| Persistent eye | 15% | All T frames occluded (eyes) | Constant per clip |
| Persistent mouth | 15% | All T frames occluded (mouth) | Constant per clip |
| Persistent both | 5% | All T frames occluded (eyes + mouth) | Constant per clip |
| Transient eye | 5% | One contiguous subsegment occluded (eyes) | Constant per segment |
| Transient mouth | 5% | One contiguous subsegment occluded (mouth) | Constant per segment |

**Total augmented:** 45% (55% clean).

### C.2 Opacity Bands

| Band | Range | Use case |
|------|-------|----------|
| Hard | 1.0 | Full occlusion |
| Medium | 0.7–0.9 | Strong occlusion |
| Light | 0.4–0.7 | Partial occlusion |

**Per clip/segment:** One opacity value, constant for all frames in that clip/segment.

**Deterministic choice:** `opacity = f(subject_id, video_id, clip_start, regime, seed)`

### C.3 Transient Subsegment

- **Length:** e.g. T/4 to T/2 frames (8–16 frames at T=32)
- **Position:** Random but contiguous; e.g. `start = randint(0, T - seg_len)` with seed
- **Rest of clip:** Clean

### C.4 Occlusion-Aware Gating

- **Opacity → occlusion estimator:** Higher opacity → higher p_eye / p_mouth from estimator
- **Gate target:** `gate = 0.3 + 0.7 * (1 - p_occ)` → low gate when p_occ high
- **Training:** Gate alignment loss teaches gates to follow this relationship

---

## D. Label-Aware Safety Constraints

### D.1 Rules

1. **Yawn clips:** Cap fraction with persistent mouth occlusion (e.g. ≤30% of yawn clips get persistent mouth)
2. **EyeClosed clips:** Cap fraction with persistent eye occlusion (e.g. ≤30%)
3. **Persistent both:** Rare for positive-class clips; apply to ≤10% of EyeClosed/Yawn clips
4. **Transient:** Safer; can apply more liberally (e.g. 50% of augmented transient for positive classes)

### D.2 Implementation

- Before assigning augmentation regime to a clip, check majority class in clip
- If majority = Yawn and regime = persistent_mouth → with probability 0.7, switch to clean or transient_mouth
- Similar for EyeClosed + persistent_eye

---

## E. Validation and Test Strategy

### E.1 Validation

- **Primary:** Clean only (no synthetic occlusion) for model selection
- **Optional:** Fixed stressed val set (one copy per regime) for monitoring — same clips, different occlusion, deterministic

### E.2 Test

1. **Clean test:** No occlusion
2. **Structured stress tests (deterministic, separate from training):**
   - Persistent eye (opacity 0.8)
   - Persistent mouth (opacity 0.8)
   - Persistent both (opacity 0.8)
   - Transient eye (middle T/4 of clip, opacity 0.8)
   - Transient mouth (middle T/4 of clip, opacity 0.8)

Each test clip is evaluated in 6 conditions: clean + 5 stress tests.

---

## F. Deterministic Seeding

Use composite seed from:

- `subject_id`
- `video_id` or `session_id`
- `clip_start` (first frame index of clip)
- `fps`, `T`, `stride`
- Global `seed`

Example: `rng = np.random.default_rng(hash((subject_id, clip_start, T, seed)) % 2**32)`

---

## G. Summary of Design Choices

| Aspect | Choice |
|--------|--------|
| Split | LOSO preferred; temporal val within train |
| Clips | T=32 default, T=16 ablation; stride 16 train, 32 eval |
| Augmentation | 55% clean, 45% augmented (persistent + transient) |
| Opacity | Constant per clip/segment; bands: hard/medium/light |
| Safety | Cap persistent occlusion on positive-class clips |
| Val | Clean; optional stressed val |
| Test | Clean + 5 stress conditions, deterministic |
