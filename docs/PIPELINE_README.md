# Pipeline Usage — Training, Validation, Stress Testing

## Visualize Occlusion Samples

Generate sample images for all occlusion types (training regimes, stress conditions, legacy):

```bash
python visualize_occlusion.py --data Data --out occlusion_samples
```

Saves to `occlusion_samples/`:
- `train_*`: Training regimes (clean, persistent_eye/mouth/both, transient_eye/mouth) at op 0.8
- `stress_*`: Stress test conditions (6 types at 0.8)
- `legacy_*`: Legacy eye_only, mouth_only, both at opacities 0.3–1.0
- `summary_training.jpg`: Grid of training regimes

---

## Quick Start: Single Command

```bash
# Clip strategy (STRATEGY_DESIGN) — default, uses regime-based occlusion
python run_train_eval.py --strategy clip --max-train-clips 50 --max-val-clips 15 --epochs 3

# Full clip run (all clips; slower)
python run_train_eval.py --strategy clip --epochs 20

# Legacy frame sampling (faster iteration)
python run_train_eval.py --strategy legacy --samples 30 --epochs 5 --stress-frames 15

# With k-fold (5 folds)
python run_train_eval.py --mode kfold --k 5 --max-train-clips 100 --epochs 10

# Skip stress test (training + eval only)
python run_train_eval.py --strategy clip --max-train-clips 30 --epochs 5 --no-stress

# Complete end-to-end: RetinaFace, fixed, clip, defer-test, latency report
python run_train_eval.py --strategy clip --mode fixed --face retina --benchmark --epochs 20
```


# Fixed split (default): 3 subjects for test
python run_train_eval.py --mode fixed --num-test 3

# k-Fold (5 folds), run fold 0
python run_train_eval.py --mode kfold --k 5 --fold 0

# Run all 5 folds
for f in 0 1 2 3 4; do
  python run_train_eval.py --mode kfold --k 5 --fold $f --epochs 20
done

# LOSO (15 folds), run fold 0
python run_train_eval.py --mode loso --fold 0

# Run all 15 LOSO folds
for f in $(seq 0 14); do
  python run_train_eval.py --mode loso --fold $f --epochs 20
done

### run_train_eval.py options

| Option | Default | Description |
|--------|---------|--------------|
| `--strategy` | clip | `clip` (STRATEGY_DESIGN) \| `legacy` (frame sampling) |
| `--samples` | 30 | Frames per video for legacy (0 = all) |
| `--max-train-clips` | None | Max train clips for clip strategy (quick test) |
| `--max-val-clips` | None | Max val clips for clip strategy |
| `--epochs` | 5 | Training epochs |
| `--batch` | 16 | Batch size |
| `--mode` | fixed | `fixed` \| `kfold` \| `loso` |
| `--k` | 5 | Folds for k-fold |
| `--num-test` | 3 | Test subjects for fixed split |
| `--face` | retina | `dlib` \| `retina` (RetinaFace default) |
| `--stress` | on | Run stress test (gating ON vs OFF on synthetic occlusion) |
| `--no-stress` | — | Skip stress test |
| `--stress-frames` | 20 | Max frames per video (legacy) or per-clip (clip) |
| `--stress-opacities` | 0,0.5,1 | Opacity levels for legacy stress test |
| `--fold` | 0 | Fold index for kfold/loso (0-based) |
| `--defer-test` | on | [clip] Skip test extraction until stress test (saves time) |
| `--benchmark` | off | Run latency benchmark for paper (saves latency_report.json) |

### Results saved to `results/run_YYYYMMDD_HHMMSS/`

| File | Description |
|------|-------------|
| `config.json` | Run configuration |
| `training_history.json` | Loss and accuracy per epoch |
| `training_curves.png` | Training loss and accuracy plots |
| `eval_metrics.json` | Test accuracy, precision, recall, F1 |
| `stress_test_details.csv` | Per-frame p_eye, p_mouth, gates, predictions per condition |
| `stress_test_summary.csv` | Gating ON vs OFF accuracy per condition |
| `gating_on_vs_off.png` | Bar chart comparing gating ON vs OFF |
| `opacity_analysis.png` | Accuracy vs opacity level (eye, mouth, both) |
| `gates_vs_opacity.png` | Gate values and occlusion probs vs opacity |
| `model_best.pt` | Best model checkpoint |

---

## Programmatic Usage

### 1. Training (Clean)

```python
from data_loading import load_csv_video_data
from split_generator import create_splits, get_subject_ids
from pipeline import extract_features_stratified

# Build splits from fixed/kfold/loso
csv_data = load_csv_video_data('Data')
subjects = get_subject_ids(csv_data)
splits_list = create_splits(subjects, mode='fixed', num_test=3)
# Map to pipeline format: {'train': [video_keys], 'test': [...], 'val': [...]}
# (see run_train_eval._splits_from_fixed_or_fold)

train, val, test = extract_features_stratified(
    csv_data, splits, face_detector, feat_extractor, occ_model,
    num_samples_per_video=100,  # or None for all
    val_ratio=0.20, random_state=42
)
```

### 2. Training (Occlusion-Aware, Experiment A+)

```python
from pipeline import extract_features_with_augmentation
from synthetic_occlusion import apply_synthetic_occlusion
# ...
train, val, test = extract_features_with_augmentation(
    csv_data, splits, face_detector, feat_extractor, occ_model,
    apply_synthetic_occlusion, opacity_levels=[0.3, 0.5, 0.7, 0.9, 1.0],
    aug_clean_fraction=0.60, ...
)
```

### 3. Stress Testing

The stress test is **integrated** into `run_train_eval.py` (run by default):

- Applies synthetic occlusion (eye, mouth, both) at configurable opacity levels
- Extracts features from occluded frames, runs inference with gating ON and OFF
- Saves `stress_test_results.csv`, `gating_on_vs_off.png`, `opacity_analysis.png`

Standalone usage:

```python
from stress_test import run_stress_test
stress_df = run_stress_test(
    csv_data, test_keys, model, face_detector, feat_extractor, occ_model,
    trainer, device='cuda', opacity_levels=[0.0, 0.5, 1.0],
    max_frames_per_video=30
)
```

See `experiment_a_gating_stress_test.ipynb` for notebook-based workflow.

---

## Clip Strategy (STRATEGY_DESIGN.md) — Default

The clip strategy implements the design from `docs/STRATEGY_DESIGN.md`:

- **Clip-level sampling:** T=32, stride 16 (train/val), stride 32 (test)
- **FPS downsampling:** 15 fps
- **Temporal val split:** Last 20% of train subjects (no leakage)
- **Regime-based augmentation:** 55% clean, 45% augmented (persistent/transient eye/mouth/both)
- **Label-aware caps:** Cap persistent occlusion on Yawn/EyeClosed clips
- **Stress test:** 6 conditions — clean + persistent_eye/mouth/both + transient_eye/mouth (opacity 0.8)

### Training

```python
from pipeline import extract_features_for_clips

train, val, test, test_clips = extract_features_for_clips(
    csv_data, split_config, face_detector, feat_extractor, occ_model,
    val_ratio=0.20, max_train_clips=100, max_val_clips=25
)
```

### Stress Testing (Clip Strategy)

- Uses test clips; each frame evaluated in 6 conditions
- Saves `stress_test_details.csv` with p_eye, p_mouth, gate_face, gate_eye, gate_mouth per frame/condition

---

## Environment

- Python 3.11, PyTorch, OpenCV, dlib or InsightFace
- See `requirements.txt`

---

## Evaluation Split Modes

```python
from split_generator import create_splits, get_train_val_test_clips
from clip_sampler import extract_clips_for_videos

# Fixed split (1 run, fast)
splits = create_splits(subjects, mode='fixed', num_test=3)

# k-fold (e.g. 5 runs, fewer than LOSO)
splits = create_splits(subjects, mode='kfold', k=5)

# LOSO (15 runs)
splits = create_splits(subjects, mode='loso')

for fold_idx, split_config in enumerate(splits):
    train_clips, val_clips, test_clips = get_train_val_test_clips(
        clips_per_video, split_config, val_ratio=0.20
    )
    # ... train on fold_idx ...
```

---

## Training + Evaluation Flow (from .py files)

```
run_train_eval.py
├── data_loading.load_csv_video_data()
├── split_generator.create_splits(mode='fixed'|'kfold'|'loso')
├── pipeline.extract_features_for_clips() [clip] or extract_features_stratified() [legacy]
├── datasets.DriverStateDataset
├── transformer_enhanced.EnhancedOcclusionAwareTransformer
├── trainer_enhanced.TinyTransformerTrainer
│   ├── train_epoch()  × N epochs
│   └── evaluate()     on val_loader
├── evaluation.compute_metrics_on_loader()  on test_loader
├── stress_test.run_stress_test()           synthetic occlusion, gating ON/OFF (clips or legacy)
└── Save to results/run_YYYYMMDD_HHMMSS/
    ├── config.json, training_history.json, training_curves.png
    ├── eval_metrics.json, model_best.pt
    └── stress_test_details.csv, stress_test_summary.csv, gating_on_vs_off.png, opacity_analysis.png, gates_vs_opacity.png
```

**Key modules:**
- `run_train_eval.py` — single entry point; wires everything
- `split_generator` — fixed / k-fold / LOSO splits; temporal val
- `clip_sampler` — clip extraction with FPS downsampling
- `occlusion_augmentation` — regime assignment, persistent/transient occlusion
- `pipeline` — feature extraction (clip or frame-level)
- `trainer_enhanced` — training loop with gate alignment loss
- `evaluation` — metrics on test set

---

## Reproducibility

- Set `SEED=42` in config
- Use deterministic split (video-level for test, temporal for val)
- Log run config to `run_config.json`
