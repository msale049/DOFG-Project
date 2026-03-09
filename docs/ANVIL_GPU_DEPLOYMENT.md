# Anvil GPU Deployment Guide

## Files You Need (End-to-End Flow)

| Category | Count |
|----------|-------|
| Python source | 20 |
| Model weights | 3 |
| InsightFace models | 2 (auto-downloaded) |
| Data | `Data/` folder |

**Output files** (saved to `results/run_YYYYMMDD_HHMMSS/`): **11–12 files** per run
- `config.json`, `training_history.json`, `training_curves.png`, `eval_metrics.json`
- `model_best.pt`
- `occlusion_visualization.png` (grid of train + stress + legacy samples)
- `stress_test_details.csv`, `stress_test_summary.csv`, `gating_on_vs_off.png`, `opacity_analysis.png`, `gates_vs_opacity.png`
- `latency_report.json` (if `--benchmark`)

### 1. Python Source (20 files)

| File | Purpose |
|------|---------|
| `run_train_eval.py` | Main entry point |
| `config.py` | Paths, constants, CLIP_CONFIG, AUGMENTATION_CONFIG |
| `data_loading.py` | CSV loading, frame sampling |
| `pipeline.py` | Feature extraction (clip + legacy) |
| `clip_sampler.py` | Clip extraction, FPS downsampling |
| `split_generator.py` | Train/val/test splits, temporal val |
| `occlusion_augmentation.py` | Regime assignment, persistent/transient occlusion |
| `synthetic_occlusion.py` | Low-level eye/mouth overlay |
| `datasets.py` | DriverStateDataset |
| `feature_extraction.py` | ResNet34 region features |
| `occlusion_estimator.py` | ResNet34 occlusion probs |
| `face_detection_retinaface.py` | RetinaFace + dlib landmarks |
| `face_detection_dlib.py` | Fallback dlib detector |
| `transformer_enhanced.py` | Occlusion-aware transformer |
| `trainer_enhanced.py` | Training loop |
| `evaluation.py` | Metrics |
| `stress_test.py` | Stress test (6 conditions) |
| `visualize_occlusion.py` | Occlusion grid PNG (train + stress + legacy) |
| `ablation_utils.py` | Gate disabling for ablation |
| `utils.py` | Bbox/landmark helpers |

### 2. Model Weights (in project root or paths in `config.py`)

| File | Used by |
|------|---------|
| `resnet34_portable.state_dict.pt` | Feature extractor |
| `resnet34_occlusion.pt` | Occlusion estimator |
| `shape_predictor_68_face_landmarks.dat` | dlib landmarks (RetinaFace uses it too) |

### 3. InsightFace Models (auto-downloaded to `~/.insightface/models/`)

- `buffalo_sc/det_500m.onnx`
- `buffalo_sc/w600k_mbf.onnx`

If RetinaFace fails (e.g. GLIBCXX on HPC), use `--face dlib` and you only need the dlib shape predictor.

### 4. Data

```
Data/
├── Sub1/
│   ├── sub1_video.mp4
│   └── sub1_video.csv
├── Sub2/
│   └── ...
└── Sub15/
    └── ...
```

### 5. Dependencies

Use `requirements-gpu.txt` (includes PyTorch with CUDA):

```bash
uv pip install -r requirements-gpu.txt
```

---

## Pipeline Flow (No Double Fetch)

### Phase 1: Metadata Only (No Video Read)

```
load_csv_video_data('Data')
  └── Reads CSV files only → {video_key: {video_path, annotations, subject}}
      No frames loaded.

create_splits(subjects) → split_config {train_subjects, test_subjects}

extract_clips_for_videos(csv_data) → clips_per_video
  └── Uses annotations only. No video read.

get_train_val_test_clips(clips_per_video, split_config)
  └── train_clips, val_clips, test_clips (clip metadata only)
```

### Phase 2: Feature Extraction (Single Pass per Clip)

For **each clip** in train_clips, then val_clips, then test_clips:

```
1. Open VideoCapture(video_path)  ← one open per clip
2. For each frame in clip:
   a. cap.set(POS_FRAMES, frame_num)
   b. cap.read() → bgr
   c. face_detector.detect_face_and_landmarks(bgr)
   d. [TRAIN ONLY] apply_occlusion_to_frame() if regime != clean
   e. feat_extractor.extract_region_features(bgr, bbox, eyes, mouth)
   f. occ_model.predict_probs(rgb, bbox)
   g. Append sample {features, occlusion_info, label} to train/val/test_samples
3. cap.release()
```

**Result:** `train_samples`, `val_samples`, `test_samples` — all in memory. No video read again for training/val/test.

### Phase 3: Training & Clean Evaluation

```
DriverStateDataset(train_samples) → train_loader
DriverStateDataset(val_samples)   → val_loader
DriverStateDataset(test_samples)  → test_loader

Train N epochs on train_loader, validate on val_loader.
Evaluate on test_loader (clean) → eval_metrics.json
```

**No video read.** All from pre-extracted samples.

### Phase 4: Stress Test (Separate Video Read)

Stress test **re-reads** test frames from video. This is intentional:

- Clean evaluation uses pre-extracted features (no occlusion).
- Stress test applies 6 occlusion conditions per frame → needs fresh frames to overlay occlusion, then run face detect → feature extract → occlusion estimator → model.

```
For each test_clip (or sampled frames in legacy):
  Open VideoCapture(video_path)
  For each frame:
    cap.read() → bgr
    For each of 6 conditions (clean, persistent_eye, persistent_mouth, persistent_both, transient_eye, transient_mouth):
      Apply occlusion to frame
      face_detect → feature_extract → occ_estimator → model (gating ON/OFF)
      Store p_eye, p_mouth, gates, pred
  cap.release()
```

**Summary:** Test video is read twice:
1. In Phase 2 (extract_features_for_clips) → for clean test_samples.
2. In Phase 4 (stress test) → for synthetic occlusion conditions.

Train and val are read only once (Phase 2).

---

## Occlusion Flow

| Stage | Train | Val | Test (clean) | Stress test |
|-------|-------|-----|--------------|-------------|
| Occlusion applied? | Yes (55% clean, 45% regime) | No | No | Yes (6 conditions) |
| When | During feature extraction | — | — | During stress test |
| Regime | assign_regime_to_clip → persistent/transient | — | — | Fixed: clean + 5 stress |

Train occlusion: `assign_regime_to_clip` → regime (clean, persistent_eye, etc.) + opacity. Then `apply_occlusion_to_frame` per frame before feature extraction.

---

## Default Settings (no arguments)

```bash
python run_train_eval.py
```

Runs with:
- `--strategy clip` (STRATEGY_DESIGN)
- `--mode fixed` (3 subjects test)
- `--face retina` (RetinaFace)
- `--defer-test` (load test only during stress test)
- `--stress` (run stress test)
- `--epochs 5`, `--batch 16`, `--seed 42`
- No `--benchmark` (add `--benchmark` for latency report)

---

## Complete End-to-End Command

RetinaFace + fixed split + clip + defer test + latency report:

```bash
python run_train_eval.py --strategy clip --mode fixed --face retina --benchmark --epochs 20
```

- `--defer-test` is default on → test frames loaded only once (during stress test)
- `--benchmark` → saves `latency_report.json` with per-phase ms

---

## Anvil GPU: Requesting GPU for Python

**Important:** If you have an **AI allocation** (e.g. `ele250032-ai`), you must use `--partition=ai`, not `-p gpu`. Otherwise you get: *"You are submitting a job to a non-AI partition while using an AI allocation"*.

### Option 1: Interactive (salloc)

**AI allocation** (use `-p ai` — required for `ele250032-ai`; `-p gpu` will fail):

```bash
salloc -A ele250032-ai -p ai -N 1 -n 1 -t 14:00:00 --gpus-per-node=1

# Once allocated:
# Activate your conda env (e.g. dms_exp_a at /anvil/projects/x-ele250032/envs/dms_exp_a):
conda activate /anvil/projects/x-ele250032/envs/dms_exp_a
# Or if using modules instead:
# module load modtree/gpu

cd /path/to/DOFG\ Project

# Run in background so it survives laptop disconnect:
nohup python run_train_eval.py --strategy clip --epochs 20 --benchmark > train.log 2>&1 &
tail -f train.log   # monitor progress
```

**Standard GPU allocation** (use `-p gpu`):

```bash
salloc -A YOUR_GPU_ALLOCATION -p gpu -N 1 -n 1 -t 4:00:00 --gpus-per-node=1
```

### Option 2: Batch job (sbatch)

Create `run_dofg_gpu.sh`:

**AI allocation** (14 hours):

```bash
#!/bin/bash
#SBATCH -A ele250032-ai
#SBATCH -p ai
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH -t 14:00:00
#SBATCH -J dofg_train
#SBATCH -o dofg_%j.out
#SBATCH -e dofg_%j.err

module purge
module load modtree/gpu
# Or: module load anaconda && conda activate your_env

cd /path/to/DOFG\ Project
python run_train_eval.py --strategy clip --mode fixed --face retina --benchmark --epochs 20
```

Submit: `sbatch run_dofg_gpu.sh`

### Using your conda env (`dms_exp_a`)

**Order matters:** Load CUDA/GPU module *before* activating conda, so PyTorch sees the GPU.

```bash
# 1. Get GPU node (salloc)
salloc -A ele250032-ai -p ai -N 1 -n 1 -t 14:00:00 --gpus-per-node=1

# 2. Load modules FIRST (CUDA/drivers before conda)
module load anaconda 2>/dev/null || true
# If nvidia-smi not found, try: module load modtree/gpu  or  module load cuda

# 3. Activate conda env
conda activate /anvil/projects/x-ele250032/envs/dms_exp_a

# 4. Verify GPU visible to PyTorch
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# 5. Run pipeline
cd /anvil/projects/x-ele250032/Occlusion_Pipeline
nohup python run_train_eval.py --strategy clip --epochs 20 --benchmark > train.log 2>&1 &
```

In `run_dofg_gpu.sh` (sbatch):
```bash
module load anaconda 2>/dev/null || true
conda activate /anvil/projects/x-ele250032/envs/dms_exp_a
```

### salloc output explained

| Message | Meaning |
|---------|---------|
| `Granted job allocation 15535569` | Job accepted; **15535569** is your job ID |
| `Waiting for resource configuration` | SLURM is finding a GPU node |
| `Nodes h014 are ready for job` | You're on compute node **h014** (GPU node; prompt changes to `x-msaleem@h014.anvil`) |

### Check remaining time

```bash
# Your jobs (shows TIME_LIMIT and TIME used):
squeue -u $USER

# Detailed info for a job (replace JOBID with e.g. 15535569):
scontrol show job JOBID
# Look for: TimeLimit=14:00:00, RunTime=00:23:15 (elapsed)
```

Remaining time ≈ `TimeLimit - RunTime`.

### GPU OOM (out of memory) fixes — keep RetinaFace on GPU

**Why Jupyter works but terminal fails:** Jupyter may use a different node/GPU with more memory, or a fresh session. Terminal `salloc` on h014 might share a smaller GPU. PyTorch now loads *before* RetinaFace so it gets clean GPU memory (RetinaFace's failed CUDA init can fragment GPU).

To run **everything on GPU** (RetinaFace + PyTorch) with limited VRAM:

1. **Smaller RetinaFace input** (default is 480; use 320 if still OOM):
   ```bash
   python run_train_eval.py --det-size 320 --strategy clip --epochs 20 --benchmark
   ```

2. **Reduce batch size**:
   ```bash
   python run_train_eval.py --det-size 480 --batch 8 --strategy clip --epochs 20
   ```

3. **Combined (lowest memory)**:
   ```bash
   python run_train_eval.py --det-size 320 --batch 8 --strategy clip --epochs 20
   ```

5. **With nohup**:
   ```bash
   nohup python run_train_eval.py --det-size 320 --batch 8 --strategy clip --epochs 20 --benchmark > train.log 2>&1 &
   ```

If still OOM, use `--face dlib` (CPU face detection).

**H100 80GB still OOM?** Try (in order):

1. **PyTorch cu124** (better H100 support; driver 13.0):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

2. **Memory allocator** (reduce fragmentation):
   ```bash
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run_train_eval.py --strategy clip --epochs 20
   ```

3. **Run GPU diagnostic** (isolates env vs pipeline):
   ```bash
   python test_gpu.py
   ```
   If test_gpu.py passes but pipeline OOMs, share the "GPU before feat_extractor" line from the log.

4. **Debug exact line**:
   ```bash
   CUDA_LAUNCH_BLOCKING=1 python run_train_eval.py --max-train-clips 5 --epochs 1
   ```

The `pthread_setaffinity_np` messages from ONNX are warnings only; they do not stop the run.

### OMP / thread-affinity fix (Jupyter or scripts)

To avoid `pthread_setaffinity_np failed` on HPC, set these **before** importing torch/ONNX:

```python
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
```

`run_train_eval.py` sets these automatically. For Jupyter, put them in the **first cell** before any imports.

### Verify GPU is being used

```python
import torch, onnxruntime as ort
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
print('ORT providers:', ort.get_available_providers())

# After loading models:
print('feat_extractor device:', next(feat_extractor.model.parameters()).device)
print('occ_model device:     ', next(occ_model.model.parameters()).device)
```

Or from terminal before running:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### Key SLURM options

| Option | Value | Notes |
|--------|-------|-------|
| `-A` | your allocation | e.g. `ele250032-ai` (AI) or `ele250032-gpu` |
| `-p ai` | partition | **Required for AI allocations** |
| `-p gpu` | partition | For standard GPU allocations only |
| `--gpus-per-node=1` | 1 GPU | Required for GPU jobs |
| `-t` | 14:00:00 | Max runtime (hh:mm:ss); use 14h for full runs |

See [Purdue RCAC Anvil GPU examples](https://www.rcac.purdue.edu/knowledge/anvil/run/examples/slurm/gpu).

---

## Split Modes: Fixed, k-Fold, LOSO

```bash
# Fixed split (default): 3 subjects for test, rest for train
python run_train_eval.py --mode fixed --num-test 3

# k-Fold: 5 folds, use fold 0 (first fold)
python run_train_eval.py --mode kfold --k 5 --fold 0

# Run all 5 folds (separate runs)
for f in 0 1 2 3 4; do
  python run_train_eval.py --mode kfold --k 5 --fold $f --epochs 20
done

# LOSO: 15 folds (one subject out each), use fold 0
python run_train_eval.py --mode loso --fold 0

# Run all 15 LOSO folds
for f in $(seq 0 14); do
  python run_train_eval.py --mode loso --fold $f --epochs 20
done
```

---

## Deterministic Training

**Yes.** With the same `--seed` (default 42):

- **Clips:** `hash((video_key, clip_start, T, seed))` → same clips, same order
- **Regime:** `assign_regime_to_clip` uses that hash → same regime per clip
- **Split:** `create_splits` uses seed → same train/test subjects
- **Transient segment:** `clip_seed = hash((video_key, clip_start, seed))` → same subsegment

Running multiple times with the same seed produces identical train/val data and occlusion assignment.
