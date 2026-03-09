# DOFG Pipeline — GPU Run

End-to-end training, evaluation, and stress test for driver state detection with occlusion-aware gating.

## Quick Start (Anvil HPC)

```bash
salloc -A ele250032-ai -p ai -N 1 -n 1 -t 14:00:00 --gpus-per-node=1
module load anaconda 2>/dev/null || true
conda activate /path/to/your/env

cd Occlusion_Pipeline
python run_train_eval.py --strategy clip --epochs 20 --benchmark
```

## Required Files (not in repo — add locally)

- `Data/` — DMD dataset (Sub1..Sub15 with video + CSV)
- `models/resnet34_portable.state_dict.pt` — Feature extractor weights
- `models/resnet34_occlusion.pt` — Occlusion estimator weights
- `models/shape_predictor_68_face_landmarks.dat` — dlib landmarks

## Dependencies

```bash
uv pip install -r requirements-gpu.txt
```

See `docs/ANVIL_GPU_DEPLOYMENT.md` for full setup and troubleshooting.
