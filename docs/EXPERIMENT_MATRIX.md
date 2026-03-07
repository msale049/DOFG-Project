# Experiment Matrix

## Dimensions

| Dimension | Values |
|-----------|--------|
| Training | Clean only \| Occlusion-aware (augmented) |
| Gating | ON \| OFF |
| Clip length T | 32 \| 16 (ablation) |
| Test condition | Clean \| Persistent eye \| Persistent mouth \| Persistent both \| Transient eye \| Transient mouth |

## Why Keep Clean Training?

Gating is learned during training, but with **clean data** the occlusion estimator rarely outputs high probabilities (DMD has little real occlusion). Gate alignment loss gets weak supervision. **Clean training** is the baseline (Experiment A) to compare against **occlusion-aware training**, which provides strong supervision from synthetic occlusion. Both are needed to show whether augmentation helps.

## Full Matrix (2 × 2 × 2 × 6 = 48 cells)

For each combination of (training, gating, T), evaluate on 6 test conditions.

| # | Training | Gating | T | Test conditions |
|---|----------|--------|---|-----------------|
| 1 | Clean | ON | 32 | All 6 |
| 2 | Clean | OFF | 32 | All 6 |
| 3 | Occlusion-aware | ON | 32 | All 6 |
| 4 | Occlusion-aware | OFF | 32 | All 6 |
| 5 | Clean | ON | 16 | All 6 |
| 6 | Clean | OFF | 16 | All 6 |
| 7 | Occlusion-aware | ON | 16 | All 6 |
| 8 | Occlusion-aware | OFF | 16 | All 6 |

## Priority Order for Running

1. **Baseline:** Clean training, Gating ON, T=32, clean test
2. **Gating ablation:** Same, Gating OFF
3. **Stress tests:** Same config, each of 5 stress conditions
4. **Occlusion-aware:** Augmented training, Gating ON, T=32, all 6 conditions
5. **T ablation:** T=16 for key configs
6. **Full matrix:** Remaining cells

## Metrics to Report

- Accuracy (overall, per class)
- Macro F1
- Per-condition accuracy (clean vs each stress)
- Gate statistics (mean, std) under stress
- Latency (optional)

## Evaluation Splits (choose one)

| Mode | Folds | Runs per config | Use case |
|------|-------|-----------------|----------|
| **Fixed split** | 1 | 1 | Fast iteration, development |
| **k-fold** | k (e.g. 5) | k | Fewer trainings than LOSO; report mean ± std |
| **LOSO** | 15 | 15 | Full subject-wise evaluation |

**Default:** k-fold (k=5) to reduce trainings vs LOSO, with option to use fixed split when needed. Both implementations available in `split_generator.py`.
