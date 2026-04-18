#!/bin/bash
# launch_v7_fold0.sh
# ==================
# Single-fold (fold 0) k-fold run of the Phase-1 minimum-viable V7 gating
# pipeline at gate_floor=0.05.
#
# Configuration matches the prior V5 baseline + V6 run exactly (seed, k,
# num_test, epochs, stress frames, discrete train opacities) so results are
# directly comparable to:
#   results/kfold_fold0_full_det_1 legacy opacity/   (V5 baseline)
#   results/gate_floor_sweep_v6_kfold/floor_0.05/fold_00/   (failed V6)
#
# V7 Phase-1 changes vs V6:
#   - attention-bias gating: ON   (default, same as V6)
#   - logit-bias head:       OFF  (was ON in V6; caused clean regression)
#   - estimator calibration: OFF  (was ON in V6)
#   - gate dropout:          0.0  (was 0.1 in V6)
#   - clean-invariance reg:  1.0  (NEW — enforces gate≈1 is a no-op)
#   - checkpoint metric:     macro_f1 (NEW — was val_accuracy)
#
# Writes results to  results/gate_floor_sweep_v7_kfold/floor_0.05/fold_00/
# Writes stdout to   logs/v7_fold0_<timestamp>.log
#
# Usage:
#   nohup bash launch_v7_fold0.sh > /dev/null 2>&1 &
#   echo "launched PID=$!"
#   tail -f logs/v7_fold0_*.log
#
# Or to keep in foreground:
#   bash launch_v7_fold0.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON=".venv/bin/python3"
mkdir -p logs results

TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/v7_fold0_${TS}.log"

echo "==============================================" | tee "$LOG"
echo "  V7 Phase-1 single-fold launch"                | tee -a "$LOG"
echo "  Start: $(date -Iseconds)"                     | tee -a "$LOG"
echo "  Log:   $LOG"                                  | tee -a "$LOG"
echo "  Host:  $(hostname)"                           | tee -a "$LOG"
echo "==============================================" | tee -a "$LOG"

exec $PYTHON -u run_gate_floor_sweep.py \
    --data Data \
    --results-root results \
    --sweep-name gate_floor_sweep_v7_kfold \
    --mode kfold \
    --k 5 \
    --folds 0 \
    --num-test 3 \
    --epochs 15 \
    --batch 16 \
    --face retina \
    --det-size 640 \
    --stress-frames 20 \
    --gate-weight 0.5 \
    --seed 42 \
    --floors 0.05 \
    --stress-opacities "0.4,0.6,0.8,1.0" \
    --gating-mode attention \
    --no-logit-bias \
    --no-estimator-calibration \
    --gate-dropout 0.0 \
    --clean-invariance-weight 1.0 \
    --clean-invariance-thresh 0.1 \
    --checkpoint-metric macro_f1 \
    >> "$LOG" 2>&1
