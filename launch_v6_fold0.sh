#!/bin/bash
# launch_v6_fold0.sh
# ==================
# Single-fold (fold 0) k-fold run of the V6 gating pipeline at gate_floor=0.05.
#
# Matches the prior gate_floor_sweep_kfold_full/floor_0.05/fold_00 configuration
# exactly (seed, k, num_test, epochs, stress frames, discrete train opacities)
# so the resulting metrics are directly comparable. The new V6 flags
# (attention-bias gating, logit-bias head, estimator calibration, gate dropout)
# are the defaults of run_gate_floor_sweep.py, so no extra args are needed.
#
# Writes results to  results/gate_floor_sweep_v6_kfold/floor_0.05/fold_00/
# Writes stdout to   logs/v6_fold0_<timestamp>.log
#
# Usage:
#   nohup bash launch_v6_fold0.sh > /dev/null 2>&1 &
#
# Or to keep in foreground:
#   bash launch_v6_fold0.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON=".venv/bin/python3"
mkdir -p logs results

TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/v6_fold0_${TS}.log"

echo "==============================================" | tee "$LOG"
echo "  V6 single-fold launch"                         | tee -a "$LOG"
echo "  Start: $(date -Iseconds)"                      | tee -a "$LOG"
echo "  Log:   $LOG"                                    | tee -a "$LOG"
echo "  Host:  $(hostname)"                             | tee -a "$LOG"
echo "==============================================" | tee -a "$LOG"

exec $PYTHON -u run_gate_floor_sweep.py \
    --data Data \
    --results-root results \
    --sweep-name gate_floor_sweep_v6_kfold \
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
    --gate-dropout 0.1 \
    >> "$LOG" 2>&1
