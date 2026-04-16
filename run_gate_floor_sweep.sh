#!/bin/bash
# run_gate_floor_sweep.sh
# =======================
# Parametric sweep over gate_floor values for the DOFG gated transformer.
#
# OPTIMIZED: Uses run_gate_floor_sweep.py which extracts features ONCE per fold,
# caches stress-test features, and only repeats training + inference per floor.
# Saves ~85% of compute vs. running each floor independently.
#
# Usage (fixed split, single fold):
#   bash run_gate_floor_sweep.sh                   # run all 7 values
#   bash run_gate_floor_sweep.sh 0.50 0.70         # run only specified values
#   bash run_gate_floor_sweep.sh --skip-existing   # skip floors already completed
#   bash run_gate_floor_sweep.sh \
#       --epochs 1 \
#       --stress-frames 4 \
#       --max-train-clips 8 --max-val-clips 4 --max-test-clips 4 \
#       --sweep-name gate_floor_sweep_smoke        # quick end-to-end smoke test
#
# Usage (k-fold cross-validation):
#   bash run_gate_floor_sweep.sh --kfold            # 5-fold CV for all 7 values
#   bash run_gate_floor_sweep.sh --kfold 0.50 0.70  # 5-fold CV for specific values
#   bash run_gate_floor_sweep.sh --kfold --folds 0  # run only a specific fold
#
# Legacy (independent per-floor runs via run_train_eval.py):
#   bash run_gate_floor_sweep.sh --legacy           # use old per-floor mode

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON=".venv/bin/python3"

ALL_FLOORS=(0.05 0.10 0.30 0.50 0.70 0.90 1.00)

SKIP_EXISTING=false
KFOLD=false
LEGACY=false
K=5
FOLDS=""
NUM_TEST=3
EPOCHS=15
BATCH=16
STRESS_FRAMES=20
SWEEP_NAME=""
MAX_TRAIN_CLIPS=""
MAX_VAL_CLIPS=""
MAX_TEST_CLIPS=""
FLOORS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --kfold)
            KFOLD=true
            shift
            ;;
        --legacy)
            LEGACY=true
            shift
            ;;
        --k)
            K="$2"
            shift 2
            ;;
        --folds)
            FOLDS="$2"
            shift 2
            ;;
        --num-test)
            NUM_TEST="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch)
            BATCH="$2"
            shift 2
            ;;
        --stress-frames)
            STRESS_FRAMES="$2"
            shift 2
            ;;
        --sweep-name)
            SWEEP_NAME="$2"
            shift 2
            ;;
        --max-train-clips)
            MAX_TRAIN_CLIPS="$2"
            shift 2
            ;;
        --max-val-clips)
            MAX_VAL_CLIPS="$2"
            shift 2
            ;;
        --max-test-clips)
            MAX_TEST_CLIPS="$2"
            shift 2
            ;;
        --*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            FLOORS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#FLOORS[@]} -eq 0 ]]; then
    FLOORS=("${ALL_FLOORS[@]}")
fi

FLOORS_CSV=$(IFS=,; echo "${FLOORS[*]}")

if [[ "$LEGACY" == true ]]; then
    # ══════════════════════════════════════════════════════════════════════════
    # Legacy mode: each floor gets a full independent run_train_eval.py call
    # ══════════════════════════════════════════════════════════════════════════
    if [[ -n "$SWEEP_NAME" ]]; then
        SWEEP_DIR="$SWEEP_NAME"
    elif [[ "$KFOLD" == true ]]; then
        SWEEP_DIR="gate_floor_sweep_kfold"
    else
        SWEEP_DIR="gate_floor_sweep"
    fi

    if [[ "$KFOLD" == true ]]; then
        MODE_LABEL="kfold (k=${K}) [legacy]"
    else
        MODE_LABEL="fixed split [legacy]"
    fi

    echo "=============================================="
    echo "  Gate Floor Sweep (LEGACY — slow)"
    echo "  Mode: ${MODE_LABEL}"
    echo "  Floors: ${FLOORS[*]}"
    if [[ -n "$FOLDS" ]]; then
        echo "  Folds: ${FOLDS}"
    fi
    echo "  Epochs: ${EPOCHS} | Batch: ${BATCH} | Stress frames: ${STRESS_FRAMES}"
    echo "=============================================="

    COMMON_ARGS=(
        --samples 0 --epochs "$EPOCHS" --batch "$BATCH" --face retina --det-size 640
        --strategy clip --train-opacity-values "0.4,0.6,0.8,1.0"
        --stress --stress-frames "$STRESS_FRAMES" --stress-opacities "0.4,0.6,0.8,1.0"
        --no-defer-test --class-weighted --gate-supervision gt
        --gate-weight 0.5 --seed 42
    )

    if [[ -n "$MAX_TRAIN_CLIPS" ]]; then
        COMMON_ARGS+=(--max-train-clips "$MAX_TRAIN_CLIPS")
    fi
    if [[ -n "$MAX_VAL_CLIPS" ]]; then
        COMMON_ARGS+=(--max-val-clips "$MAX_VAL_CLIPS")
    fi
    if [[ -n "$MAX_TEST_CLIPS" ]]; then
        COMMON_ARGS+=(--max-test-clips "$MAX_TEST_CLIPS")
    fi

    for FLOOR in "${FLOORS[@]}"; do
        RUN_NAME="${SWEEP_DIR}/floor_${FLOOR}"
        RUN_DIR="results/${RUN_NAME}"

        if [[ "$KFOLD" == true ]]; then
            DONE_MARKER="${RUN_DIR}/crossval_summary.json"
            if [[ "$SKIP_EXISTING" == true ]] && [[ -f "$DONE_MARKER" ]]; then
                echo ">>> Skipping floor=${FLOOR} (exists)"
                continue
            fi
            echo ""
            echo "  Running gate_floor=${FLOOR} (k-fold, k=${K})"
            CROSSVAL_ARGS=(
                --data Data --mode kfold --k "$K"
                --results-root results
                --run-name "${SWEEP_DIR}/floor_${FLOOR}"
                --continue-on-error --skip-existing
            )
            if [[ -n "$FOLDS" ]]; then
                CROSSVAL_ARGS+=(--folds "$FOLDS")
            fi
            $PYTHON -u run_crossval.py \
                "${CROSSVAL_ARGS[@]}" -- "${COMMON_ARGS[@]}" --gate-floor "$FLOOR"
        else
            if [[ "$SKIP_EXISTING" == true ]] && [[ -f "${RUN_DIR}/eval_metrics.json" ]]; then
                echo ">>> Skipping floor=${FLOOR} (exists)"
                continue
            fi
            echo ""
            echo "  Running gate_floor=${FLOOR} (fixed split)"
            $PYTHON -u run_train_eval.py \
                --data Data --results-root results --run-name "$RUN_NAME" \
                --mode fixed --num-test "$NUM_TEST" --fold 0 \
                "${COMMON_ARGS[@]}" --gate-floor "$FLOOR"
        fi
        echo "  >>> Completed gate_floor=${FLOOR}"
    done

    $PYTHON -u aggregate_gate_floor_sweep.py --sweep-dir "results/${SWEEP_DIR}"
    echo "Done."
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════════════
# Optimized mode (default): features extracted ONCE, shared across all floors
# ══════════════════════════════════════════════════════════════════════════════

if [[ "$KFOLD" == true ]]; then
    MODE="kfold"
    MODE_LABEL="kfold (k=${K}) [optimized]"
else
    MODE="fixed"
    MODE_LABEL="fixed split [optimized]"
fi

echo "=============================================="
echo "  Gate Floor Parametric Sweep"
echo "  Mode: ${MODE_LABEL}"
echo "  Floors: ${FLOORS_CSV}"
if [[ -n "$FOLDS" ]]; then
    echo "  Folds: ${FOLDS}"
fi
echo "  Epochs: ${EPOCHS} | Batch: ${BATCH} | Stress frames: ${STRESS_FRAMES}"
if [[ -n "$MAX_TRAIN_CLIPS" ]] || [[ -n "$MAX_VAL_CLIPS" ]] || [[ -n "$MAX_TEST_CLIPS" ]]; then
    echo "  Clip caps: train=${MAX_TRAIN_CLIPS:-all}, val=${MAX_VAL_CLIPS:-all}, test=${MAX_TEST_CLIPS:-all}"
fi
echo "  Skip existing: $SKIP_EXISTING"
echo "=============================================="

SWEEP_ARGS=(
    --data Data
    --results-root results
    --mode "$MODE"
    --k "$K"
    --num-test "$NUM_TEST"
    --epochs "$EPOCHS"
    --batch "$BATCH"
    --face retina
    --det-size 640
    --stress-frames "$STRESS_FRAMES"
    --gate-weight 0.5
    --seed 42
    --floors "$FLOORS_CSV"
    --stress-opacities "0.4,0.6,0.8,1.0"
)

if [[ "$SKIP_EXISTING" == true ]]; then
    SWEEP_ARGS+=(--skip-existing)
fi
if [[ -n "$FOLDS" ]]; then
    SWEEP_ARGS+=(--folds "$FOLDS")
fi
if [[ -n "$SWEEP_NAME" ]]; then
    SWEEP_ARGS+=(--sweep-name "$SWEEP_NAME")
fi
if [[ -n "$MAX_TRAIN_CLIPS" ]]; then
    SWEEP_ARGS+=(--max-train-clips "$MAX_TRAIN_CLIPS")
fi
if [[ -n "$MAX_VAL_CLIPS" ]]; then
    SWEEP_ARGS+=(--max-val-clips "$MAX_VAL_CLIPS")
fi
if [[ -n "$MAX_TEST_CLIPS" ]]; then
    SWEEP_ARGS+=(--max-test-clips "$MAX_TEST_CLIPS")
fi

$PYTHON -u run_gate_floor_sweep.py "${SWEEP_ARGS[@]}"

echo "Done."
