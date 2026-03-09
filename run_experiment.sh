#!/bin/bash
cd /cephfs/users/bashir/DOFG-Project
exec .venv/bin/python3 -u run_train_eval.py \
    --samples 0 \
    --epochs 15 \
    --batch 16 \
    --mode fixed \
    --num-test 3 \
    --face retina \
    --det-size 640 \
    --strategy clip \
    --stress \
    --stress-frames 20 \
    --stress-opacities "0.4,0.6,0.8,1.0" \
    --defer-test \
    --benchmark \
    --seed 42 \
    --class-weighted
