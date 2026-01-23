#!/usr/bin/env bash
# run.sh
# Example script to run fine-tuning with predefined arguments

# Exit immediately on error
set -e

# Activate conda env (optional)
# conda activate mrmd

# Run the fine-tuning script
python -m src.cgmd_paper.scripts.fine_tune \
  --dataset-config configs/datasets/ctc/DIC-C2DH-HeLa.yaml \
  --model-type cyto2 \
  --selector ALL \
  --budget 2 \
  --train-mode fixed \
  --epochs-fixed 100 \
  --lr 1e-1 \
  --wd 1e-4 \
  --runs 1 \
  --features-dir ./cache \
  --overwrite-features \
  --results-csv results.csv
