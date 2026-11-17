#!/usr/bin/env bash
# Train the AdamW and Muon baselines for one model size on one dataset, one after the other.
#
#   bash scripts/run_baselines.sh <model> <dataset> [num_gpus] [extra train.py args...]
#
#   bash scripts/run_baselines.sh gpt2_small openwebtext 8
#   bash scripts/run_baselines.sh llama_medium fineweb 4 --wandb_log=True
#   bash scripts/run_baselines.sh gpt2_small c4 1 --max_iters=20000
#
# <model>: gpt2_small | gpt2_medium | gpt2_large | llama_small | llama_medium | llama_large
# Outputs go to out/<dataset>/<model>_<optimizer>/.
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,13p' "$0"
    exit 1
fi
MODEL=$1
DATASET=$2
NGPU=${3:-1}
shift $(( $# >= 3 ? 3 : 2 ))

cd "$(dirname "$0")/.."
for OPT in adamw muon; do
    CONFIG=config/${MODEL}_${OPT}.py
    if [ "$NGPU" -gt 1 ]; then
        torchrun --standalone --nproc_per_node="$NGPU" train.py "$CONFIG" --dataset="$DATASET" "$@"
    else
        python train.py "$CONFIG" --dataset="$DATASET" "$@"
    fi
done
