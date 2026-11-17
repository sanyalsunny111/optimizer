#!/usr/bin/env bash
# Multi-node training with SLURM: one torchrun launcher per node, 8 GPUs per node.
#
#   sbatch scripts/slurm_multinode.sh config/gpt2_medium_adamw.py --dataset=fineweb
#
# Adapt the #SBATCH lines (partition, account, time, GPUs per node) to your cluster.
# The dataset (data/<dataset>/train.bin, val.bin, meta.json) must be readable from every
# node, e.g. on a shared file system. With a time limit, split the run into several jobs:
# the first job with --stop_iter=N, the following ones with --init_from=resume (see README).
#
#SBATCH --job-name=optimizer
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}
export OMP_NUM_THREADS=8

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
srun torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    train.py "$@"
