#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 24:00:00
#SBATCH -J final_train
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

echo "============================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "============================================"

# Standalone launcher for CSP post-training.
# Usage:
#   bash slurm_csp_vocab_train.sh <VISION_BACKBONE> <DATASET> [BASE_CKPT] [WANDB_PROJECT]
# Examples:
#   bash slurm_csp_vocab_train.sh ijepa cspref_mit_states /path/to/base.ckpt
#   bash slurm_csp_vocab_train.sh vjepa cspref_ut_zappos "" my-wandb-project

if [[ $# -lt 2 ]]; then
  echo "Usage: bash slurm_csp_vocab_train.sh <VISION_BACKBONE> <DATASET> [BASE_CKPT] [WANDB_PROJECT]"
  exit 2
fi

VISION_BACKBONE="${1}"
DATASET="${2}"
BASE_CKPT="${3:-}"
W_PROJECT="${4:-}"

if [[ -n "${W_PROJECT}" ]]; then
  export WANDB_PROJECT="${W_PROJECT}"
fi

FUSION_TYPE="${FUSION_TYPE:-clip_similarity}"
CMD=(uv run python run_csp_vocab_train.py --vision-backbone "${VISION_BACKBONE}" --dataset "${DATASET}" --fusion-type "${FUSION_TYPE}")
if [[ -n "${BASE_CKPT}" ]]; then
  CMD+=(--base-checkpoint "${BASE_CKPT}")
fi

echo "[csp_vocab_train] dataset: ${DATASET}"
echo "[csp_vocab_train] vision backbone: ${VISION_BACKBONE}"
echo "[csp_vocab_train] fusion type: ${FUSION_TYPE}"
if [[ -n "${BASE_CKPT}" ]]; then
  echo "[csp_vocab_train] base checkpoint: ${BASE_CKPT}"
else
  echo "[csp_vocab_train] base checkpoint: <none>"
fi
if [[ -n "${W_PROJECT}" ]]; then
  echo "[csp_vocab_train] W&B project: ${WANDB_PROJECT}"
else
  echo "[csp_vocab_train] W&B project: <default from env/.env/code>"
fi
echo "[csp_vocab_train] command: ${CMD[*]}"

"${CMD[@]}"

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"