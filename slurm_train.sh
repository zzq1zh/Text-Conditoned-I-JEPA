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

# Run from the code directory
cd "$SLURM_SUBMIT_DIR"

source ~/.local/bin/env

# Select target and dataset by args or environment variables.
# Usage examples:
#   sbatch slurm_train.sh dinov3 cspref_cgqa
#   TRAIN_TARGET=vjepa TRAIN_DATASET=cspref_ut_zappos sbatch slurm_train.sh
#   sbatch slurm_train.sh two_fusions cspref_mit_states
TARGET="${1:-${TRAIN_TARGET:-two_fusions}}"
DATASET="${2:-${TRAIN_DATASET:-cspref_mit_states}}"

case "${TARGET}" in
  two_fusions|two-stage|two_stage)
    TRAIN_CMD=(bash train_two_fusions_and_push.sh)
    ;;
  dinov3|dino|dino-v3)
    TRAIN_CMD=(uv run python train_csp_clipstyle.py --vision-backbone dino-v3 --dataset "${DATASET}")
    ;;
  ijepa|i-jepa)
    TRAIN_CMD=(uv run python train_csp_clipstyle.py --vision-backbone ijepa --dataset "${DATASET}")
    ;;
  vjepa|v-jepa)
    TRAIN_CMD=(uv run python train_csp_clipstyle.py --vision-backbone v-jepa --dataset "${DATASET}")
    ;;
  ./*.sh|*.sh)
    TRAIN_CMD=(bash "${TARGET}")
    ;;
  *)
    echo "Unsupported target: ${TARGET}"
    echo "Supported: two_fusions, dinov3, ijepa, vjepa, or a custom .sh path"
    exit 2
    ;;
esac

echo "Training target: ${TARGET}"
echo "Dataset: ${DATASET}"
echo "Command: ${TRAIN_CMD[*]}"

if [[ "${TARGET}" == "two_fusions" || "${TARGET}" == "two-stage" || "${TARGET}" == "two_stage" ]]; then
  DATASET="${DATASET}" "${TRAIN_CMD[@]}"
else
  "${TRAIN_CMD[@]}"
fi

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"
