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

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "ERROR: python3/python not found in PATH."
  exit 127
fi

# Select training script by arg or environment variable.
# Usage examples:
#   sbatch slurm_train.sh dinov3
#   TRAIN_TARGET=vjepa sbatch slurm_train.sh
#   sbatch slurm_train.sh ./train_ijepa_mit_states_clipstyle.sh
TARGET="${1:-${TRAIN_TARGET:-two_fusions}}"
case "${TARGET}" in
  two_fusions|two-stage|two_stage)
    TRAIN_CMD=(bash train_two_fusions_and_push.sh)
    ;;
  dinov3|dino|dino-v3)
    TRAIN_CMD=("${PYTHON_BIN}" train_mit_states_clipstyle.py --vision-backbone dino-v3)
    ;;
  ijepa|i-jepa)
    TRAIN_CMD=("${PYTHON_BIN}" train_mit_states_clipstyle.py --vision-backbone ijepa)
    ;;
  vjepa|v-jepa)
    TRAIN_CMD=("${PYTHON_BIN}" train_mit_states_clipstyle.py --vision-backbone v-jepa)
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
echo "Command: ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"
