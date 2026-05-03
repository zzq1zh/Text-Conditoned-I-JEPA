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

# Select target / dataset / W&B project by args or env vars.
# Usage examples:
#   sbatch slurm_train.sh dinov3 cspref_mit_states my-wandb-project
#   sbatch slurm_train.sh ijepa cspref_ut_zappos
#   sbatch slurm_train.sh vjepa cspref_cgqa
#   TRAIN_TARGET=vjepa TRAIN_DATASET=cspref_ut_zappos WANDB_PROJECT=myproj sbatch slurm_train.sh
#   sbatch slurm_train.sh two_fusions cspref_mit_states
#   sbatch slurm_train.sh csp_posttrain cspref_mit_states "" /path/to/base.ckpt
#   sbatch slurm_train.sh csp_posttrain cspref_mit_states my-wandb-project /path/to/base.ckpt vjepa
TARGET="${1:-${TRAIN_TARGET:-two_fusions}}"
DATASET="${2:-${TRAIN_DATASET:-cspref_mit_states}}"
W_PROJECT="${3:-${WANDB_PROJECT:-}}"
BASE_CKPT="${4:-${CSP_BASE_CKPT:-}}"
CSP_BACKBONE="${5:-ijepa}"

case "${TARGET}" in
  two_fusions|two-stage|two_stage)
    TRAIN_CMD=(bash train_two_fusions_and_push.sh)
    ;;
  dinov3|dino)
    TRAIN_CMD=(uv run python run_text_cond_train.py --vision-backbone dinov3 --dataset "${DATASET}")
    ;;
  ijepa|i-jepa)
    TRAIN_CMD=(uv run python run_text_cond_train.py --vision-backbone ijepa --dataset "${DATASET}")
    ;;
  vjepa|v-jepa)
    TRAIN_CMD=(uv run python run_text_cond_train.py --vision-backbone vjepa --dataset "${DATASET}")
    ;;
  csp_posttrain|csp-posttrain|csp)
    TRAIN_CMD=(bash slurm_csp_vocab_train.sh "${CSP_BACKBONE}" "${DATASET}" "${BASE_CKPT}" "${W_PROJECT}")
    ;;
  ./*.sh|*.sh)
    TRAIN_CMD=(bash "${TARGET}")
    ;;
  *)
    echo "Unsupported target: ${TARGET}"
    echo "Supported: two_fusions, dinov3, ijepa, vjepa, csp_posttrain, or a custom .sh path"
    exit 2
    ;;
esac

echo "Training target: ${TARGET}"
echo "Dataset: ${DATASET}"
if [[ "${TARGET}" == "csp_posttrain" || "${TARGET}" == "csp-posttrain" || "${TARGET}" == "csp" ]]; then
  echo "CSP vision backbone: ${CSP_BACKBONE}"
  if [[ -n "${BASE_CKPT}" ]]; then
    echo "CSP base checkpoint: ${BASE_CKPT}"
  else
    echo "CSP base checkpoint: <none; training starts from pretrained base initialization>"
  fi
fi
if [[ -n "${W_PROJECT}" ]]; then
  export WANDB_PROJECT="${W_PROJECT}"
  echo "W&B project: ${WANDB_PROJECT}"
else
  echo "W&B project: <default from env/.env/code>"
fi
echo "Command: ${TRAIN_CMD[*]}"

if [[ "${TARGET}" == "two_fusions" || "${TARGET}" == "two-stage" || "${TARGET}" == "two_stage" ]]; then
  DATASET="${DATASET}" "${TRAIN_CMD[@]}"
else
  "${TRAIN_CMD[@]}"
fi

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"