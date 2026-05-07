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
#   sbatch slurm_text_cond_train.sh dinov3 cspref_mit_states my-wandb-project
#   sbatch slurm_text_cond_train.sh ijepa cspref_ut_zappos
#   sbatch slurm_text_cond_train.sh vjepa cspref_cgqa
#   TRAIN_TARGET=vjepa TRAIN_DATASET=cspref_ut_zappos WANDB_PROJECT=myproj sbatch slurm_text_cond_train.sh
#   sbatch slurm_text_cond_train.sh csp_posttrain cspref_mit_states "" /path/to/base.ckpt
#   sbatch slurm_text_cond_train.sh csp_posttrain cspref_mit_states my-wandb-project /path/to/base.ckpt vjepa
#   FINETUNE_CSP_VOCAB=1 sbatch slurm_text_cond_train.sh ijepa cspref_mit_states
#     → appends --finetune-csp-vocab only (text_cond_train CSP path trains from scratch; 4th-arg BASE_CKPT unused here)
#   FINETUNE_VISION_BACKBONE=1 sbatch slurm_text_cond_train.sh ijepa cspref_mit_states
#     → appends --finetune-vision-backbone (with FINETUNE_CSP_VOCAB=1, unfreezes vision during CSP vocab training too)
#   FUSION_TYPE=cross_attention sbatch slurm_text_cond_train.sh dinov3 cspref_mit_states my-wandb
#     → passes --fusion-type to run_text_cond_train (default: clip_similarity)
#   TRAIN_LR=3e-4 TRAIN_BATCH_SIZE=16 sbatch slurm_text_cond_train.sh ijepa cspref_mit_states
#     → forwarded as extra args to run_text_cond_train (overrides hyperparameters.json for text_cond_train)
#     Aliases: LR / BATCH_SIZE env vars. Not applied to csp_posttrain (still uses JSON via slurm_csp_vocab_train.sh).
TARGET="${1:-${TRAIN_TARGET:-dinov3}}"
DATASET="${2:-${TRAIN_DATASET:-cspref_mit_states}}"
W_PROJECT="${3:-${WANDB_PROJECT:-}}"
BASE_CKPT="${4:-${CSP_BASE_CKPT:-}}"
CSP_BACKBONE="${5:-ijepa}"
FINETUNE_CSP_VOCAB="${FINETUNE_CSP_VOCAB:-0}"
FINETUNE_VISION_BACKBONE="${FINETUNE_VISION_BACKBONE:-0}"
FUSION_TYPE="${FUSION_TYPE:-clip_similarity}"
# Learning rate / batch size overrides for dinov3|ijepa|vjepa only (empty = use hyperparameters.json).
TRAIN_LR="${TRAIN_LR:-${LR:-}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${BATCH_SIZE:-}}"

case "${TARGET}" in
  dinov3|dino)
    TRAIN_CMD=(uv run python run_text_cond_train.py --vision-backbone dinov3 --dataset "${DATASET}" --fusion-type "${FUSION_TYPE}")
    ;;
  ijepa|i-jepa)
    TRAIN_CMD=(uv run python run_text_cond_train.py --vision-backbone ijepa --dataset "${DATASET}" --fusion-type "${FUSION_TYPE}")
    ;;
  vjepa|v-jepa)
    TRAIN_CMD=(uv run python run_text_cond_train.py --vision-backbone vjepa --dataset "${DATASET}" --fusion-type "${FUSION_TYPE}")
    ;;
  csp_posttrain|csp-posttrain|csp)
    TRAIN_CMD=(bash slurm_csp_vocab_train.sh "${CSP_BACKBONE}" "${DATASET}" "${BASE_CKPT}" "${W_PROJECT}")
    ;;
  ./*.sh|*.sh)
    TRAIN_CMD=(bash "${TARGET}")
    ;;
  *)
    echo "Unsupported target: ${TARGET}"
    echo "Supported: dinov3, ijepa, vjepa, csp_posttrain, or a custom .sh path"
    exit 2
    ;;
esac

_fcv="$(echo "${FINETUNE_CSP_VOCAB}" | tr '[:upper:]' '[:lower:]')"
if [[ "${_fcv}" == "1" || "${_fcv}" == "true" || "${_fcv}" == "yes" || "${_fcv}" == "y" ]]; then
  case "${TARGET}" in
    dinov3|dino|ijepa|i-jepa|vjepa|v-jepa)
      TRAIN_CMD+=(--finetune-csp-vocab)
      ;;
    *)
      echo "Warning: FINETUNE_CSP_VOCAB set but target '${TARGET}' is not dinov3/ijepa/vjepa; ignoring CSP finetune flags."
      ;;
  esac
fi

_fvb="$(echo "${FINETUNE_VISION_BACKBONE}" | tr '[:upper:]' '[:lower:]')"
if [[ "${_fvb}" == "1" || "${_fvb}" == "true" || "${_fvb}" == "yes" || "${_fvb}" == "y" ]]; then
  case "${TARGET}" in
    dinov3|dino|ijepa|i-jepa|vjepa|v-jepa)
      TRAIN_CMD+=(--finetune-vision-backbone)
      ;;
    *)
      echo "Warning: FINETUNE_VISION_BACKBONE set but target '${TARGET}' is not dinov3/ijepa/vjepa; ignoring --finetune-vision-backbone."
      ;;
  esac
fi

case "${TARGET}" in
  dinov3|dino|ijepa|i-jepa|vjepa|v-jepa)
    if [[ -n "${TRAIN_BATCH_SIZE}" ]]; then
      TRAIN_CMD+=(--batch-size "${TRAIN_BATCH_SIZE}")
    fi
    if [[ -n "${TRAIN_LR}" ]]; then
      TRAIN_CMD+=(--lr "${TRAIN_LR}")
    fi
    ;;
esac

echo "Training target: ${TARGET}"
echo "Dataset: ${DATASET}"
echo "FINETUNE_CSP_VOCAB: ${FINETUNE_CSP_VOCAB} (1/true/y adds --finetune-csp-vocab for ijepa/dinov3/vjepa)"
echo "FINETUNE_VISION_BACKBONE: ${FINETUNE_VISION_BACKBONE} (1/true/y adds --finetune-vision-backbone for ijepa/dinov3/vjepa)"
echo "FUSION_TYPE: ${FUSION_TYPE} (--fusion-type for run_text_cond_train; not from hyperparameters.json)"
if [[ "${TARGET}" == dinov3 || "${TARGET}" == dino || "${TARGET}" == ijepa || "${TARGET}" == i-jepa || "${TARGET}" == vjepa || "${TARGET}" == v-jepa ]]; then
  if [[ -n "${TRAIN_BATCH_SIZE}" || -n "${TRAIN_LR}" ]]; then
    echo "TRAIN_BATCH_SIZE: ${TRAIN_BATCH_SIZE:-<JSON>}  TRAIN_LR: ${TRAIN_LR:-<JSON>} (set env TRAIN_BATCH_SIZE / TRAIN_LR or BATCH_SIZE / LR; overrides hyperparameters.json via run_text_cond_train extra args)"
  else
    echo "TRAIN_BATCH_SIZE / TRAIN_LR: <unset; batch_size & lr from hyperparameters.json>"
  fi
fi
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

"${TRAIN_CMD[@]}"

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"
