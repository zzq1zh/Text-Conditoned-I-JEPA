#!/usr/bin/env bash
set -euo pipefail

# Train and upload text-conditioned I-JEPA models to Hugging Face.
# Supports:
# 1) Single run mode (default): one config x two fusion heads
# 2) Grid search mode: cartesian product over LR / BATCH / WEIGHT_DECAY, each with two fusion heads
#
# Example:
#   HF_TOKEN=xxx HF_NAMESPACE=zzq1zh bash train_two_fusions_and_push.sh
# Optional overrides:
#   DATASET=cspref_mit_states EPOCHS=20 BATCH_SIZE=128 SEED=0 bash train_two_fusions_and_push.sh
# Grid search example:
#   GRID_SEARCH=1 LR_LIST=5e-3,5e-4,5e-5 BATCH_SIZE_LIST=128,256 WEIGHT_DECAY_LIST=1e-5,5e-5 \
#   HF_TOKEN=xxx HF_NAMESPACE=zzq1zh bash train_two_fusions_and_push.sh

# Load .env automatically when present (HF_TOKEN / WANDB_* etc.)
if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is empty. Put it in .env or export HF_TOKEN before running this script."
  exit 1
fi

HF_NAMESPACE="${HF_NAMESPACE:-zzq1zh}"
DATASET="${DATASET:-cspref_mit_states}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
TEXT_BANK_CHUNK_SIZE="${TEXT_BANK_CHUNK_SIZE:-512}"
MODEL_PREFIX="${MODEL_PREFIX:-text-cond-ijepa}"
SAVE_DIR="${SAVE_DIR:-checkpoints}"
HF_PRIVATE="${HF_PRIVATE:-1}"     # 1 -> --hub-private, 0 -> public
USE_WANDB="${USE_WANDB:-1}"       # 1 -> enable W&B, 0 -> --no-wandb
GRID_SEARCH="${GRID_SEARCH:-0}"   # 1 -> use LR/BATCH/WD grid, 0 -> single config
LR_LIST="${LR_LIST:-5e-3,5e-4,5e-5}"
BATCH_SIZE_LIST="${BATCH_SIZE_LIST:-128,256}"
WEIGHT_DECAY_LIST="${WEIGHT_DECAY_LIST:-1e-5,5e-5}"

mkdir -p "${SAVE_DIR}"

extra_args=()
if [[ "${HF_PRIVATE}" == "1" ]]; then extra_args+=(--hub-private); fi
if [[ "${USE_WANDB}" != "1" ]]; then extra_args+=(--no-wandb); fi

normalize_token() {
  local x="$1"
  x="${x//./p}"
  x="${x//+/}"
  x="${x//_/-}"
  echo "${x}"
}

train_and_push() {
  local fusion_type="$1"
  local tag="$2"
  local lr="$3"
  local batch_size="$4"
  local weight_decay="$5"
  local run_suffix="$6"
  local ckpt_path="${SAVE_DIR}/${DATASET}_${tag}_${run_suffix}.pt"
  local hub_repo="${HF_NAMESPACE}/${MODEL_PREFIX}-${DATASET}-${tag}-${run_suffix}"

  local cmd=(
    uv run python text_cond_train.py
    --dataset "${DATASET}"
    --seed "${SEED}"
    --epochs "${EPOCHS}"
    --batch-size "${batch_size}"
    --lr "${lr}"
    --weight-decay "${weight_decay}"
    --text-bank-chunk-size "${TEXT_BANK_CHUNK_SIZE}"
    --hub-token "${HF_TOKEN}"
    --fusion-type "${fusion_type}"
    --save "${ckpt_path}"
    --hub-model-id "${hub_repo}"
  )
  cmd+=("${extra_args[@]}")

  echo "============================================================"
  echo "Training fusion_type=${fusion_type}"
  echo "lr=${lr} batch_size=${batch_size} weight_decay=${weight_decay}"
  echo "Checkpoint: ${ckpt_path}"
  echo "Hub repo:   ${hub_repo}"
  echo "============================================================"
  "${cmd[@]}"
}

if [[ "${GRID_SEARCH}" == "1" ]]; then
  IFS=',' read -r -a lr_grid <<< "${LR_LIST}"
  IFS=',' read -r -a bs_grid <<< "${BATCH_SIZE_LIST}"
  IFS=',' read -r -a wd_grid <<< "${WEIGHT_DECAY_LIST}"

  run_idx=0
  for lr_i in "${lr_grid[@]}"; do
    for bs_i in "${bs_grid[@]}"; do
      for wd_i in "${wd_grid[@]}"; do
        run_idx=$((run_idx + 1))
        suffix="g${run_idx}-lr$(normalize_token "${lr_i}")-bs$(normalize_token "${bs_i}")-wd$(normalize_token "${wd_i}")"
        train_and_push "clip_similarity" "clip-sim" "${lr_i}" "${bs_i}" "${wd_i}" "${suffix}"
        train_and_push "cross_attention" "cross-attn" "${lr_i}" "${bs_i}" "${wd_i}" "${suffix}"
      done
    done
  done
  echo "Done. Grid search complete. Total grids: ${run_idx}; total trainings: $((run_idx * 2))."
else
  suffix="default-lr$(normalize_token "${LR}")-bs$(normalize_token "${BATCH_SIZE}")-wd$(normalize_token "${WEIGHT_DECAY}")"
  train_and_push "clip_similarity" "clip-sim" "${LR}" "${BATCH_SIZE}" "${WEIGHT_DECAY}" "${suffix}"
  train_and_push "cross_attention" "cross-attn" "${LR}" "${BATCH_SIZE}" "${WEIGHT_DECAY}" "${suffix}"
  echo "Done. Both models were trained and upload was requested."
fi
