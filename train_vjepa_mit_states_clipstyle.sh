#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
LR="${LR:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
SEED_LIST="${SEED_LIST:-0,1,2,3,4}"
HYPERPARAMS_FILE="${HYPERPARAMS_FILE:-hyperparameters.json}"
VISION_BACKBONE="v-jepa"
DATASET_KEY="cspref_mit_states"
NO_WANDB=0
FINETUNE_CLIP_TEXT=0
DRY_RUN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --seed) SEED_LIST="$2"; shift 2 ;;
    --seed-list) SEED_LIST="$2"; shift 2 ;;
    --hyperparams-file) HYPERPARAMS_FILE="$2"; shift 2 ;;
    --no-wandb) NO_WANDB=1; shift ;;
    --finetune-clip-text) FINETUNE_CLIP_TEXT=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if command -v uv >/dev/null 2>&1; then
  PY_CMD=(uv run python)
  TRAIN_CMD_PREFIX=(uv run python text_cond_train.py)
elif command -v python3 >/dev/null 2>&1; then
  PY_CMD=(python3)
  TRAIN_CMD_PREFIX=(python3 text_cond_train.py)
elif command -v python >/dev/null 2>&1; then
  PY_CMD=(python)
  TRAIN_CMD_PREFIX=(python text_cond_train.py)
else
  echo "ERROR: neither uv, python3, nor python is available in PATH."
  exit 127
fi

resolve_hparam() {
  local key="$1"
  "${PY_CMD[@]}" - "$HYPERPARAMS_FILE" "$VISION_BACKBONE" "$DATASET_KEY" "$key" <<'PY'
import json
import pathlib
import sys

cfg_path = pathlib.Path(sys.argv[1])
backbone = sys.argv[2]
dataset = sys.argv[3]
key = sys.argv[4]

if not cfg_path.exists():
    print("")
    raise SystemExit(0)

cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
merged = {}
for src in (
    cfg.get("defaults", {}),
    cfg.get("models", {}).get(backbone, {}),
    cfg.get("datasets", {}).get(dataset, {}),
    cfg.get("model_dataset", {}).get(backbone, {}).get(dataset, {}),
):
    if isinstance(src, dict):
        merged.update(src)
v = merged.get(key, "")
print("" if v is None else str(v))
PY
}

HP_EPOCHS="$(resolve_hparam epochs)"
HP_BATCH_SIZE="$(resolve_hparam batch_size)"
HP_LR="$(resolve_hparam lr)"
HP_WEIGHT_DECAY="$(resolve_hparam weight_decay)"

EPOCHS="${EPOCHS:-${HP_EPOCHS:-20}}"
BATCH_SIZE="${BATCH_SIZE:-${HP_BATCH_SIZE:-128}}"
LR="${LR:-${HP_LR:-5e-5}}"
WEIGHT_DECAY="${WEIGHT_DECAY:-${HP_WEIGHT_DECAY:-1e-5}}"

mkdir -p checkpoints
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
IFS=',' read -r -a SEEDS <<< "${SEED_LIST}"

for SEED in "${SEEDS[@]}"; do
  CKPT_PATH="checkpoints/vjepa_mit_states_clipstyle_s${SEED}_${TIMESTAMP}.pt"
  CMD=(
    "${TRAIN_CMD_PREFIX[@]}"
    --vision-backbone "${VISION_BACKBONE}"
    --dataset "${DATASET_KEY}"
    --fusion-type clip_similarity
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --lr "${LR}"
    --weight-decay "${WEIGHT_DECAY}"
    --seed "${SEED}"
    --hyperparams-file "${HYPERPARAMS_FILE}"
    --save "${CKPT_PATH}"
  )

  if [[ "${NO_WANDB}" == "1" ]]; then
    CMD+=(--no-wandb)
  fi
  if [[ "${FINETUNE_CLIP_TEXT}" == "1" ]]; then
    CMD+=(--finetune-clip-text)
  fi
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
  fi

  echo "Running command (seed=${SEED}):"
  printf '%q ' "${CMD[@]}"
  echo
  echo "Checkpoint will be saved to: ${CKPT_PATH}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi

  "${CMD[@]}"
done
