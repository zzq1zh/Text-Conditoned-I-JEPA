#!/usr/bin/env bash
set -euo pipefail

# Two-stage auto mode (default):
#   Stage 1: run 1-seed grid search
#   Stage 2: auto pick top-k configs from val metric, then run 5 seeds
#
# Optional modes:
#   MODE=single  -> fixed hyperparams over SEED_LIST
#   MODE=grid    -> full grid over SEED_LIST
#   MODE=two_stage (default)

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

MODE="${MODE:-two_stage}"
HF_NAMESPACE="${HF_NAMESPACE:-zzq1zh}"
DATASET="${DATASET:-cspref_mit_states}"
EPOCHS="${EPOCHS:-20}"
TEXT_BANK_CHUNK_SIZE="${TEXT_BANK_CHUNK_SIZE:-512}"
MODEL_PREFIX="${MODEL_PREFIX:-text-cond-ijepa}"
SAVE_DIR="${SAVE_DIR:-checkpoints}"
RESULTS_DIR="${RESULTS_DIR:-results}"
HF_PRIVATE="${HF_PRIVATE:-1}"
USE_WANDB="${USE_WANDB:-1}"

# Hyperparameters
LR="${LR:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
LR_LIST="${LR_LIST:-5e-3,5e-4,5e-5}"
BATCH_SIZE_LIST="${BATCH_SIZE_LIST:-128,256}"
WEIGHT_DECAY_LIST="${WEIGHT_DECAY_LIST:-1e-5,5e-5}"

# Seeds
SEED_LIST="${SEED_LIST:-0,1,2,3,4}"              # used by MODE=single/grid
STAGE1_SEED="${STAGE1_SEED:-0}"                  # used by MODE=two_stage
STAGE2_SEED_LIST="${STAGE2_SEED_LIST:-0,1,2,3,4}" # used by MODE=two_stage

# Selection for stage 2
TOP_K="${TOP_K:-1}"                               # top-k per fusion
SELECT_SPLIT="${SELECT_SPLIT:-val}"
SELECT_METRIC="${SELECT_METRIC:-auc_csp_style}"   # higher is better

# Upload policy
UPLOAD_STAGE1="${UPLOAD_STAGE1:-0}"               # 1 to upload stage-1 runs
UPLOAD_STAGE2="${UPLOAD_STAGE2:-1}"               # 1 to upload stage-2 runs
UPLOAD_SINGLE_GRID="${UPLOAD_SINGLE_GRID:-1}"     # for MODE=single/grid

mkdir -p "${SAVE_DIR}" "${RESULTS_DIR}"

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

summarize_dir() {
  local metrics_dir="$1"
  local out_dir="$2"
  mkdir -p "${out_dir}" "${out_dir}/plots"
  uv run python summarize_eval_metrics.py \
    --metrics-dir "${metrics_dir}" \
    --raw-csv "${out_dir}/raw_eval_metrics.csv" \
    --summary-csv "${out_dir}/summary_metrics.csv" \
    --summary-json "${out_dir}/summary_metrics.json" \
    --plot-dir "${out_dir}/plots"
}

run_train_eval() {
  local fusion_type="$1"
  local tag="$2"
  local lr="$3"
  local batch_size="$4"
  local weight_decay="$5"
  local run_suffix="$6"
  local seed="$7"
  local metrics_dir="$8"
  local do_upload="$9"

  mkdir -p "${metrics_dir}"
  local experiment_tag="${tag}-${run_suffix}"
  local ckpt_path="${SAVE_DIR}/${DATASET}_${tag}_${run_suffix}.pt"
  local hub_repo="${HF_NAMESPACE}/${MODEL_PREFIX}-${DATASET}-${tag}-${run_suffix}"
  local val_metrics_json="${metrics_dir}/${experiment_tag}_val.json"
  local test_metrics_json="${metrics_dir}/${experiment_tag}_test.json"

  local cmd=(
    uv run python text_cond_train.py
    --dataset "${DATASET}"
    --seed "${seed}"
    --epochs "${EPOCHS}"
    --batch-size "${batch_size}"
    --lr "${lr}"
    --weight-decay "${weight_decay}"
    --text-bank-chunk-size "${TEXT_BANK_CHUNK_SIZE}"
    --hub-token "${HF_TOKEN}"
    --fusion-type "${fusion_type}"
    --save "${ckpt_path}"
  )
  if [[ "${do_upload}" == "1" ]]; then
    cmd+=(--hub-model-id "${hub_repo}")
  fi
  cmd+=("${extra_args[@]}")

  echo "============================================================"
  echo "Training fusion_type=${fusion_type} seed=${seed}"
  echo "lr=${lr} batch_size=${batch_size} weight_decay=${weight_decay}"
  echo "Checkpoint: ${ckpt_path}"
  if [[ "${do_upload}" == "1" ]]; then
    echo "Hub repo:   ${hub_repo}"
  else
    echo "Hub upload: disabled for this run"
  fi
  echo "============================================================"
  "${cmd[@]}"

  uv run python text_cond_train.py \
    --eval-only \
    --dataset "${DATASET}" \
    --seed "${seed}" \
    --batch-size "${batch_size}" \
    --lr "${lr}" \
    --weight-decay "${weight_decay}" \
    --text-bank-chunk-size "${TEXT_BANK_CHUNK_SIZE}" \
    --fusion-type "${fusion_type}" \
    --checkpoint "${ckpt_path}" \
    --eval-split val \
    --experiment-tag "${experiment_tag}" \
    --metrics-json "${val_metrics_json}" \
    "${extra_args[@]}"

  uv run python text_cond_train.py \
    --eval-only \
    --dataset "${DATASET}" \
    --seed "${seed}" \
    --batch-size "${batch_size}" \
    --lr "${lr}" \
    --weight-decay "${weight_decay}" \
    --text-bank-chunk-size "${TEXT_BANK_CHUNK_SIZE}" \
    --fusion-type "${fusion_type}" \
    --checkpoint "${ckpt_path}" \
    --eval-split test \
    --experiment-tag "${experiment_tag}" \
    --metrics-json "${test_metrics_json}" \
    "${extra_args[@]}"
}

run_grid_for_seed() {
  local seed="$1"
  local metrics_dir="$2"
  local do_upload="$3"
  IFS=',' read -r -a lr_grid <<< "${LR_LIST}"
  IFS=',' read -r -a bs_grid <<< "${BATCH_SIZE_LIST}"
  IFS=',' read -r -a wd_grid <<< "${WEIGHT_DECAY_LIST}"
  local run_idx=0
  for lr_i in "${lr_grid[@]}"; do
    for bs_i in "${bs_grid[@]}"; do
      for wd_i in "${wd_grid[@]}"; do
        run_idx=$((run_idx + 1))
        local suffix="s${seed}-g${run_idx}-lr$(normalize_token "${lr_i}")-bs$(normalize_token "${bs_i}")-wd$(normalize_token "${wd_i}")"
        run_train_eval "clip_similarity" "clip-sim" "${lr_i}" "${bs_i}" "${wd_i}" "${suffix}" "${seed}" "${metrics_dir}" "${do_upload}"
      done
    done
  done
}

select_topk_configs() {
  local metrics_dir="$1"
  local out_csv="$2"
  local stage1_seed="$3"
  local top_k="$4"
  local split="$5"
  local metric="$6"

  uv run python - <<'PY' "${metrics_dir}" "${out_csv}" "${stage1_seed}" "${top_k}" "${split}" "${metric}"
from __future__ import annotations
import csv, json, math, pathlib, sys

metrics_dir = pathlib.Path(sys.argv[1])
out_csv = pathlib.Path(sys.argv[2])
stage1_seed = int(sys.argv[3])
top_k = int(sys.argv[4])
split = sys.argv[5]
metric = sys.argv[6]

rows = []
for p in sorted(metrics_dir.glob("*.json")):
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        continue
    if str(d.get("split")) != split:
        continue
    if int(d.get("seed", -1)) != stage1_seed:
        continue
    v = float(d.get(metric, float("nan")))
    if math.isnan(v) or math.isinf(v):
        continue
    rows.append(d)

if not rows:
    raise SystemExit("No valid stage-1 rows found for top-k selection.")

best = {}
for d in rows:
    key = (
        str(d.get("fusion_type", "")),
        float(d.get("lr")),
        int(d.get("batch_size")),
        float(d.get("weight_decay")),
    )
    score = float(d.get(metric))
    if key not in best or score > best[key]["score"]:
        best[key] = {"score": score, "data": d}

per_fusion = {}
for item in best.values():
    fusion = str(item["data"]["fusion_type"])
    per_fusion.setdefault(fusion, []).append(item)

out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["fusion_type", "tag", "rank", "lr", "batch_size", "weight_decay", "score"])
    for fusion, items in sorted(per_fusion.items()):
        items.sort(key=lambda x: x["score"], reverse=True)
        for rank, item in enumerate(items[:top_k], start=1):
            d = item["data"]
            tag = "clip-sim" if fusion == "clip_similarity" else "cross-attn"
            w.writerow([
                fusion,
                tag,
                rank,
                d["lr"],
                d["batch_size"],
                d["weight_decay"],
                item["score"],
            ])

print(f"Wrote selected configs: {out_csv}")
PY
}

run_two_stage() {
  local stage1_dir="${RESULTS_DIR}/stage1"
  local stage2_dir="${RESULTS_DIR}/stage2"
  local stage1_metrics="${stage1_dir}/raw"
  local stage2_metrics="${stage2_dir}/raw"
  local selected_csv="${stage1_dir}/selected_topk.csv"
  mkdir -p "${stage1_metrics}" "${stage2_metrics}"

  echo "[Stage 1] seed=${STAGE1_SEED} grid search"
  run_grid_for_seed "${STAGE1_SEED}" "${stage1_metrics}" "${UPLOAD_STAGE1}"
  summarize_dir "${stage1_metrics}" "${stage1_dir}"

  echo "[Stage 1] selecting top-${TOP_K} per fusion by ${SELECT_METRIC} on ${SELECT_SPLIT}"
  select_topk_configs "${stage1_metrics}" "${selected_csv}" "${STAGE1_SEED}" "${TOP_K}" "${SELECT_SPLIT}" "${SELECT_METRIC}"

  IFS=',' read -r -a stage2_seeds <<< "${STAGE2_SEED_LIST}"
  echo "[Stage 2] seeds=${STAGE2_SEED_LIST}"
  while IFS=, read -r fusion tag rank lr bs wd score; do
    if [[ "${fusion}" == "fusion_type" ]]; then
      continue
    fi
    for s in "${stage2_seeds[@]}"; do
      suffix="s${s}-top${rank}-${tag}-lr$(normalize_token "${lr}")-bs$(normalize_token "${bs}")-wd$(normalize_token "${wd}")"
      run_train_eval "${fusion}" "${tag}" "${lr}" "${bs}" "${wd}" "${suffix}" "${s}" "${stage2_metrics}" "${UPLOAD_STAGE2}"
    done
  done < "${selected_csv}"

  summarize_dir "${stage2_metrics}" "${stage2_dir}"
  echo "Two-stage run complete."
  echo "Stage-1 summary: ${stage1_dir}/summary_metrics.csv"
  echo "Stage-2 summary: ${stage2_dir}/summary_metrics.csv"
}

run_single() {
  local out_dir="${RESULTS_DIR}/single"
  local metrics_dir="${out_dir}/raw"
  mkdir -p "${metrics_dir}"
  IFS=',' read -r -a seeds <<< "${SEED_LIST}"
  for s in "${seeds[@]}"; do
    local suffix="s${s}-default-lr$(normalize_token "${LR}")-bs$(normalize_token "${BATCH_SIZE}")-wd$(normalize_token "${WEIGHT_DECAY}")"
    run_train_eval "clip_similarity" "clip-sim" "${LR}" "${BATCH_SIZE}" "${WEIGHT_DECAY}" "${suffix}" "${s}" "${metrics_dir}" "${UPLOAD_SINGLE_GRID}"
  done
  summarize_dir "${metrics_dir}" "${out_dir}"
  echo "Single-mode run complete. Summary: ${out_dir}/summary_metrics.csv"
}

run_grid() {
  local out_dir="${RESULTS_DIR}/grid"
  local metrics_dir="${out_dir}/raw"
  mkdir -p "${metrics_dir}"
  IFS=',' read -r -a seeds <<< "${SEED_LIST}"
  for s in "${seeds[@]}"; do
    run_grid_for_seed "${s}" "${metrics_dir}" "${UPLOAD_SINGLE_GRID}"
  done
  summarize_dir "${metrics_dir}" "${out_dir}"
  echo "Grid-mode run complete. Summary: ${out_dir}/summary_metrics.csv"
}

case "${MODE}" in
  two_stage) run_two_stage ;;
  single) run_single ;;
  grid) run_grid ;;
  *)
    echo "ERROR: Unsupported MODE=${MODE}. Use: two_stage | single | grid"
    exit 1
    ;;
esac
