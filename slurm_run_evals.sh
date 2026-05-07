#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 24:00:00
#SBATCH -J run_evals
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

echo "============================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "============================================"

# Run from the code directory (same as slurm_text_cond_train.sh)
cd "$SLURM_SUBMIT_DIR"

source ~/.local/bin/env

# Re-run val + test eval for every checkpoint; see run_evals.py --help.
#
# Usage examples:
#   sbatch slurm_run_evals.sh
#       → defaults to: checkpoints/ --recurse
#   sbatch slurm_run_evals.sh my_ckpts/ --recurse
#   sbatch slurm_run_evals.sh checkpoints/ --glob 'csp_vocab_*.pt'
#   sbatch slurm_run_evals.sh ckpt1.pt ckpt2.pt --vision-backbone dinov3 --dataset cspref_cgqa
#   HYPERPARAMS_FILE=/path/to/hyperparameters.json sbatch slurm_run_evals.sh
#   WANDB_PROJECT=myproj sbatch slurm_run_evals.sh checkpoints/ --recurse --wandb
#
# Extra tokens can be passed via env (evaluated before positional args):
#   RERUN_PREFIX_ARGS="--dry-run" sbatch slurm_run_evals.sh

if [[ -n "${HYPERPARAMS_FILE:-}" ]]; then
  export HYPERPARAMS_FILE
  echo "HYPERPARAMS_FILE: ${HYPERPARAMS_FILE}"
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  export WANDB_PROJECT="${WANDB_PROJECT}"
  echo "W&B project: ${WANDB_PROJECT}"
fi

# shellcheck disable=SC2206
PREFIX=( ${RERUN_PREFIX_ARGS:-} )
if [[ $# -eq 0 ]]; then
  EVAL_CMD=(uv run python run_evals.py "${PREFIX[@]}" checkpoints/ --recurse)
else
  EVAL_CMD=(uv run python run_evals.py "${PREFIX[@]}" "$@")
fi

echo "Command: ${EVAL_CMD[*]}"
"${EVAL_CMD[@]}"

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"
