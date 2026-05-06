#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH -J dinov3_attn
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

echo "============================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "============================================"

cd "$SLURM_SUBMIT_DIR"

source ~/.local/bin/env

# Pass-through to visualize_dinov3_attention.py. First non-option token must be present or we print help.
#
# Single-image:
#   sbatch slurm_visualize_dinov3_attention.sh --image path/to.jpg --checkpoint path/to.pt --out attn.png
#   sbatch slurm_visualize_dinov3_attention.sh --image path/to.jpg --layers 6 8 9 10 11 --device cuda
#
# CSP backbone compare (one dataset):
#   sbatch slurm_visualize_dinov3_attention.sh --csp-compare \
#     --csp-checkpoint-tuned path/with_backbone.pt --csp-checkpoint-base path/heads_only.pt \
#     --csp-dataset cspref_mit_states --csp-n-samples 5 --csp-out-dir csp_attention_out
#
# Extra flags before passthrough (array split on whitespace):
#   VIZ_PREFIX_ARGS="--amp" sbatch slurm_visualize_dinov3_attention.sh --csp-compare ...

if [[ $# -eq 0 ]]; then
  echo "Usage: sbatch $0 -- <args to visualize_dinov3_attention.py>"
  echo "  (Slurm forwards arguments after the script name to this job script.)"
  echo "Examples:"
  echo "  sbatch slurm_visualize_dinov3_attention.sh --image sample.jpg --out maps.png"
  echo "  sbatch slurm_visualize_dinov3_attention.sh --csp-compare --csp-dataset cspref_mit_states \\"
  echo "    --csp-checkpoint-tuned a.pt --csp-checkpoint-base b.pt"
  exit 2
fi

# shellcheck disable=SC2206
PREFIX=( ${VIZ_PREFIX_ARGS:-} )
CMD=(uv run python visualize_dinov3_attention.py "${PREFIX[@]}" "$@")

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"
