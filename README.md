# Understanding DinoV3 Representations with Compositional Reasoning

This CSCI 1430 final project explores the use of a lightweight fusion head to align vision backbones such as DINOv3 with textual labels, and evaluates the approach on three datasets: MIT-States, UT-Zappos, and C-GQA.

**Default `--vision-backbone` preset is DINOv3** (`timm/vit_base_patch16_dinov3.lvd1689m`). Use `--vision-backbone {dinov3, ijepa, vjepa}` to switch encoders.

## What is implemented

- **Vision backbone**: loaded with `transformers.AutoModel` + image/video processor; **frozen by default** (unfreeze with `--finetune-vision-backbone`).
- **CLIP text tower** + adapter; optionally train CLIP text with `--finetune-clip-text`.
- **Fusion heads**: `cross_attention`, `clip_similarity`.
- **Single training objective**: bidirectional InfoNCE.
- **Eval metrics**: overall top-1/top-5, seen/unseen splits where applicable, `auc_csp_style`.
- Optional **Weights & Biases** logging and optional **Hugging Face Hub** upload of trainable head weights.

## Setup

```bash
uv sync
cp .env.example .env
```

Recommended `.env` keys:

- `WANDB_API_KEY` (if using W&B)
- `WANDB_PROJECT` (optional default project name)

Hugging Face Hub upload or gated model access uses the credential from `huggingface-cli login`.

## Quick Start

Hyperparameters from `hyperparameters.json` unless overridden. Add `--no-wandb` to skip W&B.

### Training

```bash
# Text-conditioned head (+ bundled val/test); checkpoints → checkpoints/
uv run python run_text_cond_train.py --vision-backbone dinov3 --dataset cspref_mit_states --seed 42

# CSP vocab post-train (needs a base .pt from above or from JSON base_checkpoint)
uv run python run_csp_vocab_train.py --vision-backbone dinov3 --dataset cspref_mit_states --seed 42 \
  --base-checkpoint checkpoints/<your_checkpoint>.pt
```

### Eval

**All checkpoints in a folder**:

```bash
<<<<<<< HEAD
# Re-run val+test on every checkpoint under checkpoints/
=======
>>>>>>> fb8accd (Refactor codebase)
uv run python run_evals.py checkpoints/
```

**One checkpoint**:

```bash
uv run python run_evals.py checkpoints/<your_checkpoint>.pt
```

If backbone, dataset, or seed cannot be inferred from the path or bundle, add e.g. `--vision-backbone dinov3 --dataset cspref_mit_states --seed 42`.

## Main scripts

- `text_cond_train.py`: train / eval entrypoint
- `csp_vocab_train.py`: CSP-style compositional vocabulary training entrypoint
- `run_text_cond_train.py`: config-driven multi-seed train+eval launcher (reads `hyperparameters.json`; see **Quick Start**)
- `run_csp_vocab_train.py`: CSP vocab multi-seed launcher over `csp_vocab_train.py` (see **Quick Start**)
- `main.py`: `TextConditionedVisionModel`, fusion modules, backbone loading helpers
- `vision_data.py`: dataset registry and split logic
- `build_csp_hf_datasets.py`: build/push **CSP reference** Hugging Face datasets (MIT-States / UT-Zappos / C-GQA); optional `--download` / `--prepare`, `--ref-push` to Hub (`--namespace`, `--repo-prefix`, `--ref-only`, `--ref-public`, `--token`; see `python build_csp_hf_datasets.py --help`)
- `run_evals.py` / `slurm_run_evals.sh`: batch re-run val+test `--eval-only` on checkpoints (see **Quick Start** for `run_evals.py`)
- `visualize_dinov3_attention.py`: visualize ViT **CLS→patch** self-attention (see below)

## Training

Example:

```bash
uv run python text_cond_train.py \
  --dataset cspref_mit_states \
  --fusion-type cross_attention \
  --epochs 20 \
  --batch-size 128 \
  --save ckpt_cross_attn.pt
```

Useful flags:

- `--fusion-type {cross_attention,clip_similarity}`
- `--finetune-csp-vocab` (train CSP-style attr/object compositional soft prompts in `text_cond_train.py`; see script help for constraints)
- `--csp-vocab-init {random,text}`
- `--csp-attr-dropout 0.3`
- `--csp-pair-separator " "`
- `--vision-backbone {dinov3,ijepa,vjepa}` — default **`dinov3`** if omitted; or set explicit HF vision model id with `--ijepa <id>`
- `--hyperparams-file hyperparameters.json`
- `--finetune-clip-text`
- `--finetune-vision-backbone`
- `--hub-model-id user/repo` (push trainable weights after training)
- `--no-wandb` (disable W&B)

Backbone examples:

```bash
# DINOv3 ViT-B/16
uv run python text_cond_train.py --vision-backbone dinov3 --dataset cspref_mit_states --epochs 1

# I-JEPA
uv run python text_cond_train.py --vision-backbone ijepa --dataset cspref_mit_states --epochs 1

# V-JEPA 2
uv run python text_cond_train.py --vision-backbone vjepa --dataset cspref_mit_states --epochs 1
```

Dedicated CSP vocab training with `csp_vocab_train.py` (defaults include `--csp-vocab-init text`):

```bash
uv run python csp_vocab_train.py \
  --base-checkpoint checkpoints/<your_checkpoint>.pt \
  --dataset cspref_mit_states \
  --epochs 20 \
  --batch-size 128 \
  --save ckpt_csp_vocab.pt
```

## Eval only

```bash
uv run python text_cond_train.py --eval-only --dataset cspref_mit_states --checkpoint checkpoints/<your_checkpoint>.pt --eval-split val
uv run python text_cond_train.py --eval-only --dataset cspref_mit_states --checkpoint checkpoints/<your_checkpoint>.pt --eval-split test
```

## Attention visualization (`visualize_dinov3_attention.py`)

Utility to plot **encoder self-attention** maps (mean over heads: **CLS token → image patches**) for the **vision backbone** inside your saved models. Which Hub checkpoint is used (**DINOv3**, I-JEPA, V-JEPA, …) comes from the bundle’s training `args` (`vision_backbone` / `ijepa`), not from separate CLI flags on this script.

**CLI modes**

1. **Single image** — pass `--image` and a **required** `--checkpoint` that is a **CSP vocab bundle** `.pt` (same schema as `--csp-checkpoint-tuned`: `csp_vocab`, `meta`, `backbone`, `adapter`, `fusion`, etc.). The script builds `TextConditionedVisionModel` from the bundle, applies the same eager-backbone reload as `--csp-compare`, then runs attention on `model.backbone`. Set `--layers` (0-based block indices) and `--out`.
2. **`--csp-compare`** — load two **CSP vocab** `.pt` bundles (`--csp-checkpoint-tuned` with finetuned `backbone`, `--csp-checkpoint-base` with heads only / backbone ignored), scan **`--csp-dataset` official test split** for **contrast samples** (tuned top-1 correct, pretrained-only wrong), then draw side-by-side attention grids. Optional export of PNGs + `manifest.json` under `--csp-out-dir` (see `--csp-save-samples-dir`, `--csp-no-save-samples`).

Examples:

```bash
uv run python visualize_dinov3_attention.py --image path/to.jpg --checkpoint checkpoints/xxx.pt \
  --out attn.png --layers 6 8 9 10 11

uv run python visualize_dinov3_attention.py --csp-compare \
  --csp-checkpoint-tuned checkpoints/<your_tuned_checkpoint>.pt \
  --csp-checkpoint-base checkpoints/<your_base_checkpoint>.pt \
  --csp-dataset cspref_mit_states --csp-n-samples 5 --csp-out-dir out_attn
```

## Datasets attribution

Parts of the **datasets** code paths—especially registry entries, loaders, and eval helpers for **CSP / compositional** Hub splits in `vision_data.py` (and related dataset build tooling)—derive from or follow the setup in:

```bibtex
@inproceedings{nayak2023learning,
  title     = {Learning to Compose Soft Prompts for Compositional Zero-Shot Learning},
  author    = {Nayak, Nihal V. and Yu, Peilin and Bach, Stephen H.},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}
```
