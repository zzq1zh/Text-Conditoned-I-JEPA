# Text-Conditioned I-JEPA

Minimal training/evaluation pipeline for text-conditioned I-JEPA on CIFAR and CSP-style datasets.

## What is implemented now

- Frozen I-JEPA image backbone + CLIP text tower (text tower can be unfrozen with `--finetune-clip-text`)
- Fusion heads: `cross_attention`, `linear`, `clip_similarity`
- **Single training objective**: bidirectional InfoNCE (CLIP-style contrastive loss)
- Eval metrics: overall top-1/top-5, seen top-1/top-5, unseen top-1/top-5, `auc_csp_style`
- Optional W&B logging and optional Hugging Face model upload

## Setup

```bash
cd /path/to/final-project
uv sync
cp .env.example .env
```

Recommended `.env` keys:

- `HF_TOKEN` (for pushing/loading private Hub repos)
- `WANDB_API_KEY` (if using W&B)
- `WANDB_PROJECT` (optional default project name)

## Main scripts

- `text_cond_train.py`: train / eval entrypoint
- `main.py`: model definitions (`TextConditionedIJepa`, fusion heads)
- `vision_data.py`: dataset registry and split logic
- `build_csp_reference_hf_datasets.py`: convert CSP reference datasets to HF format and push
- `train_two_fusions_and_push.sh`: run `clip_similarity` and `cross_attention`, upload both, optional grid search + multi-seed

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

- `--fusion-type {cross_attention,linear,clip_similarity}`
- `--vision-backbone {ijepa,vjepa,dinov3}` (or pass explicit model id with `--ijepa ...`)
- `--hyperparams-file hyperparameters.json`
- `--finetune-clip-text`
- `--hub-model-id user/repo` (push after training)
- `--no-wandb` (disable W&B)

Backbone examples:

```bash
# V-JEPA 2
uv run python text_cond_train.py --vision-backbone vjepa --dataset cspref_mit_states --epochs 1

# DINOv3 ViT-B/16
uv run python text_cond_train.py --vision-backbone dinov3 --dataset cspref_mit_states --epochs 1
```

## Hyperparameters file

1. `defaults`
2. `models.<vision-backbone>`
3. `datasets.<dataset>`
4. `model_dataset.<vision-backbone>.<dataset>`

## Eval only

```bash
uv run python text_cond_train.py --eval-only --dataset cspref_mit_states --checkpoint ckpt_cross_attn.pt --eval-split val
uv run python text_cond_train.py --eval-only --dataset cspref_mit_states --checkpoint ckpt_cross_attn.pt --eval-split test
uv run python text_cond_train.py --eval-only --from-hub user/model --dataset cspref_mit_states --eval-split test
```

## Split behavior

- For `csp_*` and `cspref_*`: use official Hub `train/val/test` directly.
- For non-CSP datasets: split Hub train into train/val and use configured test logic from `vision_data.py`.

## Batch script: automatic two-stage / single / grid

Default (`MODE=two_stage`, reads `.env`):

- Stage 1: run grid on one seed (`STAGE1_SEED`, default `0`)
- Auto-select top-k configs per fusion on val (`TOP_K`, default `1`)
- Stage 2: run selected configs on 5 seeds (`STAGE2_SEED_LIST`, default `0,1,2,3,4`)

```bash
bash train_two_fusions_and_push.sh
```

Other modes:

```bash
MODE=single bash train_two_fusions_and_push.sh
MODE=grid bash train_two_fusions_and_push.sh
```

Optional overrides:

- `STAGE1_SEED=0`
- `STAGE2_SEED_LIST=0,1,2,3,4`
- `TOP_K=1`
- `SELECT_SPLIT=val`
- `SELECT_METRIC=auc_csp_style`
- `SEED_LIST=0,1,2,3,4` (for `MODE=single/grid`)
- `LR_LIST=5e-3,5e-4,5e-5`
- `BATCH_SIZE_LIST=128,256`
- `WEIGHT_DECAY_LIST=1e-5,5e-5`
- `DATASET`, `EPOCHS`, `HF_NAMESPACE`, `MODEL_PREFIX`, `SAVE_DIR`

Output artifacts from batch script:

- `MODE=two_stage`: `results/stage1/*` and `results/stage2/*`
- `MODE=single`: `results/single/*`
- `MODE=grid`: `results/grid/*`
