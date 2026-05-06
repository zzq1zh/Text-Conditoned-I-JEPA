# Text-conditioned vision models

Training and evaluation pipeline for **vision–language contrastive fine-tuning**: a Hugging Face **vision backbone** (ViT-style encoders) paired with a **CLIP text encoder**, fused for classification-style scoring on CIFAR and CSP-style datasets.

**Default `--vision-backbone` preset is DINOv3** (`facebook/dinov3-vitb16-pretrain-lvd1689m`). Use `--vision-backbone {ijepa,vjepa}` or an explicit HF id via **`--ijepa`** (legacy flag name) to switch encoders.

## What is implemented

- **Vision backbone**: loaded with `transformers.AutoModel` + image/video processor; **frozen by default** (unfreeze with `--finetune-vision-backbone`). Presets: `dinov3` (default), `ijepa`, `vjepa`; override with `--ijepa <hub/model-id>`.
- **CLIP text tower** + adapter; optionally train CLIP text with `--finetune-clip-text`.
- **Fusion heads**: `cross_attention`, `linear`, `clip_similarity`.
- **Single training objective**: bidirectional InfoNCE (CLIP-style contrastive loss).
- **Eval metrics**: overall top-1/top-5, seen/unseen splits where applicable, `auc_csp_style`.
- Optional **Weights & Biases** logging and optional **Hugging Face Hub** upload of trainable head weights (backbone loaded from Hub by id at eval time).

## Setup

```bash
cd /path/to/final-project
uv sync
cp .env.example .env
```

Recommended `.env` keys:

- `WANDB_API_KEY` (if using W&B)
- `WANDB_PROJECT` (optional default project name)

Hugging Face Hub upload or gated model access uses the credential from `huggingface-cli login` (no project `.env` token required).

## Main scripts

- `text_cond_train.py`: train / eval entrypoint
- `csp_vocab_train.py`: CSP-style compositional vocabulary training entrypoint
- `run_text_cond_train.py`: config-driven multi-seed train+eval launcher (reads `hyperparameters.json`)
- `main.py`: `TextConditionedVisionModel`, fusion modules, backbone loading helpers
- `vision_data.py`: dataset registry and split logic
- `build_csp_hf_datasets.py`: build/push CSP Hugging Face datasets (`--mode clevr` for CLEVR-style `csp_*` releases; `--mode reference` for MIT-States / UT-Zappos / C-GQA)
- `train_two_fusions_and_push.sh`: run `clip_similarity` and `cross_attention`, upload both, optional grid search + multi-seed
- `run_evals.py` / `slurm_run_evals.sh`: batch re-run val+test `--eval-only` on checkpoints
- `visualize_dinov3_attention.py`: visualize ViT **CLS→patch** self-attention (see below)

## Attention visualization (`visualize_dinov3_attention.py`)

Utility to plot **encoder self-attention** maps (mean over heads: **CLS token → image patches**) for a Hugging Face **ViT-style** backbone—default preset **DINOv3** (`--vision-backbone dinov3`, or `--model-id`).

**CLI modes**

1. **Single image** — pass `--image`, optional `--checkpoint` (loads `backbone` / `backbone.*` from a bundle or full model dict if present; else Hub weights). Choose `--layers` (0-based block indices) and `--out` for the figure.
2. **`--csp-compare`** — load two **CSP vocab** `.pt` bundles (`--csp-checkpoint-tuned` with finetuned `backbone`, `--csp-checkpoint-base` with heads only / backbone ignored), scan **`--csp-dataset` official test split** for **contrast samples** (tuned top-1 correct, pretrained-only wrong), then draw side-by-side attention grids. Optional export of PNGs + `manifest.json` under `--csp-out-dir` (see `--csp-save-samples-dir`, `--csp-no-save-samples`).

**Main functions (by role)**

| Area | Functions | Purpose |
|------|-----------|---------|
| Backbone I/O | `_extract_backbone_state`, `_load_backbone` | Pull vision `state_dict` from checkpoint; build `AutoModel` with `attn_implementation="eager"` and optional tuned weights. |
| ViT layout | `_encoder_layers`, `_patch_grid` | Locate transformer blocks; infer patch grid height/width and prefix token count from `config`. |
| Attention | `_forward_attentions`, `_forward_with_attention_hooks`, `_cls_to_patch_map`, `_upsample_map` | Run forward with attentions (or hooks fallback); turn last-layer weights into a patch map; resize to image size for overlay. |
| CSP bundles | `_require_csp_bundle`, `_bundle_training_args`, `_ijepa_id_from_bundle`, `_meta_from_bundle`, `_load_csp_textconditioned`, `_build_csp_vocab`, `_assert_meta_pairs_equal`, `_backbone_to_eager_attn` | Parse checkpoints; rebuild `TextConditionedVisionModel` + `CspCompositionVocab`; align `meta` across bundles; swap backbone to eager attention when needed. |
| CSP contrast + figures | `_csp_logits_one_image`, `_extract_layer_attention_maps`, `_scan_csp_contrast_samples`, `_save_csp_compare_sample_artifacts`, `_figure_csp_backbone_compare`, `run_csp_backbone_compare` | Per-image CSP logits; layer-wise maps for two models; search test split; save samples; build comparison figure; orchestrate compare mode. |
| Entry | `main` | Argument parsing; dispatch single-image vs `--csp-compare`. |

Examples (from the script docstring):

```bash
uv run python visualize_dinov3_attention.py --image path/to.jpg --checkpoint checkpoints/xxx.pt \
  --out attn.png --layers 6 8 9 10 11

uv run python visualize_dinov3_attention.py --csp-compare \
  --csp-checkpoint-tuned path/to/with_backbone.pt \
  --csp-checkpoint-base path/to/heads_only.pt \
  --csp-dataset cspref_mit_states --csp-n-samples 5 --csp-out-dir out_attn
```

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
- `--finetune-csp-vocab` (train CSP-style attr/object compositional soft prompts in `text_cond_train.py`; see script help for constraints)
- `--csp-vocab-init {random,text}`
- `--csp-attr-dropout 0.3`
- `--csp-pair-separator " "`
- `--vision-backbone {dinov3,ijepa,vjepa}` — default **`dinov3`** if omitted; or set explicit HF vision model id with `--ijepa <id>`
- `--hyperparams-file hyperparameters.json`
- `--finetune-clip-text`
- `--finetune-vision-backbone`
- `--hub-model-id user/repo` (push trainable weights after training; backbone still referenced by id in config)
- `--no-wandb` (disable W&B)

Backbone examples (DINOv3 is the default; the first command omits `--vision-backbone` on purpose):

```bash
# DINOv3 ViT-B/16 (default preset)
uv run python text_cond_train.py --dataset cspref_mit_states --epochs 1

# I-JEPA (preset)
uv run python text_cond_train.py --vision-backbone ijepa --dataset cspref_mit_states --epochs 1

# V-JEPA 2 (preset)
uv run python text_cond_train.py --vision-backbone vjepa --dataset cspref_mit_states --epochs 1
```

Dedicated CSP vocab training with `csp_vocab_train.py` (defaults include `--csp-vocab-init text`):

```bash
uv run python csp_vocab_train.py \
  --dataset cspref_mit_states \
  --epochs 20 \
  --batch-size 128 \
  --save ckpt_csp_vocab.pt
```

## Config-driven experiments

`run_text_cond_train.py` reads runtime/training parameters from `hyperparameters.json` (see script for how CLI interacts with the file):

```bash
uv run python run_text_cond_train.py
```

## Hyperparameters file

Merge order (CLI still wins when a flag is explicitly passed): `defaults` → `models.<vision_backbone>` → `datasets.<dataset>` → `model_dataset.<vision_backbone>.<dataset>`.

## Eval only

```bash
uv run python text_cond_train.py --eval-only --dataset cspref_mit_states --checkpoint ckpt_cross_attn.pt --eval-split val
uv run python text_cond_train.py --eval-only --dataset cspref_mit_states --checkpoint ckpt_cross_attn.pt --eval-split test
uv run python text_cond_train.py --eval-only --from-hub user/model --dataset cspref_mit_states --eval-split test
```

## Split behavior

Registry datasets (`cspref_*`) use the published Hub `train` / `val` / `test` splits only.
CLI flags `--val-fraction` and `--split-seed` are accepted for compatibility but ignored.

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
