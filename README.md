# Text-Conditioned I-JEPA & Vision Pipelines

This repository trains and evaluates **text-conditioned I-JEPA** classifiers: a frozen I-JEPA vision backbone, a CLIP text tower (and optional finetune), and a small fusion MLP. It also provides **CLIP zero-shot** baselines and shared **Hugging Face Datasets** loading for CIFAR, MIT-States, and custom CSP (CLEVR-style) Hub datasets.

## Requirements

- **Python 3.12** (see `pyproject.toml`)
- **[uv](https://github.com/astral-sh/uv)** (recommended) or another PEP 508 installer

## Setup

```bash
cd /path/to/final-project
uv sync
cp .env.example .env
# Set HF_TOKEN (optional, for private Hub models / uploads) and WANDB_API_KEY (optional) in .env
```

Environment variables are loaded via `project_env` before training scripts run.

## Key scripts

| Script | Purpose |
|--------|--------|
| `text_cond_train.py` | Main entry: finetune the text adapter + fusion head; optional W&B; optional push to the Hugging Face Hub. |
| `main.py` | Model definitions (`TextConditionedIJepa`, I-JEPA helpers) and a small **I-JEPA / text-fusion smoke test** (`--text-fusion`). |
| `clip_pipeline.py` | **CLIP zero-shot** classification on datasets registered in `vision_data.py`. |
| `vision_data.py` | **Dataset registry** and split logic (`load_vision_train_val_test_specs`, etc.). |
| `build_csp_hf_datasets.py` | Build/prep CSP-style Parquet/zip data for the Hub (see script docstring). |

## Text-conditioned I-JEPA (training)

Default: I-JEPA vision encoder + CLIP text + fusion; backbone frozen unless you change the code.

```bash
uv run python text_cond_train.py --dataset cifar100 --epochs 1 --no-wandb
```

**Useful flags**

- `--device cuda` or `--cpu` for device selection.
- `--finetune-clip-text` to unfreeze the CLIP text stack (heavier VRAM).
- `--save path/to.pt` to save trainable weights (backbone is not stored).
- `--hub-model-id user/repo` to upload trainable weights + `tc_ijepa_config.json` after training (requires `HF_TOKEN`).

**Evaluation only**

```bash
uv run python text_cond_train.py --eval-only --dataset cifar100 --eval-split val
uv run python text_cond_train.py --eval-only --from-hub user/model --dataset cifar100
```

## CSP datasets (`pos` / `neg_0` … `neg_3`)

For Hub datasets with columns `pos` and `neg_0`–`neg_3` (e.g. `csp_single_object`, `csp_two_object`), training uses:

- **Positive prompt** from `pos` (via `--text-template`, e.g. `a photo of a {c}.` with `c` = phrase).
- **Optional contrastive term** (default on): among one positive and four negative phrases, the logit for the **true class** should be largest on the **positive** text. Set `--contrast-loss-weight 0` to keep phrase-based CE only and skip extra negative-text batches.

Datasets that only expose class names (e.g. CIFAR) keep the original **class-name prompts**; no `neg_*` term is used.

## CLIP zero-shot (baseline)

```bash
uv run python clip_pipeline.py --dataset cifar100
```

## Project layout (high level)

- `text_cond_train.py` — training loop, `HfPilImageDataset`, collate (phrase + optional negatives), Hub export/load.
- `main.py` — `TextConditionedIJepa` forward (CE + optional multi-text contrast), small smoke tests.
- `vision_data.py` — `DATASET_CONFIG` keys: `cifar10`, `cifar100`, `mit_states`, `csp_two_object`, `csp_single_object`, `csp_rel`.

## Optional: Weights & Biases

Omit `--no-wandb` to log runs. Configure `WANDB_API_KEY` / `WANDB_PROJECT` in `.env` or the CLI (see `text_cond_train.py --help`).

## License / course policy

If this is a course submission, follow your instructor’s rules for attribution, dataset use, and what may be published publicly.
