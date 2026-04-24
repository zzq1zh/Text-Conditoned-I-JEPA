"""
Fine-tune the text-conditioning adapter + fusion head on a training split (I-JEPA frozen),
with optional Weights & Biases logging. Inline comments in English.
By default uses a CUDA GPU when ``torch.cuda.is_available()``; pass ``--cpu`` or ``--device cpu`` otherwise.
Default ``--batch-size`` / ``--num-workers`` / ``--log-interval`` assume a large GPU; reduce if OOM.
"""

from __future__ import annotations

import project_env

project_env.load_project_env()

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer

# Project imports (run from repo root, e.g. `uv run python text_cond_train.py ...`)
from main import (  # noqa: E402
    DEFAULT_CLIP_TEXT_ID,
    DEFAULT_IJEPA_ID,
    DEFAULT_PROMPT_TEMPLATE,
    TextConditionedIJepa,
    _resolve_device,
)
from vision_data import (  # noqa: E402
    list_vision_dataset_keys,
    load_vision_train_val_test_specs,
    prompts_for_label_indices,
    set_seed,
)


def _hf_image_to_pil_rgb(img: Any) -> Any:
    """
    Normalize Hugging Face ``Image`` feature values to RGB :class:`PIL.Image.Image`.
    Parquet / DataLoader workers may yield a ``dict`` (``path`` / ``bytes``) or ndarray
    without a channel layout the ViT image processor accepts.
    """
    from io import BytesIO

    from PIL import Image as PILImage

    if isinstance(img, PILImage.Image):
        return img.convert("RGB")
    if isinstance(img, dict):
        b = img.get("bytes")
        if b is not None:
            return PILImage.open(BytesIO(b)).convert("RGB")
        p = img.get("path")
        if p:
            return PILImage.open(p).convert("RGB")
    try:
        import numpy as np

        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                return PILImage.fromarray(img).convert("RGB")
            if img.ndim == 3 and img.shape[2] in (1, 3, 4):
                return PILImage.fromarray(np.asarray(img, dtype=np.uint8)).convert("RGB")
    except (TypeError, ValueError):
        pass
    raise TypeError(
        f"Expected PIL Image, HF image dict, or HxW[xC] array; got {type(img).__name__!r}"
    )


# Hugging Face Hub: trainable blocks only (no frozen I-JEPA backbone keys)
HUB_CONFIG_FILENAME = "tc_ijepa_config.json"
HUB_WEIGHTS_FILENAME = "trainable_model.safetensors"

# W&B: change this to your preferred default, or set WANDB_PROJECT in .env
DEFAULT_WANDB_PROJECT = "csci1430-tc-ijepa"


def _default_wandb_project() -> str:
    """``WANDB_PROJECT`` in .env overrides this (loaded before argparse via ``project_env``)."""
    p = (os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT).strip()
    return p or DEFAULT_WANDB_PROJECT


def _resolve_wandb_run_name(explicit_name: str, fallback_name: str) -> str:
    s = (explicit_name or "").strip()
    return s or fallback_name


def _default_train_wandb_run_name(args: argparse.Namespace) -> str:
    return f"train-{args.dataset}-{args.fusion_type}-seed{args.seed}"


def _default_eval_wandb_run_name(args: argparse.Namespace, *, fusion_type: str) -> str:
    return f"eval-{args.dataset}-{args.eval_split}-{fusion_type}-seed{args.seed}"


# Default dataloader: tuned for a single large GPU; lower --batch-size if OOM; --finetune-clip-text may need 8–12
DEFAULT_TRAIN_BATCH = 16
DEFAULT_TRAIN_NUM_WORKERS = 6
DEFAULT_LOG_INTERVAL = 32
CSPREF_RECOMMENDED: dict[str, dict[str, Any]] = {
    # Close to CSP-style hyperparameters, but adapted to this training script.
    "cspref_mit_states": {
        "epochs": 20,
        "batch_size": 128,
        "num_workers": 8,
        "lr": 5e-5,
        "weight_decay": 1e-5,
        "fusion_type": "linear",
        "text_template": "a photo of a {c}.",
    },
    "cspref_ut_zappos": {
        "epochs": 20,
        "batch_size": 128,
        "num_workers": 8,
        "lr": 5e-5,
        "weight_decay": 1e-5,
        "fusion_type": "linear",
        "text_template": "a photo of a {c}.",
    },
    "cspref_cgqa": {
        "epochs": 20,
        "batch_size": 128,
        "num_workers": 8,
        "lr": 5e-5,
        "weight_decay": 1e-5,
        "fusion_type": "linear",
        "text_template": "a photo of a {c}.",
    },
}


@dataclass
class TrainState:
    """Hyperparameters + paths logged to W&B (and used for the run)."""

    dataset_key: str
    ijepa_id: str
    clip_id: str
    val_fraction: float
    split_seed: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    max_grad_norm: float
    cond_dim: int
    fusion_hidden: int
    fusion_type: str
    text_template: str
    max_train_samples: int | None
    max_val_samples: int | None
    finetune_clip_text: bool
    use_amp: bool
    num_workers: int
    seed: int
    device: str
    device_arg: str
    wandb_log_images: bool
    wandb_image_log_interval: int
    wandb_max_images: int


def _msamples(x: int) -> int | None:
    return x if x and x > 0 else None


def _resolve_hub_repo_id(hub_model_id: str) -> str:
    """
    - Empty -> ``""`` (no upload / no Hub eval).
    - Non-empty must be a full Hub id ``namespace/name`` (must contain ``/``); no auto prefix.
    """
    s = (hub_model_id or "").strip()
    if not s:
        return ""
    if "/" not in s:
        raise ValueError(
            "Hub repo id must be a full id like ``user/model`` or ``org/model`` (include the ``/``); "
            "bare names are not accepted."
        )
    return s


def _arg_was_explicit(argv: list[str], key: str) -> bool:
    """Whether ``--key`` (or ``--key=...``) was explicitly passed by the user."""
    opt = f"--{key.replace('_', '-')}"
    for t in argv:
        if t == opt or t.startswith(opt + "="):
            return True
    return False


def _apply_cspref_recommended_config(args: argparse.Namespace, argv: list[str]) -> None:
    """
    Apply dataset-specific recommended settings for cspref_* datasets.
    Explicit CLI flags always override recommendations.
    """
    rec = CSPREF_RECOMMENDED.get(args.dataset)
    if rec is None:
        return
    applied: list[str] = []
    for k, v in rec.items():
        if _arg_was_explicit(argv, k):
            continue
        old_v = getattr(args, k)
        if old_v != v:
            setattr(args, k, v)
            applied.append(f"{k}={v!r}")
    if applied:
        print(
            f"Applied recommended config for {args.dataset}: " + ", ".join(applied),
            flush=True,
        )


def _resolve_train_device(args: argparse.Namespace) -> torch.device:
    """
    Default training device: use the first available CUDA GPU (same as :func:`main._resolve_device`
    with ``None``). Use ``--cpu`` or ``--device cpu`` to force CPU, or ``--device cuda`` to
    require a GPU and fail if CUDA is not available.
    """
    if args.cpu or args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "You passed --device cuda, but torch.cuda.is_available() is False. "
                "Install a CUDA build of PyTorch, or use the default (omit --device / --cpu) to fall back to CPU, "
                "or set --device auto."
            )
        return torch.device("cuda")
    # auto: prefer GPU, then CPU
    return _resolve_device(None)


class HfPilImageDataset(Dataset):
    """
    HuggingFace row -> ``dict`` with ``image``, ``label``, and optional ``pos`` / ``neg_0``..``neg_3``
    (CLEVR-style CSP Hub rows).
    """

    def __init__(self, hf_subset: Any, image_key: str, label_key: str) -> None:
        self.hf = hf_subset
        self.image_key = image_key
        self.label_key = label_key
        self._phrase_keys = ("pos", "neg_0", "neg_1", "neg_2", "neg_3")
        self._has_phrase_columns = all(
            k in self.hf.column_names for k in self._phrase_keys
        )

    def __len__(self) -> int:
        return len(self.hf)

    @property
    def has_phrase_columns(self) -> bool:
        return self._has_phrase_columns

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.hf[idx]
        out: dict[str, Any] = {
            "image": _hf_image_to_pil_rgb(row[self.image_key]),
            "label": int(row[self.label_key]),
        }
        if "pair_seen_in_train" in self.hf.column_names:
            out["pair_seen_in_train"] = bool(row["pair_seen_in_train"])
        if self._has_phrase_columns:
            for k in self._phrase_keys:
                v = row[k]
                out[k] = "" if v is None else str(v)
        return out


def make_collate_fn(
    processor: Any,
) -> Any:
    """
    Batches samples into ``pixel_values`` + long labels (+ optional seen flag).
    Text embeddings are precomputed from the global candidate pool.
    """

    def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        label_list = [int(d["label"]) for d in batch]
        images = [d["image"] for d in batch]
        vis = processor(images=images, return_tensors="pt")
        out: dict[str, Any] = {
            "pixel_values": vis["pixel_values"],
            "labels": torch.tensor(label_list, dtype=torch.long),
        }
        if "pair_seen_in_train" in batch[0]:
            out["pair_seen_in_train"] = torch.tensor(
                [bool(d["pair_seen_in_train"]) for d in batch], dtype=torch.bool
            )
        return out

    return _collate


@torch.inference_mode()
def _build_candidate_text_bank(
    model: TextConditionedIJepa,
    tokenizer: Any,
    class_names: list[str],
    text_template: str,
    device: torch.device,
    *,
    chunk_size: int = 512,
) -> torch.Tensor:
    """
    Precompute candidate text embeddings for all classes.
    Returns tensor of shape (C, cond_dim) on ``device``.
    """
    prompts = [text_template.format(c=c) for c in class_names]
    chunks: list[torch.Tensor] = []
    for s in range(0, len(prompts), chunk_size):
        e = min(len(prompts), s + chunk_size)
        enc = tokenizer(prompts[s:e], padding=True, return_tensors="pt", truncation=True)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        t = model.encode_text(ids, mask)
        chunks.append(t.detach())
    return torch.cat(chunks, dim=0)


def _clip_contrastive_loss(pair_scores: torch.Tensor) -> torch.Tensor:
    """
    CLIP-style symmetric InfoNCE loss from a pair score matrix.
    pair_scores: (B, B), where diagonal entries are positives.
    """
    if pair_scores.ndim != 2 or pair_scores.size(0) != pair_scores.size(1):
        raise ValueError(
            f"pair_scores must be square (B,B), got shape={tuple(pair_scores.shape)}"
        )
    target = torch.arange(pair_scores.size(0), device=pair_scores.device, dtype=torch.long)
    loss_i2t = nn.functional.cross_entropy(pair_scores, target)
    loss_t2i = nn.functional.cross_entropy(pair_scores.t(), target)
    return 0.5 * (loss_i2t + loss_t2i)


@torch.inference_mode()
def _compute_auc_ovr_macro(
    logits_list: list[torch.Tensor], labels_list: list[torch.Tensor], num_classes: int
) -> float:
    """
    Compute ROC-AUC for multiclass classification using OVR macro averaging.
    Returns NaN when it cannot be computed (e.g., missing classes in the split).
    """
    if not logits_list or not labels_list:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan")

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    if logits.ndim != 2 or labels.ndim != 1:
        return float("nan")
    if logits.size(0) != labels.size(0) or logits.size(1) != num_classes:
        return float("nan")
    if labels.numel() == 0:
        return float("nan")

    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    y = labels.cpu().numpy()
    try:
        if num_classes == 2:
            return float(roc_auc_score(y, probs[:, 1]))
        return float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
    except ValueError:
        # Typical on tiny subsets where not all classes are present.
        return float("nan")


@torch.inference_mode()
def _make_wandb_images(
    pixel_values: torch.Tensor,
    labels: torch.Tensor,
    preds: torch.Tensor,
    *,
    max_images: int,
    class_names: list[str] | None = None,
) -> list[Any]:
    """
    Build a small list of ``wandb.Image`` from a training batch.
    Uses per-image min-max normalization for visualization only.
    """
    import wandb

    pv = pixel_values.detach().cpu().float()
    y = labels.detach().cpu().long()
    p = preds.detach().cpu().long()
    n = min(int(max_images), int(pv.size(0)))
    out: list[Any] = []
    def _label_text(idx: int) -> str:
        if class_names is not None and 0 <= idx < len(class_names):
            return class_names[idx]
        return str(idx)

    for i in range(n):
        img = pv[i]
        img = img - img.min()
        den = img.max().clamp(min=1e-6)
        img = img / den
        yi = int(y[i])
        pi = int(p[i])
        out.append(
            wandb.Image(
                img,
                caption=(
                    f"predict label: {_label_text(pi)} ({pi}) ; "
                    f"true label: {_label_text(yi)} ({yi})"
                ),
            )
        )
    return out


@torch.inference_mode()
def _compute_auc_csp_style(
    logits_list: list[torch.Tensor],
    labels_list: list[torch.Tensor],
    seen_flags_list: list[torch.Tensor],
    num_classes: int,
) -> float:
    """
    CSP-style AUC:
    sweep an unseen-class bias and integrate seen/unseen top-1 trade-off.

    Returns NaN when seen/unseen supervision is unavailable (non-compositional datasets)
    or when the split does not contain enough seen/unseen examples.
    """
    if not logits_list or not labels_list or not seen_flags_list:
        return float("nan")
    logits = torch.cat(logits_list, dim=0).float()
    labels = torch.cat(labels_list, dim=0).long()
    seen_flags = torch.cat(seen_flags_list, dim=0).bool()
    if logits.ndim != 2 or labels.ndim != 1 or seen_flags.ndim != 1:
        return float("nan")
    if logits.size(0) != labels.numel() or labels.numel() != seen_flags.numel():
        return float("nan")
    if logits.size(1) != num_classes or num_classes <= 1:
        return float("nan")

    class_seen = torch.zeros(num_classes, dtype=torch.bool)
    for y, sf in zip(labels.tolist(), seen_flags.tolist(), strict=False):
        if 0 <= y < num_classes and bool(sf):
            class_seen[y] = True
    if (not class_seen.any()) or bool(class_seen.all()):
        return float("nan")

    unseen_sample_mask = ~class_seen[labels]
    if unseen_sample_mask.sum().item() == 0:
        return float("nan")
    seen_sample_mask = ~unseen_sample_mask
    if seen_sample_mask.sum().item() == 0:
        return float("nan")

    unseen_logits = logits[unseen_sample_mask]
    unseen_labels = labels[unseen_sample_mask]
    correct_scores = unseen_logits.gather(1, unseen_labels.view(-1, 1)).squeeze(1)
    max_seen_scores = unseen_logits[:, class_seen].max(dim=1).values
    unseen_score_diff = max_seen_scores - correct_scores

    pred0 = logits.argmax(dim=1)
    unseen_matches0 = pred0[unseen_sample_mask].eq(unseen_labels)
    correct_unseen_diff = unseen_score_diff[unseen_matches0] - 1e-4
    if correct_unseen_diff.numel() == 0:
        return float("nan")
    correct_unseen_diff = torch.sort(correct_unseen_diff).values
    magic_binsize = 20
    bias_skip = max(correct_unseen_diff.numel() // magic_binsize, 1)
    bias_list = correct_unseen_diff[::bias_skip]

    seen_acc: list[float] = []
    unseen_acc: list[float] = []
    unseen_class_mask = ~class_seen
    for bias in bias_list.tolist():
        s = logits.clone()
        s[:, unseen_class_mask] += float(bias)
        pred = s.argmax(dim=1)
        seen_acc.append(float(pred[seen_sample_mask].eq(labels[seen_sample_mask]).float().mean().item()))
        unseen_acc.append(float(pred[unseen_sample_mask].eq(labels[unseen_sample_mask]).float().mean().item()))

    seen_acc.append(float(pred0[seen_sample_mask].eq(labels[seen_sample_mask]).float().mean().item()))
    unseen_acc.append(float(pred0[unseen_sample_mask].eq(labels[unseen_sample_mask]).float().mean().item()))
    try:
        import numpy as np

        x = np.asarray(unseen_acc)
        y = np.asarray(seen_acc)
        if hasattr(np, "trapezoid"):
            return float(np.trapezoid(y, x))
        return float(np.trapz(y, x))
    except (ValueError, TypeError):
        return float("nan")


@torch.inference_mode()
def _eval_loss_topk(
    model: TextConditionedIJepa,
    loader: DataLoader,
    device: torch.device,
    candidate_text_bank: torch.Tensor,
    *,
    use_amp: bool,
    num_classes: int,
    use_bidirectional_infonce: bool = True,
) -> tuple[float, float, float, float, float]:
    """
    Mean loss (if labels provided), top-1 accuracy, top-5 accuracy.
    Top-k uses k = min(5, num_classes).
    """
    model.eval()
    n_ok1 = 0
    n_ok5 = 0
    n = 0
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_seen_flags: list[torch.Tensor] = []
    k_top = min(5, num_classes)
    for batch in loader:
        pv = batch["pixel_values"].to(device)
        y = batch["labels"].to(device)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda" and use_amp),
            dtype=torch.float16,
        ):
            z = model.encode_image(pv)
            logits = model.score_candidates(z, candidate_text_bank)
            if not use_bidirectional_infonce:
                raise RuntimeError("Only bidirectional_infonce loss is supported.")
            pos_text = candidate_text_bank.index_select(0, y)
            pair_scores = model.score_candidates(z, pos_text)
            loss = _clip_contrastive_loss(pair_scores)
        total_loss += float(loss.item()) * y.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())
        if "pair_seen_in_train" in batch:
            all_seen_flags.append(batch["pair_seen_in_train"].detach().cpu().bool())
        pred1 = logits.argmax(dim=-1)
        n_ok1 += (pred1 == y).sum().item()
        if k_top > 0:
            topk_idx = logits.topk(k_top, dim=-1).indices
            n_ok5 += (topk_idx == y.unsqueeze(-1)).any(dim=-1).sum().item()
        n += y.size(0)
    if n == 0:
        return 0.0, 0.0, 0.0, float("nan"), float("nan")
    auc_ovr_macro = _compute_auc_ovr_macro(all_logits, all_labels, num_classes)
    auc_csp_style = _compute_auc_csp_style(all_logits, all_labels, all_seen_flags, num_classes)
    return total_loss / n, n_ok1 / n, n_ok5 / n, auc_ovr_macro, auc_csp_style


def _export_trainable_state_dict(model: TextConditionedIJepa) -> dict[str, torch.Tensor]:
    """I-JEPA backbone is frozen and loaded from the Hub; omit it to keep uploads small."""
    out: dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        if k.startswith("backbone."):
            continue
        t = v.contiguous()
        if t.is_floating_point() and t.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            t = t.float()
        out[k] = t.cpu()
    if not out:
        raise RuntimeError("No non-backbone weights to export; unexpected model state.")
    return out


def _hub_config_dict(
    args: argparse.Namespace,
    class_names: list[str],
    num_labels: int,
) -> dict[str, Any]:
    return {
        "model_type": "text_cond_ijepa",
        "ijepa_id": args.ijepa,
        "clip_id": args.clip,
        "num_labels": int(num_labels),
        "cond_dim": int(args.cond_dim),
        "fusion_hidden": int(args.fusion_hidden),
        "fusion_type": str(args.fusion_type),
        "loss_mode": "bidirectional_infonce",
        "finetune_clip_text": bool(args.finetune_clip_text),
        "dataset": args.dataset,
        "text_template": args.text_template,
        "val_fraction": float(args.val_fraction),
        "split_seed": int(args.split_seed),
        "class_names": list(class_names),
        "text_bank_chunk_size": int(args.text_bank_chunk_size),
    }


def push_text_cond_to_hub(
    model: TextConditionedIJepa,
    config: dict[str, Any],
    repo_id: str,
    *,
    private: bool = False,
    token: str | None = None,
) -> str:
    from huggingface_hub import HfApi, get_token
    from safetensors.torch import save_file

    token = (token or "").strip() or get_token()
    if not token:
        raise RuntimeError(
            "Hugging Face token not found. Set HF_TOKEN in .env (see .env.example) or run: huggingface-cli login"
        )
    trainable = _export_trainable_state_dict(model)
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        wpath = d / HUB_WEIGHTS_FILENAME
        cpath = d / HUB_CONFIG_FILENAME
        save_file(trainable, str(wpath))
        cpath.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        api.upload_file(
            path_in_repo=HUB_WEIGHTS_FILENAME,
            path_or_fileobj=str(wpath),
            repo_id=repo_id,
            repo_type="model",
        )
        api.upload_file(
            path_in_repo=HUB_CONFIG_FILENAME,
            path_or_fileobj=str(cpath),
            repo_id=repo_id,
            repo_type="model",
        )
    return f"https://huggingface.co/{repo_id}"


def load_text_cond_trainable_from_hub(
    repo_id: str,
    device: torch.device,
    *,
    token: str | None = None,
) -> tuple[TextConditionedIJepa, dict[str, Any]]:
    from huggingface_hub import get_token, hf_hub_download
    from safetensors.torch import load_file

    token = (token or "").strip() or get_token()
    cfg_p = hf_hub_download(repo_id, HUB_CONFIG_FILENAME, token=token)
    w_p = hf_hub_download(repo_id, HUB_WEIGHTS_FILENAME, token=token)
    with open(cfg_p, encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("model_type") != "text_cond_ijepa":
        raise ValueError(
            f"Config model_type must be 'text_cond_ijepa', got {cfg.get('model_type')!r}"
        )
    model = TextConditionedIJepa(
        num_labels=int(cfg["num_labels"]),
        ijepa_id=str(cfg["ijepa_id"]),
        clip_id=str(cfg["clip_id"]),
        cond_dim=int(cfg["cond_dim"]),
        fusion_hidden=int(cfg["fusion_hidden"]),
        fusion_type=str(cfg.get("fusion_type", "cross_attention")),
        freeze_text_encoder=not bool(cfg.get("finetune_clip_text", False)),
    )
    model.to(device)
    sd = load_file(w_p)
    r = model.load_state_dict(sd, strict=False)
    miss = [k for k in r.missing_keys if not k.startswith("backbone.")]
    if miss:
        raise RuntimeError(
            f"After loading Hub trainable weights, still missing (non-backbone) keys: {miss[:12]}..."
        )
    uexp = [k for k in r.unexpected_keys if not k.startswith("backbone.")]
    if uexp:
        raise RuntimeError(f"Unexpected keys in checkpoint: {uexp[:12]}...")
    return model, cfg


def run_finetune(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = _resolve_train_device(args)
    if device.type == "cuda":
        print(f"Training on GPU: {torch.cuda.get_device_name(device.index or 0)}", flush=True)
    else:
        print(
            "Training on CPU (to use a GPU, install CUDA-enabled PyTorch; omit --cpu / --device cpu).",
            flush=True,
        )
    tvt = load_vision_train_val_test_specs(
        args.dataset,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        max_train_samples=_msamples(int(args.max_train_samples)),
        max_val_samples=_msamples(int(args.max_val_samples)),
        max_test_samples=None,
    )
    n_classes = len(tvt.train.class_names)
    train_ds = HfPilImageDataset(
        tvt.train.dataset, tvt.train.image_column, tvt.train.label_key
    )
    val_ds = HfPilImageDataset(
        tvt.val.dataset, tvt.val.image_column, tvt.val.label_key
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Train or val split is empty. Check --max-train/val-samples and val-fraction")

    im_proc = AutoImageProcessor.from_pretrained(args.ijepa)
    model = TextConditionedIJepa(
        num_labels=n_classes,
        ijepa_id=args.ijepa,
        clip_id=args.clip,
        cond_dim=args.cond_dim,
        fusion_hidden=args.fusion_hidden,
        fusion_type=args.fusion_type,
        freeze_text_encoder=not args.finetune_clip_text,
    )
    model.to(device)

    tok = AutoTokenizer.from_pretrained(args.clip)
    collate = make_collate_fn(im_proc)

    # Avoid empty batches on tiny subsampled training sets
    _drop = len(train_ds) >= 2 * args.batch_size
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
        drop_last=_drop,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters; try --finetune-clip-text for CLIP text adapter + fusion")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    resolved_loss_mode = "bidirectional_infonce"
    use_bidirectional_infonce = True
    print(
        f"Using loss_mode={resolved_loss_mode}: CLIP-style contrastive loss (symmetric InfoNCE).",
        flush=True,
    )

    state = TrainState(
        dataset_key=args.dataset,
        ijepa_id=args.ijepa,
        clip_id=args.clip,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        cond_dim=args.cond_dim,
        fusion_hidden=args.fusion_hidden,
        fusion_type=args.fusion_type,
        text_template=args.text_template,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        finetune_clip_text=args.finetune_clip_text,
        use_amp=bool(device.type == "cuda" and args.amp),
        num_workers=args.num_workers,
        seed=args.seed,
        device=str(device),
        device_arg=args.device,
        wandb_log_images=bool(args.wandb_log_images),
        wandb_image_log_interval=int(args.wandb_image_log_interval),
        wandb_max_images=int(args.wandb_max_images),
    )

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb

        # project_env already loaded .env; W&B also reads WANDB_API_KEY from the environment
        w_key = (os.environ.get("WANDB_API_KEY") or "").strip()
        if w_key:
            wandb.login(key=w_key, relogin=True)
        run_name = _resolve_wandb_run_name(
            args.wandb_run_name,
            _default_train_wandb_run_name(args),
        )
        w = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=run_name,
            config=asdict(state),
        )
    else:
        w = None

    global_step = 0
    use_cuda_amp = device.type == "cuda" and args.amp
    grad_scaler: torch.amp.GradScaler | None
    if use_cuda_amp:
        grad_scaler = torch.amp.GradScaler("cuda")
    else:
        grad_scaler = None
    t0 = time.time()

    for epoch in range(args.epochs):
        # Recompute candidate text embeddings each epoch from the current text adapter.
        candidate_text_bank = _build_candidate_text_bank(
            model,
            tok,
            tvt.train.class_names,
            args.text_template,
            device,
            chunk_size=int(args.text_bank_chunk_size),
        )
        model.train()
        pbar = tqdm(
            train_loader, desc=f"train ep {epoch+1}/{args.epochs}", file=sys.stdout
        )
        for batch in pbar:
            pv = batch["pixel_values"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_cuda_amp,
                dtype=torch.float16,
            ):
                z = model.encode_image(pv)
                pos_text = candidate_text_bank.index_select(0, y)
                pair_scores = model.score_candidates(z, pos_text)
                loss = _clip_contrastive_loss(pair_scores)
            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
            else:
                loss.backward()
            if args.max_grad_norm > 0:
                if grad_scaler is not None:
                    grad_scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            if grad_scaler is not None:
                grad_scaler.step(opt)
                grad_scaler.update()
            else:
                opt.step()

            with torch.inference_mode():
                full_logits = model.score_candidates(z, candidate_text_bank)
                pred = full_logits.argmax(-1)
                batch_acc = (pred == y).float().mean().item()
            if use_wandb and w is not None:
                import wandb

                log_d: dict[str, Any] = {}
                if global_step % args.log_interval == 0:
                    log_d.update(
                        {
                            "train/loss": float(loss.item()),
                            "train/batch_acc": batch_acc,
                            "train/lr": opt.param_groups[0]["lr"],
                            "time_elapsed_s": time.time() - t0,
                        }
                    )
                    log_d["train/loss_contrastive"] = float(loss.item())

                if (
                    args.wandb_log_images
                    and args.wandb_image_log_interval > 0
                    and global_step % args.wandb_image_log_interval == 0
                ):
                    log_d["train/images"] = _make_wandb_images(
                        pv,
                        y,
                        pred,
                        max_images=int(args.wandb_max_images),
                        class_names=tvt.train.class_names,
                    )
                if log_d:
                    wandb.log(log_d, step=global_step)
            pbar.set_postfix(
                loss=f"{float(loss.detach()):.3f}", acc=f"{batch_acc:.3f}"
            )
            global_step += 1

        val_loss, val_top1, val_top5, val_auc_ovr, val_auc_csp = _eval_loss_topk(
            model,
            val_loader,
            device,
            candidate_text_bank,
            use_amp=(device.type == "cuda" and args.amp),
            num_classes=n_classes,
            use_bidirectional_infonce=use_bidirectional_infonce,
        )
        if use_wandb and w is not None:
            import wandb

            wandb.log(
                {
                    "epoch": epoch,
                    "val/loss": val_loss,
                    "val/acc@1": val_top1,
                    "val/acc@5": val_top5,
                    "val/auc_ovr_macro": val_auc_ovr,
                    "val/auc_csp_style": val_auc_csp,
                },
                step=global_step,
            )
        print(
            f"Epoch {epoch+1}/{args.epochs}  val_loss={val_loss:.4f}  "
            f"top-1={val_top1*100:.2f}%  top-5={val_top5*100:.2f}%  "
            f"auc_csp_style={val_auc_csp:.4f}  auc_ovr_macro={val_auc_ovr:.4f}  "
            f"(train samples={len(train_ds)}, val samples={len(val_ds)})"
        )

    if use_wandb and w is not None:
        import wandb

        wandb.finish()

    if args.save and args.save.strip():
        torch.save(model.state_dict(), args.save)
        print(f"Saved state_dict to {args.save}", flush=True)

    hub_raw = (args.hub_model_id or "").strip()
    if hub_raw:
        rid = _resolve_hub_repo_id(hub_raw)
        url = push_text_cond_to_hub(
            model,
            _hub_config_dict(args, tvt.train.class_names, n_classes),
            rid,
            private=bool(args.hub_private),
            token=(args.hub_token or "").strip() or None,
        )
        print(
            f"Pushed trainable weights to Hub (I-JEPA backbone excluded: use ijepa_id from config to load it). {url}",
            flush=True,
        )


def run_eval_only(args: argparse.Namespace) -> None:
    """Top-1 / Top-5 on val or the merged test split; optional checkpoint or Hub."""
    set_seed(args.seed)
    device = _resolve_train_device(args)
    if device.type == "cuda":
        print(f"Eval on GPU: {torch.cuda.get_device_name(device.index or 0)}", flush=True)
    else:
        print("Eval on CPU.", flush=True)

    from_hub_raw = (args.from_hub or "").strip()
    if from_hub_raw:
        from_hub = _resolve_hub_repo_id(from_hub_raw)
    else:
        from_hub = ""
    ckpt = (args.checkpoint or "").strip()
    if from_hub and ckpt:
        raise ValueError("Use either --from-hub or --checkpoint, not both.")

    me = _msamples(int(args.max_eval_samples))
    tvt = load_vision_train_val_test_specs(
        args.dataset,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        max_train_samples=None,
        max_val_samples=me if args.eval_split == "val" else None,
        max_test_samples=me if args.eval_split == "test" else None,
    )
    if args.eval_split == "val":
        spec = tvt.val
    else:
        spec = tvt.test
    eval_ds = HfPilImageDataset(spec.dataset, spec.image_column, spec.label_key)
    if len(eval_ds) == 0:
        raise RuntimeError("Eval split is empty. Check --max-eval-samples and data config.")

    if from_hub:
        model, cfg = load_text_cond_trainable_from_hub(
            from_hub, device, token=(args.hub_token or "").strip() or None
        )
        class_names: list[str] = list(cfg["class_names"])
        n_classes = len(class_names)
        if len(tvt.train.class_names) != n_classes:
            raise ValueError(
                f"Hub model num_labels={n_classes} but current --dataset has "
                f"{len(tvt.train.class_names)} classes. Use the same --dataset and split as training."
            )
        ijepa_id = str(cfg["ijepa_id"])
        clip_id = str(cfg["clip_id"])
        tpl = str(cfg.get("text_template", args.text_template))
        print(f"Loaded trainable weights from the Hub: {from_hub}", flush=True)
    else:
        n_classes = len(tvt.train.class_names)
        class_names = tvt.train.class_names
        ijepa_id = args.ijepa
        clip_id = args.clip
        tpl = args.text_template
        model = TextConditionedIJepa(
            num_labels=n_classes,
            ijepa_id=ijepa_id,
            clip_id=clip_id,
            cond_dim=args.cond_dim,
            fusion_hidden=args.fusion_hidden,
            fusion_type=args.fusion_type,
            freeze_text_encoder=not args.finetune_clip_text,
        )
        model.to(device)
        if ckpt:
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            print(f"Loaded weights from {ckpt}", flush=True)
        else:
            print(
                "No --checkpoint: evaluation uses a freshly initialized head (I-JEPA/CLIP are pretrained).",
                flush=True,
            )

    im_proc = AutoImageProcessor.from_pretrained(ijepa_id)
    tok = AutoTokenizer.from_pretrained(clip_id)
    collate = make_collate_fn(im_proc)
    candidate_text_bank = _build_candidate_text_bank(
        model,
        tok,
        class_names,
        tpl,
        device,
        chunk_size=int(args.text_bank_chunk_size),
    )
    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )

    loss, top1, top5, auc_ovr, auc_csp = _eval_loss_topk(
        model,
        loader,
        device,
        candidate_text_bank,
        use_amp=(device.type == "cuda" and args.amp),
        num_classes=n_classes,
        use_bidirectional_infonce=True,
    )
    split_name = f"{args.eval_split} (n={len(eval_ds)})"
    print(
        f"\n{split_name}\n  loss: {loss:.4f}\n  top-1: {top1*100:.2f}%\n  top-5: {top5*100:.2f}%\n  auc_csp_style: {auc_csp:.4f}\n  auc_ovr_macro: {auc_ovr:.4f}\n",
        flush=True,
    )
    if not args.no_wandb:
        import wandb

        w_key = (os.environ.get("WANDB_API_KEY") or "").strip()
        if w_key:
            wandb.login(key=w_key, relogin=True)
        run_name = _resolve_wandb_run_name(
            args.wandb_run_name,
            _default_eval_wandb_run_name(
                args,
                fusion_type=str(getattr(model.fusion, "fusion_type", args.fusion_type)),
            ),
        )
        w = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=run_name,
            config={
                "mode": "eval_only",
                "dataset": args.dataset,
                "eval_split": args.eval_split,
                "fusion_type": str(getattr(model.fusion, "fusion_type", args.fusion_type)),
                "seed": int(args.seed),
                "from_hub": from_hub,
                "checkpoint": ckpt,
            },
        )
        wandb.log(
            {
                f"{args.eval_split}/loss": float(loss),
                f"{args.eval_split}/acc@1": float(top1),
                f"{args.eval_split}/acc@5": float(top5),
                f"{args.eval_split}/auc_csp_style": float(auc_csp),
                f"{args.eval_split}/auc_ovr_macro": float(auc_ovr),
            },
            step=0,
        )
        wandb.finish()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fine-tune text-conditioned I-JEPA (adapter + fusion) with optional W&B",
        epilog="Example:  uv run python text_cond_train.py --epochs 2  # add --no-wandb to skip W&B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ijepa", default=DEFAULT_IJEPA_ID, help="HuggingFace I-JEPA id")
    p.add_argument("--clip", default=DEFAULT_CLIP_TEXT_ID, help="HuggingFace CLIP id (text + tokenizer)")
    p.add_argument(
        "--dataset",
        default="cifar100",
        choices=list_vision_dataset_keys(),
    )
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--text-template", default=DEFAULT_PROMPT_TEMPLATE)
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Cap train size after split; 0 = full",
    )
    p.add_argument(
        "--max-val-samples", type=int, default=0, help="Cap val size; 0 = full"
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_TRAIN_BATCH,
        help="Lower on low VRAM or if OOM; with --finetune-clip-text try 8-12",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument(
        "--text-bank-chunk-size",
        type=int,
        default=512,
        help="Chunk size when precomputing all candidate text embeddings.",
    )
    p.add_argument("--cond-dim", type=int, default=256)
    p.add_argument("--fusion-hidden", type=int, default=512)
    p.add_argument(
        "--fusion-type",
        choices=("cross_attention", "linear", "clip_similarity"),
        default="cross_attention",
        help="Fusion head type for visual+text conditioning. `clip_similarity` uses CLIP-style normalized dot-product scoring.",
    )
    p.add_argument(
        "--finetune-clip-text",
        action="store_true",
        help="Unfreeze CLIP text encoder in addition to adapter+fusion (heavier, needs more VRAM)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Compute device. Default `auto` uses a CUDA GPU when available (training defaults to GPU).",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU; same as --device cpu (overrides --device).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_TRAIN_NUM_WORKERS,
        help="DataLoader workers (0 = main process only, slower I/O)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True, help="CUDA autocast fp16 (default: on for CUDA)"
    )
    p.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL)
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B (offline mode still uses login unless WANDB_MODE=disabled)",
    )
    p.add_argument(
        "--wandb-project",
        default=_default_wandb_project(),
        help=(
            f"W&B project name (code default: {DEFAULT_WANDB_PROJECT!r}; override with "
            "WANDB_PROJECT in .env or this flag)"
        ),
    )
    p.add_argument("--wandb-entity", default="", help="W&B team/entity (optional)")
    p.add_argument("--wandb-run-name", default="", help="W&B run display name (optional)")
    p.add_argument(
        "--wandb-log-images",
        action="store_true",
        help="Log training batch images to W&B (caption includes y/pred).",
    )
    p.add_argument(
        "--wandb-image-log-interval",
        type=int,
        default=1,
        help="Log images every N global steps when --wandb-log-images is enabled.",
    )
    p.add_argument(
        "--wandb-max-images",
        type=int,
        default=8,
        help="Maximum number of images logged per image logging step.",
    )
    p.add_argument(
        "--save", default="", help="Optional path to save model.state_dict() after training"
    )
    h = p.add_argument_group("Hugging Face Hub (set --hub-model-id to push after training; or use --from-hub)")
    h.add_argument(
        "--hub-model-id",
        default="",
        help=(
            f"If set, after training upload ``trainable_model.safetensors`` + {HUB_CONFIG_FILENAME} "
            f"(I-JEPA backbone not uploaded; needs HF_TOKEN). Use full id ``user/model`` or ``org/model``"
        ),
    )
    h.add_argument(
        "--hub-private", action="store_true", help="Create or update the repo as private"
    )
    h.add_argument(
        "--hub-token",
        default="",
        help="Write token; default: HF_TOKEN in .env or the huggingface-cli login cache",
    )
    g = p.add_argument_group("evaluation (use with --eval-only)")
    g.add_argument(
        "--eval-only",
        action="store_true",
        help="Run top-1 / top-5 on --eval-split and exit (no training)",
    )
    g.add_argument(
        "--from-hub",
        default="",
        help="Full Hub id ``user/model`` (or ``org/model``). Mutually exclusive with --checkpoint",
    )
    g.add_argument(
        "--checkpoint",
        default="",
        help="Path to a saved state_dict; omit for an untrained fusion head (still pretrained backbone/text)",
    )
    g.add_argument(
        "--eval-split",
        choices=("val", "test"),
        default="val",
        help="val = held-out from Hub train; test = val ∪ official Hub test (see vision_data)",
    )
    g.add_argument(
        "--max-eval-samples",
        type=int,
        default=0,
        help="Cap eval split size; 0 = full (applies to the chosen --eval-split only)",
    )
    return p


def main() -> None:
    argv = sys.argv[1:]
    args = build_parser().parse_args(argv)
    _apply_cspref_recommended_config(args, argv)
    if args.eval_only:
        run_eval_only(args)
    else:
        run_finetune(args)


if __name__ == "__main__":
    main()
