"""
Standalone post-training for CSP-style compositional vocabulary.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

import project_env

project_env.load_project_env()

from main import (  # noqa: E402
    DEFAULT_CLIP_TEXT_ID,
    DEFAULT_PROMPT_TEMPLATE,
    TextConditionedVisionModel,
    VISION_BACKBONE_PRESETS,
    _extract_model_pixel_values,
    load_vision_processor,
    resolve_vision_model_id,
)
from vision_data import (  # noqa: E402
    csp_vocab_allowed_class_indices,
    list_vision_dataset_keys,
    load_vision_train_val_test_specs,
    set_seed,
)

from csp_eval import clip_contrastive_loss, eval_clip_style_classification  # noqa: E402

# W&B: same default project as text_cond_train; override with WANDB_PROJECT in .env
DEFAULT_WANDB_PROJECT = "csci1430-tc-ijepa"


def _default_wandb_project() -> str:
    p = (os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT).strip()
    return p or DEFAULT_WANDB_PROJECT


def _resolve_wandb_run_name(explicit_name: str, fallback_name: str) -> str:
    s = (explicit_name or "").strip()
    return s or fallback_name


def _wandb_model_suffix(args: argparse.Namespace) -> str:
    backbone = (getattr(args, "vision_backbone", "") or "").strip()
    if backbone:
        return backbone
    model_id = (getattr(args, "ijepa", "") or "").strip()
    if model_id:
        return model_id.rsplit("/", 1)[-1]
    return "model"


def _default_csp_vocab_wandb_run_name(args: argparse.Namespace) -> str:
    return (
        f"csp-vocab-{args.dataset}-{args.fusion_type}-seed{args.seed}-{_wandb_model_suffix(args)}"
    )


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
    One W&B image panel (2x4-style grid) with predicted / true labels.
    Matches ``text_cond_train._make_wandb_images`` (duplicated to avoid importing that module).
    """
    import wandb
    from PIL import Image, ImageDraw

    def _to_wandb_image_tensor(img: torch.Tensor) -> torch.Tensor:
        if img.ndim == 4:
            img = img[img.shape[0] // 2]
        if img.ndim == 3 and img.shape[0] not in (1, 3) and img.shape[-1] in (1, 3):
            img = img.permute(2, 0, 1)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        if img.ndim != 3:
            raise ValueError(f"Unsupported image tensor shape for wandb.Image: {tuple(img.shape)}")
        return img

    def _to_uint8_hwc(img_chw: torch.Tensor) -> Any:
        x = img_chw.float()
        x = x - x.min()
        x = x / x.max().clamp(min=1e-6)
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        x = (x * 255.0).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        return x

    pv = pixel_values.detach().cpu().float()
    y = labels.detach().cpu().long()
    p = preds.detach().cpu().long()
    n = min(int(max_images), int(pv.size(0)))
    if n <= 0:
        return []

    def _label_text(idx: int) -> str:
        if class_names is not None and 0 <= idx < len(class_names):
            return class_names[idx]
        return str(idx)

    cols = 4
    rows = max((n + cols - 1) // cols, 1)
    tile_w = 196
    tile_h = 196
    text_h = 40
    canvas = Image.new("RGB", (cols * tile_w, rows * (tile_h + text_h)), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for i in range(n):
        img = _to_wandb_image_tensor(pv[i])
        arr = _to_uint8_hwc(img)
        tile = Image.fromarray(arr).resize((tile_w, tile_h))
        yi = int(y[i])
        pi = int(p[i])
        r = i // cols
        c = i % cols
        x0 = c * tile_w
        y0 = r * (tile_h + text_h)
        canvas.paste(tile, (x0, y0))
        draw.text((x0 + 4, y0 + tile_h + 2), f"P: {_label_text(pi)} ({pi})", fill=(20, 20, 20))
        draw.text((x0 + 4, y0 + tile_h + 20), f"T: {_label_text(yi)} ({yi})", fill=(20, 20, 20))
    return [wandb.Image(canvas, caption=f"Preview panel ({n} images)")]


DEFAULT_TRAIN_BATCH = 128
DEFAULT_TRAIN_NUM_WORKERS = 4
DEFAULT_LOG_INTERVAL = 32
DEFAULT_HYPERPARAMS_FILE = "hyperparameters.json"
HYPERPARAM_OVERRIDABLE_KEYS: frozenset[str] = frozenset(
    {
        "epochs",
        "batch_size",
        "num_workers",
        "lr",
        "scheduler_type",
        "warmup_ratio",
        "min_lr_ratio",
        "weight_decay",
        "cond_dim",
        "fusion_hidden",
        "fusion_type",
        "text_template",
        "text_bank_chunk_size",
        "max_grad_norm",
        "amp",
        "csp_vocab_init",
        "csp_attr_dropout",
        "csp_pair_separator",
        "csp_context_length",
    }
)


def _msamples(x: int) -> int | None:
    return x if x and x > 0 else None


def _default_save_from_base(base_checkpoint: str) -> str:
    """``path/model.pt`` -> ``path/model_csp_vocab.pt``; no base -> ``csp_vocab_posttrain.pt``."""
    b = (base_checkpoint or "").strip()
    if not b:
        return "csp_vocab_posttrain.pt"
    p = Path(b)
    return str(p.with_name(f"{p.stem}_csp_vocab{p.suffix}"))


def _arg_was_explicit(argv: list[str], key: str) -> bool:
    opt = f"--{key.replace('_', '-')}"
    for t in argv:
        if t == opt or t.startswith(opt + "="):
            return True
    return False


def _apply_hparam_overrides(
    args: argparse.Namespace, argv: list[str], overrides: dict[str, Any], *, source: str
) -> None:
    applied: list[str] = []
    for k, v in overrides.items():
        if k not in HYPERPARAM_OVERRIDABLE_KEYS:
            continue
        if _arg_was_explicit(argv, k):
            continue
        old_v = getattr(args, k)
        if old_v != v:
            setattr(args, k, v)
            applied.append(f"{k}={v!r}")
    if applied:
        print(f"Applied hyperparameters from {source}: " + ", ".join(applied), flush=True)


def _load_hyperparams_config(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in hyperparameters file {p}: {e}") from e
    if not isinstance(raw, dict):
        raise ValueError(f"Hyperparameters file must be a JSON object: {p}")
    return raw


def apply_hyperparams_from_file(args: argparse.Namespace, argv: list[str]) -> None:
    cfg = _load_hyperparams_config(args.hyperparams_file)
    if not cfg:
        return
    model_key = str(args.vision_backbone)
    dataset_key = str(args.dataset)
    defaults = cfg.get("defaults", {})
    models = cfg.get("models", {})
    datasets = cfg.get("datasets", {})
    model_dataset = cfg.get("model_dataset", {})
    if isinstance(defaults, dict):
        _apply_hparam_overrides(args, argv, defaults, source="defaults")
    if isinstance(models, dict) and isinstance(models.get(model_key), dict):
        _apply_hparam_overrides(args, argv, models[model_key], source=f"models.{model_key}")
    if isinstance(datasets, dict) and isinstance(datasets.get(dataset_key), dict):
        _apply_hparam_overrides(
            args, argv, datasets[dataset_key], source=f"datasets.{dataset_key}"
        )
    if isinstance(model_dataset, dict):
        by_model = model_dataset.get(model_key, {})
        if isinstance(by_model, dict) and isinstance(by_model.get(dataset_key), dict):
            _apply_hparam_overrides(
                args,
                argv,
                by_model[dataset_key],
                source=f"model_dataset.{model_key}.{dataset_key}",
            )


def _resolve_train_device(args: argparse.Namespace) -> torch.device:
    if args.cpu or args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("You passed --device cuda, but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _hf_image_to_pil_rgb(img: Any) -> Any:
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
    raise TypeError(f"Unsupported image type: {type(img).__name__}")


class HfPilImageDataset(Dataset):
    def __init__(self, hf_subset: Any, image_key: str, label_key: str) -> None:
        self.hf = hf_subset
        self.image_key = image_key
        self.label_key = label_key
        self._phrase_keys = ("pos", "neg_0", "neg_1", "neg_2", "neg_3")
        self._has_phrase_columns = all(k in self.hf.column_names for k in self._phrase_keys)

    def __len__(self) -> int:
        return len(self.hf)

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
                out[k] = "" if row[k] is None else str(row[k])
        return out


def make_collate_fn(processor: Any) -> Any:
    call_sig = inspect.signature(processor.__call__)
    expects_videos = ("videos" in call_sig.parameters) and ("images" not in call_sig.parameters)

    def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = torch.tensor([int(d["label"]) for d in batch], dtype=torch.long)
        images = [d["image"] for d in batch]
        if expects_videos:
            vis = processor(videos=[[im] for im in images], return_tensors="pt")
        else:
            vis = processor(images=images, return_tensors="pt")
        out: dict[str, Any] = {
            "pixel_values": _extract_model_pixel_values(vis),
            "labels": labels,
        }
        if "pair_seen_in_train" in batch[0]:
            out["pair_seen_in_train"] = torch.tensor(
                [bool(d["pair_seen_in_train"]) for d in batch], dtype=torch.bool
            )
        return out

    return _collate


@dataclass(frozen=True)
class CspVocabMeta:
    attrs: list[str]
    objs: list[str]
    pairs: list[str]
    pair_attr_idx: torch.Tensor
    pair_obj_idx: torch.Tensor
    pair_separator: str


def _split_pair_name(name: str, pair_separator: str) -> tuple[str, str]:
    s = (name or "").strip()
    if not s:
        return "__none__", "unknown"
    if pair_separator and pair_separator in s:
        left, right = s.split(pair_separator, 1)
        return (left.strip() or "__none__", right.strip() or "unknown")
    toks = s.split()
    if len(toks) <= 1:
        return "__none__", toks[0]
    return " ".join(toks[:-1]), toks[-1]


def build_csp_vocab_meta(class_names: list[str], *, pair_separator: str = " ") -> CspVocabMeta:
    if not class_names:
        raise ValueError("class_names is empty; cannot build CSP metadata")
    attrs_for_pairs: list[str] = []
    objs_for_pairs: list[str] = []
    for cname in class_names:
        a, o = _split_pair_name(str(cname), pair_separator)
        attrs_for_pairs.append(a)
        objs_for_pairs.append(o)
    attrs = sorted(set(attrs_for_pairs))
    objs = sorted(set(objs_for_pairs))
    attr_to_idx = {a: i for i, a in enumerate(attrs)}
    obj_to_idx = {o: i for i, o in enumerate(objs)}
    pair_attr_idx = torch.tensor([attr_to_idx[a] for a in attrs_for_pairs], dtype=torch.long)
    pair_obj_idx = torch.tensor([obj_to_idx[o] for o in objs_for_pairs], dtype=torch.long)
    return CspVocabMeta(
        attrs=attrs,
        objs=objs,
        pairs=list(class_names),
        pair_attr_idx=pair_attr_idx,
        pair_obj_idx=pair_obj_idx,
        pair_separator=pair_separator,
    )


def _clip_text_embeds_from_inputs_embeds(
    clip_text_with_proj: nn.Module,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Run CLIP text tower on pre-built ``inputs_embeds`` (BOS + soft prompts + EOS).

    Recent ``transformers`` builds reject ``CLIPTextModelWithProjection(..., inputs_embeds=)``
    because the inner ``CLIPTextModel`` raises when ``input_ids`` is omitted. This path
    mirrors ``CLIPTextModel.forward`` but starts from ``CLIPTextEmbeddings(inputs_embeds=...)``.
    The composed sequence ends with the EOS patch, so we pool the **last** token (same as
    one-token EOS at the end of a short prompt).
    """
    tm = clip_text_with_proj.text_model
    hidden = tm.embeddings(inputs_embeds=inputs_embeds)
    try:
        from transformers.masking_utils import create_causal_mask
    except ImportError as exc:
        raise RuntimeError(
            "CSP compose() needs ``transformers.masking_utils.create_causal_mask`` "
            "(upgrade transformers) when the CLIP wrapper no longer accepts inputs_embeds-only "
            f"calls: {exc}"
        ) from exc
    causal_attention_mask = create_causal_mask(
        config=tm.config,
        inputs_embeds=hidden,
        attention_mask=attention_mask,
        past_key_values=None,
    )
    encoder_outputs = tm.encoder(
        inputs_embeds=hidden,
        attention_mask=causal_attention_mask,
        is_causal=True,
    )
    last_hidden_state = tm.final_layer_norm(encoder_outputs.last_hidden_state)
    pooled_output = last_hidden_state[:, -1, :]
    return clip_text_with_proj.text_projection(pooled_output)


class CspCompositionVocab(nn.Module):
    def __init__(
        self,
        *,
        num_attrs: int,
        num_objs: int,
        text_encoder: nn.Module,
        adapter: nn.Module,
        tokenizer: Any,
        text_hidden_dim: int,
        cond_dim: int,
        context_length: int = 8,
        attr_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if context_length < 2:
            raise ValueError(f"context_length must be >= 2, got {context_length}")
        self.num_attrs = int(num_attrs)
        self.num_objs = int(num_objs)
        self.text_hidden_dim = int(text_hidden_dim)
        self.cond_dim = int(cond_dim)
        self.context_length = int(context_length)
        self.attr_ctx_tokens = max(1, self.context_length // 2)
        self.obj_ctx_tokens = max(1, self.context_length - self.attr_ctx_tokens)
        self.attr_dropout = float(attr_dropout)
        self.text_encoder = text_encoder
        self.adapter = adapter
        self.attr_prompt = nn.Parameter(
            torch.empty(self.num_attrs, self.attr_ctx_tokens, self.text_hidden_dim)
        )
        self.obj_prompt = nn.Parameter(
            torch.empty(self.num_objs, self.obj_ctx_tokens, self.text_hidden_dim)
        )
        self.bos_token_id = int(
            tokenizer.bos_token_id
            if tokenizer.bos_token_id is not None
            else tokenizer.cls_token_id
        )
        self.eos_token_id = int(
            tokenizer.eos_token_id
            if tokenizer.eos_token_id is not None
            else (
                tokenizer.sep_token_id
                if tokenizer.sep_token_id is not None
                else self.bos_token_id
            )
        )
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.attr_prompt, std=0.02)
        nn.init.normal_(self.obj_prompt, std=0.02)

    def _compose_raw(self, attr_idx: torch.Tensor, obj_idx: torch.Tensor) -> torch.Tensor:
        a = self.attr_prompt.index_select(0, attr_idx)
        o = self.obj_prompt.index_select(0, obj_idx)
        if self.training and self.attr_dropout > 0.0:
            keep = (torch.rand(a.shape[0], 1, 1, device=a.device) >= self.attr_dropout).to(a.dtype)
            a = a * keep
        return a, o

    def compose(self, attr_idx: torch.Tensor, obj_idx: torch.Tensor) -> torch.Tensor:
        a, o = self._compose_raw(attr_idx, obj_idx)
        bsz = a.shape[0]
        token_embed = self.text_encoder.get_input_embeddings().weight
        bos = token_embed[self.bos_token_id].to(a.device).view(1, 1, -1).expand(bsz, -1, -1)
        eos = token_embed[self.eos_token_id].to(a.device).view(1, 1, -1).expand(bsz, -1, -1)
        inputs_embeds = torch.cat([bos, a, o, eos], dim=1)
        attention_mask = torch.ones(
            (bsz, inputs_embeds.shape[1]), dtype=torch.long, device=inputs_embeds.device
        )
        try:
            out = self.text_encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
            text_embeds = out.text_embeds
        except ValueError as err:
            # transformers >= ~4.48: CLIPTextModel requires input_ids; inputs_embeds-only fails.
            if "input_ids" not in str(err).lower():
                raise
            text_embeds = _clip_text_embeds_from_inputs_embeds(
                self.text_encoder, inputs_embeds, attention_mask
            )
        if text_embeds is None:
            raise RuntimeError("CLIPTextModelWithProjection did not return text_embeds")
        cond = self.adapter(text_embeds)
        return nn.functional.normalize(cond, dim=-1)

    def compose_all_pairs(
        self,
        meta: CspVocabMeta,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        dev = self.attr_prompt.device if device is None else torch.device(device)
        return self.compose(meta.pair_attr_idx.to(dev), meta.pair_obj_idx.to(dev))

    def compose_pair_indices(
        self,
        meta: CspVocabMeta,
        pair_row_indices: torch.Tensor,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """
        Embed only selected global pair rows (indices into ``meta.pairs``, shape (K,)).
        """
        dev = self.attr_prompt.device if device is None else torch.device(device)
        idx = pair_row_indices.detach().cpu().long().view(-1)
        attr = meta.pair_attr_idx.index_select(0, idx).to(dev)
        obj = meta.pair_obj_idx.index_select(0, idx).to(dev)
        return self.compose(attr, obj)

    @torch.inference_mode()
    def init_from_label_text(self, meta: CspVocabMeta, tokenizer: Any) -> None:
        token_embed = self.text_encoder.get_input_embeddings().weight.detach().float().cpu()

        def _avg_token_embedding(text: str) -> torch.Tensor:
            s = (text or "").strip()
            if not s or s == "__none__":
                return torch.zeros(self.text_hidden_dim, dtype=torch.float32)
            ids = tokenizer(
                s, add_special_tokens=False, return_attention_mask=False
            )["input_ids"]
            if not ids:
                return torch.zeros(self.text_hidden_dim, dtype=torch.float32)
            idx = torch.tensor(ids, dtype=torch.long)
            return token_embed.index_select(0, idx).mean(dim=0)

        for i, a in enumerate(meta.attrs):
            v = _avg_token_embedding(a)
            self.attr_prompt[i].copy_(v.unsqueeze(0).repeat(self.attr_ctx_tokens, 1).to(self.attr_prompt.device))
        for i, o in enumerate(meta.objs):
            v = _avg_token_embedding(o)
            self.obj_prompt[i].copy_(v.unsqueeze(0).repeat(self.obj_ctx_tokens, 1).to(self.obj_prompt.device))


@torch.inference_mode()
def _build_candidate_text_bank(
    model: TextConditionedVisionModel,
    tokenizer: Any,
    class_names: list[str],
    text_template: str,
    device: torch.device,
    *,
    chunk_size: int,
) -> torch.Tensor:
    prompts = [text_template.format(c=c) for c in class_names]
    chunks: list[torch.Tensor] = []
    for s in range(0, len(prompts), chunk_size):
        enc = tokenizer(prompts[s : s + chunk_size], padding=True, return_tensors="pt", truncation=True)
        t = model.encode_text(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        chunks.append(t.detach())
    return torch.cat(chunks, dim=0)


def _make_csp_eval_forward(
    model: TextConditionedVisionModel,
    csp_vocab: CspCompositionVocab,
    csp_meta: CspVocabMeta,
    device: torch.device,
    *,
    use_amp: bool,
    allowed_class_indices: list[int] | None = None,
) -> Callable[[dict[str, Any]], tuple[torch.Tensor, torch.Tensor | None]]:
    """Per-batch forward for :func:`eval_clip_style_classification` (composed text bank + contrastive loss)."""

    c_full = len(csp_meta.pairs)
    allowed_t: torch.Tensor | None
    g2l: torch.Tensor | None
    if allowed_class_indices is None:
        allowed_t = None
        g2l = None
    else:
        allowed_unique = sorted({int(i) for i in allowed_class_indices})
        if any(i < 0 or i >= c_full for i in allowed_unique):
            raise ValueError(
                f"allowed_class_indices out of range [0,{c_full}); got "
                f"{min(allowed_unique)}/{max(allowed_unique)}"
            )
        allowed_t = torch.tensor(allowed_unique, dtype=torch.long, device=device)
        g2l = torch.full((c_full,), -1, dtype=torch.long, device=device)
        g2l[allowed_t] = torch.arange(allowed_t.numel(), device=device, dtype=torch.long)

    def forward_batch(
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pv = batch["pixel_values"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda" and use_amp),
            dtype=torch.float16,
        ):
            candidate_text_bank = csp_vocab.compose_all_pairs(csp_meta, device=device)
            z = model.encode_image(pv)
            if allowed_t is None:
                logits = model.score_candidates(z, candidate_text_bank)
                pos_text = candidate_text_bank.index_select(0, y)
            else:
                assert g2l is not None
                y_loc = g2l[y]
                if not (y_loc >= 0).all():
                    raise RuntimeError(
                        "Label id not in allowed_class_indices for this eval split "
                        "(batch contains a class outside train∪val or train∪test)."
                    )
                bank_sub = candidate_text_bank.index_select(0, allowed_t)
                logits_sub = model.score_candidates(z, bank_sub)
                finfo_min = torch.finfo(logits_sub.dtype).min
                mask_val = finfo_min / 2 if finfo_min > -3.4e38 else -3.4e38
                logits = torch.full(
                    (z.size(0), c_full),
                    mask_val,
                    device=z.device,
                    dtype=logits_sub.dtype,
                )
                logits[:, allowed_t] = logits_sub
                pos_text = bank_sub.index_select(0, y_loc)
            loss = clip_contrastive_loss(model.score_candidates(z, pos_text))
        return logits, loss

    return forward_batch


_CSP_EVAL_CLI_RESTORE_KEYS: frozenset[str] = frozenset(
    {
        "eval_only",
        "checkpoint",
        "eval_split",
        "metrics_json",
        "experiment_tag",
        "max_eval_samples",
        "seed",
        "device",
        "cpu",
        "no_wandb",
        "wandb_project",
        "wandb_entity",
        "wandb_run_name",
        "wandb_log_images",
        "wandb_image_log_interval",
        "wandb_max_images",
        "hyperparams_file",
        "batch_size",
        "num_workers",
        "amp",
        "log_interval",
    }
)


def _meta_from_bundle(meta_d: dict[str, Any]) -> CspVocabMeta:
    return CspVocabMeta(
        attrs=list(meta_d["attrs"]),
        objs=list(meta_d["objs"]),
        pairs=list(meta_d["pairs"]),
        pair_attr_idx=meta_d["pair_attr_idx"],
        pair_obj_idx=meta_d["pair_obj_idx"],
        pair_separator=str(meta_d.get("pair_separator", " ")),
    )


def run_csp_eval_only(args: argparse.Namespace) -> None:
    """Load a saved CSP vocab artifact and run clip-style metrics on val or test."""
    cli_snap = argparse.Namespace(**vars(args))
    artifact_path = (args.checkpoint or "").strip()
    if not artifact_path:
        raise ValueError("CSP --eval-only requires --checkpoint (path to the saved .pt artifact).")

    bundle = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict) or "csp_vocab" not in bundle or "meta" not in bundle:
        raise TypeError(
            f"--checkpoint must be a CSP vocab bundle dict with 'csp_vocab' and 'meta'; got {type(bundle).__name__}"
        )
    saved_args = bundle.get("args") or {}
    if not isinstance(saved_args, dict):
        raise TypeError(f"bundle['args'] must be a dict, got {type(saved_args).__name__}")

    for k, v in saved_args.items():
        if hasattr(args, k):
            setattr(args, k, v)
    for k in _CSP_EVAL_CLI_RESTORE_KEYS:
        if hasattr(args, k) and hasattr(cli_snap, k):
            setattr(args, k, getattr(cli_snap, k))
    args.checkpoint = artifact_path

    set_seed(int(args.seed))
    args.ijepa = resolve_vision_model_id(args.vision_backbone, args.ijepa)
    device = _resolve_train_device(args)
    if device.type == "cuda":
        print(f"Eval on GPU: {torch.cuda.get_device_name(device.index or 0)}", flush=True)
    else:
        print("Eval on CPU.", flush=True)

    csp_meta = _meta_from_bundle(bundle["meta"])
    me = _msamples(int(args.max_eval_samples))
    tvt = load_vision_train_val_test_specs(
        args.dataset,
        val_fraction=float(args.val_fraction),
        split_seed=int(args.split_seed),
        max_train_samples=None,
        max_val_samples=me if args.eval_split == "val" else None,
        max_test_samples=me if args.eval_split == "test" else None,
    )
    if list(tvt.train.class_names) != list(csp_meta.pairs):
        raise ValueError(
            "Training class order in the checkpoint meta does not match the current dataset "
            f"(n_checkpoint={len(csp_meta.pairs)} vs n_dataset={len(tvt.train.class_names)}). "
            "Use the same --dataset and split settings as training."
        )

    if args.eval_split == "val":
        spec = tvt.val
    else:
        spec = tvt.test
    eval_ds = HfPilImageDataset(spec.dataset, spec.image_column, spec.label_key)
    if len(eval_ds) == 0:
        raise RuntimeError("Eval split is empty. Check --max-eval-samples and data config.")

    n_classes = len(tvt.train.class_names)
    im_proc = load_vision_processor(args.ijepa)
    collate = make_collate_fn(im_proc)
    loader = DataLoader(
        eval_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )

    finetune_v = bool(getattr(args, "finetune_vision_backbone", False))
    model = TextConditionedVisionModel(
        num_labels=n_classes,
        ijepa_id=args.ijepa,
        clip_id=args.clip,
        cond_dim=int(args.cond_dim),
        fusion_hidden=int(args.fusion_hidden),
        fusion_type=str(args.fusion_type),
        freeze_text_encoder=True,
        freeze_vision_backbone=not finetune_v,
    ).to(device)
    _load_base_checkpoint_if_any(model, args.base_checkpoint)
    model.eval()
    model.backbone.eval()
    for p in model.parameters():
        p.requires_grad = False
    ad = bundle.get("adapter")
    fu = bundle.get("fusion")
    if isinstance(ad, dict):
        model.text_cond.adapter.load_state_dict(ad, strict=True)
    if isinstance(fu, dict):
        model.fusion.load_state_dict(fu, strict=True)
    bb = bundle.get("backbone")
    if isinstance(bb, dict) and bb:
        model.backbone.load_state_dict(bb, strict=True)
        print("Loaded vision backbone weights from CSP bundle.", flush=True)
    elif finetune_v:
        print(
            "Warning: training used finetune_vision_backbone but this bundle has no 'backbone' tensor dict; "
            "using Hugging Face pretrained backbone (metrics will not match that training run). "
            "Re-save with a current text_cond_train.py after finetune-vision-backbone training.",
            flush=True,
        )

    tok = AutoTokenizer.from_pretrained(args.clip)
    csp_vocab = CspCompositionVocab(
        num_attrs=len(csp_meta.attrs),
        num_objs=len(csp_meta.objs),
        text_encoder=model.text_cond.text_encoder,
        adapter=model.text_cond.adapter,
        tokenizer=tok,
        text_hidden_dim=int(model.text_cond.text_encoder.config.hidden_size),
        cond_dim=int(args.cond_dim),
        context_length=int(args.csp_context_length),
        attr_dropout=float(args.csp_attr_dropout),
    ).to(device)
    csp_vocab.load_state_dict(bundle["csp_vocab"], strict=True)
    csp_vocab.eval()

    use_amp_eval = device.type == "cuda" and bool(args.amp)
    eval_allowed = csp_vocab_allowed_class_indices(tvt, args.eval_split)
    eval_allowed_arg: list[int] | None = eval_allowed if len(eval_allowed) < n_classes else None
    if eval_allowed_arg is not None:
        split_tag = "train∪val" if args.eval_split == "val" else "train∪test"
        print(
            f"Eval: candidate softmax uses {len(eval_allowed)}/{n_classes} classes ({split_tag} label union).",
            flush=True,
        )
    (
        loss,
        top1,
        top5,
        auc_csp,
        seen_top1,
        seen_top5,
        unseen_top1,
        unseen_top5,
    ) = eval_clip_style_classification(
        loader,
        device,
        num_classes=n_classes,
        use_amp=use_amp_eval,
        forward_batch=_make_csp_eval_forward(
            model,
            csp_vocab,
            csp_meta,
            device,
            use_amp=use_amp_eval,
            allowed_class_indices=eval_allowed_arg,
        ),
        modules_to_eval=(model, csp_vocab),
    )
    split_name = f"{args.eval_split} (n={len(eval_ds)})"
    print(
        f"\n{split_name}\n  loss: {loss:.4f}\n  top-1: {top1*100:.2f}%\n  top-5: {top5*100:.2f}%\n  "
        f"seen_top-1: {seen_top1*100:.2f}%\n  seen_top-5: {seen_top5*100:.2f}%\n  "
        f"unseen_top-1: {unseen_top1*100:.2f}%\n  unseen_top-5: {unseen_top5*100:.2f}%\n  "
        f"auc_csp_style: {auc_csp:.4f}\n",
        flush=True,
    )

    metrics_json = (args.metrics_json or "").strip()
    if metrics_json:
        p = Path(metrics_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_type": "csp_vocab_eval",
            "dataset": args.dataset,
            "split": args.eval_split,
            "seed": int(args.seed),
            "fusion_type": str(getattr(model.fusion, "fusion_type", args.fusion_type)),
            "experiment_tag": (args.experiment_tag or "").strip(),
            "checkpoint": artifact_path,
            "from_hub": "",
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "loss": float(loss),
            "top1": float(top1),
            "top5": float(top5),
            "seen_top1": float(seen_top1),
            "seen_top5": float(seen_top5),
            "unseen_top1": float(unseen_top1),
            "unseen_top5": float(unseen_top5),
            "auc_csp_style": float(auc_csp),
            "batch_size": int(args.batch_size),
            "timestamp_unix": int(time.time()),
        }
        p.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote eval metrics JSON to {p}", flush=True)

    if not args.no_wandb:
        os.environ["WANDB_LOG_MODEL"] = "false"
        import wandb

        w_key = (os.environ.get("WANDB_API_KEY") or "").strip()
        if w_key:
            wandb.login(key=w_key, relogin=True)
        run_name = _resolve_wandb_run_name(
            (args.wandb_run_name or "").strip(),
            f"csp-vocab-eval-{args.dataset}-{args.eval_split}-seed{args.seed}",
        )
        w = wandb.init(
            project=args.wandb_project,
            entity=(args.wandb_entity or "").strip() or None,
            name=run_name,
            config={
                "mode": "csp_vocab_eval_only",
                "dataset": args.dataset,
                "eval_split": args.eval_split,
                "fusion_type": str(getattr(model.fusion, "fusion_type", args.fusion_type)),
                "seed": int(args.seed),
                "checkpoint": artifact_path,
            },
        )
        wandb.log(
            {
                f"{args.eval_split}/loss": float(loss),
                f"{args.eval_split}/acc@1": float(top1),
                f"{args.eval_split}/acc@5": float(top5),
                f"{args.eval_split}/seen_acc@1": float(seen_top1),
                f"{args.eval_split}/seen_acc@5": float(seen_top5),
                f"{args.eval_split}/unseen_acc@1": float(unseen_top1),
                f"{args.eval_split}/unseen_acc@5": float(unseen_top5),
                f"{args.eval_split}/auc_csp_style": float(auc_csp),
            },
            step=0,
        )
        wandb.finish()
        del w


def _load_base_checkpoint_if_any(model: TextConditionedVisionModel, checkpoint: str) -> None:
    ckpt = (checkpoint or "").strip()
    if not ckpt:
        return
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint should be a state_dict mapping, got {type(state).__name__}")
    res = model.load_state_dict(state, strict=False)
    if res.unexpected_keys:
        raise RuntimeError(f"Unexpected keys in checkpoint: {res.unexpected_keys[:12]}")
    print(
        "Loaded base checkpoint with strict=False "
        f"(missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)})",
        flush=True,
    )


def run_post_training(args: argparse.Namespace) -> None:
    set_seed(int(args.seed))
    args.ijepa = resolve_vision_model_id(args.vision_backbone, args.ijepa)
    device = _resolve_train_device(args)
    print(f"Post-training device: {device}", flush=True)
    tvt = load_vision_train_val_test_specs(
        args.dataset,
        val_fraction=float(args.val_fraction),
        split_seed=int(args.split_seed),
        max_train_samples=_msamples(int(args.max_train_samples)),
        max_val_samples=_msamples(int(args.max_val_samples)),
        max_test_samples=None,
    )
    n_classes = len(tvt.train.class_names)
    train_ds = HfPilImageDataset(tvt.train.dataset, tvt.train.image_column, tvt.train.label_key)
    val_ds = HfPilImageDataset(tvt.val.dataset, tvt.val.image_column, tvt.val.label_key)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Train or val split is empty.")

    im_proc = load_vision_processor(args.ijepa)
    collate = make_collate_fn(im_proc)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
        drop_last=(len(train_ds) >= 2 * int(args.batch_size)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )

    val_allowed = csp_vocab_allowed_class_indices(tvt, "val")
    val_eval_allowed: list[int] | None = val_allowed if len(val_allowed) < n_classes else None
    train_allowed = csp_vocab_allowed_class_indices(tvt, "train")
    if len(train_allowed) < n_classes:
        print(
            f"Train: composed pair bank uses {len(train_allowed)}/{n_classes} classes "
            "(labels that appear in train rows only).",
            flush=True,
        )
    if val_eval_allowed is not None:
        print(
            f"Val eval: candidate softmax restricted to {len(val_allowed)}/{n_classes} classes "
            "(train ∪ val label union).",
            flush=True,
        )

    model = TextConditionedVisionModel(
        num_labels=n_classes,
        ijepa_id=args.ijepa,
        clip_id=args.clip,
        cond_dim=int(args.cond_dim),
        fusion_hidden=int(args.fusion_hidden),
        fusion_type=str(args.fusion_type),
        freeze_text_encoder=True,
    ).to(device)
    _load_base_checkpoint_if_any(model, args.base_checkpoint)
    model.eval()
    model.backbone.eval()
    for p in model.parameters():
        p.requires_grad = False

    tok = AutoTokenizer.from_pretrained(args.clip)
    csp_meta = build_csp_vocab_meta(
        tvt.train.class_names,
        pair_separator=str(args.csp_pair_separator),
    )
    csp_vocab = CspCompositionVocab(
        num_attrs=len(csp_meta.attrs),
        num_objs=len(csp_meta.objs),
        text_encoder=model.text_cond.text_encoder,
        adapter=model.text_cond.adapter,
        tokenizer=tok,
        text_hidden_dim=int(model.text_cond.text_encoder.config.hidden_size),
        cond_dim=int(args.cond_dim),
        context_length=int(args.csp_context_length),
        attr_dropout=float(args.csp_attr_dropout),
    ).to(device)
    # Freeze everything in CSP module first, then unfreeze only soft prompt vocab tables.
    for p in csp_vocab.parameters():
        p.requires_grad = False
    csp_vocab.attr_prompt.requires_grad = True
    csp_vocab.obj_prompt.requires_grad = True
    if args.csp_vocab_init == "text":
        csp_vocab.init_from_label_text(csp_meta, tok)

    params = [csp_vocab.attr_prompt, csp_vocab.obj_prompt]
    opt = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    total_steps = max(int(args.epochs) * max(len(train_loader), 1), 1)
    warmup_steps = min(max(int(round(total_steps * float(args.warmup_ratio))), 0), total_steps)
    if args.scheduler_type == "none":
        sched: torch.optim.lr_scheduler.LRScheduler | None = None
    elif args.scheduler_type == "cosine":
        min_lr_ratio = float(args.min_lr_ratio)
        if not (0.0 <= min_lr_ratio <= 1.0):
            raise ValueError("--min-lr-ratio must be in [0,1]")

        def _lr_factor(step: int) -> float:
            s = int(step)
            if warmup_steps > 0 and s < warmup_steps:
                return float(s + 1) / float(warmup_steps)
            denom = max(total_steps - warmup_steps, 1)
            t = min(max((s - warmup_steps) / denom, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * t))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_factor)
    else:
        raise ValueError(f"Unsupported scheduler_type={args.scheduler_type!r}")

    use_cuda_amp = (device.type == "cuda") and bool(args.amp)
    grad_scaler = torch.amp.GradScaler("cuda") if use_cuda_amp else None
    global_step = 0
    t0 = time.time()

    use_wandb = not bool(args.no_wandb)
    w: Any = None
    if use_wandb:
        os.environ["WANDB_LOG_MODEL"] = "false"
        import wandb

        w_key = (os.environ.get("WANDB_API_KEY") or "").strip()
        if w_key:
            wandb.login(key=w_key, relogin=True)
        run_name = _resolve_wandb_run_name(
            (args.wandb_run_name or "").strip(),
            _default_csp_vocab_wandb_run_name(args),
        )
        wandb_config: dict[str, Any] = {
            "run_type": "csp_vocab_posttrain",
            "dataset": args.dataset,
            "vision_backbone": args.vision_backbone,
            "ijepa_id": args.ijepa,
            "clip_id": args.clip,
            "fusion_type": args.fusion_type,
            "cond_dim": int(args.cond_dim),
            "fusion_hidden": int(args.fusion_hidden),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "scheduler_type": args.scheduler_type,
            "warmup_ratio": float(args.warmup_ratio),
            "min_lr_ratio": float(args.min_lr_ratio),
            "max_grad_norm": float(args.max_grad_norm),
            "seed": int(args.seed),
            "amp": bool(args.amp),
            "num_workers": int(args.num_workers),
            "csp_context_length": int(args.csp_context_length),
            "csp_vocab_init": args.csp_vocab_init,
            "csp_attr_dropout": float(args.csp_attr_dropout),
            "base_checkpoint": (args.base_checkpoint or "").strip(),
            "save": (args.save or "").strip(),
            "device": str(device),
        }
        w = wandb.init(
            project=args.wandb_project,
            entity=(args.wandb_entity or "").strip() or None,
            name=run_name,
            config=wandb_config,
        )

    train_allowed_t = torch.tensor(train_allowed, dtype=torch.long)
    g2l_train = torch.full((n_classes,), -1, dtype=torch.long)
    g2l_train[train_allowed_t] = torch.arange(train_allowed_t.numel())
    g2l_train = g2l_train.to(device)
    train_allowed_dev = train_allowed_t.to(device)

    try:
        for epoch in range(int(args.epochs)):
            csp_vocab.train()
            # ``text_encoder`` / ``adapter`` are registered submodules of ``csp_vocab``; a bare
            # ``.train()`` would flip them out of eval (Dropout etc.) even though only soft prompts
            # get gradients. They stay frozen; keep them in eval like ``model``.
            csp_vocab.text_encoder.eval()
            csp_vocab.adapter.eval()
            pbar = tqdm(
                train_loader,
                desc=f"csp post-train ep {epoch + 1}/{args.epochs}",
                file=sys.stdout,
            )
            for batch_idx, batch in enumerate(pbar):
                pv = batch["pixel_values"].to(device, non_blocking=True)
                y = batch["labels"].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=use_cuda_amp,
                    dtype=torch.float16,
                ):
                    candidate_text_bank = csp_vocab.compose_pair_indices(
                        csp_meta, train_allowed_t, device=device
                    )
                    z = model.encode_image(pv)
                    y_loc = g2l_train[y]
                    if not (y_loc >= 0).all():
                        raise RuntimeError(
                            "Train batch contains a label outside csp_vocab_allowed_class_indices "
                            "(train-only pool); check data vs allowed set."
                        )
                    pos_text = candidate_text_bank.index_select(0, y_loc)
                    pair_scores = model.score_candidates(z, pos_text)
                    loss = clip_contrastive_loss(pair_scores)
                if grad_scaler is not None:
                    grad_scaler.scale(loss).backward()
                else:
                    loss.backward()
                if float(args.max_grad_norm) > 0:
                    if grad_scaler is not None:
                        grad_scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, float(args.max_grad_norm))
                if grad_scaler is not None:
                    grad_scaler.step(opt)
                    grad_scaler.update()
                else:
                    opt.step()
                if sched is not None:
                    sched.step()

                log_this_batch = (batch_idx % int(args.log_interval)) == 0
                img_this_batch = bool(
                    use_wandb
                    and w is not None
                    and args.wandb_log_images
                    and int(args.wandb_image_log_interval) > 0
                    and ((epoch + 1) % int(args.wandb_image_log_interval) == 0)
                    and batch_idx == 0
                )
                if log_this_batch or img_this_batch:
                    with torch.inference_mode():
                        full_logits = model.score_candidates(z, candidate_text_bank)
                        pred_local = full_logits.argmax(-1)
                        pred = train_allowed_dev.index_select(0, pred_local)
                        batch_acc = float((pred == y).float().mean().item())
                else:
                    pred = None
                    batch_acc = 0.0

                if log_this_batch:
                    pbar.set_postfix(loss=f"{float(loss.detach()):.3f}", acc=f"{batch_acc:.3f}")

                if use_wandb and w is not None:
                    import wandb

                    log_d: dict[str, Any] = {}
                    if log_this_batch:
                        log_d.update(
                            {
                                "train/loss": float(loss.item()),
                                "train/batch_acc": batch_acc,
                                "train/lr": opt.param_groups[0]["lr"],
                                "time_elapsed_s": time.time() - t0,
                            }
                        )
                    if img_this_batch and pred is not None:
                        log_d["train/images"] = _make_wandb_images(
                            pv,
                            y,
                            pred,
                            max_images=int(args.wandb_max_images),
                            class_names=tvt.train.class_names,
                        )
                    if log_d:
                        wandb.log(log_d, step=global_step)

                global_step += 1

            (
                val_loss,
                val_top1,
                val_top5,
                val_auc_csp,
                val_seen_top1,
                val_seen_top5,
                val_unseen_top1,
                val_unseen_top5,
            ) = eval_clip_style_classification(
                val_loader,
                device,
                num_classes=n_classes,
                use_amp=bool(args.amp),
                forward_batch=_make_csp_eval_forward(
                    model,
                    csp_vocab,
                    csp_meta,
                    device,
                    use_amp=bool(args.amp),
                    allowed_class_indices=val_eval_allowed,
                ),
                modules_to_eval=(model, csp_vocab),
            )
            print(
                f"Epoch {epoch + 1}/{args.epochs}  val_loss={val_loss:.4f}  "
                f"val_top1={val_top1 * 100:.2f}%  val_top5={val_top5 * 100:.2f}%  "
                f"seen_top1={val_seen_top1 * 100:.2f}%  seen_top5={val_seen_top5 * 100:.2f}%  "
                f"unseen_top1={val_unseen_top1 * 100:.2f}%  unseen_top5={val_unseen_top5 * 100:.2f}%  "
                f"auc_csp_style={val_auc_csp:.4f}  steps={global_step}",
                flush=True,
            )

            if use_wandb and w is not None:
                import wandb

                wandb.log(
                    {
                        "epoch": epoch,
                        "val/loss": val_loss,
                        "val/acc@1": val_top1,
                        "val/acc@5": val_top5,
                        "val/auc_csp_style": val_auc_csp,
                        "val/seen_acc@1": val_seen_top1,
                        "val/seen_acc@5": val_seen_top5,
                        "val/unseen_acc@1": val_unseen_top1,
                        "val/unseen_acc@5": val_unseen_top5,
                    },
                    step=global_step,
                )

        if args.save and args.save.strip():
            out = {
                "csp_vocab": csp_vocab.state_dict(),
                "adapter": model.text_cond.adapter.state_dict(),
                "fusion": model.fusion.state_dict(),
                "meta": {
                    "attrs": csp_meta.attrs,
                    "objs": csp_meta.objs,
                    "pairs": csp_meta.pairs,
                    "pair_attr_idx": csp_meta.pair_attr_idx.cpu(),
                    "pair_obj_idx": csp_meta.pair_obj_idx.cpu(),
                    "pair_separator": csp_meta.pair_separator,
                },
                "args": vars(args),
            }
            torch.save(out, args.save)
            print(f"Saved CSP post-training artifact to {args.save}", flush=True)
            if use_wandb and w is not None:
                import wandb

                wandb.log({"artifact/save_path": str(args.save)}, step=global_step)
    finally:
        if use_wandb and w is not None:
            import wandb

            wandb.finish()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standalone CSP vocab post-training (freeze base model, train compositional vocab).",
        epilog="Example: uv run python csp_vocab_train.py --dataset cspref_mit_states --base-checkpoint runs/base.pt  # add --no-wandb to skip W&B",
    )
    p.add_argument("--dataset", default="cspref_mit_states", choices=list_vision_dataset_keys())
    p.add_argument(
        "--vision-backbone",
        default="dinov3",
        choices=tuple(sorted(VISION_BACKBONE_PRESETS.keys())),
        help="Preset HF vision model when --ijepa is empty (must match the base checkpoint).",
    )
    p.add_argument(
        "--ijepa",
        default="",
        help="Explicit HuggingFace vision backbone id. Empty (default) uses --vision-backbone preset.",
    )
    p.add_argument("--clip", default=DEFAULT_CLIP_TEXT_ID)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=DEFAULT_TRAIN_BATCH)
    p.add_argument("--num-workers", type=int, default=DEFAULT_TRAIN_NUM_WORKERS)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--scheduler-type", choices=("none", "cosine"), default="none")
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--max-grad-norm", type=float, default=0.0)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--cond-dim", type=int, default=256)
    p.add_argument("--fusion-hidden", type=int, default=512)
    p.add_argument(
        "--fusion-type",
        choices=("cross_attention", "clip_similarity"),
        default="clip_similarity",
    )
    p.add_argument("--text-template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    p.add_argument("--text-bank-chunk-size", type=int, default=512)
    p.add_argument(
        "--csp-context-length",
        type=int,
        default=8,
        help="Total number of learnable soft prompt tokens per composed pair (split across attr/object).",
    )
    p.add_argument("--csp-vocab-init", choices=("random", "text"), default="text")
    p.add_argument("--csp-attr-dropout", type=float, default=0.3)
    p.add_argument("--csp-pair-separator", type=str, default=" ")
    p.add_argument(
        "--base-checkpoint",
        type=str,
        default="",
        help="Optional pretrained TextConditionedVisionModel state_dict for post-training initialization.",
    )
    p.add_argument(
        "--save",
        type=str,
        default=None,
        help=(
            "Output path for the CSP vocab artifact (dict with csp_vocab / meta / args). "
            "Default when omitted: next to --base-checkpoint as <stem>_csp_vocab.pt, "
            "or csp_vocab_posttrain.pt if no base checkpoint."
        ),
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    p.add_argument(
        "--wandb-project",
        default=_default_wandb_project(),
        help=(
            f"W&B project (code default: {DEFAULT_WANDB_PROJECT!r}; override with WANDB_PROJECT in .env)."
        ),
    )
    p.add_argument("--wandb-entity", default="", help="W&B entity/team (optional).")
    p.add_argument("--wandb-run-name", default="", help="W&B run name (optional).")
    p.add_argument(
        "--wandb-log-images",
        action="store_true",
        help="Log a train image panel each --wandb-image-log-interval epochs (epoch 0 = first batch).",
    )
    p.add_argument(
        "--wandb-image-log-interval",
        type=int,
        default=1,
        help="When --wandb-log-images is set, every N epochs log train images (default: 1).",
    )
    p.add_argument(
        "--wandb-max-images",
        type=int,
        default=8,
        help="Max images per W&B panel when --wandb-log-images is set.",
    )
    p.add_argument("--hyperparams-file", type=str, default=DEFAULT_HYPERPARAMS_FILE)
    g = p.add_argument_group("evaluation (use with --eval-only)")
    g.add_argument(
        "--eval-only",
        action="store_true",
        help="Load a saved CSP vocab artifact (--checkpoint) and evaluate on --eval-split (no training).",
    )
    g.add_argument(
        "--checkpoint",
        default="",
        help="Path to a saved CSP bundle (dict with csp_vocab / meta / args); required with --eval-only.",
    )
    g.add_argument(
        "--eval-split",
        choices=("val", "test"),
        default="val",
        help="val = held-out split; test = official test merge (see vision_data).",
    )
    g.add_argument(
        "--max-eval-samples",
        type=int,
        default=0,
        help="Cap eval split size; 0 = full (applies to the chosen --eval-split only).",
    )
    g.add_argument(
        "--metrics-json",
        default="",
        help="Optional JSON path for eval metrics (same schema keys as text_cond_train eval).",
    )
    g.add_argument(
        "--experiment-tag",
        default="",
        help="Optional tag stored in --metrics-json.",
    )
    return p


def main() -> None:
    argv = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_hyperparams_from_file(args, argv)
    if args.eval_only:
        run_csp_eval_only(args)
        return
    if not _arg_was_explicit(argv, "save"):
        args.save = _default_save_from_base(args.base_checkpoint)
    run_post_training(args)


if __name__ == "__main__":
    main()
