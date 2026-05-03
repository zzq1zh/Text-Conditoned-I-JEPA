"""
Standalone post-training for CSP-style compositional vocabulary.

This script is intentionally self-contained:
- no dependency on ``text_cond_train.py`` or ``csp_vocab.py``
- shares validation metrics with ``text_cond_train`` through ``csp_eval``
- trains only CSP attr/object embeddings while freezing TextConditionedIJepa
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
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
    DEFAULT_IJEPA_ID,
    DEFAULT_PROMPT_TEMPLATE,
    TextConditionedIJepa,
    VISION_BACKBONE_PRESETS,
    _extract_model_pixel_values,
    _resolve_device,
    load_vision_processor,
    resolve_vision_model_id,
)
from vision_data import (  # noqa: E402
    list_vision_dataset_keys,
    load_vision_train_val_test_specs,
    set_seed,
)

from csp_eval import clip_contrastive_loss, eval_clip_style_classification  # noqa: E402


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
    return _resolve_device(None)


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
        out = self.text_encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        text_embeds = out.text_embeds
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
    model: TextConditionedIJepa,
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
    model: TextConditionedIJepa,
    csp_vocab: CspCompositionVocab,
    csp_meta: CspVocabMeta,
    device: torch.device,
    *,
    use_amp: bool,
) -> Callable[[dict[str, Any]], tuple[torch.Tensor, torch.Tensor | None]]:
    """Per-batch forward for :func:`eval_clip_style_classification` (composed text bank + contrastive loss)."""

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
            logits = model.score_candidates(z, candidate_text_bank)
            pos_text = candidate_text_bank.index_select(0, y)
            pair_scores = model.score_candidates(z, pos_text)
            loss = clip_contrastive_loss(pair_scores)
        return logits, loss

    return forward_batch


def _load_base_checkpoint_if_any(model: TextConditionedIJepa, checkpoint: str) -> None:
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

    model = TextConditionedIJepa(
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

    for epoch in range(int(args.epochs)):
        csp_vocab.train()
        pbar = tqdm(train_loader, desc=f"csp post-train ep {epoch + 1}/{args.epochs}", file=sys.stdout)
        for batch_idx, batch in enumerate(pbar):
            pv = batch["pixel_values"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_cuda_amp,
                dtype=torch.float16,
            ):
                # Re-compose every step so gradients flow into CSP vocab parameters.
                candidate_text_bank = csp_vocab.compose_all_pairs(csp_meta, device=device)
                z = model.encode_image(pv)
                pos_text = candidate_text_bank.index_select(0, y)
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
            if (batch_idx % int(args.log_interval)) == 0:
                with torch.inference_mode():
                    full_logits = model.score_candidates(z, candidate_text_bank)
                    batch_acc = float((full_logits.argmax(-1) == y).float().mean().item())
                pbar.set_postfix(loss=f"{float(loss.detach()):.3f}", acc=f"{batch_acc:.3f}")
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

    if args.save and args.save.strip():
        out = {
            "csp_vocab": csp_vocab.state_dict(),
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standalone CSP vocab post-training (freeze base model, train compositional vocab).",
        epilog=(
            "Example: uv run python csp_vocab_train.py "
            "--dataset cspref_mit_states --base-checkpoint runs/base.pt  "
            "(default save: runs/base_csp_vocab.pt)"
        ),
    )
    p.add_argument("--dataset", default="cspref_mit_states", choices=list_vision_dataset_keys())
    p.add_argument(
        "--vision-backbone",
        default="ijepa",
        choices=tuple(sorted(VISION_BACKBONE_PRESETS.keys())),
        help="Vision backbone preset key used to resolve --ijepa.",
    )
    p.add_argument("--ijepa", default=DEFAULT_IJEPA_ID)
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
        choices=("cross_attention", "linear", "clip_similarity"),
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
        help="Optional pretrained TextConditionedIJepa state_dict for post-training initialization.",
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
    p.add_argument("--hyperparams-file", type=str, default=DEFAULT_HYPERPARAMS_FILE)
    return p


def main() -> None:
    argv = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_hyperparams_from_file(args, argv)
    if not _arg_was_explicit(argv, "save"):
        args.save = _default_save_from_base(args.base_checkpoint)
    run_post_training(args)


if __name__ == "__main__":
    main()
