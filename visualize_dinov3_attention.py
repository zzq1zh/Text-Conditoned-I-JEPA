#!/usr/bin/env python3
"""
Visualize self-attention maps from a Hugging Face DINOv3 ViT backbone.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoTokenizer

import project_env

project_env.load_project_env()

from csp_vocab_train import (  # noqa: E402
    CspCompositionVocab,
    CspVocabMeta,
)

from main import (  # noqa: E402
    DEFAULT_CLIP_TEXT_ID,
    TextConditionedVisionModel,
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


def _extract_backbone_state(ckpt_obj: Any) -> dict[str, torch.Tensor] | None:
    """Return backbone state dict from a raw checkpoint, or None."""
    if not isinstance(ckpt_obj, dict):
        return None
    if isinstance(ckpt_obj.get("backbone"), dict):
        return ckpt_obj["backbone"]  # type: ignore[return-value]
    prefix = "backbone."
    out: dict[str, torch.Tensor] = {}
    for k, v in ckpt_obj.items():
        if not isinstance(k, str):
            continue
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
    return out or None


def _load_backbone(model_id: str, device: torch.device, *, tuned_sd: dict[str, torch.Tensor] | None) -> nn.Module:
    dtype = torch.float32
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    if tuned_sd:
        res = model.load_state_dict(tuned_sd, strict=False)
        print(
            f"Loaded tuned backbone weights: missing={len(res.missing_keys)} "
            f"unexpected={len(res.unexpected_keys)}",
            flush=True,
        )
        if res.missing_keys[:8]:
            print(f"  missing (first): {res.missing_keys[:8]}", flush=True)
        if res.unexpected_keys[:8]:
            print(f"  unexpected (first): {res.unexpected_keys[:8]}", flush=True)
    else:
        print("Using pretrained backbone from Hub (no tuned weights in checkpoint).", flush=True)
    return model


def _encoder_layers(model: nn.Module) -> nn.ModuleList:
    """
    Return the ``ModuleList`` of transformer blocks.

    - Dinov2-style ViTs: ``model.encoder.layer``
    - DINOv3 ViT (``DINOv3ViTModel``): ``model.model.layer`` (encoder submodule is named ``model``).
    """
    enc = getattr(model, "encoder", None)
    if enc is None:
        enc = getattr(model, "model", None)
    if enc is None:
        raise RuntimeError(
            "Backbone has neither .encoder nor .model; expected a Dinov2- or DINOv3-ViT-style HF vision model."
        )
    layers = getattr(enc, "layer", None)
    if layers is None:
        raise RuntimeError("Vision encoder submodule has no .layer ModuleList.")
    return layers


def _patch_grid(config: Any) -> tuple[int, int, int]:
    """Return (num_patch_h, num_patch_w, num_leading_special). Leading specials = CLS + register tokens."""
    ps = getattr(config, "patch_size", 16)
    if isinstance(ps, (list, tuple)):
        ps = int(ps[0])
    else:
        ps = int(ps)
    img = getattr(config, "image_size", 224)
    if isinstance(img, (list, tuple)):
        img = int(img[0])
    else:
        img = int(img)
    nh, nw = img // ps, img // ps
    n_prefix = 1 + int(getattr(config, "num_register_tokens", 0))
    return nh, nw, n_prefix


def _cls_to_patch_map(
    attn_bhss: torch.Tensor,
    *,
    nh: int,
    nw: int,
    n_prefix: int,
) -> np.ndarray:
    """
      attn: (B, H, S, S) -> map (nh, nw) averaged over batch and heads,
      CLS query (index 0) attending to patch key positions after prefix tokens.
    """
    b, h, s, _ = attn_bhss.shape
    n_patch = nh * nw
    if s < n_prefix + n_patch:
        raise ValueError(
            f"Attention seq {s} shorter than prefix({n_prefix})+patches({n_patch}). Check model layout."
        )
    # CLS is first token in DINOv2/DINOv3 ViT
    vec = attn_bhss[:, :, 0, n_prefix : n_prefix + n_patch]  # (B,H,P)
    vec = vec.mean(dim=(0, 1)).float().cpu().numpy().reshape(nh, nw)
    vec = vec - vec.min()
    den = vec.max() + 1e-8
    vec = vec / den
    return vec


def _forward_attentions(model: nn.Module, pixel_values: torch.Tensor) -> tuple[tuple[torch.Tensor, ...] | None, str]:
    """Run forward; return (attentions tuple or None, message if fallback needed)."""
    with torch.inference_mode():
        try:
            out = model(pixel_values=pixel_values, output_attentions=True)
        except TypeError:
            return None, "forward does not accept output_attentions"
        att = getattr(out, "attentions", None)
        if att is None or len(att) == 0:
            return None, "output has no attentions"
        return att, "ok"


def _forward_with_attention_hooks(
    model: nn.Module,
    pixel_values: torch.Tensor,
    n_layers: int,
) -> list[torch.Tensor]:
    """
    Fallback: register forward hooks on each encoder layer's attention submodule
    and capture softmax probabilities when available.
    """
    layers = _encoder_layers(model)
    if len(layers) != n_layers:
        n_layers = len(layers)
    storage: list[torch.Tensor | None] = [None] * n_layers
    hooks: list[Any] = []

    def make_hook(i: int):
        def hook(_mod: nn.Module, _args: Any, output: Any) -> None:
            # HF often returns (hidden_states, attentions?, ...)
            if isinstance(output, tuple) and len(output) >= 2:
                att = output[1]
                if att is not None and torch.is_tensor(att):
                    storage[i] = att.detach()
            elif torch.is_tensor(output):
                storage[i] = output.detach()

        return hook

    for i, layer in enumerate(layers):
        attn_mod = getattr(layer, "attention", None)
        if attn_mod is None:
            continue
        # self_attn or attention.core — try common names
        sub = getattr(attn_mod, "attention", attn_mod)
        hooks.append(sub.register_forward_hook(make_hook(i)))

    with torch.inference_mode():
        model(pixel_values=pixel_values)

    for h in hooks:
        h.remove()

    if all(x is None for x in storage):
        raise RuntimeError(
            "Could not capture attentions via hooks. Either upgrade transformers "
            "or use a checkpoint/model that supports output_attentions=True."
        )
    out_list: list[torch.Tensor] = []
    for i, t in enumerate(storage):
        if t is None:
            raise RuntimeError(f"Missing attention capture at layer {i}")
        out_list.append(t)
    return out_list


def _upsample_map(m: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    oh, ow = out_hw
    return cv2.resize(m.astype(np.float32), (ow, oh), interpolation=cv2.INTER_LINEAR)


def _meta_from_bundle(meta_d: dict[str, Any]) -> CspVocabMeta:
    return CspVocabMeta(
        attrs=list(meta_d["attrs"]),
        objs=list(meta_d["objs"]),
        pairs=list(meta_d["pairs"]),
        pair_attr_idx=meta_d["pair_attr_idx"],
        pair_obj_idx=meta_d["pair_obj_idx"],
        pair_separator=str(meta_d.get("pair_separator", " ")),
    )


def _require_csp_bundle(raw: Any, path: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or "csp_vocab" not in raw or "meta" not in raw:
        raise TypeError(
            f"{path} must be a CSP bundle dict with 'csp_vocab' and 'meta'; got {type(raw).__name__}"
        )
    return raw


def _bundle_training_args(bundle: dict[str, Any]) -> SimpleNamespace:
    d = bundle.get("args")
    if not isinstance(d, dict):
        d = {}
    return SimpleNamespace(**d)


def _resolve_args_field(ns: SimpleNamespace, key: str, default: Any) -> Any:
    return getattr(ns, key, default)


def _ijepa_id_from_bundle(bundle: dict[str, Any]) -> str:
    ba = _bundle_training_args(bundle)
    vb = str(_resolve_args_field(ba, "vision_backbone", "dinov3") or "dinov3")
    ij_raw = str(_resolve_args_field(ba, "ijepa", "") or "")
    return resolve_vision_model_id(vb, ij_raw)


def _load_csp_textconditioned(
    bundle: dict[str, Any],
    csp_meta: CspVocabMeta,
    device: torch.device,
    *,
    load_backbone_weights: bool,
) -> TextConditionedVisionModel:
    """Instantiate :class:`TextConditionedVisionModel` from bundle; optionally skip backbone tensors."""
    ba = _bundle_training_args(bundle)
    finetune_v = bool(_resolve_args_field(ba, "finetune_vision_backbone", False))
    clip_id = str(_resolve_args_field(ba, "clip", DEFAULT_CLIP_TEXT_ID) or DEFAULT_CLIP_TEXT_ID)
    ijepa_id = _ijepa_id_from_bundle(bundle)
    cond_dim = int(_resolve_args_field(ba, "cond_dim", 256))
    fusion_hidden = int(_resolve_args_field(ba, "fusion_hidden", 512))
    fusion_type = str(_resolve_args_field(ba, "fusion_type", "cross_attention"))

    n_classes = len(csp_meta.pairs)
    model = TextConditionedVisionModel(
        num_labels=n_classes,
        ijepa_id=ijepa_id,
        clip_id=clip_id,
        cond_dim=cond_dim,
        fusion_hidden=fusion_hidden,
        fusion_type=fusion_type,
        freeze_text_encoder=True,
        freeze_vision_backbone=not finetune_v,
    ).to(device)
    model.eval()

    ad = bundle.get("adapter")
    fu = bundle.get("fusion")
    if isinstance(ad, dict):
        model.text_cond.adapter.load_state_dict(ad, strict=True)
    if isinstance(fu, dict):
        model.fusion.load_state_dict(fu, strict=True)

    bb = bundle.get("backbone")
    if load_backbone_weights:
        if isinstance(bb, dict) and bb:
            model.backbone.load_state_dict(bb, strict=True)
        else:
            raise ValueError(
                "CSP compare: tuned checkpoint must contain a non-empty 'backbone' dict "
                "(vision-finetuned bundle)."
            )
    elif isinstance(bb, dict) and bb:
        print(
            "Note: base checkpoint contains 'backbone' tensors; they are ignored (pretrained backbone only).",
            flush=True,
        )

    for p in model.parameters():
        p.requires_grad = False
    return model


def _build_csp_vocab(
    bundle: dict[str, Any],
    model: TextConditionedVisionModel,
    csp_meta: CspVocabMeta,
    tok: Any,
    device: torch.device | str,
) -> CspCompositionVocab:
    ba = _bundle_training_args(bundle)
    ctx = int(_resolve_args_field(ba, "csp_context_length", 8))
    attr_do = float(_resolve_args_field(ba, "csp_attr_dropout", 0.3))
    cond_dim = int(_resolve_args_field(ba, "cond_dim", 256))
    return CspCompositionVocab(
        num_attrs=len(csp_meta.attrs),
        num_objs=len(csp_meta.objs),
        text_encoder=model.text_cond.text_encoder,
        adapter=model.text_cond.adapter,
        tokenizer=tok,
        text_hidden_dim=int(model.text_cond.text_encoder.config.hidden_size),
        cond_dim=cond_dim,
        context_length=ctx,
        attr_dropout=attr_do,
    ).to(device)


def _assert_meta_pairs_equal(meta_a: dict[str, Any], meta_b: dict[str, Any], path_a: str, path_b: str) -> None:
    pa, pb = list(meta_a["pairs"]), list(meta_b["pairs"])
    if pa != pb:
        raise ValueError(
            f"CSP meta.pairs mismatch between {path_a} and {path_b} "
            f"(len {len(pa)} vs {len(pb)} or different order)."
        )


def _backbone_to_eager_attn(model: TextConditionedVisionModel, ijepa_id: str, device: torch.device) -> None:
    bb = AutoModel.from_pretrained(
        ijepa_id,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    bb.load_state_dict(model.backbone.state_dict(), strict=True)
    bb.to(device)
    bb.eval()
    model.backbone = bb


@torch.inference_mode()
def _csp_logits_one_image(
    model: TextConditionedVisionModel,
    csp_vocab: CspCompositionVocab,
    csp_meta: CspVocabMeta,
    pixel_values: torch.Tensor,
    device: torch.device,
    *,
    allowed_class_indices: list[int] | None,
    use_amp: bool,
) -> torch.Tensor:
    """Classification logits ``(1, num_classes)`` (optional eval-style class mask)."""
    c_full = len(csp_meta.pairs)
    pv = pixel_values.to(device, dtype=torch.float32, non_blocking=True)
    with torch.amp.autocast(
        device_type=device.type,
        enabled=(device.type == "cuda" and use_amp),
        dtype=torch.float16,
    ):
        candidate_text_bank = csp_vocab.compose_all_pairs(csp_meta, device=device)
        z = model.encode_image(pv)

        if allowed_class_indices is None:
            return model.score_candidates(z, candidate_text_bank)

        allowed_unique = sorted({int(i) for i in allowed_class_indices})
        allowed_t = torch.tensor(allowed_unique, dtype=torch.long, device=device)
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
    return logits


def _extract_layer_attention_maps(
    backbone: nn.Module,
    pixel_values: torch.Tensor,
    *,
    layers: list[int],
    nh: int,
    nw: int,
    n_prefix: int,
) -> dict[int, np.ndarray]:
    n_enc = len(_encoder_layers(backbone))
    for idx in layers:
        if idx < 0 or idx >= n_enc:
            raise ValueError(f"Layer index {idx} out of range [0, {n_enc - 1}]")

    att_tuple, msg = _forward_attentions(backbone, pixel_values)
    if att_tuple is None:
        print(f"output_attentions failed ({msg}); using hooks.", flush=True)
        att_list = _forward_with_attention_hooks(backbone, pixel_values, n_enc)
    else:
        att_list = list(att_tuple)

    return {li: _cls_to_patch_map(att_list[li], nh=nh, nw=nw, n_prefix=n_prefix) for li in layers}


@dataclass
class _CspContrastSample:
    pil: Image.Image
    label: int
    y_name: str
    pred_base: int
    pred_tuned: int
    split_row_index: int
    pos_phrase: str | None = None


def _scan_csp_contrast_samples(
    *,
    dataset_key: str,
    model_t: TextConditionedVisionModel,
    model_b: TextConditionedVisionModel,
    csp_t: CspCompositionVocab,
    csp_b: CspCompositionVocab,
    csp_meta: CspVocabMeta,
    bundle_tuned: dict[str, Any],
    proc: Any,
    device: torch.device,
    n_want: int,
    max_scan: int,
    use_amp: bool,
    seed: int,
) -> list[_CspContrastSample]:
    """Scan only ``tvt.test`` (for CSP Hub datasets this is the official ``test`` split only)."""
    ba = _bundle_training_args(bundle_tuned)
    vf = float(_resolve_args_field(ba, "val_fraction", 0.1))
    ss = int(_resolve_args_field(ba, "split_seed", 0))
    tvt = load_vision_train_val_test_specs(
        dataset_key,
        val_fraction=vf,
        split_seed=ss,
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=None,
    )
    data_names = list(tvt.train.class_names)
    ckpt_pairs = list(csp_meta.pairs)
    if data_names != ckpt_pairs:
        trained_on = _resolve_args_field(ba, "dataset", None)
        hint = ""
        if trained_on and str(trained_on) != str(dataset_key):
            hint = (
                f" This checkpoint's saved args.dataset is {trained_on!r} (you passed --csp-dataset {dataset_key!r}). "
                "Use the same dataset key as training."
            )
        elif trained_on:
            hint = (
                " Same key was used but class lists differ—re-download the Hub dataset or rebuild the bundle "
                "on this machine."
            )
        raise ValueError(
            f"CSP bundle label space does not match dataset {dataset_key!r}: "
            f"len(train.class_names)={len(data_names)} vs len(meta.pairs)={len(ckpt_pairs)}. "
            f"CSP compare requires an exact ClassLabel order match with the bundle.{hint}"
        )

    spec = tvt.test
    ds = spec.dataset
    image_col = spec.image_column
    lk = spec.label_key
    class_names = list(tvt.train.class_names)

    n_classes = len(class_names)
    allowed = csp_vocab_allowed_class_indices(tvt, "test")
    allowed_arg: list[int] | None = allowed if len(allowed) < n_classes else None
    if allowed_arg is not None:
        print(
            f"  {dataset_key}: test split uses {len(allowed_arg)}/{n_classes} allowed classes (train∪test labels).",
            flush=True,
        )

    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    found: list[_CspContrastSample] = []
    scanned = 0
    for idx in indices:
        if len(found) >= n_want or scanned >= max_scan:
            break
        scanned += 1
        row = ds[idx]
        pil = row[image_col]
        if not isinstance(pil, Image.Image):
            pil = Image.fromarray(np.asarray(pil)).convert("RGB")
        else:
            pil = pil.convert("RGB")
        label = int(row[lk])
        if allowed_arg is not None and label not in allowed_arg:
            continue

        vis = proc(images=pil, return_tensors="pt")
        pv = _extract_model_pixel_values(vis).to(device, dtype=torch.float32)

        logits_t = _csp_logits_one_image(
            model_t, csp_t, csp_meta, pv, device, allowed_class_indices=allowed_arg, use_amp=use_amp
        )
        logits_b = _csp_logits_one_image(
            model_b, csp_b, csp_meta, pv, device, allowed_class_indices=allowed_arg, use_amp=use_amp
        )

        pred_t = int(logits_t.argmax(dim=-1).item())
        pred_b = int(logits_b.argmax(dim=-1).item())
        if pred_t == label and pred_b != label:
            pos_phrase: str | None = None
            if "pos" in ds.column_names:
                raw_p = row["pos"]
                pos_phrase = None if raw_p is None else str(raw_p)
            found.append(
                _CspContrastSample(
                    pil=pil,
                    label=label,
                    y_name=class_names[label],
                    pred_base=pred_b,
                    pred_tuned=pred_t,
                    split_row_index=int(idx),
                    pos_phrase=pos_phrase,
                )
            )

    print(
        f"  {dataset_key}: found {len(found)}/{n_want} contrast samples "
        f"(scanned {scanned}, max_scan={max_scan}).",
        flush=True,
    )
    return found


def _save_csp_compare_sample_artifacts(
    samples: list[_CspContrastSample],
    *,
    pair_names: list[str],
    out_dir: Path,
    dataset_key: str,
) -> None:
    """Write one PNG per sample plus ``manifest.json`` with labels and pair strings."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_samples: list[dict[str, Any]] = []
    for i, s in enumerate(samples):
        fname = f"sample_{i:03d}.png"
        s.pil.save(out_dir / fname)
        entry: dict[str, Any] = {
            "index_in_manifest": i,
            "image_file": fname,
            "split_row_index": s.split_row_index,
            "label_id": s.label,
            "pair_label_text": s.y_name,
            "pred_pretrained_id": s.pred_base,
            "pred_tuned_id": s.pred_tuned,
            "pred_pretrained_pair_text": pair_names[s.pred_base],
            "pred_tuned_pair_text": pair_names[s.pred_tuned],
        }
        if s.pos_phrase is not None:
            entry["dataset_pos_phrase"] = s.pos_phrase
        manifest_samples.append(entry)

    payload = {
        "dataset": dataset_key,
        "eval_split": "test",
        "samples": manifest_samples,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(samples)} images and manifest.json under {out_dir.resolve()}", flush=True)


def _figure_csp_backbone_compare(
    samples: list[_CspContrastSample],
    *,
    model_t: TextConditionedVisionModel,
    model_b: TextConditionedVisionModel,
    bundle_tuned: dict[str, Any],
    proc: Any,
    device: torch.device,
    layers: list[int],
    dataset_tag: str,
    out_path: Path,
    ckpt_tuned: Path,
    ckpt_base: Path,
) -> None:
    ijepa_id = _ijepa_id_from_bundle(bundle_tuned)
    nh, nw, n_prefix = _patch_grid(model_t.backbone.config)
    n_layers = len(layers)
    ncols = 1 + 2 * n_layers
    nrows = len(samples)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.5 * max(nrows, 1)), squeeze=False)

    for r, sample in enumerate(samples):
        vis = proc(images=sample.pil, return_tensors="pt")
        pv = _extract_model_pixel_values(vis).to(device, dtype=torch.float32)
        np_img = np.asarray(sample.pil).astype(np.float32) / 255.0
        H, W = np_img.shape[0], np_img.shape[1]

        maps_t = _extract_layer_attention_maps(
            model_t.backbone, pv, layers=layers, nh=nh, nw=nw, n_prefix=n_prefix
        )
        maps_b = _extract_layer_attention_maps(
            model_b.backbone, pv, layers=layers, nh=nh, nw=nw, n_prefix=n_prefix
        )

        row_title = (
            f"{r + 1}: y={sample.y_name} | tuned_ok pretrained_wrong | pretrained_top1={sample.pred_base}"
        )
        for c in range(ncols):
            ax = axes[r, c]
            if c == 0:
                ax.imshow(np_img)
                ax.set_title(row_title, fontsize=7)
            elif c <= n_layers:
                li = layers[c - 1]
                um = _upsample_map(maps_t[li], (H, W))
                ax.imshow(np_img)
                ax.imshow(um, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
                ax.set_title(f"tuned L{li}", fontsize=8)
            else:
                li = layers[c - 1 - n_layers]
                um = _upsample_map(maps_b[li], (H, W))
                ax.imshow(np_img)
                ax.imshow(um, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
                ax.set_title(f"pretrained L{li}", fontsize=8)
            ax.axis("off")

    sup = f"{dataset_tag}\ntuned_ckpt: {ckpt_tuned}\nbase_ckpt (pretrained backbone): {ckpt_base}\n{ijepa_id}"
    fig.suptitle(sup, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_path.resolve()}", flush=True)


def run_csp_backbone_compare(args: argparse.Namespace) -> None:
    p_t = args.csp_checkpoint_tuned
    p_b = args.csp_checkpoint_base
    if not p_t.is_file() or not p_b.is_file():
        raise SystemExit(f"Missing checkpoint: tuned={p_t} base={p_b}")

    set_seed(int(args.seed))
    device = torch.device(args.device)
    use_amp = bool(args.amp) and device.type == "cuda"

    raw_t = torch.load(p_t, map_location="cpu", weights_only=True)
    raw_b = torch.load(p_b, map_location="cpu", weights_only=True)
    bundle_t = _require_csp_bundle(raw_t, str(p_t))
    bundle_b = _require_csp_bundle(raw_b, str(p_b))
    _assert_meta_pairs_equal(bundle_t["meta"], bundle_b["meta"], str(p_t), str(p_b))

    if _ijepa_id_from_bundle(bundle_t) != _ijepa_id_from_bundle(bundle_b):
        raise ValueError(
            f"Vision backbone id mismatch: tuned={_ijepa_id_from_bundle(bundle_t)!r} "
            f"base={_ijepa_id_from_bundle(bundle_b)!r}"
        )

    csp_meta = _meta_from_bundle(bundle_t["meta"])

    clip_t = str(_resolve_args_field(_bundle_training_args(bundle_t), "clip", DEFAULT_CLIP_TEXT_ID))
    clip_b = str(_resolve_args_field(_bundle_training_args(bundle_b), "clip", DEFAULT_CLIP_TEXT_ID))
    if clip_t != clip_b:
        raise ValueError(f"CLIP id mismatch: tuned={clip_t!r} base={clip_b!r}")

    model_t = _load_csp_textconditioned(bundle_t, csp_meta, device, load_backbone_weights=True)
    model_b = _load_csp_textconditioned(bundle_b, csp_meta, device, load_backbone_weights=False)

    tok = AutoTokenizer.from_pretrained(clip_t)
    csp_mod_t = _build_csp_vocab(bundle_t, model_t, csp_meta, tok, device)
    csp_mod_t.load_state_dict(bundle_t["csp_vocab"], strict=True)
    csp_mod_t.eval()
    csp_mod_b = _build_csp_vocab(bundle_b, model_b, csp_meta, tok, device)
    csp_mod_b.load_state_dict(bundle_b["csp_vocab"], strict=True)
    csp_mod_b.eval()

    ijepa_id = _ijepa_id_from_bundle(bundle_t)
    _backbone_to_eager_attn(model_t, ijepa_id, device)
    _backbone_to_eager_attn(model_b, ijepa_id, device)

    proc = load_vision_processor(ijepa_id)
    n = int(args.csp_n_samples)
    max_scan = int(args.csp_max_scan)
    ds_key = str(args.csp_dataset)

    if ds_key not in list_vision_dataset_keys():
        raise SystemExit(f"Unknown dataset {ds_key!r}. Choose from {list_vision_dataset_keys()}")

    samples = _scan_csp_contrast_samples(
        dataset_key=ds_key,
        model_t=model_t,
        model_b=model_b,
        csp_t=csp_mod_t,
        csp_b=csp_mod_b,
        csp_meta=csp_meta,
        bundle_tuned=bundle_t,
        proc=proc,
        device=device,
        n_want=n,
        max_scan=max_scan,
        use_amp=use_amp,
        seed=int(args.seed),
    )
    if not samples:
        raise SystemExit(
            f"No contrast samples found on the test split for {ds_key!r} (try increasing --csp-max-scan)."
        )

    if not args.csp_no_save_samples:
        save_dir = (
            Path(args.csp_save_samples_dir)
            if args.csp_save_samples_dir is not None
            else Path(args.csp_out_dir) / f"{ds_key}_samples"
        )
        _save_csp_compare_sample_artifacts(
            samples,
            pair_names=list(csp_meta.pairs),
            out_dir=save_dir,
            dataset_key=ds_key,
        )

    out_p = Path(args.csp_out_dir) / f"{ds_key}_csp_attn_compare.png"
    _figure_csp_backbone_compare(
        samples,
        model_t=model_t,
        model_b=model_b,
        bundle_tuned=bundle_t,
        proc=proc,
        device=device,
        layers=list(args.layers),
        dataset_tag=ds_key,
        out_path=out_p,
        ckpt_tuned=p_t,
        ckpt_base=p_b,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--csp-compare",
        action="store_true",
        help="CSP mode: on one --csp-dataset, search **test split only**; contrast samples must have finetuned "
        "backbone top-1 correct and pretrained-only backbone top-1 wrong.",
    )
    p.add_argument(
        "--csp-checkpoint-tuned",
        type=Path,
        default=None,
        help="CSP bundle with finetuned vision backbone (non-empty backbone dict).",
    )
    p.add_argument(
        "--csp-checkpoint-base",
        type=Path,
        default=None,
        help="CSP bundle for adapter/fusion/csp_vocab; 'backbone' tensors are ignored (pretrained weights only). "
        "May be the same file as --csp-checkpoint-tuned when only the vision backbone should differ.",
    )
    p.add_argument(
        "--csp-n-samples",
        type=int,
        default=3,
        dest="csp_n_samples",
        help="Number of contrast image+label pairs to collect from --csp-dataset (default: 3).",
    )
    p.add_argument(
        "--csp-dataset",
        default=None,
        help="Single vision_data key for --csp-compare (e.g. cspref_mit_states). Required with --csp-compare.",
    )
    p.add_argument(
        "--csp-max-scan",
        type=int,
        default=8000,
        help="Max test-split rows to scan when searching for contrast pairs (default: 8000).",
    )
    p.add_argument(
        "--csp-out-dir",
        type=Path,
        default=Path("csp_attention_compare"),
        help="Output directory; writes {dataset}_csp_attn_compare.png and, by default, {dataset}_samples/ with PNGs + manifest.",
    )
    p.add_argument(
        "--csp-no-save-samples",
        action="store_true",
        help="With --csp-compare: do not write per-sample PNGs or manifest.json (attention grid only).",
    )
    p.add_argument(
        "--csp-save-samples-dir",
        type=Path,
        default=None,
        help="With --csp-compare: directory for PNGs + manifest.json. "
        "Default: {csp_out_dir}/{dataset}_samples/ when saving is enabled.",
    )
    p.add_argument("--seed", type=int, default=0, help="Shuffle seed for scanning order.")
    p.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA autocast fp16 for CSP logits (attention viz stays fp32).",
    )
    p.add_argument("--image", type=Path, default=None, help="Input RGB image path (single-image mode)")
    p.add_argument("--checkpoint", type=Path, default=None, help="Optional .pt (full model or CSP bundle)")
    p.add_argument("--model-id", default="", help="Override HuggingFace vision model id")
    p.add_argument("--vision-backbone", default="dinov3", help="Preset when --model-id empty (default: dinov3)")
    p.add_argument("--out", type=Path, default=Path("dinov3_attention_maps.png"))
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[6, 8, 9, 10, 11],
        help="0-based encoder layer indices (default: 6 8 9 10 11)",
    )
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()

    if args.csp_compare:
        if args.csp_checkpoint_tuned is None or args.csp_checkpoint_base is None:
            raise SystemExit("--csp-compare requires --csp-checkpoint-tuned and --csp-checkpoint-base.")
        if not args.csp_dataset:
            raise SystemExit("--csp-compare requires --csp-dataset (single vision_data registry key).")
        run_csp_backbone_compare(args)
        return

    if args.image is None:
        raise SystemExit("Provide --image for single-image mode, or use --csp-compare with CSP checkpoints.")

    device = torch.device(args.device)
    model_id = resolve_vision_model_id(args.vision_backbone, args.model_id)

    tuned: dict[str, torch.Tensor] | None = None
    if args.checkpoint and args.checkpoint.is_file():
        raw = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        tuned = _extract_backbone_state(raw)
        if tuned is None:
            print(
                f"Checkpoint {args.checkpoint} has no backbone weights; using pretrained backbone only.",
                flush=True,
            )
    elif args.checkpoint:
        print(f"No file at {args.checkpoint}; using pretrained backbone only.", flush=True)

    model = _load_backbone(model_id, device, tuned_sd=tuned)
    proc = load_vision_processor(model_id)
    config = model.config
    nh, nw, n_prefix = _patch_grid(config)
    n_enc = len(_encoder_layers(model))

    for idx in args.layers:
        if idx < 0 or idx >= n_enc:
            raise SystemExit(f"Layer index {idx} out of range [0, {n_enc - 1}] for this model.")

    pil = Image.open(args.image).convert("RGB")
    batch = proc(images=pil, return_tensors="pt")
    pv = batch["pixel_values"].to(device, dtype=torch.float32)

    att_tuple, msg = _forward_attentions(model, pv)
    if att_tuple is None:
        print(f"output_attentions path failed ({msg}); trying hooks.", flush=True)
        att_list = _forward_with_attention_hooks(model, pv, n_enc)
    else:
        att_list = list(att_tuple)

    maps: list[tuple[int, np.ndarray]] = []
    for li in args.layers:
        m = _cls_to_patch_map(att_list[li], nh=nh, nw=nw, n_prefix=n_prefix)
        maps.append((li, m))

    np_img = np.asarray(pil).astype(np.float32) / 255.0
    H, W = np_img.shape[0], np_img.shape[1]

    ncols = min(3, len(maps) + 1)
    nrows = (len(maps) + 1 + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    flat = axes.ravel()
    flat[0].imshow(np_img)
    flat[0].set_title("input")
    flat[0].axis("off")
    for i, (li, m) in enumerate(maps, start=1):
        um = _upsample_map(m, (H, W))
        ax = flat[i]
        ax.imshow(np_img)
        ax.imshow(um, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
        ax.set_title(f"layer {li} (CLS→patch)")
        ax.axis("off")
    for j in range(len(maps) + 1, len(flat)):
        flat[j].axis("off")
    fig.suptitle(f"{model_id}\n{args.checkpoint}" if args.checkpoint else model_id, fontsize=10)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"Wrote {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
