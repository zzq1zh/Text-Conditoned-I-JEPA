#!/usr/bin/env python3
"""
Convert a TextConditionedVisionModel ``torch.save`` checkpoint (``.pt``) into the Hugging Face Hub layout
used by ``text_cond_train`` / ``load_text_cond_trainable_from_hub``:

- ``trainable_model.safetensors`` — non-backbone weights only (backbone keys are stripped on upload)
- ``tc_ijepa_config.json`` — metadata for loading (class names, fusion, HF backbone ids, …)

Supports full checkpoints from ``text_cond_train.py`` (``torch.save(model.state_dict(), ...)``).

Does **not** support CSP vocab bundles (dict with ``csp_vocab`` / ``meta``); those are a different artifact.

Examples::

  uv run python push_text_cond_checkpoint_to_hub.py checkpoints/model.pt \\
      --repo-id YOUR_USER/your-model --dataset cspref_cgqa --vision-backbone dinov3

  uv run python push_text_cond_checkpoint_to_hub.py checkpoints/model.pt \\
      --hub-config-json hub_config_partial.json --dataset cspref_mit_states --vision-backbone ijepa \\
      --repo-id YOUR_USER/your-model

  uv run python push_text_cond_checkpoint_to_hub.py checkpoints/model.pt \\
      --dataset cspref_cgqa --vision-backbone dinov3 --output-dir ./hf_model_bundle --no-push

Run ``huggingface-cli login`` before uploading (or pass ``--token``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import project_env

project_env.load_project_env()

import torch  # noqa: E402

from main import (  # noqa: E402
    DEFAULT_CLIP_TEXT_ID,
    DEFAULT_PROMPT_TEMPLATE,
    TextConditionedVisionModel,
    VISION_BACKBONE_PRESETS,
    resolve_vision_model_id,
)
from text_cond_train import (  # noqa: E402
    HUB_CONFIG_FILENAME,
    HUB_WEIGHTS_FILENAME,
    _export_trainable_state_dict,
    push_text_cond_to_hub,
)
from vision_data import (  # noqa: E402
    list_vision_dataset_keys,
    load_vision_train_val_test_specs,
)


def _load_state_dict_from_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "csp_vocab" in obj and "meta" in obj:
        raise ValueError(
            "This file looks like a CSP vocab post-training bundle (contains 'csp_vocab' / 'meta'). "
            "This script only supports TextConditionedVisionModel state_dict .pt checkpoints."
        )
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            inner = obj.get(key)
            if isinstance(inner, dict) and inner and all(isinstance(k, str) for k in inner.keys()):
                obj = inner
                break
        if all(isinstance(k, str) for k in obj.keys()):
            tensors = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
            if tensors:
                return tensors  # type: ignore[return-value]
    raise TypeError(
        f"Could not parse state_dict from {path}. Expected a dict of tensor parameters "
        f"(optionally wrapped under 'state_dict' / 'model')."
    )


def _merged_hyperparams(path: Path, vision_backbone: str, dataset_key: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Hyperparameters file must be a JSON object: {path}")
    out: dict[str, Any] = {}
    model_key = str(vision_backbone)
    dkey = str(dataset_key)
    for src in (
        raw.get("defaults", {}),
        raw.get("models", {}).get(model_key, {}),
        raw.get("datasets", {}).get(dkey, {}),
        raw.get("model_dataset", {}).get(model_key, {}).get(dkey, {}),
    ):
        if isinstance(src, dict):
            out.update(src)
    return out


def _build_hub_config(
    *,
    hparams: dict[str, Any],
    hub_partial: dict[str, Any] | None,
    dataset_key: str,
    vision_backbone: str,
    ijepa: str,
    clip: str,
    val_fraction: float,
    split_seed: int,
    finetune_clip_text: bool,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if hub_partial:
        cfg.update(hub_partial)

    if "class_names" not in cfg:
        if dataset_key not in list_vision_dataset_keys():
            raise ValueError(
                f"--dataset {dataset_key!r} is not a known vision_data key; "
                "put class_names in --hub-config-json or fix --dataset."
            )
        tvt = load_vision_train_val_test_specs(
            dataset_key,
            val_fraction=float(val_fraction),
            split_seed=int(split_seed),
            max_train_samples=None,
            max_val_samples=None,
            max_test_samples=None,
        )
        cfg["class_names"] = list(tvt.train.class_names)

    for k in (
        "fusion_type",
        "cond_dim",
        "fusion_hidden",
        "text_template",
        "text_bank_chunk_size",
    ):
        if k not in cfg and k in hparams:
            cfg[k] = hparams[k]

    if (ijepa or "").strip():
        cfg["ijepa_id"] = resolve_vision_model_id(vision_backbone, ijepa)
    else:
        cfg.setdefault("ijepa_id", resolve_vision_model_id(vision_backbone, ""))

    cfg.setdefault("clip_id", DEFAULT_CLIP_TEXT_ID)
    if (clip or "").strip():
        cfg["clip_id"] = clip.strip()

    cfg.setdefault("cond_dim", 256)
    cfg.setdefault("fusion_hidden", 512)
    cfg.setdefault("fusion_type", "clip_similarity")
    cfg.setdefault("text_template", DEFAULT_PROMPT_TEMPLATE)
    cfg.setdefault("text_bank_chunk_size", 512)
    cfg.setdefault("finetune_clip_text", bool(finetune_clip_text))
    cfg.setdefault("dataset", dataset_key)
    cfg.setdefault("val_fraction", float(val_fraction))
    cfg.setdefault("split_seed", int(split_seed))

    cfg["num_labels"] = len(cfg["class_names"])
    cfg["model_type"] = "text_cond_ijepa"
    cfg["loss_mode"] = "bidirectional_infonce"

    if cfg["model_type"] != "text_cond_ijepa":
        raise ValueError(f"hub config model_type must be 'text_cond_ijepa', got {cfg['model_type']!r}")
    return cfg


def _verify_loads(model: TextConditionedVisionModel, sd: dict[str, torch.Tensor]) -> None:
    r = model.load_state_dict(sd, strict=False)
    miss = [k for k in r.missing_keys if not k.startswith("backbone.")]
    if miss:
        raise RuntimeError(
            "Checkpoint is missing trainable keys (non-backbone). "
            f"First few: {miss[:16]!r}"
        )
    uexp = [k for k in r.unexpected_keys if not k.startswith("backbone.")]
    if uexp:
        raise RuntimeError(f"Checkpoint has unexpected non-backbone keys: {uexp[:16]!r}")


def _write_local_bundle(model: TextConditionedVisionModel, hub_cfg: dict[str, Any], out_dir: Path) -> None:
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    wpath = out_dir / HUB_WEIGHTS_FILENAME
    cpath = out_dir / HUB_CONFIG_FILENAME
    save_file(_export_trainable_state_dict(model), str(wpath))
    cpath.write_text(json.dumps(hub_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {wpath} and {cpath}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert a .pt TextConditionedVisionModel checkpoint to Hub safetensors + config, optionally upload.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("checkpoint", type=Path, help="Path to .pt state_dict checkpoint")
    p.add_argument(
        "--repo-id",
        default="",
        help="Hugging Face model repo id (e.g. user/name). Required for upload unless --no-push.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If set, write trainable_model.safetensors + tc_ijepa_config.json to this directory.",
    )
    p.add_argument(
        "--hub-config-json",
        type=Path,
        default=None,
        help="Optional JSON with hub fields (e.g. class_names, ijepa_id). "
        "Missing fields filled from CLI / dataset / hyperparameters.json.",
    )
    p.add_argument(
        "--dataset",
        default="",
        metavar="DATASET",
        help="Used for class_names and hyperparameter lookup. "
        "Can be omitted only if --hub-config-json already contains class_names. "
        f"Choices: {', '.join(sorted(list_vision_dataset_keys()))}",
    )
    p.add_argument(
        "--vision-backbone",
        choices=tuple(sorted(VISION_BACKBONE_PRESETS.keys())),
        default="dinov3",
        help="Preset for ijepa_id when not overridden.",
    )
    p.add_argument(
        "--ijepa",
        default="",
        metavar="MODEL_ID",
        help="Explicit HF vision model id. Empty (default) uses --vision-backbone preset.",
    )
    p.add_argument(
        "--clip",
        default="",
        help=f"HF CLIP text model id (default: keep from JSON if any, else {DEFAULT_CLIP_TEXT_ID}).",
    )
    p.add_argument("--hyperparams-file", type=Path, default=Path("hyperparameters.json"))
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument(
        "--finetune-clip-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recorded in hub JSON (must match how the checkpoint was trained).",
    )
    p.add_argument("--private", action="store_true", help="Create/use a private Hub repo.")
    p.add_argument("--token", default="", help="HF write token (else huggingface-cli login cache).")
    p.add_argument(
        "--no-push",
        action="store_true",
        help="Do not upload; requires --output-dir.",
    )
    args = p.parse_args()

    ckpt_path = args.checkpoint.expanduser().resolve()
    if not ckpt_path.is_file():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(2)

    if not args.no_push and not (args.repo_id or "").strip():
        print("Provide --repo-id to upload, or pass --no-push with --output-dir.", file=sys.stderr)
        sys.exit(2)

    if args.no_push and args.output_dir is None:
        print("--no-push requires --output-dir.", file=sys.stderr)
        sys.exit(2)

    hub_partial: dict[str, Any] | None = None
    if args.hub_config_json:
        hub_partial = json.loads(Path(args.hub_config_json).read_text(encoding="utf-8"))
        if not isinstance(hub_partial, dict):
            raise SystemExit("--hub-config-json must contain a JSON object.")

    dataset_key = (args.dataset or "").strip()
    if not dataset_key:
        if not hub_partial or "class_names" not in hub_partial:
            raise SystemExit(
                "Provide --dataset (to load class_names), or supply class_names in --hub-config-json."
            )
        dataset_key = str(hub_partial.get("dataset") or "cspref_mit_states")

    if dataset_key not in list_vision_dataset_keys():
        if hub_partial and "class_names" in hub_partial:
            pass
        else:
            raise SystemExit(
                f"Unknown --dataset {dataset_key!r}. Use a vision_data key or put class_names in JSON."
            )

    hparams = _merged_hyperparams(args.hyperparams_file, args.vision_backbone, dataset_key)

    hub_cfg = _build_hub_config(
        hparams=hparams,
        hub_partial=hub_partial,
        dataset_key=dataset_key,
        vision_backbone=args.vision_backbone,
        ijepa=(args.ijepa or "").strip(),
        clip=args.clip,
        val_fraction=float(args.val_fraction),
        split_seed=int(args.split_seed),
        finetune_clip_text=bool(args.finetune_clip_text),
    )

    sd = _load_state_dict_from_checkpoint(ckpt_path)
    model = TextConditionedVisionModel(
        num_labels=int(hub_cfg["num_labels"]),
        ijepa_id=str(hub_cfg["ijepa_id"]),
        clip_id=str(hub_cfg["clip_id"]),
        cond_dim=int(hub_cfg["cond_dim"]),
        fusion_hidden=int(hub_cfg["fusion_hidden"]),
        fusion_type=str(hub_cfg.get("fusion_type", "clip_similarity")),
        freeze_text_encoder=not bool(hub_cfg.get("finetune_clip_text", False)),
    )
    _verify_loads(model, sd)
    print(f"Verified checkpoint matches hub config ({hub_cfg['num_labels']} classes).", flush=True)

    if args.output_dir is not None:
        out = args.output_dir.expanduser().resolve()
        _write_local_bundle(model, hub_cfg, out)

    if not args.no_push:
        url = push_text_cond_to_hub(
            model,
            hub_cfg,
            (args.repo_id or "").strip(),
            private=bool(args.private),
            token=(args.token or "").strip() or None,
        )
        print(url, flush=True)


if __name__ == "__main__":
    main()
