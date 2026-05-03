#!/usr/bin/env python3
"""
Unified launcher for CSP vocabulary post-training across seeds.

Mirrors ``run_text_cond_train.py``: reads ``hyperparameters.json`` (defaults →
``models.<backbone>`` → ``datasets.<dataset>`` → ``model_dataset``) for the
full training config. The spawned ``csp_vocab_train.py`` process gets the same
file and applies any keys not overridden on the command line (e.g.
``max_grad_norm``, ``num_workers``, schedulers, CSP soft-prompt fields).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="Unified CSP vocab post-training launcher for {ijepa, vjepa, dinov3}."
    )
    p.add_argument(
        "--vision-backbone",
        required=True,
        choices=("ijepa", "vjepa", "dinov3"),
        help="Vision backbone alias.",
    )
    p.add_argument("--dataset", default="cspref_mit_states")
    p.add_argument(
        "--seed-list",
        default="",
        help="Comma-separated seeds.",
    )
    p.add_argument(
        "--seed",
        default="",
        help="Single seed shortcut (overrides --seed-list when provided).",
    )
    p.add_argument(
        "--hyperparams-file",
        default=os.environ.get("HYPERPARAMS_FILE", "hyperparameters.json"),
    )
    p.add_argument(
        "--base-checkpoint",
        default="",
        help="Optional base checkpoint path (overrides base_checkpoint / csp_base_checkpoint in JSON). Supports '{seed}' template.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_known_args()


def _load_hparams(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Hyperparameters file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _merged_hparams(cfg: dict[str, Any], backbone: str, dataset: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for src in (
        cfg.get("defaults", {}),
        cfg.get("models", {}).get(backbone, {}),
        cfg.get("datasets", {}).get(dataset, {}),
    ):
        if isinstance(src, dict):
            merged.update(src)
    md = cfg.get("model_dataset", {})
    if isinstance(md, dict):
        by_model = md.get(backbone, {})
        if isinstance(by_model, dict) and isinstance(by_model.get(dataset), dict):
            merged.update(by_model[dataset])
    return merged


def _require_hparam(merged: dict[str, Any], key: str, cast: Any) -> Any:
    if key not in merged or merged[key] is None:
        raise KeyError(
            f"Missing required hyperparameter '{key}' in hyperparameters.json for "
            f"the selected model/dataset combination."
        )
    return cast(merged[key])


def _seeds_from_hparam_value(v: Any) -> list[str]:
    if isinstance(v, (list, tuple)):
        out = [str(x).strip() for x in v if str(x).strip()]
    else:
        out = [x.strip() for x in str(v).split(",") if x.strip()]
    if not out:
        raise ValueError("seed_list in hyperparameters.json is empty.")
    return out


def _resolve_base_checkpoint(template: str, seed: str) -> str:
    t = str(template or "").strip()
    if not t:
        return ""
    return t.format(seed=seed) if "{seed}" in t else t


def main() -> None:
    args, extra_args = _parse_args()

    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)
    (repo_root / "checkpoints").mkdir(exist_ok=True)
    dataset_tag = args.dataset.replace("-", "_")

    hp_cfg = _load_hparams(repo_root / args.hyperparams_file)
    merged = _merged_hparams(hp_cfg, args.vision_backbone, args.dataset)

    # Strict mode: all training hyperparameters come from hyperparameters.json.
    epochs = _require_hparam(merged, "epochs", int)
    batch_size = _require_hparam(merged, "batch_size", int)
    lr = _require_hparam(merged, "lr", float)
    weight_decay = _require_hparam(merged, "weight_decay", float)
    max_grad_norm = _require_hparam(merged, "max_grad_norm", float)
    fusion_type = _require_hparam(merged, "fusion_type", str)

    if str(args.seed).strip():
        seeds = [str(args.seed).strip()]
    elif str(args.seed_list).strip():
        seeds = [s.strip() for s in str(args.seed_list).split(",") if s.strip()]
    else:
        seeds = _seeds_from_hparam_value(_require_hparam(merged, "seed_list", lambda x: x))

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_tag = args.vision_backbone.replace("-", "")

    print("========== CSP Vocab Run Configuration ==========", flush=True)
    print(f"vision_backbone: {args.vision_backbone}", flush=True)
    print(f"dataset: {args.dataset}", flush=True)
    print(f"dataset_tag: {dataset_tag}", flush=True)
    print(f"hyperparams_file: {args.hyperparams_file}", flush=True)
    print(f"epochs: {epochs}", flush=True)
    print(f"batch_size: {batch_size}", flush=True)
    print(f"lr: {lr}", flush=True)
    print(f"weight_decay: {weight_decay}", flush=True)
    print(f"max_grad_norm: {max_grad_norm}", flush=True)
    print(f"fusion_type: {fusion_type}", flush=True)
    print(f"seed_list: {','.join(seeds)}", flush=True)
    base_ckpt_tpl = (args.base_checkpoint or "").strip() or str(
        merged.get("base_checkpoint") or merged.get("csp_base_checkpoint") or ""
    ).strip()
    print(f"base_checkpoint_template: {base_ckpt_tpl or '<none>'}", flush=True)
    print(f"dry_run: {args.dry_run}", flush=True)
    if extra_args:
        print(f"extra_args: {' '.join(extra_args)}", flush=True)
    else:
        print("extra_args: <none>", flush=True)
    print(f"checkpoints_dir: {repo_root / 'checkpoints'}", flush=True)
    print(f"timestamp: {ts}", flush=True)
    print("=================================================", flush=True)

    for seed in seeds:
        ckpt = f"checkpoints/csp_vocab_{model_tag}_{dataset_tag}_s{seed}_{ts}.pt"
        train_cmd = [
            sys.executable,
            "csp_vocab_train.py",
            "--vision-backbone",
            args.vision_backbone,
            "--dataset",
            args.dataset,
            "--fusion-type",
            fusion_type,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--lr",
            str(lr),
            "--weight-decay",
            str(weight_decay),
            "--seed",
            str(seed),
            "--hyperparams-file",
            args.hyperparams_file,
            "--save",
            ckpt,
        ]
        base_ckpt = _resolve_base_checkpoint(base_ckpt_tpl, seed)
        if base_ckpt:
            train_cmd.extend(["--base-checkpoint", base_ckpt])
        if extra_args:
            train_cmd.extend(extra_args)

        print(f"Running CSP vocab train command (seed={seed}):", flush=True)
        print(" ".join(train_cmd), flush=True)
        print(f"CSP vocab checkpoint will be saved to: {ckpt}", flush=True)

        if args.dry_run:
            continue
        subprocess.run(train_cmd, check=True)


if __name__ == "__main__":
    main()
