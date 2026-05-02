#!/usr/bin/env python3
"""
Unified launcher for CSP-reference datasets + clip-style fusion across vision backbones.

Replaces:
- train_dinov3_mit_states_clipstyle.sh
- train_ijepa_mit_states_clipstyle.sh
- train_vjepa_mit_states_clipstyle.sh
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
        description="Unified training launcher for {ijepa, v-jepa, dino-v3} on cspref_mit_states with clip_similarity fusion."
    )
    p.add_argument(
        "--vision-backbone",
        required=True,
        choices=("ijepa", "v-jepa", "dino-v3"),
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
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument(
        "--wandb-log-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable W&B image logging (default: on).",
    )
    p.add_argument(
        "--wandb-max-images",
        type=int,
        default=8,
        help="Number of images shown per W&B panel (default: 8).",
    )
    p.add_argument("--finetune-clip-text", action="store_true")
    p.add_argument(
        "--plot-metric",
        default="top1",
        help="Metric key in eval JSON for seed-performance line plot (e.g. top1, top5, auc_csp_style).",
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


def _plot_seed_performance(
    records: list[dict[str, Any]],
    *,
    metric_key: str,
    out_path: Path,
) -> None:
    if not records:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skip seed-performance plot.", file=sys.stderr)
        return

    sorted_records = sorted(records, key=lambda r: int(r["seed"]))
    x = [int(r["seed"]) for r in sorted_records]
    y_val = [float(r["val"][metric_key]) for r in sorted_records]
    y_test = [float(r["test"][metric_key]) for r in sorted_records]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y_val, marker="o", label=f"val/{metric_key}")
    plt.plot(x, y_test, marker="o", label=f"test/{metric_key}")
    plt.xlabel("Seed")
    plt.ylabel(metric_key)
    plt.title(f"Eval performance across seeds ({metric_key})")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Wrote seed-performance plot to: {out_path}")


def main() -> None:
    args, extra_args = _parse_args()

    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)
    (repo_root / "checkpoints").mkdir(exist_ok=True)
    dataset_tag = args.dataset.replace("-", "_")
    results_dir = repo_root / "results" / f"{dataset_tag}_clipstyle"
    results_dir.mkdir(parents=True, exist_ok=True)

    hp_cfg = _load_hparams(repo_root / args.hyperparams_file)
    merged = _merged_hparams(hp_cfg, args.vision_backbone, args.dataset)

    # Strict mode: all training hyperparameters come from hyperparameters.json.
    epochs = _require_hparam(merged, "epochs", int)
    batch_size = _require_hparam(merged, "batch_size", int)
    lr = _require_hparam(merged, "lr", float)
    weight_decay = _require_hparam(merged, "weight_decay", float)
    fusion_type = _require_hparam(merged, "fusion_type", str)

    if str(args.seed).strip():
        seeds = [str(args.seed).strip()]
    elif str(args.seed_list).strip():
        seeds = [s.strip() for s in str(args.seed_list).split(",") if s.strip()]
    else:
        seeds = _seeds_from_hparam_value(_require_hparam(merged, "seed_list", lambda x: x))

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_tag = args.vision_backbone.replace("-", "")
    eval_records: list[dict[str, Any]] = []

    # Print a full run configuration snapshot before training starts.
    print("========== Run Configuration ==========")
    print(f"vision_backbone: {args.vision_backbone}")
    print(f"dataset: {args.dataset}")
    print(f"dataset_tag: {dataset_tag}")
    print(f"hyperparams_file: {args.hyperparams_file}")
    print(f"epochs: {epochs}")
    print(f"batch_size: {batch_size}")
    print(f"lr: {lr}")
    print(f"weight_decay: {weight_decay}")
    print(f"fusion_type: {fusion_type}")
    print(f"seed_list: {','.join(seeds)}")
    print(f"no_wandb: {args.no_wandb}")
    print(f"wandb_log_images: {args.wandb_log_images}")
    print(f"wandb_max_images: {args.wandb_max_images}")
    print(f"finetune_clip_text: {args.finetune_clip_text}")
    print(f"plot_metric: {args.plot_metric}")
    print(f"dry_run: {args.dry_run}")
    if extra_args:
        print(f"extra_args: {' '.join(extra_args)}")
    else:
        print("extra_args: <none>")
    print(f"checkpoints_dir: {repo_root / 'checkpoints'}")
    print(f"results_dir: {results_dir}")
    print(f"timestamp: {ts}")
    print("======================================")

    for seed in seeds:
        ckpt = f"checkpoints/{model_tag}_{dataset_tag}_clipstyle_s{seed}_{ts}.pt"
        train_cmd = [
            sys.executable,
            "text_cond_train.py",
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
        if args.no_wandb:
            train_cmd.append("--no-wandb")
        if args.wandb_log_images:
            train_cmd.extend(["--wandb-log-images", "--wandb-max-images", str(args.wandb_max_images)])
        if args.finetune_clip_text:
            train_cmd.append("--finetune-clip-text")
        if extra_args:
            train_cmd.extend(extra_args)

        base_eval_cmd = [
            sys.executable,
            "text_cond_train.py",
            "--eval-only",
            "--vision-backbone",
            args.vision_backbone,
            "--dataset",
            args.dataset,
            "--seed",
            str(seed),
            "--hyperparams-file",
            args.hyperparams_file,
            "--checkpoint",
            ckpt,
        ]
        if args.no_wandb:
            base_eval_cmd.append("--no-wandb")
        if args.wandb_log_images:
            base_eval_cmd.extend(["--wandb-log-images", "--wandb-max-images", str(args.wandb_max_images)])
        val_json = results_dir / f"{model_tag}_s{seed}_{ts}_val.json"
        test_json = results_dir / f"{model_tag}_s{seed}_{ts}_test.json"
        eval_val_cmd = base_eval_cmd + [
            "--eval-split",
            "val",
            "--experiment-tag",
            f"{model_tag}-s{seed}-{ts}",
            "--metrics-json",
            str(val_json),
        ]
        eval_test_cmd = base_eval_cmd + [
            "--eval-split",
            "test",
            "--experiment-tag",
            f"{model_tag}-s{seed}-{ts}",
            "--metrics-json",
            str(test_json),
        ]

        print(f"Running train command (seed={seed}):")
        print(" ".join(train_cmd))
        print(f"Checkpoint will be saved to: {ckpt}")
        print(f"Running eval command (val):")
        print(" ".join(eval_val_cmd))
        print(f"Running eval command (test):")
        print(" ".join(eval_test_cmd))

        if args.dry_run:
            continue

        subprocess.run(train_cmd, check=True)
        subprocess.run(eval_val_cmd, check=True)
        subprocess.run(eval_test_cmd, check=True)

        if val_json.exists() and test_json.exists():
            val_data = json.loads(val_json.read_text(encoding="utf-8"))
            test_data = json.loads(test_json.read_text(encoding="utf-8"))
            if args.plot_metric not in val_data or args.plot_metric not in test_data:
                raise KeyError(
                    f"plot metric '{args.plot_metric}' not found in eval json; "
                    f"available keys include: {sorted(val_data.keys())}"
                )
            eval_records.append({"seed": int(seed), "val": val_data, "test": test_data})

    if (not args.dry_run) and eval_records:
        plot_path = (
            results_dir / f"{model_tag}_{dataset_tag}_{ts}_seed_curve_{args.plot_metric}.png"
        )
        _plot_seed_performance(eval_records, metric_key=args.plot_metric, out_path=plot_path)


if __name__ == "__main__":
    main()
