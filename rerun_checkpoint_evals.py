#!/usr/bin/env python3
"""
Re-run val + test ``--eval-only`` for every checkpoint under a path list or directory.

- Files whose load result is a dict containing ``csp_vocab`` and ``meta`` →
  ``csp_vocab_train.py --eval-only``.
- Otherwise → ``text_cond_train.py --eval-only`` (plain ``state_dict``).

For CSP vocab bundles, ``--dataset``, ``--vision-backbone``, and ``--seed`` default from
the saved ``bundle['args']`` when omitted. For text-cond checkpoints you should pass them
explicitly (or only ``--seed`` if parseable from the filename ``*_s<seed>_*.pt``).

Examples::

    uv run python rerun_checkpoint_evals.py checkpoints/*.pt \\
      --vision-backbone ijepa --dataset cspref_mit_states --seed 42

    uv run python rerun_checkpoint_evals.py checkpoints/ --glob 'csp_vocab_*.pt'

    uv run python rerun_checkpoint_evals.py checkpoints/ --recurse
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


def _detect_kind(path: Path) -> str:
    obj: Any = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "csp_vocab" in obj and "meta" in obj:
        return "csp_vocab"
    return "text_cond"


def _parse_seed_from_name(path: Path) -> int | None:
    m = re.search(r"_s(\d+)_", path.name)
    return int(m.group(1)) if m else None


def _bundle_defaults(path: Path) -> dict[str, Any]:
    obj: Any = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict) or "args" not in obj:
        return {}
    a = obj.get("args")
    return dict(a) if isinstance(a, dict) else {}


def _collect_files(paths: list[Path], *, glob_pat: str | None, recurse: bool) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = raw.resolve()
        if glob_pat:
            if not p.is_dir():
                raise ValueError(f"--glob requires a directory path, got: {p}")
            out.extend(sorted(p.glob(glob_pat)))
            continue
        if p.is_file() and p.suffix.lower() in {".pt", ".pth"}:
            out.append(p)
        elif p.is_dir():
            if recurse:
                out.extend(sorted(p.rglob("*.pt")))
                out.extend(sorted(p.rglob("*.pth")))
            else:
                out.extend(sorted(p.glob("*.pt")))
                out.extend(sorted(p.glob("*.pth")))
        else:
            raise FileNotFoundError(f"Not a checkpoint file or directory: {p}")
    # Stable de-dupe while preserving order
    seen: set[str] = set()
    uniq: list[Path] = []
    for f in out:
        k = str(f.resolve())
        if k not in seen:
            seen.add(k)
            uniq.append(f.resolve())
    return sorted(uniq)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("Examples::", 1)[0].strip())
    p.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Checkpoint .pt/.pth files and/or directories to scan.",
    )
    p.add_argument(
        "--glob",
        dest="glob_pat",
        default="",
        help="If set, each path must be a directory; append this glob (e.g. 'csp_vocab_*.pt').",
    )
    p.add_argument(
        "--recurse",
        action="store_true",
        help="When a path is a directory, search recursively for *.pt / *.pth",
    )
    p.add_argument("--vision-backbone", default="", choices=("ijepa", "vjepa", "dinov3", ""))
    p.add_argument("--dataset", default="")
    p.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Global default seed when not inferrable (-1 = try filename / bundle only).",
    )
    p.add_argument(
        "--hyperparams-file",
        default=os.environ.get("HYPERPARAMS_FILE", "hyperparameters.json"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Metrics JSON root (default: results/rerun_evals/<dataset_tag>_<kind>/).",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B image logging during eval (default: pass --no-wandb to eval).",
    )
    p.add_argument("--wandb-max-images", type=int, default=8)
    p.add_argument(
        "--forward",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra argv token for each eval (repeatable), e.g. --forward --finetune-clip-text",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.paths:
        print("usage: rerun_checkpoint_evals.py <checkpoint.pt|dir> ...", file=sys.stderr)
        sys.exit(2)

    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    files = _collect_files(
        [Path(p) for p in args.paths],
        glob_pat=(args.glob_pat or "").strip() or None,
        recurse=bool(args.recurse),
    )
    if not files:
        print("No checkpoint files matched.", file=sys.stderr)
        sys.exit(1)

    use_wandb = bool(args.wandb)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    for ck in files:
        kind = _detect_kind(ck)
        script = "csp_vocab_train.py" if kind == "csp_vocab" else "text_cond_train.py"

        bd = _bundle_defaults(ck) if kind == "csp_vocab" else {}
        dataset = (args.dataset or "").strip() or str(bd.get("dataset") or "").strip()
        backbone = (args.vision_backbone or "").strip() or str(
            bd.get("vision_backbone") or ""
        ).strip()
        seed: int | None
        if args.seed >= 0:
            seed = int(args.seed)
        elif bd.get("seed") is not None:
            seed = int(bd["seed"])
        else:
            seed = _parse_seed_from_name(ck)

        if not dataset:
            raise SystemExit(f"{ck}: set --dataset (not in bundle args).")
        if not backbone:
            raise SystemExit(f"{ck}: set --vision-backbone (not in bundle args).")
        if seed is None:
            raise SystemExit(
                f"{ck}: could not resolve seed; pass --seed or use a filename like *_s42_*"
            )

        dataset_tag = dataset.replace("-", "_")
        kind_tag = "csp_vocab" if kind == "csp_vocab" else "clipstyle"
        if args.out_dir is not None:
            results_dir = Path(args.out_dir)
        else:
            results_dir = repo_root / "results" / f"rerun_evals_{dataset_tag}_{kind_tag}"
        results_dir.mkdir(parents=True, exist_ok=True)

        base = [
            sys.executable,
            str(repo_root / script),
            "--eval-only",
            "--vision-backbone",
            backbone,
            "--dataset",
            dataset,
            "--seed",
            str(seed),
            "--hyperparams-file",
            str(args.hyperparams_file),
            "--checkpoint",
            str(ck),
        ]
        if use_wandb:
            base.extend(["--wandb-log-images", "--wandb-max-images", str(args.wandb_max_images)])
        else:
            base.append("--no-wandb")

        base.extend(x for x in args.forward if x)

        stem = ck.stem
        tag = f"rerun-{stem}-{ts}"
        val_json = results_dir / f"{stem}_val.json"
        test_json = results_dir / f"{stem}_test.json"

        val_cmd = base + [
            "--eval-split",
            "val",
            "--experiment-tag",
            tag,
            "--metrics-json",
            str(val_json),
        ]
        test_cmd = base + [
            "--eval-split",
            "test",
            "--experiment-tag",
            tag,
            "--metrics-json",
            str(test_json),
        ]

        print(f"\n=== {ck.name}  ({kind}) ===", flush=True)
        print(" ".join(val_cmd), flush=True)
        print(" ".join(test_cmd), flush=True)

        if args.dry_run:
            continue

        subprocess.run(val_cmd, check=True, cwd=str(repo_root))
        subprocess.run(test_cmd, check=True, cwd=str(repo_root))


if __name__ == "__main__":
    main()
