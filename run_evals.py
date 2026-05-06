#!/usr/bin/env python3
"""
run val + test ``--eval-only`` for every checkpoint under a path list or directory.
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


_BACKBONE = r"(?:ijepa|vjepa|dinov3)"
# Saved by run_text_cond_train: {model_tag}_{dataset_tag}_clipstyle_s{seed}_{ts}
_RE_CLIPSTYLE_STRICT = re.compile(
    rf"^({_BACKBONE})_(.+)_clipstyle_s(\d+)_([0-9]{{8}}-[0-9]{{6}})$"
)
_RE_CLIPSTYLE_LOOSE = re.compile(rf"^({_BACKBONE})_(.+)_clipstyle_s(\d+)_(.+)$")
# Saved by run_csp_vocab_train: csp_vocab_{model_tag}_{dataset_tag}_s{seed}_{ts}
_RE_CSPV_STRICT = re.compile(
    rf"^csp_vocab_({_BACKBONE})_(.+)_s(\d+)_([0-9]{{8}}-[0-9]{{6}})$"
)
_RE_CSPV_LOOSE = re.compile(rf"^csp_vocab_({_BACKBONE})_(.+)_s(\d+)_(.+)$")


def _parse_run_checkpoint_stem(path: Path, kind: str) -> dict[str, str | int] | None:
    """
    Parse ``run_*_train.py`` checkpoint stem into vision_backbone / dataset / seed.

    ``kind`` is ``text_cond`` or ``csp_vocab`` from :func:`_detect_kind` (file contents).
    """
    stem = path.stem
    if kind == "csp_vocab":
        m = _RE_CSPV_STRICT.match(stem) or _RE_CSPV_LOOSE.match(stem)
    else:
        m = _RE_CLIPSTYLE_STRICT.match(stem) or _RE_CLIPSTYLE_LOOSE.match(stem)
    if not m:
        return None
    return {
        "vision_backbone": m.group(1),
        "dataset": m.group(2),
        "seed": int(m.group(3)),
    }


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


def _run_summarize(repo_root: Path, metrics_dir: Path) -> None:
    """Run ``summarize_eval_metrics.py`` on one directory of ``*.json`` metrics."""
    metrics_dir = metrics_dir.resolve()
    json_files = list(metrics_dir.glob("*.json"))
    if not json_files:
        print(f"[summarize] skip (no *.json in {metrics_dir})", flush=True)
        return
    summarize_script = repo_root / "summarize_eval_metrics.py"
    if not summarize_script.is_file():
        raise FileNotFoundError(f"Missing {summarize_script}")
    cmd = [
        sys.executable,
        str(summarize_script),
        "--metrics-dir",
        str(metrics_dir),
        "--raw-csv",
        str(metrics_dir / "eval_metrics_raw.csv"),
        "--summary-csv",
        str(metrics_dir / "eval_metrics_summary.csv"),
        "--summary-json",
        str(metrics_dir / "eval_metrics_summary.json"),
        "--plot-dir",
        str(metrics_dir / "eval_plots"),
    ]
    print(f"\n[summarize] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(repo_root))


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
    p.add_argument(
        "--no-summarize",
        action="store_true",
        help="Do not run summarize_eval_metrics.py after eval (default: run it per results dir).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.paths:
        print("usage: run_evals.py <checkpoint.pt|dir> ...", file=sys.stderr)
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
    metrics_dirs: set[Path] = set()

    for ck in files:
        kind = _detect_kind(ck)
        script = "csp_vocab_train.py" if kind == "csp_vocab" else "text_cond_train.py"

        bd = _bundle_defaults(ck) if kind == "csp_vocab" else {}
        parsed = _parse_run_checkpoint_stem(ck, kind)

        dataset = (args.dataset or "").strip() or str(bd.get("dataset") or "").strip()
        if not dataset and parsed is not None:
            dataset = str(parsed["dataset"])

        backbone = (args.vision_backbone or "").strip() or str(
            bd.get("vision_backbone") or ""
        ).strip()
        if not backbone and parsed is not None:
            backbone = str(parsed["vision_backbone"])

        seed: int | None
        if args.seed >= 0:
            seed = int(args.seed)
        elif bd.get("seed") is not None:
            seed = int(bd["seed"])
        elif parsed is not None:
            seed = int(parsed["seed"])
        else:
            seed = _parse_seed_from_name(ck)

        if not dataset:
            raise SystemExit(f"{ck}: set --dataset (not in bundle / launcher filename).")
        if not backbone:
            raise SystemExit(f"{ck}: set --vision-backbone (not in bundle / launcher filename).")
        if seed is None:
            raise SystemExit(
                f"{ck}: could not resolve seed; pass --seed or use a launcher filename "
                f"(e.g. …_clipstyle_s42_* or csp_vocab_*_s42_*)."
            )

        dataset_tag = dataset.replace("-", "_")
        kind_tag = "csp_vocab" if kind == "csp_vocab" else "clipstyle"
        if args.out_dir is not None:
            results_dir = Path(args.out_dir)
        else:
            results_dir = repo_root / "results" / f"rerun_evals_{dataset_tag}_{kind_tag}"
        results_dir.mkdir(parents=True, exist_ok=True)
        metrics_dirs.add(results_dir.resolve())

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

    if not args.dry_run and not args.no_summarize:
        for d in sorted(metrics_dirs, key=str):
            _run_summarize(repo_root, d)


if __name__ == "__main__":
    main()
