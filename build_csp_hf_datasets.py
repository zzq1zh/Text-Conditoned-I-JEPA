#!/usr/bin/env python3
"""
Assemble the ``two_object``, ``rel``, and ``single_object`` CLEVR-style releases into
Hugging Face :class:`datasets.DatasetDict` with splits ``train`` / ``val`` / ``test``
and push three separate Hub dataset repos: ``<hub_user>/csp_two_object``, ``.../csp_rel``,
``.../csp_single_object``.

- **On-disk** ``gen`` (``gen.csv`` / ``gen.json``) is mapped to the Hub split **``test``**
  (standard benchmark naming); ``val`` stays ``val``.

- **Images** must sit next to the manifest files; paths are checked before upload.

Requires ``HF_TOKEN`` (or ``huggingface-cli login``) and ``datasets`` / ``Pillow``.
Optional: ``pip install pyarrow`` for faster local saves. Inline comments in English.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import project_env

project_env.load_project_env()


def _read_csv_table(csv_path: Path) -> list[dict[str, str]]:
    """Read a CSV; drop an unnamed first index column if present (leading comma in header)."""
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        rows: list[dict[str, str]] = []
        for raw in reader:
            row = {k.strip(): (v or "").strip() for k, v in raw.items() if k and str(k).strip()}
            if row:
                rows.append(row)
    return rows


def _build_csv_split(
    root: Path,
    split: str,
    rows: list[dict[str, Any]],
    image_path_fn: Callable[[str, str], Path],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    missing = 0
    for rec in rows:
        fn = str(rec.get("file_name", "")).strip()
        if not fn:
            continue
        p = image_path_fn(split, fn)
        if not p.is_file():
            missing += 1
            continue
        out.append(
            {
                "image": str(p.resolve()),
                "file_name": fn,
                "pos": str(rec.get("pos", "")),
                "neg_0": str(rec.get("neg_0", "")),
                "neg_1": str(rec.get("neg_1", "")),
                "neg_2": str(rec.get("neg_2", "")),
                "neg_3": str(rec.get("neg_3", "")),
            }
        )
    if missing:
        print(
            f"  [{split}] skipped {missing} rows with missing image files under {root}",
            file=sys.stderr,
            flush=True,
        )
    return out


def _build_rel_split(
    root: Path,
    split: str,  # train | val | test (disk folder name train|val|gen)
    data: dict[str, Any],
) -> list[dict[str, Any]]:
    folder = "gen" if split == "test" else split
    out: list[dict[str, Any]] = []
    missing = 0
    for fname, v in data.items():
        p = root / "images" / folder / fname
        if not p.is_file():
            missing += 1
            continue
        pos = v.get("pos") or []
        neg = v.get("neg") or []
        if not isinstance(pos, list) or not isinstance(neg, list):
            continue
        out.append(
            {
                "image": str(p.resolve()),
                "file_name": str(fname),
                "positives": [str(x) for x in pos],
                "negatives": [str(x) for x in neg],
            }
        )
    if missing:
        print(
            f"  [rel {split}] skipped {missing} keys with missing image files",
            file=sys.stderr,
            flush=True,
        )
    return out


def _dataset_from_csv_rows(rows: list[dict[str, Any]]):
    from datasets import Dataset, Features, Image, Value

    if not rows:
        return Dataset.from_dict({})
    features = Features(
        {
            "image": Image(),
            "file_name": Value("string"),
            "pos": Value("string"),
            "neg_0": Value("string"),
            "neg_1": Value("string"),
            "neg_2": Value("string"),
            "neg_3": Value("string"),
        }
    )
    batch = {k: [r[k] for r in rows] for k in rows[0]}
    return Dataset.from_dict(batch, features=features)


def _dataset_from_rel_rows(rows: list[dict[str, Any]]):
    from datasets import Dataset, Features, Image, List, Value

    if not rows:
        return Dataset.from_dict({})
    features = Features(
        {
            "image": Image(),
            "file_name": Value("string"),
            "positives": List(Value("string")),
            "negatives": List(Value("string")),
        }
    )
    batch = {k: [r[k] for r in rows] for k in rows[0]}
    return Dataset.from_dict(batch, features=features)


def build_two_object(root: Path) -> "DatasetDict":
    from datasets import DatasetDict

    def img_path(split: str, fn: str) -> Path:
        d = "gen" if split == "test" else split
        return root / "images" / d / fn

    split_to_csv = {"train": "train.csv", "val": "val.csv", "test": "gen.csv"}
    splits: dict[str, Any] = {}
    for split, csv_name in split_to_csv.items():
        path = root / csv_name
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")
        rows = _read_csv_table(path)
        if not rows:
            raise RuntimeError(f"Empty {path}")
        built = _build_csv_split(root, split, rows, img_path)
        if not built:
            raise RuntimeError(f"No valid rows for split {split} in {root}")
        splits[split] = _dataset_from_csv_rows(built)
    return DatasetDict(splits)


def build_single_object(root: Path) -> "DatasetDict":
    from datasets import DatasetDict

    def img_path(_split: str, fn: str) -> Path:
        return root / "images" / fn

    split_to_csv = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}
    splits: dict[str, Any] = {}
    for split, csv_name in split_to_csv.items():
        path = root / csv_name
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")
        rows = _read_csv_table(path)
        if not rows:
            raise RuntimeError(f"Empty {path}")
        built = _build_csv_split(root, split, rows, img_path)
        if not built:
            raise RuntimeError(f"No valid rows for split {split} in {root}")
        splits[split] = _dataset_from_csv_rows(built)
    return DatasetDict(splits)


def build_rel(root: Path) -> "DatasetDict":
    from datasets import DatasetDict

    split_to_file = {
        "train": "train.json",
        "val": "val.json",
        "test": "gen.json",
    }
    splits: dict[str, Any] = {}
    for split, jname in split_to_file.items():
        p = root / jname
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p}")
        with p.open("r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        if not data:
            raise RuntimeError(f"Empty {p}")
        rows = _build_rel_split(root, split, data)
        if not rows:
            raise RuntimeError(f"No valid rows for split {split} in {root}")
        splits[split] = _dataset_from_rel_rows(rows)
    return DatasetDict(splits)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build csp_two_object / csp_rel / csp_single_object HF datasets and optionally push"
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root (default: directory containing this script)",
    )
    p.add_argument(
        "--hub-user",
        type=str,
        default=os.environ.get("HF_HUB_USER", "").strip(),
        help="Hub namespace (user or org) for <hub_user>/csp_* (or set HF_HUB_USER)",
    )
    p.add_argument(
        "--push",
        action="store_true",
        help="Call push_to_hub for each DatasetDict (requires token)",
    )
    p.add_argument(
        "--public",
        action="store_true",
        help="Public Hub datasets (default: private)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="If set, save each DatasetDict to disk under this directory (no push needed)",
    )
    p.add_argument(
        "--only",
        choices=("two_object", "rel", "single_object", "all"),
        default="all",
        help="Build only one dataset, or all",
    )
    args = p.parse_args()

    root = args.data_root.resolve()
    p_two = root / "two_object" / "two_object"
    p_rel = root / "rel" / "rel"
    p_one = root / "single_object" / "single_object"

    builders: dict[str, Any] = {
        "two_object": (p_two, build_two_object),
        "rel": (p_rel, build_rel),
        "single_object": (p_one, build_single_object),
    }

    if not args.hub_user and args.push:
        print(
            "Pushing needs --hub-user (or HF_HUB_USER in the environment).",
            file=sys.stderr,
        )
        raise SystemExit(2)

    from huggingface_hub import get_token

    token: str | None = None
    if args.push:
        token = get_token()
        if not token:
            print("No HF token; set HF_TOKEN or run huggingface-cli login", file=sys.stderr)
            raise SystemExit(2)

    to_run = list(builders) if args.only == "all" else [args.only]
    for key in to_run:
        bpath, fn = builders[key]
        if not bpath.is_dir():
            print(f"Skip {key}: not found {bpath}", file=sys.stderr)
            continue
        print(f"Building {key} from {bpath} ...", flush=True)
        dsd = fn(bpath)
        for sp, s in dsd.items():
            print(f"  split {sp}: {len(s)} rows", flush=True)
        if args.out_dir is not None:
            dest = (args.out_dir / f"csp_{key}").resolve()
            dest.mkdir(parents=True, exist_ok=True)
            dsd.save_to_disk(str(dest))
            print(f"  saved to {dest}", flush=True)
        if args.push and args.hub_user:
            repo_id = f"{args.hub_user}/csp_{key}"
            private = not args.public
            print(f"  pushing to {repo_id} (private={private}) ...", flush=True)
            dsd.push_to_hub(
                repo_id,
                private=private,
                token=token,
            )
            print(f"  done: https://huggingface.co/datasets/{repo_id}", flush=True)


if __name__ == "__main__":
    main()
