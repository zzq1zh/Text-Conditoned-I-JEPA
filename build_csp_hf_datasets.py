#!/usr/bin/env python3
"""
Build and publish Hugging Face dataset repos for compositional tasks in two families:

1) ``clevr`` — CLEVR-style CSP releases (``two_object`` / ``rel`` / ``single_object``).
   On-disk ``gen`` maps to Hub split ``test``; images must sit next to manifests.

2) ``reference`` — CSP reference benchmarks (MIT-States, UT-Zappos, C-GQA), following
   the data flow described in https://github.com/BatsResearch/csp

Requires ``datasets`` and ``Pillow``; Hub push uses ``huggingface-cli login`` (or an explicit ``--token`` where supported).
Optional: ``pip install pyarrow`` for faster local saves.
"""


from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
import shutil
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
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


def run_clevr(args: argparse.Namespace) -> None:
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
            print("No Hugging Face token; run huggingface-cli login", file=sys.stderr)
            raise SystemExit(2)

    to_run = list(builders) if args.clevr_only == "all" else [args.clevr_only]
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




@dataclass(frozen=True)
class DatasetPaths:
    name: str
    root: Path
    split_name: str = "compositional-split-natural"


DATASET_ROOTS: dict[str, str] = {
    "mit-states": "mit-states",
    "ut-zappos": "ut-zappos",
    "cgqa": "cgqa",
}


def _run(cmd: str, cwd: Path) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), shell=True, check=True)


def _extract_zip_all(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def _extract_zip_prefix(zip_path: Path, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.startswith(prefix):
                continue
            zf.extract(name, out_dir)


def _extract_targz_all(tar_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, mode="r:gz") as tf:
        tf.extractall(out_dir)


def download_raw_data(data_dir: Path) -> None:
    """Reproduce the download commands from CSP's download_data.sh."""
    data_dir.mkdir(parents=True, exist_ok=True)
    _run(
        'wget -c "http://wednesday.csail.mit.edu/joseph_result/state_and_transformation/release_dataset.zip" -O mitstates.zip',
        data_dir,
    )
    _run(
        'wget -c "http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images.zip" -O utzap.zip',
        data_dir,
    )
    _run(
        'wget -c "https://senthilpurushwalkam.com/publications/compositional/compositional_split_natural.tar.gz" -O compositional_split_natural.tar.gz',
        data_dir,
    )
    _run(
        'wget -c "https://huggingface.co/datasets/nihalnayak/cgqa/resolve/main/cgqa.zip" -O cgqa.zip',
        data_dir,
    )


def prepare_raw_layout(data_dir: Path) -> None:
    """Reproduce extraction/reorganization logic from CSP scripts."""
    # MIT-States
    mit_root = data_dir / "mit-states"
    mit_root.mkdir(parents=True, exist_ok=True)
    mit_images = mit_root / "images"
    if not mit_images.exists() or not any(mit_images.iterdir()):
        _extract_zip_prefix(data_dir / "mitstates.zip", mit_root, "release_dataset/images/")
        mit_images.mkdir(parents=True, exist_ok=True)
        src_images = mit_root / "release_dataset" / "images"
        if src_images.exists():
            for item in src_images.iterdir():
                dst = mit_images / item.name
                if dst.exists():
                    if dst.is_dir():
                        shutil.rmtree(dst)
                    else:
                        dst.unlink()
                shutil.move(str(item), str(dst))
            shutil.rmtree(mit_root / "release_dataset", ignore_errors=True)

    # Replace spaces with underscores for top-level MIT folders.
    for p in (data_dir / "mit-states" / "images").glob("*"):
        if " " in p.name:
            p.rename(p.with_name(p.name.replace(" ", "_")))

    # UT-Zappos (keep original _images layout; no expensive full copy to images/attr_obj)
    ut_root = data_dir / "ut-zap50k"
    ut_root.mkdir(parents=True, exist_ok=True)
    ut_images = ut_root / "_images"
    if not ut_images.exists() or not any(ut_images.iterdir()):
        _extract_zip_all(data_dir / "utzap.zip", ut_root)
        ut_images.mkdir(parents=True, exist_ok=True)
        src_utz = ut_root / "ut-zap50k-images"
        if src_utz.exists():
            for item in src_utz.iterdir():
                dst = ut_images / item.name
                if dst.exists():
                    if dst.is_dir():
                        shutil.rmtree(dst)
                    else:
                        dst.unlink()
                shutil.move(str(item), str(dst))
            shutil.rmtree(src_utz, ignore_errors=True)

    # C-GQA and split files
    cgqa_root = data_dir / "cgqa"
    if not cgqa_root.exists() or not any(cgqa_root.iterdir()):
        _extract_zip_all(data_dir / "cgqa.zip", data_dir)
    split_anchor = data_dir / "compositional-split-natural"
    if not split_anchor.exists():
        _extract_targz_all(data_dir / "compositional_split_natural.tar.gz", data_dir)

    # Match CSP directory naming.
    src = data_dir / "ut-zap50k"
    dst = data_dir / "ut-zappos"
    if src.exists() and not dst.exists():
        src.rename(dst)


def _parse_pairs_file(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if not path.is_file():
        return pairs
    for line in path.read_text(encoding="utf-8").splitlines():
        t = line.strip().split()
        if len(t) >= 2:
            pairs.append((t[0], t[1]))
    return pairs


def _resolve_split_dir(root: Path, split_name: str) -> Path:
    candidates = [root / split_name, root.parent / split_name]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(f"Cannot find split directory {split_name!r} under {root} or {root.parent}")


def _instance_image_rel(inst: dict[str, Any], ds_name: str) -> str:
    if ds_name == "ut-zappos" and "_image" in inst and inst["_image"]:
        return str(inst["_image"])
    if "image" in inst and inst["image"]:
        return str(inst["image"])
    if "_image" in inst and inst["_image"]:
        return str(inst["_image"])
    return ""


def build_hf_datasetdict(ds_paths: DatasetPaths) -> "Any":
    """Build train/val/test DatasetDict with compositional metadata."""
    from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, Value

    root = ds_paths.root
    split_dir = _resolve_split_dir(root, ds_paths.split_name)

    train_pairs = _parse_pairs_file(split_dir / "train_pairs.txt")
    val_pairs = _parse_pairs_file(split_dir / "val_pairs.txt")
    test_pairs = _parse_pairs_file(split_dir / "test_pairs.txt")
    all_pairs = sorted(set(train_pairs + val_pairs + test_pairs))
    if not all_pairs:
        raise RuntimeError(f"No pairs parsed for {ds_paths.name} from {split_dir}")

    pair_to_idx = {p: i for i, p in enumerate(all_pairs)}
    attrs = sorted({a for a, _ in all_pairs})
    objs = sorted({o for _, o in all_pairs})
    attr_to_idx = {a: i for i, a in enumerate(attrs)}
    obj_to_idx = {o: i for i, o in enumerate(objs)}
    train_pair_set = set(train_pairs)

    md_file = root / f"metadata_{ds_paths.split_name}.t7"
    metadata: list[dict[str, Any]] | None
    if md_file.is_file():
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"Found metadata file at {md_file}, but torch is not installed. "
                "Install torch or remove metadata usage."
            ) from exc
        metadata = torch.load(str(md_file), map_location="cpu")
    else:
        metadata = None

    by_split: dict[str, dict[str, list[Any]]] = {
        "train": {
            "image": [],
            "attr": [],
            "obj": [],
            "pair": [],
            "attr_id": [],
            "obj_id": [],
            "pair_id": [],
            "pair_seen_in_train": [],
            "source_set": [],
            "image_relpath": [],
        },
        "val": {
            "image": [],
            "attr": [],
            "obj": [],
            "pair": [],
            "attr_id": [],
            "obj_id": [],
            "pair_id": [],
            "pair_seen_in_train": [],
            "source_set": [],
            "image_relpath": [],
        },
        "test": {
            "image": [],
            "attr": [],
            "obj": [],
            "pair": [],
            "attr_id": [],
            "obj_id": [],
            "pair_id": [],
            "pair_seen_in_train": [],
            "source_set": [],
            "image_relpath": [],
        },
    }

    valid_splits = {"train", "val", "test"}
    valid_pair_set = set(all_pairs)
    image_roots = [root / "images", root / "_images"]

    def _append_row(
        *,
        settype: str,
        attr: str,
        obj: str,
        pair: tuple[str, str],
        img_path: Path,
        rel: str,
    ) -> None:
        out = by_split[settype]
        out["image"].append(str(img_path.resolve()))
        out["attr"].append(attr)
        out["obj"].append(obj)
        out["pair"].append(f"{attr} {obj}")
        out["attr_id"].append(attr_to_idx[attr])
        out["obj_id"].append(obj_to_idx[obj])
        out["pair_id"].append(pair_to_idx[pair])
        out["pair_seen_in_train"].append(pair in train_pair_set)
        out["source_set"].append(settype)
        out["image_relpath"].append(rel)

    if metadata is not None:
        for inst in metadata:
            attr = str(inst.get("attr", ""))
            obj = str(inst.get("obj", ""))
            settype = str(inst.get("set", "")).lower()
            if attr == "NA" or settype == "na":
                continue
            pair = (attr, obj)
            if pair not in valid_pair_set or settype not in valid_splits:
                continue
            rel = _instance_image_rel(inst, ds_paths.name)
            if not rel:
                continue
            img_path: Path | None = None
            for ir in image_roots:
                cand = ir / rel
                if cand.is_file():
                    img_path = cand
                    break
            if img_path is None:
                continue
            _append_row(
                settype=settype,
                attr=attr,
                obj=obj,
                pair=pair,
                img_path=img_path,
                rel=rel,
            )
    else:
        # Fallback path used by cgqa.zip releases that do not include metadata_*.t7.
        slug_to_pair: dict[str, tuple[str, str]] = {}
        for a, o in all_pairs:
            aa = a.replace(" ", "-")
            oo = o.replace(" ", "-")
            slug_to_pair[f"{aa}-{oo}"] = (a, o)
        train_set = set(train_pairs)
        val_set = set(val_pairs)
        test_set = set(test_pairs)
        img_dir = root / "images"
        if not img_dir.is_dir():
            raise FileNotFoundError(
                f"Missing metadata file ({md_file}) and image directory ({img_dir}) for fallback parse."
            )
        for p in img_dir.iterdir():
            if not p.is_file():
                continue
            rel = p.name
            stem = p.stem
            parts = stem.split("-", 2)
            if len(parts) < 3:
                continue
            slug = parts[2]
            pair = slug_to_pair.get(slug)
            if pair is None:
                continue
            if pair in train_set:
                settype = "train"
            elif pair in val_set:
                settype = "val"
            elif pair in test_set:
                settype = "test"
            else:
                continue
            attr, obj = pair
            _append_row(
                settype=settype,
                attr=attr,
                obj=obj,
                pair=pair,
                img_path=p,
                rel=rel,
            )

    features = Features(
        {
            "image": Image(),
            "attr": Value("string"),
            "obj": Value("string"),
            "pair": Value("string"),
            "attr_id": ClassLabel(names=attrs),
            "obj_id": ClassLabel(names=objs),
            "pair_id": ClassLabel(names=[f"{a} {o}" for a, o in all_pairs]),
            "pair_seen_in_train": Value("bool"),
            "source_set": Value("string"),
            "image_relpath": Value("string"),
        }
    )

    ds_dict: dict[str, Dataset] = {}
    for sp in ("train", "val", "test"):
        rows = by_split[sp]
        if len(rows["pair"]) == 0:
            raise RuntimeError(f"{ds_paths.name}: no rows for split {sp}")
        ds_dict[sp] = Dataset.from_dict(rows, features=features)
    return DatasetDict(ds_dict)


def push_dataset(ds: "Any", repo_id: str, private: bool, token: str | None) -> str:
    ds.push_to_hub(repo_id, private=private, token=token)
    return f"https://huggingface.co/datasets/{repo_id}"


def run_reference(args: argparse.Namespace) -> None:
    if args.download:
        download_raw_data(args.data_dir)
    if args.prepare:
        prepare_raw_layout(args.data_dir)

    from huggingface_hub import get_token

    token: str | None = (args.token or "").strip() or None
    if args.ref_push:
        if not token:
            token = get_token()
        if not token:
            print("No Hugging Face token; run huggingface-cli login", file=sys.stderr)
            raise SystemExit(2)

    targets = list(DATASET_ROOTS.keys()) if args.ref_only == "all" else [args.ref_only]
    for name in targets:
        root = args.data_dir / DATASET_ROOTS[name]
        bundle = DatasetPaths(name=name, root=root)
        ds = build_hf_datasetdict(bundle)
        print(
            f"[{name}] train/val/test = {len(ds['train'])}/{len(ds['val'])}/{len(ds['test'])}",
            flush=True,
        )
        if args.ref_push:
            repo_name = f"{args.repo_prefix}-{name}"
            repo_id = f"{args.namespace}/{repo_name}"
            private = not args.ref_public
            url = push_dataset(ds, repo_id, private=private, token=token)
            print(f"[{name}] pushed: {url}", flush=True)





def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Build CSP-related Hugging Face datasets. Use --mode clevr for CLEVR-style "
            "releases or --mode reference for CSP reference benchmarks."
        )
    )
    p.add_argument(
        "--mode",
        choices=("clevr", "reference"),
        default="clevr",
        help="Which dataset family to build (default: clevr).",
    )

    # clevr
    clevr_g = p.add_argument_group("clevr (CLEVR-style csp_two_object / csp_rel / csp_single_object)")
    clevr_g.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root containing two_object/rel/single_object trees",
    )
    clevr_g.add_argument(
        "--hub-user",
        type=str,
        default=os.environ.get("HF_HUB_USER", "").strip(),
        help="Hub namespace for <hub_user>/csp_* (or set HF_HUB_USER)",
    )
    clevr_g.add_argument("--push", action="store_true", help="push_to_hub each DatasetDict")
    clevr_g.add_argument("--public", action="store_true", help="public Hub datasets (default private)")
    clevr_g.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="save each DatasetDict to disk under this directory",
    )
    clevr_g.add_argument(
        "--clevr-only",
        choices=("two_object", "rel", "single_object", "all"),
        default="all",
        dest="clevr_only",
        help="Build one CLEVR-style tree or all (default: all)",
    )

    # reference
    ref_g = p.add_argument_group("reference (MIT-States / UT-Zappos / C-GQA)")
    ref_g.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory for downloads and extracted reference data",
    )
    ref_g.add_argument("--namespace", type=str, default="zzq1zh")
    ref_g.add_argument("--repo-prefix", type=str, default="csp-ref")
    ref_g.add_argument(
        "--ref-only",
        choices=("all", "mit-states", "ut-zappos", "cgqa"),
        default="all",
        dest="ref_only",
        help="Which reference dataset to convert (default: all)",
    )
    ref_g.add_argument("--download", action="store_true", help="Run wget downloads (CSP-style)")
    ref_g.add_argument("--prepare", action="store_true", help="Extract/reorganize archives")
    ref_g.add_argument("--ref-push", action="store_true", dest="ref_push", help="Push to HF Hub")
    ref_g.add_argument(
        "--ref-public",
        action="store_true",
        dest="ref_public",
        help="Make reference Hub datasets public (default: private)",
    )
    ref_g.add_argument(
        "--token",
        type=str,
        default="",
    )

    args = p.parse_args()
    if args.mode == "clevr":
        run_clevr(args)
    else:
        run_reference(args)


if __name__ == "__main__":
    main()
