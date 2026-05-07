from __future__ import annotations

import argparse
import sys
from pathlib import Path
import shutil
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from typing import Any

import project_env

project_env.load_project_env()

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

    # UT-Zappos
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
    if not md_file.is_file():
        raise FileNotFoundError(
            f"Missing metadata file {md_file} for {ds_paths.name}. "
            f"Expected metadata next to dataset root (compositional split: {ds_paths.split_name!r})."
        )
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Found metadata file at {md_file}, but torch is not installed. "
            "Install torch or remove metadata usage."
        ) from exc
    metadata = torch.load(str(md_file), map_location="cpu")

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
        description="Build CSP reference Hugging Face datasets (MIT-States / UT-Zappos / C-GQA)."
    )
    ref_g = p.add_argument_group("MIT-States / UT-Zappos / C-GQA")
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
    run_reference(args)


if __name__ == "__main__":
    main()
