"""
Dataset utilities aligned with the compositional view used in BatsResearch/csp:

- expose per-example ``attr`` / ``obj`` / ``pair`` strings
- keep split names (train / val / test)
- annotate whether a pair is seen in train

This file is intentionally additive: it does not replace ``vision_data.py``.
Use it when you want CSP-style analysis/evaluation metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset, DatasetDict

from vision_data import DATASET_CONFIG, load_vision_huggingface_as_dataset_dict


def _split_phrase_attr_obj(phrase: str) -> tuple[str, str]:
    """
    Heuristic phrase parser: last token -> object, preceding tokens -> attribute.
    If phrase has one token, attr is ``__none__``.
    """
    s = (phrase or "").strip()
    if not s:
        return "__none__", "unknown"
    toks = s.split()
    if len(toks) == 1:
        return "__none__", toks[0]
    return " ".join(toks[:-1]), toks[-1]


def _row_positive_phrase(row: dict[str, Any]) -> str:
    """Pick a positive phrase in the same spirit as project CSP loaders."""
    if "pos" in row:
        return str(row.get("pos") or "").strip()
    positives = row.get("positives")
    if isinstance(positives, list) and positives:
        return str(positives[0] or "").strip()
    return ""


def _to_csp_style_split(split_ds: Dataset, split_name: str) -> Dataset:
    """
    Convert one split into CSP-style records with columns:
    ``image``, ``phrase``, ``attr``, ``obj``, ``pair``, ``split``, ``label`` (if present).
    """
    rows: dict[str, list[Any]] = {
        "image": [],
        "phrase": [],
        "attr": [],
        "obj": [],
        "pair": [],
        "split": [],
    }
    include_label = "label" in split_ds.column_names
    if include_label:
        rows["label"] = []

    img_col = "image" if "image" in split_ds.column_names else "img"
    for i in range(len(split_ds)):
        row = split_ds[i]
        phrase = _row_positive_phrase(row)
        attr, obj = _split_phrase_attr_obj(phrase)
        pair = f"{attr}||{obj}"
        rows["image"].append(row[img_col])
        rows["phrase"].append(phrase)
        rows["attr"].append(attr)
        rows["obj"].append(obj)
        rows["pair"].append(pair)
        rows["split"].append(split_name)
        if include_label:
            rows["label"].append(int(row["label"]))
    return Dataset.from_dict(rows)


def _add_seen_pair_flag(ds: Dataset, seen_pairs: set[str]) -> Dataset:
    return ds.map(lambda b: {"pair_seen_in_train": [p in seen_pairs for p in b["pair"]]}, batched=True)


@dataclass(frozen=True)
class CspStyleBundle:
    """
    ``data``: split dict with train/val/test, and columns including attr/obj/pair/seen flag.
    ``seen_pairs``: pair vocabulary from train.
    ``all_pairs``: pair vocabulary from all splits.
    """

    data: DatasetDict
    seen_pairs: set[str]
    all_pairs: set[str]


def load_csp_style_dataset(dataset_key: str) -> CspStyleBundle:
    """
    Build CSP-style split datasets from a registered dataset key.

    Best fit: ``csp_two_object`` / ``csp_single_object`` / ``csp_rel``.
    For non-CSP datasets this still runs, but attr/object are phrase heuristics.
    """
    if dataset_key not in DATASET_CONFIG:
        known = ", ".join(sorted(DATASET_CONFIG))
        raise ValueError(f"Unknown dataset {dataset_key!r}. Choose one of: {known}")

    raw = load_vision_huggingface_as_dataset_dict(dataset_key)
    out: dict[str, Dataset] = {}
    for split_name in ("train", "val", "test"):
        if split_name in raw:
            out[split_name] = _to_csp_style_split(raw[split_name], split_name)
    if "train" not in out:
        raise KeyError(f"{dataset_key!r} must provide a train split for CSP-style view")

    seen_pairs = set(out["train"]["pair"])
    all_pairs: set[str] = set()
    for split_name, ds in out.items():
        all_pairs.update(ds["pair"])
        out[split_name] = _add_seen_pair_flag(ds, seen_pairs)

    return CspStyleBundle(data=DatasetDict(out), seen_pairs=seen_pairs, all_pairs=all_pairs)


def summarize_csp_style(bundle: CspStyleBundle) -> dict[str, dict[str, int]]:
    """
    Return basic pair statistics per split:
    n_rows / n_pairs / n_seen_pairs / n_unseen_pairs.
    """
    stats: dict[str, dict[str, int]] = {}
    for split_name, ds in bundle.data.items():
        pairs = set(ds["pair"])
        seen = sum(1 for p in pairs if p in bundle.seen_pairs)
        unseen = len(pairs) - seen
        stats[split_name] = {
            "n_rows": len(ds),
            "n_pairs": len(pairs),
            "n_seen_pairs": seen,
            "n_unseen_pairs": unseen,
        }
    return stats

