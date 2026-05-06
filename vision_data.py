"""
Shared vision dataset loading (Hugging Face Datasets) for CSP-reference
compositional benchmarks: Hub repos with fixed ``train`` / ``val`` / ``test``
and pair-based labels (``cspref_*``).

See README for attribution to Nayak et al., ICLR 2023.
Inline comments in English.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from datasets import Dataset, DatasetDict, ClassLabel, load_dataset


# CSP-reference datasets (see ``build_csp_hf_datasets.py --mode reference``).
DATASET_CONFIG: dict[str, dict[str, Any]] = {
    "cspref_mit_states": {
        "path": "zzq1zh/csp-ref-mit-states",
        "name": None,
        "label_key": "label",
    },
    "cspref_ut_zappos": {
        "path": "zzq1zh/csp-ref-ut-zappos",
        "name": None,
        "label_key": "label",
    },
    "cspref_cgqa": {
        "path": "zzq1zh/csp-ref-cgqa",
        "name": None,
        "label_key": "label",
    },
}


def list_vision_dataset_keys() -> list[str]:
    """Return sorted keys accepted by :func:`load_vision_dataset`."""
    return sorted(DATASET_CONFIG.keys())


def _prepare_csp_ref_datasetdict(raw: DatasetDict, path_for_err: str) -> DatasetDict:
    """
    Build/normalize pair-based labels for CSP-reference HF repos.
    Uses the ``pair`` string when present, otherwise falls back to ``attr`` + ``obj``.
    """
    if not isinstance(raw, DatasetDict):
        raise TypeError(f"Expected DatasetDict for CSP ref {path_for_err!r}, got {type(raw).__name__}")
    if "train" not in raw:
        raise KeyError(f"CSP ref {path_for_err!r} must contain a train split, got {list(raw.keys())}")

    all_pairs: set[str] = set()
    for sp in raw.values():
        cm = set(sp.column_names)
        if "pair" in cm:
            for i in range(len(sp)):
                all_pairs.add(str(sp[i]["pair"]).strip())
        elif "attr" in cm and "obj" in cm:
            for i in range(len(sp)):
                all_pairs.add(f"{str(sp[i]['attr']).strip()} {str(sp[i]['obj']).strip()}")
        else:
            raise ValueError(
                f"CSP ref {path_for_err!r}: each split must provide either pair or attr+obj columns; got {sorted(cm)}"
            )
    class_names = sorted(x for x in all_pairs if x)
    if not class_names:
        raise ValueError(f"CSP ref {path_for_err!r}: empty pair label set")
    name_to_id = {n: j for j, n in enumerate(class_names)}

    def _batch_pair_from_pair(examples: dict[str, list[Any]]) -> dict[str, list[int]]:
        out: list[int] = []
        for p in examples["pair"]:
            key = str(p).strip()
            out.append(name_to_id[key])
        return {"label": out}

    def _batch_pair_from_attr_obj(examples: dict[str, list[Any]]) -> dict[str, list[int]]:
        out: list[int] = []
        attrs = examples["attr"]
        objs = examples["obj"]
        for a, o in zip(attrs, objs, strict=False):
            key = f"{str(a).strip()} {str(o).strip()}"
            out.append(name_to_id[key])
        return {"label": out}

    cl = ClassLabel(names=class_names)
    out_s: dict[str, Dataset] = {}
    for name, d in raw.items():
        cm = set(d.column_names)
        if len(d) == 0:
            out_s[name] = d
            continue
        if "pair" in cm:
            d2 = d.map(_batch_pair_from_pair, batched=True)
        else:
            d2 = d.map(_batch_pair_from_attr_obj, batched=True)
        d2 = d2.cast_column("label", cl)
        out_s[name] = d2
    return DatasetDict(out_s)


def load_vision_huggingface_as_dataset_dict(dataset_key: str) -> DatasetDict:
    """
    Load a registry entry: HF ``DatasetDict`` with ``train`` / ``val`` / ``test``,
    normalized via :func:`_prepare_csp_ref_datasetdict`.
    """
    if dataset_key not in DATASET_CONFIG:
        known = ", ".join(list_vision_dataset_keys())
        raise ValueError(f"Unknown dataset {dataset_key!r}. Choose one of: {known}")
    cfg = DATASET_CONFIG[dataset_key]
    pth = str(cfg["path"])
    raw0 = load_dataset(pth, cfg.get("name"))
    if not isinstance(raw0, DatasetDict):
        raise TypeError(
            f"Expected DatasetDict (train/val/test) for CSP ref dataset {pth!r}, got {type(raw0).__name__}"
        )
    return _prepare_csp_ref_datasetdict(raw0, pth)


def set_seed(seed: int) -> None:
    """Reproducible shuffles / subsampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_split(ds: DatasetDict | Dataset, split: str) -> Dataset:
    """Pick one split; map ``test`` -> ``validation`` if ``test`` is missing."""
    if isinstance(ds, Dataset):
        return ds
    if split in ds:
        return ds[split]
    if split == "test" and "validation" in ds and "test" not in ds:
        return ds["validation"]
    available = list(ds.keys())
    raise KeyError(f"Split {split!r} not found. Available: {available}")


def load_vision_dataset(
    dataset_key: str,
    split: str = "test",
) -> tuple[Dataset, list[str], str]:
    """
    Load one CSP-reference split from the registry.

    Returns:
        dataset: column ``img`` or ``image`` (PIL), labels under ``label_key``
        class_names: ClassLabel order matching integer ids
        label_key: feature name for the label column
    """
    if dataset_key not in DATASET_CONFIG:
        known = ", ".join(list_vision_dataset_keys())
        raise ValueError(f"Unknown dataset {dataset_key!r}. Choose one of: {known}")

    cfg = DATASET_CONFIG[dataset_key]
    raw = load_vision_huggingface_as_dataset_dict(dataset_key)
    part = resolve_split(raw, split)
    label_key = cfg["label_key"]
    names_feature = part.features[label_key]
    if not hasattr(names_feature, "names"):
        raise TypeError(
            f"Expected ClassLabel for {label_key!r}, got {type(names_feature).__name__}"
        )
    class_names: list[str] = list(names_feature.names)
    return part, class_names, label_key


def get_image_column(dataset: Dataset) -> str:
    """Return ``img`` or ``image`` if present."""
    for key in ("img", "image"):
        if key in dataset.column_names:
            return key
    raise KeyError(
        f"No image column; expected 'img' or 'image' in {dataset.column_names}"
    )


def limit_dataset_size(dataset: Dataset, max_samples: int | None) -> Dataset:
    """Subsample the first ``max_samples`` rows after a fixed shuffle (seed=0)."""
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return dataset.shuffle(seed=0).select(range(max_samples))


@dataclass(frozen=True)
class VisionBatchSpec:
    """Convenience bundle after load + limit + image column resolution."""

    dataset: Dataset
    class_names: list[str]
    label_key: str
    image_column: str


@dataclass(frozen=True)
class VisionTrainTestVal:
    """
    Published Hub ``train`` / ``val`` / ``test`` for CSP-reference datasets.

    For closed-world scoring, use :func:`csp_vocab_allowed_class_indices` with
    ``role="val"`` or ``"test"`` so val eval uses train∪val candidates and test
    eval uses train∪test.
    """

    train: VisionBatchSpec
    val: VisionBatchSpec
    test: VisionBatchSpec


def _build_spec_from_part(
    part: Dataset,
    class_names: list[str],
    label_key: str,
    max_samples: int | None,
) -> VisionBatchSpec:
    part = limit_dataset_size(part, max_samples)
    return VisionBatchSpec(
        dataset=part,
        class_names=class_names,
        label_key=label_key,
        image_column=get_image_column(part),
    )


def load_vision_train_val_test_specs(
    dataset_key: str,
    train_hub_split: str = "train",
    test_hub_split: str = "test",
    val_fraction: float = 0.1,
    split_seed: int = 0,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
) -> VisionTrainTestVal:
    """
    Build **train / val / test** from published CSP-reference Hub splits.

    ``train_hub_split``, ``test_hub_split``, ``val_fraction``, and ``split_seed`` are
    kept for call-site compatibility but **ignored** (splits are always Hub ``train``,
    ``val``, and ``test``).
    """
    if dataset_key not in DATASET_CONFIG:
        known = ", ".join(list_vision_dataset_keys())
        raise ValueError(f"Unknown dataset {dataset_key!r}. Choose one of: {known}")

    cfg = DATASET_CONFIG[dataset_key]
    raw = load_vision_huggingface_as_dataset_dict(dataset_key)
    for k in ("train", "val", "test"):
        if k not in raw:
            raise KeyError(
                f"CSP-reference dataset {dataset_key!r} must have splits train, val, test; got {list(raw)}"
            )
    label_key = cfg["label_key"]
    hub_train, hub_val, hub_test = raw["train"], raw["val"], raw["test"]
    names_tr = hub_train.features[label_key]
    if not hasattr(names_tr, "names"):
        raise TypeError(
            f"Expected ClassLabel for {label_key!r} on train, got {type(names_tr).__name__}"
        )
    class_names: list[str] = list(names_tr.names)
    if list(hub_val.features[label_key].names) != class_names:
        raise ValueError("CSP ref: train and val ClassLabel name lists do not match")
    if list(hub_test.features[label_key].names) != class_names:
        raise ValueError("CSP ref: train and test ClassLabel name lists do not match")
    return VisionTrainTestVal(
        train=_build_spec_from_part(hub_train, class_names, label_key, max_train_samples),
        val=_build_spec_from_part(hub_val, class_names, label_key, max_val_samples),
        test=_build_spec_from_part(hub_test, class_names, label_key, max_test_samples),
    )


def csp_vocab_allowed_class_indices(tvt: VisionTrainTestVal, role: str) -> list[int]:
    """
    Global class indices allowed for closed-world classification / composed-pair banks:

    - ``role == "train"``: labels that appear in **train** rows only.
    - ``role == "val"``: labels that appear in **train** or **val** rows.
    - ``role == "test"``: labels that appear in **train** or **test** rows.

    Uses ``tvt.train.label_key`` on each split (shared ``ClassLabel`` order across splits).

    Used by ``csp_vocab_train``, ``text_cond_train`` (--finetune-csp-vocab, eval, and
    main training val metrics) to restrict softmax candidates from actual split label sets.
    """
    if role not in ("train", "val", "test"):
        raise ValueError(f"role must be 'train', 'val', or 'test', got {role!r}")
    n = len(tvt.train.class_names)
    lk = tvt.train.label_key
    ids_train = {int(x) for x in tvt.train.dataset.unique(lk)}
    if role == "train":
        allowed = sorted(ids_train)
    elif role == "val":
        ids_other = {int(x) for x in tvt.val.dataset.unique(lk)}
        allowed = sorted(ids_train | ids_other)
    else:
        ids_other = {int(x) for x in tvt.test.dataset.unique(lk)}
        allowed = sorted(ids_train | ids_other)
    if len(allowed) > n:
        raise RuntimeError(f"Allowed class count {len(allowed)} exceeds num_classes {n}")
    return allowed


def load_vision_batch_spec(
    dataset_key: str,
    split: str = "test",
    max_samples: int | None = None,
) -> VisionBatchSpec:
    """
    One-call loader: CSP-reference split + image column name.

    ``num_labels`` for classification heads is ``len(class_names)``.
    """
    ds, class_names, label_key = load_vision_dataset(dataset_key, split=split)
    ds = limit_dataset_size(ds, max_samples)
    img_col = get_image_column(ds)
    return VisionBatchSpec(
        dataset=ds,
        class_names=class_names,
        label_key=label_key,
        image_column=img_col,
    )


def build_text_prompts(class_names: Iterable[str], template: str) -> list[str]:
    """
    One prompt string per class (zero-shot CLIP, or class vocabulary for templates).
    Use ``{c}`` in ``template`` for the class name.
    """
    return [template.format(c=c) for c in class_names]


def prompts_for_label_indices(
    class_names: list[str],
    template: str,
    label_ids: list[int] | np.ndarray | torch.Tensor,
) -> list[str]:
    """
    For supervised training: one prompt per example from its class index, e.g.
    ``"a photo of a {c}."`` with ``c = class_names[y]``.
    """
    if isinstance(label_ids, torch.Tensor):
        label_ids = label_ids.detach().cpu().tolist()
    elif isinstance(label_ids, np.ndarray):
        label_ids = label_ids.tolist()
    return [template.format(c=class_names[i]) for i in label_ids]
