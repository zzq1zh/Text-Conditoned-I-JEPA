"""
Shared vision dataset loading (Hugging Face Datasets) for CLIP, I-JEPA, and
text-conditioned I-JEPA. Keep registry and split logic in one place.
Inline comments in English.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from datasets import Dataset, DatasetDict, Features, ClassLabel, Image, concatenate_datasets, load_dataset


# CIFAR-10/100 are common CLIP / vision-language baselines; extend this table to add more sets.
# "mit_states" uses Hugging Face nested format (noun + list of images) → flattened object classification
# (one row per image, label = noun); see :func:`load_vision_huggingface_as_dataset_dict`.
# "csp_*" are project Hub datasets (see ``build_csp_hf_datasets.py --mode clevr``) with pre-split
# train/val/test; ``load_vision_train_val_test_specs`` uses them as-is (no re-split of train).
DATASET_CONFIG: dict[str, dict[str, Any]] = {
    "cifar10": {
        "path": "cifar10",
        "name": None,
        "label_key": "label",
    },
    "cifar100": {
        "path": "cifar100",
        "name": None,
        "label_key": "fine_label",
    },
    "mit_states": {
        "path": "nirmalendu01/MIT-States-Cleaned-Subset",
        "name": None,
        "label_key": "label",
        "loader": "mit_states_flatten",
    },
    "csp_two_object": {
        "path": "zzq1zh/csp_two_object",
        "name": None,
        "label_key": "label",
        "loader": "csp_hub",
        "csp_label_source": "pos",
    },
    "csp_single_object": {
        "path": "zzq1zh/csp_single_object",
        "name": None,
        "label_key": "label",
        "loader": "csp_hub",
        "csp_label_source": "pos",
    },
    "csp_rel": {
        "path": "zzq1zh/csp_rel",
        "name": None,
        "label_key": "label",
        "loader": "csp_hub",
        "csp_label_source": "positives_first",
    },
    "cspref_mit_states": {
        "path": "zzq1zh/csp-ref-mit-states",
        "name": None,
        "label_key": "label",
        "loader": "csp_ref_hub",
    },
    "cspref_ut_zappos": {
        "path": "zzq1zh/csp-ref-ut-zappos",
        "name": None,
        "label_key": "label",
        "loader": "csp_ref_hub",
    },
    "cspref_cgqa": {
        "path": "zzq1zh/csp-ref-cgqa",
        "name": None,
        "label_key": "label",
        "loader": "csp_ref_hub",
    },
}


def list_vision_dataset_keys() -> list[str]:
    """Return sorted keys accepted by :func:`load_vision_dataset`."""
    return sorted(DATASET_CONFIG.keys())


def _flatten_mit_states_split_to_labeled(
    split_ds: Dataset, *, label_column: str = "noun", image_list_column: str = "images"
) -> Dataset:
    """
    One MIT-States row = one object (noun) and a list of images; expand to
    one row per image, classification target = object name (noun).
    """
    images: list[Any] = []
    nouns: list[str] = []
    for i in range(len(split_ds)):
        row = split_ds[i]
        n = str(row[label_column])
        for im in row[image_list_column]:
            images.append(im)
            nouns.append(n)
    if not images:
        raise ValueError("MIT-States: no images after flatten; check Hub column names (noun, images).")
    class_names = sorted(set(nouns))
    name_to_id = {c: j for j, c in enumerate(class_names)}
    labels = [name_to_id[t] for t in nouns]
    features = Features(
        {
            "image": Image(),
            "label": ClassLabel(names=class_names),
        }
    )
    return Dataset.from_dict({"image": images, "label": labels}, features=features)


def _mit_states_huggingface_to_train_test(
    path: str,
    name: str | None,
    *,
    test_size: float = 0.2,
    seed: int = 0,
) -> DatasetDict:
    """Nested MIT-States Hub data → flat ``train``/``test`` (hold-out) for the shared split logic."""
    raw = load_dataset(path, name)
    parts: list[Dataset] = []
    for key in ("train", "validation", "test"):
        if key in raw:
            parts.append(_flatten_mit_states_split_to_labeled(raw[key]))
    if not parts:
        for k in raw:
            parts.append(_flatten_mit_states_split_to_labeled(raw[k]))
    if not parts:
        raise ValueError("MIT-States: no splits in loaded DatasetDict")
    if len(parts) == 1:
        flat = parts[0]
    else:
        flat = concatenate_datasets(parts)
    try:
        ttv = flat.train_test_split(
            test_size=test_size, shuffle=True, seed=seed, stratify_by_column="label"
        )
    except (ValueError, TypeError):
        ttv = flat.train_test_split(test_size=test_size, shuffle=True, seed=seed)
    return DatasetDict({"train": ttv["train"], "test": ttv["test"]})


def _row_label_str_csp(row: Any, source: str) -> str:
    if isinstance(row, dict):
        r: dict[str, Any] = row
    else:
        r = row
    if source == "pos":
        p = r.get("pos")
        s = (str(p).strip() if p is not None else "") or "unknown"
        return s
    if source == "positives_first":
        pl = r.get("positives")
        if pl is not None and len(pl) > 0:
            return str(pl[0]).strip() or "unknown"
        return "unknown"
    raise ValueError(f"Unknown csp_label_source: {source!r}")


def _prepare_csp_datasetdict(raw: DatasetDict, path_for_err: str, csp_label_source: str) -> DatasetDict:
    """
    Add a ``label`` column (:class:`ClassLabel`) from string ``pos`` or from the
    first string in ``positives`` (``positives_first`` for ``csp_rel``).
    """
    all_labels: set[str] = set()
    for sp in raw.values():
        for i in range(len(sp)):
            s = _row_label_str_csp(sp[i], csp_label_source)
            all_labels.add(s)
    class_names = sorted(all_labels)
    if not class_names:
        raise ValueError(f"CSP {path_for_err!r}: empty label set")
    name_to_id: dict[str, int] = {n: j for j, n in enumerate(class_names)}

    def _batch_pos(examples: dict[str, list[Any]]) -> dict[str, list[int]]:
        labs: list[int] = []
        for p in examples["pos"]:
            k = (str(p).strip() if p is not None else "") or "unknown"
            if k not in name_to_id:
                k = "unknown"
            labs.append(name_to_id[k])
        return {"label": labs}

    def _batch_rel(examples: dict[str, list[Any]]) -> dict[str, list[int]]:
        out: list[int] = []
        for plist in examples["positives"]:
            t = "unknown"
            if plist is not None and len(plist) > 0:
                t = str(plist[0]).strip() or "unknown"
            if t not in name_to_id:
                t = "unknown"
            out.append(name_to_id[t])
        return {"label": out}

    if csp_label_source not in ("pos", "positives_first"):
        raise ValueError(f"csp_label_source must be pos or positives_first, got {csp_label_source!r}")
    k0 = "train" if "train" in raw else list(raw.keys())[0]
    cm = raw[k0].column_names
    if csp_label_source == "pos" and "pos" not in cm:
        raise ValueError(f"CSP {path_for_err!r}: expected a pos column, got {cm}")
    if csp_label_source == "positives_first" and "positives" not in cm:
        raise ValueError(f"CSP {path_for_err!r}: expected a positives column, got {cm}")

    cl = ClassLabel(names=class_names)
    out_s: dict[str, Dataset] = {}
    for name, d in raw.items():
        if len(d) == 0:
            out_s[name] = d
            continue
        if csp_label_source == "pos":
            d2 = d.map(_batch_pos, batched=True)
        else:
            d2 = d.map(_batch_rel, batched=True)
        d2 = d2.cast_column("label", cl)
        out_s[name] = d2
    return DatasetDict(out_s)


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
    Load a registry entry as a :class:`DatasetDict`. CIFAR uses native Hub
    ``train``/``test``; ``mit_states`` is flattened to ``image``/``label`` and
    given a fresh train/test split.
    """
    if dataset_key not in DATASET_CONFIG:
        known = ", ".join(list_vision_dataset_keys())
        raise ValueError(f"Unknown dataset {dataset_key!r}. Choose one of: {known}")
    cfg = DATASET_CONFIG[dataset_key]
    if cfg.get("loader") == "mit_states_flatten":
        return _mit_states_huggingface_to_train_test(
            str(cfg["path"]),
            cfg.get("name"),
        )
    if cfg.get("loader") == "csp_hub":
        pth = str(cfg["path"])
        raw0 = load_dataset(pth, cfg.get("name"))
        if not isinstance(raw0, DatasetDict):
            raise TypeError(
                f"Expected DatasetDict (train/val/test) for CSP dataset {pth!r}, got {type(raw0).__name__}"
            )
        src = str(cfg.get("csp_label_source", "pos"))
        return _prepare_csp_datasetdict(raw0, pth, src)
    if cfg.get("loader") == "csp_ref_hub":
        pth = str(cfg["path"])
        raw0 = load_dataset(pth, cfg.get("name"))
        if not isinstance(raw0, DatasetDict):
            raise TypeError(
                f"Expected DatasetDict (train/val/test) for CSP ref dataset {pth!r}, got {type(raw0).__name__}"
            )
        return _prepare_csp_ref_datasetdict(raw0, pth)
    raw = load_dataset(cfg["path"], cfg["name"])
    if isinstance(raw, Dataset):
        return DatasetDict({"train": raw})
    return raw


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
    Load a public image dataset (same registry for CLIP and I-JEPA pipelines).

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
class VisionTrainVal:
    """
    Two splits only (e.g. Hub ``train`` and Hub ``test`` with val = test for benchmarks).
    ``class_names`` and ``label_key`` are shared.
    """

    train: VisionBatchSpec
    val: VisionBatchSpec


@dataclass(frozen=True)
class VisionTrainTestVal:
    """
    Three-way split: **train** and **val** are carved from the Hub **train**;
    **test** = **val ∪ Hub test** (``concat``). So the *test* pool **contains** the
    validation set plus the official test split, for a single combined evaluation
    set while **val** remains available for training-time monitoring.
    """

    train: VisionBatchSpec
    val: VisionBatchSpec
    test: VisionBatchSpec


def get_available_split_names(dataset_key: str) -> list[str]:
    """
    List split keys on the Hub (e.g. ``train`` / ``test`` for CIFAR) without double-loading
    the examples array twice if you will call :func:`load_vision_train_val_specs` next
    (this still runs ``load_dataset`` once).
    """
    if dataset_key not in DATASET_CONFIG:
        known = ", ".join(list_vision_dataset_keys())
        raise ValueError(f"Unknown dataset {dataset_key!r}. Choose one of: {known}")
    raw = load_vision_huggingface_as_dataset_dict(dataset_key)
    if isinstance(raw, Dataset):
        return ["train"]
    return list(raw.keys())


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


def load_vision_train_val_specs(
    dataset_key: str,
    train_split: str = "train",
    val_split: str = "test",
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> VisionTrainVal:
    """
    Load **training** and **validation** HF splits the same way CLIP / classification
    pipelines use CIFAR: ``train`` for optimization, official ``test`` (or a separate
    ``validation``) for monitoring.

    CIFAR-10/100 expose ``train`` and ``test``; there is no separate val split, so
    **validation = test** is the common benchmark choice (same as holding out the test set
    for CLIP eval scripts).
    """
    if dataset_key not in DATASET_CONFIG:
        known = ", ".join(list_vision_dataset_keys())
        raise ValueError(f"Unknown dataset {dataset_key!r}. Choose one of: {known}")
    cfg = DATASET_CONFIG[dataset_key]
    if cfg.get("loader") == "csp_hub":
        raw = load_vision_huggingface_as_dataset_dict(dataset_key)
        if "train" not in raw or "val" not in raw:
            raise KeyError(
                f"CSP dataset {dataset_key!r} must have train and val splits, got {list(raw)}"
            )
        train_part, val_part = raw["train"], raw["val"]
        label_key = cfg["label_key"]
        for name, p in (("train", train_part), ("val", val_part)):
            names_feature = p.features[label_key]
            if not hasattr(names_feature, "names"):
                raise TypeError(
                    f"Expected ClassLabel for {label_key!r} in {name}, got {type(names_feature).__name__}"
                )
        class_names: list[str] = list(train_part.features[label_key].names)
        if list(val_part.features[label_key].names) != class_names:
            raise ValueError("CSP: train and val ClassLabel name lists do not match")
        return VisionTrainVal(
            train=_build_spec_from_part(train_part, class_names, label_key, max_train_samples),
            val=_build_spec_from_part(val_part, class_names, label_key, max_val_samples),
        )

    raw = load_vision_huggingface_as_dataset_dict(dataset_key)
    train_part = resolve_split(raw, train_split)
    val_part = resolve_split(raw, val_split)
    label_key = cfg["label_key"]
    for name, p in (("train", train_part), ("val", val_part)):
        names_feature = p.features[label_key]
        if not hasattr(names_feature, "names"):
            raise TypeError(
                f"Expected ClassLabel for {label_key!r} in {name}, got {type(names_feature).__name__}"
            )
    # Same ClassLabel order on both CIFAR splits
    class_names: list[str] = list(train_part.features[label_key].names)
    if list(val_part.features[label_key].names) != class_names:
        raise ValueError("Train and val ClassLabel name lists do not match")

    return VisionTrainVal(
        train=_build_spec_from_part(train_part, class_names, label_key, max_train_samples),
        val=_build_spec_from_part(val_part, class_names, label_key, max_val_samples),
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
    Build **train / val / test** for CIFAR-style Hub data:

    - The Hub ``train`` is split into *train* (1 - val_fraction) and *val* (stratified
      when possible).
    - *test* is **``concat(val, hub_test)``** — it **includes** all validation
      examples and all official test examples (``max_test`` caps this combined set).

    ``val_fraction`` is the fraction of Hub training rows that become *val* (e.g. 0.1).

    For **csp_** and **cspref_** hub datasets, ``val_fraction`` and ``split_seed``
    are ignored: the published ``train`` / ``val`` / ``test`` splits are used directly
    (no merge of val into the test pool).

    For **cspref_*** closed-world scoring, use :func:`csp_style_eval_allowed_class_indices`
    so val eval uses train∪val candidates and test eval uses train∪test (see training/eval scripts).
    """
    if dataset_key not in DATASET_CONFIG:
        known = ", ".join(list_vision_dataset_keys())
        raise ValueError(f"Unknown dataset {dataset_key!r}. Choose one of: {known}")

    cfg = DATASET_CONFIG[dataset_key]
    if cfg.get("loader") in {"csp_hub", "csp_ref_hub"}:
        raw = load_vision_huggingface_as_dataset_dict(dataset_key)
        for k in ("train", "val", "test"):
            if k not in raw:
                raise KeyError(
                    f"CSP-style dataset {dataset_key!r} must have splits train, val, test; got {list(raw)}"
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
            raise ValueError("CSP: train and val ClassLabel name lists do not match")
        if list(hub_test.features[label_key].names) != class_names:
            raise ValueError("CSP: train and test ClassLabel name lists do not match")
        return VisionTrainTestVal(
            train=_build_spec_from_part(hub_train, class_names, label_key, max_train_samples),
            val=_build_spec_from_part(hub_val, class_names, label_key, max_val_samples),
            test=_build_spec_from_part(hub_test, class_names, label_key, max_test_samples),
        )

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction!r}")
    label_key = cfg["label_key"]
    raw = load_vision_huggingface_as_dataset_dict(dataset_key)
    hub_train = resolve_split(raw, train_hub_split)
    hub_test = resolve_split(raw, test_hub_split)

    names_tr = hub_train.features[label_key]
    if not hasattr(names_tr, "names"):
        raise TypeError(f"Expected ClassLabel for {label_key!r} on train, got {type(names_tr).__name__}")
    class_names: list[str] = list(names_tr.names)
    if list(hub_test.features[label_key].names) != class_names:
        raise ValueError("Train and test ClassLabel name lists do not match")

    try:
        ttv = hub_train.train_test_split(
            test_size=val_fraction,
            shuffle=True,
            seed=split_seed,
            stratify_by_column=label_key,
        )
    except (ValueError, TypeError):
        ttv = hub_train.train_test_split(
            test_size=val_fraction,
            shuffle=True,
            seed=split_seed,
        )
    # HF returns keys "train" and "test" for the two parts of the split
    sub_train, sub_val = ttv["train"], ttv["test"]
    # Test includes val: combined eval pool = val (from train) + official Hub test
    test_merged = concatenate_datasets([sub_val, hub_test])

    return VisionTrainTestVal(
        train=_build_spec_from_part(sub_train, class_names, label_key, max_train_samples),
        val=_build_spec_from_part(sub_val, class_names, label_key, max_val_samples),
        test=_build_spec_from_part(test_merged, class_names, label_key, max_test_samples),
    )


def csp_style_eval_allowed_class_indices(
    dataset_key: str,
    tvt: VisionTrainTestVal,
    eval_split: str,
) -> list[int]:
    """
    For CSP-reference hub datasets (``cspref_*``), restrict closed-world candidates:

    - ``eval_split == "val"``: classes that appear in **train** or **val** rows
      (exclude test-only pair types from the softmax).
    - ``eval_split == "test"``: classes that appear in **train** or **test** rows
      (exclude val-only pair types).

    For all other registry entries, returns ``range(num_classes)`` (no restriction).
    """
    if eval_split not in ("val", "test"):
        raise ValueError(f"eval_split must be 'val' or 'test', got {eval_split!r}")
    n = len(tvt.train.class_names)
    cfg = DATASET_CONFIG.get(dataset_key) or {}
    if cfg.get("loader") != "csp_ref_hub":
        return list(range(n))
    lk = tvt.train.label_key
    ids_train = {int(x) for x in tvt.train.dataset.unique(lk)}
    if eval_split == "val":
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
    One-call loader: same data CLIP and I-JEPA use, plus image column name.

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
