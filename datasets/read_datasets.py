from __future__ import annotations

from typing import Any

import project_env
from vision_data import DATASET_CONFIG, load_vision_huggingface_as_dataset_dict

project_env.load_project_env()

# CSP code-style names -> project vision_data keys backed by HF Hub repos.
DATASET_TO_VISION_KEY = {
    "mit-states": "cspref_mit_states",
    "ut-zappos": "cspref_ut_zappos",
    "cgqa": "cspref_cgqa",
}

# Backward-compatible alias kept for older scripts.
DATASET_PATHS = DATASET_TO_VISION_KEY


def get_hf_repo_id(dataset_name: str) -> str:
    if dataset_name not in DATASET_TO_VISION_KEY:
        known = ", ".join(sorted(DATASET_TO_VISION_KEY))
        raise KeyError(f"Unknown dataset {dataset_name!r}. Choose one of: {known}")
    key = DATASET_TO_VISION_KEY[dataset_name]
    return str(DATASET_CONFIG[key]["path"])


def load_composition_datasetdict(dataset_name: str) -> Any:
    """
    Load CSP-style compositional datasets from Hugging Face Hub.
    Returns a DatasetDict with train/val/test and pair-based label metadata.
    """
    if dataset_name not in DATASET_TO_VISION_KEY:
        known = ", ".join(sorted(DATASET_TO_VISION_KEY))
        raise KeyError(f"Unknown dataset {dataset_name!r}. Choose one of: {known}")
    return load_vision_huggingface_as_dataset_dict(DATASET_TO_VISION_KEY[dataset_name])