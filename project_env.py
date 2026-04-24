"""
Load key=value pairs from a ``.env`` file at the repository root (see ``.env.example``).
Call :func:`load_project_env` before Hugging Face Hub, W&B, or other services read tokens
from the environment.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent


def load_project_env() -> bool:
    """
    Read ``.env`` next to this package (project root), then fall back to ``load_dotenv()``
    in the current working directory. Returns whether any variable was set from a file
    (see ``python-dotenv``).
    """
    path = _ROOT / ".env"
    if path.is_file():
        return bool(load_dotenv(path))
    return bool(load_dotenv())
