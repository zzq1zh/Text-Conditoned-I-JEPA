from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

_ROOT = Path(__file__).resolve().parent

# Parsed repo `.env`; ``WANDB_API_KEY`` is never copied into ``os.environ``.
_env_file_cache: dict[str, str | None] | None = None


def load_project_env() -> bool:
    """
    Load variables from repo ``.env`` into ``os.environ`` for keys other than
    ``WANDB_API_KEY`` (existing environment variables are not overridden).
    The API key stays in the file only; use :func:`get_wandb_api_key`.
    """
    global _env_file_cache
    path = _ROOT / ".env"
    if not path.is_file():
        _env_file_cache = {}
        return False
    _env_file_cache = dict(dotenv_values(path))
    for key, value in _env_file_cache.items():
        if not key or value is None:
            continue
        if key == "WANDB_API_KEY":
            continue
        if key not in os.environ:
            os.environ[key] = value
    return True


def get_wandb_api_key() -> str:
    """Return ``WANDB_API_KEY`` from repo ``.env`` (stripped). Not read from ``os.environ``."""
    global _env_file_cache
    if _env_file_cache is None:
        load_project_env()
    assert _env_file_cache is not None
    return str(_env_file_cache.get("WANDB_API_KEY") or "").strip()


def wandb_configured() -> bool:
    """True when repo ``.env`` defines a non-empty ``WANDB_API_KEY``."""
    return bool(get_wandb_api_key())
