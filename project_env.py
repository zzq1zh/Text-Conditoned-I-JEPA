from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent


def load_project_env() -> bool:
    path = _ROOT / ".env"
    if path.is_file():
        return bool(load_dotenv(path))
    return bool(load_dotenv())
