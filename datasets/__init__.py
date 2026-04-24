"""
Bridge package:
- keeps local modules under ``datasets/`` importable (e.g. ``datasets.composition_dataset``)
- also exposes Hugging Face ``datasets`` symbols used by project code.
"""

from __future__ import annotations

import sysconfig
from pathlib import Path

_local_dir = Path(__file__).resolve().parent
_hf_dir = Path(sysconfig.get_paths()["purelib"]) / "datasets"

# Search local modules first, then HF datasets package modules.
__path__ = [str(_local_dir), str(_hf_dir)]
_hf_init = _hf_dir / "__init__.py"
if not _hf_init.is_file():
    raise ModuleNotFoundError(f"Hugging Face datasets package not found at {_hf_init}")

# Execute HF datasets __init__ in this module namespace so imports like
# ``from datasets import load_dataset`` keep working, while local submodules
# (e.g. ``datasets.composition_dataset``) remain available.
exec(_hf_init.read_text(encoding="utf-8"), globals(), globals())
