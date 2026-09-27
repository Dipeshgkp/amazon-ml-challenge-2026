"""Paths and global settings. Override locations with env vars ER_DATA_DIR / ER_CACHE_DIR / ER_OUTPUT_DIR."""
import os
from pathlib import Path

_p = Path(__file__).resolve()
ROOT = _p.parents[3] if len(_p.parents) > 3 and (_p.parents[3] / "dataset").exists() else _p.parents[2]
DATA_DIR = Path(os.environ.get("ER_DATA_DIR", ROOT / "dataset"))
CACHE_DIR = Path(os.environ.get("ER_CACHE_DIR", ROOT / "cache"))
OUTPUT_DIR = Path(os.environ.get("ER_OUTPUT_DIR", ROOT / "output"))

SEED = 42

for d in (CACHE_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)


def raw_path(split: str, name: str) -> Path:
    return DATA_DIR / split / f"{split}_{name}.tsv"


def cache_path(name: str) -> Path:
    return CACHE_DIR / name
