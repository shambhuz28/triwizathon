"""Settings files: prices.yaml and thresholds.yaml, editable without touching code."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent / "config"


@lru_cache(maxsize=None)
def prices() -> dict:
    return yaml.safe_load((CONFIG_DIR / "prices.yaml").read_text())


@lru_cache(maxsize=None)
def thresholds() -> dict:
    return yaml.safe_load((CONFIG_DIR / "thresholds.yaml").read_text())


def model_config(model_id: str | None) -> tuple[str, dict]:
    p = prices()
    model_id = model_id if model_id in p["models"] else p["default_model"]
    return model_id, p["models"][model_id]
