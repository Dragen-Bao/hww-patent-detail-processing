"""Configuration loader for the independent patent-detail pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(__file__).with_name("config") / "default.json"


def load_config(path: Path = CONFIG_PATH, project_root: Path | None = None) -> dict[str, Any]:
    """Load one JSON configuration and resolve project-relative paths."""

    root = project_root or Path(__file__).resolve().parents[2]
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("patent_csv", "dictionary", "output_root"):
        config[key] = str((root / config[key]).resolve())
    return config
