from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def save_model_artifact(artifact: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_model_artifact(path: Path) -> dict[str, Any]:
    return joblib.load(path)

