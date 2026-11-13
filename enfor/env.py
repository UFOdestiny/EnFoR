"""Filesystem locations, read from the environment or the repository-level ``.env`` file.

Variables already set in the environment take precedence over ``.env``. Relative paths are resolved against the
repository root.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Export ``KEY=VALUE`` lines of ``path`` that are not already set."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.split(" #", 1)[0].strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    path = Path(value).expanduser() if value else Path(default)
    return path if path.is_absolute() else REPO_ROOT / path


load_dotenv()

MODELS_DIR = env_path("ENFOR_MODELS", REPO_ROOT / "resource" / "models")
DATASETS_DIR = env_path("ENFOR_DATASETS", REPO_ROOT / "resource" / "datasets")
OUTPUT_DIR = env_path("ENFOR_OUTPUT", REPO_ROOT / "outputs")
DATA_DIR = REPO_ROOT / "data"   # evaluation sets derived for the paper (shipped with the code)
