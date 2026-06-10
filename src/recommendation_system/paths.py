"""
Central project paths — single source of truth for everything on disk.

Leaf module: stdlib only, no heavy imports; safe to import from anywhere.

Layout:
    <root>/data/{raw,processed/{movies,tv,cache}}   — datasets (outside package)
    <root>/models/{movies,tv}                       — checkpoints (outside package)
    <root>/reports/figures                          — reports (outside package)
    <package>/faiss_index                           — the only package-internal data
    <package>/models/gnn/config/genre_map.json      — dataset config

Override: set RECSYS_PROJECT_ROOT to relocate everything that lives outside
the package (data / models / reports). Package-internal paths (FAISS index,
genre map) always follow the installed package location.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent  # .../src/recommendation_system


def _find_project_root() -> Path:
    env = os.environ.get("RECSYS_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    for parent in PACKAGE_DIR.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    # src-layout fallback: src/recommendation_system -> parents[1] == repo root
    return PACKAGE_DIR.parents[1]


PROJECT_ROOT = _find_project_root()

# --- data (outside the package) ---------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MOVIES_DIR = PROCESSED_DIR / "movies"
TV_DIR = PROCESSED_DIR / "tv"
CACHE_DIR = PROCESSED_DIR / "cache"

# --- model checkpoints -------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_CHECKPOINT_VERSION = "v4"


def default_checkpoint(domain: str) -> Path:
    """Fallback checkpoint the bot/GUI/tests load when no newer one is chosen."""
    return MODELS_DIR / domain / f"lightgcn_{domain}_best_{DEFAULT_CHECKPOINT_VERSION}.pt"


MOVIES_CHECKPOINT = default_checkpoint("movies")
TV_CHECKPOINT = default_checkpoint("tv")

# --- package-internal artifacts ----------------------------------------------
FAISS_DIR = PACKAGE_DIR / "faiss_index"
FAISS_INDEX = FAISS_DIR / "catalog.faiss"
FAISS_META = FAISS_DIR / "catalog_meta.json"
GENRE_MAP_PATH = PACKAGE_DIR / "models" / "gnn" / "config" / "genre_map.json"

# --- reports -------------------------------------------------------------------
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- env file ------------------------------------------------------------------
ENV_FILE = PROJECT_ROOT / ".env"
