"""
Shared fixtures.

Two layers of fixtures live here:

1. **Lightweight mocks** (`mock_metadata`, `temp_db_path`) — used by
   existing unit tests that should never touch real artifacts.
2. **Real-engine fixtures** (`movies_engine_real`, `tv_engine_real`,
   `dual_engine_real`, `faiss_catalog_real`) — session-scoped, load
   once. Each `pytest.skip()`s if its on-disk artifacts are missing,
   so the suite runs cleanly in a clean checkout / CI without models.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

# Project code is importable as the `recommendation_system` package: either via
# the editable install (`pip install -e .`) or via `pythonpath = ["src"]` in
# pyproject's pytest config. All on-disk paths come from the central paths module.
from recommendation_system.paths import (
    CACHE_DIR,
    FAISS_INDEX,
    FAISS_META,
    MOVIES_CHECKPOINT,
    MOVIES_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    TV_CHECKPOINT,
    TV_DIR,
)


# --------------------------------------------------------------------------
# Lightweight mocks (kept for legacy tests)
# --------------------------------------------------------------------------

@pytest.fixture
def mock_metadata():
    """Расширенные тестовые данные для проверки фильтрации"""
    return pd.DataFrame({
        'item_id': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'title': [
            'Saw', 'Saw II', 'Saw III',
            'The Matrix', 'The Matrix Reloaded',
            'Shrek', 'Shrek 2',
            'Inception', 'Interstellar', 'The Dark Knight'
        ],
        'year': [2004, 2005, 2006, 1999, 2003, 2001, 2004, 2010, 2014, 2008],
        'genres': [['Horror'], ['Horror'], ['Horror'], ['Sci-Fi'], ['Sci-Fi'], ['Animation'], ['Animation'], ['Sci-Fi'], ['Sci-Fi'], ['Action']],
        'tmdb_id': [176, 215, 824, 603, 604, 808, 809, 27205, 157336, 155],
        'type': ['movie'] * 10
    })


@pytest.fixture
def temp_db_path(tmp_path):
    """Создает путь к временному файлу базы данных SQLite для тестов"""
    return tmp_path / "test_translations.db"


# --------------------------------------------------------------------------
# Real-engine fixtures (session-scoped, skip-if-missing)
# --------------------------------------------------------------------------

def _require(*paths: Path) -> None:
    for p in paths:
        if not p.exists():
            pytest.skip(f"required artifact missing: {p}", allow_module_level=False)


def _build_real_engine(dataset_dir: Path, checkpoint: Path):
    from recommendation_system.models.gnn.inference_engine import InferenceEngine
    from recommendation_system.models.gnn.universal_search import UniversalSearchEngine

    _require(dataset_dir / "items_metadata_final.parquet", checkpoint)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    infer = InferenceEngine(dataset_dir, checkpoint, device="cpu")
    ok, msg = infer.load_resources()
    if not ok:
        pytest.skip(f"InferenceEngine failed to load {dataset_dir.name}: {msg}")

    metadata = pd.read_parquet(dataset_dir / "items_metadata_final.parquet")
    embeddings = dataset_dir / "overview_embeddings.npy"
    return UniversalSearchEngine(
        metadata=metadata,
        cache_dir=CACHE_DIR,
        tmdb_api_key=None,  # no live TMDb in tests
        inference_engine=infer,
        embeddings_path=embeddings if embeddings.exists() else None,
        model_num_items=infer.model.num_items,
    )


@pytest.fixture(scope="session")
def movies_engine_real():
    # Mirror production (movie_bot): movies engine runs with popularity de-bias.
    engine = _build_real_engine(MOVIES_DIR, MOVIES_CHECKPOINT)
    engine.popularity_debias = 0.5
    return engine


@pytest.fixture(scope="session")
def tv_engine_real():
    return _build_real_engine(TV_DIR, TV_CHECKPOINT)


@pytest.fixture(scope="session")
def faiss_catalog_real():
    from recommendation_system.models.gnn.faiss_bridge import FaissCatalog

    _require(FAISS_INDEX, FAISS_META)
    return FaissCatalog.load(FAISS_INDEX, FAISS_META)


@pytest.fixture(scope="session")
def dual_engine_real(movies_engine_real, tv_engine_real, faiss_catalog_real):
    from recommendation_system.models.gnn.dual_domain_engine import DualDomainEngine

    return DualDomainEngine(
        movies_engine=movies_engine_real,
        tv_engine=tv_engine_real,
        faiss_catalog=faiss_catalog_real,
    )


# --------------------------------------------------------------------------
# Quality-report infrastructure
# --------------------------------------------------------------------------

_QUALITY_LOGGER = logging.getLogger("recs.quality")


class _QualityReportsWriter:
    """Accumulates quality metric records and flushes to a JSON file at session end."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: list[dict] = []
        self._meta = {
            "started_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "path": str(path),
        }

    def append(self, test_name: str, scenario: str, metrics: dict) -> None:
        record = {
            "test": test_name,
            "scenario": scenario,
            "metrics": metrics,
        }
        self._entries.append(record)
        _QUALITY_LOGGER.info("%s | %s | %s", test_name, scenario, json.dumps(metrics, default=str))

    def flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": self._meta | {"finished_at": datetime.utcnow().isoformat(timespec="seconds") + "Z"},
            "entries": self._entries,
        }
        self._path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


@pytest.fixture(scope="session")
def reports_writer():
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"quality_trajectory_{timestamp}.json"
    writer = _QualityReportsWriter(path)
    yield writer
    writer.flush()


@pytest.fixture(scope="session")
def popular_tmdb_ids_movies(movies_engine_real):
    from tests._quality_helpers import build_popular_set

    return build_popular_set(movies_engine_real.metadata, pct=0.10)


@pytest.fixture(scope="session")
def popular_tmdb_ids_tv(tv_engine_real):
    from tests._quality_helpers import build_popular_set

    return build_popular_set(tv_engine_real.metadata, pct=0.10)


@pytest.fixture(scope="session")
def graph_neighbors_fn(movies_engine_real, tv_engine_real):
    """
    Returns a callable: (liked_tmdb_ids, media_type, k) -> set[int].
    Dispatches to the appropriate InferenceEngine based on media_type.
    """
    from tests._quality_helpers import compute_graph_neighbors

    def _fn(liked_tmdb_ids, media_type: str, k: int = 50) -> set[int]:
        if media_type == "movie":
            engine = movies_engine_real
        elif media_type == "tv":
            engine = tv_engine_real
        else:
            raise ValueError(f"unknown media_type: {media_type!r}")
        return compute_graph_neighbors(
            inference_engine=engine.inference_engine,
            search_engine=engine,
            liked_tmdb_ids=liked_tmdb_ids,
            k=k,
        )

    return _fn
