"""Tests for DualDomainEngine — router over 2× UniversalSearchEngine + FaissCatalog.

Uses a mock metadata (3 movies + 3 tv shows), a real FaissCatalog over random
unit embeddings, and real UniversalSearchEngine instances (they're cheap on
tiny metadata). No LightGCN: inference_engine=None skips tier 1 everywhere,
leaving tier 2 (content) + the FAISS bridge as the paths under test.

Overview texts are kept < 10 chars so EnhancedContentEngine._get_embedding
returns None without triggering a live SBERT download.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recommendation_system.models.gnn.dual_domain_engine import DualDomainEngine
from recommendation_system.models.gnn.faiss_bridge import TV_OFFSET, FaissCatalog
from recommendation_system.models.gnn.universal_search import UniversalSearchEngine


DIM = 384


def _unit(rng: np.random.Generator, dim: int = DIM) -> np.ndarray:
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _movie_row(item_id: int, tmdb_id: int, title: str, genres: list[str]) -> dict:
    return {
        "item_id": item_id,
        "tmdb_id": tmdb_id,
        "type": "movie",
        "title": title,
        "title_ru": None,
        "year": 2010,
        "genres": genres,
        "keywords": [],
        "overview": "x",
        "overview_ru": None,
        "vote_average": 7.0,
        "vote_count": 100,
        "popularity": 50.0,
    }


def _tv_row(item_id: int, tmdb_raw: int, title: str, genres: list[str]) -> dict:
    return {
        "item_id": item_id,
        "tmdb_id": tmdb_raw + TV_OFFSET,
        "type": "tv",
        "title": title,
        "title_ru": None,
        "year": 2015,
        "genres": genres,
        "keywords": [],
        "overview": "x",
        "overview_ru": None,
        "vote_average": 8.0,
        "vote_count": 200,
        "popularity": 60.0,
    }


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def movies_metadata():
    return pd.DataFrame([
        _movie_row(0, 603, "The Matrix", ["Sci-Fi"]),
        _movie_row(1, 27205, "Inception", ["Sci-Fi"]),
        _movie_row(2, 155, "The Dark Knight", ["Action"]),
    ])


@pytest.fixture
def tv_metadata():
    return pd.DataFrame([
        _tv_row(0, 1399, "Game of Thrones", ["Fantasy"]),
        _tv_row(1, 1396, "Breaking Bad", ["Drama"]),
        _tv_row(2, 1668, "Friends", ["Comedy"]),
    ])


@pytest.fixture
def faiss_catalog(rng):
    cat = FaissCatalog(embedding_dim=DIM)
    for tid in (603, 27205, 155):
        cat.add(tmdb_id=tid, media_type="movie", embedding=_unit(rng))
    for tid in (1399, 1396, 1668):
        cat.add(tmdb_id=tid, media_type="tv", embedding=_unit(rng))
    return cat


@pytest.fixture
def movies_engine(movies_metadata, tmp_path):
    cache_dir = tmp_path / "movies_cache"
    cache_dir.mkdir()
    return UniversalSearchEngine(
        metadata=movies_metadata,
        cache_dir=cache_dir,
        tmdb_api_key=None,
        inference_engine=None,
        embeddings_path=None,
        model_num_items=0,
    )


@pytest.fixture
def tv_engine(tv_metadata, tmp_path):
    cache_dir = tmp_path / "tv_cache"
    cache_dir.mkdir()
    return UniversalSearchEngine(
        metadata=tv_metadata,
        cache_dir=cache_dir,
        tmdb_api_key=None,
        inference_engine=None,
        embeddings_path=None,
        model_num_items=0,
    )


@pytest.fixture
def engine(movies_engine, tv_engine, faiss_catalog):
    return DualDomainEngine(movies_engine, tv_engine, faiss_catalog)


# --- static helpers ---------------------------------------------------------

def test_split_by_domain_separates_on_tv_offset():
    movie_ids, tv_ids = DualDomainEngine._split_by_domain(
        [603, 27205, 1399 + TV_OFFSET, 1396 + TV_OFFSET]
    )
    assert movie_ids == [603, 27205]
    assert tv_ids == [1399 + TV_OFFSET, 1396 + TV_OFFSET]


def test_split_by_domain_empty_and_none():
    assert DualDomainEngine._split_by_domain(None) == ([], [])
    assert DualDomainEngine._split_by_domain([]) == ([], [])
    movie_ids, tv_ids = DualDomainEngine._split_by_domain(
        [603, None, 1399 + TV_OFFSET]
    )
    assert movie_ids == [603]
    assert tv_ids == [1399 + TV_OFFSET]


def test_to_item_ids_splits_known_unknown(movies_engine):
    known, unknown = DualDomainEngine._to_item_ids([603, 99999], movies_engine)
    # 603 → item_id 0 (known); 99999 has no mapping
    assert known == [0]
    assert unknown == [99999]


def test_to_item_ids_empty(movies_engine):
    assert DualDomainEngine._to_item_ids([], movies_engine) == ([], [])


def test_split_quota_pure_movie():
    assert DualDomainEngine._split_quota(10, 5, 0) == (10, 0)


def test_split_quota_pure_tv():
    assert DualDomainEngine._split_quota(10, 0, 3) == (0, 10)


def test_split_quota_proportional():
    q_movie, q_tv = DualDomainEngine._split_quota(10, 6, 2)
    assert q_movie + q_tv == 10
    assert q_movie == 8 and q_tv == 2


def test_split_quota_guarantees_both_sides_get_at_least_one():
    # Extreme imbalance, but both sides non-zero → each must keep ≥1 slot.
    q_movie, q_tv = DualDomainEngine._split_quota(10, 100, 1)
    assert q_movie >= 1 and q_tv >= 1
    assert q_movie + q_tv == 10


# --- recs_movie / recs_tv ---------------------------------------------------

def test_recs_movie_returns_only_movies(engine):
    recs = engine.recs_movie(liked_tmdb_ids=[603], top_k=3)
    assert recs, "content tier should produce at least one candidate"
    for r in recs:
        assert r.media_type == "movie"
    # liked item must never appear in its own recommendations
    assert 603 not in {r.tmdb_id for r in recs}


def test_recs_movie_ignores_tv_ids(engine):
    # Only tv ids → no movie ids to drive the movie engine → empty.
    assert engine.recs_movie(liked_tmdb_ids=[1399 + TV_OFFSET], top_k=3) == []


def test_recs_tv_returns_only_tv(engine):
    recs = engine.recs_tv(liked_tmdb_ids=[1399 + TV_OFFSET], top_k=3)
    assert recs
    for r in recs:
        assert r.media_type == "tv"
    assert (1399 + TV_OFFSET) not in {r.tmdb_id for r in recs}


def test_recs_tv_ignores_movie_ids(engine):
    assert engine.recs_tv(liked_tmdb_ids=[603], top_k=3) == []


def test_recs_empty_and_none_inputs(engine):
    assert engine.recs_movie(liked_tmdb_ids=None, top_k=3) == []
    assert engine.recs_movie(liked_tmdb_ids=[], top_k=3) == []
    assert engine.recs_tv(liked_tmdb_ids=None, top_k=3) == []
    assert engine.recs_tv(liked_tmdb_ids=[], top_k=3) == []
    assert engine.recs_all(liked_tmdb_ids=None, top_k=3) == []
    assert engine.recs_all(liked_tmdb_ids=[], top_k=3) == []


def test_recs_movie_unknown_tmdb_returns_empty(engine):
    # Unknown tmdb_id, no hot_cache entry, no tmdb_client → engine returns [].
    assert engine.recs_movie(liked_tmdb_ids=[7_777_777], top_k=3) == []


# --- recs_cross -------------------------------------------------------------

def test_recs_cross_movie_to_tv(engine):
    recs = engine.recs_cross(
        liked_tmdb_ids=[603], target_media_type="tv", top_k=2
    )
    assert recs
    for r in recs:
        assert r.media_type == "tv"
    assert 603 not in {r.tmdb_id for r in recs}


def test_recs_cross_tv_to_movie(engine):
    recs = engine.recs_cross(
        liked_tmdb_ids=[1399 + TV_OFFSET], target_media_type="movie", top_k=2
    )
    assert recs
    for r in recs:
        assert r.media_type == "movie"


def test_recs_cross_self_excludes_liked_items(engine):
    # target=tv with a liked tv item — the liked item itself scores 1.0
    # against its own embedding but must be filtered out.
    recs = engine.recs_cross(
        liked_tmdb_ids=[1399 + TV_OFFSET], target_media_type="tv", top_k=5
    )
    returned = {r.tmdb_id for r in recs}
    assert (1399 + TV_OFFSET) not in returned
    # Remaining tv items should surface (stored with +TV_OFFSET in metadata).
    assert returned.issubset({1396 + TV_OFFSET, 1668 + TV_OFFSET})


def test_recs_cross_rejects_bad_media_type(engine):
    with pytest.raises(ValueError):
        engine.recs_cross(
            liked_tmdb_ids=[603], target_media_type="anime", top_k=3
        )


def test_recs_cross_empty_input(engine):
    assert engine.recs_cross(
        liked_tmdb_ids=None, target_media_type="movie", top_k=3
    ) == []
    assert engine.recs_cross(
        liked_tmdb_ids=[], target_media_type="movie", top_k=3
    ) == []


def test_recs_cross_unknown_ids_return_empty(engine):
    # Not in FAISS → _reconstruct_query returns None → [].
    assert engine.recs_cross(
        liked_tmdb_ids=[99999], target_media_type="tv", top_k=3
    ) == []


# --- recs_all ---------------------------------------------------------------

def test_recs_all_mixes_domains_when_input_is_mixed(engine):
    recs = engine.recs_all(
        liked_tmdb_ids=[603, 1399 + TV_OFFSET], top_k=4
    )
    assert recs
    assert len(recs) <= 4
    assert {r.media_type for r in recs} == {"movie", "tv"}


def test_recs_all_no_duplicates(engine):
    recs = engine.recs_all(
        liked_tmdb_ids=[603, 1399 + TV_OFFSET], top_k=6
    )
    tmdb_ids = [r.tmdb_id for r in recs]
    assert len(tmdb_ids) == len(set(tmdb_ids))


def test_recs_all_movie_only_input(engine):
    recs = engine.recs_all(liked_tmdb_ids=[603], top_k=2)
    assert recs
    for r in recs:
        assert r.media_type == "movie"


def test_recs_all_tv_only_input(engine):
    recs = engine.recs_all(liked_tmdb_ids=[1399 + TV_OFFSET], top_k=2)
    assert recs
    for r in recs:
        assert r.media_type == "tv"
