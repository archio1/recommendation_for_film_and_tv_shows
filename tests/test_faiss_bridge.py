"""Tests for FaissCatalog — the cross-domain content-bridge index."""

import numpy as np
import pytest

from recommendation_system.models.gnn.faiss_bridge import (
    TV_OFFSET,
    FaissCatalog,
    FaissSearchResult,
)


DIM = 384


def _random_unit(rng: np.random.Generator, dim: int = DIM) -> np.ndarray:
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def populated_catalog(rng):
    cat = FaissCatalog(embedding_dim=DIM)
    # 3 movies + 2 tv, different tmdb_ids
    cat.add(tmdb_id=603, media_type="movie", embedding=_random_unit(rng))
    cat.add(tmdb_id=27205, media_type="movie", embedding=_random_unit(rng))
    cat.add(tmdb_id=155, media_type="movie", embedding=_random_unit(rng))
    cat.add(tmdb_id=1399, media_type="tv", embedding=_random_unit(rng))  # GoT
    cat.add(tmdb_id=1396, media_type="tv", embedding=_random_unit(rng))  # Breaking Bad
    return cat


def test_add_increases_size(rng):
    cat = FaissCatalog(embedding_dim=DIM)
    assert cat.size == 0
    cat.add(tmdb_id=603, media_type="movie", embedding=_random_unit(rng))
    assert cat.size == 1
    cat.add(tmdb_id=1399, media_type="tv", embedding=_random_unit(rng))
    assert cat.size == 2


def test_faiss_id_offset_scheme():
    # movie → raw tmdb_id
    assert FaissCatalog._faiss_id(603, "movie") == 603
    # tv without offset → gets offset applied
    assert FaissCatalog._faiss_id(1399, "tv") == 1399 + TV_OFFSET
    # tv with offset already baked in → kept as-is
    assert FaissCatalog._faiss_id(1399 + TV_OFFSET, "tv") == 1399 + TV_OFFSET


def test_faiss_id_rejects_unknown_media_type():
    with pytest.raises(ValueError):
        FaissCatalog._faiss_id(100, "anime")


def test_add_rejects_wrong_dim():
    cat = FaissCatalog(embedding_dim=DIM)
    with pytest.raises(ValueError):
        cat.add(tmdb_id=1, media_type="movie", embedding=np.zeros(128, dtype=np.float32))


def test_add_normalizes_embedding(rng):
    cat = FaissCatalog(embedding_dim=DIM)
    unit = _random_unit(rng)
    scaled = unit * 7.5  # not unit norm
    cat.add(tmdb_id=603, media_type="movie", embedding=scaled)
    # self-query must score ~1.0 despite the non-unit input
    results = cat.search(unit, top_k=1)
    assert results[0].tmdb_id == 603
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_search_self_match_is_top_result(populated_catalog, rng):
    # Re-encode an item already in the catalog and expect it at position 0.
    query = _random_unit(rng)
    populated_catalog.add(tmdb_id=999, media_type="movie", embedding=query)
    results = populated_catalog.search(query, top_k=3)
    assert len(results) == 3
    assert results[0].tmdb_id == 999
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_search_cosine_matches_numpy(populated_catalog, rng):
    """FAISS IndexFlatIP on normalized vectors == numpy dot product."""
    query = _random_unit(rng)
    # Pack all stored embeddings back out by querying the raw index.
    # We reconstruct via faiss.reconstruct(faiss_id).
    all_fids = list(populated_catalog.mapping.keys())
    stored = np.vstack(
        [populated_catalog.index.reconstruct(fid) for fid in all_fids]
    )
    numpy_scores = stored @ query
    numpy_order = np.argsort(-numpy_scores)

    results = populated_catalog.search(query, top_k=len(all_fids))
    faiss_order_fids = [
        FaissCatalog._faiss_id(r.tmdb_id, r.media_type) for r in results
    ]
    numpy_order_fids = [all_fids[i] for i in numpy_order]
    assert faiss_order_fids == numpy_order_fids

    for r in results:
        fid = FaissCatalog._faiss_id(r.tmdb_id, r.media_type)
        idx = all_fids.index(fid)
        assert r.score == pytest.approx(float(numpy_scores[idx]), abs=1e-5)


def test_search_respects_media_type_filter(populated_catalog, rng):
    query = _random_unit(rng)
    only_tv = populated_catalog.search(query, top_k=5, media_type_filter="tv")
    only_movies = populated_catalog.search(query, top_k=5, media_type_filter="movie")

    assert len(only_tv) == 2
    assert len(only_movies) == 3
    assert all(r.media_type == "tv" for r in only_tv)
    assert all(r.media_type == "movie" for r in only_movies)


def test_search_rejects_bad_filter(populated_catalog, rng):
    with pytest.raises(ValueError):
        populated_catalog.search(_random_unit(rng), top_k=3, media_type_filter="anime")


def test_search_on_empty_catalog():
    cat = FaissCatalog(embedding_dim=DIM)
    assert cat.search(np.zeros(DIM, dtype=np.float32), top_k=5) == []


def test_upsert_replaces_existing_vector(rng):
    cat = FaissCatalog(embedding_dim=DIM)
    v1 = _random_unit(rng)
    v2 = _random_unit(rng)
    cat.add(tmdb_id=603, media_type="movie", embedding=v1)
    cat.add(tmdb_id=603, media_type="movie", embedding=v2)  # upsert

    assert cat.size == 1  # not 2
    # v2 should now be the match, v1 should no longer match perfectly
    top = cat.search(v2, top_k=1)[0]
    assert top.tmdb_id == 603
    assert top.score == pytest.approx(1.0, abs=1e-5)

    top_v1 = cat.search(v1, top_k=1)[0]
    # v1 now matches whatever is stored (v2) — not perfectly
    assert top_v1.score < 0.999


def test_add_batch_equivalent_to_individual(rng):
    cat_single = FaissCatalog(embedding_dim=DIM)
    cat_batch = FaissCatalog(embedding_dim=DIM)

    items = [
        (603, "movie", _random_unit(rng)),
        (1399, "tv", _random_unit(rng)),
        (155, "movie", _random_unit(rng)),
    ]
    for tid, mt, emb in items:
        cat_single.add(tid, mt, emb)
    cat_batch.add_batch(items)

    assert cat_single.mapping == cat_batch.mapping
    assert cat_single.size == cat_batch.size

    query = _random_unit(rng)
    r_single = cat_single.search(query, top_k=3)
    r_batch = cat_batch.search(query, top_k=3)
    assert [r.tmdb_id for r in r_single] == [r.tmdb_id for r in r_batch]
    for a, b in zip(r_single, r_batch):
        assert a.score == pytest.approx(b.score, abs=1e-6)


def test_persist_and_load_roundtrip(populated_catalog, rng, tmp_path):
    index_path = tmp_path / "catalog.faiss"
    meta_path = tmp_path / "catalog_meta.json"
    populated_catalog.persist(index_path, meta_path)

    assert index_path.exists()
    assert meta_path.exists()

    loaded = FaissCatalog.load(index_path, meta_path)
    assert loaded.size == populated_catalog.size
    assert loaded.mapping == populated_catalog.mapping
    assert loaded.embedding_dim == populated_catalog.embedding_dim

    query = _random_unit(rng)
    r_orig = populated_catalog.search(query, top_k=5)
    r_loaded = loaded.search(query, top_k=5)
    assert [r.tmdb_id for r in r_orig] == [r.tmdb_id for r in r_loaded]
    for a, b in zip(r_orig, r_loaded):
        assert a.score == pytest.approx(b.score, abs=1e-6)


def test_search_result_shape(populated_catalog, rng):
    results = populated_catalog.search(_random_unit(rng), top_k=2)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, FaissSearchResult)
        assert isinstance(r.tmdb_id, int)
        assert r.media_type in {"movie", "tv"}
        assert -1.0 <= r.score <= 1.0001  # cosine; allow small fp slack
