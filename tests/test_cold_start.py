"""Tests for ColdStartIngestor — TMDb → HotCache → SBERT → FAISS pipeline.

TMDBLiveClient is faked (no network); SBERT encoder is faked (no model
download) by pre-setting `ingestor._encoder`. FaissCatalog and both
HotCache instances are real — they're cheap on a handful of items.
"""

from __future__ import annotations

import numpy as np
import pytest

from recommendation_system.models.gnn.cold_start import ColdStartIngestor
from recommendation_system.models.gnn.faiss_bridge import (
    TV_OFFSET,
    FaissCatalog,
)
from recommendation_system.models.gnn.universal_search import (
    HotCache,
    UniversalMediaItem,
)


DIM = 384


class FakeTMDBClient:
    """Stand-in for TMDBLiveClient.get_details — no network."""

    def __init__(self):
        self.calls: list[tuple[int, str]] = []
        self._responses: dict[tuple[int, str], UniversalMediaItem] = {}

    def register(self, raw_tmdb: int, media_type: str, item: UniversalMediaItem):
        self._responses[(raw_tmdb, media_type)] = item

    def get_details(self, tid: int, mt: str = "movie"):
        self.calls.append((tid, mt))
        return self._responses.get((int(tid), mt))


class FakeEncoder:
    """Deterministic stand-in for a SentenceTransformer."""

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.encode_calls = 0

    def encode(self, text, normalize_embeddings: bool = True):
        self.encode_calls += 1
        # Make the vector depend on text length so different items differ.
        v = np.full(self.dim, 0.1 * (len(text) % 7 + 1), dtype=np.float32)
        v[self.encode_calls % self.dim] += 1.0
        if normalize_embeddings:
            v = v / np.linalg.norm(v)
        return v


def _make_tmdb_item(tmdb_id_raw: int, media_type: str, title: str) -> UniversalMediaItem:
    return UniversalMediaItem(
        tmdb_id=tmdb_id_raw,
        media_type=media_type,
        title=title,
        year=2024,
        genres=["Drama"],
        keywords=[],
        overview=f"A story about {title}, long enough to embed.",
        source="tmdb_live",
        popularity=10.0,
        vote_average=7.5,
        vote_count=100,
    )


@pytest.fixture
def fake_tmdb():
    return FakeTMDBClient()


@pytest.fixture
def faiss_catalog():
    return FaissCatalog(embedding_dim=DIM)


@pytest.fixture
def hot_caches(tmp_path):
    movies = HotCache(tmp_path / "movies.db")
    tv = HotCache(tmp_path / "tv.db")
    return movies, tv


@pytest.fixture
def ingestor(tmp_path, fake_tmdb, faiss_catalog, hot_caches):
    movies_hc, tv_hc = hot_caches
    ing = ColdStartIngestor(
        tmdb_client=fake_tmdb,
        faiss_catalog=faiss_catalog,
        movies_hot_cache=movies_hc,
        tv_hot_cache=tv_hc,
        faiss_index_path=tmp_path / "catalog.faiss",
        faiss_meta_path=tmp_path / "catalog_meta.json",
    )
    # Pre-inject the encoder so the real SBERT never loads during tests.
    ing._encoder = FakeEncoder()
    return ing


# --- canonical tmdb ---------------------------------------------------------

def test_canonical_tmdb_movie_is_identity():
    assert ColdStartIngestor._canonical_tmdb(603, "movie") == (603, 603)


def test_canonical_tmdb_tv_applies_offset_if_missing():
    assert ColdStartIngestor._canonical_tmdb(1399, "tv") == (1399, 1399 + TV_OFFSET)


def test_canonical_tmdb_tv_respects_already_offset_input():
    assert ColdStartIngestor._canonical_tmdb(1399 + TV_OFFSET, "tv") == (
        1399,
        1399 + TV_OFFSET,
    )


def test_canonical_tmdb_rejects_bad_media_type():
    with pytest.raises(ValueError):
        ColdStartIngestor._canonical_tmdb(1, "anime")


# --- movie cold-start -------------------------------------------------------

def test_ensure_movie_hits_tmdb_and_populates_everything(
    ingestor, fake_tmdb, faiss_catalog, hot_caches
):
    movies_hc, _ = hot_caches
    fake_tmdb.register(7777, "movie", _make_tmdb_item(7777, "movie", "New Movie"))

    item = ingestor.ensure(7777, "movie")

    assert item is not None
    assert item.tmdb_id == 7777
    assert item.media_type == "movie"
    assert item.in_graph is False
    assert item.source == "tmdb_live"

    assert FaissCatalog._faiss_id(7777, "movie") in faiss_catalog.mapping
    assert movies_hc.get_by_tmdb_id(7777) is not None
    assert fake_tmdb.calls == [(7777, "movie")]

    # persist happened — both files should exist
    assert ingestor.faiss_index_path.exists()
    assert ingestor.faiss_meta_path.exists()


# --- tv cold-start (both raw and offset inputs) -----------------------------

def test_ensure_tv_accepts_raw_tmdb_and_applies_offset(
    ingestor, fake_tmdb, faiss_catalog, hot_caches
):
    _, tv_hc = hot_caches
    fake_tmdb.register(8888, "tv", _make_tmdb_item(8888, "tv", "New Show"))

    item = ingestor.ensure(8888, "tv")

    assert item is not None
    # stored tmdb must carry offset (matches parquet convention)
    assert item.tmdb_id == 8888 + TV_OFFSET
    assert item.media_type == "tv"
    assert item.in_graph is False

    # FAISS key is based on raw+offset — same as parquet items
    assert (8888 + TV_OFFSET) in faiss_catalog.mapping
    # TV HotCache indexed by stored tmdb
    assert tv_hc.get_by_tmdb_id(8888 + TV_OFFSET) is not None
    # TMDb was called with the RAW id
    assert fake_tmdb.calls == [(8888, "tv")]


def test_ensure_tv_accepts_already_offset_tmdb(
    ingestor, fake_tmdb, faiss_catalog
):
    fake_tmdb.register(9999, "tv", _make_tmdb_item(9999, "tv", "Another Show"))

    item = ingestor.ensure(9999 + TV_OFFSET, "tv")

    assert item is not None
    assert item.tmdb_id == 9999 + TV_OFFSET
    # TMDb was still called with raw id, not double-offset
    assert fake_tmdb.calls == [(9999, "tv")]


# --- isolation between domains ----------------------------------------------

def test_movie_and_tv_go_to_correct_hot_caches(
    ingestor, fake_tmdb, hot_caches
):
    movies_hc, tv_hc = hot_caches
    fake_tmdb.register(111, "movie", _make_tmdb_item(111, "movie", "M"))
    fake_tmdb.register(222, "tv", _make_tmdb_item(222, "tv", "T"))

    ingestor.ensure(111, "movie")
    ingestor.ensure(222, "tv")

    assert movies_hc.get_by_tmdb_id(111) is not None
    assert movies_hc.get_by_tmdb_id(222 + TV_OFFSET) is None
    assert tv_hc.get_by_tmdb_id(222 + TV_OFFSET) is not None
    assert tv_hc.get_by_tmdb_id(111) is None


# --- idempotency ------------------------------------------------------------

def test_ensure_is_idempotent_no_second_tmdb_call(
    ingestor, fake_tmdb, faiss_catalog
):
    fake_tmdb.register(333, "movie", _make_tmdb_item(333, "movie", "X"))

    first = ingestor.ensure(333, "movie")
    second = ingestor.ensure(333, "movie")

    assert first is not None and second is not None
    assert first.tmdb_id == second.tmdb_id
    # TMDb called exactly once
    assert len(fake_tmdb.calls) == 1
    # FAISS stays size 1 (no duplicate add)
    assert faiss_catalog.size == 1
    # Encoder ran exactly once
    assert ingestor._encoder.encode_calls == 1


def test_ensure_tv_idempotent_across_raw_and_offset_forms(
    ingestor, fake_tmdb, faiss_catalog
):
    fake_tmdb.register(444, "tv", _make_tmdb_item(444, "tv", "Y"))

    ingestor.ensure(444, "tv")
    ingestor.ensure(444 + TV_OFFSET, "tv")

    assert len(fake_tmdb.calls) == 1
    assert faiss_catalog.size == 1


# --- negative paths ---------------------------------------------------------

def test_ensure_returns_none_when_tmdb_has_no_record(
    ingestor, fake_tmdb, faiss_catalog, hot_caches
):
    movies_hc, _ = hot_caches
    # Nothing registered for 555.

    result = ingestor.ensure(555, "movie")

    assert result is None
    assert fake_tmdb.calls == [(555, "movie")]
    assert faiss_catalog.size == 0
    assert movies_hc.get_by_tmdb_id(555) is None
    # No persist file either, since nothing was added.
    assert not ingestor.faiss_index_path.exists()


def test_ensure_rejects_bad_media_type(ingestor):
    with pytest.raises(ValueError):
        ingestor.ensure(1, "anime")


# --- persist round-trip -----------------------------------------------------

def test_ensure_persists_reloadable_catalog(
    ingestor, fake_tmdb, faiss_catalog
):
    fake_tmdb.register(603, "movie", _make_tmdb_item(603, "movie", "Matrix-like"))
    fake_tmdb.register(1399, "tv", _make_tmdb_item(1399, "tv", "Thrones-like"))

    ingestor.ensure(603, "movie")
    ingestor.ensure(1399, "tv")

    reloaded = FaissCatalog.load(
        ingestor.faiss_index_path, ingestor.faiss_meta_path
    )
    assert reloaded.size == 2
    assert reloaded.mapping == faiss_catalog.mapping


# --- integration with DualDomainEngine -------------------------------------

def test_dual_domain_reconstruct_query_triggers_cold_start(
    ingestor, fake_tmdb, faiss_catalog
):
    """_reconstruct_query should call ensure() for unknown tmdb_ids."""
    from recommendation_system.models.gnn.dual_domain_engine import (
        DualDomainEngine,
    )

    fake_tmdb.register(1234, "movie", _make_tmdb_item(1234, "movie", "Cold"))

    # Build a DualDomainEngine with stub engines — we only need _reconstruct_query.
    class _StubEngine:
        pass
    engine = DualDomainEngine(
        movies_engine=_StubEngine(),
        tv_engine=_StubEngine(),
        faiss_catalog=faiss_catalog,
        cold_start=ingestor,
    )

    query = engine._reconstruct_query([1234])

    assert query is not None
    assert query.shape == (DIM,)
    # Cold-start actually ran
    assert fake_tmdb.calls == [(1234, "movie")]
    assert FaissCatalog._faiss_id(1234, "movie") in faiss_catalog.mapping
