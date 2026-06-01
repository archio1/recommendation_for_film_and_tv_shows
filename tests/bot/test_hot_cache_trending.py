"""
HotCache.get_trending — vote-count filter regressions.

Locks down the vote-count filter behaviour:
- Default min_vote_count=100 cuts TMDB-noise (high `popularity` but
  almost no actual ratings — typical of marketing-heavy releases).
- Override path returns everything when needed.
- media_type filter composes with the vote filter (AND, not OR).
- Only source='trending' rows surface — search-cache / live-fetched
  rows must not leak into /trending output.
"""

from __future__ import annotations

import pytest

from universal_search import HotCache, UniversalMediaItem


def _item(
    tmdb_id: int,
    *,
    media_type: str,
    vote_count: int,
    popularity: float,
    source: str = "trending",
    title: str | None = None,
) -> UniversalMediaItem:
    return UniversalMediaItem(
        tmdb_id=tmdb_id,
        media_type=media_type,
        title=title or f"Item {tmdb_id}",
        year=2024,
        genres=[],
        popularity=popularity,
        vote_count=vote_count,
        source=source,
    )


@pytest.fixture
def trending_cache(tmp_path) -> HotCache:
    """Two legit hits and two noise rows per domain."""
    cache = HotCache(tmp_path / "hot.db")
    cache.upsert_batch([
        _item(1, media_type="movie", vote_count=200, popularity=90.0),  # legit
        _item(2, media_type="movie", vote_count=50, popularity=80.0),   # noise
        _item(3, media_type="tv", vote_count=500, popularity=70.0),     # legit
        _item(4, media_type="tv", vote_count=30, popularity=85.0),      # noise
    ])
    return cache


def test_default_min_vote_count_filters_noise(trending_cache):
    """Default 100-vote cutoff drops items with too few real ratings.
    Noise items #2 (vote_count=50) and #4 (vote_count=30) get cut even
    though their popularity is high."""
    out = trending_cache.get_trending()
    assert {i.tmdb_id for i in out} == {1, 3}


def test_min_vote_count_zero_returns_everything(trending_cache):
    """Override to 0 disables the filter — escape hatch for callers
    who want raw popularity ordering."""
    out = trending_cache.get_trending(min_vote_count=0)
    assert {i.tmdb_id for i in out} == {1, 2, 3, 4}


def test_media_type_filter_composes_with_vote_filter(trending_cache):
    """media_type='movie' narrows to movies *and* keeps the vote cutoff —
    a regression where AND turned into OR would surface low-vote tv."""
    out = trending_cache.get_trending(media_type="movie")
    assert [i.tmdb_id for i in out] == [1]


def test_non_trending_rows_excluded(tmp_path):
    """Items from other sources (trained / tmdb_live / cold-start) must
    not surface in /trending — the source='trending' clause is the only
    thing keeping search-cache rows out of the trends list."""
    cache = HotCache(tmp_path / "hot.db")
    cache.upsert_batch([
        _item(10, media_type="movie", vote_count=1000, popularity=100.0, source="trending"),
        _item(11, media_type="movie", vote_count=1000, popularity=100.0, source="tmdb_live"),
    ])
    out = cache.get_trending()
    assert [i.tmdb_id for i in out] == [10]
