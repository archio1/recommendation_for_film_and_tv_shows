"""
Search regression tests for `UniversalSearchEngine.search()`.

The point of these is **lock down behavior the user actually depends
on**: typing a known title in the bot must return that title. We split
this into two flavors:

- **Should-pass**: well-known titles that ARE in the parquet — search
  must rank them at the top.
- **Low-popularity regression**: cult fantasy TV with low TMDb
  popularity used to be dropped by the `popularity < 10` floor in the
  live-TMDb fallback even on an exact title match. The fix bypasses that
  floor when the title matches strongly (rel_score >= 0.9). Because the
  engine fixtures run with no live TMDb (tmdb_api_key=None), we exercise
  this deterministically with an injected fake client.

Tests skip cleanly if the per-domain engine fixtures can't be built.
"""

from __future__ import annotations

import pytest


# --------------------------------------------------------------------------
# Movies — should pass
# --------------------------------------------------------------------------

KNOWN_MOVIES = [
    "Inception",
    "The Matrix",
    "Pulp Fiction",
    "John Wick",
    "Fight Club",
]


@pytest.mark.parametrize("query", KNOWN_MOVIES)
def test_search_finds_known_movies(movies_engine_real, query):
    res = movies_engine_real.search(query, media_type="movie", limit=10)
    titles = [item.title for item in res.results]
    assert res.results, f"no results for {query!r}; got {titles}"
    # The exact title should rank in the top 3 for these classics.
    top3 = [t.lower() for t in titles[:3]]
    assert query.lower() in top3, (
        f"{query!r} not in top-3 results: {titles[:3]}"
    )


# --------------------------------------------------------------------------
# TV — should pass (commonly-known shows present in the Trakt collection)
# --------------------------------------------------------------------------

KNOWN_TV_SHOWS = [
    "Game of Thrones",
    "Breaking Bad",
    "The Boys",
    "House of the Dragon",
    "Stranger Things",
]


@pytest.mark.parametrize("query", KNOWN_TV_SHOWS)
def test_search_finds_known_tv_shows(tv_engine_real, query):
    res = tv_engine_real.search(query, media_type="tv", limit=10)
    titles = [item.title for item in res.results]
    assert res.results, f"no results for {query!r}; got {titles}"
    top3 = [t.lower() for t in titles[:3]]
    assert query.lower() in top3, (
        f"{query!r} not in top-3 results: {titles[:3]}"
    )


# --------------------------------------------------------------------------
# Filter & edge cases
# --------------------------------------------------------------------------

def test_search_movies_engine_only_returns_movies(movies_engine_real):
    """Per-domain engine searches must not leak the other type."""
    res = movies_engine_real.search("Game", media_type="movie", limit=10)
    bad = [r for r in res.results if r.media_type != "movie"]
    assert not bad, f"movies engine returned non-movie items: {bad}"


def test_search_tv_engine_only_returns_tv(tv_engine_real):
    res = tv_engine_real.search("Friends", media_type="tv", limit=10)
    bad = [r for r in res.results if r.media_type != "tv"]
    assert not bad, f"tv engine returned non-tv items: {bad}"


def test_search_handles_empty_query_gracefully(movies_engine_real):
    """Empty input must not raise — bot users do hit Enter on empty input."""
    res = movies_engine_real.search("", media_type="movie", limit=5)
    assert isinstance(res.results, list)


def test_search_handles_garbage_query_gracefully(movies_engine_real):
    res = movies_engine_real.search("xqz!@#$%qzx", media_type="movie", limit=5)
    assert isinstance(res.results, list)


# --------------------------------------------------------------------------
# Low-popularity regression: a strong title match must survive the TMDb-live
# popularity floor (universal_search.py popularity<10 gate).
# --------------------------------------------------------------------------

LOW_POP_FANTASY_TV = [
    "Shadow and Bone",
    "Willow",
]


class _FakeTMDBClient:
    """Minimal stand-in for TMDBLiveClient: returns one low-popularity,
    exact-title TV match for any query, mirroring search_multi()."""

    def __init__(self, title, tmdb_id, popularity):
        self._title = title
        self._tmdb_id = tmdb_id
        self._popularity = popularity

    def search_multi(self, query, limit=10):
        from universal_search import UniversalMediaItem

        return [
            UniversalMediaItem(
                tmdb_id=self._tmdb_id,
                media_type="tv",
                title=self._title,
                year=2021,
                genres=["Fantasy"],
                popularity=self._popularity,  # below the 10.0 floor
                vote_average=7.5,
                vote_count=400,
                source="tmdb_live",
            )
        ]


@pytest.mark.parametrize("query", LOW_POP_FANTASY_TV)
def test_search_finds_low_popularity_tv_shows(tv_engine_real, query, monkeypatch):
    # The show is absent from the local parquet, so search() falls through to
    # the live path. The real fixture has no TMDb client (tmdb_api_key=None),
    # so inject a fake one returning the show with popularity < 10. The fixed
    # popularity gate must let it through because the title matches exactly
    # (rel_score == 1.0). Without the fix the floor would drop it.
    fake = _FakeTMDBClient(title=query, tmdb_id=900000 + len(query), popularity=5.0)
    monkeypatch.setattr(tv_engine_real, "tmdb_client", fake)
    # Don't pollute the session-scoped engine's hot cache with the fake item.
    monkeypatch.setattr(
        tv_engine_real.hot_cache, "upsert_batch", lambda *a, **k: None
    )

    res = tv_engine_real.search(query, media_type="tv", limit=10)
    titles_lower = [item.title.lower() for item in res.results]
    assert query.lower() in titles_lower, (
        f"{query!r} dropped by popularity floor despite exact title match; "
        f"got {titles_lower}"
    )
