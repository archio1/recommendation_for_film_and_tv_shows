"""
Search regression tests for `UniversalSearchEngine.search()`.

The point of these is **lock down behavior the user actually depends
on**: typing a known title in the bot must return that title. We split
this into two flavors:

- **Should-pass**: well-known titles that ARE in the parquet — search
  must rank them at the top.
- **xfail**: titles known to be missing today (cult fantasy with low
  TMDb popularity, or not in Trakt). The bug is in
  `universal_search.py:602` — `popularity < 10` cutoff drops these
  even when TMDb live search returns them. We document the bug as
  `@pytest.mark.xfail` so the test goes green automatically once the
  fix lands.

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
# Known-bug regression — xfail until universal_search.py:602 is fixed
# --------------------------------------------------------------------------

LOW_POP_FANTASY_TV = [
    "Shadow and Bone",
    "Willow",
]


@pytest.mark.xfail(
    reason=(
        "universal_search.py:602 drops items with popularity<10 from TMDb live "
        "results. Cult fantasy TV often falls below this threshold. Fix: lower "
        "the threshold for tv-domain or skip the popularity gate when title "
        "matches exactly. Remove this xfail once fixed."
    ),
    strict=False,
)
@pytest.mark.parametrize("query", LOW_POP_FANTASY_TV)
def test_search_finds_low_popularity_tv_shows(tv_engine_real, query):
    res = tv_engine_real.search(query, media_type="tv", limit=10)
    titles_lower = [item.title.lower() for item in res.results]
    assert any(query.lower() in t or t in query.lower() for t in titles_lower), (
        f"{query!r} not found; got {titles_lower}"
    )
