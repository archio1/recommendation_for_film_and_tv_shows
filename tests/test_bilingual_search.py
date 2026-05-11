"""
Trilingual title search regressions.

Covers `UniversalSearchEngine.search` (the pandas LIKE-mask path) and
`HotCache.search` (the SQLite path) — both should hit a tmdb_id when the
query matches *any* of the title/title_ru/title_uk columns. Each engine
is constructed with a tiny in-memory DataFrame so the tests don't need
real model artifacts.

The "Крепкий орешек / Міцний горішок" case is the bug the user hit: with
a populated title_ru column the Russian query has to find Die Hard
without falling through to the Ukrainian title.
"""

from __future__ import annotations

import pandas as pd
import pytest

from universal_search import HotCache, UniversalMediaItem, UniversalSearchEngine


def _trilingual_metadata() -> pd.DataFrame:
    """Five rows with varying localization + numeric-style titles.

    - Inception: all three titles populated → 1:1:1 match check.
    - Die Hard: title_ru and title_uk populated → regression on the bug.
    - Joker: only English title → uk/ru queries shouldn't surface it.
    - Se7en: digit-as-letter literal (tmdb=807) — exercises the numeric
      normalizer path on the LIKE mask.
    - 7 Samurai: phrase with leading digit + alphabetic context — query
      'seven samurai' should hit it via the word→digit variant.
    """
    return pd.DataFrame(
        {
            "item_id": [0, 1, 2, 3, 4],
            "tmdb_id": [27205, 562, 475557, 807, 346],
            "type": ["movie", "movie", "movie", "movie", "movie"],
            "title": ["Inception", "Die Hard", "Joker", "Se7en", "7 Samurai"],
            "title_ru": ["Начало", "Крепкий орешек", None, "Семь", "Семь самураев"],
            "title_uk": ["Початок", "Міцний горішок", None, "Сім", "Сім самураїв"],
            "year": [2010, 1988, 2019, 1995, 1954],
            "genres": [
                ["Action", "Sci-Fi"],
                ["Action", "Thriller"],
                ["Crime", "Drama"],
                ["Crime", "Mystery"],
                ["Action", "Drama"],
            ],
            "popularity": [80.0, 50.0, 60.0, 40.0, 30.0],
            "vote_average": [8.4, 7.8, 8.2, 8.3, 8.6],
            "vote_count": [30000, 8000, 20000, 6000, 3000],
            "keywords": [[], [], [], [], []],
            "overview": ["", "", "", "", ""],
            "overview_ru": [None, None, None, None, None],
            "overview_uk": [None, None, None, None, None],
        }
    )


@pytest.fixture
def trilingual_engine(tmp_path) -> UniversalSearchEngine:
    return UniversalSearchEngine(
        metadata=_trilingual_metadata(),
        cache_dir=tmp_path,
        tmdb_api_key=None,
        inference_engine=None,
        embeddings_path=None,
        model_num_items=0,
    )


# ---------------------------------------------------------------------------
# UniversalSearchEngine.search — pandas LIKE-mask path
# ---------------------------------------------------------------------------


def test_inception_found_by_english(trilingual_engine):
    result = trilingual_engine.search("Inception", limit=5)
    assert any(r.tmdb_id == 27205 for r in result.results)


def test_inception_found_by_russian(trilingual_engine):
    result = trilingual_engine.search("Начало", limit=5)
    assert any(r.tmdb_id == 27205 for r in result.results)


def test_inception_found_by_ukrainian(trilingual_engine):
    result = trilingual_engine.search("Початок", limit=5)
    assert any(r.tmdb_id == 27205 for r in result.results)


def test_three_languages_return_same_tmdb_id(trilingual_engine):
    """Inception/Начало/Початок must all resolve to one tmdb_id (27205)."""
    en = trilingual_engine.search("Inception", limit=5).results
    ru = trilingual_engine.search("Начало", limit=5).results
    uk = trilingual_engine.search("Початок", limit=5).results

    assert en and ru and uk, "all three queries should return at least one hit"
    assert en[0].tmdb_id == ru[0].tmdb_id == uk[0].tmdb_id == 27205


def test_die_hard_found_by_russian_when_title_ru_populated(trilingual_engine):
    """Regression: 'крепкий орешек' must hit Die Hard when title_ru exists.

    The user-reported bug: this query returned nothing because title_ru
    was missing from parquet, so the search fell back to title_uk and
    only Ukrainian queries ('міцний горішок') worked. With the column
    populated, the Russian query should find tmdb_id=562.
    """
    result = trilingual_engine.search("крепкий орешек", limit=5)
    tmdb_ids = [r.tmdb_id for r in result.results]
    assert 562 in tmdb_ids, (
        f"expected Die Hard (562) in results for 'крепкий орешек', "
        f"got {tmdb_ids}"
    )


def test_die_hard_found_by_ukrainian(trilingual_engine):
    result = trilingual_engine.search("міцний горішок", limit=5)
    assert any(r.tmdb_id == 562 for r in result.results)


def test_search_is_case_insensitive(trilingual_engine):
    """LIKE matching lowercases both the query and the columns."""
    upper = trilingual_engine.search("INCEPTION", limit=5).results
    lower = trilingual_engine.search("inception", limit=5).results
    mixed = trilingual_engine.search("InCePtIoN", limit=5).results
    assert upper and lower and mixed
    assert upper[0].tmdb_id == lower[0].tmdb_id == mixed[0].tmdb_id == 27205


def test_movie_with_only_english_title_unfindable_in_cyrillic(trilingual_engine):
    """Joker has no title_ru/title_uk — Cyrillic queries shouldn't match it.

    Guards against the mask accidentally matching null/empty columns:
    `fillna('').str.lower().str.contains(q, na=False)` should never
    surface Joker for a Russian or Ukrainian search.
    """
    ru = trilingual_engine.search("джокер", limit=5).results
    uk = trilingual_engine.search("джокер", limit=5).results
    assert all(r.tmdb_id != 475557 for r in ru)
    assert all(r.tmdb_id != 475557 for r in uk)
    # English still works.
    en = trilingual_engine.search("Joker", limit=5).results
    assert any(r.tmdb_id == 475557 for r in en)


def test_localized_titles_propagated_to_result(trilingual_engine):
    """Result items must carry title_ru/title_uk through to the formatter."""
    result = trilingual_engine.search("Inception", limit=5)
    inception = next(r for r in result.results if r.tmdb_id == 27205)
    assert inception.title == "Inception"
    assert inception.title_ru == "Начало"
    assert inception.title_uk == "Початок"


# ---------------------------------------------------------------------------
# Stage L — numeric normalization (Se7en ↔ seven, 7 ↔ семь / сім)
# ---------------------------------------------------------------------------


def test_se7en_found_by_literal_query(trilingual_engine):
    """Original query with the digit-as-letter must still hit. The
    normalizer always preserves the lowered original among its variants."""
    result = trilingual_engine.search("Se7en", limit=5).results
    assert any(r.tmdb_id == 807 for r in result), [r.tmdb_id for r in result]


def test_se7en_found_by_russian_word(trilingual_engine):
    """Catalog has title_ru='Семь' — query 'семь' alone should hit it
    via the title_ru column (no numeric normalization needed for this
    direction; the column-OR mask covers it)."""
    result = trilingual_engine.search("Семь", limit=5).results
    assert any(r.tmdb_id == 807 for r in result), [r.tmdb_id for r in result]


def test_se7en_found_by_ukrainian_word(trilingual_engine):
    result = trilingual_engine.search("Сім", limit=5).results
    assert any(r.tmdb_id == 807 for r in result), [r.tmdb_id for r in result]


def test_seven_samurai_found_by_alphabetic_phrase(trilingual_engine):
    """Catalog title is '7 Samurai' (digit). Query 'seven samurai' must
    hit via the word→digit variant ('7 samurai'). This is the canonical
    Stage L payoff in the alphabetic→digit direction."""
    result = trilingual_engine.search("seven samurai", limit=5).results
    assert any(r.tmdb_id == 346 for r in result), [r.tmdb_id for r in result]


def test_seven_samurai_found_by_russian_phrase(trilingual_engine):
    """title_ru='Семь самураев' — query 'семь самураев' hits via the
    title_ru column directly. This is the easy column-OR path; the
    Stage L variant ('7 самураев') is bonus coverage."""
    result = trilingual_engine.search("семь самураев", limit=5).results
    assert any(r.tmdb_id == 346 for r in result), [r.tmdb_id for r in result]


def test_seven_samurai_found_by_ukrainian_word_to_digit(trilingual_engine):
    """Even if title_uk had only the digit form, 'сім самураїв' must
    still hit via the word→digit variant. With the current fixture
    title_uk='Сім самураїв' covers it directly too."""
    result = trilingual_engine.search("сім самураїв", limit=5).results
    assert any(r.tmdb_id == 346 for r in result), [r.tmdb_id for r in result]


# ---------------------------------------------------------------------------
# Stage L — same checks on the SQLite (HotCache) path
# ---------------------------------------------------------------------------


def test_hotcache_se7en_literal(tmp_path):
    cache = HotCache(tmp_path / "hot.db")
    cache.upsert_batch(
        [
            UniversalMediaItem(
                tmdb_id=807,
                media_type="movie",
                title="Se7en",
                title_ru="Семь",
                title_uk="Сім",
                year=1995,
                genres=["Crime", "Mystery"],
                popularity=40.0,
            ),
        ]
    )
    hits = cache.search("Se7en", media_type=None, limit=5)
    assert any(i.tmdb_id == 807 for i in hits)


def test_hotcache_seven_samurai_word_to_digit(tmp_path):
    """SQLite path mirrors the pandas behaviour — query 'seven samurai'
    must produce '7 samurai' as one of the OR-clauses and hit the row
    that only has the digit form in title."""
    cache = HotCache(tmp_path / "hot.db")
    cache.upsert_batch(
        [
            UniversalMediaItem(
                tmdb_id=346,
                media_type="movie",
                title="7 Samurai",
                title_ru=None,
                title_uk=None,
                year=1954,
                genres=["Action", "Drama"],
                popularity=30.0,
            ),
        ]
    )
    hits = cache.search("seven samurai", media_type=None, limit=5)
    assert any(i.tmdb_id == 346 for i in hits), [i.tmdb_id for i in hits]


def test_media_type_filter_applies(tmp_path):
    """media_type='movie' must exclude tv rows even if titles match."""
    metadata = pd.DataFrame(
        {
            "item_id": [0, 1],
            "tmdb_id": [100, 200],
            "type": ["movie", "tv"],
            "title": ["Same Name", "Same Name"],
            "title_ru": [None, None],
            "title_uk": [None, None],
            "year": [2010, 2015],
            "genres": [["Drama"], ["Drama"]],
            "popularity": [10.0, 10.0],
            "vote_average": [7.0, 7.0],
            "vote_count": [100, 100],
            "keywords": [[], []],
            "overview": ["", ""],
            "overview_ru": [None, None],
            "overview_uk": [None, None],
        }
    )
    engine = UniversalSearchEngine(
        metadata=metadata,
        cache_dir=tmp_path,
        tmdb_api_key=None,
        inference_engine=None,
        embeddings_path=None,
        model_num_items=0,
    )
    movies = engine.search("Same Name", media_type="movie", limit=5).results
    tv = engine.search("Same Name", media_type="tv", limit=5).results
    assert all(r.media_type == "movie" for r in movies)
    assert all(r.media_type == "tv" for r in tv)


# ---------------------------------------------------------------------------
# HotCache.search — SQLite path (mirrors the pandas LIKE behaviour)
# ---------------------------------------------------------------------------


@pytest.fixture
def populated_hot_cache(tmp_path) -> HotCache:
    cache = HotCache(tmp_path / "hot.db")
    cache.upsert_batch(
        [
            UniversalMediaItem(
                tmdb_id=27205,
                media_type="movie",
                title="Inception",
                title_ru="Начало",
                title_uk="Початок",
                year=2010,
                genres=["Action", "Sci-Fi"],
                popularity=80.0,
            ),
            UniversalMediaItem(
                tmdb_id=562,
                media_type="movie",
                title="Die Hard",
                title_ru="Крепкий орешек",
                title_uk="Міцний горішок",
                year=1988,
                genres=["Action", "Thriller"],
                popularity=50.0,
            ),
        ]
    )
    return cache


def test_hotcache_finds_by_each_language(populated_hot_cache):
    en_hits = populated_hot_cache.search("Inception", media_type=None, limit=5)
    ru_hits = populated_hot_cache.search("Начало", media_type=None, limit=5)
    uk_hits = populated_hot_cache.search("Початок", media_type=None, limit=5)

    assert any(i.tmdb_id == 27205 for i in en_hits)
    assert any(i.tmdb_id == 27205 for i in ru_hits)
    assert any(i.tmdb_id == 27205 for i in uk_hits)


def test_hotcache_die_hard_russian_regression(populated_hot_cache):
    """Same regression check on the SQLite path."""
    hits = populated_hot_cache.search("крепкий орешек", media_type=None, limit=5)
    assert any(i.tmdb_id == 562 for i in hits)


def test_hotcache_round_trips_localized_titles(populated_hot_cache):
    hits = populated_hot_cache.search("Inception", media_type=None, limit=5)
    inception = next(i for i in hits if i.tmdb_id == 27205)
    assert inception.title_ru == "Начало"
    assert inception.title_uk == "Початок"
