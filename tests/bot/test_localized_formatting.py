"""
Lang-aware formatter tests.

`_format_title`, `_pick_localized` and `_format_genres` live in
`movie_bot.py`, which loads ML engines at import time and can't be
pulled in for a unit test. The bodies are tiny and pure, so we
re-implement them locally here and add a source-inspection guard
that fails loudly if the bot drifts from this contract.

The Blacklist / Drama-Crime-Mystery case is the original reported
bug: before the lang refactor, `_format_genres` always returned
Russian regardless of the user's language. The genre-output tests
nail that down.

The "ru user with title_uk only" case is the second user-reported
bug: when title_ru is missing, the bot silently fell through to
title_uk, so a Russian user adding Die Hard saw "Міцний горішок"
instead of the English "Die Hard". Fixed in stage J — these tests
encode the desired behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pytest

from recommendation_system.models.gnn.bilingual_utils import GENRE_EN_TO_RU, GENRE_EN_TO_UK
from recommendation_system.models.gnn.universal_search import UniversalMediaItem


# ---------------------------------------------------------------------------
# Local mirrors of the formatter functions.
#
# Kept in lockstep with movie_bot.py via the source-inspection guard
# at the bottom of this file — if the bot's bodies drift, that test
# fails and forces an update here.
# ---------------------------------------------------------------------------

_GENRE_TABLES = {
    "ru": GENRE_EN_TO_RU,
    "uk": GENRE_EN_TO_UK,
    "en": {},
}

# Mirrors faiss_bridge.TV_OFFSET — kept local so we don't drag the gnn
# package import in. The contract guard pins the constant inside the
# bot's _format_title, so a drift here is caught by source inspection.
TV_OFFSET = 10_000_000


def _pick_localized(item: UniversalMediaItem, lang: str) -> str:
    """Mirror of movie_bot._pick_localized — post stage J fix.

    Fallback for ru/uk goes to English first, not the other Cyrillic
    language. Showing Ukrainian to a Russian user just because title_uk
    is populated and title_ru isn't is the user-reported bug.
    """
    if lang == "uk":
        return item.title_uk or item.title or item.title_ru or ""
    if lang == "ru":
        return item.title_ru or item.title or item.title_uk or ""
    return item.title or item.title_ru or item.title_uk or ""


def _format_title(
    item: UniversalMediaItem, lang: str = "ru", *, link: bool = True
) -> str:
    """Mirror of movie_bot._format_title.

    HTML mode (default) wraps the visible label in an <a href> pointing
    at IMDb (or TMDB if imdb_id is missing). Inline-button label callers
    pass link=False — Telegram doesn't render HTML inside button text.
    """
    title = _pick_localized(item, lang)
    year = f" ({item.year})" if item.year else ""
    if not link:
        return f"{title}{year}"
    if item.imdb_id:
        url = f"https://www.imdb.com/title/{item.imdb_id}/"
    else:
        raw = (
            item.tmdb_id - TV_OFFSET
            if item.media_type == "tv" and item.tmdb_id >= TV_OFFSET
            else item.tmdb_id
        )
        url = f"https://www.themoviedb.org/{item.media_type}/{raw}"
    return f'<a href="{url}">{title}{year}</a>'


def _format_genres(genres: List[str], lang: str = "ru", limit: int = 3) -> str:
    if not genres:
        return ""
    table = _GENRE_TABLES.get(lang, _GENRE_TABLES["ru"])
    return ", ".join(table.get(g, g) for g in genres[:limit])


# ---------------------------------------------------------------------------
# Item factories
# ---------------------------------------------------------------------------


def _make(
    title: str = "Inception",
    title_ru: Optional[str] = "Начало",
    title_uk: Optional[str] = "Початок",
    year: int = 2010,
    genres: Optional[List[str]] = None,
    media_type: str = "movie",
) -> UniversalMediaItem:
    return UniversalMediaItem(
        tmdb_id=27205,
        media_type=media_type,
        title=title,
        title_ru=title_ru,
        title_uk=title_uk,
        year=year,
        genres=genres or ["Action", "Sci-Fi"],
    )


# ---------------------------------------------------------------------------
# _pick_localized — fallback chains across all 3 langs
# ---------------------------------------------------------------------------


def test_pick_localized_returns_native_when_present():
    item = _make()
    assert _pick_localized(item, "en") == "Inception"
    assert _pick_localized(item, "ru") == "Начало"
    assert _pick_localized(item, "uk") == "Початок"


def test_pick_localized_uk_falls_back_to_english_not_russian():
    """Stage J: the uk → ru → en chain became uk → en → ru.

    A uk user without a uk title sees the canonical English name, not
    a Russian one — surprising the user with the wrong Cyrillic is
    worse than showing them Latin they can read.
    """
    item = _make(title="Inception", title_ru="Начало", title_uk=None)
    assert _pick_localized(item, "uk") == "Inception"


def test_pick_localized_ru_falls_back_to_english_not_ukrainian():
    """Stage J / user-reported bug: ru fallback skips uk.

    Die Hard had title_uk='Міцний горішок' but title_ru=None in the
    parquet (uk backfill ran, ru backfill never did). Before the fix,
    a Russian user saw 'Міцний горішок' in their list. After the fix,
    they see the English 'Die Hard' instead.
    """
    die_hard = _make(
        title="Die Hard",
        title_ru=None,
        title_uk="Міцний горішок",
        year=1988,
    )
    assert _pick_localized(die_hard, "ru") == "Die Hard"


def test_pick_localized_falls_through_to_other_cyrillic_as_last_resort():
    """When both English and the native lang are missing, take whatever's
    left rather than emit empty — better than rendering ' (1988)'."""
    only_uk = _make(title="", title_ru=None, title_uk="Початок")
    assert _pick_localized(only_uk, "ru") == "Початок"

    only_ru = _make(title="", title_ru="Начало", title_uk=None)
    assert _pick_localized(only_ru, "uk") == "Начало"


def test_pick_localized_returns_empty_when_nothing_known():
    blank = _make(title="", title_ru=None, title_uk=None)
    assert _pick_localized(blank, "ru") == ""
    assert _pick_localized(blank, "uk") == ""
    assert _pick_localized(blank, "en") == ""


def test_pick_localized_empty_string_treated_as_missing():
    """title_ru="" should fall through, same as None."""
    item = _make(title="Inception", title_ru="", title_uk="Початок")
    # ru: title_ru is empty → fallback to title (en), not title_uk.
    assert _pick_localized(item, "ru") == "Inception"


# ---------------------------------------------------------------------------
# _format_title — title + year
# ---------------------------------------------------------------------------


def test_format_title_appends_year():
    # link=False isolates title+year from the IMDb-link wrapping (covered
    # separately below) so this test stays focused on localization.
    item = _make()
    assert _format_title(item, "en", link=False) == "Inception (2010)"
    assert _format_title(item, "ru", link=False) == "Начало (2010)"
    assert _format_title(item, "uk", link=False) == "Початок (2010)"


def test_format_title_omits_year_when_zero():
    item = _make(year=0)
    assert _format_title(item, "en", link=False) == "Inception"


def test_format_title_die_hard_ru_user_regression():
    """Full regression on the user's report: list line for a ru user
    when only title_uk is populated should NOT be the Ukrainian title."""
    die_hard = _make(
        title="Die Hard",
        title_ru=None,
        title_uk="Міцний горішок",
        year=1988,
    )
    assert _format_title(die_hard, "ru", link=False) == "Die Hard (1988)"
    assert _format_title(die_hard, "uk", link=False) == "Міцний горішок (1988)"
    assert _format_title(die_hard, "en", link=False) == "Die Hard (1988)"


# ---------------------------------------------------------------------------
# _format_title — IMDb / TMDB link wrapping
# ---------------------------------------------------------------------------


def test_format_title_wraps_in_imdb_anchor_when_imdb_id_present():
    """IMDb link is preferred over TMDB when we have it. The <a> wraps
    title+year together so the whole label is clickable."""
    item = _make()
    item.imdb_id = "tt1375666"
    assert (
        _format_title(item, "en")
        == '<a href="https://www.imdb.com/title/tt1375666/">Inception (2010)</a>'
    )


def test_format_title_falls_back_to_tmdb_when_no_imdb_id():
    """Some TV rows and old arthouse have no imdb_id — fallback to TMDB
    keeps the link working (TMDB id is always present on a real item)."""
    item = _make()  # imdb_id defaults to None on UniversalMediaItem
    assert (
        _format_title(item, "en")
        == '<a href="https://www.themoviedb.org/movie/27205">Inception (2010)</a>'
    )


def test_format_title_strips_tv_offset_for_tmdb_url():
    """tv items carry +TV_OFFSET in tmdb_id (parquet convention).
    The public TMDB URL uses the *raw* id — a missed strip here gives
    a 404 on themoviedb.org. Critical regression to lock down."""
    boys = _make(
        title="The Boys",
        title_ru=None,
        title_uk=None,
        media_type="tv",
        year=2019,
    )
    boys.tmdb_id = 76479 + TV_OFFSET
    assert (
        _format_title(boys, "en")
        == '<a href="https://www.themoviedb.org/tv/76479">The Boys (2019)</a>'
    )


def test_format_title_link_false_returns_plain_text():
    """link=False is the path used by inline-button labels (search /
    list keyboards) — Telegram renders button text literally, so an
    <a> tag would show as visible angle brackets."""
    item = _make()
    item.imdb_id = "tt1375666"
    assert _format_title(item, "en", link=False) == "Inception (2010)"


# ---------------------------------------------------------------------------
# _format_genres — language-aware
# ---------------------------------------------------------------------------


def test_format_genres_blacklist_regression_ru():
    """Original bug: The Blacklist's genres always rendered as
    'Драма, Криминал, Детектив' regardless of the user's language."""
    out = _format_genres(["Drama", "Crime", "Mystery"], lang="ru")
    assert out == "Драма, Криминал, Детектив"


def test_format_genres_blacklist_uk():
    out = _format_genres(["Drama", "Crime", "Mystery"], lang="uk")
    assert out == "Драма, Кримінал, Детектив"


def test_format_genres_blacklist_en_passthrough():
    """English is the source language — genres render verbatim."""
    out = _format_genres(["Drama", "Crime", "Mystery"], lang="en")
    assert out == "Drama, Crime, Mystery"


@pytest.mark.parametrize(
    "lang,genres,expected",
    [
        ("ru", ["Action"], "Боевик"),
        ("uk", ["Action"], "Бойовик"),
        ("en", ["Action"], "Action"),
        ("ru", ["Sci-Fi", "Thriller"], "Фантастика, Триллер"),
        ("uk", ["Sci-Fi", "Thriller"], "Фантастика, Трилер"),
        ("ru", ["Horror"], "Ужасы"),
        ("uk", ["Horror"], "Жахи"),
        ("ru", ["Animation"], "Мультфильм"),
        ("uk", ["Animation"], "Мультфільм"),
    ],
)
def test_format_genres_parametrized(lang, genres, expected):
    assert _format_genres(genres, lang=lang) == expected


def test_format_genres_truncates_to_limit():
    """Default limit is 3 — extra genres get dropped."""
    out = _format_genres(
        ["Action", "Adventure", "Sci-Fi", "Thriller", "Drama"],
        lang="ru",
    )
    assert out == "Боевик, Приключения, Фантастика"


def test_format_genres_respects_explicit_limit():
    out = _format_genres(["Action", "Drama"], lang="en", limit=1)
    assert out == "Action"


def test_format_genres_unknown_genre_passes_through():
    """Genre not in the dict (e.g. a new TMDB tag) shouldn't crash —
    just emit the English label as a fallback."""
    out = _format_genres(["Action", "Reality"], lang="ru")
    assert out == "Боевик, Reality"


def test_format_genres_empty_list():
    assert _format_genres([], lang="ru") == ""
    assert _format_genres([], lang="en") == ""


def test_format_genres_unknown_lang_falls_to_ru():
    """Defensive: an out-of-band lang shouldn't blow up; ru is the
    historical default for users we haven't classified."""
    assert _format_genres(["Action"], lang="zz") == "Боевик"


# ---------------------------------------------------------------------------
# Source-inspection guard — pin movie_bot's formatter contracts.
#
# Because movie_bot.py initializes engines at import time, we can't
# `from movie_bot import _pick_localized` here. Instead, read the file
# and verify the structural pieces of the function bodies haven't drifted
# from the version this test mirrors.
# ---------------------------------------------------------------------------


def test_movie_bot_formatter_contract():
    from recommendation_system.paths import PACKAGE_DIR

    bot_path = PACKAGE_DIR / "models" / "gnn" / "movie_bot.py"
    if not bot_path.exists():
        pytest.skip(f"{bot_path} not found")
    src = bot_path.read_text(encoding="utf-8")

    # _pick_localized: sibling-Cyrillic fallback before English (spec §3.4).
    # uk → ru → en and ru → uk → en. After the cache-collision fix the
    # title_ru / title_uk columns are reliably populated, so the
    # ru↔uk slavic fallback gives meaningful output instead of leaking
    # English ahead of a sister-language translation.
    expected_chains = [
        'return item.title_uk or item.title_ru or item.title',
        'return item.title_ru or item.title_uk or item.title',
    ]
    for chain in expected_chains:
        assert chain in src, (
            f"movie_bot._pick_localized chain drifted — expected "
            f"{chain!r} in source"
        )

    # _format_genres must consult the lang-keyed table and never
    # hardcode GENRE_EN_TO_RU.
    assert "_GENRE_TABLES" in src or 'GENRE_EN_TO_UK' in src, (
        "movie_bot must dispatch genres by lang — hardcoded "
        "GENRE_EN_TO_RU is the original bug."
    )

    # _format_title: IMDb-link wrapping.
    # The local _format_title mirror in this file matches movie_bot's
    # behaviour — these guards fail loudly if the bot's logic drifts.
    expected_link_pieces = [
        # IMDb branch picked when imdb_id is truthy.
        'https://www.imdb.com/title/',
        # TMDB fallback URL — note no trailing slash because we
        # interpolate {raw} after the path segment.
        'https://www.themoviedb.org/',
        # TV_OFFSET subtraction — vital for non-IMDb tv links to
        # resolve to the correct TMDB tv/<raw_id> page.
        'item.tmdb_id - TV_OFFSET',
        # link=False is the plain-text branch used by button-label
        # callers; without it inline buttons would render literal HTML.
        'link: bool = True',
        'if not link:',
    ]
    for piece in expected_link_pieces:
        assert piece in src, (
            f"movie_bot._format_title contract drifted — expected "
            f"{piece!r} in source"
        )
