"""
Data-quality regression for the trilingual metadata parquets.

This test would have caught the Stage K bug before users hit it: the uk
backfill ran, the ru backfill never did, and `title_ru` ended up at 0%
non-null in production. The search engine masks across `title`,
`title_ru`, `title_uk`, so the moment any of these columns goes empty
in the deployed parquet, RU/UK queries silently miss matches.

Skips cleanly if the parquet doesn't exist (fresh checkout, CI without
data artifacts) — runs as a real assertion when artifacts are present.
"""

from __future__ import annotations

import pandas as pd
import pytest

from recommendation_system.paths import MOVIES_DIR, TV_DIR

PARQUET_PATHS = {
    "movies": MOVIES_DIR / "items_metadata_final.parquet",
    "tv": TV_DIR / "items_metadata_final.parquet",
}

REQUIRED_COLUMNS = ("title", "title_ru", "title_uk", "overview", "overview_ru", "overview_uk")
TITLE_COVERAGE_FLOOR = 0.5


def _load_or_skip(domain: str) -> pd.DataFrame:
    path = PARQUET_PATHS[domain]
    if not path.exists():
        pytest.skip(f"{path} not present — data-quality check is opt-in")
    return pd.read_parquet(path)


@pytest.mark.parametrize("domain", ["movies", "tv"])
def test_localized_columns_exist(domain: str):
    """All four localized columns must exist (Stage A schema + Stage K).

    Catches the case where Stage A migration regresses or someone drops
    a column. `overview_ru` was missing before Stage K — this test pins
    it down.
    """
    df = _load_or_skip(domain)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, (
        f"{domain}: required columns missing from parquet: {missing}. "
        f"Run recsys-backfill --lang ru and --lang uk for this domain."
    )


@pytest.mark.parametrize("domain", ["movies", "tv"])
def test_title_ru_populated(domain: str):
    """title_ru must be at least 50% populated.

    The reported bug had title_ru at 0% — RU queries hit empty mask and
    `_pick_localized` fell through to title_uk, surfacing UK titles to
    Russian users. TMDB has near-complete RU coverage so 50% is a loose
    floor that still catches a fully-empty column or a partial backfill.
    """
    df = _load_or_skip(domain)
    if "title_ru" not in df.columns:
        pytest.fail(f"{domain}: title_ru column missing — see test_localized_columns_exist")
    coverage = df["title_ru"].notna().mean()
    assert coverage >= TITLE_COVERAGE_FLOOR, (
        f"{domain}: title_ru non-null = {coverage:.1%} < {TITLE_COVERAGE_FLOOR:.0%}. "
        f"Run recsys-backfill --lang ru --domain {domain}."
    )


@pytest.mark.parametrize("domain", ["movies", "tv"])
def test_title_uk_populated(domain: str):
    """title_uk must be at least 50% populated (Stage B already ran)."""
    df = _load_or_skip(domain)
    if "title_uk" not in df.columns:
        pytest.fail(f"{domain}: title_uk column missing — see test_localized_columns_exist")
    coverage = df["title_uk"].notna().mean()
    assert coverage >= TITLE_COVERAGE_FLOOR, (
        f"{domain}: title_uk non-null = {coverage:.1%} < {TITLE_COVERAGE_FLOOR:.0%}. "
        f"Run recsys-backfill --lang uk --domain {domain}."
    )


def test_fringe_title_ru_is_correct():
    """End-to-end проверка фикса коллизии: Fringe (TV, raw tmdb=1705,
    full=10001705) должен иметь title_ru='Грань', а не 'Битва за планету
    обезьян' (это title_ru для movie tmdb=1705).

    До фикса PK кэша переводов был просто INTEGER, без media_type — и
    backfill писал movie-перевод в TV-строку парка. Если этот тест падает
    с не-Грань значением — фикс коллизии кэша переводов регрессировал.
    """
    df = _load_or_skip("tv")
    fringe = df[df["tmdb_id"] == 10001705]
    if fringe.empty:
        pytest.skip("Fringe (tmdb=10001705) not present in TV parquet")
    title_ru = fringe["title_ru"].iloc[0]
    assert title_ru, f"Fringe title_ru is empty / NULL: {title_ru!r}"
    assert "Грань" in title_ru, (
        f"Fringe title_ru should contain 'Грань', got {title_ru!r} — "
        f"likely cache-collision regression"
    )
