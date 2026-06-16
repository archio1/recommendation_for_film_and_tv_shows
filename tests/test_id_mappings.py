"""
Data-validation tests for the per-domain datasets.

These tests assert structural invariants that the rest of the system
relies on: id_mapping.json must agree with the parquet, TV ids must
carry the +TV_OFFSET offset (and movie ids must NOT), required columns
must be present, and critical fields must be non-null.

If a domain's artifacts are missing, the relevant fixtures skip the
test cleanly — no failure on a clean checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from recommendation_system.models.gnn.faiss_bridge import TV_OFFSET
from recommendation_system.paths import MOVIES_DIR, TV_DIR

REQUIRED_COLUMNS = {
    "tmdb_id", "item_id", "title", "type", "genres",
    "year", "popularity", "has_embeddings",
}

CRITICAL_NON_NULL = ["tmdb_id", "item_id", "title", "type"]


def _load_domain(domain_dir: Path) -> tuple[pd.DataFrame, dict]:
    parquet = domain_dir / "items_metadata_final.parquet"
    mapping = domain_dir / "id_mapping.json"
    if not parquet.exists() or not mapping.exists():
        pytest.skip(f"missing artifact in {domain_dir}")
    df = pd.read_parquet(parquet)
    with open(mapping, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return df, meta


@pytest.fixture(scope="module")
def movies_data():
    return _load_domain(MOVIES_DIR)


@pytest.fixture(scope="module")
def tv_data():
    return _load_domain(TV_DIR)


# --- per-domain consistency -------------------------------------------------

def test_movies_id_mapping_consistent_with_parquet(movies_data):
    df, meta = movies_data
    assert meta["num_items"] == len(df), (
        f"id_mapping.num_items={meta['num_items']} but parquet has {len(df)} rows"
    )


def test_tv_id_mapping_consistent_with_parquet(tv_data):
    df, meta = tv_data
    assert meta["num_items"] == len(df)


def test_movies_tmdb_to_item_id_unique(movies_data):
    _, meta = movies_data
    item_ids = list(meta["tmdb_to_item"].values())
    assert len(item_ids) == len(set(item_ids))
    assert min(item_ids) >= 0
    assert max(item_ids) < meta["num_items"]


def test_tv_tmdb_to_item_id_unique(tv_data):
    _, meta = tv_data
    item_ids = list(meta["tmdb_to_item"].values())
    assert len(item_ids) == len(set(item_ids))
    assert min(item_ids) >= 0
    assert max(item_ids) < meta["num_items"]


# --- TV_OFFSET convention ---------------------------------------------------

def test_tv_parquet_tmdb_ids_have_offset(tv_data):
    df, _ = tv_data
    assert (df["tmdb_id"] >= TV_OFFSET).all(), (
        "TV parquet must store tmdb_ids with the +TV_OFFSET offset baked in"
    )


def test_movies_parquet_tmdb_ids_no_offset(movies_data):
    df, _ = movies_data
    assert (df["tmdb_id"] < TV_OFFSET).all(), (
        "Movie parquet tmdb_ids must NOT carry the TV_OFFSET"
    )


def test_tv_id_mapping_keys_have_offset(tv_data):
    _, meta = tv_data
    bad = [k for k in meta["tmdb_to_item"].keys() if int(k) < TV_OFFSET]
    assert not bad, f"TV id_mapping has {len(bad)} keys without TV_OFFSET, e.g. {bad[:3]}"


def test_movies_id_mapping_keys_no_offset(movies_data):
    _, meta = movies_data
    bad = [k for k in meta["tmdb_to_item"].keys() if int(k) >= TV_OFFSET]
    assert not bad, f"Movie id_mapping has {len(bad)} keys with TV_OFFSET, e.g. {bad[:3]}"


# --- trained-vs-catalog accounting -----------------------------------------

@pytest.mark.parametrize("fixture_name", ["movies_data", "tv_data"])
def test_num_trained_items_le_num_items(fixture_name, request):
    _, meta = request.getfixturevalue(fixture_name)
    assert meta["num_trained_items"] <= meta["num_items"]


@pytest.mark.parametrize("fixture_name", ["movies_data", "tv_data"])
def test_has_embeddings_count_matches_num_trained(fixture_name, request):
    df, meta = request.getfixturevalue(fixture_name)
    if "has_embeddings" not in df.columns:
        pytest.skip("parquet has no has_embeddings column")
    trained_in_parquet = int(df["has_embeddings"].sum())
    assert trained_in_parquet == meta["num_trained_items"], (
        f"parquet has_embeddings=True count ({trained_in_parquet}) "
        f"!= id_mapping.num_trained_items ({meta['num_trained_items']})"
    )


# --- schema -----------------------------------------------------------------

@pytest.mark.parametrize("fixture_name", ["movies_data", "tv_data"])
def test_required_columns_present(fixture_name, request):
    df, _ = request.getfixturevalue(fixture_name)
    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"missing required columns: {missing}"


@pytest.mark.parametrize("fixture_name", ["movies_data", "tv_data"])
def test_critical_fields_no_nulls(fixture_name, request):
    df, _ = request.getfixturevalue(fixture_name)
    for col in CRITICAL_NON_NULL:
        if col in df.columns:
            assert df[col].notna().all(), f"column '{col}' has NULLs"


@pytest.mark.parametrize("fixture_name", ["movies_data", "tv_data"])
def test_item_id_is_dense_range(fixture_name, request):
    """item_id must cover [0, num_items) without gaps."""
    df, meta = request.getfixturevalue(fixture_name)
    expected = set(range(meta["num_items"]))
    actual = set(df["item_id"].astype(int).tolist())
    assert actual == expected, (
        f"item_id is not a dense range; "
        f"missing={list(expected - actual)[:5]}, extra={list(actual - expected)[:5]}"
    )


@pytest.mark.parametrize("fixture_name", ["movies_data", "tv_data"])
def test_type_field_matches_domain(fixture_name, request):
    df, _ = request.getfixturevalue(fixture_name)
    expected_type = "tv" if fixture_name == "tv_data" else "movie"
    bad = df[df["type"] != expected_type]
    assert bad.empty, (
        f"{fixture_name}: {len(bad)} rows have type != {expected_type!r}, "
        f"first: {bad['title'].head(3).tolist()}"
    )
