"""
Unit tests for MovieDatasetProcessor.load_amazon_interactions().

The loader maps Amazon Reviews 2023 (Movies_and_TV) onto the project's
tmdb_id space by title, then streams the review JSONL into an
interactions DataFrame. These tests build tiny on-the-fly JSONL files and
exercise the loader in isolation — no real Amazon dump, no network.

`load_amazon_interactions` only touches `self._section` (a staticmethod
logger), so we construct the processor via __new__ to skip the heavy
__init__ (which would read config/settings.json + raw data dirs).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from recommendation_system.data.make_dataset import MovieDatasetProcessor


# --------------------------------------------------------------------------
# Helpers / fixtures
# --------------------------------------------------------------------------

def _processor() -> MovieDatasetProcessor:
    """Bare processor instance — bypasses __init__ (load_amazon_interactions
    needs no instance state beyond the static `_section` logger)."""
    return MovieDatasetProcessor.__new__(MovieDatasetProcessor)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_amazon(amazon_dir: Path, meta: list[dict], reviews: list[dict]) -> Path:
    amazon_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(amazon_dir / "meta_Movies_and_TV.jsonl", meta)
    _write_jsonl(amazon_dir / "Movies_and_TV.jsonl", reviews)
    return amazon_dir


@pytest.fixture
def full_metadata() -> pd.DataFrame:
    """Project-side catalog the loader maps Amazon titles against.

    Only `title`, `popularity`, `tmdb_id` are read by the loader.
    """
    return pd.DataFrame({
        "title": ["The Matrix", "Inception", "Breaking Bad", "Game of Thrones", "Battery"],
        "popularity": [200, 250, 300, 280, 10],
        "tmdb_id": [603, 27205, 10001396, 10001399, 999],
    })


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

class TestAmazonLoaderMapping:
    def test_load_basic(self, full_metadata, tmp_path):
        """Reviews for a mapped title produce rows with the expected schema."""
        amazon = _write_amazon(
            tmp_path,
            meta=[{"parent_asin": "B001", "title": "The Matrix"}],
            reviews=[
                {"parent_asin": "B001", "user_id": "U1", "rating": 5.0, "timestamp": 1672531200000},
                {"parent_asin": "B001", "user_id": "U2", "rating": 4.0, "timestamp": 1672617600000},
            ],
        )
        df = _processor().load_amazon_interactions(full_metadata, amazon)

        assert list(df.columns) == ["user_id", "tmdb_id", "rating", "timestamp"]
        assert len(df) == 2
        assert set(df["tmdb_id"]) == {603}

    def test_title_cleaning_brackets(self, full_metadata, tmp_path):
        """'The Matrix [Blu-ray]' is normalized to 'the matrix' before matching."""
        amazon = _write_amazon(
            tmp_path,
            meta=[{"parent_asin": "B001", "title": "The Matrix [Blu-ray]"}],
            reviews=[{"parent_asin": "B001", "user_id": "U1", "rating": 5.0, "timestamp": 0}],
        )
        df = _processor().load_amazon_interactions(full_metadata, amazon)
        assert df["tmdb_id"].tolist() == [603]

    def test_title_cleaning_tv_complete_series(self, full_metadata, tmp_path):
        """'Breaking Bad: The Complete Series' → 'breaking bad' (TV-offset tmdb)."""
        amazon = _write_amazon(
            tmp_path,
            meta=[{"parent_asin": "B010", "title": "Breaking Bad: The Complete Series"}],
            reviews=[{"parent_asin": "B010", "user_id": "U1", "rating": 5.0, "timestamp": 0}],
        )
        df = _processor().load_amazon_interactions(full_metadata, amazon)
        assert df["tmdb_id"].tolist() == [10001396]

    def test_title_cleaning_season_suffix(self, full_metadata, tmp_path):
        """'Game of Thrones Season 1' → 'game of thrones'."""
        amazon = _write_amazon(
            tmp_path,
            meta=[{"parent_asin": "B020", "title": "Game of Thrones Season 1"}],
            reviews=[{"parent_asin": "B020", "user_id": "U1", "rating": 4.0, "timestamp": 0}],
        )
        df = _processor().load_amazon_interactions(full_metadata, amazon)
        assert df["tmdb_id"].tolist() == [10001399]

    def test_unmatched_title_ignored(self, full_metadata, tmp_path):
        """A product whose cleaned title isn't in the catalog yields no rows."""
        amazon = _write_amazon(
            tmp_path,
            meta=[{"parent_asin": "B999", "title": "Some Random Direct-to-DVD Flick"}],
            reviews=[{"parent_asin": "B999", "user_id": "U1", "rating": 5.0, "timestamp": 0}],
        )
        df = _processor().load_amazon_interactions(full_metadata, amazon)
        assert df.empty

    def test_trash_keyword_filtered(self, full_metadata, tmp_path):
        """Accessory junk is dropped even when its title matches a catalog entry.

        'Battery' is a catalog title (tmdb 999) but also a trash keyword, so
        the ASIN is never mapped and its review never loaded.
        """
        amazon = _write_amazon(
            tmp_path,
            meta=[{"parent_asin": "B777", "title": "Battery"}],
            reviews=[{"parent_asin": "B777", "user_id": "U1", "rating": 5.0, "timestamp": 0}],
        )
        df = _processor().load_amazon_interactions(full_metadata, amazon)
        assert df.empty
        assert 999 not in set(df["tmdb_id"]) if not df.empty else True

    def test_duplicate_title_resolves_deterministically(self, tmp_path):
        """Duplicate catalog titles collapse to a single tmdb_id.

        NOTE: the current implementation keeps the *least* popular row
        (pandas Series.to_dict() last-wins after a descending sort), which
        contradicts the loader's "оставляя самые популярные" comment. This
        test locks the actual behaviour; flip the expected id if the dedup
        bug is fixed.
        """
        meta_df = pd.DataFrame({
            "title": ["Heat", "Heat"],
            "popularity": [500, 5],
            "tmdb_id": [111, 222],
        })
        amazon = _write_amazon(
            tmp_path,
            meta=[{"parent_asin": "B001", "title": "Heat"}],
            reviews=[{"parent_asin": "B001", "user_id": "U1", "rating": 5.0, "timestamp": 0}],
        )
        df = _processor().load_amazon_interactions(meta_df, amazon)
        assert df["tmdb_id"].tolist() == [222]


class TestAmazonLoaderValues:
    def test_rating_scale_preserved(self, full_metadata, tmp_path):
        """Ratings stay on the raw 1–5 scale (already matches MovieLens)."""
        amazon = _write_amazon(
            tmp_path,
            meta=[{"parent_asin": "B001", "title": "Inception"}],
            reviews=[
                {"parent_asin": "B001", "user_id": "U1", "rating": 1.0, "timestamp": 0},
                {"parent_asin": "B001", "user_id": "U2", "rating": 5.0, "timestamp": 0},
            ],
        )
        df = _processor().load_amazon_interactions(full_metadata, amazon)
        assert sorted(df["rating"].tolist()) == [1.0, 5.0]

    def test_timestamp_ms_to_seconds(self, full_metadata, tmp_path):
        """Unix-ms timestamps are converted to integer seconds."""
        amazon = _write_amazon(
            tmp_path,
            meta=[{"parent_asin": "B001", "title": "Inception"}],
            reviews=[{"parent_asin": "B001", "user_id": "U1", "rating": 4.0, "timestamp": 1672531200000}],
        )
        df = _processor().load_amazon_interactions(full_metadata, amazon)
        assert df["timestamp"].tolist() == [1672531200]

    def test_missing_timestamp_defaults_zero(self, full_metadata, tmp_path):
        """A review without a timestamp field defaults to 0."""
        amazon = _write_amazon(
            tmp_path,
            meta=[{"parent_asin": "B001", "title": "Inception"}],
            reviews=[{"parent_asin": "B001", "user_id": "U1", "rating": 4.0}],
        )
        df = _processor().load_amazon_interactions(full_metadata, amazon)
        assert df["timestamp"].tolist() == [0]

    def test_user_id_factorized_to_uint(self, full_metadata, tmp_path):
        """String Amazon user_ids become contiguous numeric ids."""
        amazon = _write_amazon(
            tmp_path,
            meta=[{"parent_asin": "B001", "title": "Inception"}],
            reviews=[
                {"parent_asin": "B001", "user_id": "AHJK", "rating": 5.0, "timestamp": 0},
                {"parent_asin": "B001", "user_id": "BXYZ", "rating": 4.0, "timestamp": 0},
                {"parent_asin": "B001", "user_id": "AHJK", "rating": 3.0, "timestamp": 0},
            ],
        )
        df = _processor().load_amazon_interactions(full_metadata, amazon)

        assert str(df["user_id"].dtype) == "uint32"
        # Two distinct users → ids {0, 1}; the repeated reviewer keeps one id.
        assert set(df["user_id"]) == {0, 1}
        assert df["user_id"].tolist()[0] == df["user_id"].tolist()[2]


class TestAmazonLoaderRobustness:
    def test_missing_files_returns_empty(self, full_metadata, tmp_path):
        """No Amazon files on disk → empty DataFrame, no exception."""
        df = _processor().load_amazon_interactions(full_metadata, tmp_path)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_only_one_file_present_returns_empty(self, full_metadata, tmp_path):
        """Half a dataset (meta only, no reviews) is treated as missing."""
        (tmp_path / "meta_Movies_and_TV.jsonl").write_text("{}\n", encoding="utf-8")
        df = _processor().load_amazon_interactions(full_metadata, tmp_path)
        assert df.empty

    def test_malformed_lines_skipped(self, full_metadata, tmp_path):
        """Corrupt JSON lines in either file are skipped, not fatal."""
        amazon_dir = tmp_path
        amazon_dir.mkdir(parents=True, exist_ok=True)
        with open(amazon_dir / "meta_Movies_and_TV.jsonl", "w", encoding="utf-8") as f:
            f.write("{ this is not json\n")
            f.write(json.dumps({"parent_asin": "B001", "title": "Inception"}) + "\n")
        with open(amazon_dir / "Movies_and_TV.jsonl", "w", encoding="utf-8") as f:
            f.write("garbage line >>>\n")
            f.write(json.dumps({"parent_asin": "B001", "user_id": "U1", "rating": 5.0, "timestamp": 0}) + "\n")

        df = _processor().load_amazon_interactions(full_metadata, amazon_dir)
        assert df["tmdb_id"].tolist() == [27205]

    def test_review_for_unmapped_asin_ignored(self, full_metadata, tmp_path):
        """Reviews whose ASIN never mapped to a tmdb_id are dropped."""
        amazon = _write_amazon(
            tmp_path,
            meta=[{"parent_asin": "B001", "title": "Inception"}],
            reviews=[
                {"parent_asin": "B001", "user_id": "U1", "rating": 5.0, "timestamp": 0},
                {"parent_asin": "BZZZ", "user_id": "U2", "rating": 5.0, "timestamp": 0},
            ],
        )
        df = _processor().load_amazon_interactions(full_metadata, amazon)
        assert len(df) == 1
        assert df["tmdb_id"].tolist() == [27205]
