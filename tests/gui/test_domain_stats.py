"""
Тесты сбора статистики датасетов/моделей — `gui/domain_stats.py`.

Это фундамент, на котором стоят вкладки «Данные» и «Тестирование» (резолв того,
какой датасет/чекпоинт показать и загрузить). Без сети и реальных моделей —
только крошечные tmp-артефакты.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from recommendation_system.models.gnn.gui import domain_stats as ds


# ======================================================================
# _collect_dataset_stats
# ======================================================================

class TestCollectDatasetStats:
    def test_happy_path(self, tmp_path, make_dataset_dir):
        folder = make_dataset_dir(
            tmp_path / "movies", n_interactions=200, num_users=100, num_items=10
        )
        result = ds._collect_dataset_stats(folder)

        assert result is not None
        assert result["num_users"] == 100
        assert result["num_items"] == 10
        assert result["num_interactions"] == 200
        assert isinstance(result["interactions_mtime"], datetime)
        assert isinstance(result["items_mtime"], datetime)

    def test_num_items_falls_back_to_num_trained_items(self, tmp_path, make_dataset_dir):
        folder = make_dataset_dir(
            tmp_path / "tv", num_items=None, num_trained_items=42
        )
        result = ds._collect_dataset_stats(folder)

        assert result is not None
        assert result["num_items"] == 42

    @pytest.mark.parametrize(
        "missing_kwarg",
        [
            {"write_interactions": False},
            {"write_items": False},
            {"write_mapping": False},
        ],
    )
    def test_returns_none_if_any_required_file_missing(
        self, tmp_path, make_dataset_dir, missing_kwarg
    ):
        folder = make_dataset_dir(tmp_path / "d", **missing_kwarg)
        assert ds._collect_dataset_stats(folder) is None

    def test_nonexistent_dir_returns_none(self, tmp_path):
        assert ds._collect_dataset_stats(tmp_path / "nope") is None

    def test_broken_mapping_json_keeps_interactions_but_nulls_counts(
        self, tmp_path, make_dataset_dir
    ):
        folder = make_dataset_dir(tmp_path / "movies", mapping_text="{ broken json")
        result = ds._collect_dataset_stats(folder)

        # Файл есть → не None; json битый → счётчики None, но num_interactions из parquet.
        assert result is not None
        assert result["num_users"] is None
        assert result["num_items"] is None
        assert result["num_interactions"] == 200


# ======================================================================
# _collect_model_stats
# ======================================================================

class TestCollectModelStats:
    def test_with_full_sidecar(self, tmp_path, make_model_dir):
        pt = make_model_dir(
            tmp_path / "models" / "movies",
            recall=0.2058, sanity_passed=True,
            trained_at="2026-05-29T13:59:00Z",
        )
        result = ds._collect_model_stats(pt)

        assert result["last_train_checkpoint"] == pt.name
        assert result["last_train_recall"] == pytest.approx(0.2058)
        assert result["last_train_sanity_ok"] is True
        assert result["last_train_at"] == "2026-05-29T13:59:00Z"
        assert isinstance(result["model_mtime"], datetime)
        assert result["model_size_mb"] >= 0

    def test_sidecar_checkpoint_name_wins(self, tmp_path, make_model_dir):
        pt = make_model_dir(
            tmp_path / "m", name="renamed.pt",
            checkpoint_in_sidecar="canonical_v9.pt",
        )
        result = ds._collect_model_stats(pt)
        assert result["last_train_checkpoint"] == "canonical_v9.pt"

    def test_without_sidecar_only_file_info(self, tmp_path, make_model_dir):
        pt = make_model_dir(tmp_path / "m", name="old_v4.pt", sidecar=False)
        result = ds._collect_model_stats(pt)

        assert result["last_train_checkpoint"] == "old_v4.pt"
        assert "model_mtime" in result and "model_size_mb" in result
        assert "last_train_at" not in result
        assert "last_train_recall" not in result
        assert "last_train_sanity_ok" not in result

    def test_broken_sidecar_falls_back_to_file_info(self, tmp_path, make_model_dir):
        pt = make_model_dir(tmp_path / "m", sidecar_text="{ not json")
        result = ds._collect_model_stats(pt)

        assert result["last_train_checkpoint"] == pt.name
        assert "last_train_at" not in result

    def test_sanity_failed(self, tmp_path, make_model_dir):
        pt = make_model_dir(tmp_path / "m", sanity_passed=False)
        result = ds._collect_model_stats(pt)
        assert result["last_train_sanity_ok"] is False

    def test_non_numeric_recall_is_ignored(self, tmp_path, make_model_dir):
        pt = make_model_dir(
            tmp_path / "m",
            sidecar_text='{"metrics": {"recall@10": "n/a"}, "checkpoint": "x.pt"}',
        )
        result = ds._collect_model_stats(pt)
        assert "last_train_recall" not in result


# ======================================================================
# _find_latest_model
# ======================================================================

class TestFindLatestModel:
    def test_picks_most_recent_by_mtime(self, tmp_path, make_model_dir):
        d = tmp_path / "models" / "movies"
        make_model_dir(d, name="v1.pt", domain="movies", mtime=1_000_000)
        make_model_dir(d, name="v2.pt", domain="movies", mtime=2_000_000)

        latest = ds._find_latest_model(d, "movies")
        assert latest is not None and latest.name == "v2.pt"

    def test_skips_sidecar_with_mismatched_domain(self, tmp_path, make_model_dir):
        d = tmp_path / "models" / "mixed"
        make_model_dir(d, name="movies_v5.pt", domain="movies", mtime=1_000_000)
        # tv-чекпоинт новее, но при запросе movies должен быть пропущен.
        make_model_dir(d, name="tv_v5.pt", domain="tv", mtime=2_000_000)

        latest = ds._find_latest_model(d, "movies")
        assert latest is not None and latest.name == "movies_v5.pt"

    def test_includes_pt_without_sidecar(self, tmp_path, make_model_dir):
        d = tmp_path / "models" / "movies"
        make_model_dir(d, name="legacy_v4.pt", sidecar=False)
        latest = ds._find_latest_model(d, "movies")
        assert latest is not None and latest.name == "legacy_v4.pt"

    def test_missing_dir_returns_none(self, tmp_path):
        assert ds._find_latest_model(tmp_path / "nope", "movies") is None

    def test_empty_dir_returns_none(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        assert ds._find_latest_model(d, "movies") is None


# ======================================================================
# _collect_domain_stats (интеграция)
# ======================================================================

class TestCollectDomainStats:
    def test_with_overrides(self, tmp_path, make_dataset_dir, make_model_dir):
        dataset = make_dataset_dir(tmp_path / "ds", num_users=7, num_items=3)
        model = make_model_dir(tmp_path / "md", name="ovr.pt", recall=0.5)

        stats = ds._collect_domain_stats(
            "movies", dataset_override=dataset, model_override=model
        )

        assert stats.dataset_exists is True
        assert stats.num_users == 7
        assert stats.num_items == 3
        assert stats.last_train_checkpoint == "ovr.pt"
        assert stats.last_train_recall == pytest.approx(0.5)
        assert stats.dataset_dir == dataset

    def test_missing_dataset_sets_exists_false(self, tmp_path, make_model_dir):
        empty = tmp_path / "empty_ds"
        empty.mkdir()
        model = make_model_dir(tmp_path / "md", name="m.pt")

        stats = ds._collect_domain_stats(
            "movies", dataset_override=empty, model_override=model
        )

        assert stats.dataset_exists is False
        assert stats.num_users is None
        # модель из override всё равно подхватилась
        assert stats.last_train_checkpoint == "m.pt"

    def test_default_paths_via_project_root(
        self, tmp_path, monkeypatch, make_dataset_dir, make_model_dir
    ):
        # Без override'ов пути строятся от модульного PROJECT_ROOT.
        monkeypatch.setattr(ds, "PROJECT_ROOT", tmp_path)
        make_dataset_dir(tmp_path / "data" / "processed" / "movies", num_users=50)
        make_model_dir(
            tmp_path / "models" / "movies", name="def_v5.pt", domain="movies"
        )

        stats = ds._collect_domain_stats("movies")

        assert stats.dataset_exists is True
        assert stats.num_users == 50
        assert stats.last_train_checkpoint == "def_v5.pt"
