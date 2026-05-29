"""
Тесты логики вкладки «Создание датасета» — `gui/dataset_tab.py`.

Без запуска сборки: проверяем парсинг параметров (UI→kwargs), pre-flight
проверку источников, post-build sanity и состояние пресетов. Конструируем
DatasetTab с фейковой page (без окна Flet).
"""

from __future__ import annotations

import pytest

from recommendation_system.models.gnn.gui import dataset_tab as dt
from recommendation_system.models.gnn.gui.dataset_tab import DatasetTab


@pytest.fixture
def tab(fake_app):
    return DatasetTab(fake_app)


# ======================================================================
# _collect_params — UI → kwargs для MovieDatasetProcessor
# ======================================================================

class TestCollectParams:
    def test_default_preset_values(self, tab):
        params = tab._collect_params()
        assert params["top_n_movies"] == 15000
        assert params["top_n_tv"] == 10000
        assert params["min_user_interactions"] == 10
        assert params["min_item_interactions"] == 10
        assert params["rating_threshold"] == pytest.approx(3.5)
        assert params["max_interactions"] == 15_000_000
        assert params["min_year"] is None  # пустое поле
        assert params["languages"] == ["en"]

    def test_empty_fields_use_sentinels(self, tab):
        for name in dt._PARAM_ORDER:
            tab.param_inputs[name].value = ""
        params = tab._collect_params()

        assert params["top_n_movies"] == dt._HUGE_INT
        assert params["top_n_tv"] == dt._HUGE_INT
        assert params["max_interactions"] == dt._HUGE_INT
        assert params["min_user_interactions"] == 0
        assert params["min_item_interactions"] == 0
        assert params["rating_threshold"] == 0.0
        assert params["min_year"] is None
        assert params["languages"] == ["en"]  # пустой список → дефолт en

    def test_languages_parsed_and_stripped(self, tab):
        tab.param_inputs["languages"].value = "en, ru ,uk"
        params = tab._collect_params()
        assert params["languages"] == ["en", "ru", "uk"]

    def test_min_year_parsed_when_present(self, tab):
        tab.param_inputs["min_year"].value = "2010"
        params = tab._collect_params()
        assert params["min_year"] == 2010

    def test_non_numeric_raises_value_error(self, tab):
        tab.param_inputs["top_n_movies"].value = "abc"
        with pytest.raises(ValueError):
            tab._collect_params()


# ======================================================================
# Пресеты / состояние параметров
# ======================================================================

class TestPresets:
    def test_apply_smoke_preset(self, tab, make_event):
        tab._on_preset_change(make_event(value="smoke"))
        assert tab.param_inputs["top_n_movies"].value == "500"
        assert tab.param_inputs["top_n_tv"].value == "200"
        assert tab.param_inputs["min_year"].value == "2010"

    def test_apply_full_preset_clears_limits(self, tab, make_event):
        tab._on_preset_change(make_event(value="full"))
        assert tab.param_inputs["top_n_movies"].value == ""
        assert tab.param_inputs["max_interactions"].value == ""

    def test_custom_preset_is_noop(self, tab, make_event):
        before = tab.param_inputs["top_n_movies"].value
        tab._on_preset_change(make_event(value="custom"))
        assert tab.param_inputs["top_n_movies"].value == before

    def test_manual_edit_flips_preset_to_custom(self, tab, make_event):
        tab.preset_dd.value = "default"
        tab._on_param_change(make_event())
        assert tab.preset_dd.value == "custom"

    def test_programmatic_preset_does_not_flip_to_custom(self, tab, make_event):
        # _on_preset_change применяет значения под флагом _preset_silent —
        # вызванный им on_change не должен сваливать preset в custom.
        tab._on_preset_change(make_event(value="smoke"))
        assert tab.preset_dd.value != "custom"


# ======================================================================
# _preflight_check — обязательные источники
# ======================================================================

class TestPreflightCheck:
    def _make_ml(self, root, files=("ratings.csv", "movies.csv", "links.csv")):
        ml = root / "data" / "raw" / "ml-32m"
        ml.mkdir(parents=True, exist_ok=True)
        for f in files:
            (ml / f).write_text("x", encoding="utf-8")

    def test_movies_all_present_ok(self, tab, tmp_path, monkeypatch):
        monkeypatch.setattr(dt, "PROJECT_ROOT", tmp_path)
        self._make_ml(tmp_path)
        assert tab._preflight_check("movies") == []

    def test_movies_missing_files_reported(self, tab, tmp_path, monkeypatch):
        monkeypatch.setattr(dt, "PROJECT_ROOT", tmp_path)
        self._make_ml(tmp_path, files=("ratings.csv",))  # нет movies.csv/links.csv
        missing = tab._preflight_check("movies")
        labels = [label for label, _p, _h in missing]
        assert "MovieLens / movies.csv" in labels
        assert "MovieLens / links.csv" in labels
        assert len(missing) == 2

    def test_tv_missing_sources_reported(self, tab, tmp_path, monkeypatch):
        monkeypatch.setattr(dt, "PROJECT_ROOT", tmp_path)
        # ничего не создаём → оба trakt-файла отсутствуют
        missing = tab._preflight_check("tv")
        assert len(missing) == 2

    def test_all_domain_combines_movies_and_tv(self, tab, tmp_path, monkeypatch):
        monkeypatch.setattr(dt, "PROJECT_ROOT", tmp_path)
        # ничего нет → 3 movies (ratings/movies/links) + 2 tv
        missing = tab._preflight_check("all")
        assert len(missing) == 5


# ======================================================================
# _dataset_sanity — post-build валидация
# ======================================================================

class TestDatasetSanity:
    def _processed(self, root, name):
        return root / "data" / "processed" / name

    def test_happy(self, tmp_path, monkeypatch, make_dataset_dir):
        monkeypatch.setattr(dt, "PROJECT_ROOT", tmp_path)
        make_dataset_dir(self._processed(tmp_path, "smoke"), n_interactions=120_000)
        ok, failures = dt._dataset_sanity("smoke")
        assert ok is True
        assert failures == []

    def test_missing_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dt, "PROJECT_ROOT", tmp_path)
        self._processed(tmp_path, "empty").mkdir(parents=True)
        ok, failures = dt._dataset_sanity("empty")
        assert ok is False
        assert any("missing" in f for f in failures)

    def test_too_few_interactions(self, tmp_path, monkeypatch, make_dataset_dir):
        monkeypatch.setattr(dt, "PROJECT_ROOT", tmp_path)
        make_dataset_dir(self._processed(tmp_path, "tiny"), n_interactions=200)
        ok, failures = dt._dataset_sanity("tiny")
        assert ok is False
        assert any("100k" in f for f in failures)

    def test_duplicate_tmdb_ids(self, tmp_path, monkeypatch, make_dataset_dir):
        monkeypatch.setattr(dt, "PROJECT_ROOT", tmp_path)
        make_dataset_dir(
            self._processed(tmp_path, "dup"),
            n_interactions=120_000, duplicate_tmdb=True,
        )
        ok, failures = dt._dataset_sanity("dup")
        assert ok is False
        assert any("duplicate" in f for f in failures)

    def test_items_schema_mismatch(self, tmp_path, monkeypatch, make_dataset_dir):
        monkeypatch.setattr(dt, "PROJECT_ROOT", tmp_path)
        make_dataset_dir(
            self._processed(tmp_path, "badcols"),
            n_interactions=120_000, items_columns=("tmdb_id", "title"),  # нет genres
        )
        ok, failures = dt._dataset_sanity("badcols")
        assert ok is False
        assert any("schema mismatch" in f for f in failures)

    def test_invalid_mapping_counts(self, tmp_path, monkeypatch, make_dataset_dir):
        monkeypatch.setattr(dt, "PROJECT_ROOT", tmp_path)
        make_dataset_dir(
            self._processed(tmp_path, "badmap"),
            n_interactions=120_000, num_users=10,
            mapping_text='{"num_users": 0, "num_items": 5}',
        )
        ok, failures = dt._dataset_sanity("badmap")
        assert ok is False
        assert any("num_users" in f for f in failures)
