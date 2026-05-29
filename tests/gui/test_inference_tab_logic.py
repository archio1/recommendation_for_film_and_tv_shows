"""
Тесты логики вкладки «Тестирование» — `gui/inference_tab.py`.

Без загрузки реальных движков/FAISS: чистые функции (URL/локализация),
избранное (seed), резолв имени/пути чекпоинта (override→latest→v4) и
seed-диагностика по TV_OFFSET. Движки имитируем/обходим.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from recommendation_system.models.gnn.gui import inference_tab as it
from recommendation_system.models.gnn.gui.inference_tab import InferenceTab
from recommendation_system.models.gnn.gui.domain_stats import DomainStats


def _item(**kw):
    base = dict(tmdb_id=603, title="The Matrix", media_type="movie", imdb_id=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _stats(**kw) -> DomainStats:
    base = dict(
        domain="movies",
        dataset_dir=Path("/ds"),
        models_dir=Path("/models"),
        dataset_exists=True,
    )
    base.update(kw)
    return DomainStats(**base)


@pytest.fixture
def tab(fake_app):
    return InferenceTab(fake_app)


# ======================================================================
# _tmdb_url — построение ссылки (TV_OFFSET, IMDb-fallback)
# ======================================================================

class TestTmdbUrl:
    def test_imdb_id_wins(self):
        url = it._tmdb_url(_item(imdb_id="tt0133093"))
        assert url == "https://www.imdb.com/title/tt0133093/"

    def test_movie_without_imdb_uses_tmdb(self):
        url = it._tmdb_url(_item(tmdb_id=603, media_type="movie"))
        assert url == "https://www.themoviedb.org/movie/603"

    def test_tv_subtracts_offset(self):
        url = it._tmdb_url(_item(tmdb_id=10_000_000 + 1399, media_type="tv"))
        assert url == "https://www.themoviedb.org/tv/1399"

    def test_tv_below_offset_not_subtracted(self):
        url = it._tmdb_url(_item(tmdb_id=1399, media_type="tv"))
        assert url == "https://www.themoviedb.org/tv/1399"


# ======================================================================
# _display_title — выбор локализованного названия
# ======================================================================

class TestDisplayTitle:
    def test_en_uses_default_title(self):
        assert it._display_title(_item(title="Matrix"), "en") == "Matrix"

    def test_ru_prefers_title_ru(self):
        item = _item(title="Matrix", title_ru="Матрица")
        assert it._display_title(item, "ru") == "Матрица"

    def test_ru_falls_back_when_empty(self):
        item = _item(title="Matrix", title_ru=None)
        assert it._display_title(item, "ru") == "Matrix"

    def test_uk_prefers_title_uk(self):
        item = _item(title="Matrix", title_uk="Матриця")
        assert it._display_title(item, "uk") == "Матриця"


# ======================================================================
# Избранное (seed)
# ======================================================================

class TestFavorites:
    def test_add_one(self, tab):
        tab._add_to_favorites(_item(tmdb_id=603, title="Matrix", media_type="movie"))
        assert tab.favorites == [(603, "Matrix", "movie")]

    def test_dedup_by_tmdb_id(self, tab):
        tab._add_to_favorites(_item(tmdb_id=603))
        tab._add_to_favorites(_item(tmdb_id=603, title="dup"))
        assert len(tab.favorites) == 1

    def test_add_distinct(self, tab):
        tab._add_to_favorites(_item(tmdb_id=1))
        tab._add_to_favorites(_item(tmdb_id=2))
        assert {t for t, _, _ in tab.favorites} == {1, 2}

    def test_remove(self, tab):
        tab._add_to_favorites(_item(tmdb_id=1))
        tab._add_to_favorites(_item(tmdb_id=2))
        tab._remove_favorite(1)
        assert [t for t, _, _ in tab.favorites] == [2]


# ======================================================================
# _resolve_latest_name — что покажет/загрузит при Switch ON
# ======================================================================

class TestResolveLatestName:
    def test_uses_sidecar_checkpoint(self, tab, monkeypatch):
        monkeypatch.setattr(
            it, "_collect_domain_stats",
            lambda domain: _stats(last_train_checkpoint="best_v7.pt"),
        )
        assert tab._resolve_latest_name("movies") == "best_v7.pt"

    def test_falls_back_to_v4(self, tab, monkeypatch):
        monkeypatch.setattr(
            it, "_collect_domain_stats",
            lambda domain: _stats(last_train_checkpoint=None),
        )
        assert tab._resolve_latest_name("tv") == "lightgcn_tv_best_v4.pt"


# ======================================================================
# Резолв пути чекпоинта в _build_domain_engine (override → latest → v4)
#
# Проверяем выбор пути через FileNotFoundError: метод бросает его ДО
# конструирования тяжёлых движков, как только видит несуществующий .pt.
# ======================================================================

class TestCheckpointResolution:
    def _call(self, tab):
        # device/классы движков не используются — исключение раньше.
        return tab._build_domain_engine("movies", "cpu", object(), object())

    def test_override_wins(self, tab, monkeypatch, tmp_path):
        monkeypatch.setattr(it, "_collect_domain_stats",
                            lambda domain: _stats(last_train_checkpoint="x.pt"))
        tab.movies_checkpoint_override = tmp_path / "my_override.pt"
        with pytest.raises(FileNotFoundError) as ei:
            self._call(tab)
        assert "my_override.pt" in str(ei.value)

    def test_latest_sidecar_path(self, tab, monkeypatch, tmp_path):
        monkeypatch.setattr(
            it, "_collect_domain_stats",
            lambda domain: _stats(models_dir=tmp_path, last_train_checkpoint="best_v7.pt"),
        )
        tab.movies_checkpoint_override = None
        with pytest.raises(FileNotFoundError) as ei:
            self._call(tab)
        assert "best_v7.pt" in str(ei.value)

    def test_v4_fallback_path(self, tab, monkeypatch, tmp_path):
        monkeypatch.setattr(
            it, "_collect_domain_stats",
            lambda domain: _stats(models_dir=tmp_path, last_train_checkpoint=None),
        )
        tab.movies_checkpoint_override = None
        with pytest.raises(FileNotFoundError) as ei:
            self._call(tab)
        assert "lightgcn_movies_best_v4.pt" in str(ei.value)


# ======================================================================
# _seed_diagnostics — split по TV_OFFSET + охват графа
# ======================================================================

class TestSeedDiagnostics:
    def test_split_movie_tv_no_engines(self, tab):
        # Движки не загружены → только строка про seed.
        out = tab._seed_diagnostics([1, 2, 10_000_005], scope="all")
        assert "2 movie" in out and "1 tv" in out

    def test_counts_known_in_movies_graph(self, tab):
        tab.movies_engine = types.SimpleNamespace(tmdb_to_item_id={1: 0})
        out = tab._seed_diagnostics([1, 2], scope="movie")
        assert "movies graph" in out
        assert "1/2" in out
