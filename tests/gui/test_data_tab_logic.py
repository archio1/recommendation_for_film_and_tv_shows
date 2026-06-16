"""
Тесты логики вкладки «Данные» — `gui/data_tab.py`.

Форматтеры, override-свитчи (latest ↔ выбранный путь) и refresh() с
tmp-датасетом/моделью.
"""

from __future__ import annotations

import types
from datetime import datetime
from pathlib import Path

import pytest

from recommendation_system.models.gnn.gui import data_tab as dtab
from recommendation_system.models.gnn.gui.data_tab import DataTab


@pytest.fixture
def tab(fake_app):
    return DataTab(fake_app)


# ======================================================================
# Форматтеры
# ======================================================================

class TestFormatters:
    @pytest.mark.parametrize("value,expected", [
        (None, "—"),
        (0, "0"),
        (1234567, "1,234,567"),
    ])
    def test_fmt_int(self, value, expected):
        assert dtab._fmt_int(value) == expected

    def test_fmt_int_non_int(self):
        assert dtab._fmt_int("x") == "—"

    def test_fmt_mtime(self):
        assert dtab._fmt_mtime(None) == "—"
        dt = datetime(2026, 5, 29, 13, 59)
        assert dtab._fmt_mtime(dt) == "2026-05-29 13:59"

    def test_fmt_iso_valid(self):
        assert dtab._fmt_iso("2026-05-29T13:59:00Z") == "2026-05-29 13:59 UTC"

    def test_fmt_iso_empty(self):
        assert dtab._fmt_iso("") == "—"
        assert dtab._fmt_iso(None) == "—"

    def test_fmt_iso_invalid_returns_asis(self):
        assert dtab._fmt_iso("not-a-date") == "not-a-date"


# ======================================================================
# Override-свитчи и пикеры
# ======================================================================

class TestOverrides:
    def test_switch_on_clears_override(self, tab):
        tab._dataset_overrides["movies"] = Path("/some/where")
        tab._dataset_switches["movies"].value = True
        tab._on_dataset_switch("movies")
        assert tab._dataset_overrides["movies"] is None

    def test_dataset_picked_sets_override(self, tab, tmp_path):
        ev = types.SimpleNamespace(path=str(tmp_path))
        tab._on_dataset_picked("movies", ev)
        assert tab._dataset_overrides["movies"] == Path(str(tmp_path))

    def test_dataset_picked_cancel_resets_switch(self, tab):
        ev = types.SimpleNamespace(path=None)
        tab._on_dataset_picked("movies", ev)
        assert tab._dataset_switches["movies"].value is True
        assert tab._dataset_overrides["movies"] is None

    def test_model_picked_sets_override(self, tab, tmp_path):
        pt = tmp_path / "m.pt"
        ev = types.SimpleNamespace(files=[types.SimpleNamespace(path=str(pt))])
        tab._on_model_picked("movies", ev)
        assert tab._model_overrides["movies"] == Path(str(pt))

    def test_model_picked_cancel_resets_switch(self, tab):
        ev = types.SimpleNamespace(files=[])
        tab._on_model_picked("tv", ev)
        assert tab._model_switches["tv"].value is True


# ======================================================================
# refresh() с override на tmp-артефакты
# ======================================================================

class TestRefresh:
    def test_refresh_with_overrides(self, tab, make_dataset_dir, make_model_dir, tmp_path):
        ds = make_dataset_dir(tmp_path / "myds", num_users=11)
        pt = make_model_dir(tmp_path / "mymodels", name="picked_v9.pt")

        tab._dataset_overrides["movies"] = ds
        tab._model_overrides["movies"] = pt
        tab.refresh()

        # _sync_controls отразил override: switch OFF, в поле — имя выбранного.
        assert tab._dataset_switches["movies"].value is False
        assert tab._dataset_inputs["movies"].value == ds.name
        assert tab._model_switches["movies"].value is False
        assert tab._model_inputs["movies"].value == "picked_v9.pt"
