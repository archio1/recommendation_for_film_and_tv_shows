"""
Тесты корневого приложения — `gui/app.py`.

Фокус: логика адаптивной ширины (`_page_width`/`_apply_content_width`) с
регрессом на баг «пустых вкладок на старте», переключатель устройства и
smoke-конструкция всех 4 вкладок + единая высота кнопок.
"""

from __future__ import annotations

import types

import pytest

from recommendation_system.models.gnn.gui.app import TrainerGuiApp
from recommendation_system.models.gnn.gui.theme import BUTTON_HEIGHT, CONTENT_MAX_WIDTH


@pytest.fixture
def app(make_fake_page):
    return TrainerGuiApp(make_fake_page(1920))


def _side(content) -> float:
    """Горизонтальный padding контента (0, если padding не задан)."""
    p = content.padding
    return 0 if not p else p.left


# ======================================================================
# _page_width
# ======================================================================

class TestPageWidth:
    def test_prefers_page_width(self, app):
        app.page.width = 1234
        assert app._page_width() == 1234

    def test_falls_back_to_window_width(self, app):
        app.page.width = None
        app.page.window.width = 800
        assert app._page_width() == 800

    def test_none_when_both_unknown(self, app):
        app.page.width = None
        app.page.window.width = None
        assert app._page_width() is None


# ======================================================================
# _apply_content_width — адаптив + max-width
# ======================================================================

class TestContentWidth:
    def test_unknown_width_stays_visible(self, app):
        # Регресс: на старте ширина неизвестна → контент НЕ должен схлопываться.
        app.page.width = None
        app.page.window.width = None
        app._apply_content_width()
        assert app._content.expand is True
        assert _side(app._content) == 0

    def test_narrow_window_full_width(self, app):
        app.page.width = 700
        app._apply_content_width()
        assert _side(app._content) == 0  # на всю ширину, без боковых полей

    def test_exactly_max_no_padding(self, app):
        app.page.width = CONTENT_MAX_WIDTH + 32  # ровно укладывается с учётом page.padding
        app._apply_content_width()
        assert _side(app._content) == 0

    def test_wide_window_centers(self, app):
        app.page.width = 1920
        app._apply_content_width()
        expected = (1920 - 32 - CONTENT_MAX_WIDTH) / 2
        assert _side(app._content) == pytest.approx(expected)


# ======================================================================
# Переключатель устройства
# ======================================================================

class TestDeviceSwitch:
    def _event(self, value):
        return types.SimpleNamespace(control=types.SimpleNamespace(selected={value}))

    def test_switch_to_cpu(self, app):
        app._on_device_changed(self._event("cpu"))
        assert app.device == "cpu"
        assert app.current_device() == "cpu"

    def test_empty_selection_keeps_device(self, app):
        app.device = "cpu"
        app._on_device_changed(types.SimpleNamespace(
            control=types.SimpleNamespace(selected=set())
        ))
        assert app.device == "cpu"


# ======================================================================
# Smoke: конструкция всех вкладок + консистентность кнопок
# ======================================================================

class TestSmokeAndButtons:
    def test_all_tabs_constructed(self, app):
        assert app.training_tab is not None
        assert app.dataset_tab is not None
        assert app.inference_tab is not None
        assert app.data_tab is not None

    def test_filepickers_registered_in_overlay(self, app):
        # training 1 + dataset 5 + inference 2 + data 4 = 12
        assert len(app.page.overlay) == 12

    def test_action_buttons_share_height(self, app):
        buttons = [
            app.training_tab.start_btn, app.training_tab.stop_btn,
            app.training_tab.sidecar_btn, app.training_tab.folder_btn,
            app.dataset_tab.build_btn, app.dataset_tab.stop_btn,
            app.dataset_tab.open_folder_btn,
            app.inference_tab.recs_movie_btn, app.inference_tab.recs_tv_btn,
            app.inference_tab.recs_all_btn, app.inference_tab.recs_cross_btn,
        ]
        assert all(b.height == BUTTON_HEIGHT for b in buttons)
