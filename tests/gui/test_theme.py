"""
Тесты дизайн-токенов и фабрик кнопок — `gui/theme.py`.

Дешёвый guard консистентности: все фабрики дают единую высоту и ожидаемые
цвета, а kwargs (icon/on_click/disabled/visible) пробрасываются.
"""

from __future__ import annotations

import flet as ft

from recommendation_system.models.gnn.gui import theme


def test_tokens_present():
    assert theme.BUTTON_HEIGHT == 40
    assert theme.CONTENT_MAX_WIDTH == 1100


class TestButtonFactories:
    def test_primary(self):
        b = theme.primary_button("Go")
        assert isinstance(b, ft.ElevatedButton)
        assert b.height == theme.BUTTON_HEIGHT
        assert b.bgcolor == theme.COLORS["primary"]
        assert b.color == ft.Colors.WHITE

    def test_danger(self):
        b = theme.danger_button("Stop")
        assert b.height == theme.BUTTON_HEIGHT
        assert b.bgcolor == theme.COLORS["err"]

    def test_accent(self):
        b = theme.accent_button("Cross")
        assert b.height == theme.BUTTON_HEIGHT
        assert b.bgcolor == theme.COLORS["accent"]

    def test_neutral(self):
        b = theme.neutral_button("Idle")
        assert b.height == theme.BUTTON_HEIGHT
        assert b.bgcolor == ft.Colors.GREY_400

    def test_secondary_is_outlined(self):
        b = theme.secondary_button("Open")
        assert isinstance(b, ft.OutlinedButton)
        assert b.height == theme.BUTTON_HEIGHT

    def test_all_factories_share_height(self):
        heights = {
            theme.primary_button("a").height,
            theme.danger_button("b").height,
            theme.accent_button("c").height,
            theme.neutral_button("d").height,
            theme.secondary_button("e").height,
        }
        assert heights == {theme.BUTTON_HEIGHT}

    def test_kwargs_passthrough(self):
        clicked = []
        b = theme.primary_button(
            "X", icon=ft.Icons.PLAY_ARROW,
            on_click=lambda e: clicked.append(e), disabled=True, visible=False,
        )
        assert b.icon == ft.Icons.PLAY_ARROW
        assert b.disabled is True
        assert b.visible is False
        assert callable(b.on_click)
