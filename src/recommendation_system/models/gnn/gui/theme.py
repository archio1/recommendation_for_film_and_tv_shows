"""Общие константы оформления GUI (нулевой слой — без внутренних импортов)."""

from __future__ import annotations

import flet as ft

COLORS = {
    "primary": ft.Colors.BLUE_700,
    "primary_bg": ft.Colors.BLUE_50,
    "accent": ft.Colors.ORANGE_400,
    "ok": ft.Colors.GREEN_600,
    "ok_bg": ft.Colors.GREEN_50,
    "warn": ft.Colors.AMBER_700,
    "err": ft.Colors.RED_600,
    "err_bg": ft.Colors.RED_50,
    "muted": ft.Colors.GREY_600,
    "card_border": ft.Colors.GREY_300,
}


# --- Дизайн-токены: размеры и отступы (единый источник правды) ---
BUTTON_HEIGHT = 40        # высота всех текстовых кнопок действий
CONTENT_MAX_WIDTH = 1100  # центрируемая макс-ширина контента на широком окне

GAP_S = 8
GAP_M = 12
GAP_L = 16
SECTION_GAP = 20


def primary_button(text: str, **kwargs) -> ft.ElevatedButton:
    """Главная кнопка действия — единая высота/цвет (primary)."""
    return ft.ElevatedButton(
        text, bgcolor=COLORS["primary"], color=ft.Colors.WHITE,
        height=BUTTON_HEIGHT, **kwargs,
    )


def danger_button(text: str, **kwargs) -> ft.ElevatedButton:
    """Деструктивное действие (Стоп/отмена) — красная заливка."""
    return ft.ElevatedButton(
        text, bgcolor=COLORS["err"], color=ft.Colors.WHITE,
        height=BUTTON_HEIGHT, **kwargs,
    )


def accent_button(text: str, **kwargs) -> ft.ElevatedButton:
    """Акцентное действие (cross-domain) — оранжевая заливка."""
    return ft.ElevatedButton(
        text, bgcolor=COLORS["accent"], color=ft.Colors.WHITE,
        height=BUTTON_HEIGHT, **kwargs,
    )


def neutral_button(text: str, **kwargs) -> ft.ElevatedButton:
    """Неактивное/декоративное действие — серая заливка."""
    return ft.ElevatedButton(
        text, bgcolor=ft.Colors.GREY_400, color=ft.Colors.WHITE,
        height=BUTTON_HEIGHT, **kwargs,
    )


def secondary_button(text: str, **kwargs) -> ft.OutlinedButton:
    """Вторичное действие — контурная кнопка той же высоты."""
    return ft.OutlinedButton(text, height=BUTTON_HEIGHT, **kwargs)
