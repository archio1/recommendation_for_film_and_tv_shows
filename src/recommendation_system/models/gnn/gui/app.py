"""Главное Flet-приложение: общее состояние (device) + 4-табный layout."""

from __future__ import annotations

import flet as ft
import torch

from recommendation_system.models.gnn.gui.data_tab import DataTab
from recommendation_system.models.gnn.gui.dataset_tab import DatasetTab
from recommendation_system.models.gnn.gui.inference_tab import InferenceTab
from recommendation_system.models.gnn.gui.theme import COLORS, CONTENT_MAX_WIDTH
from recommendation_system.models.gnn.gui.training_tab import TrainingTab


class TrainerGuiApp:
    """Main Flet app holding shared state and the 4-tab layout."""

    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.training_tab = TrainingTab(self)
        self.dataset_tab = DatasetTab(self)
        self.inference_tab = InferenceTab(self)
        self.data_tab = DataTab(self)
        self._build_layout()

    def current_device(self) -> str:
        return self.device

    def _build_layout(self) -> None:
        self.page.title = "Recommendation System — Trainer GUI"
        self.page.padding = 16
        self.page.theme_mode = ft.ThemeMode.LIGHT

        header = self._build_header()
        tabs = self._build_tabs()

        # Контент всегда expand=True (поэтому виден при любой/неизвестной ширине,
        # в т.ч. сразу на старте). Ограничение по ширине (CONTENT_MAX_WIDTH) на
        # широком окне делаем симметричным горизонтальным padding'ом — это даёт
        # эффект «центрированной max-width колонки» без фиксированной ширины,
        # которая схлопывалась бы в центрирующем Row.
        self._content = ft.Container(
            content=ft.Column(
                [header, ft.Divider(height=1), tabs], expand=True, spacing=12
            ),
            expand=True,
        )
        self.page.on_resized = self._on_page_resized
        self.page.add(self._content)
        self._apply_content_width()
        # Отложенно домерить ширину: на старте page.width ещё None, поэтому
        # центрирование на широком мониторе включаем, как только размер известен.
        self.page.run_task(self._poll_initial_width)

    def _page_width(self) -> float | None:
        """Текущая ширина окна, если известна (иначе None)."""
        w = getattr(self.page, "width", None)
        if not w:
            window = getattr(self.page, "window", None)
            w = getattr(window, "width", None) if window else None
        return w

    def _apply_content_width(self) -> None:
        # Центрируем через горизонтальный padding: side = половина «лишней»
        # ширины сверх CONTENT_MAX_WIDTH. -32 компенсирует page.padding (16×2).
        w = self._page_width()
        side = 0.0
        if w and (w - 32) > CONTENT_MAX_WIDTH:
            side = (w - 32 - CONTENT_MAX_WIDTH) / 2
        self._content.padding = (
            ft.padding.symmetric(horizontal=side) if side else 0
        )

    async def _poll_initial_width(self) -> None:
        # На старте размер окна приходит не сразу — опрашиваем до ~1с.
        import asyncio
        for _ in range(20):
            if self._page_width():
                break
            await asyncio.sleep(0.05)
        self._apply_content_width()
        self.page.update()

    def _on_page_resized(self, e) -> None:
        self._apply_content_width()
        self.page.update()

    def _build_header(self) -> ft.Control:
        title = ft.Text(
            "Recommendation Trainer",
            size=22,
            weight=ft.FontWeight.BOLD,
            color=COLORS["primary"],
        )
        # Domain selector живёт внутри каждой вкладки (TrainingTab.domain_dd,
        # DatasetTab.domain_dd, InferenceTab.scope_dd). Глобальным остаётся
        # только Device — он влияет и на train, и на загрузку inference-движков.
        self.device_selector = ft.SegmentedButton(
            selected={self.device},
            allow_multiple_selection=False,
            segments=[
                ft.Segment(value="cpu", label=ft.Text("CPU"),
                           icon=ft.Icon(ft.Icons.COMPUTER)),
                ft.Segment(
                    value="cuda",
                    label=ft.Text("GPU"),
                    icon=ft.Icon(ft.Icons.BOLT),
                    disabled=not torch.cuda.is_available(),
                ),
            ],
            on_change=self._on_device_changed,
        )
        return ft.Row(
            [title, ft.Container(expand=True), self.device_selector],
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=16,
        )

    def _build_tabs(self) -> ft.Control:
        return ft.Tabs(
            selected_index=2,
            expand=True,
            tabs=[
                ft.Tab(text="Обучение", icon=ft.Icons.SCHOOL,
                       content=self.training_tab.build()),
                ft.Tab(text="Создание датасета", icon=ft.Icons.BUILD,
                       content=self.dataset_tab.build()),
                ft.Tab(text="Тестирование", icon=ft.Icons.SCIENCE,
                       content=self.inference_tab.build()),
                ft.Tab(text="Данные", icon=ft.Icons.STORAGE,
                       content=self.data_tab.build()),
            ],
        )

    def _on_device_changed(self, e: ft.ControlEvent) -> None:
        selected = list(e.control.selected)
        if selected:
            self.device = selected[0]
        self.page.update()


def main(page: ft.Page) -> None:
    TrainerGuiApp(page)
