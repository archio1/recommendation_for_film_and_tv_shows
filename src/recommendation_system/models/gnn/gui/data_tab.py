"""Вкладка «Данные» — readonly дашборд статистики датасетов/моделей по доменам."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import flet as ft

from recommendation_system.models.gnn.gui.common import _open_in_explorer
from recommendation_system.models.gnn.gui.domain_stats import (
    DomainStats,
    _collect_domain_stats,
)
from recommendation_system.models.gnn.gui.theme import (
    COLORS,
    PROJECT_ROOT,
    secondary_button,
)

if TYPE_CHECKING:
    from recommendation_system.models.gnn.gui.app import TrainerGuiApp


class DataTab:
    """Per-domain dashboard with arbitrary dataset/model pickers.
    Default: latest in `data/processed/{domain}` and `models/{domain}`.
    Switch OFF + picker = inspect any folder / any .pt (incl. v4 without sidecar)."""

    def __init__(self, app: "TrainerGuiApp") -> None:
        self.app = app
        self._dataset_overrides: dict[str, Optional[Path]] = {"movies": None, "tv": None}
        self._model_overrides: dict[str, Optional[Path]] = {"movies": None, "tv": None}
        self._dataset_switches: dict[str, ft.Switch] = {}
        self._model_switches: dict[str, ft.Switch] = {}
        self._dataset_inputs: dict[str, ft.TextField] = {}
        self._model_inputs: dict[str, ft.TextField] = {}
        self._dataset_pickers: dict[str, ft.FilePicker] = {}
        self._model_pickers: dict[str, ft.FilePicker] = {}
        self._build()
        self.refresh()

    def build(self) -> ft.Control:
        return self._root

    def refresh(self) -> None:
        for d in ("movies", "tv"):
            stats = _collect_domain_stats(
                d,
                dataset_override=self._dataset_overrides[d],
                model_override=self._model_overrides[d],
            )
            card = self._movies_card if d == "movies" else self._tv_card
            card.content = self._build_card(stats, d)
            self._sync_controls(d, stats)

    def _sync_controls(self, domain: str, stats: DomainStats) -> None:
        ds_override = self._dataset_overrides[domain]
        ds_input = self._dataset_inputs[domain]
        if ds_override is not None:
            ds_input.value = ds_override.name
            ds_input.tooltip = str(ds_override)
        else:
            ds_input.value = f"latest → {stats.dataset_dir.name}"
            ds_input.tooltip = str(stats.dataset_dir)
        self._dataset_switches[domain].value = ds_override is None

        m_override = self._model_overrides[domain]
        m_input = self._model_inputs[domain]
        if m_override is not None:
            m_input.value = m_override.name
            m_input.tooltip = str(m_override)
        elif stats.last_train_checkpoint:
            m_input.value = f"latest → {stats.last_train_checkpoint}"
            m_input.tooltip = str(stats.models_dir / stats.last_train_checkpoint)
        else:
            m_input.value = "(нет моделей)"
            m_input.tooltip = str(stats.models_dir)
        self._model_switches[domain].value = m_override is None

    def _build(self) -> None:
        for d in ("movies", "tv"):
            ds_picker = ft.FilePicker(
                on_result=lambda e, dom=d: self._on_dataset_picked(dom, e)
            )
            m_picker = ft.FilePicker(
                on_result=lambda e, dom=d: self._on_model_picked(dom, e)
            )
            self._dataset_pickers[d] = ds_picker
            self._model_pickers[d] = m_picker
            self.app.page.overlay.append(ds_picker)
            self.app.page.overlay.append(m_picker)

            self._dataset_inputs[d] = ft.TextField(
                read_only=True, dense=True, expand=True,
                text_size=12, content_padding=8,
            )
            self._model_inputs[d] = ft.TextField(
                read_only=True, dense=True, expand=True,
                text_size=12, content_padding=8,
            )
            self._dataset_switches[d] = ft.Switch(
                value=True, scale=0.7,
                tooltip="ON = latest. OFF → выбрать папку через 📂",
                on_change=lambda e, dom=d: self._on_dataset_switch(dom),
            )
            self._model_switches[d] = ft.Switch(
                value=True, scale=0.7,
                tooltip="ON = latest. OFF → выбрать .pt через 📂",
                on_change=lambda e, dom=d: self._on_model_switch(dom),
            )

        self._movies_card = ft.Container(
            padding=16, bgcolor=ft.Colors.WHITE, border_radius=8,
            border=ft.border.all(1, COLORS["card_border"]),
            expand=True,
        )
        self._tv_card = ft.Container(
            padding=16, bgcolor=ft.Colors.WHITE, border_radius=8,
            border=ft.border.all(1, COLORS["card_border"]),
            expand=True,
        )

        header = ft.Row(
            [
                ft.Text(
                    "Состояние датасетов и моделей",
                    size=18,
                    weight=ft.FontWeight.BOLD,
                    color=COLORS["primary"],
                ),
                ft.Container(expand=True),
                ft.IconButton(
                    ft.Icons.REFRESH,
                    tooltip="Обновить",
                    on_click=self._on_refresh,
                ),
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        self._root = ft.Container(
            content=ft.Column(
                [
                    header,
                    ft.Row(
                        [self._movies_card, self._tv_card],
                        spacing=16,
                        vertical_alignment=ft.CrossAxisAlignment.START,
                    ),
                ],
                spacing=16,
                expand=True,
                scroll=ft.ScrollMode.AUTO,
            ),
            padding=16,
        )

    # ---- picker / switch handlers -------------------------------------

    def _open_dataset_picker(self, domain: str) -> None:
        initial = self._dataset_overrides[domain] or (PROJECT_ROOT / "data" / "processed" / domain)
        if not initial.exists():
            initial = PROJECT_ROOT / "data" / "processed"
        if not initial.exists():
            initial = PROJECT_ROOT
        self._dataset_pickers[domain].get_directory_path(
            dialog_title=f"Выбери папку датасета ({domain})",
            initial_directory=str(initial),
        )

    def _open_model_picker(self, domain: str) -> None:
        current = self._model_overrides[domain]
        initial = current.parent if current else (PROJECT_ROOT / "models" / domain)
        if not initial.exists():
            initial = PROJECT_ROOT / "models"
        if not initial.exists():
            initial = PROJECT_ROOT
        self._model_pickers[domain].pick_files(
            dialog_title=f"Выбери .pt модели ({domain})",
            initial_directory=str(initial),
            allowed_extensions=["pt"],
            allow_multiple=False,
        )

    def _on_dataset_picked(self, domain: str, e: ft.FilePickerResultEvent) -> None:
        if not e.path:
            # User cancelled — restore latest.
            self._dataset_switches[domain].value = True
            self.app.page.update()
            return
        self._dataset_overrides[domain] = Path(e.path)
        self.refresh()
        self.app.page.update()

    def _on_model_picked(self, domain: str, e: ft.FilePickerResultEvent) -> None:
        if not e.files:
            self._model_switches[domain].value = True
            self.app.page.update()
            return
        self._model_overrides[domain] = Path(e.files[0].path)
        self.refresh()
        self.app.page.update()

    def _on_dataset_switch(self, domain: str) -> None:
        if self._dataset_switches[domain].value:
            self._dataset_overrides[domain] = None
            self.refresh()
            self.app.page.update()
        else:
            # Flipped OFF — open picker so user can choose.
            self._open_dataset_picker(domain)

    def _on_model_switch(self, domain: str) -> None:
        if self._model_switches[domain].value:
            self._model_overrides[domain] = None
            self.refresh()
            self.app.page.update()
        else:
            self._open_model_picker(domain)

    def _on_refresh(self, e: ft.ControlEvent) -> None:
        self.refresh()
        self.app.page.update()

    # ---- card layout --------------------------------------------------

    def _dataset_controls_row(self, domain: str) -> ft.Control:
        return ft.Row(
            [
                self._dataset_switches[domain],
                ft.Text("latest", size=11, color=COLORS["muted"]),
                self._dataset_inputs[domain],
                ft.IconButton(
                    ft.Icons.FOLDER_OPEN,
                    tooltip="Выбрать папку датасета",
                    icon_size=18,
                    on_click=lambda _e, dom=domain: self._open_dataset_picker(dom),
                ),
            ],
            spacing=4,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

    def _model_controls_row(self, domain: str) -> ft.Control:
        return ft.Row(
            [
                self._model_switches[domain],
                ft.Text("latest", size=11, color=COLORS["muted"]),
                self._model_inputs[domain],
                ft.IconButton(
                    ft.Icons.UPLOAD_FILE,
                    tooltip="Выбрать .pt модели",
                    icon_size=18,
                    on_click=lambda _e, dom=domain: self._open_model_picker(dom),
                ),
            ],
            spacing=4,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

    def _build_card(self, stats: DomainStats, domain: str) -> ft.Control:
        title = "🎬 Movies" if stats.domain == "movies" else "📺 TV"
        title_row = ft.Text(title, size=20, weight=ft.FontWeight.BOLD,
                            color=COLORS["primary"])

        if stats.dataset_exists:
            dataset_section: ft.Control = ft.Column(
                [
                    _stat_row("Users", _fmt_int(stats.num_users)),
                    _stat_row("Items", _fmt_int(stats.num_items)),
                    _stat_row("Interactions", _fmt_int(stats.num_interactions)),
                    _stat_row("interactions mtime", _fmt_mtime(stats.interactions_mtime)),
                    _stat_row("items mtime", _fmt_mtime(stats.items_mtime)),
                ],
                spacing=4,
            )
        else:
            dataset_section = ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.WARNING_AMBER, color=COLORS["warn"], size=16),
                            ft.Text(
                                "Файлы датасета не найдены в выбранной папке",
                                color=COLORS["warn"], size=12,
                            ),
                        ],
                        spacing=4,
                    ),
                    ft.Text(
                        f"Ожидаются: interactions_final.parquet / "
                        f"items_metadata_final.parquet / id_mapping.json",
                        size=11, italic=True, color=COLORS["muted"],
                    ),
                    ft.Text(
                        f"Путь: {stats.dataset_dir}",
                        size=11, color=COLORS["muted"], selectable=True,
                    ),
                ],
                spacing=4,
            )

        train_block = self._build_train_block(stats)

        return ft.Column(
            [
                title_row,
                ft.Divider(height=4),
                ft.Text("Датасет", size=13, weight=ft.FontWeight.W_500,
                        color=COLORS["muted"]),
                self._dataset_controls_row(domain),
                dataset_section,
                ft.Divider(height=4),
                ft.Text("Последняя тренировка", size=13, weight=ft.FontWeight.W_500,
                        color=COLORS["muted"]),
                self._model_controls_row(domain),
                train_block,
                ft.Container(expand=True),
                ft.Row(
                    [
                        secondary_button(
                            "Папка датасета",
                            icon=ft.Icons.FOLDER_OPEN,
                            on_click=lambda _e: _open_in_explorer(stats.dataset_dir),
                        ),
                        secondary_button(
                            "Папка моделей",
                            icon=ft.Icons.MODEL_TRAINING,
                            on_click=lambda _e: _open_in_explorer(stats.models_dir)
                            if stats.models_dir.exists() else None,
                            disabled=not stats.models_dir.exists(),
                        ),
                    ],
                    spacing=8,
                ),
            ],
            spacing=8,
        )

    def _build_train_block(self, stats: DomainStats) -> ft.Control:
        # Three branches:
        # 1. Have sidecar (full metrics) — last_train_at set.
        # 2. Have .pt but no sidecar (old Colab v4) — checkpoint set, last_train_at None.
        # 3. No model at all — both None.
        if stats.last_train_at is None and stats.last_train_checkpoint is None:
            return ft.Row(
                [
                    ft.Icon(ft.Icons.INFO_OUTLINE, color=COLORS["muted"], size=16),
                    ft.Text("Модель не выбрана и не найдена",
                            color=COLORS["muted"], size=13),
                ],
                spacing=4,
            )

        if stats.last_train_at is None:
            # Sidecar-less branch (v4 Colab or any .pt without .json).
            return ft.Column(
                [
                    _stat_row("checkpoint", stats.last_train_checkpoint or "—"),
                    _stat_row("file mtime", _fmt_mtime(stats.model_mtime)),
                    _stat_row(
                        "size",
                        f"{stats.model_size_mb:.1f} MB"
                        if stats.model_size_mb is not None else "—",
                    ),
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.INFO_OUTLINE, color=COLORS["muted"], size=14),
                            ft.Text(
                                "Sidecar отсутствует — нет метрик/sanity "
                                "(модель обучалась не локальным trainer.py)",
                                color=COLORS["muted"], size=11, italic=True,
                            ),
                        ],
                        spacing=4,
                    ),
                ],
                spacing=4,
            )

        sanity_chip: list = []
        if stats.last_train_sanity_ok is True:
            sanity_chip = [ft.Icon(ft.Icons.CHECK_CIRCLE, color=COLORS["ok"], size=16),
                           ft.Text("sanity OK", color=COLORS["ok"], size=12)]
        elif stats.last_train_sanity_ok is False:
            sanity_chip = [ft.Icon(ft.Icons.ERROR_OUTLINE, color=COLORS["err"], size=16),
                           ft.Text("sanity FAIL", color=COLORS["err"], size=12)]

        rows: list[ft.Control] = [
            _stat_row("trained at", _fmt_iso(stats.last_train_at)),
            _stat_row("checkpoint", stats.last_train_checkpoint or "—"),
            _stat_row("recall@10",
                      f"{stats.last_train_recall:.4f}"
                      if stats.last_train_recall is not None else "—"),
        ]
        if sanity_chip:
            rows.append(ft.Row(sanity_chip, spacing=4))
        return ft.Column(rows, spacing=4)


def _stat_row(label: str, value: str) -> ft.Control:
    return ft.Row(
        [
            ft.Text(label, size=13, color=COLORS["muted"], width=160),
            ft.Text(value, size=13, weight=ft.FontWeight.W_500, selectable=True),
        ],
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )


def _fmt_int(n: Optional[int]) -> str:
    return f"{n:,}" if isinstance(n, int) else "—"


def _fmt_mtime(mt: Optional[datetime]) -> str:
    return mt.strftime("%Y-%m-%d %H:%M") if mt else "—"


def _fmt_iso(iso: Optional[str]) -> str:
    if not iso:
        return "—"
    try:
        # ISO with timezone (e.g. ...+00:00) → drop timezone for display.
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError):
        return iso
