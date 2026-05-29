"""Вкладка «Создание датасета» — обёртка над make_dataset.MovieDatasetProcessor."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import flet as ft

from recommendation_system.models.gnn.gui.common import (
    _QueueLogHandler,
    _first_existing_ancestor,
    _open_in_explorer,
)
from recommendation_system.models.gnn.gui.theme import (
    COLORS,
    GAP_M,
    PROJECT_ROOT,
    neutral_button,
    primary_button,
    secondary_button,
)

if TYPE_CHECKING:
    from recommendation_system.models.gnn.gui.app import TrainerGuiApp


def _dataset_sanity(folder_name: str) -> tuple[bool, list[str]]:
    """Post-build validation. Reads parquet metadata + small column slices only —
    safe to run on the UI thread (< 1s for 20M-row interactions).

    folder_name = subfolder name under data/processed/ (movies, tv, или custom).
    """
    base = PROJECT_ROOT / "data" / "processed" / folder_name
    failures: list[str] = []
    inter = base / "interactions_final.parquet"
    items = base / "items_metadata_final.parquet"
    mapping = base / "id_mapping.json"

    for p in (inter, items, mapping):
        if not p.exists():
            failures.append(f"missing: {p.name}")
    if failures:
        return False, failures

    try:
        import pyarrow.parquet as pq
        n_inter = pq.ParquetFile(inter).metadata.num_rows
    except Exception as e:
        failures.append(f"interactions_final.parquet unreadable: {e}")
        return False, failures
    if n_inter < 100_000:
        failures.append(f"interactions: {n_inter:,} rows < 100k (smoke-run?)")

    try:
        import pandas as pd
        items_df = pd.read_parquet(items, columns=["tmdb_id", "title", "genres"])
    except Exception as e:
        failures.append(f"items_metadata_final.parquet schema mismatch: {e}")
        return False, failures

    required = {"tmdb_id", "title", "genres"}
    missing_cols = required - set(items_df.columns)
    if missing_cols:
        failures.append(f"items: missing columns {sorted(missing_cols)}")
    if items_df["tmdb_id"].duplicated().any():
        n_dup = int(items_df["tmdb_id"].duplicated().sum())
        failures.append(f"items: {n_dup} duplicate tmdb_ids")

    try:
        with open(mapping, encoding="utf-8") as f:
            m = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        failures.append(f"id_mapping.json unreadable: {e}")
        return False, failures
    num_users = m.get("num_users")
    num_items = m.get("num_items") or m.get("num_trained_items")
    if not isinstance(num_users, int) or num_users <= 0:
        failures.append("id_mapping: invalid num_users")
    if not isinstance(num_items, int) or num_items <= 0:
        failures.append("id_mapping: invalid num_items / num_trained_items")

    return (len(failures) == 0), failures


# Preset values for the 8 MovieDatasetProcessor params (spec 3.2 + 3.3).
# Keys MUST match MovieDatasetProcessor.__init__ param names.
_PARAM_ORDER = (
    "top_n_movies", "top_n_tv",
    "min_user_interactions", "min_item_interactions",
    "rating_threshold", "max_interactions",
    "min_year", "languages",
)

_PARAM_PRESETS: dict[str, dict[str, str]] = {
    "smoke": {
        "top_n_movies": "500", "top_n_tv": "200",
        "min_user_interactions": "5", "min_item_interactions": "5",
        "rating_threshold": "3.5", "max_interactions": "",
        "min_year": "2010", "languages": "en",
    },
    "default": {
        "top_n_movies": "15000", "top_n_tv": "10000",
        "min_user_interactions": "10", "min_item_interactions": "10",
        "rating_threshold": "3.5", "max_interactions": "15000000",
        "min_year": "", "languages": "en",
    },
    "full": {
        "top_n_movies": "", "top_n_tv": "",
        "min_user_interactions": "", "min_item_interactions": "",
        "rating_threshold": "", "max_interactions": "",
        "min_year": "", "languages": "en",
    },
}

_PARAM_TOOLTIPS: dict[str, str] = {
    "top_n_movies": "Сколько фильмов взять (по популярности). Пусто = без лимита.",
    "top_n_tv": "Сколько TV-шоу взять (по популярности). Пусто = без лимита.",
    "min_user_interactions": "K-core: минимум оценок у пользователя. Пусто = без фильтра.",
    "min_item_interactions": "K-core: минимум оценок у item. Пусто = без фильтра.",
    "rating_threshold": "Порог «нравится» (0-5). 3.5 = ≥3.5 считается положительным. Пусто = 0.",
    "max_interactions": "Лимит общего числа взаимодействий. Пусто = без лимита.",
    "min_year": "Минимальный год выпуска. Пусто = без фильтра.",
    "languages": "Языки оригинала через запятую (ISO 639-1): en / en,ru,uk. Пусто = en.",
}

# Sentinel for params whose backend signature is non-Optional (top_n_*, max_interactions, etc.).
# 10^9 effectively means «без лимита» — .head() / boolean filter no-op.
_HUGE_INT = 10 ** 9


# spec 3.3.b — Required/Optional + how to check existence for each source field.
# Key matches DatasetTab attribute name. `default_rel` is relative to PROJECT_ROOT/data.
# `check_file` (optional): for directory inputs, the representative file to test inside.
_SOURCE_META: dict[str, dict] = {
    "ml_dir_input": {
        "required": True,
        "default_rel": "raw/ml-32m",
        "check_file": "ratings.csv",
    },
    "tmdb_csv_input": {
        "required": False,
        "default_rel": "raw/TMDB_movie_dataset_v11.csv",
        "check_file": None,
    },
    "trakt_shows_input": {
        "required": True,
        "default_rel": "raw/trakt_shows.csv",
        "check_file": None,
    },
    "trakt_inter_input": {
        "required": True,
        "default_rel": "raw/trakt_interactions.csv",
        "check_file": None,
    },
    "extra_dataset_input": {
        "required": False,
        "default_rel": None,
        "check_file": "meta_Movies_and_TV.jsonl",
    },
}


class DatasetTab:
    """
    Wraps make_dataset.MovieDatasetProcessor.build_*_dataset() with a small UI:
    domain switcher (movies/tv/all), Amazon-dir field, indeterminate progress,
    post-build sanity validation. After success refreshes Tab 4 "Данные".
    """

    def __init__(self, app: "TrainerGuiApp") -> None:
        self.app = app
        self._thread: threading.Thread | None = None
        self._update_q: queue.Queue = queue.Queue()
        self._preset_silent: bool = False
        self._build()

    def build(self) -> ft.Control:
        return self._root

    def _build(self) -> None:
        self.domain_dd = ft.Dropdown(
            label="Домен",
            value="movies",
            options=[
                ft.dropdown.Option("movies", "Movies"),
                ft.dropdown.Option("tv", "TV"),
                ft.dropdown.Option("all", "All (последовательно)"),
            ],
            width=220,
            on_change=self._on_domain_change,
        )
        self.name_input = ft.TextField(
            label="Имя датасета",
            hint_text="Пусто → 'movies' / 'tv' (зависит от домена)",
            expand=True,
            dense=True,
        )

        raw_default = PROJECT_ROOT / "data" / "raw"

        # 4 path-override строки + 4 FilePicker'а в overlay.
        # on_change → пересчёт счётчика в title ExpansionTile «Источники».
        self.ml_dir_input = ft.TextField(
            label="MovieLens dir", hint_text=str(raw_default / "ml-32m"),
            expand=True, dense=True,
            on_change=self._on_source_change,
        )
        self.ml_dir_picker = ft.FilePicker(on_result=self._on_ml_dir_picked)
        self.ml_dir_btn = ft.IconButton(
            ft.Icons.FOLDER_OPEN, tooltip="Выбрать папку",
            on_click=lambda _e: self.ml_dir_picker.get_directory_path(
                dialog_title="MovieLens directory",
                initial_directory=str(self._picker_initial_dir(
                    self.ml_dir_input.value, raw_default / "ml-32m"
                )),
            ),
        )

        self.tmdb_csv_input = ft.TextField(
            label="TMDB CSV",
            hint_text=str(raw_default / "TMDB_movie_dataset_v11.csv"),
            expand=True, dense=True,
            on_change=self._on_source_change,
        )
        self.tmdb_csv_picker = ft.FilePicker(on_result=self._on_tmdb_csv_picked)
        self.tmdb_csv_btn = ft.IconButton(
            ft.Icons.UPLOAD_FILE, tooltip="Выбрать файл",
            on_click=lambda _e: self.tmdb_csv_picker.pick_files(
                allow_multiple=False, allowed_extensions=["csv"],
                dialog_title="TMDB metadata CSV",
                initial_directory=str(self._picker_initial_dir(
                    self.tmdb_csv_input.value, raw_default
                )),
            ),
        )

        self.trakt_shows_input = ft.TextField(
            label="Trakt shows CSV", hint_text=str(raw_default / "trakt_shows.csv"),
            expand=True, dense=True,
            on_change=self._on_source_change,
        )
        self.trakt_shows_picker = ft.FilePicker(on_result=self._on_trakt_shows_picked)
        self.trakt_shows_btn = ft.IconButton(
            ft.Icons.UPLOAD_FILE, tooltip="Выбрать файл",
            on_click=lambda _e: self.trakt_shows_picker.pick_files(
                allow_multiple=False, allowed_extensions=["csv"],
                dialog_title="Trakt shows CSV",
                initial_directory=str(self._picker_initial_dir(
                    self.trakt_shows_input.value, raw_default
                )),
            ),
        )

        self.trakt_inter_input = ft.TextField(
            label="Trakt interactions CSV",
            hint_text=str(raw_default / "trakt_interactions.csv"),
            expand=True, dense=True,
            on_change=self._on_source_change,
        )
        self.trakt_inter_picker = ft.FilePicker(on_result=self._on_trakt_inter_picked)
        self.trakt_inter_btn = ft.IconButton(
            ft.Icons.UPLOAD_FILE, tooltip="Выбрать файл",
            on_click=lambda _e: self.trakt_inter_picker.pick_files(
                allow_multiple=False, allowed_extensions=["csv"],
                initial_directory=str(self._picker_initial_dir(
                    self.trakt_inter_input.value, raw_default
                )),
                dialog_title="Trakt interactions CSV",
            ),
        )

        self.extra_dataset_input = ft.TextField(
            label="Amazon dir (опц.) — Optional Amazon Reviews",
            value="",
            hint_text="Пусто = Amazon не используется; путь к папке с meta_Movies_and_TV.jsonl",
            expand=True,
            dense=True,
            on_change=self._on_source_change,
        )
        self.amazon_picker = ft.FilePicker(on_result=self._on_amazon_picked)
        self.amazon_btn = ft.IconButton(
            ft.Icons.FOLDER_OPEN, tooltip="Выбрать папку Amazon",
            on_click=lambda _e: self.amazon_picker.get_directory_path(
                dialog_title="Amazon Reviews directory",
                initial_directory=str(self._picker_initial_dir(
                    self.extra_dataset_input.value, Path.home()
                )),
            ),
        )

        # FilePicker'ы должны жить в page.overlay.
        self.app.page.overlay.extend([
            self.ml_dir_picker, self.tmdb_csv_picker,
            self.trakt_shows_picker, self.trakt_inter_picker,
            self.amazon_picker,
        ])

        # spec 3.3.b — live ✓/✗ status icons (1 на каждое из 5 source-полей).
        self.source_status_icons: dict[str, ft.Icon] = {
            name: ft.Icon(
                ft.Icons.REMOVE_CIRCLE_OUTLINE,
                color=COLORS["muted"], size=20,
                tooltip="Статус источника",
            )
            for name in _SOURCE_META
        }

        # --- Параметры пайплайна (8 шт.) + Preset Dropdown — spec 3.2 + 3.3 ---
        self.preset_dd = ft.Dropdown(
            label="Preset",
            value="default",
            options=[
                ft.dropdown.Option("smoke", "Smoke (быстрый тест)"),
                ft.dropdown.Option("default", "Default (бывшие хардкоды)"),
                ft.dropdown.Option("full", "Full (без лимитов)"),
                ft.dropdown.Option("custom", "Custom (ручная правка)"),
            ],
            width=260,
            on_change=self._on_preset_change,
        )

        self.param_inputs: dict[str, ft.TextField] = {}
        for name in _PARAM_ORDER:
            self.param_inputs[name] = ft.TextField(
                label=name,
                tooltip=_PARAM_TOOLTIPS[name],
                value=_PARAM_PRESETS["default"][name],
                # Адаптивная сетка: 2 поля в ряд на узком, 3 на среднем, 4 на широком.
                col={"xs": 6, "sm": 4, "md": 3},
                dense=True,
                on_change=self._on_param_change,
            )

        self.build_btn = primary_button(
            "Собрать",
            icon=ft.Icons.BUILD,
            on_click=self._on_build,
        )
        self.stop_btn = neutral_button(
            "Стоп",
            icon=ft.Icons.STOP_CIRCLE,
            disabled=True,
            tooltip="Сборка не прерывается — make_dataset не поддерживает cancel-флаг.",
        )

        self.status_text = ft.Text("Готов к работе", size=14, color=COLORS["muted"])
        self.progress_bar = ft.ProgressBar(
            value=0, color=COLORS["primary"], bgcolor=ft.Colors.GREY_200, height=8
        )

        self.log_view = ft.ListView(expand=True, spacing=2, padding=8, auto_scroll=True)

        self.result_banner = ft.Container(
            visible=False, padding=12, border_radius=8,
        )

        self.open_folder_btn = secondary_button(
            "Открыть папку датасета",
            icon=ft.Icons.FOLDER_OPEN,
            on_click=self._on_open_folder,
            visible=False,
        )

        def _badge(required: bool) -> ft.Container:
            return ft.Container(
                content=ft.Text(
                    "Required" if required else "Optional",
                    size=10, weight=ft.FontWeight.BOLD,
                    color=ft.Colors.WHITE,
                ),
                bgcolor=COLORS["err"] if required else COLORS["muted"],
                padding=ft.padding.symmetric(horizontal=8, vertical=2),
                border_radius=10,
                width=70, alignment=ft.alignment.center,
            )

        def source_row(field_attr: str, text_field, button):
            """Row: [badge | text_field | (optional)button | status_icon]."""
            required = _SOURCE_META[field_attr]["required"]
            status = self.source_status_icons[field_attr]
            children = [_badge(required), text_field]
            if button is not None:
                children.append(button)
            children.append(status)
            return ft.Row(
                children, spacing=6,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )

        # Контейнеры с visible-toggle по домену.
        self.movies_sources_box = ft.Container(
            content=ft.Column(
                [
                    ft.Text("Источники Movies (пусто → дефолт data/raw/...)",
                            size=12, color=COLORS["muted"]),
                    source_row("ml_dir_input", self.ml_dir_input, self.ml_dir_btn),
                    source_row("tmdb_csv_input", self.tmdb_csv_input, self.tmdb_csv_btn),
                ],
                spacing=4,
            ),
        )
        self.tv_sources_box = ft.Container(
            content=ft.Column(
                [
                    ft.Text("Источники TV (пусто → дефолт data/raw/...)",
                            size=12, color=COLORS["muted"]),
                    source_row("trakt_shows_input", self.trakt_shows_input,
                               self.trakt_shows_btn),
                    source_row("trakt_inter_input", self.trakt_inter_input,
                               self.trakt_inter_btn),
                ],
                spacing=4,
            ),
        )
        self.amazon_box = ft.Container(
            content=ft.Column(
                [
                    ft.Text("Amazon (опц., используется и для movies, и для TV)",
                            size=12, color=COLORS["muted"]),
                    source_row("extra_dataset_input", self.extra_dataset_input, self.amazon_btn),
                ],
                spacing=4,
            ),
        )

        # 8 параметров — единая адаптивная сетка (col на каждом поле) + Preset сверху.
        self.params_box = ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Text("Параметры пайплайна",
                                    size=12, color=COLORS["muted"]),
                            self.preset_dd,
                        ],
                        spacing=GAP_M,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.ResponsiveRow(
                        [self.param_inputs[n] for n in _PARAM_ORDER],
                        spacing=GAP_M,
                        run_spacing=GAP_M,
                    ),
                ],
                spacing=GAP_M,
            ),
        )

        controls_row = ft.Row(
            [
                self.domain_dd,
                self.name_input,
                self.build_btn,
                self.stop_btn,
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=GAP_M,
        )

        result_row = ft.Row(
            [self.result_banner, self.open_folder_btn],
            spacing=8, wrap=True,
        )

        log_box = ft.Container(
            content=self.log_view,
            bgcolor=ft.Colors.GREY_50,
            border=ft.border.all(1, COLORS["card_border"]),
            border_radius=8,
            height=380,
        )

        # --- 2 ExpansionTile-обёртки (spec 3.3.a) ---
        self.sources_title = ft.Text(
            "Источники / Sources", weight=ft.FontWeight.W_500, size=14,
        )
        self.sources_tile = ft.ExpansionTile(
            title=self.sources_title,
            initially_expanded=False,
            controls=[
                self.movies_sources_box,
                self.tv_sources_box,
                self.amazon_box,
            ],
            collapsed_bgcolor=ft.Colors.GREY_50,
            bgcolor=ft.Colors.GREY_50,
        )

        self.params_title = ft.Text(
            "Параметры / Parameters", weight=ft.FontWeight.W_500, size=14,
        )
        self.params_tile = ft.ExpansionTile(
            title=self.params_title,
            initially_expanded=False,
            controls=[self.params_box],
            collapsed_bgcolor=ft.Colors.GREY_50,
            bgcolor=ft.Colors.GREY_50,
        )

        self._root = ft.Container(
            content=ft.Column(
                [
                    controls_row,
                    self.sources_tile,
                    self.params_tile,
                    ft.Row([self.progress_bar], expand=True),
                    self.status_text,
                    result_row,
                    ft.Text("Логи", size=14, weight=ft.FontWeight.W_500,
                            color=COLORS["muted"]),
                    log_box,
                ],
                spacing=12,
                expand=True,
                scroll=ft.ScrollMode.AUTO,
            ),
            padding=16,
        )

        # Применить начальную visibility по дефолтному домену + посчитать заголовки.
        self._apply_domain_visibility(self.domain_dd.value)
        self._update_titles()
        self._update_source_status()

    # ---- handlers ----

    def _on_domain_change(self, e: ft.ControlEvent) -> None:
        self._apply_domain_visibility(e.control.value)
        self._update_titles()
        self._update_source_status()
        self.app.page.update()

    def _apply_domain_visibility(self, domain: str) -> None:
        # movies/all → показать movies-источники; tv/all → tv. Amazon всегда.
        self.movies_sources_box.visible = domain in ("movies", "all")
        self.tv_sources_box.visible = domain in ("tv", "all")
        self.amazon_box.visible = True

    # ---- titles / counters (spec 3.3.a) ----

    def _on_source_change(self, _e: ft.ControlEvent) -> None:
        self._update_titles()
        self._update_source_status()
        self.app.page.update()

    # ---- live source ✓/✗ status (spec 3.3.b) ----

    def _resolve_source_check_target(self, field_attr: str) -> Optional[Path]:
        """Returns the Path to actually .exists()-check, or None if unresolvable.

        Empty field + no default → None (Amazon when blank).
        For dir-inputs (check_file set), returns dir/check_file representative.
        """
        meta = _SOURCE_META[field_attr]
        field: ft.TextField = getattr(self, field_attr)
        raw = (field.value or "").strip()
        if raw:
            base = Path(raw).expanduser()
        elif meta["default_rel"]:
            base = PROJECT_ROOT / "data" / meta["default_rel"]
        else:
            return None
        check_file = meta.get("check_file")
        return (base / check_file) if check_file else base

    def _update_source_status(self) -> None:
        """Recompute icon + color for all 5 source-fields."""
        for field_attr, meta in _SOURCE_META.items():
            required = meta["required"]
            icon: ft.Icon = self.source_status_icons[field_attr]
            target = self._resolve_source_check_target(field_attr)
            if target is None:
                # Empty field, no default (Amazon blank) → neutral.
                icon.name = ft.Icons.REMOVE_CIRCLE_OUTLINE
                icon.color = COLORS["muted"]
                icon.tooltip = "Не задано (optional)"
                continue
            if target.exists():
                icon.name = ft.Icons.CHECK_CIRCLE
                icon.color = COLORS["ok"]
                icon.tooltip = f"OK: {target}"
            elif required:
                icon.name = ft.Icons.CANCEL
                icon.color = COLORS["err"]
                icon.tooltip = f"Не найден (required): {target}"
            else:
                icon.name = ft.Icons.WARNING_AMBER
                icon.color = COLORS["warn"]
                icon.tooltip = f"Не найден (optional): {target}"

    def _count_source_overrides(self) -> int:
        domain = self.domain_dd.value
        fields: list[ft.TextField] = []
        if domain in ("movies", "all"):
            fields += [self.ml_dir_input, self.tmdb_csv_input]
        if domain in ("tv", "all"):
            fields += [self.trakt_shows_input, self.trakt_inter_input]
        fields.append(self.extra_dataset_input)
        return sum(1 for f in fields if (f.value or "").strip())

    def _count_param_overrides(self) -> int:
        defaults = _PARAM_PRESETS["default"]
        return sum(
            1 for name, f in self.param_inputs.items()
            if (f.value or "").strip() != defaults[name]
        )

    def _update_titles(self) -> None:
        n_src = self._count_source_overrides()
        n_par = self._count_param_overrides()
        src_suffix = f"  ({n_src} заполнено)" if n_src else ""
        par_suffix = f"  ({n_par} изменено)" if n_par else ""
        self.sources_title.value = f"Источники / Sources{src_suffix}"
        self.params_title.value = f"Параметры / Parameters{par_suffix}"

    # ---- preset / params handlers (spec 3.2 + 3.3) ----

    def _on_preset_change(self, e: ft.ControlEvent) -> None:
        preset = e.control.value
        if preset == "custom":
            # Ничего не меняем — пользователь сам редактирует.
            return
        values = _PARAM_PRESETS.get(preset)
        if values is None:
            return
        self._preset_silent = True
        try:
            for name, val in values.items():
                self.param_inputs[name].value = val
        finally:
            self._preset_silent = False
        self._update_titles()
        self.app.page.update()

    def _on_param_change(self, e: ft.ControlEvent) -> None:
        # При программном применении preset'а — не сваливаемся в custom,
        # но title всё равно пересчитываем.
        if self._preset_silent:
            return
        if self.preset_dd.value != "custom":
            self.preset_dd.value = "custom"
        self._update_titles()
        self.app.page.update()

    # ---- pre-flight check (spec 3.3.c) ----

    def _preflight_check(self, domain: str) -> list[tuple[str, Path, str]]:
        """Returns [(field_label, missing_path, help_text)] for Required sources.

        Movies: ratings.csv, movies.csv, links.csv (все три обязательны —
        make_dataset.py:312-327 читает без exists() проверки).
        TV: trakt_shows.csv + trakt_interactions.csv (make_dataset.py:505, 564).
        """
        missing: list[tuple[str, Path, str]] = []

        if domain in ("movies", "all"):
            raw = (self.ml_dir_input.value or "").strip()
            ml_root = Path(raw) if raw else (PROJECT_ROOT / "data" / "raw" / "ml-32m")
            ml_hint = (
                "Скачайте MovieLens-32M с grouplens.org и распакуйте "
                "в data/raw/ml-32m/ (подробнее: docs/data_sources.md)"
            )
            for fname in ("ratings.csv", "movies.csv", "links.csv"):
                p = ml_root / fname
                if not p.exists():
                    missing.append((f"MovieLens / {fname}", p, ml_hint))

        if domain in ("tv", "all"):
            trakt_specs = [
                ("trakt_shows_input", "Trakt shows CSV",
                 "Запустите trakt_collector (~2 суток) или подключите готовый "
                 "CSV (schema в docs/data_sources.md)"),
                ("trakt_inter_input", "Trakt interactions CSV",
                 "Запустите trakt_collector или подключите готовый CSV "
                 "(schema в docs/data_sources.md)"),
            ]
            for attr, label, hint in trakt_specs:
                p = self._resolve_source_check_target(attr)
                if p and not p.exists():
                    missing.append((label, p, hint))

        return missing

    def _show_missing_sources_banner(
        self, missing: list[tuple[str, Path, str]],
    ) -> None:
        lines: list[ft.Control] = [
            ft.Text(
                "❌ Не найдены обязательные источники:",
                color=COLORS["err"], weight=ft.FontWeight.BOLD, size=14,
            ),
        ]
        for label, path, hint in missing:
            lines.append(ft.Text(
                f"  • {label}: {path}",
                color=COLORS["err"], size=12,
            ))
            lines.append(ft.Text(
                f"    → {hint}", color=COLORS["muted"], size=11,
            ))

        docs_path = PROJECT_ROOT / "docs" / "data_sources.md"
        lines.append(
            ft.TextButton(
                "Как настроить → docs/data_sources.md",
                icon=ft.Icons.OPEN_IN_NEW,
                on_click=lambda _e: _open_in_explorer(docs_path),
            )
        )

        self.result_banner.bgcolor = COLORS["err_bg"]
        self.result_banner.content = ft.Column(lines, spacing=2, tight=True)
        self.result_banner.visible = True

    def _collect_params(self) -> dict:
        """Парсинг 8 полей в kwargs для MovieDatasetProcessor.

        Пустое поле → None для Optional-параметров (min_year),
        либо sentinel _HUGE_INT / 0 / 0.0 для non-Optional, чтобы пайплайн
        обрабатывал как «без лимита».
        """
        raw = {n: (f.value or "").strip() for n, f in self.param_inputs.items()}

        def _int_or(s: str, default: int) -> int:
            return int(s) if s else default

        def _float_or(s: str, default: float) -> float:
            return float(s) if s else default

        languages = [t.strip() for t in raw["languages"].split(",") if t.strip()]
        if not languages:
            languages = ["en"]

        return {
            "top_n_movies": _int_or(raw["top_n_movies"], _HUGE_INT),
            "top_n_tv": _int_or(raw["top_n_tv"], _HUGE_INT),
            "min_user_interactions": _int_or(raw["min_user_interactions"], 0),
            "min_item_interactions": _int_or(raw["min_item_interactions"], 0),
            "rating_threshold": _float_or(raw["rating_threshold"], 0.0),
            "max_interactions": _int_or(raw["max_interactions"], _HUGE_INT),
            "min_year": int(raw["min_year"]) if raw["min_year"] else None,
            "languages": languages,
        }

    def _on_ml_dir_picked(self, e: ft.FilePickerResultEvent) -> None:
        if e.path:
            self.ml_dir_input.value = e.path
            self._update_titles()
            self._update_source_status()
            self.app.page.update()

    def _on_tmdb_csv_picked(self, e: ft.FilePickerResultEvent) -> None:
        if e.files:
            self.tmdb_csv_input.value = e.files[0].path
            self._update_titles()
            self._update_source_status()
            self.app.page.update()

    def _on_trakt_shows_picked(self, e: ft.FilePickerResultEvent) -> None:
        if e.files:
            self.trakt_shows_input.value = e.files[0].path
            self._update_titles()
            self._update_source_status()
            self.app.page.update()

    def _on_trakt_inter_picked(self, e: ft.FilePickerResultEvent) -> None:
        if e.files:
            self.trakt_inter_input.value = e.files[0].path
            self._update_titles()
            self._update_source_status()
            self.app.page.update()

    def _on_amazon_picked(self, e: ft.FilePickerResultEvent) -> None:
        if e.path:
            self.extra_dataset_input.value = e.path
            self._update_titles()
            self._update_source_status()
            self.app.page.update()

    def _picker_initial_dir(self, current_value: str | None, default: Path) -> Path:
        # Если поле непустое и путь существует — стартуем от него (или родителя
        # если это файл/нет). Иначе walk-up от дефолта (data/raw/...).
        raw = (current_value or "").strip()
        candidate = Path(raw).expanduser() if raw else default
        return _first_existing_ancestor(candidate, PROJECT_ROOT / "data")

    def _on_build(self, e: ft.ControlEvent) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        domain = self.domain_dd.value

        # spec 3.3.c — pre-flight check Required sources; build не стартует если есть пропуски.
        missing = self._preflight_check(domain)
        if missing:
            self._show_missing_sources_banner(missing)
            self.app.page.update()
            return

        name = (self.name_input.value or "").strip() or None
        amazon_raw = (self.extra_dataset_input.value or "").strip()
        amazon_dir = Path(amazon_raw) if amazon_raw else None

        def _opt_path(field: ft.TextField) -> Optional[Path]:
            v = (field.value or "").strip()
            return Path(v) if v else None

        ml_dir = _opt_path(self.ml_dir_input)
        tmdb_csv = _opt_path(self.tmdb_csv_input)
        trakt_shows = _opt_path(self.trakt_shows_input)
        trakt_inter = _opt_path(self.trakt_inter_input)

        try:
            params = self._collect_params()
        except ValueError as exc:
            self._add_log(f"❌ Невалидный параметр: {exc}", COLORS["err"])
            self.app.page.update()
            return

        self.build_btn.disabled = True
        self.domain_dd.disabled = True
        self.extra_dataset_input.disabled = True
        self.name_input.disabled = True
        self.preset_dd.disabled = True
        for f in self.param_inputs.values():
            f.disabled = True
        self.progress_bar.value = None  # indeterminate
        self.status_text.value = "Запуск сборки..."
        self.result_banner.visible = False
        self.open_folder_btn.visible = False
        self.log_view.controls.clear()
        self._add_log(
            f"🛠 Старт сборки (domain={domain}, name={name or 'default'}, "
            f"amazon_dir={amazon_dir or '—'})",
            COLORS["primary"],
        )
        self._add_log(
            f"  preset={self.preset_dd.value}, "
            f"top_n=({params['top_n_movies']}/{params['top_n_tv']}), "
            f"min_user/item=({params['min_user_interactions']}/"
            f"{params['min_item_interactions']}), "
            f"rating≥{params['rating_threshold']}, "
            f"max_interactions={params['max_interactions']}, "
            f"min_year={params['min_year']}, languages={params['languages']}",
            COLORS["muted"],
        )
        if any((ml_dir, tmdb_csv, trakt_shows, trakt_inter)):
            self._add_log(
                f"  overrides: ml={ml_dir or '—'}, tmdb={tmdb_csv or '—'}, "
                f"trakt_shows={trakt_shows or '—'}, "
                f"trakt_inter={trakt_inter or '—'}",
                COLORS["muted"],
            )
        self.app.page.update()

        self._thread = threading.Thread(
            target=self._run_build,
            args=(domain, name, amazon_dir, ml_dir, tmdb_csv,
                  trakt_shows, trakt_inter, params),
            daemon=True,
        )
        self._thread.start()
        self.app.page.run_task(self._process_updates)

    def _on_open_folder(self, e: ft.ControlEvent) -> None:
        domain = self.domain_dd.value
        name = (self.name_input.value or "").strip()
        if name:
            target = PROJECT_ROOT / "data" / "processed" / name
        elif domain in ("movies", "tv"):
            target = PROJECT_ROOT / "data" / "processed" / domain
        else:
            target = PROJECT_ROOT / "data" / "processed"
        _open_in_explorer(target)

    # ---- worker thread ----

    def _run_build(
        self,
        domain: str,
        name: Optional[str],
        amazon_dir: Optional[Path],
        ml_dir: Optional[Path],
        tmdb_csv: Optional[Path],
        trakt_shows: Optional[Path],
        trakt_inter: Optional[Path],
        params: dict,
    ) -> None:
        gui_handler = _QueueLogHandler(self._update_q)
        gui_handler.setLevel(logging.INFO)
        gui_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                              datefmt="%H:%M:%S")
        )

        from recommendation_system.data import make_dataset as md
        md.logger.addHandler(gui_handler)
        try:
            config_path = (
                PROJECT_ROOT / "src" / "recommendation_system"
                / "models" / "gnn" / "config" / "genre_map.json"
            )
            data_dir = PROJECT_ROOT / "data"

            built: list[str] = []
            # Для domain='all' имя пользователя действует только если оба
            # домена пишутся в одну папку — что не имеет смысла. Поэтому при
            # 'all' игнорируем custom name и используем дефолтные movies/tv.
            single_name_ok = domain in ("movies", "tv")

            if domain in ("movies", "all"):
                target_subdir = name if (single_name_ok and name) else "movies"
                processor = md.MovieDatasetProcessor(
                    data_dir=data_dir,
                    config_path=str(config_path),
                    output_subdir=target_subdir,
                    ml_dir=ml_dir,
                    tmdb_csv=tmdb_csv,
                    **params,
                )
                self._update_q.put({"type": "stage",
                                    "text": f"Сборка movies → {target_subdir}..."})
                ok_m = processor.build_movie_dataset(amazon_dir=amazon_dir)
                if not ok_m:
                    self._update_q.put({"type": "done", "rc": 1,
                                        "built": built, "stage_fail": "movies"})
                    return
                built.append(target_subdir)

            if domain in ("tv", "all"):
                target_subdir = name if (single_name_ok and name) else "tv"
                processor = md.MovieDatasetProcessor(
                    data_dir=data_dir,
                    config_path=str(config_path),
                    output_subdir=target_subdir,
                    trakt_shows_csv=trakt_shows,
                    trakt_interactions_csv=trakt_inter,
                    **params,
                )
                self._update_q.put({"type": "stage",
                                    "text": f"Сборка tv → {target_subdir}..."})
                ok_t = processor.build_tv_dataset(amazon_dir=amazon_dir)
                if not ok_t:
                    self._update_q.put({"type": "done", "rc": 1,
                                        "built": built, "stage_fail": "tv"})
                    return
                built.append(target_subdir)

            self._update_q.put({"type": "done", "rc": 0, "built": built})
        except Exception:
            self._update_q.put({"type": "error", "tb": traceback.format_exc()})
        finally:
            md.logger.removeHandler(gui_handler)

    async def _process_updates(self) -> None:
        while self._thread is not None and self._thread.is_alive():
            self._drain_queue()
            await asyncio.sleep(0.1)
        self._drain_queue()
        self.build_btn.disabled = False
        self.domain_dd.disabled = False
        self.extra_dataset_input.disabled = False
        self.name_input.disabled = False
        self.preset_dd.disabled = False
        for f in self.param_inputs.values():
            f.disabled = False
        if self.progress_bar.value is None:
            self.progress_bar.value = 0
        self.app.page.update()

    def _drain_queue(self) -> None:
        drained = False
        try:
            while True:
                msg = self._update_q.get_nowait()
                self._handle_message(msg)
                drained = True
        except queue.Empty:
            pass
        if drained:
            self.app.page.update()

    def _handle_message(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "logger":
            self._add_log(msg["text"], msg.get("color") or ft.Colors.GREY_700)
        elif kind == "stage":
            self.status_text.value = msg["text"]
            self._add_log(f"▶ {msg['text']}", COLORS["primary"])
        elif kind == "done":
            self.progress_bar.value = 1.0
            built: list[str] = msg.get("built") or []
            stage_fail = msg.get("stage_fail")

            if msg["rc"] == 0 and built:
                # Sanity-check каждого собранного домена.
                all_ok = True
                fail_lines: list[str] = []
                for d in built:
                    ok, failures = _dataset_sanity(d)
                    if not ok:
                        all_ok = False
                        for f in failures:
                            fail_lines.append(f"[{d}] {f}")

                if all_ok:
                    self.result_banner.bgcolor = COLORS["ok_bg"]
                    self.result_banner.content = ft.Text(
                        f"✅ Собрано: {', '.join(built)}. Sanity-check пройден.",
                        color=COLORS["ok"], weight=ft.FontWeight.BOLD,
                    )
                    self.status_text.value = "Готово"
                    self._add_log(
                        f"✅ Сборка завершена. Sanity: OK для {', '.join(built)}",
                        COLORS["ok"],
                    )
                else:
                    self.result_banner.bgcolor = COLORS["err_bg"]
                    self.result_banner.content = ft.Text(
                        f"⚠ Собрано: {', '.join(built)}. Sanity-check ПРОВАЛЕН:\n"
                        + "\n".join(fail_lines),
                        color=COLORS["err"], weight=ft.FontWeight.BOLD,
                    )
                    self.status_text.value = "Sanity FAIL"
                    for line in fail_lines:
                        self._add_log(f"❌ {line}", COLORS["err"])
            else:
                where = f" на этапе {stage_fail}" if stage_fail else ""
                self.result_banner.bgcolor = COLORS["err_bg"]
                self.result_banner.content = ft.Text(
                    f"❌ Сборка завершилась ошибкой{where} (см. логи).",
                    color=COLORS["err"], weight=ft.FontWeight.BOLD,
                )
                self.status_text.value = "Ошибка"
                self._add_log(f"❌ Сборка прервана{where}", COLORS["err"])

            self.result_banner.visible = True
            self.open_folder_btn.visible = True

            # Обновить Tab 4 чтобы пользователь сразу увидел новые mtime / числа.
            try:
                self.app.data_tab.refresh()
            except Exception:
                pass
        elif kind == "error":
            self._add_log("❌ Exception:\n" + msg["tb"], COLORS["err"])
            self.result_banner.bgcolor = COLORS["err_bg"]
            self.result_banner.content = ft.Text(
                "❌ Необработанное исключение (см. логи)",
                color=COLORS["err"], weight=ft.FontWeight.BOLD,
            )
            self.result_banner.visible = True
            self.status_text.value = "Ошибка"

    def _add_log(self, text: str, color) -> None:
        self.log_view.controls.append(
            ft.Text(text, size=12, color=color, selectable=True, font_family="Consolas")
        )
        if len(self.log_view.controls) > 500:
            del self.log_view.controls[: len(self.log_view.controls) - 500]
