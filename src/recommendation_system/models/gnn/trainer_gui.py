"""
trainer_gui.py — Flet desktop GUI for the dual-LightGCN stack.

Four tabs share two global controls (Domain switcher, Device selector):

    1. Обучение           — wraps trainer.main() with hyperparameter UI (A2.2 — done)
    2. Создание датасета  — wraps make_dataset.build_*_dataset() (A2.4 — pending)
    3. Тестирование       — DualDomainEngine + UniversalSearchEngine (A2.5 — pending)
    4. Данные             — readonly per-domain statistics (A2.3 — pending)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import threading
import traceback
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import flet as ft
import torch

# TV_OFFSET — корректный TMDB-id (TV) хранится в каталоге как +10_000_000.
# Тот же константный сдвиг используется ботом, faiss_bridge и dataset pipeline.
_TV_OFFSET = 10_000_000


def _tmdb_url(item) -> str:
    """Бот-совместимая ссылка: IMDb если есть, иначе themoviedb.org с raw tmdb_id."""
    if getattr(item, "imdb_id", None):
        return f"https://www.imdb.com/title/{item.imdb_id}/"
    tid = int(item.tmdb_id)
    if item.media_type == "tv" and tid >= _TV_OFFSET:
        tid -= _TV_OFFSET
    return f"https://www.themoviedb.org/{item.media_type}/{tid}"

PROJECT_ROOT = Path(__file__).resolve().parents[4]

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


# ======================================================================
# Tab 1 — Training
# ======================================================================


class TrainingTab:
    """
    Wraps trainer.main() with a hyperparameter UI + live progress.
    Reads `domain` and `device` from the parent app on Start (no re-read mid-run).
    """

    def __init__(self, app: "TrainerGuiApp") -> None:
        self.app = app
        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self._update_q: queue.Queue = queue.Queue()
        self._sidecar_path: Path | None = None
        self._output_path: Path | None = None
        self._user_overrode_data_dir = False
        self._build()

    # ---- public ----

    def build(self) -> ft.Control:
        return self._root

    # ---- construction ----

    def _build(self) -> None:
        # Domain switcher теперь локальный для вкладки «Обучение»
        # (DatasetTab/InferenceTab имеют свои). По умолчанию — movies.
        self.domain_dd = ft.Dropdown(
            label="Домен",
            value="movies",
            options=[
                ft.dropdown.Option("movies", "Movies"),
                ft.dropdown.Option("tv", "TV"),
            ],
            width=160,
            on_change=self._on_domain_changed,
        )

        # data_dir picker — произвольная папка (TextField + FilePicker).
        # Дефолт идёт за domain switcher, пока пользователь не отредактировал поле.
        default_data_dir = PROJECT_ROOT / "data" / "processed" / self.domain_dd.value
        self.data_dir_input = ft.TextField(
            label="Папка с датасетом",
            value=str(default_data_dir),
            hint_text=str(PROJECT_ROOT / "data" / "processed" / self.domain_dd.value),
            width=420,
            dense=True,
            on_change=self._on_data_dir_edited,
        )
        self.data_dir_picker = ft.FilePicker(on_result=self._on_data_dir_picked)
        self.data_dir_btn = ft.IconButton(
            ft.Icons.FOLDER_OPEN,
            tooltip="Выбрать папку",
            on_click=lambda _e: self.data_dir_picker.get_directory_path(
                dialog_title="Папка датасета",
                initial_directory=str(self._data_dir_initial()),
            ),
        )

        # Custom output name — если пусто, отрабатывает auto-bump v{N+1}.
        self.model_name_input = ft.TextField(
            label="Имя модели (опц.)",
            hint_text="Пусто → auto-bump v{N+1}",
            width=240,
            text_size=14,
            dense=True,
        )

        self.epochs_input = _hp_field("Эпохи", "30")
        self.batch_size_input = _hp_field("Батч", "2048")
        self.lr_input = _hp_field("LR", "0.002")
        self.embedding_dim_input = _hp_field("Embedding", "32")
        self.num_layers_input = _hp_field("Слои GCN", "2")
        self.patience_input = _hp_field("Patience", "3")
        self.eval_every_input = _hp_field("Eval every", "5")

        self.start_btn = ft.ElevatedButton(
            "Старт",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._on_start,
            bgcolor=COLORS["primary"],
            color=ft.Colors.WHITE,
            height=44,
        )
        self.stop_btn = ft.ElevatedButton(
            "Стоп",
            icon=ft.Icons.STOP_CIRCLE,
            on_click=self._on_stop,
            bgcolor=COLORS["err"],
            color=ft.Colors.WHITE,
            disabled=True,
            height=44,
        )

        self.status_text = ft.Text("Готов к работе", size=14, color=COLORS["muted"])
        self.progress_bar = ft.ProgressBar(
            value=0, color=COLORS["primary"], bgcolor=ft.Colors.GREY_200, height=8
        )

        self.epoch_label = ft.Text("0 / 0", size=20, weight=ft.FontWeight.BOLD)
        self.best_epoch_label = ft.Text("—", size=20, weight=ft.FontWeight.BOLD)
        self.loss_label = ft.Text("—", size=20, weight=ft.FontWeight.BOLD)
        self.recall_label = ft.Text("—", size=20, weight=ft.FontWeight.BOLD)
        self.ndcg_label = ft.Text("—", size=20, weight=ft.FontWeight.BOLD)

        self._loss_series = ft.LineChartData(
            data_points=[],
            stroke_width=2,
            color=COLORS["primary"],
            stroke_cap_round=True,
        )
        self.loss_chart = ft.LineChart(
            data_series=[self._loss_series],
            border=ft.border.all(1, COLORS["card_border"]),
            horizontal_grid_lines=ft.ChartGridLines(color=ft.Colors.GREY_200, width=1),
            vertical_grid_lines=ft.ChartGridLines(color=ft.Colors.GREY_200, width=1),
            left_axis=ft.ChartAxis(labels_size=48, labels_interval=0.05),
            bottom_axis=ft.ChartAxis(labels_size=32, labels_interval=1),
            min_y=0,
            expand=True,
        )

        self.log_view = ft.ListView(expand=True, spacing=2, padding=8, auto_scroll=True)

        self.result_banner = ft.Container(
            visible=False,
            padding=12,
            border_radius=8,
        )
        self.sidecar_btn = ft.OutlinedButton(
            "Открыть sidecar.json",
            icon=ft.Icons.DESCRIPTION,
            on_click=self._on_open_sidecar,
            visible=False,
        )
        self.folder_btn = ft.OutlinedButton(
            "Открыть папку",
            icon=ft.Icons.FOLDER_OPEN,
            on_click=self._on_open_folder,
            visible=False,
        )

        data_dir_row = ft.Row(
            [self.domain_dd, self.data_dir_input, self.data_dir_btn],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.END,
        )
        output_row = ft.Row(
            [self.model_name_input],
            spacing=4,
            vertical_alignment=ft.CrossAxisAlignment.END,
        )
        dataset_row = ft.Column([data_dir_row, output_row], spacing=8)

        hp_row = ft.Row(
            [
                self.epochs_input, self.batch_size_input, self.lr_input,
                self.embedding_dim_input, self.num_layers_input,
                self.patience_input, self.eval_every_input,
            ],
            wrap=True, spacing=8,
        )

        controls_row = ft.Row(
            [self.start_btn, self.stop_btn, ft.Container(expand=True), self.status_text],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=12,
        )

        metrics_row = ft.Row(
            [
                _metric_card("Эпоха", self.epoch_label),
                _metric_card("Лучшая эпоха", self.best_epoch_label),
                _metric_card("Loss", self.loss_label),
                _metric_card("Recall@10", self.recall_label),
                _metric_card("NDCG@10", self.ndcg_label),
            ],
            spacing=8,
        )

        chart_box = ft.Container(
            content=ft.Column(
                [
                    ft.Container(
                        content=ft.Text("Loss", size=12, color=COLORS["muted"],
                                        weight=ft.FontWeight.W_500),
                        padding=ft.padding.only(left=8),
                    ),
                    ft.Container(content=self.loss_chart, expand=True),
                    ft.Container(
                        content=ft.Text("Эпоха", size=12, color=COLORS["muted"],
                                        weight=ft.FontWeight.W_500),
                        alignment=ft.alignment.center,
                    ),
                ],
                spacing=4,
                expand=True,
            ),
            height=280,
            padding=8,
        )

        result_row = ft.Row(
            [self.result_banner, self.sidecar_btn, self.folder_btn],
            spacing=8,
            wrap=True,
        )

        log_box = ft.Container(
            content=self.log_view,
            bgcolor=ft.Colors.GREY_50,
            border=ft.border.all(1, COLORS["card_border"]),
            border_radius=8,
            height=300,
        )

        self._root = ft.Container(
            content=ft.Column(
                [
                    ft.Text("Источник и выход", size=14, weight=ft.FontWeight.W_500,
                            color=COLORS["muted"]),
                    dataset_row,
                    ft.Text("Гиперпараметры", size=14, weight=ft.FontWeight.W_500,
                            color=COLORS["muted"]),
                    hp_row,
                    controls_row,
                    self.progress_bar,
                    metrics_row,
                    chart_box,
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

        # FilePicker'ы должны жить в page.overlay.
        self.app.page.overlay.append(self.data_dir_picker)

    # ---- handlers ----

    def _on_start(self, e: ft.ControlEvent) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        try:
            epochs = int(self.epochs_input.value)
            batch_size = int(self.batch_size_input.value)
            lr = float(self.lr_input.value)
            embedding_dim = int(self.embedding_dim_input.value)
            num_layers = int(self.num_layers_input.value)
            patience = int(self.patience_input.value)
            eval_every = int(self.eval_every_input.value)
        except ValueError as exc:
            self._add_log(f"Невалидные параметры: {exc}", COLORS["err"])
            self.app.page.update()
            return

        domain = self.domain_dd.value
        device = self.app.current_device()
        data_dir_raw = (self.data_dir_input.value or "").strip()
        if data_dir_raw:
            data_dir = Path(data_dir_raw).expanduser().resolve()
        else:
            data_dir = PROJECT_ROOT / "data" / "processed" / domain
        if not data_dir.exists():
            self._add_log(f"data_dir не существует: {data_dir}", COLORS["err"])
            self.app.page.update()
            return

        argv = [
            "--domain", domain,
            "--data-dir", str(data_dir),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--lr", str(lr),
            "--embedding-dim", str(embedding_dim),
            "--num-layers", str(num_layers),
            "--patience", str(patience),
            "--eval-every", str(eval_every),
            "--device", device,
        ]

        custom_name = (self.model_name_input.value or "").strip()
        if custom_name:
            if not custom_name.endswith(".pt"):
                custom_name += ".pt"
            output_path = PROJECT_ROOT / "models" / domain / custom_name
            argv += ["--output", str(output_path)]

        self._stop_requested = False
        self.start_btn.disabled = True
        self.stop_btn.disabled = False
        self.status_text.value = "Подготовка..."
        self.progress_bar.value = None  # indeterminate until first epoch reports
        self.result_banner.visible = False
        self.sidecar_btn.visible = False
        self.folder_btn.visible = False
        self._sidecar_path = None
        self._output_path = None

        self._loss_series.data_points = []
        self.loss_chart.max_x = float(max(epochs, 1))
        self.loss_chart.min_x = 0

        self.epoch_label.value = f"0 / {epochs}"
        self.best_epoch_label.value = "—"
        self.loss_label.value = "—"
        self.recall_label.value = "—"
        self.ndcg_label.value = "—"
        self.log_view.controls.clear()
        self._add_log(
            f"🚀 Старт обучения (domain={domain}, device={device}, epochs={epochs})",
            COLORS["primary"],
        )
        self.app.page.update()

        self._thread = threading.Thread(
            target=self._run_training, args=(argv,), daemon=True
        )
        self._thread.start()
        self.app.page.run_task(self._process_updates)

    def _on_stop(self, e: ft.ControlEvent) -> None:
        self._stop_requested = True
        self.status_text.value = "Останавливаем..."
        self._add_log("⏹ Запрос остановки...", COLORS["warn"])
        self.app.page.update()

    def _on_open_sidecar(self, e: ft.ControlEvent) -> None:
        if self._sidecar_path and self._sidecar_path.exists():
            _open_in_explorer(self._sidecar_path)

    def _on_open_folder(self, e: ft.ControlEvent) -> None:
        if self._output_path:
            _open_in_explorer(self._output_path.parent)

    # ---- data_dir picker ----

    def _data_dir_initial(self) -> Path:
        current = (self.data_dir_input.value or "").strip()
        candidate = (
            Path(current).expanduser() if current
            else PROJECT_ROOT / "data" / "processed" / self.domain_dd.value
        )
        return _first_existing_ancestor(candidate, PROJECT_ROOT / "data")

    def _on_data_dir_picked(self, e: ft.FilePickerResultEvent) -> None:
        if not e.path:
            return
        self.data_dir_input.value = e.path
        self._user_overrode_data_dir = True
        self.app.page.update()

    def _on_data_dir_edited(self, e: ft.ControlEvent) -> None:
        # Любая ручная правка → отключаем auto-sync с domain switcher.
        self._user_overrode_data_dir = True

    def _on_domain_changed(self, e: ft.ControlEvent) -> None:
        """Local handler: пересинхронизирует data_dir под выбранный домен."""
        if self._user_overrode_data_dir:
            return
        domain = self.domain_dd.value
        self.data_dir_input.value = str(
            PROJECT_ROOT / "data" / "processed" / domain
        )
        self.data_dir_input.hint_text = str(
            PROJECT_ROOT / "data" / "processed" / domain
        )
        self.app.page.update()

    # ---- worker thread ----

    def _run_training(self, argv: list) -> None:
        gui_handler = _QueueLogHandler(self._update_q)
        gui_handler.setLevel(logging.INFO)
        gui_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                              datefmt="%H:%M:%S")
        )

        from recommendation_system.models.gnn import trainer as trainer_mod
        trainer_mod.logger.addHandler(gui_handler)
        try:
            rc = trainer_mod.main(
                argv,
                on_epoch_end=lambda ev: self._update_q.put({"type": "epoch", "data": ev}),
                stop_flag=lambda: self._stop_requested,
            )
            self._update_q.put({"type": "done", "rc": int(rc) if rc is not None else 0})
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            self._update_q.put({"type": "done", "rc": code})
        except Exception:
            self._update_q.put({"type": "error", "tb": traceback.format_exc()})
        finally:
            trainer_mod.logger.removeHandler(gui_handler)

    async def _process_updates(self) -> None:
        while self._thread is not None and self._thread.is_alive():
            self._drain_queue()
            await asyncio.sleep(0.1)
        self._drain_queue()
        self.start_btn.disabled = False
        self.stop_btn.disabled = True
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
            self._capture_paths(msg["text"])
        elif kind == "epoch":
            d = msg["data"]
            ep, total = int(d["epoch"]), int(d["total_epochs"])
            self.epoch_label.value = f"{ep} / {total}"
            self.loss_label.value = f"{d['loss']:.4f}"
            if d.get("recall") is not None:
                self.recall_label.value = f"{d['recall']:.4f}"
            if d.get("ndcg") is not None:
                self.ndcg_label.value = f"{d['ndcg']:.4f}"
            if d.get("best_epoch"):
                self.best_epoch_label.value = str(d["best_epoch"])
            self._loss_series.data_points.append(
                ft.LineChartDataPoint(float(ep), float(d["loss"]))
            )
            self.progress_bar.value = ep / max(total, 1)
            self.status_text.value = f"Эпоха {ep}/{total}"
        elif kind == "done":
            rc = msg["rc"]
            self.progress_bar.value = 1.0
            if rc == 0:
                self.result_banner.bgcolor = COLORS["ok_bg"]
                self.result_banner.content = ft.Text(
                    "✅ Обучение завершено, sanity-check пройден",
                    color=COLORS["ok"], weight=ft.FontWeight.BOLD,
                )
                self.status_text.value = "Готово"
            elif rc == 1:
                self.result_banner.bgcolor = COLORS["err_bg"]
                self.result_banner.content = ft.Text(
                    "⚠ Обучение завершено, sanity-check ПРОВАЛЕН (см. sidecar)",
                    color=COLORS["err"], weight=ft.FontWeight.BOLD,
                )
                self.status_text.value = "Sanity FAIL"
            else:
                self.result_banner.bgcolor = COLORS["err_bg"]
                self.result_banner.content = ft.Text(
                    f"❌ Ошибка обучения (rc={rc})",
                    color=COLORS["err"], weight=ft.FontWeight.BOLD,
                )
                self.status_text.value = "Ошибка"
            self.result_banner.visible = True
            if self._sidecar_path is not None:
                self.sidecar_btn.visible = True
            if self._output_path is not None:
                self.folder_btn.visible = True
        elif kind == "error":
            self._add_log("❌ Exception:\n" + msg["tb"], COLORS["err"])
            self.result_banner.bgcolor = COLORS["err_bg"]
            self.result_banner.content = ft.Text(
                "❌ Необработанное исключение (см. логи)",
                color=COLORS["err"], weight=ft.FontWeight.BOLD,
            )
            self.result_banner.visible = True
            self.status_text.value = "Ошибка"

    def _capture_paths(self, text: str) -> None:
        # Trainer logs include "Sidecar:    <path>" and "Checkpoint: <path>" on success
        # and "Output: <path>" on the config dump.
        for line in text.splitlines():
            stripped = line.strip()
            if "Sidecar:" in stripped:
                p = stripped.split("Sidecar:", 1)[1].strip()
                if p.endswith(".json"):
                    self._sidecar_path = Path(p)
            elif "Checkpoint:" in stripped:
                p = stripped.split("Checkpoint:", 1)[1].strip()
                if p.endswith(".pt"):
                    self._output_path = Path(p)
            elif stripped.startswith("Output:"):
                p = stripped.split("Output:", 1)[1].strip()
                if p.endswith(".pt"):
                    self._output_path = Path(p)

    def _add_log(self, text: str, color) -> None:
        self.log_view.controls.append(
            ft.Text(text, size=12, color=color, selectable=True, font_family="Consolas")
        )
        if len(self.log_view.controls) > 500:
            del self.log_view.controls[: len(self.log_view.controls) - 500]


class _QueueLogHandler(logging.Handler):
    """Forwards stdlib log records into the GUI's update queue."""

    def __init__(self, q: queue.Queue) -> None:
        super().__init__()
        self._q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record)
        except Exception:
            return
        if record.levelno >= logging.ERROR:
            color = COLORS["err"]
        elif record.levelno >= logging.WARNING:
            color = COLORS["warn"]
        else:
            color = None
        self._q.put({"type": "logger", "text": text, "color": color})


# ======================================================================
# Tab 2 — Dataset creation
# ======================================================================


def _first_existing_ancestor(path: Path, fallback: Path) -> Path:
    """Walk up from `path` until existing dir, else fallback. Flet иногда игнорирует
    initial_directory если путь не существует и открывает picker в неожиданном месте."""
    p = path
    while p != p.parent and not p.is_dir():
        p = p.parent
    return p if p.is_dir() else fallback


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
    "amazon_field": {
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
            width=260,
            dense=True,
        )

        raw_default = PROJECT_ROOT / "data" / "raw"

        # 4 path-override строки + 4 FilePicker'а в overlay.
        # on_change → пересчёт счётчика в title ExpansionTile «Источники».
        self.ml_dir_input = ft.TextField(
            label="MovieLens dir", hint_text=str(raw_default / "ml-32m"),
            width=380, dense=True,
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
            width=380, dense=True,
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
            width=380, dense=True,
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
            width=380, dense=True,
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

        self.amazon_field = ft.TextField(
            label="Amazon dir (опц.) — Optional Amazon Reviews",
            value="",
            hint_text="Пусто = Amazon не используется; путь к папке с meta_Movies_and_TV.jsonl",
            width=400,
            dense=True,
            on_change=self._on_source_change,
        )
        self.amazon_picker = ft.FilePicker(on_result=self._on_amazon_picked)
        self.amazon_btn = ft.IconButton(
            ft.Icons.FOLDER_OPEN, tooltip="Выбрать папку Amazon",
            on_click=lambda _e: self.amazon_picker.get_directory_path(
                dialog_title="Amazon Reviews directory",
                initial_directory=str(self._picker_initial_dir(
                    self.amazon_field.value, Path.home()
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
                width=180,
                dense=True,
                on_change=self._on_param_change,
            )

        self.build_btn = ft.ElevatedButton(
            "Собрать",
            icon=ft.Icons.BUILD,
            on_click=self._on_build,
            bgcolor=COLORS["primary"],
            color=ft.Colors.WHITE,
            height=44,
        )
        self.stop_btn = ft.ElevatedButton(
            "Стоп",
            icon=ft.Icons.STOP_CIRCLE,
            bgcolor=ft.Colors.GREY_400,
            color=ft.Colors.WHITE,
            disabled=True,
            height=44,
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

        self.open_folder_btn = ft.OutlinedButton(
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
                    source_row("amazon_field", self.amazon_field, self.amazon_btn),
                ],
                spacing=4,
            ),
        )

        # 8 параметров в 4 ряда по 2 + Preset Dropdown сверху (spec 3.2 + 3.3).
        # ExpansionTile-обёртка появится в Шаге 3 (reorganization).
        def _param_row(*names: str) -> ft.Row:
            return ft.Row(
                [self.param_inputs[n] for n in names], spacing=12,
                vertical_alignment=ft.CrossAxisAlignment.START,
            )

        self.params_box = ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Text("Параметры пайплайна",
                                    size=12, color=COLORS["muted"]),
                            self.preset_dd,
                        ],
                        spacing=12,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    _param_row("top_n_movies", "top_n_tv"),
                    _param_row("min_user_interactions", "min_item_interactions"),
                    _param_row("rating_threshold", "max_interactions"),
                    _param_row("min_year", "languages"),
                ],
                spacing=6,
            ),
        )

        controls_row = ft.Row(
            [
                self.domain_dd,
                self.name_input,
                ft.Container(expand=True),
                self.build_btn,
                self.stop_btn,
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=12,
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
        fields.append(self.amazon_field)
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
            self.amazon_field.value = e.path
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
        amazon_raw = (self.amazon_field.value or "").strip()
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
        self.amazon_field.disabled = True
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
        self.amazon_field.disabled = False
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


# ======================================================================
# Tab 4 — Data (readonly per-domain stats)
# ======================================================================


@dataclass
class DomainStats:
    domain: str
    dataset_dir: Path
    models_dir: Path
    dataset_exists: bool
    interactions_mtime: Optional[datetime] = None
    items_mtime: Optional[datetime] = None
    num_users: Optional[int] = None
    num_items: Optional[int] = None
    num_interactions: Optional[int] = None
    last_train_at: Optional[str] = None
    last_train_recall: Optional[float] = None
    last_train_checkpoint: Optional[str] = None
    last_train_sanity_ok: Optional[bool] = None
    model_mtime: Optional[datetime] = None
    model_size_mb: Optional[float] = None


def _collect_dataset_stats(dataset_dir: Path) -> Optional[dict]:
    """Read interactions/items/id_mapping from any folder.
    Returns None if any of the three required files is missing."""
    interactions_path = dataset_dir / "interactions_final.parquet"
    items_path = dataset_dir / "items_metadata_final.parquet"
    mapping_path = dataset_dir / "id_mapping.json"

    if not (interactions_path.exists() and items_path.exists() and mapping_path.exists()):
        return None

    interactions_mtime = datetime.fromtimestamp(interactions_path.stat().st_mtime)
    items_mtime = datetime.fromtimestamp(items_path.stat().st_mtime)

    num_users = None
    num_items = None
    try:
        with open(mapping_path, encoding="utf-8") as f:
            mapping = json.load(f)
        num_users = mapping.get("num_users")
        # Prefer total catalog count; fall back to trained subset.
        num_items = mapping.get("num_items") or mapping.get("num_trained_items")
    except (json.JSONDecodeError, OSError):
        pass

    num_interactions = None
    try:
        import pyarrow.parquet as pq
        num_interactions = pq.ParquetFile(interactions_path).metadata.num_rows
    except Exception:
        try:
            import pandas as pd
            num_interactions = len(pd.read_parquet(interactions_path, columns=["user_id"]))
        except Exception:
            pass

    return {
        "interactions_mtime": interactions_mtime,
        "items_mtime": items_mtime,
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions": num_interactions,
    }


def _collect_model_stats(model_path: Path) -> dict:
    """Read sidecar .json next to a .pt checkpoint. If sidecar is missing,
    return only filename + mtime + size_mb. Always populates last_train_checkpoint."""
    result: dict = {
        "last_train_checkpoint": model_path.name,
        "model_mtime": datetime.fromtimestamp(model_path.stat().st_mtime),
        "model_size_mb": model_path.stat().st_size / (1024 * 1024),
    }

    sidecar = model_path.with_suffix(".json")
    if not sidecar.exists():
        return result

    try:
        with open(sidecar, encoding="utf-8") as f:
            s = json.load(f)
    except (json.JSONDecodeError, OSError):
        return result

    result["last_train_at"] = s.get("trained_at")
    metrics = s.get("metrics") or {}
    recall = metrics.get("recall@10")
    if isinstance(recall, (int, float)):
        result["last_train_recall"] = float(recall)
    # Sidecar's checkpoint name wins if present (handles renamed .pt files).
    sidecar_ckpt = s.get("checkpoint")
    if sidecar_ckpt:
        result["last_train_checkpoint"] = sidecar_ckpt
    sanity = s.get("sanity_check") or {}
    sanity_passed = sanity.get("passed")
    if isinstance(sanity_passed, bool):
        result["last_train_sanity_ok"] = sanity_passed
    return result


def _find_latest_model(models_dir: Path, domain: str) -> Optional[Path]:
    """Find the most recent .pt in models_dir. If a sidecar exists and its
    domain mismatches, skip; .pt files without a sidecar are still considered
    (covers old Colab v4 checkpoints)."""
    if not models_dir.exists():
        return None

    candidates: list[Path] = []
    for pt in models_dir.glob("*.pt"):
        sidecar = pt.with_suffix(".json")
        if sidecar.exists():
            try:
                with open(sidecar, encoding="utf-8") as f:
                    s = json.load(f)
                if s.get("domain") and s.get("domain") != domain:
                    continue
            except (json.JSONDecodeError, OSError):
                pass
        candidates.append(pt)

    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _collect_domain_stats(
    domain: str,
    dataset_override: Optional[Path] = None,
    model_override: Optional[Path] = None,
) -> DomainStats:
    dataset_dir = dataset_override or (PROJECT_ROOT / "data" / "processed" / domain)
    models_dir = PROJECT_ROOT / "models" / domain

    ds = _collect_dataset_stats(dataset_dir)

    if model_override and model_override.exists():
        ms = _collect_model_stats(model_override)
    else:
        latest = _find_latest_model(models_dir, domain)
        ms = _collect_model_stats(latest) if latest else {}

    return DomainStats(
        domain=domain,
        dataset_dir=dataset_dir,
        models_dir=models_dir,
        dataset_exists=ds is not None,
        interactions_mtime=ds.get("interactions_mtime") if ds else None,
        items_mtime=ds.get("items_mtime") if ds else None,
        num_users=ds.get("num_users") if ds else None,
        num_items=ds.get("num_items") if ds else None,
        num_interactions=ds.get("num_interactions") if ds else None,
        last_train_at=ms.get("last_train_at"),
        last_train_recall=ms.get("last_train_recall"),
        last_train_checkpoint=ms.get("last_train_checkpoint"),
        last_train_sanity_ok=ms.get("last_train_sanity_ok"),
        model_mtime=ms.get("model_mtime"),
        model_size_mb=ms.get("model_size_mb"),
    )


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
                        ft.OutlinedButton(
                            "Папка датасета",
                            icon=ft.Icons.FOLDER_OPEN,
                            on_click=lambda _e: _open_in_explorer(stats.dataset_dir),
                        ),
                        ft.OutlinedButton(
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


# ======================================================================
# Helpers
# ======================================================================


def _hp_field(label: str, default: str) -> ft.TextField:
    return ft.TextField(label=label, value=default, width=120, text_size=14, dense=True)


def _metric_card(title: str, value_control: ft.Control) -> ft.Control:
    return ft.Container(
        content=ft.Column(
            [
                ft.Text(title, size=11, color=COLORS["muted"]),
                value_control,
            ],
            spacing=2,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        padding=12,
        border_radius=8,
        bgcolor=ft.Colors.WHITE,
        border=ft.border.all(1, COLORS["card_border"]),
        expand=True,
    )


def _placeholder(title: str, description: str) -> ft.Control:
    return ft.Container(
        content=ft.Column(
            [
                ft.Text(title, size=18, weight=ft.FontWeight.BOLD, color=COLORS["primary"]),
                ft.Text(description, size=14, color=COLORS["muted"]),
                ft.Text(
                    "(будет реализовано на следующих этапах)",
                    size=12, italic=True, color=COLORS["muted"],
                ),
            ],
            spacing=8,
        ),
        padding=24,
        bgcolor=COLORS["primary_bg"],
        border_radius=8,
        margin=ft.margin.symmetric(vertical=12),
    )


def _open_in_explorer(path: Path) -> None:
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif os.name == "posix":
            import subprocess
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


# ======================================================================
# Tab 3 — Inference (DualDomainEngine offline tester)
# ======================================================================


_FAISS_INDEX = PROJECT_ROOT / "src" / "recommendation_system" / "faiss_index" / "catalog.faiss"
_FAISS_META = PROJECT_ROOT / "src" / "recommendation_system" / "faiss_index" / "catalog_meta.json"
_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "cache"


def _display_title(item, lang: str) -> str:
    """Pick localized title field; fall back to default `title` if missing."""
    if lang == "ru":
        return getattr(item, "title_ru", None) or item.title
    if lang == "uk":
        return getattr(item, "title_uk", None) or item.title
    return item.title


def _discover_checkpoints(domain: str) -> list[str]:
    """List `.pt` files in models/{domain}/ sorted by mtime descending."""
    models_dir = PROJECT_ROOT / "models" / domain
    if not models_dir.exists():
        return []
    files = list(models_dir.glob("*.pt"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in files]


def _discover_datasets() -> list[str]:
    """List data/processed/* subfolders that contain a full built dataset
    (parquet pair + id_mapping.json). Sorted alphabetically for stability."""
    base = PROJECT_ROOT / "data" / "processed"
    if not base.exists():
        return []
    out: list[str] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if (
            (child / "interactions_final.parquet").exists()
            and (child / "items_metadata_final.parquet").exists()
            and (child / "id_mapping.json").exists()
        ):
            out.append(child.name)
    return out


class InferenceTab:
    """
    Offline 'bot in a window' for testing the DualDomainEngine router with
    domain separation. Search → ★ Favorites → 4 recs buttons (movie/tv/all/cross).

    Engines load lazily on the first recs click (30-60s) to avoid blocking
    GUI startup; until then the tab is fully usable for browsing/wiring.
    """

    def __init__(self, app: "TrainerGuiApp") -> None:
        self.app = app
        self.router = None  # DualDomainEngine | None — lazy
        self.movies_engine = None  # UniversalSearchEngine | None
        self.tv_engine = None  # UniversalSearchEngine | None
        # None = «latest» (sidecar fallback chain). Иначе — абсолютный путь до .pt
        # (выбран через FilePicker; A2.5-rev).
        self.movies_checkpoint_override: Optional[Path] = None
        self.tv_checkpoint_override: Optional[Path] = None
        self.favorites: list[tuple[int, str, str]] = []  # (tmdb_id, title, media_type)
        self.last_results: list = []  # last list[UniversalMediaItem] for relang re-render
        self._thread: threading.Thread | None = None
        self._update_q: queue.Queue = queue.Queue()
        self._faiss_available = _FAISS_INDEX.exists() and _FAISS_META.exists()
        self._build()

    def build(self) -> ft.Control:
        return self._root

    # ---- UI ----

    def _build(self) -> None:
        self.scope_dd = ft.Dropdown(
            label="Каталог для поиска",
            value="movies",
            options=[
                ft.dropdown.Option("movies", "Movies"),
                ft.dropdown.Option("tv", "TV"),
            ],
            width=180,
            on_change=lambda _e: None,
        )
        self.lang_dd = ft.Dropdown(
            label="Язык названий",
            value="en",
            options=[
                ft.dropdown.Option("en", "English"),
                ft.dropdown.Option("ru", "Русский"),
                ft.dropdown.Option("uk", "Українська"),
            ],
            width=160,
            on_change=self._on_lang_change,
        )
        self.cross_target_dd = ft.Dropdown(
            label="Cross target",
            value="tv",
            options=[
                ft.dropdown.Option("movie", "→ Movies"),
                ft.dropdown.Option("tv", "→ TV"),
            ],
            width=140,
        )

        # Model pickers per domain (A2.5-rev). UX:
        #   Switch ON  = latest (sidecar chain → fallback v4.pt), TextField пуст/disabled.
        #   Switch OFF = override активен, TextField показывает абсолютный путь к .pt.
        # Клик по FilePicker → Switch автоматически OFF + override = выбранный путь.
        self.movies_latest_switch = ft.Switch(
            value=True, tooltip="Использовать latest (sidecar)",
            on_change=lambda e: self._on_latest_switch("movies", e),
        )
        self.movies_ckpt_input = ft.TextField(
            label="Movies model (.pt)",
            value=f"latest → {self._resolve_latest_name('movies')}",
            hint_text="latest (sidecar)",
            width=380, dense=True, disabled=True,
        )
        self.movies_ckpt_picker = ft.FilePicker(
            on_result=lambda e: self._on_ckpt_picked("movies", e),
        )
        self.movies_ckpt_btn = ft.IconButton(
            ft.Icons.UPLOAD_FILE, tooltip="Выбрать .pt",
            on_click=lambda _e: self.movies_ckpt_picker.pick_files(
                allow_multiple=False, allowed_extensions=["pt"],
                dialog_title="Movies checkpoint (.pt)",
                initial_directory=str(_first_existing_ancestor(
                    PROJECT_ROOT / "models" / "movies", PROJECT_ROOT / "models",
                )),
            ),
        )

        self.tv_latest_switch = ft.Switch(
            value=True, tooltip="Использовать latest (sidecar)",
            on_change=lambda e: self._on_latest_switch("tv", e),
        )
        self.tv_ckpt_input = ft.TextField(
            label="TV model (.pt)",
            value=f"latest → {self._resolve_latest_name('tv')}",
            hint_text="latest (sidecar)",
            width=380, dense=True, disabled=True,
        )
        self.tv_ckpt_picker = ft.FilePicker(
            on_result=lambda e: self._on_ckpt_picked("tv", e),
        )
        self.tv_ckpt_btn = ft.IconButton(
            ft.Icons.UPLOAD_FILE, tooltip="Выбрать .pt",
            on_click=lambda _e: self.tv_ckpt_picker.pick_files(
                allow_multiple=False, allowed_extensions=["pt"],
                dialog_title="TV checkpoint (.pt)",
                initial_directory=str(_first_existing_ancestor(
                    PROJECT_ROOT / "models" / "tv", PROJECT_ROOT / "models",
                )),
            ),
        )

        self.search_field = ft.TextField(
            label="Поиск (bilingual RU/UK/EN)",
            hint_text="например: Inception, Володар Перснів, Игра престолов",
            on_submit=self._on_search,
            expand=True,
            dense=True,
        )
        self.search_btn = ft.IconButton(
            ft.Icons.SEARCH, on_click=self._on_search, tooltip="Найти",
        )

        self.search_results_list = ft.ListView(spacing=4, padding=4, expand=True)
        self.favorites_list = ft.ListView(spacing=4, padding=4, expand=True)
        self.recs_list = ft.ListView(
            spacing=6,
            padding=8,
            expand=True,
            auto_scroll=False,
        )

        self.recs_movie_btn = ft.ElevatedButton(
            "Рекомендовать фильмы",
            icon=ft.Icons.MOVIE, tooltip="/recs_movie — похожие фильмы",
            on_click=lambda _e: self._on_recs("movie"),
            bgcolor=COLORS["primary"], color=ft.Colors.WHITE,
        )
        self.recs_tv_btn = ft.ElevatedButton(
            "Рекомендовать сериалы",
            icon=ft.Icons.TV, tooltip="/recs_tv — похожие сериалы",
            on_click=lambda _e: self._on_recs("tv"),
            bgcolor=COLORS["primary"], color=ft.Colors.WHITE,
        )
        self.recs_all_btn = ft.ElevatedButton(
            "Фильмы и сериалы",
            icon=ft.Icons.APPS, tooltip="/recs_all — общая лента",
            on_click=lambda _e: self._on_recs("all"),
            bgcolor=COLORS["primary"], color=ft.Colors.WHITE,
        )
        self.recs_cross_btn = ft.ElevatedButton(
            "Кросс-домен",
            icon=ft.Icons.SWAP_HORIZ,
            on_click=lambda _e: self._on_recs("cross"),
            bgcolor=COLORS["accent"], color=ft.Colors.WHITE,
            disabled=not self._faiss_available,
            tooltip=(
                "/recs_cross — переход между доменами (фильм↔сериал) через FAISS"
                if self._faiss_available
                else "FAISS catalog not found — запустите "
                     "`compute_embeddings --to-faiss`"
            ),
        )

        self.status_text = ft.Text("Готов к работе", size=13, color=COLORS["muted"])
        self.router_status = ft.Text(
            "⏳ Движки не загружены (загрузятся по первому /recs_*)",
            size=12, color=COLORS["muted"], italic=True,
        )

        self.faiss_banner = ft.Container(visible=not self._faiss_available)
        if not self._faiss_available:
            self.faiss_banner.content = ft.Text(
                "⚠ FAISS catalog отсутствует — /recs_cross недоступен. "
                "Соберите индекс: python -m recommendation_system.models.gnn."
                "compute_embeddings --to-faiss",
                color=COLORS["warn"], size=12,
            )
            self.faiss_banner.bgcolor = ft.Colors.AMBER_50
            self.faiss_banner.padding = 8
            self.faiss_banner.border_radius = 6

        search_row = ft.Row(
            [self.scope_dd, self.search_field, self.search_btn],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.END,
        )

        recs_row = ft.Row(
            [
                self.recs_movie_btn,
                self.recs_tv_btn,
                self.recs_all_btn,
                self.recs_cross_btn,
                self.cross_target_dd,
                self.lang_dd,
            ],
            spacing=8,
            wrap=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        left_col = ft.Container(
            content=ft.Column(
                [
                    ft.Text("Поиск", size=14, weight=ft.FontWeight.W_500,
                            color=COLORS["muted"]),
                    search_row,
                    ft.Container(
                        content=self.search_results_list,
                        bgcolor=ft.Colors.GREY_50,
                        border=ft.border.all(1, COLORS["card_border"]),
                        border_radius=6,
                        expand=True,  # тянется по высоте, ListView сам скроллит
                    ),
                    ft.Text("Избранное (seed)", size=14, weight=ft.FontWeight.W_500,
                            color=COLORS["muted"]),
                    ft.Container(
                        content=self.favorites_list,
                        bgcolor=ft.Colors.GREY_50,
                        border=ft.border.all(1, COLORS["card_border"]),
                        border_radius=6,
                        height=180,
                    ),
                ],
                spacing=8,
                expand=True,
            ),
            expand=2,
        )

        movies_ckpt_row = ft.Row(
            [
                self.movies_latest_switch,
                ft.Text("latest", size=12, color=COLORS["muted"]),
                self.movies_ckpt_input,
                self.movies_ckpt_btn,
            ],
            spacing=6,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        tv_ckpt_row = ft.Row(
            [
                self.tv_latest_switch,
                ft.Text("latest", size=12, color=COLORS["muted"]),
                self.tv_ckpt_input,
                self.tv_ckpt_btn,
            ],
            spacing=6,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        models_row = ft.Column(
            [movies_ckpt_row, tv_ckpt_row],
            spacing=4,
        )

        # FilePicker'ы должны жить в page.overlay.
        self.app.page.overlay.append(self.movies_ckpt_picker)
        self.app.page.overlay.append(self.tv_ckpt_picker)

        status_box = ft.Container(
            content=ft.Column([self.status_text, self.router_status], spacing=4),
            bgcolor=ft.Colors.AMBER_50,
            padding=15,
            border_radius=6,
            border=ft.border.all(1, COLORS["warn"]),
        )

        scrollable_col = ft.Column(
            [
                ft.Text("Чекпоинты моделей", size=14, weight=ft.FontWeight.W_500, color=COLORS["muted"]),
                models_row,
                ft.Text("Рекомендации", size=14, weight=ft.FontWeight.W_500, color=COLORS["muted"]),
                recs_row,
                self.faiss_banner,
                ft.Container(
                    content=self.recs_list,
                    bgcolor=ft.Colors.WHITE,
                    border=ft.border.all(1, COLORS["card_border"]),
                    border_radius=6,
                    expand=True,
                ),
            ],
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

        right_col = ft.Container(
            content=ft.Column(
                [
                    scrollable_col,
                    status_box,
                ],
                spacing=8,
                expand=True,
            ),
            expand=3,
        )

        # 3. Корень вкладки
        self._root = ft.Container(
            content=ft.Row(
                [left_col, right_col],
                spacing=16,
                expand=True,
                vertical_alignment=ft.CrossAxisAlignment.START,
            ),
            padding=16,
            expand=True,
        )

    # ---- handlers ----

    def _on_search(self, e) -> None:
        if not self.search_field.value:
            return
        query = self.search_field.value.strip()
        scope = self.scope_dd.value

        # Если движки ещё не подняты — показать индикатор в списке результатов
        # (а не только в маленьком router_status внизу), запустить load и выйти.
        # router_ready сам перевызовет _on_search через after="search-retry".
        engine_missing = (
            (scope == "movies" and self.movies_engine is None)
            or (scope == "tv" and self.tv_engine is None)
        )
        if engine_missing:
            self.search_results_list.controls.clear()
            self.search_results_list.controls.append(
                ft.Text(
                    "⏳ Загрузка движков (30-60 с). Подожди — поиск повторится автоматически.",
                    size=12, color=COLORS["primary"], italic=True,
                ),
            )
            self.app.page.update()
            self._trigger_router_load(after="search-retry")
            return

        engine = self.movies_engine if scope == "movies" else self.tv_engine
        try:
            res = engine.search(query, limit=20)
            items = list(res.results)
        except Exception as exc:
            self._set_status(f"Ошибка поиска: {exc}", COLORS["err"])
            return

        self.search_results_list.controls.clear()
        if not items:
            self.search_results_list.controls.append(
                ft.Text("(ничего не найдено)", size=12,
                        color=COLORS["muted"], italic=True),
            )
        else:
            for it in items:
                self.search_results_list.controls.append(self._build_search_row(it))
        self._set_status(f"Найдено: {len(items)}", COLORS["muted"])
        self.app.page.update()

    def _resolve_latest_name(self, domain: str) -> str:
        """Имя файла, который загрузится при Switch ON (latest sidecar → fallback v4)."""
        stats = _collect_domain_stats(domain)
        if stats.last_train_checkpoint:
            return stats.last_train_checkpoint
        return f"lightgcn_{domain}_best_v4.pt"

    def _refresh_ckpt_display(self, domain: str) -> None:
        """Перерисовать TextField с тем, что реально загрузится."""
        if domain == "movies":
            inp, sw, override = (
                self.movies_ckpt_input,
                self.movies_latest_switch,
                self.movies_checkpoint_override,
            )
        else:
            inp, sw, override = (
                self.tv_ckpt_input,
                self.tv_latest_switch,
                self.tv_checkpoint_override,
            )
        if sw.value:
            inp.value = f"latest → {self._resolve_latest_name(domain)}"
            inp.tooltip = None
        elif override is not None:
            inp.value = f"override → {override.name}"
            inp.tooltip = str(override)
        else:
            inp.value = "(выберите .pt через иконку →)"
            inp.tooltip = None

    def _on_latest_switch(self, domain: str, e) -> None:
        """Switch ON = latest (override снять). OFF = override (если уже выбран — оставить)."""
        use_latest = bool(e.control.value)
        if use_latest:
            if domain == "movies":
                self.movies_checkpoint_override = None
            else:
                self.tv_checkpoint_override = None
        self._refresh_ckpt_display(domain)
        self._maybe_warn_hot_swap()
        self.app.page.update()

    def _on_ckpt_picked(self, domain: str, e: ft.FilePickerResultEvent) -> None:
        """Файл выбран через picker → Switch OFF, override = абсолютный путь."""
        if not e.files:
            return
        picked = Path(e.files[0].path)
        if domain == "movies":
            self.movies_latest_switch.value = False
            self.movies_checkpoint_override = picked
        else:
            self.tv_latest_switch.value = False
            self.tv_checkpoint_override = picked
        self._refresh_ckpt_display(domain)
        self._maybe_warn_hot_swap()
        self.app.page.update()

    def _maybe_warn_hot_swap(self) -> None:
        """Если движки уже загружены — намекнуть про restart."""
        if self.router is not None:
            self.router_status.value = (
                "⚠ Чекпоинт изменён — перезапустите GUI, чтобы применить "
                "(hot-swap не поддерживается)"
            )
            self.router_status.color = COLORS["warn"]

    def _on_lang_change(self, e) -> None:
        # Re-render current results + favorites + search list with the new lang.
        self._render_recs(self.last_results)
        self._rebuild_favorites()
        # Search-result list rebuild requires the source items, which we
        # didn't keep; skip until the user re-runs the search.
        self.app.page.update()

    def _on_recs(self, scope: str) -> None:
        if not self.favorites:
            self.recs_list.controls.clear()
            self.recs_list.controls.append(
                ft.Text(
                    "Сначала добавь элементы в избранное (★ из списка поиска).",
                    size=12, color=COLORS["warn"], italic=True,
                ),
            )
            self._set_status("Нужны элементы в ★", COLORS["warn"])
            self.app.page.update()
            return

        if self.router is None:
            self.recs_list.controls.clear()
            self.recs_list.controls.append(
                ft.Text(
                    "⏳ Загрузка движков (30-60 с). Рекомендации повторятся автоматически.",
                    size=12, color=COLORS["primary"], italic=True,
                ),
            )
            self.app.page.update()
            self._trigger_router_load(after=("recs", scope))
            return

        if self._thread is not None and self._thread.is_alive():
            return

        self._toggle_buttons(False)
        self.recs_list.controls.clear()
        self.recs_list.controls.append(
            ft.Text(
                f"⏳ Запрос «{scope}»... ждём ответ модели.",
                size=12, color=COLORS["primary"], italic=True,
            ),
        )
        self._set_status(f"Запрос {scope}...", COLORS["muted"])
        self.app.page.update()

        self._thread = threading.Thread(
            target=self._run_recs, args=(scope,), daemon=True,
        )
        self._thread.start()
        self.app.page.run_task(self._process_updates)

    def _trigger_router_load(self, after) -> None:
        """First-time engine build. Heavy (~30-60s). `after` is what to do
        once loaded — `"search-retry"` or `("recs", scope)`."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._toggle_buttons(False)
        self.router_status.value = "⏳ Загрузка движков (30-60с)..."
        self.router_status.color = COLORS["primary"]
        self.app.page.update()

        self._thread = threading.Thread(
            target=self._run_router_load, args=(after,), daemon=True,
        )
        self._thread.start()
        self.app.page.run_task(self._process_updates)

    def _add_to_favorites(self, item) -> None:
        tid = int(item.tmdb_id)
        if any(t == tid for t, _, _ in self.favorites):
            return
        self.favorites.append((tid, item.title, str(item.media_type)))
        self._rebuild_favorites()
        self.app.page.update()

    def _remove_favorite(self, tmdb_id: int) -> None:
        self.favorites = [(t, ti, mt) for t, ti, mt in self.favorites if t != tmdb_id]
        self._rebuild_favorites()
        self.app.page.update()

    def _rebuild_favorites(self) -> None:
        self.favorites_list.controls.clear()
        if not self.favorites:
            self.favorites_list.controls.append(
                ft.Text("(пусто — добавьте через поиск)", size=12,
                        color=COLORS["muted"], italic=True),
            )
            return
        for tid, title, mt in self.favorites:
            badge = "🎬" if mt == "movie" else "📺"
            self.favorites_list.controls.append(
                ft.Row(
                    [
                        ft.Text(f"{badge} {title}", size=12, expand=True,
                                no_wrap=False),
                        ft.Text(f"tmdb={tid}", size=10, color=COLORS["muted"]),
                        ft.IconButton(
                            ft.Icons.CLOSE, icon_size=14,
                            tooltip="Убрать из избранного",
                            on_click=lambda _e, t=tid: self._remove_favorite(t),
                        ),
                    ],
                    spacing=6,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                )
            )

    # ---- worker threads ----

    def _run_router_load(self, after) -> None:
        try:
            from recommendation_system.models.gnn.inference_engine import InferenceEngine
            from recommendation_system.models.gnn.universal_search import UniversalSearchEngine
            from recommendation_system.models.gnn.dual_domain_engine import DualDomainEngine
            from recommendation_system.models.gnn.faiss_bridge import FaissCatalog
            import pandas as pd  # noqa: F401  (used inline below)

            device = self.app.current_device()
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)

            self._update_q.put({"type": "router_progress", "text": "Movies: загрузка checkpoint..."})
            movies = self._build_domain_engine("movies", device, InferenceEngine, UniversalSearchEngine)

            self._update_q.put({"type": "router_progress", "text": "TV: загрузка checkpoint..."})
            tv = self._build_domain_engine("tv", device, InferenceEngine, UniversalSearchEngine)

            self._update_q.put({"type": "router_progress", "text": "FAISS: загрузка индекса..."})
            faiss_catalog = (
                FaissCatalog.load(_FAISS_INDEX, _FAISS_META)
                if self._faiss_available else None
            )

            self._update_q.put({
                "type": "router_ready",
                "movies": movies,
                "tv": tv,
                "router": DualDomainEngine(
                    movies_engine=movies, tv_engine=tv,
                    faiss_catalog=faiss_catalog,
                ),
                "after": after,
            })
        except Exception:
            self._update_q.put({"type": "router_error", "tb": traceback.format_exc()})

    def _build_domain_engine(self, domain, device, InferenceEngine, UniversalSearchEngine):
        import pandas as pd

        stats = _collect_domain_stats(domain)
        override: Optional[Path] = (
            self.movies_checkpoint_override if domain == "movies"
            else self.tv_checkpoint_override
        )
        # Приоритет: явный override (абс. путь) → latest sidecar → хардкод v4.
        if override is not None:
            checkpoint = override
        elif stats.last_train_checkpoint:
            checkpoint = stats.models_dir / stats.last_train_checkpoint
        else:
            checkpoint = stats.models_dir / f"lightgcn_{domain}_best_v4.pt"
        dataset_dir = stats.dataset_dir

        if not checkpoint.exists():
            raise FileNotFoundError(f"{domain} checkpoint not found: {checkpoint}")
        if not (dataset_dir / "items_metadata_final.parquet").exists():
            raise FileNotFoundError(
                f"{domain} dataset not built — см. вкладку «Создание датасета»"
            )

        infer = InferenceEngine(dataset_dir, checkpoint, device=device)
        ok, msg = infer.load_resources()
        if not ok:
            raise RuntimeError(f"InferenceEngine.{domain}: {msg}")

        metadata = pd.read_parquet(dataset_dir / "items_metadata_final.parquet")
        embeddings = dataset_dir / "overview_embeddings.npy"
        return UniversalSearchEngine(
            metadata=metadata,
            cache_dir=_CACHE_DIR,
            tmdb_api_key=os.getenv("TMDB_API_KEY"),
            inference_engine=infer,
            embeddings_path=embeddings if embeddings.exists() else None,
            model_num_items=infer.model.num_items,
        )

    def _run_recs(self, scope: str) -> None:
        try:
            tmdb_ids = [t for t, _, _ in self.favorites]
            if scope == "movie":
                recs = self.router.recs_movie(tmdb_ids, top_k=8)
            elif scope == "tv":
                recs = self.router.recs_tv(tmdb_ids, top_k=8)
            elif scope == "all":
                recs = self.router.recs_all(tmdb_ids, top_k=8)
            elif scope == "cross":
                target = self.cross_target_dd.value
                recs = self.router.recs_cross(
                    liked_tmdb_ids=tmdb_ids,
                    target_media_type=target,
                    top_k=8,
                )
            else:
                recs = []
            diagnostics = None
            if not recs and scope in {"movie", "tv", "all"}:
                diagnostics = self._seed_diagnostics(tmdb_ids, scope)
            self._update_q.put({
                "type": "recs", "scope": scope,
                "items": list(recs), "diagnostics": diagnostics,
            })
        except Exception:
            self._update_q.put({"type": "error", "tb": traceback.format_exc()})

    def _seed_diagnostics(self, tmdb_ids: list[int], scope: str) -> str:
        """Для empty-result случая: сколько seed-id известно каждому graph."""
        from recommendation_system.models.gnn.faiss_bridge import TV_OFFSET
        movie_seed = [t for t in tmdb_ids if t < TV_OFFSET]
        tv_seed = [t for t in tmdb_ids if t >= TV_OFFSET]
        lines = [f"seed: {len(movie_seed)} movie, {len(tv_seed)} tv"]
        if scope in {"movie", "all"} and self.movies_engine is not None:
            known = sum(
                1 for t in movie_seed
                if t in self.movies_engine.tmdb_to_item_id
            )
            lines.append(f"movies graph знает: {known}/{len(movie_seed)}")
        if scope in {"tv", "all"} and self.tv_engine is not None:
            known = sum(
                1 for t in tv_seed
                if t in self.tv_engine.tmdb_to_item_id
            )
            lines.append(f"tv graph знает: {known}/{len(tv_seed)}")
        return " | ".join(lines)

    async def _process_updates(self) -> None:
        try:
            while self._thread is not None and self._thread.is_alive():
                self._drain_queue()
                await asyncio.sleep(0.1)
            self._drain_queue()
        finally:
            self._toggle_buttons(True)
            self.app.page.update()

    def _drain_queue(self) -> None:
        drained = False
        try:
            while True:
                msg = self._update_q.get_nowait()
                try:
                    self._handle_message(msg)
                except Exception:
                    import sys
                    tb = traceback.format_exc()
                    print(tb, file=sys.stderr)
                    try:
                        self._handle_message({"type": "error", "tb": tb})
                    except Exception:
                        print("(_drain_queue: error-ветка тоже упала)", file=sys.stderr)
                drained = True
        except queue.Empty:
            pass
        if drained:
            self.app.page.update()

    def _handle_message(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "router_progress":
            self.router_status.value = "⏳ " + msg["text"]
            self.router_status.color = COLORS["primary"]
        elif kind == "router_ready":
            self.movies_engine = msg["movies"]
            self.tv_engine = msg["tv"]
            self.router = msg["router"]
            self.router_status.value = "✅ Движки загружены"
            self.router_status.color = COLORS["ok"]
            self._set_status("Готов", COLORS["muted"])
            after = msg.get("after")
            if after == "search-retry":
                self._on_search(None)
            elif isinstance(after, tuple) and after[0] == "recs":
                self._on_recs(after[1])
        elif kind == "router_error":
            self.router_status.value = "❌ Ошибка загрузки движков (см. ниже)"
            self.router_status.color = COLORS["err"]
            self.recs_list.controls.clear()
            self.recs_list.controls.append(
                ft.Text(msg["tb"], size=11, color=COLORS["err"],
                        selectable=True, font_family="Consolas"),
            )
        elif kind == "recs":
            items = msg["items"]
            self.last_results = items
            if items:
                self._render_recs(items)
            else:
                self._render_empty_recs(msg.get("diagnostics"))
            self._set_status(
                f"{msg['scope']}: получено {len(items)} рекомендаций",
                COLORS["ok"] if items else COLORS["warn"],
            )
            # Diagnostic: explicit updates to force redraw of recs_list + page
            self.recs_list.update()
            self.app.page.update()
        elif kind == "error":
            self.recs_list.controls.clear()
            self.recs_list.controls.append(
                ft.Text(msg["tb"], size=11, color=COLORS["err"],
                        selectable=True, font_family="Consolas"),
            )
            self._set_status("Ошибка (см. вывод)", COLORS["err"])

    # ---- rendering ----

    def _build_search_row(self, item) -> ft.Control:
        lang = self.lang_dd.value
        title = _display_title(item, lang)
        year = item.year if item.year else "?"
        badge = "🎬" if item.media_type == "movie" else "📺"
        url = _tmdb_url(item)
        title_text = ft.Text(
            f"{badge} {title} ({year})",
            size=12, no_wrap=False,
            color=COLORS["primary"],
            tooltip=f"Открыть {url}",
        )
        return ft.Row(
            [
                ft.Container(
                    content=title_text,
                    expand=True,
                    on_click=lambda _e, u=url: webbrowser.open(u),
                    ink=True,
                    padding=2,
                    border_radius=4,
                ),
                ft.IconButton(
                    ft.Icons.STAR_BORDER, icon_size=18, icon_color=COLORS["accent"],
                    tooltip="Добавить в избранное",
                    on_click=lambda _e, it=item: self._add_to_favorites(it),
                ),
            ],
            spacing=4,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

    def _render_recs(self, items) -> None:
        self.recs_list.controls.clear()
        if not items:
            self.recs_list.controls.append(
                ft.Text("(нет результатов)", size=12,
                        color=COLORS["muted"], italic=True),
            )
            self.recs_list.update()
            self.app.page.update()
            return
        lang = self.lang_dd.value
        for it in items:
            self.recs_list.controls.append(self._build_recs_card(it, lang))
        self.recs_list.update()
        self.app.page.update()

    def _render_empty_recs(self, diagnostics: Optional[str]) -> None:
        self.recs_list.controls.clear()
        self.recs_list.controls.append(
            ft.Container(
                content=ft.Column(
                    [
                        ft.Text("⚠ Получено 0 рекомендаций",
                                size=14, weight=ft.FontWeight.BOLD,
                                color=COLORS["warn"]),
                        ft.Text(diagnostics or "(нет диагностики)",
                                size=12, color=COLORS["muted"],
                                selectable=True),
                    ],
                    spacing=4,
                ),
                padding=10,
                bgcolor=ft.Colors.AMBER_50,
                border=ft.border.all(1, COLORS["warn"]),
                border_radius=6,
            ),
        )
        self.recs_list.update()
        self.app.page.update()

    def _build_recs_card(self, item, lang: str) -> ft.Control:
        title = _display_title(item, lang)
        year = item.year if item.year else "?"
        badge = "🎬 Movies" if item.media_type == "movie" else "📺 TV"
        genres_text = ", ".join(item.genres[:5]) if item.genres else "—"
        url = _tmdb_url(item)
        score_parts: list[str] = []
        if getattr(item, "vote_average", 0):
            score_parts.append(f"TMDB {item.vote_average:.1f}")
        if getattr(item, "source", None):
            score_parts.append(f"src={item.source}")

        return ft.Container(
            padding=10,
            border=ft.border.all(1, COLORS["card_border"]),
            border_radius=6,
            bgcolor=ft.Colors.WHITE,
            on_click=lambda _e, u=url: webbrowser.open(u),
            ink=True,
            tooltip=f"Открыть {url}",
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Text(f"{title} ({year})", size=14,
                                    weight=ft.FontWeight.BOLD, expand=True,
                                    color=COLORS["primary"]),
                            ft.Text(badge, size=11, color=COLORS["muted"]),
                        ],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Text(genres_text, size=11, color=COLORS["muted"]),
                    ft.Text(" · ".join(score_parts) if score_parts else "",
                            size=10, color=COLORS["muted"], italic=True),
                ],
                spacing=2,
            ),
        )

    # ---- helpers ----

    def _toggle_buttons(self, enabled: bool) -> None:
        self.recs_movie_btn.disabled = not enabled
        self.recs_tv_btn.disabled = not enabled
        self.recs_all_btn.disabled = not enabled
        # Cross always respects FAISS availability.
        self.recs_cross_btn.disabled = (not enabled) or (not self._faiss_available)
        self.search_btn.disabled = not enabled

    def _set_status(self, text: str, color) -> None:
        self.status_text.value = text
        self.status_text.color = color


# ======================================================================
# Main app
# ======================================================================


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
        self.page.add(
            ft.Column([header, ft.Divider(height=1), tabs], expand=True, spacing=12)
        )

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
        if selected:            self.device = selected[0]
        self.page.update()


def main(page: ft.Page) -> None:
    TrainerGuiApp(page)


if __name__ == "__main__":
    # Windows-консоль (cp1251) не умеет печатать эмодзи (📊 ⏳ ✅ …),
    # из-за чего фоновый _run_router_load молча падал UnicodeEncodeError-ом
    # внутри inference_engine.print(...). Заворачиваем stdout/stderr в UTF-8.
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    ft.app(target=main)
