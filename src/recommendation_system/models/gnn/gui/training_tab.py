"""Вкладка «Обучение» — обёртка над trainer.main() с UI гиперпараметров и live-прогрессом."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

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
    danger_button,
    primary_button,
    secondary_button,
)

if TYPE_CHECKING:
    from recommendation_system.models.gnn.gui.app import TrainerGuiApp


def _hp_field(label: str, default: str) -> ft.TextField:
    # col: 2 поля в ряд на узком, 3 на среднем, 4 на широком окне.
    return ft.TextField(
        label=label, value=default, text_size=14, dense=True,
        col={"xs": 6, "sm": 4, "md": 3},
    )


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
            expand=True,
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
            expand=True,
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

        self.start_btn = primary_button(
            "Старт",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._on_start,
        )
        self.stop_btn = danger_button(
            "Стоп",
            icon=ft.Icons.STOP_CIRCLE,
            on_click=self._on_stop,
            disabled=True,
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
        self.sidecar_btn = secondary_button(
            "Открыть sidecar.json",
            icon=ft.Icons.DESCRIPTION,
            on_click=self._on_open_sidecar,
            visible=False,
        )
        self.folder_btn = secondary_button(
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

        hp_row = ft.ResponsiveRow(
            [
                self.epochs_input, self.batch_size_input, self.lr_input,
                self.embedding_dim_input, self.num_layers_input,
                self.patience_input, self.eval_every_input,
            ],
            spacing=GAP_M, run_spacing=GAP_M,
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
