"""
Фикстуры для тестов подпакета `gui/` (trainer_gui).

Все тяжёлое имитируем крошечными tmp-артефактами — реальные модели/датасеты не нужны:

- `fake_page`            — минимальная заглушка `ft.Page` (overlay/update/run_task/width/window).
- `make_dataset_dir`     — пишет interactions/items parquet + id_mapping.json в указанную папку.
- `make_model_dir`       — кладёт фейковый `*.pt` (+ опц. sidecar `*.json`) в указанную папку.

Пакет импортируется через editable install / `pythonpath = ["src"]` в pyproject,
поэтому `from recommendation_system.models.gnn.gui import ...` работает.
"""

from __future__ import annotations

import json
import os
import types
from pathlib import Path

import pandas as pd
import pytest


# --------------------------------------------------------------------------
# Fake Flet page
# --------------------------------------------------------------------------

class FakePage:
    """Заглушка ft.Page: достаточно для конструирования вкладок и проверки state.

    Не рендерит ничего; копит вызовы update() и хранит overlay (FilePicker'ы
    туда регистрируются во время _build).
    """

    def __init__(self, width: float | None = 1920, height: float | None = 1080):
        self.overlay: list = []
        self.title = None
        self.padding = None
        self.theme_mode = None
        self.width = width
        self.height = height
        self.window = types.SimpleNamespace(width=width, height=height)
        self.on_resized = None
        self.update_calls = 0
        self.added: tuple = ()

    def add(self, *controls):
        self.added = controls

    def update(self):
        self.update_calls += 1

    def run_task(self, *_args, **_kwargs):
        # В тестах асинхронные задачи не запускаем.
        return None


@pytest.fixture
def fake_page():
    return FakePage()


@pytest.fixture
def make_fake_page():
    """Фабрика fake_page с произвольной шириной (для тестов max-width)."""
    def _factory(width: float | None = 1920, height: float | None = 1080) -> FakePage:
        return FakePage(width=width, height=height)
    return _factory


# --------------------------------------------------------------------------
# Fake app (родитель вкладок)
# --------------------------------------------------------------------------

class FakeApp:
    """Минимальный родитель для изолированного конструирования вкладок.

    Вкладки читают только `self.app.page`, `self.app.current_device()` и
    (DatasetTab после сборки) `self.app.data_tab.refresh()` — последнее обёрнуто
    в try/except, поэтому отсутствие data_tab безопасно.
    """

    def __init__(self, page: FakePage, device: str = "cpu"):
        self.page = page
        self.device = device

    def current_device(self) -> str:
        return self.device


@pytest.fixture
def fake_app(fake_page):
    return FakeApp(fake_page)


class FakeEvent:
    """Заглушка ft.ControlEvent: e.control.value."""

    def __init__(self, value=None, control=None):
        self.control = control if control is not None else types.SimpleNamespace(value=value)


@pytest.fixture
def make_event():
    def _factory(value=None, control=None) -> FakeEvent:
        return FakeEvent(value=value, control=control)
    return _factory


# --------------------------------------------------------------------------
# Tmp dataset / model builders
# --------------------------------------------------------------------------

def _write_dataset(
    folder: Path,
    *,
    n_interactions: int = 200,
    num_users: int = 100,
    num_items: int | None = 10,
    num_trained_items: int | None = None,
    duplicate_tmdb: bool = False,
    items_columns: tuple[str, ...] = ("tmdb_id", "title", "genres"),
    write_interactions: bool = True,
    write_items: bool = True,
    write_mapping: bool = True,
    mapping_text: str | None = None,
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)

    if write_interactions:
        inter = pd.DataFrame({
            "user_id": [i % num_users for i in range(n_interactions)],
            "item_id": [i % 10 for i in range(n_interactions)],
        })
        inter.to_parquet(folder / "interactions_final.parquet")

    if write_items:
        n = 5
        tmdb = [1, 1, 3, 4, 5] if duplicate_tmdb else [1, 2, 3, 4, 5]
        data = {
            "tmdb_id": tmdb,
            "title": [f"Title {i}" for i in range(n)],
            "genres": [["Action"], ["Drama"], ["Sci-Fi"], ["Horror"], ["Comedy"]],
        }
        # Оставляем только запрошенные колонки (для проверки missing-cols в sanity).
        df = pd.DataFrame(data)[list(items_columns)]
        df.to_parquet(folder / "items_metadata_final.parquet")

    if write_mapping:
        if mapping_text is not None:
            (folder / "id_mapping.json").write_text(mapping_text, encoding="utf-8")
        else:
            mapping: dict = {"num_users": num_users}
            if num_items is not None:
                mapping["num_items"] = num_items
            if num_trained_items is not None:
                mapping["num_trained_items"] = num_trained_items
            (folder / "id_mapping.json").write_text(
                json.dumps(mapping), encoding="utf-8"
            )

    return folder


@pytest.fixture
def make_dataset_dir():
    """Фабрика: пишет файлы датасета в указанную папку и возвращает её путь."""
    return _write_dataset


def _write_model(
    folder: Path,
    name: str = "lightgcn_movies_best_v5.pt",
    *,
    sidecar: bool = True,
    domain: str | None = "movies",
    recall: float | None = 0.2058,
    trained_at: str | None = "2026-05-29T13:59:00Z",
    sanity_passed: bool | None = True,
    checkpoint_in_sidecar: str | None = None,
    sidecar_text: str | None = None,
    mtime: float | None = None,
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    pt = folder / name
    pt.write_bytes(b"\x00\x01\x02")  # _collect_model_stats не делает torch.load

    if sidecar:
        scar = pt.with_suffix(".json")
        if sidecar_text is not None:
            scar.write_text(sidecar_text, encoding="utf-8")
        else:
            payload: dict = {}
            if domain is not None:
                payload["domain"] = domain
            if trained_at is not None:
                payload["trained_at"] = trained_at
            if recall is not None:
                payload["metrics"] = {"recall@10": recall}
            payload["checkpoint"] = checkpoint_in_sidecar or name
            if sanity_passed is not None:
                payload["sanity_check"] = {"passed": sanity_passed}
            scar.write_text(json.dumps(payload), encoding="utf-8")

    if mtime is not None:
        os.utime(pt, (mtime, mtime))
    return pt


@pytest.fixture
def make_model_dir():
    """Фабрика: кладёт фейковый .pt (+ опц. sidecar) и возвращает путь к .pt."""
    return _write_model
