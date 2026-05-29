"""Кросс-табные утилиты GUI: открытие в проводнике, поиск папок, логгер-мост."""

from __future__ import annotations

import logging
import queue
import os
from pathlib import Path

from recommendation_system.models.gnn.gui.theme import COLORS


def _first_existing_ancestor(path: Path, fallback: Path) -> Path:
    """Walk up from `path` until existing dir, else fallback. Flet иногда игнорирует
    initial_directory если путь не существует и открывает picker в неожиданном месте."""
    p = path
    while p != p.parent and not p.is_dir():
        p = p.parent
    return p if p.is_dir() else fallback


def _open_in_explorer(path: Path) -> None:
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif os.name == "posix":
            import subprocess
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


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
