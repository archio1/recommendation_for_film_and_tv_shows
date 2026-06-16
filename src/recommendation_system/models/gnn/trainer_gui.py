"""
trainer_gui.py — точка входа Flet-GUI для dual-LightGCN стека.

Реализация вынесена в подпакет `gui/`:
    gui/theme.py         — общие константы (COLORS, PROJECT_ROOT)
    gui/common.py        — кросс-табные утилиты (логгер-мост, file-picker хелперы)
    gui/domain_stats.py  — сбор статистики датасетов/моделей
    gui/training_tab.py  — вкладка «Обучение»
    gui/dataset_tab.py   — вкладка «Создание датасета»
    gui/inference_tab.py — вкладка «Тестирование»
    gui/data_tab.py      — вкладка «Данные»
    gui/app.py           — TrainerGuiApp + main()

Запуск: recsys-gui  (или python -m recommendation_system.models.gnn.trainer_gui)
"""

from __future__ import annotations

from recommendation_system.models.gnn.gui.app import TrainerGuiApp, main  # noqa: F401


def run() -> None:
    """Console-script entry point (`recsys-gui`, см. [project.scripts]).

    Windows-консоль (cp1251) не умеет печатать эмодзи (📊 ⏳ ✅ …),
    из-за чего фоновый _run_router_load молча падал UnicodeEncodeError-ом
    внутри inference_engine.print(...). Заворачиваем stdout/stderr в UTF-8.
    """
    import sys

    import flet as ft

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    ft.app(target=main)


if __name__ == "__main__":
    run()
