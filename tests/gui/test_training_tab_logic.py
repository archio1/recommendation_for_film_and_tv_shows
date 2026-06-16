"""
Тесты логики вкладки «Обучение» — `gui/training_tab.py`.

Без реального обучения: парсинг путей из логов, sync data_dir под домен,
валидация гиперпараметров и сбор argv (с замоканным trainer.main).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from recommendation_system.models.gnn.gui import training_tab as tt
from recommendation_system.models.gnn.gui.training_tab import TrainingTab


@pytest.fixture
def tab(fake_app):
    return TrainingTab(fake_app)


# ======================================================================
# _capture_paths — извлечение sidecar/checkpoint из логов trainer
# ======================================================================

class TestCapturePaths:
    def test_captures_sidecar(self, tab):
        tab._capture_paths("12:00 | INFO | Sidecar:    /models/movies/run_v5.json")
        assert tab._sidecar_path == Path("/models/movies/run_v5.json")

    def test_captures_checkpoint(self, tab):
        tab._capture_paths("Checkpoint: /models/movies/run_v5.pt")
        assert tab._output_path == Path("/models/movies/run_v5.pt")

    def test_captures_output_line(self, tab):
        tab._capture_paths("Output: /models/movies/out.pt")
        assert tab._output_path == Path("/models/movies/out.pt")

    def test_ignores_non_matching(self, tab):
        tab._capture_paths("just a normal log line")
        assert tab._sidecar_path is None
        assert tab._output_path is None

    def test_sidecar_must_end_json(self, tab):
        tab._capture_paths("Sidecar: /models/movies/notjson.txt")
        assert tab._sidecar_path is None


# ======================================================================
# _on_domain_changed — синхронизация data_dir под выбранный домен
# ======================================================================

class TestDomainSync:
    def test_syncs_data_dir(self, tab, make_event):
        tab.domain_dd.value = "tv"
        tab._on_domain_changed(make_event())
        expected = str(tt.PROJECT_ROOT / "data" / "processed" / "tv")
        assert tab.data_dir_input.value == expected

    def test_manual_override_disables_sync(self, tab, make_event):
        tab._on_data_dir_edited(make_event())  # пометили ручную правку
        tab.data_dir_input.value = "/my/custom/path"
        tab.domain_dd.value = "tv"
        tab._on_domain_changed(make_event())
        assert tab.data_dir_input.value == "/my/custom/path"  # не перезаписан


# ======================================================================
# _on_start — валидация и сбор argv
# ======================================================================

class TestStartValidation:
    def test_invalid_hyperparam_no_thread(self, tab, make_event):
        tab.epochs_input.value = "abc"
        tab._on_start(make_event())
        assert tab._thread is None  # поток не стартовал

    def test_missing_data_dir_no_thread(self, tab, make_event, tmp_path):
        tab.data_dir_input.value = str(tmp_path / "does_not_exist")
        tab._on_start(make_event())
        assert tab._thread is None

    def test_builds_expected_argv(self, tab, make_event, tmp_path, monkeypatch):
        import recommendation_system.models.gnn.trainer as trainer_mod

        captured = {}

        def fake_main(argv, on_epoch_end=None, stop_flag=None):
            captured["argv"] = argv
            return 0

        monkeypatch.setattr(trainer_mod, "main", fake_main)

        tab.domain_dd.value = "movies"
        tab.epochs_input.value = "3"
        tab.data_dir_input.value = str(tmp_path)  # существует

        tab._on_start(make_event())
        assert tab._thread is not None
        tab._thread.join(timeout=10)

        argv = captured["argv"]
        assert "--domain" in argv and argv[argv.index("--domain") + 1] == "movies"
        assert "--epochs" in argv and argv[argv.index("--epochs") + 1] == "3"
        assert "--device" in argv and argv[argv.index("--device") + 1] == "cpu"
        data_dir_arg = argv[argv.index("--data-dir") + 1]
        assert Path(data_dir_arg) == Path(tmp_path).resolve()

    def test_custom_model_name_adds_output_flag(self, tab, make_event, tmp_path, monkeypatch):
        import recommendation_system.models.gnn.trainer as trainer_mod
        captured = {}
        monkeypatch.setattr(
            trainer_mod, "main",
            lambda argv, on_epoch_end=None, stop_flag=None: captured.setdefault("argv", argv) or 0,
        )
        tab.data_dir_input.value = str(tmp_path)
        tab.model_name_input.value = "my_model"
        tab._on_start(make_event())
        tab._thread.join(timeout=10)

        argv = captured["argv"]
        assert "--output" in argv
        out = argv[argv.index("--output") + 1]
        assert out.endswith("my_model.pt")  # авто-добавление .pt
