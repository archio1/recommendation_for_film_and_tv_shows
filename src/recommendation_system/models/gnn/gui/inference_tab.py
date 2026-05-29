"""Вкладка «Тестирование» — офлайн-тестер DualDomainEngine (бот в окне)."""

from __future__ import annotations

import asyncio
import os
import queue
import threading
import traceback
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import flet as ft

from recommendation_system.models.gnn.gui.common import _first_existing_ancestor
from recommendation_system.models.gnn.gui.domain_stats import _collect_domain_stats
from recommendation_system.models.gnn.gui.theme import (
    COLORS,
    PROJECT_ROOT,
    accent_button,
    primary_button,
)

if TYPE_CHECKING:
    from recommendation_system.models.gnn.gui.app import TrainerGuiApp


# TV_OFFSET — корректный TMDB-id (TV) хранится в каталоге как +10_000_000.
# Тот же константный сдвиг используется ботом, faiss_bridge и dataset pipeline.
_TV_OFFSET = 10_000_000

_FAISS_INDEX = PROJECT_ROOT / "src" / "recommendation_system" / "faiss_index" / "catalog.faiss"
_FAISS_META = PROJECT_ROOT / "src" / "recommendation_system" / "faiss_index" / "catalog_meta.json"
_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "cache"


def _tmdb_url(item) -> str:
    """Бот-совместимая ссылка: IMDb если есть, иначе themoviedb.org с raw tmdb_id."""
    if getattr(item, "imdb_id", None):
        return f"https://www.imdb.com/title/{item.imdb_id}/"
    tid = int(item.tmdb_id)
    if item.media_type == "tv" and tid >= _TV_OFFSET:
        tid -= _TV_OFFSET
    return f"https://www.themoviedb.org/{item.media_type}/{tid}"


def _display_title(item, lang: str) -> str:
    """Pick localized title field; fall back to default `title` if missing."""
    if lang == "ru":
        return getattr(item, "title_ru", None) or item.title
    if lang == "uk":
        return getattr(item, "title_uk", None) or item.title
    return item.title


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
            expand=True, dense=True, disabled=True,
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
            expand=True, dense=True, disabled=True,
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

        self.recs_movie_btn = primary_button(
            "Рекомендовать фильмы",
            icon=ft.Icons.MOVIE, tooltip="/recs_movie — похожие фильмы",
            on_click=lambda _e: self._on_recs("movie"),
        )
        self.recs_tv_btn = primary_button(
            "Рекомендовать сериалы",
            icon=ft.Icons.TV, tooltip="/recs_tv — похожие сериалы",
            on_click=lambda _e: self._on_recs("tv"),
        )
        self.recs_all_btn = primary_button(
            "Фильмы и сериалы",
            icon=ft.Icons.APPS, tooltip="/recs_all — общая лента",
            on_click=lambda _e: self._on_recs("all"),
        )
        self.recs_cross_btn = accent_button(
            "Кросс-домен",
            icon=ft.Icons.SWAP_HORIZ,
            on_click=lambda _e: self._on_recs("cross"),
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
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
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
