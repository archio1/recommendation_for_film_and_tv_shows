import flet as ft
import threading
import time
from pathlib import Path
import sys
import queue
import json
import asyncio
import torch
import traceback
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent))
from trainer import LightGCNTrainer
from lightgcn import LightGCN
from graph_builder import MovieGraphBuilder


# ==============================================================================
# ЛОГИКА ПРЕДСКАЗАНИЯ (INFERENCE ENGINE)
# ==============================================================================
class InferenceEngine:
    def __init__(self, data_dir: Path, model_path: Path, device='cpu'):
        self.data_dir = data_dir
        self.model_path = model_path
        self.device = device
        self.model = None
        self.metadata = None
        self.id_mapping = None
        self.is_loaded = False

    def load_resources(self):
        """Загружает метаданные и веса модели"""
        try:
            # ИСПРАВЛЕНИЕ: Убрали / 'processed', так как data_dir уже указывает на нее
            # Файлы ищутся прямо в папке data_dir
            with open(self.data_dir / 'id_mapping.json', 'r') as f:
                self.id_mapping = json.load(f)

            self.metadata = pd.read_parquet(self.data_dir / 'items_metadata_final.parquet')

            # Создаем быстрый поиск: Title -> Item ID
            self.title_to_id = pd.Series(
                self.metadata['item_id'].values,
                index=self.metadata['title'].str.lower()
            ).to_dict()

            # 2. Инициализация модели
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint['model_state_dict']

            num_users = state_dict['user_embedding.weight'].shape[0]
            num_items = state_dict['item_embedding.weight'].shape[0]
            embedding_dim = state_dict['user_embedding.weight'].shape[1]

            self.model = LightGCN(num_users, num_items, embedding_dim)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            self.is_loaded = True
            return True, "Модель и данные успешно загружены"
        except Exception as e:
            return False, f"Ошибка загрузки: {e}"

    def search_movies(self, query: str, limit=5):
        """Поиск фильмов по названию"""
        if not self.is_loaded or not query:
            return []

        query = query.lower()
        mask = self.metadata['title'].str.lower().str.contains(query)
        results = self.metadata[mask].head(limit)
        return results[['title', 'year', 'item_id']].to_dict('records')

    def get_recommendations(self, liked_item_ids: list, top_k=10):
        """
        Генерирует рекомендации на основе списка понравившихся фильмов.
        Метод: Усреднение эмбеддингов выбранных фильмов (User Projection).
        """
        if not self.is_loaded or not liked_item_ids:
            return []

        # 1. Получаем эмбеддинги выбранных фильмов
        item_emb = self.model.item_embedding.weight.detach()  # [num_items, dim]

        selected_indices = torch.tensor(liked_item_ids).to(self.device)
        selected_vectors = item_emb[selected_indices]  # [num_liked, dim]

        # 2. Создаем "виртуального пользователя" (среднее арифметическое)
        user_vector = torch.mean(selected_vectors, dim=0).unsqueeze(0)  # [1, dim]

        # 3. Считаем скоры для всех фильмов (Dot Product)
        scores = torch.matmul(user_vector, item_emb.t()).squeeze(0)  # [num_items]

        # 4. Исключаем уже выбранные фильмы (ставим им -inf)
        scores[selected_indices] = -float('inf')

        # 5. Топ-K
        top_scores, top_indices = torch.topk(scores, top_k)
        top_indices = top_indices.cpu().numpy()

        # 6. Достаем информацию о фильмах
        recs = []
        for idx in top_indices:
            row = self.metadata[self.metadata['item_id'] == idx].iloc[0]
            recs.append({
                'title': row['title'],
                'year': row['year'],
                'genres': row['genres'],
                'overview': row['overview']
            })

        return recs


class TrainingGUI:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "🎬 LightGCN — Обучение и Тестирование"
        self.page.window_width = 1500
        self.page.window_height = 980
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self.page.padding = 20
        self.page.bgcolor = ft.Colors.with_opacity(0.98, "#F5F7FA")

        # Состояние обучения
        self.update_queue = queue.Queue()
        self.is_training = False
        self.training_thread = None

        # Состояние инференса
        self.inference_engine = None
        self.selected_movies = []  # Список dict: {'title':..., 'item_id':...}

        # Данные графиков
        self.epochs_data = []
        self.loss_data = []
        self.recall_data = []
        self.ndcg_data = []

        # --- Picker для папки данных ---
        self.file_picker = ft.FilePicker(on_result=self._on_dir_selected)
        self.page.overlay.append(self.file_picker)

        # --- НОВОЕ: Picker для файла модели ---
        self.model_picker = ft.FilePicker(on_result=self._on_model_file_selected)
        self.page.overlay.append(self.model_picker)

        self._init_controls()
        self._build_ui()
        self.page.run_task(self._process_updates)

    def _on_dir_selected(self, e: ft.FilePickerResultEvent):
        if e.path:
            self.data_path_text.value = e.path
            self.data_path_text.color = ft.Colors.BLACK
            self.data_path_icon.name = ft.Icons.FOLDER_OPEN
            self.data_path_icon.color = ft.Colors.BLUE_600
            self.page.update()

    def _init_controls(self):
        # === Вкладка 1: Обучение ===
        self.data_path_text = ft.Text("Автоопределение (папка проекта)", color=ft.Colors.GREY_500, size=14, expand=True)
        self.data_path_icon = ft.Icon(ft.Icons.AUTO_MODE, color=ft.Colors.GREY_400)
        self.btn_select_data = ft.ElevatedButton("Обзор...", icon=ft.Icons.FOLDER,
                                                 on_click=lambda _: self.file_picker.get_directory_path())

        self.embedding_dim_input = ft.TextField(label="Размерность", value="16", width=150, text_size=14)
        self.num_layers_input = ft.TextField(label="Слои GCN", value="1", width=150, text_size=14)
        self.epochs_input = ft.TextField(label="Эпохи", value="20", width=150, text_size=14)
        self.batch_size_input = ft.TextField(label="Батч", value="4096", width=150, text_size=14)
        self.lr_input = ft.TextField(label="LR", value="0.003", width=150, text_size=14)

        self.device_selector = ft.SegmentedButton(
            selected={"cuda" if torch.cuda.is_available() else "cpu"},
            allow_multiple_selection=False,
            segments=[
                ft.Segment(value="cpu", label=ft.Text("CPU"), icon=ft.Icon(ft.Icons.COMPUTER)),
                ft.Segment(value="cuda", label=ft.Text("GPU"), icon=ft.Icon(ft.Icons.BOLT),
                           disabled=not torch.cuda.is_available()),
            ]
        )

        self.start_button = ft.ElevatedButton("Начать обучение", icon=ft.Icons.ROCKET_LAUNCH,
                                              on_click=self._start_training,
                                              style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_600, color=ft.Colors.WHITE),
                                              height=50)
        self.stop_button = ft.ElevatedButton("Стоп", icon=ft.Icons.STOP_CIRCLE, on_click=self._stop_training,
                                             disabled=True,
                                             style=ft.ButtonStyle(bgcolor=ft.Colors.RED_100, color=ft.Colors.RED_700),
                                             height=50)

        # Прогресс
        self.batch_progress_bar = ft.ProgressBar(width=600, height=6, bgcolor=ft.Colors.GREY_200,
                                                 color=ft.Colors.ORANGE_400, value=0)
        self.batch_progress_text = ft.Text("Ожидание...", size=12, color=ft.Colors.GREY_600, font_family="Consolas")
        self.batch_progress_section = ft.Container(
            content=ft.Column([self.batch_progress_text, self.batch_progress_bar], spacing=5),
            margin=ft.margin.only(top=10), visible=False)

        self.progress_bar = ft.ProgressBar(expand=True, color=ft.Colors.BLUE_700, bgcolor=ft.Colors.BLUE_100, height=10,
                                           border_radius=5)
        self.progress_text = ft.Text("Готов к работе", size=16, weight=ft.FontWeight.BOLD,
                                     color=ft.Colors.BLUE_GREY_800)

        # Метрики
        self.epoch_card = self._metric_card("Эпоха", "0/0", ft.Colors.BLUE_500, ft.Icons.CALENDAR_MONTH)
        self.loss_card = self._metric_card("Loss", "0.0000", ft.Colors.ORANGE_500, ft.Icons.TRENDING_DOWN)
        self.recall_card = self._metric_card("Recall@10", "0.0000", ft.Colors.GREEN_500, ft.Icons.STAR)
        self.ndcg_card = self._metric_card("NDCG@10", "0.0000", ft.Colors.PURPLE_500, ft.Icons.SORT)
        self.time_card = self._metric_card("Время", "00:00", ft.Colors.CYAN_500, ft.Icons.TIMER)

        self.chart_text = ft.Text("График...", size=12, font_family="Consolas")
        self.chart_container = ft.Container(content=ft.Column([ft.Text("График", weight="bold"), self.chart_text]),
                                            bgcolor=ft.Colors.WHITE, padding=15, border_radius=12,
                                            shadow=ft.BoxShadow(blur_radius=10,
                                                                color=ft.Colors.with_opacity(0.1, "black")))

        self.log_list = ft.ListView(spacing=5, padding=10, auto_scroll=True, height=300)
        self.log_container = ft.Container(content=self.log_list, bgcolor=ft.Colors.WHITE, padding=15, border_radius=12,
                                          shadow=ft.BoxShadow(blur_radius=10,
                                                              color=ft.Colors.with_opacity(0.1, "black")))

        self.dataset_info = ft.Container(content=ft.Text("Инфо..."), bgcolor=ft.Colors.WHITE, padding=15,
                                         border_radius=12, shadow=ft.BoxShadow(blur_radius=10,
                                                                               color=ft.Colors.with_opacity(0.1,
                                                                                                            "black")))

        # === Вкладка 2: Проверка (Inference) ===
        self.search_field = ft.TextField(
            label="Найти фильм (на английском)",
            prefix_icon=ft.Icons.SEARCH,
            on_change=self._on_search_change,
            expand=True,
            disabled=True  # Сразу выключено, пока модель не загружена
        )
        self.search_results = ft.ListView(height=200, spacing=5)
        self.selected_movies_view = ft.Row(wrap=True, spacing=10)

        self.recommendations_view = ft.ListView(expand=True, spacing=10)

        # --- ИЗМЕНЕННАЯ КНОПКА ---
        self.btn_load_model = ft.ElevatedButton(
            "Выбрать файл модели (.pt)",
            on_click=lambda _: self.model_picker.pick_files(
                allow_multiple=False,
                allowed_extensions=["pt"],
                dialog_title="Выберите файл lightgcn_best.pt"
            ),
            icon=ft.Icons.FOLDER_OPEN
        )
        self.btn_get_recs = ft.ElevatedButton("Получить рекомендации", on_click=self._get_recommendations,
                                              icon=ft.Icons.MOVIE_FILTER, disabled=True)
        self.inference_status = ft.Text("Модель не выбрана", color=ft.Colors.GREY)

    def _metric_card(self, title, value, color, icon):
        return ft.Container(
            content=ft.Row([
                ft.Container(content=ft.Icon(icon, color=color, size=24), padding=10,
                             bgcolor=ft.Colors.with_opacity(0.1, color), border_radius=10),
                ft.Column([ft.Text(title, size=12, color=ft.Colors.GREY_600),
                           ft.Text(value, size=20, weight="bold", color=ft.Colors.BLUE_GREY_900)], spacing=2)
            ]),
            padding=15, bgcolor=ft.Colors.WHITE, border_radius=12,
            shadow=ft.BoxShadow(blur_radius=10, color=ft.Colors.with_opacity(0.05, "black")), width=190
        )

    def _build_ui(self):
        # 1. Training Tab Content
        training_content = ft.Column([
            ft.Container(
                content=ft.Column([
                    ft.Row([ft.Icon(ft.Icons.TUNE, color=ft.Colors.BLUE_700),
                            ft.Text("Настройки обучения", size=20, weight="bold")], spacing=10),
                    ft.Divider(height=10, color="transparent"),
                    ft.Container(
                        content=ft.Row([self.data_path_icon, self.data_path_text, self.btn_select_data],
                                       alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        padding=ft.padding.symmetric(horizontal=15, vertical=8),
                        border=ft.border.all(1, ft.Colors.GREY_300), border_radius=10, bgcolor=ft.Colors.WHITE
                    ),
                    ft.Divider(height=10, color="transparent"),
                    ft.Row([
                        ft.Column([ft.Text("Параметры модели", weight="bold"),
                                   ft.Row([self.embedding_dim_input, self.num_layers_input])]),
                        ft.Column([ft.Text("Гиперпараметры", weight="bold"),
                                   ft.Row([self.epochs_input, self.batch_size_input, self.lr_input])]),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, wrap=True),
                    ft.Divider(height=10, color="transparent"),
                    ft.Row([
                        ft.Column([ft.Text("Устройство", weight="bold"), self.device_selector]),
                        ft.Container(expand=True),
                        self.stop_button, ft.Container(width=10), ft.Container(content=self.start_button, width=200)
                    ], alignment=ft.MainAxisAlignment.END, vertical_alignment=ft.CrossAxisAlignment.END)
                ]),
                padding=25, bgcolor=ft.Colors.WHITE, border_radius=16,
                shadow=ft.BoxShadow(blur_radius=15, color=ft.Colors.with_opacity(0.08, "black"))
            ),
            ft.Container(content=ft.Column(
                [ft.Row([self.progress_text], alignment=ft.MainAxisAlignment.SPACE_BETWEEN), self.progress_bar,
                 self.batch_progress_section]), margin=ft.margin.only(top=20, bottom=20)),
            ft.Row([self.epoch_card, self.loss_card, self.recall_card, self.ndcg_card, self.time_card],
                   alignment=ft.MainAxisAlignment.SPACE_BETWEEN, wrap=True),
            ft.Container(height=20),
            ft.Row([
                ft.Column([
                    ft.Row([ft.Container(content=self.dataset_info, expand=1),
                            ft.Container(content=self.chart_container, expand=2)], expand=True)
                ], expand=2),
                ft.Container(content=self.log_container, width=400, expand=1)
            ], vertical_alignment=ft.CrossAxisAlignment.START, expand=True)
        ], scroll=ft.ScrollMode.AUTO)

        # 2. Inference Tab Content
        inference_content = ft.Container(
            content=ft.Column([
                ft.Container(
                    content=ft.Row([
                        self.btn_load_model,
                        self.inference_status,
                        ft.Container(expand=True),
                        self.btn_get_recs
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    padding=15, bgcolor=ft.Colors.WHITE, border_radius=12
                ),
                ft.Row([
                    # Левая колонка: Поиск и выбор
                    ft.Container(
                        content=ft.Column([
                            ft.Text("1. Выберите фильмы (3-5 шт)", weight="bold", size=16),
                            self.search_field,
                            ft.Container(content=self.search_results, height=150, visible=False,
                                         bgcolor=ft.Colors.WHITE,
                                         border_radius=10,
                                         shadow=ft.BoxShadow(blur_radius=5,
                                                             color=ft.Colors.with_opacity(0.1, "black"))),
                            ft.Text("Выбрано:", color=ft.Colors.GREY),
                            self.selected_movies_view
                        ]),
                        expand=1, padding=20, bgcolor=ft.Colors.WHITE, border_radius=12
                    ),
                    # Правая колонка: Результат
                    ft.Container(
                        content=ft.Column([
                            ft.Text("2. Рекомендации для вас", weight="bold", size=16, color=ft.Colors.BLUE_800),
                            self.recommendations_view
                        ]),
                        expand=1, padding=20, bgcolor=ft.Colors.WHITE, border_radius=12
                    )
                ], expand=True, spacing=20, vertical_alignment=ft.CrossAxisAlignment.START)
            ]),
            padding=20  # Padding теперь у контейнера
        )

        # TABS
        self.tabs = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            tabs=[
                ft.Tab(
                    text="Обучение модели",
                    icon=ft.Icons.MODEL_TRAINING,
                    content=training_content
                ),
                ft.Tab(
                    text="Проверка (Inference)",
                    icon=ft.Icons.MOVIE_FILTER,
                    content=inference_content
                ),
            ],
            expand=True
        )

        self.page.add(self.tabs)

    # --- INFERENCE METHODS ---
    def _on_model_file_selected(self, e: ft.FilePickerResultEvent):
        """Вызывается, когда пользователь выбрал файл в проводнике"""
        if e.files and len(e.files) > 0:
            file_path = Path(e.files[0].path)
            self._execute_model_loading(file_path)

    def _execute_model_loading(self, model_path: Path):
        """Загружает движок с указанным путем к модели"""
        self.inference_status.value = f"Загрузка: {model_path.name}..."
        self.inference_status.color = ft.Colors.ORANGE
        self.page.update()

        try:
            # 1. Данные берем через автопоиск
            # Функция _find_data_directory уже возвращает путь к .../data/processed
            data_dir = self._find_data_directory()

            # ИСПРАВЛЕНИЕ: Проверяем файл прямо в data_dir, без добавления / "processed"
            if not (data_dir / "id_mapping.json").exists():
                raise FileNotFoundError(f"В папке {data_dir} нет id_mapping.json. Проверьте данные.")

            # 2. Инициализация движка
            device = list(self.device_selector.selected)[0]

            self.inference_engine = InferenceEngine(data_dir, model_path, device=device)
            success, msg = self.inference_engine.load_resources()

            if success:
                self.inference_status.value = "✅ Модель успешно загружена!"
                self.inference_status.color = ft.Colors.GREEN
                self.btn_get_recs.disabled = False
                self.search_field.disabled = False
                self.btn_load_model.text = "Выбрать другую модель"
            else:
                self.inference_status.value = f"Ошибка движка: {msg}"
                self.inference_status.color = ft.Colors.RED

        except Exception as ex:
            self.inference_status.value = f"Ошибка: {str(ex)}"
            self.inference_status.color = ft.Colors.RED
            print(traceback.format_exc())

        self.page.update()

    def _on_search_change(self, e):
        if not self.inference_engine or not self.inference_engine.is_loaded:
            return

        query = e.control.value
        if len(query) < 2:
            self.search_results.parent.visible = False
            self.page.update()
            return

        results = self.inference_engine.search_movies(query)

        self.search_results.controls.clear()
        for movie in results:
            self.search_results.controls.append(
                ft.ListTile(
                    title=ft.Text(movie['title'], weight="bold"),
                    subtitle=ft.Text(str(int(movie['year'])) if movie['year'] else "Unknown"),
                    on_click=lambda _, m=movie: self._add_movie(m)
                )
            )

        self.search_results.parent.visible = True
        self.page.update()

    def _add_movie(self, movie):
        # Проверка на дубликаты
        if any(m['item_id'] == movie['item_id'] for m in self.selected_movies):
            return

        self.selected_movies.append(movie)

        # Добавляем chip
        chip = ft.Chip(
            label=ft.Text(movie['title']),
            on_delete=lambda e: self._remove_movie(movie),
            leading=ft.Icon(ft.Icons.MOVIE)
        )
        self.selected_movies_view.controls.append(chip)

        # Очистка поиска
        self.search_field.value = ""
        self.search_results.parent.visible = False
        self.page.update()

    def _remove_movie(self, movie):
        self.selected_movies = [m for m in self.selected_movies if m['item_id'] != movie['item_id']]
        # Перерисовка чипсов (простой способ - очистить и создать заново, или найти и удалить)
        # Для простоты:
        self.selected_movies_view.controls.clear()
        for m in self.selected_movies:
            self.selected_movies_view.controls.append(
                ft.Chip(
                    label=ft.Text(m['title']),
                    on_delete=lambda e, mov=m: self._remove_movie(mov),
                    leading=ft.Icon(ft.Icons.MOVIE)
                )
            )
        self.page.update()

    def _get_recommendations(self, e):
        if not self.selected_movies:
            return

        item_ids = [m['item_id'] for m in self.selected_movies]
        recs = self.inference_engine.get_recommendations(item_ids)

        self.recommendations_view.controls.clear()
        for i, rec in enumerate(recs):
            self.recommendations_view.controls.append(
                ft.Container(
                    content=ft.Row([
                        ft.Text(f"#{i + 1}", size=20, weight="bold", color=ft.Colors.BLUE_200),
                        ft.Icon(ft.Icons.MOVIE_CREATION, color=ft.Colors.BLUE_500),
                        ft.Column([
                            ft.Text(f"{rec['title']} ({int(rec['year']) if rec['year'] else ''})", weight="bold",
                                    size=16),
                            ft.Text(f"{', '.join(rec['genres'][:3])}", color=ft.Colors.GREY_600, size=12),
                            ft.Text(rec['overview'][:100] + "..." if rec['overview'] else "No description", size=12,
                                    italic=True)
                        ], spacing=2, expand=True)
                    ]),
                    padding=10, border=ft.border.all(1, ft.Colors.GREY_200), border_radius=10
                )
            )
        self.page.update()

    # --- EXISTING HELPER METHODS ---
    def _add_log(self, message: str, color=ft.Colors.BLACK):
        timestamp = time.strftime("%H:%M:%S")
        self.log_list.controls.append(
            ft.Text(f"[{timestamp}] {message}", size=13, color=color, font_family="Consolas")
        )
        if len(self.log_list.controls) > 100:
            self.log_list.controls.pop(0)
        self.log_list.scroll_to(offset=-1, duration=100)
        self.page.update()

    def _find_data_directory(self):
        selected_path = self.data_path_text.value
        if "Автоопределение" not in selected_path and selected_path.strip():
            custom_path = Path(selected_path)
            if custom_path.exists():
                self._add_log(f"✓ Выбранная папка: {custom_path}", ft.Colors.GREEN_400)
                return custom_path

        current_file = Path(__file__).resolve()

        # Ищем корень проекта
        current = current_file.parent
        for _ in range(10):
            if (current / "src").exists():
                project_root = current
                break
            if (current / "data" / "processed").exists():
                data_dir = current / "data" / "processed"
                return data_dir
            current = current.parent
        else:
            cwd = Path.cwd()
            possible_paths = [
                cwd / "data" / "processed",
                cwd / "src" / "recommendation_system" / "data" / "processed",
                cwd.parent / "data" / "processed",
            ]
            for path in possible_paths:
                if path.exists():
                    return path

            raise FileNotFoundError("Не удалось найти папку с данными")

        # Если нашли root
        data_dir = project_root / "src" / "recommendation_system" / "data" / "processed"
        if not data_dir.exists():
            data_dir = project_root / "data" / "processed"

        if data_dir.exists():
            return data_dir

        raise FileNotFoundError(f"Папка данных не найдена в {project_root}")

    def _start_training(self, e):
        if self.is_training:
            return

        self.is_training = True
        self.start_button.disabled = True
        self.stop_button.disabled = False
        self.progress_text.value = "Инициализация..."
        self._add_log("🚀 Запуск обучения...", ft.Colors.BLUE_400)
        self.page.update()

        try:
            data_dir = self._find_data_directory()
            required_files = ["interactions_final.parquet", "items_metadata_final.parquet", "id_mapping.json"]
            for file in required_files:
                if not (data_dir / file).exists():
                    raise FileNotFoundError(f"Файл не найден: {file}")
            self._add_log("✅ Файлы найдены", ft.Colors.GREEN_400)

            builder = MovieGraphBuilder(data_dir)
            data = builder.prepare_for_training(test_size=0.2, temporal=False)

            self.update_queue.put({
                'type': 'dataset_info',
                'data': {
                    'num_users': data['num_users'],
                    'num_items': data['num_items'],
                    'train_interactions': len(data['train_df']),
                    'test_interactions': len(data['test_data']['interactions'])
                }
            })

            epochs = int(self.epochs_input.value)
            batch_size = int(self.batch_size_input.value)
            lr = float(self.lr_input.value)
            embedding_dim = int(self.embedding_dim_input.value)
            num_layers = int(self.num_layers_input.value)
            device = list(self.device_selector.selected)[0]

            model = LightGCN(data['num_users'], data['num_items'], embedding_dim, num_layers).to(device)
            self._add_log(f"🔧 Модель: {embedding_dim}D, {num_layers} layers, {device}", ft.Colors.CYAN_400)

            self.trainer = CustomTrainerWithCallback(model, self.update_queue, lambda: self.is_training)
            self.training_thread = threading.Thread(target=self.trainer.train,
                                                    args=(data, epochs, batch_size, lr, 5, 3))
            self.training_thread.daemon = True
            self.training_thread.start()

        except Exception as ex:
            self._add_log(f"❌ ОШИБКА: {str(ex)}", ft.Colors.RED_400)
            print(traceback.format_exc())
            self.is_training = False
            self.start_button.disabled = False
            self.stop_button.disabled = True
            self.progress_text.value = "Ошибка"
            self.page.update()

    def _stop_training(self, e):
        if not self.is_training: return
        self.is_training = False
        self.start_button.disabled = False
        self.stop_button.disabled = True
        self._add_log("⏹️ Остановка...", ft.Colors.ORANGE_400)
        self.page.update()

    async def _process_updates(self):
        while True:
            try:
                if not self.update_queue.empty():
                    update = self.update_queue.get_nowait()

                    if update['type'] == 'dataset_info':
                        info_content = ft.Column([
                            ft.Row([ft.Icon(ft.Icons.DATASET, color=ft.Colors.BLUE_700),
                                    ft.Text("Информация о данных", weight="bold", size=16,
                                            color=ft.Colors.BLUE_GREY_900)]),
                            ft.Divider(),
                            ft.Container(content=ft.Column([
                                ft.Row([ft.Icon(ft.Icons.PEOPLE, size=18, color=ft.Colors.BLUE_500),
                                        ft.Text(f"Пользователей: {update['data']['num_users']:,}", size=14)]),
                                ft.Row([ft.Icon(ft.Icons.MOVIE, size=18, color=ft.Colors.ORANGE_500),
                                        ft.Text(f"Элементов: {update['data']['num_items']:,}", size=14)]),
                                ft.Row([ft.Icon(ft.Icons.SCHOOL, size=18, color=ft.Colors.GREEN_500),
                                        ft.Text(f"Train: {update['data']['train_interactions']:,}", size=14)]),
                                ft.Row([ft.Icon(ft.Icons.SCIENCE, size=18, color=ft.Colors.PURPLE_500),
                                        ft.Text(f"Test: {update['data']['test_interactions']:,}", size=14)]),
                            ], spacing=12), padding=ft.padding.only(top=10))
                        ])
                        self.dataset_info.content = info_content
                        self.page.update()

                    elif update['type'] == 'epoch_update':
                        progress = update['epoch'] / update['total_epochs']
                        self.progress_bar.value = progress
                        self.progress_text.value = f"Эпоха {update['epoch']}/{update['total_epochs']} — {progress * 100:.1f}%"

                        self.epoch_card.content.controls[1].controls[
                            1].value = f"{update['epoch']}/{update['total_epochs']}"
                        self.loss_card.content.controls[1].controls[1].value = f"{update['loss']:.4f}"
                        self.time_card.content.controls[1].controls[1].value = update['time']
                        if 'recall' in update:
                            self.recall_card.content.controls[1].controls[1].value = f"{update['recall']:.4f}"
                            self.ndcg_card.content.controls[1].controls[1].value = f"{update['ndcg']:.4f}"

                        self._update_chart(update)
                        self.page.update()

                    elif update['type'] == 'batch_progress':
                        if not self.batch_progress_section.visible: self.batch_progress_section.visible = True
                        self.batch_progress_bar.value = update['current_batch'] / update['total_batches']
                        self.batch_progress_text.value = f"Training: {int(self.batch_progress_bar.value * 100)}% | {update['current_batch']}/{update['total_batches']} [{update['elapsed_str']}<{update['eta_str']}, {update['speed']:.1f}it/s, loss={update['loss']:.4f}]"
                        self.page.update()

                    elif update['type'] == 'epoch_complete':
                        self.batch_progress_bar.value = 0
                        self.batch_progress_text.value = "Валидация..."
                        self.page.update()

                    elif update['type'] == 'log':
                        self._add_log(update['message'], update.get('color', ft.Colors.BLACK))

                    elif update['type'] == 'training_complete':
                        self.is_training = False
                        self.start_button.disabled = False
                        self.stop_button.disabled = True
                        self.batch_progress_section.visible = False
                        self.progress_text.value = "✅ Завершено!"
                        self._add_log("🎉 Обучение завершено!", ft.Colors.GREEN_600)
                        self.page.update()

                await asyncio.sleep(0.05)
            except Exception:
                await asyncio.sleep(0.1)

    def _update_chart(self, update):
        self.epochs_data.append(update['epoch'])
        self.loss_data.append(update['loss'])
        if 'recall' in update:
            self.recall_data.append(update['recall'])
            self.ndcg_data.append(update['ndcg'])

        # Текстовый график
        txt = f"📉 Loss: {self.loss_data[-1]:.4f}\n"
        txt += "█" * int(20 * (1 - min(self.loss_data[-1], 1.0))) + "░" * (
                20 - int(20 * (1 - min(self.loss_data[-1], 1.0)))) + "\n\n"
        if self.recall_data:
            txt += f"⭐ Recall@10: {self.recall_data[-1]:.4f}\n"
            txt += "█" * int(20 * self.recall_data[-1]) + "░" * (20 - int(20 * self.recall_data[-1])) + "\n\n"
            txt += f"🎯 NDCG@10: {self.ndcg_data[-1]:.4f}\n"
            txt += "█" * int(20 * self.ndcg_data[-1]) + "░" * (20 - int(20 * self.ndcg_data[-1])) + "\n"

        self.chart_text.value = txt
        self.page.update()


class CustomTrainerWithCallback(LightGCNTrainer):
    """Trainer с callback'ами для GUI"""

    def __init__(self, model, update_queue, is_training_flag):
        super().__init__(model)
        self.update_queue = update_queue
        self.is_training_flag = is_training_flag
        self.start_time = time.time()

    def train(self, data, num_epochs, batch_size, lr, eval_every, early_stopping_patience):
        """Метод обучения с обработкой ошибок"""
        try:
            import torch.optim as optim
            from lightgcn import BPRLoss

            # --- 1. ОПРЕДЕЛЯЕМ ПУТЬ ДЛЯ СОХРАНЕНИЯ ---
            # Ищем корень проекта (поднимаемся на 4 уровня вверх от trainer_gui.py)
            project_root = Path(__file__).resolve().parents[4]
            save_dir = project_root / 'models'
            save_dir.mkdir(parents=True, exist_ok=True)
            # -----------------------------------------

            self.update_queue.put({
                'type': 'log',
                'message': f'💾 Модели будут сохраняться в: {save_dir}',
                'color': ft.Colors.BLUE_300
            })

            train_graph = data['train_graph'].to(self.device)
            train_df = data['train_df']
            train_matrix = data['train_matrix']
            test_data = data['test_data']

            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            loss_fn = BPRLoss()

            best_recall = 0
            best_epoch = 0
            patience_counter = 0

            # Подготовка данных для батчей
            num_batches = (len(train_df) + batch_size - 1) // batch_size

            for epoch in range(1, num_epochs + 1):
                if not self.is_training_flag():
                    self.update_queue.put({
                        'type': 'log',
                        'message': '⏹️ Обучение остановлено пользователем',
                        'color': ft.Colors.ORANGE_400
                    })
                    break

                epoch_start = time.time()

                # Тренировка с прогрессом батчей
                train_loss = self._train_epoch_with_progress(
                    train_graph, train_df, train_matrix,
                    optimizer, loss_fn, batch_size, epoch, num_epochs
                )

                # Отправляем сигнал о завершении эпохи
                self.update_queue.put({'type': 'epoch_complete'})

                epoch_time = time.time() - epoch_start
                elapsed_total = time.time() - self.start_time
                time_str = f"{int(elapsed_total // 60):02d}:{int(elapsed_total % 60):02d}"

                update = {
                    'type': 'epoch_update',
                    'epoch': epoch,
                    'total_epochs': num_epochs,
                    'loss': train_loss,
                    'time': time_str
                }

                if epoch % eval_every == 0 or epoch == 1:
                    self.update_queue.put({
                        'type': 'log',
                        'message': f"🔍 Эпоха {epoch}/{num_epochs}: оценка метрик...",
                        'color': ft.Colors.BLUE_200
                    })

                    metrics = self.evaluate(
                        train_graph, test_data, train_matrix,
                        k=10, sample_users=1000
                    )

                    recall = metrics['recall@10']
                    ndcg = metrics['ndcg@10']

                    update['recall'] = recall
                    update['ndcg'] = ndcg

                    self.update_queue.put({
                        'type': 'log',
                        'message': f"✅ Эпоха {epoch}/{num_epochs} | Loss: {train_loss:.4f} | R@10: {recall:.4f} | N@10: {ndcg:.4f}",
                        'color': ft.Colors.GREEN_400
                    })

                    if recall > best_recall:
                        best_recall = recall
                        best_epoch = epoch
                        patience_counter = 0

                        # Определяем полный путь к файлу
                        model_path = save_dir / 'lightgcn_best.pt'

                        # Сохраняем
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'recall@10': recall,
                            'ndcg@10': ndcg,
                            'metrics': metrics
                        }, model_path)

                        # 1. Сообщение о рекорде
                        self.update_queue.put({
                            'type': 'log',
                            'message': f"🏆 Новый рекорд! Recall: {recall:.4f}",
                            'color': ft.Colors.AMBER_600
                        })

                        # 2. Сообщение с путем к файлу (НОВОЕ)
                        self.update_queue.put({
                            'type': 'log',
                            'message': f"💾 Сохранено в: {model_path}",
                            'color': ft.Colors.BLUE_GREY_500
                        })
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            self.update_queue.put({
                                'type': 'log',
                                'message': f"⏹️ Early stopping сработал на эпохе {epoch} (лучший результат на эпохе {best_epoch})",
                                'color': ft.Colors.ORANGE_400
                            })
                            break
                else:
                    self.update_queue.put({
                        'type': 'log',
                        'message': f"📝 Эпоха {epoch}/{num_epochs} | Loss: {train_loss:.4f}",
                        'color': ft.Colors.GREY_600
                    })

                self.update_queue.put(update)

            self.update_queue.put({'type': 'training_complete'})

        except Exception as e:
            self.update_queue.put({
                'type': 'log',
                'message': f"❌ ОШИБКА в обучении: {str(e)}",
                'color': ft.Colors.RED_400
            })
            self.update_queue.put({
                'type': 'error',
                'message': str(e)
            })
            # Вывод трейса в консоль для отладки
            print(traceback.format_exc())

    def _train_epoch_with_progress(self, train_graph, train_df, train_matrix,
                                   optimizer, loss_fn, batch_size, epoch, num_epochs):
        """Обучение эпохи с отправкой детального прогресса в стиле tqdm (GPU OPTIMIZED)"""
        import torch
        import numpy as np
        import time

        self.model.train()
        total_loss = 0.0

        # Подготовка данных
        user_ids = train_df['user_id'].values
        item_ids = train_df['item_id'].values
        num_samples = len(train_df)
        num_batches = (num_samples + batch_size - 1) // batch_size

        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        # Таймер начала эпохи
        epoch_start_time = time.time()

        for batch_idx in range(num_batches):
            if not self.is_training_flag():
                break

            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]

            # 1. Сразу переносим батч на видеокарту
            batch_users = torch.LongTensor(user_ids[batch_indices]).to(self.device)
            batch_pos_items = torch.LongTensor(item_ids[batch_indices]).to(self.device)

            # 2. БЫСТРЫЙ Negative Sampling (прямо на GPU)
            # Вместо медленных циклов Python используем torch.randint
            batch_neg_items = torch.randint(
                0, self.model.num_items,
                (len(batch_users),),
                device=self.device
            )

            optimizer.zero_grad()

            # 3. Forward Pass
            # Передаем граф на устройство
            user_emb, item_emb = self.model(train_graph.edge_index.to(self.device))

            u_emb = user_emb[batch_users]
            pos_emb = item_emb[batch_pos_items]
            neg_emb = item_emb[batch_neg_items]

            # 4. Расчет скоров и Loss
            pos_scores = torch.sum(u_emb * pos_emb, dim=1)
            neg_scores = torch.sum(u_emb * neg_emb, dim=1)

            loss = loss_fn(pos_scores, neg_scores)

            # L2 Regularization
            l2_reg = 1e-4 * (u_emb.norm(2).pow(2) +
                             pos_emb.norm(2).pow(2) +
                             neg_emb.norm(2).pow(2)) / float(len(batch_users))

            loss = loss + l2_reg

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # --- ЛОГИКА PROGRESS BAR ---
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches:
                current_time = time.time()
                elapsed = current_time - epoch_start_time
                processed = batch_idx + 1
                speed = processed / elapsed if elapsed > 0 else 0

                remaining_batches = num_batches - processed
                eta_seconds = remaining_batches / speed if speed > 0 else 0

                elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
                eta_str = f"{int(eta_seconds // 60):02d}:{int(eta_seconds % 60):02d}"

                self.update_queue.put({
                    'type': 'batch_progress',
                    'current_batch': processed,
                    'total_batches': num_batches,
                    'elapsed_str': elapsed_str,
                    'eta_str': eta_str,
                    'speed': speed,
                    'loss': total_loss / processed
                })

        return total_loss / num_batches


def main(page: ft.Page):
    """Точка входа в приложение"""
    app = TrainingGUI(page)


if __name__ == "__main__":
    ft.app(target=main)
