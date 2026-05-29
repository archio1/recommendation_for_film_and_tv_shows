# Руководство по `trainer_gui` — административный GUI для dual-LightGCN стека

> **Версия:** 2026-05-12 (после реализации `spec/local-training.md`)
> **Файлы:** точка входа `src/recommendation_system/models/gnn/trainer_gui.py` (тонкий шим); реализация — подпакет `src/recommendation_system/models/gnn/gui/` (`theme`, `common`, `domain_stats`, `training_tab`, `dataset_tab`, `inference_tab`, `data_tab`, `app`)
> **Аудитория:** разработчики/админы системы. Это **не** end-user UI — для пользователей есть Telegram-бот (`movie_bot.py`).

## Что это и зачем

GUI на Flet (desktop), 4 вкладки. Один процесс — одна установка стека. Используется для:

- локального обучения LightGCN-моделей вместо Colab (для итераций по гиперпараметрам / починки регрессий);
- сборки датасетов через UI вместо CLI;
- офлайн-тестирования рекомендаций (тот же `DualDomainEngine`, что и в боте, но без Telegram);
- быстрого взгляда на состояние датасетов и моделей.

**Что GUI НЕ делает:**

- не запускает Trakt-collector (это 2 суток сбора, отдельный скрипт);
- не пушит модели в бот / production — это ручной шаг;
- не делает hyperparameter sweep — крутите параметры вручную;
- не имеет hot-swap моделей: при смене device или checkpoint — перезапуск GUI;
- не отменяет сборку датасета (`make_dataset` не поддерживает cancel-флаг).

---

## Запуск

```powershell
# Из корня репозитория. PYTHONPATH=src обязателен.
$env:PYTHONPATH = "src"; python -m recommendation_system.models.gnn.trainer_gui
```

При старте GUI:

1. Импортирует `flet`, `torch` (5-10с холодный старт).
2. Создаёт 4 вкладки. **Движки моделей НЕ загружаются** — это происходит лениво в Tab 3 при первом запросе.
3. Tab 4 (Данные) сразу читает parquet/sidecar и показывает статистику.

Окно открывается ~10-15с от запуска команды. Если не открылось — проверь `python` процесс в `tasklist`.

---

## Общий header

Сверху над вкладками — два глобальных контрола:

### Domain switcher (Dropdown)

`movies` / `tv`. Используется **только** Tab 1 (Обучение) — определяет какой датасет/модель будут собираться. Tab 2 (Dataset) и Tab 3 (Inference) имеют **свои** локальные dropdown'ы. Tab 4 (Data) всегда показывает оба домена.

### Device selector (SegmentedButton)

CPU / GPU. GPU-сегмент disabled, если `torch.cuda.is_available() = False`.

- **Tab 1:** значение передаётся в `trainer.main(--device=...)` без override. CPU = ~10× медленнее GPU.
- **Tab 3:** значение фиксируется при **первой** загрузке движков. Сменишь после — перезапусти GUI.
- Tab 2 / Tab 4: device не используется.

---

## Tab 1 «Обучение»

Wrap вокруг `trainer.main()`. Реализация: `gui/training_tab.py:59` (`TrainingTab`).

### Когда использовать

- Натренировать новую версию модели локально (Colab больше не обязателен).
- Подобрать гиперпараметры (быстрее чем по очереди в Colab).
- Починить регрессию: «после правки `lightgcn.py` старый чекпоинт всё ещё работает, новый — нет».

### Поля гиперпараметров

| Поле | Default | Что меняет |
|---|---|---|
| **Эпохи** | 30 | Максимум эпох. Early stopping (`patience`) обычно режет раньше. |
| **Батч** | 2048 | Размер минибатча. На CPU — поставь 512; на GPU 12GB — 2048 ок; 24GB — 4096. |
| **LR** | 0.002 | Learning rate. Стандартный Adam, не трогай если не знаешь зачем. |
| **Embedding** | 32 | Размерность user/item векторов. 16-64 разумно; 128 → переобучение на нашем размере. |
| **Слои GCN** | 2 | Глубина LightGCN. 1-3 разумно; >4 — overshoot, размывание сигнала. |
| **Patience** | 3 | Сколько эпох ждать улучшения recall@10 до early-stop. |
| **Eval every** | 5 | Каждые N эпох считать recall/ndcg на val. Дороже → реже. |

Тултипы на полях показывают defaults; при сомнении — оставь как есть.

### Старт / Стоп

- **Старт** запускает `trainer.main(argv, on_epoch_end=..., stop_flag=...)` в `threading.Thread`. UI остаётся отзывчивым.
- **Стоп** ставит флаг → next epoch прерывается. Уже завершённые эпохи сохраняются в sidecar.

### Что показывают метрики и чарт

| Метрика | Что это |
|---|---|
| **Эпоха** | текущая / всего. Прогресс-бар по эпохам. |
| **Лучшая эпоха** | эпоха с максимальным recall@10. Чекпоинт сохраняется именно с этой эпохи. |
| **Loss** | BPR loss на батче. Должен монотонно падать; рост = LR слишком высокий. |
| **Recall@10 / NDCG@10** | измеряются раз в `eval_every` эпох на validation split. |
| **Loss-чарт** | каждая точка = эпоха. Если флэт после первых 5 — модель не учится (слишком низкий LR / битый граф). |

### Sanity-check и баннер результата

После train `trainer.main` валидирует модель:

- `recall@10 > 0.05` (низкий threshold; реальные модели > 0.2).
- `||user_emb||.mean() > 0.01` (эмбеддинги не все нули).
- `||item_emb||.mean() > 0.01`.

Результат:

- **Зелёный** «sanity-check пройден» → модель ок, sidecar записан, auto-bump указывает на новый чекпоинт.
- **Красный** «sanity-check ПРОВАЛЕН (см. sidecar)» → exit code 1, sidecar **записан** с `passed=false` и списком failures. Не используйте такую модель в боте.
- **Красный** «Ошибка обучения (rc=...)» → exception. Смотри лог внизу.

### Sidecar JSON

После каждого train рядом с `.pt` пишется JSON-сосед — `lightgcn_{domain}_best_v{N}.json`:

```json
{
  "checkpoint": "lightgcn_movies_best_v5.pt",
  "domain": "movies",
  "device": "cuda",
  "trained_at": "2026-05-12T14:30:00Z",
  "epochs_run": 28,
  "best_epoch": 22,
  "metrics": {"recall@10": 0.234, "ndcg@10": 0.187},
  "dataset": {"users": 145032, "items": 18124, "interactions": 23900000},
  "hyperparameters": {"lr": 0.002, "batch_size": 2048, ...},
  "sanity_check": {"passed": true, "failures": []}
}
```

Кнопка «Открыть sidecar.json» — открывает в текущей системной программе (Notepad / VS Code). Используй чтобы сверить гиперы между запусками.

### Auto-bump версий

`trainer.py` сканирует `models/{domain}/lightgcn_{domain}_best_v*.pt` и подбирает следующий `vN+1`. Старые версии **не удаляются** — чисти руками когда диск кончается.

### Логи

Внизу вкладки — окно с логами `trainer.py` через `logging.Handler`. Ошибки красные, warnings жёлтые. До 500 строк (старые удаляются автоматом).

---

## Tab 2 «Создание датасета»

Wrap вокруг `MovieDatasetProcessor.build_movie_dataset()` / `.build_tv_dataset()` (`make_dataset.py:1238, 1289`). Реализация: `gui/dataset_tab.py:170` (`DatasetTab`).

### Когда использовать

- После сбора свежих Trakt-данных (`trakt_*.csv` обновлены).
- После изменения схемы parquet (новые колонки, bilingual поля).
- При первом разворачивании проекта на новой машине.

Обычная итерация — раз в недели, не в день.

### Источники (readonly)

Показывает где лежат raw-данные:

- **Movies:** `data/raw/ml-32m/` (MovieLens 32M) + `data/raw/TMDB_movie_dataset_v11.csv`
- **TV:** `data/raw/trakt_shows.csv` + `data/raw/trakt_interactions.csv`
- **Amazon (опционально):** см. поле ниже.

### Domain dropdown (локальный)

`movies` / `tv` / `all`. **Не использует** глобальный domain из header (там есть только movies/tv).

- `all` → последовательно: сначала movies, потом tv. Если movies упадёт — tv не запустится.

### Amazon dir

Default `D:/amazon_data`. Если папки нет — пайплайн делает graceful fallback (без Amazon-интеракций, просто лог `warning`). Чтобы выключить совсем — оставь поле пустым.

### Прогресс и логи

- **ProgressBar — indeterminate** (бегущая полоса). `make_dataset` не отдаёт numeric callbacks, поэтому % посчитать нельзя.
- **Status text** — короткий статус (последняя «стадия»: `Сборка movies...` / `Сборка tv...`).
- **Логи** — реальный вывод `make_dataset.logger`. Видно прогресс через INFO-сообщения.

Сборка movies на полном датасете занимает ~30-60 минут (TMDB API hits для keywords + filter + k-core). TV — быстрее. Кнопка «Стоп» декоративная — `make_dataset` не имеет cancel-флага. **Если запустил по ошибке — придётся ждать или убить процесс целиком.**

### Sanity-check после сборки

После успешного return `True` запускается `_dataset_sanity(domain)` (`gui/dataset_tab.py:34`):

- `interactions_final.parquet`, `items_metadata_final.parquet`, `id_mapping.json` существуют.
- `interactions_final.parquet`: ≥ 100k строк (иначе warning «smoke-run?»).
- `items_metadata_final.parquet`: колонки `tmdb_id`, `title`, `genres` присутствуют; нет дубликатов `tmdb_id`.
- `id_mapping.json`: `num_users` и `num_items` (или `num_trained_items`) — int > 0.

Результат:

- **Зелёный** «✅ Sanity-check пройден» → используй датасет.
- **Красный** «⚠ Sanity-check ПРОВАЛЕН» с конкретным списком failures → не доверяй датасету, разбирайся.

### Автообновление Tab 4

После успеха `data_tab.refresh()` вызывается автоматически — переключись на Tab 4, увидишь свежие mtime/числа.

---

## Tab 3 «Тестирование (Inference)»

Офлайн-аналог бота: тот же `DualDomainEngine` + `UniversalSearchEngine`, без Telegram. Реализация: `gui/inference_tab.py:57` (`InferenceTab`).

### Когда использовать

- После train новой модели — проверить, что recommendations не сломались.
- При отладке `DualDomainEngine` (cross-domain, FAISS, cold-start).
- Для bilingual-проверки: «нашёл ли поиск Володар Перснів».
- Для smoke-теста перед деплоем в бот.

### Layout

Левая колонка (≈40%): **Поиск + Избранное (seed)**.
Правая (≈60%): **Рекомендации + 4 кнопки + статус**.

### Lazy-init движков

Самое важное: при создании Tab движки **НЕ загружаются** (это 30-60с на CPU, ~30с на GPU). Загрузка стартует при первом нажатии:

- любой кнопки `/recs_*` (если ещё не загружено), или
- Поиска (если активный domain ещё не загружен).

В правом нижнем углу — статус `router_status`:

- `⏳ Движки не загружены (загрузятся по первому /recs_*)` — начальное состояние.
- `⏳ Movies: загрузка checkpoint...` → `⏳ TV: ...` → `⏳ FAISS: ...` — прогресс.
- `✅ Движки загружены` — готово.
- `❌ Ошибка загрузки движков` — traceback в области рекомендаций.

Все 4 кнопки disabled во время загрузки. Device берётся из header **в момент** загрузки.

### Выбор чекпоинтов

`InferenceTab` вызывает `_collect_domain_stats(domain).last_train_checkpoint` (см. Tab 4) — берёт самый свежий sidecar JSON и читает оттуда имя `.pt`. Fallback — хардкод `lightgcn_{domain}_best_v4.pt` (то же, что использует `movie_bot.py`).

Чтобы InferenceTab подхватил новую модель: натренируй на Tab 1 → закрой GUI → открой заново. Hot-swap не поддерживается.

### Поиск (bilingual)

- **Dropdown «Каталог»:** `movies` или `tv` — определяет в каком каталоге искать.
- **Поле поиска:** EN / RU / UK любой строкой.
- **Кнопка 🔍 / Enter** → `engine.search(query, limit=20)`.

`UniversalSearchEngine.search` уже умеет искать по `title`, `title_ru`, `title_uk` параллельно (`universal_search.py:627-...`). То есть «Володар Перснів», «Властелин колец», «Lord of the Rings» — все найдут один и тот же фильм.

Результаты — список строк с эмодзи (🎬 movie / 📺 tv), кнопка ★ добавляет в избранное.

### Избранное (seed)

Список tmdb_id с title и media_type. Используется как **seed** для всех `/recs_*` кнопок. Дубли отсекаются. Кнопка × убирает.

Хранится только в памяти процесса — при перезапуске GUI обнуляется (намеренно: это admin-инструмент, не сессия пользователя).

### 4 кнопки рекомендаций

| Кнопка | Что делает | Что нужно в избранном |
|---|---|---|
| **/recs_movie** | `router.recs_movie(favorites, top_k=8)` | хотя бы 1 фильм |
| **/recs_tv** | `router.recs_tv(favorites, top_k=8)` | хотя бы 1 сериал |
| **/recs_all** | `router.recs_all(favorites, top_k=8)` | любое |
| **/recs_cross** | `router.recs_cross(favorites, target_media_type=...)` | любое + Dropdown «Cross target» (`movie` / `tv`) |

Семантика **идентична** командам в Telegram-боте (`movie_bot.py:790-862`). Если результат отличается от бота — значит баг в самом боте, не в GUI.

### Cross-domain (`/recs_cross`)

Требует FAISS индекс:

- `src/recommendation_system/faiss_index/catalog.faiss`
- `src/recommendation_system/faiss_index/catalog_meta.json`

Если файлов нет:

- При старте Tab 3 показывается **жёлтый баннер** с инструкцией.
- Кнопка `/recs_cross` disabled (tooltip объясняет почему).

Собрать индекс:

```powershell
$env:PYTHONPATH = "src"; python -m recommendation_system.models.gnn.compute_embeddings --to-faiss
```

После сборки — **перезапусти GUI** (Tab 3 проверяет наличие индекса только при создании).

### Lang switcher

Dropdown «Язык названий»: `en` / `ru` / `uk`. Переключает поле, которое отображается в результатах и избранном:

- `en` → `item.title`
- `ru` → `item.title_ru` (fallback `title` если пусто)
- `uk` → `item.title_uk` (fallback `title` если пусто)

Поиск работает на **всех языках одновременно** независимо от switcher'а — switcher только про отображение.

Search-list **не перерисовывается** при смене языка (не храним source items). Чтобы увидеть переключение в поиске — повтори запрос.

### Карточка результата

- Title + год (в выбранном языке).
- Badge: 🎬 Movies / 📺 TV.
- Жанры (до 5 первых).
- TMDB rating + source (`trained` / `catalog` / `cold_start`).

### Типичные граблиf

- **«Сначала добавьте элементы в избранное»** → пустое seed.
- **«Получено 0 рекомендаций»** → tmdb_id из favorites не известны движку соответствующего домена (например `/recs_movie` с TV-сидами).
- **«Ошибка загрузки движков»** → `checkpoint not found` (`v4.pt` отсутствует) или `dataset not built` (parquet нет). Возвращайся на Tab 1 или Tab 2.

---

## Tab 4 «Данные»

Readonly dashboard. Реализация: `gui/data_tab.py:26` (`DataTab`) + `gui/domain_stats.py:138` (`_collect_domain_stats`).

### Что показывает

Две карточки бок о бок: 🎬 Movies и 📺 TV. Каждая:

**Блок «Датасет»:**
- Users / Items / Interactions (из `id_mapping.json` + parquet metadata).
- `interactions mtime` / `items mtime` (когда parquet был перезаписан).

**Блок «Последняя тренировка»** (читает свежайший sidecar JSON из `models/{domain}/`):
- `trained at` (ISO timestamp).
- `checkpoint` (имя `.pt`).
- `recall@10`.
- Chip: **sanity OK** / **sanity FAIL**.

Если sidecar нет (модель тренировалась на Colab без `local-training` инфры) — «Sidecar не найден — модель не обучалась локально». Это нормально для исторических v1-v4.

**Кнопки:**
- «Папка датасета» / «Папка моделей» → открывает в Explorer.

**Refresh (↻ вверху):** обновляет всё. Tab 2 после успешной сборки вызывает refresh автоматически — кнопка нужна только если меняешь что-то снаружи.

### Если датасет не собран

Карточка показывает «⚠ Датасет не создан» + ожидаемый путь + подсказку «Создайте на вкладке «Создание датасета»». Сборка через Tab 2.

---

## Типичные сценарии end-to-end

### Сценарий 1: «Натренировал новую модель — хочу проверить»

1. **Tab 1**: domain=movies, epochs=30, device=GPU → Старт.
2. Дождись зелёного баннера → запомни путь к `.pt` (или открой sidecar).
3. **Закрой GUI и открой заново** (Tab 3 загружает чекпоинт только при первом запуске).
4. **Tab 4**: проверь карточку Movies → «Последняя тренировка» = свежая, chip = sanity OK.
5. **Tab 3**: добавь 3-5 знакомых фильмов в избранное → /recs_movie → сравни глазами с тем, что раньше выдавала старая модель.
6. Если ок — выкатывай в бот (отдельный шаг, не из GUI).

### Сценарий 2: «Свежий Trakt-collect — пересобираю TV-датасет»

1. **Tab 2**: domain=tv, Amazon dir = default → Собрать.
2. Жди (TV ~10-20 минут). Зелёный баннер «Sanity OK».
3. **Tab 4**: Movies-карточка не изменилась; TV-карточка показывает свежие числа.
4. **Tab 1**: domain=tv, device=GPU → Старт (новый чекпоинт TV нужен потому что catalog поменялся).
5. (Опционально) **Tab 2**: domain=movies → Собрать заново, потом Tab 1 movies. Делать **только** если структура изменилась.
6. Перезапусти GUI и проверь через Tab 3 (см. Сценарий 1).

### Сценарий 3: «Сравнить две версии модели»

GUI этого не делает прямо. Workflow:

1. Сделай первый train (v5) на Tab 1.
2. Скопируй sidecar v5 → блокнот: запомни `recall@10`.
3. Поменяй гиперы → второй train (v6).
4. Открой sidecar v6 → сравни глазами.
5. **Tab 3** покажет рекомендации только от **последней** модели (latest sidecar). Чтобы протестировать v5 — временно удали sidecar v6.

Полноценное сравнение → `spec/retrain-pipeline.md` или custom скрипт.

### Сценарий 4: «GUI открылся, но Tab 3 не загружает движки»

Симптом: красное «Ошибка загрузки движков», в логе traceback.

- **`FileNotFoundError: checkpoint not found`** → нет `lightgcn_{domain}_best_v4.pt` ИЛИ latest sidecar указывает на отсутствующий файл. Удали битый sidecar / положи v4.pt.
- **`FileNotFoundError: dataset not built`** → пусто в `data/processed/{domain}/`. Tab 2 → Собрать.
- **`RuntimeError: InferenceEngine.movies: <error>`** → checkpoint несовместим с текущим `lightgcn.py` (поменялась архитектура модели). Сравни state_dict keys; пересобери модель свежим train.
- **`FaissCatalog not found`** → запусти `compute_embeddings --to-faiss`, перезапусти GUI.

---

## Ограничения и риски

1. **Hot-swap моделей не поддерживается.** Чтобы InferenceTab подхватил свежий чекпоинт — закрой и открой GUI.
2. **Device фиксируется при первом lazy-load в Tab 3.** Сменишь header device после — реально движок останется на старом.
3. **Сборка датасета не прерывается** (Tab 2). Запустил по ошибке → ждёшь или убиваешь процесс.
4. **Concurrency:** не запускай Tab 1 (train) и Tab 2 (build) одновременно — оба тяжёлые. Tab 3 (inference) на момент train конфликтует за GPU, на момент build — за диск.
5. **Логи только в окне GUI** (Tab 1, Tab 2). При крэше окна — теряются. Для архивации используй sidecar (Tab 1) или `make_dataset.log` (Tab 2).

---

## Ссылки на код

| Что | Файл / строки |
|---|---|
| Точка входа GUI | `gui/app.py:144 main()` (+ шим `trainer_gui.py`) |
| Дизайн-токены / фабрики кнопок | `gui/theme.py` (`COLORS`, `BUTTON_HEIGHT`, `CONTENT_MAX_WIDTH`, `primary/danger/accent/neutral/secondary_button`) |
| Tab 1 «Обучение» | `gui/training_tab.py:59 TrainingTab` |
| Tab 2 «Создание датасета» | `gui/dataset_tab.py:170 DatasetTab` |
| Tab 3 «Тестирование» | `gui/inference_tab.py:57 InferenceTab` |
| Tab 4 «Данные» | `gui/data_tab.py:26 DataTab` |
| Sanity-check датасета | `gui/dataset_tab.py:34 _dataset_sanity()` |
| Stat collector для Tab 4 | `gui/domain_stats.py:138 _collect_domain_stats()` |
| Логи → UI bridge | `gui/common.py:33 _QueueLogHandler` |
| Backend CLI | `trainer.py:445-635 main()` |
| Backend dataset | `make_dataset.py:127 MovieDatasetProcessor` |
| Inference loader | `inference_engine.py:31 InferenceEngine` |
| Router | `dual_domain_engine.py:32 DualDomainEngine` |

## См. также

- `spec/local-training.md` — спецификация, по которой собран GUI.
- `spec/two-model-architecture.md` — почему movies и tv разделены.
- `spec/retrain-pipeline.md` (будущее) — что будет вместо ручного train в Tab 1.
- `movie_bot.py` — production-вариант того же `DualDomainEngine`.
