# Data Sources — Setup & Schema Reference

Все датасеты хранятся в `data/raw/`. Это руководство объясняет, какие источники нужны для какой модели, где их взять, и какой формат CSV ожидает пайплайн `make_dataset.py`.

При сборке через GUI (`python -m recommendation_system.models.gnn.trainer_gui` → вкладка «Создание датасета») отсутствие обязательного источника блокируется pre-flight проверкой со ссылкой на этот документ.

---

## TL;DR — что нужно

| Источник | Модель | Обязательность | Размер | Где взять |
|---|---|---|---|---|
| MovieLens 32M | movies | **Required** | ~1.1 GB | [grouplens.org/datasets/movielens/32m](https://grouplens.org/datasets/movielens/32m/) |
| TMDB metadata v11 | movies | Optional | ~400 MB | Kaggle: «TMDB Movies Dataset 2024» |
| Trakt shows CSV | tv | **Required** | ~5 MB | `trakt_collector.py` (~2 суток) или сторонний CSV |
| Trakt interactions CSV | tv | **Required** | ~80 MB | то же |
| Amazon Reviews 2023 | movies/tv | Optional | ~100 GB | [amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io/) |

---

## Movies pipeline

### MovieLens 32M — **Required**

GroupLens MovieLens 32M даёт основную user-item матрицу для модели фильмов. Без него пайплайн `build_movie_dataset()` упадёт на чтении `ratings.csv` (см. `make_dataset.py:312-327` — read без `exists()` проверки).

**Как установить:**

1. Скачать `ml-32m.zip` (~250 MB) с https://grouplens.org/datasets/movielens/32m/.
2. Распаковать в `data/raw/ml-32m/`. Ожидаемая структура:
   ```
   data/raw/ml-32m/
     ├── ratings.csv     (~870 MB, ~32M строк)
     ├── movies.csv      (~3 MB, ~87K фильмов)
     ├── links.csv       (~1.6 MB, movieId ↔ tmdbId)
     ├── tags.csv        (используется опционально)
     ├── README.txt
     └── checksums.txt
   ```
3. Проверить (в GUI → DatasetTab → Источники → MovieLens dir): должна быть ✓ зелёная.

**Схема (всё это GroupLens задаёт сам — менять не нужно):**

- `ratings.csv`: `userId`, `movieId`, `rating` (0.5–5.0), `timestamp` (epoch).
- `movies.csv`: `movieId`, `title`, `genres` (pipe-separated).
- `links.csv`: `movieId`, `imdbId`, `tmdbId` (может быть NaN).

### TMDB metadata v11 — Optional

Расширяет метаданные фильмов (overview, keywords, popularity, vote_count) и улучшает классификацию жанров. Без него movies-пайплайн работает, но без обогащённых метаданных.

**Как установить:** скачать с Kaggle (поиск «TMDB Movies Dataset» v11) → `data/raw/TMDB_movie_dataset_v11.csv`.

Reference: `make_dataset.py:clean_tmdb_movies` (строки 586+).

---

## TV pipeline

### Trakt CSV — **Required**

Trakt — основная user-item матрица для TV-модели. Есть два способа получить:

#### (a) Самосбор через `trakt_collector.py` (≈2 суток)

```bash
python -m recommendation_system.data.trakt_collector
```

Скрипт идёт через Trakt API с rate-limit'ами; собирает популярные шоу + ratings ~80K пользователей. Промежуточное состояние сохраняется в `data/raw/trakt_collector.db`, можно прерывать и возобновлять.

Результат — два файла в `data/raw/`:
- `trakt_shows.csv`
- `trakt_interactions.csv`

Это рекомендованный способ для воспроизводимости.

#### (b) Подключение готового CSV — schema-контракт

Если у вас есть готовый TV-датасет (Kaggle, IMDb dump, чужой Trakt-snapshot), подключите его через GUI → DatasetTab → ExpansionTile «Источники» → FilePicker на полях `Trakt shows CSV` и `Trakt interactions CSV`.

CSV должен соответствовать следующему контракту, иначе пайплайн упадёт или вернёт пустой DataFrame.

##### `trakt_shows.csv`

Reference: `make_dataset.py:load_trakt_metadata` (строки 492–551).

| Колонка | Тип | Обязательность | Замечания |
|---|---|---|---|
| `tmdb_id` | int | **Required** | TMDB show ID, **без** offset (+10M добавляется автоматически) |
| `title` | str | **Required** | оригинальное название |
| `year` | int | **Required** | год премьеры; NaN → строка отбрасывается |
| `genres` | str | **Required** | список жанров: comma-separated (`"Drama, Sci-Fi"`) или JSON-list (`'["Drama"]'`); пустые отбрасываются |
| `overview` | str | **Required** | описание; короче 10 символов → строка отбрасывается |
| `language` | str (ISO 639-1) | используется если есть | фильтр `languages` из GUI применяется только если колонка существует |
| `vote_average` | float | Optional (default 0.0) | TMDB-style рейтинг 0–10 |
| `vote_count` | int | Optional (default 0) | используется в фильтре «non-target language ≥ 100 votes» |
| `popularity` | float | Optional (default 0.0) | используется для top-N сортировки |
| `title_ru` | str | Optional (default NULL) | RU-перевод названия для bilingual UI |

Минимальный валидный пример:
```csv
tmdb_id,title,year,genres,overview,language,vote_average,vote_count,popularity
1399,Game of Thrones,2011,"Drama, Fantasy","Seven noble families fight for control of the mythical land of Westeros.",en,8.4,21000,500.0
1396,Breaking Bad,2008,"Drama, Crime","A high school chemistry teacher diagnosed with cancer turns to manufacturing meth.",en,8.9,11000,400.0
```

##### `trakt_interactions.csv`

Reference: `make_dataset.py:load_trakt_interactions` (строки 553–580).

| Колонка | Тип | Обязательность | Замечания |
|---|---|---|---|
| `user_id` | uint32 | **Required** | произвольные неотрицательные целые; уникальность не требуется (за rating'ом) |
| `tmdb_id` | uint32 | **Required** | **БЕЗ** +10M offset — добавится в pipeline (строка 573) |
| `rating` | float32 | **Required** | шкала 0.5–5.0 (MovieLens-совместимая) |
| `timestamp` | uint32 | **Required** | unix epoch в секундах |

Минимальный валидный пример:
```csv
user_id,tmdb_id,rating,timestamp
1,1399,5.0,1604188800
1,1396,4.5,1604189000
2,1399,4.0,1604190000
```

Строки с `tmdb_id`, отсутствующими в `trakt_shows.csv`, будут отброшены при join (строка 577).

---

## Amazon Reviews 2023 — Optional (кросс-доменный)

Очень большой датасет (~100 GB) с user-item-rating взаимодействиями по фильмам и сериалам с Amazon Prime. Если подключён — добавляет десятки миллионов взаимодействий поверх MovieLens / Trakt.

**Как использовать:**
1. Скачать [Movies_and_TV](https://amazon-reviews-2023.github.io/) (метаданные + ревью), распаковать.
2. Структура папки:
   ```
   <amazon_dir>/
     ├── meta_Movies_and_TV.jsonl     (~3 GB)
     └── Movies_and_TV.jsonl          (~90 GB)
   ```
3. В GUI → DatasetTab → ExpansionTile «Источники» → поле «Amazon dir» указать путь к папке.

Если файлы отсутствуют — пайплайн делает graceful fallback (просто пропускает Amazon, `make_dataset.py:357-359`). На иконке источника покажется ⚠ amber (optional missing).

Mapping ASIN → tmdb_id идёт по очищенному title (см. `load_amazon_interactions`, строки 347+).

---

## Подмена источников из GUI

Любой из источников можно переопределить через GUI без редактирования `make_dataset.py`:

1. Запустить `python -m recommendation_system.models.gnn.trainer_gui`.
2. Вкладка «Создание датасета».
3. Развернуть `ExpansionTile «Источники»`.
4. Для нужного поля нажать иконку папки/файла → FilePicker.
5. Иконка справа (`✓` / `✗` / `⚠`) мгновенно подтверждает наличие.

Пустое поле = использовать дефолт из `data/raw/...`. Подменённый путь — сохранится только на текущий запуск сборки (не записывается в файл настроек).

---

## Smoke vs Full preset

Полная сборка movies+tv требует много памяти и времени. На локальном железе с ограниченным RAM/VRAM выбирайте подходящий preset в `ExpansionTile «Параметры»`:

| Preset | top_n_movies | top_n_tv | min_year | rating_threshold | max_interactions | Время сборки | Куда годится |
|---|---|---|---|---|---|---|---|
| **Smoke** | 500 | 200 | 2010 | 3.5 | (без лимита) | <5 мин | проверка пайплайна, отладка кода |
| **Default** | 15 000 | 10 000 | (нет) | 3.5 | 15 000 000 | 20–40 мин | боевая сборка (текущие хардкоды) |
| **Full** | (все) | (все) | (нет) | (все) | (все) | 1–3 ч + ≥16 GB RAM | максимальное покрытие |
| **Custom** | — | — | — | — | — | — | ручная правка под задачу |

**Локальное железо** (`RTX 4060 Ti 8GB`): использовать Smoke preset — полный movies-датасет (16M рёбер) на этом GPU HW-bound (~10 ч/эпоха), Smoke даёт ~1 мин/эпоха. Полные тренировки — Colab.

После смены любого из 8 параметров вручную preset автоматически переключается в `Custom`.

---

## Что не делаем (явно)

- Auto-download MovieLens/TMDB — только инструкции и ссылки в этом файле.
- Замена MovieLens на «свой источник» для movies pipeline — структурно невозможно без переписывания `load_data()` и `link_to_movielens()`. MovieLens нужен и за тем, что `links.csv` даёт MovieID → TMDBID маппинг.
- Кнопка запуска `trakt_collector.py` из GUI — сбор идёт ~2 суток с rate-limit'ами Trakt API, не вписывается в GUI-сценарий.
- Валидация schema-контракта при подмене CSV — пользователь сам соблюдает; ошибки увидите в логах сборки (вкладка «Создание датасета», панель «Логи»).
