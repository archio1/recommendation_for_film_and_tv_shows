import pytest
import pandas as pd
from src.recommendation_system.models.gnn.bilingual_utils import (
    ScalableMovieIntelligence,
    TMDBTranslationCache,
    GENRE_EN_TO_RU,
    GENRE_EN_TO_UK,
)
#ScalableMovieIntelligence вынести в confest

# --- ТЕСТЫ ЛИНГВИСТИКИ И НОРМАЛИЗАЦИИ ---

def test_cyrillic_detection():
    """Проверяем, что код правильно отличает русский текст от английского
    Гарантирует, что когда пользователь пишет в Telegram на русском, бот поймет, что нужно лезть в русскую базу.
    """
    # Создаем пустой объект для доступа к методу
    intel = ScalableMovieIntelligence(metadata=pd.DataFrame())

    assert intel._is_cyrillic("Матрица") is True
    assert intel._is_cyrillic("Matrix") is False
    assert intel._is_cyrillic("Movie (Фильм)") is True
    assert intel._is_cyrillic("12345") is False


# --- UK-DETECTION (украинские title'ы тоже кириллица) ---

@pytest.mark.parametrize(
    "text,expected",
    [
        # Чисто украинский — должен опознаваться как кириллица.
        ("Початок", True),               # Inception (uk)
        ("Міцний горішок", True),         # Die Hard (uk) — со специфичными і
        ("Володар перснів", True),        # Lord of the Rings (uk)
        ("Їжак у тумані", True),          # ї в начале слова
        ("Її серце", True),               # ї capital
        ("Ґедзь", True),                  # ґ — украинская специфика
        ("Тіні забутих предків", True),   # і
        ("Євангеліон", True),             # є
        # Чистая латиница не должна классифицироваться как кириллица.
        ("Inception", False),
        ("Lord of the Rings", False),
        # Mixed — найдена хотя бы одна кирил. буква → True (нужно для поиска).
        ("Inception (Початок)", True),
        ("Die Hard / Міцний горішок", True),
        # Punctuation/числа — не кириллица.
        ("!@#$%^", False),
        ("2010", False),
    ],
)
def test_cyrillic_detection_ukrainian(text, expected):
    """Регрессия: regex `[а-яёА-ЯЁ]` без явного `і/ї/є/ґ` — но эти буквы
    лежат в Unicode-блоке Cyrillic, и Python re по умолчанию их матчит.
    Тест фиксит поведение, чтобы рефакторинг regex не сломал uk-ввод.
    """
    intel = ScalableMovieIntelligence(metadata=pd.DataFrame())
    assert intel._is_cyrillic(text) is expected


def test_text_normalization():
    """Проверяем очистку текста: нижний регистр и удаление знаков препинания
    Важно для поиска. Если ты напишешь "Матрица!", а в базе она "Матрица", нормализация сотрет восклицательный знак и найдет фильм.
    """
    intel = ScalableMovieIntelligence(metadata=pd.DataFrame())

    # Должно убирать лишние пробелы и знаки, делать всё строчным
    assert intel._normalize_text("The Matrix!") == "the matrix"
    assert intel._normalize_text("  Inception  ") == "inception"
    assert intel._normalize_text("Fast & Furious") == "fast furious"


def test_genre_translation_map():
    """Проверяем, что словарь жанров содержит правильные переводы
    Простая проверка, что мы не удалили случайно важные жанры из словаря.
    """
    assert GENRE_EN_TO_RU["Action"] == "Боевик"
    assert GENRE_EN_TO_RU["Sci-Fi"] == "Фантастика"
    assert GENRE_EN_TO_RU["Horror"] == "Ужасы"


# --- GENRE_EN_TO_UK (украинский словарь) ---

def test_genre_translation_map_ukrainian():
    """Базовые проверки украинских переводов — словарь должен содержать
    все ключевые TMDB-жанры. Если отсутствует — рендер списка для uk-юзера
    отвалится в English-fallback по этому жанру."""
    assert GENRE_EN_TO_UK["Action"] == "Бойовик"
    assert GENRE_EN_TO_UK["Sci-Fi"] == "Фантастика"
    assert GENRE_EN_TO_UK["Horror"] == "Жахи"
    assert GENRE_EN_TO_UK["Drama"] == "Драма"
    assert GENRE_EN_TO_UK["Crime"] == "Кримінал"
    # The Blacklist regression: Mystery → Детектив (а не «Загадка»).
    assert GENRE_EN_TO_UK["Mystery"] == "Детектив"
    assert GENRE_EN_TO_UK["Thriller"] == "Трилер"
    assert GENRE_EN_TO_UK["Animation"] == "Мультфільм"


def test_genre_dictionaries_have_same_keys():
    """RU и UK словари должны быть симметричны по ключам — иначе
    появятся жанры, которые рендерятся локализованно для одного юзера
    и проваливаются в English для другого. Замечается guard-тестом
    при добавлении нового жанра в один словарь без второго."""
    ru_keys = set(GENRE_EN_TO_RU.keys())
    uk_keys = set(GENRE_EN_TO_UK.keys())
    only_ru = ru_keys - uk_keys
    only_uk = uk_keys - ru_keys
    assert not only_ru, f"keys present in RU but missing in UK: {only_ru}"
    assert not only_uk, f"keys present in UK but missing in RU: {only_uk}"


@pytest.mark.parametrize(
    "genre",
    [
        "Action", "Adventure", "Animation", "Comedy", "Crime",
        "Documentary", "Drama", "Family", "Fantasy", "History",
        "Horror", "Music", "Mystery", "Romance", "Science Fiction",
        "Thriller", "War", "Western",
    ],
)
def test_uk_translation_present_for_each_tmdb_genre(genre):
    """Каждый канонический TMDB-жанр должен иметь uk-перевод."""
    assert genre in GENRE_EN_TO_UK
    assert GENRE_EN_TO_UK[genre], f"empty UK translation for {genre}"


def test_uk_translations_are_in_cyrillic():
    """Все uk-переводы должны быть кириллическими (вестерн — спец-кейс,
    исторически пишется латиницей в обеих локалях)."""
    LATIN_OK = {"Western"}  # «Вестерн» в uk — но если так, должна быть кир.
    for en, uk in GENRE_EN_TO_UK.items():
        if en in LATIN_OK:
            continue
        assert any("Ѐ" <= ch <= "ӿ" for ch in uk), (
            f"UK translation for {en!r} should contain Cyrillic, got {uk!r}"
        )


def test_ru_and_uk_translations_differ_where_expected():
    """Отдельный sanity-чек: для жанров, где ru и uk реально различаются,
    словари не возвращают одно и то же. Если кто-то скопировал ru в uk —
    тест поймает."""
    distinct_pairs = [
        ("Action", "Боевик", "Бойовик"),
        ("Adventure", "Приключения", "Пригоди"),
        ("Animation", "Мультфильм", "Мультфільм"),
        ("Comedy", "Комедия", "Комедія"),
        ("Crime", "Криминал", "Кримінал"),
        ("Horror", "Ужасы", "Жахи"),
        ("Thriller", "Триллер", "Трилер"),
        ("History", "Исторический", "Історичний"),
        ("Family", "Семейный", "Сімейний"),
    ]
    for en, ru, uk in distinct_pairs:
        assert GENRE_EN_TO_RU[en] == ru, f"RU drift for {en}: {GENRE_EN_TO_RU[en]!r}"
        assert GENRE_EN_TO_UK[en] == uk, f"UK drift for {en}: {GENRE_EN_TO_UK[en]!r}"
        assert GENRE_EN_TO_RU[en] != GENRE_EN_TO_UK[en], (
            f"RU and UK translation collapsed for {en!r}"
        )


def test_genre_emoji_logic():
    """Проверяем получение эмодзи для жанров"""
    intel = ScalableMovieIntelligence(metadata=pd.DataFrame())

    # Должен возвращать эмодзи для известных жанров
    emojis = intel.get_genre_emojis(["Action", "Sci-Fi"])
    assert "💥" in emojis
    assert "🚀" in emojis

    # Если жанр неизвестен, должен вернуть стандартную иконку 🎬
    assert intel.get_genre_emojis(["Unknown Genre"]) == "🎬"


# --- ТЕСТЫ БАЗЫ ДАННЫХ (SQLite) ---

def test_cache_set_get(temp_db_path):
    """Проверяем, что SQLite правильно сохраняет и выдает переводы
    Это критический тест. Он создает временную базу в памяти, записывает туда данные и читает их.
    Так мы проверяем работу SQLite, не трогая твою основную базу translations.db.
    """
    cache = TMDBTranslationCache(temp_db_path)

    cache.set(
        tmdb_id=123, media_type="movie",
        title_ru="Тестовое название", overview_ru="Описание",
    )

    data = cache.get(123, "movie")

    assert data is not None
    assert data['title_ru'] == "Тестовое название"
    assert data['overview_ru'] == "Описание"

    cache.close()


def test_cache_empty_result(temp_db_path):
    """Проверяем поведение базы, если фильма в ней нет"""
    cache = TMDBTranslationCache(temp_db_path)

    assert cache.get(999999, "movie") is None

    cache.close()


def test_cache_isolates_movie_and_tv_by_media_type(temp_db_path):
    """Корневая регрессия: movie tmdb=1705 и TV tmdb=1705 — разные записи.

    До фикса PK был просто INTEGER, и backfill movies сначала писал «Битву
    за планету обезьян» в строку 1705, а backfill tv затем читал её как
    «уже закешировано» и портил title_ru у Fringe в TV-parquet. Композитный
    PK (tmdb_id, media_type) лечит это в корне — тест ловит регрессию,
    если кто-то снова уберёт media_type из ключа.
    """
    cache = TMDBTranslationCache(temp_db_path)
    try:
        cache.set_ru(1705, "movie", "Битва за планету обезьян")
        cache.set_ru(1705, "tv", "Грань")
        cache.set_uk(1705, "movie", "Битва за планету мавп")
        cache.set_uk(1705, "tv", "Межа")

        movie_row = cache.get(1705, "movie")
        tv_row = cache.get(1705, "tv")

        assert movie_row is not None
        assert tv_row is not None
        assert movie_row["title_ru"] == "Битва за планету обезьян"
        assert tv_row["title_ru"] == "Грань"
        assert movie_row["title_uk"] == "Битва за планету мавп"
        assert tv_row["title_uk"] == "Межа"

        # get_batch is per-domain too — must not leak the other domain's row.
        batch_movie = cache.get_batch([1705], "movie")
        batch_tv = cache.get_batch([1705], "tv")
        assert batch_movie[1705]["title_ru"] == "Битва за планету обезьян"
        assert batch_tv[1705]["title_ru"] == "Грань"
    finally:
        cache.close()


# --- ИНТЕГРАЦИОННЫЙ ТЕСТ ПОИСКА ---

def test_search_routing(mock_metadata):
    """Проверяем, что поиск находит английский фильм в mock-данных
    Проверяет весь путь — от ввода слова до получения объекта MovieInfo.

    Регрессия: ранжирование fuzzy-search должно ставить точный совпавший
    alt-title ('matrix') выше substring-matches ('the matrix reloaded',
    'matrix reloaded'). Если кто-то меняет _fuzzy_search и матрица-1999
    перестаёт быть первой — тест ловит.
    """
    intel = ScalableMovieIntelligence(metadata=mock_metadata)

    result = intel.search("Matrix")

    assert len(result.matches) > 0
    assert result.matches[0].title_en == "The Matrix"
    assert result.matches[0].item_id == 3