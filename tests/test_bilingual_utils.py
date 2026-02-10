import pytest
import pandas as pd
from src.recommendation_system.models.gnn.bilingual_utils import (
    ScalableMovieIntelligence,
    TMDBTranslationCache,
    GENRE_EN_TO_RU
)


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

    # Сохраняем тестовый перевод
    cache.set(tmdb_id=123, title_ru="Тестовое название", overview_ru="Описание")

    # Пытаемся достать
    data = cache.get(123)

    assert data is not None
    assert data['title_ru'] == "Тестовое название"
    assert data['overview_ru'] == "Описание"

    cache.close()


def test_cache_empty_result(temp_db_path):
    """Проверяем поведение базы, если фильма в ней нет"""
    cache = TMDBTranslationCache(temp_db_path)

    # Ищем ID, которого нет в базе
    assert cache.get(999999) is None

    cache.close()


# --- ИНТЕГРАЦИОННЫЙ ТЕСТ ПОИСКА ---

def test_search_routing(mock_metadata):
    """Проверяем, что поиск находит английский фильм в mock-данных
    Проверяет весь путь — от ввода слова до получения объекта MovieInfo.
    """
    intel = ScalableMovieIntelligence(metadata=mock_metadata)

    # Ищем "Matrix"
    result = intel.search("Matrix")

    assert len(result.matches) > 0
    # Проверяем, что нашелся именно нужный фильм
    assert result.matches[0].title_en == "The Matrix"
    assert result.matches[0].item_id == 0