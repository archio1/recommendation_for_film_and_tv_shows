import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from src.recommendation_system.models.gnn.trainer_gui import InferenceEngine


# --- ВСПОМОГАТЕЛЬНЫЙ КЛАСС (MOCK) ---
@pytest.fixture
def engine_with_mock_model(mock_metadata):
    """Создает InferenceEngine с фейковой нейросетью для тестов
    Это хитрая штука. Мы не хотим, чтобы тесты ждали 10 секунд, пока загрузится твоя реальная модель lightgcn_best_v3.pt.
    Поэтому мы подменяем нейросеть «пустышкой», которая мгновенно выдает случайные числа.
    Нам не важно, какие это числа, нам важно проверить, как работает фильтр после них.
    """
    # Создаем объект без вызова __init__, чтобы не грузить файлы
    engine = InferenceEngine.__new__(InferenceEngine)

    # Вручную прописываем нужные атрибуты
    engine.metadata = mock_metadata
    engine.device = 'cpu'
    engine.is_loaded = True
    engine.item_features = (None, None)

    # Создаем фейковую модель и её эмбеддинги
    # Для 10 фильмов создаем векторы размерности 64
    mock_embeddings = torch.randn(len(mock_metadata), 64)

    engine.model = MagicMock()
    # Метод get_item_embedding будет возвращать наши случайные векторы
    engine.model.get_item_embedding.return_value = mock_embeddings

    return engine


# --- ТЕСТЫ ЛОГИКИ ---

def test_recommendation_quantity(engine_with_mock_model):
    """Проверяем, что бот возвращает ровно столько фильмов, сколько просили (8)"""
    # Выбираем Матрицу (item_id 3)
    liked_ids = [3]
    recs = engine_with_mock_model.get_recommendations(liked_ids, top_k=5)

    assert len(recs) == 5


def test_exclusion_of_liked_movies(engine_with_mock_model):
    """Проверяем, что выбранный фильм не попадает в рекомендации"""
    liked_ids = [0]  # Выбрали Saw
    recs = engine_with_mock_model.get_recommendations(liked_ids, top_k=5)

    rec_ids = [r['item_id'] for r in recs]
    assert 0 not in rec_ids, "Выбранный фильм Saw не должен быть в рекомендациях"


def test_sequel_filtering_saw(engine_with_mock_model):
    """
    КРИТИЧЕСКИЙ ТЕСТ: Проверка нашего бага с короткими названиями.
    Если выбрана 'Saw', в рекомендациях не должно быть 'Saw II' и 'Saw III'.
    Это проверка того исправления, которое мы делали (len >= 3).
    Если ты случайно удалишь этот фикс в будущем, этот тест «покраснеет» и спасет тебя.
    """
    liked_ids = [0]  # Saw
    # Просим рекомендации
    recs = engine_with_mock_model.get_recommendations(liked_ids, top_k=5)

    rec_titles = [r['title'].lower() for r in recs]

    # Проверяем, что ни одна рекомендация не содержит слово 'saw'
    for title in rec_titles:
        assert 'saw' not in title, f"Сиквел '{title}' просочился в рекомендации к 'Saw'"


def test_sequel_filtering_matrix(engine_with_mock_model):
    """Проверка фильтрации для Матрицы"""
    liked_ids = [3]  # The Matrix
    recs = engine_with_mock_model.get_recommendations(liked_ids, top_k=5)

    rec_titles = [r['title'].lower() for r in recs]

    # 'The Matrix Reloaded' (item_id 4) должна быть отфильтрована
    for title in rec_titles:
        assert 'reloaded' not in title
        # Проверяем по корню названия
        assert title != 'the matrix reloaded'


def test_empty_input_handling(engine_with_mock_model):
    """Проверяем, что движок не падает, если пришел пустой список лайков"""
    assert engine_with_mock_model.get_recommendations([], top_k=8) == []


def test_imdb_url_generation(engine_with_mock_model):
    """Проверяем, что ссылка на IMDb генерируется для каждой рекомендации
     Проверяет, что мы не забыли добавить ссылки, которые нужны боту в Telegram.
    """
    recs = engine_with_mock_model.get_recommendations([7], top_k=1)
    assert 'imdb_url' in recs[0]
    assert recs[0]['imdb_url'].startswith('https://www.imdb.com/')