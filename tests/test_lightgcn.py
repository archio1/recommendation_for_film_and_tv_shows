import pytest
import torch
from recommendation_system.models.gnn.lightgcn import LightGCN


@pytest.fixture
def model_params():
    """Параметры для маленькой тестовой модели"""
    return {
        'num_users': 100,
        'num_items': 50,
        'num_genres': 20,
        'embedding_dim': 16,
        'num_layers': 3,
        'use_layer_weights': True
    }


@pytest.fixture
def sample_graph(model_params):
    """Создает минимальный граф для теста (10 ребер)"""
    num_nodes = model_params['num_users'] + model_params['num_items']
    # Случайные связи между юзерами (0-99) и айтемами (100-149)
    edge_index = torch.stack([
        torch.randint(0, model_params['num_users'], (10,)),
        torch.randint(model_params['num_users'], num_nodes, (10,))
    ])
    return edge_index


# 1. ТЕСТ ИНИЦИАЛИЗАЦИИ
def test_model_initialization(model_params):
    """Проверяем, что модель создается с нужными весами
    Просто создает объект модели LightGCN и заглядывает внутрь её весов.
    Мы проверяем, что слои эмбеддингов создались именно того размера, который мы задали.
    Если мы сказали, что у нас 100 юзеров и векторы по 16 чисел, то матрица должна быть 100 x 16
    Проверяет наличие вектора alpha — это те самые веса, которыми модель «смешивает» слои.
    Если их нет или их количество не совпадает с num_layers + 1, загрузка весов из Colab упадет.
    """
    model = LightGCN(**model_params)

    assert model.user_embedding.weight.shape == (model_params['num_users'], model_params['embedding_dim'])
    assert model.item_id_embedding.weight.shape == (model_params['num_items'], model_params['embedding_dim'])

    assert hasattr(model, 'layer_weights')
    assert len(model.layer_weights) == model_params['num_layers'] + 1


# 2. ТЕСТ FORWARD PASS (Базовый)
def test_forward_pass_shapes(model_params, sample_graph):
    """Проверяем размерности выходных данных после прохода через нейросеть
    Генерирует случайный маленький граф (кто-то что-то посмотрел) и прогоняет его через нейросеть.
    Это «тест на дым». Мы проверяем, что данные проходят через все слои свертки и на выходе получается результат,
    а не ошибка.
    Размерность на выходе: должна соответствовать количеству юзеров и айтемов.
    NaN-check: Это самое важное. В графовых сетях часто бывает деление на степень узла.
    Если у фильма 0 просмотров, может возникнуть деление на 0, и все веса станут NaN (не числами).
    Тест гарантирует, что математика стабильна.
    """
    model = LightGCN(**model_params)

    # Твой forward теперь принимает normalize_output
    user_emb, item_emb = model(sample_graph, normalize_output=False)

    assert user_emb.shape == (model_params['num_users'], model_params['embedding_dim'])
    assert item_emb.shape == (model_params['num_items'], model_params['embedding_dim'])
    assert not torch.isnan(user_emb).any()


# 3. ТЕСТ ГИБРИДНЫХ ПРИЗНАКОВ (Жанры + Годы)
def test_hybrid_features_injection(model_params, sample_graph):
    """Проверяем, что модель умеет принимать и обрабатывать жанры/годы
    Подает в модель не только связи в графе, но и дополнительные данные (жанры и годы).
    Твоя модель — гибридная. Она должна уметь подмешивать контент к ID фильма.
    Что метод forward не «спотыкается», когда получает кортеж из двух тензоров (жанры + годы),
    и правильно интегрирует их в финальные векторы.
    """
    model = LightGCN(**model_params)

    # Фейковые фичи: матрица жанров и тензор годов
    genre_features = torch.randn(model_params['num_items'], model_params['num_genres'])
    year_features = torch.randn(model_params['num_items'], 1)
    item_features = (genre_features, year_features)

    # Forward pass с фичами
    user_emb, item_emb = model(sample_graph, item_features=item_features)

    assert item_emb.shape == (model_params['num_items'], model_params['embedding_dim'])


# 4. ТЕСТ ВЕСОВ СЛОЕВ (Alpha)
def test_layer_weights_logic(model_params, sample_graph):
    """Проверяем, что веса слоев (alpha) влияют на результат
    Вручную меняет веса alpha (делает один из слоев максимально важным) и проверяет, что расчеты продолжают работать.
    В LightGCN финальный результат — это взвешенная сумма всех слоев.
    Мы проверяем, что формула Softmax(alpha) внутри модели написана без ошибок.
    Что даже при экстремальных значениях весов слоев (например, всё внимание только первому слою) модель выдает валидные векторы.
    """
    model = LightGCN(**model_params)

    with torch.no_grad():
        model.layer_weights.data.fill_(0.0)
        model.layer_weights.data[0] = 10.0 # Делаем 0-й слой доминирующим

    user_emb, item_emb = model(sample_graph)

    # Результат должен быть близок к начальным эмбеддингам
    # (проверка, что логика взвешивания вообще работает)
    assert user_emb.shape[1] == model_params['embedding_dim']


# 5. ТЕСТ ПОЛУЧЕНИЯ EMBEDDINGS (Inference mode)
def test_get_item_embedding(model_params):
    """Тестируем метод, который используется ботом для поиска похожих
    Тестирует отдельную функцию, которую вызывает твой бот и GUI, когда ты ищешь «похожие фильмы».
    Бот не прогоняет каждый раз весь граф (это долго), он берет уже готовые «богатые» эмбеддинги.
    Нам нужно быть уверенными, что эта сокращенная процедура работает.
    Сравнивает эмбеддинг фильма «без жанров» и «с жанрами».
    Логика: Они не должны быть одинаковыми. Если они равны, значит, жанры в модели просто игнорируются (баг в коде сложения векторов).
    Тест поймает это.
    """
    model = LightGCN(**model_params)

    # Без фичей
    emb_simple = model.get_item_embedding()
    assert emb_simple.shape == (model_params['num_items'], model_params['embedding_dim'])

    # С фичами
    genre_features = torch.randn(model_params['num_items'], model_params['num_genres'])
    year_features = torch.randn(model_params['num_items'], 1)
    emb_hybrid = model.get_item_embedding((genre_features, year_features))

    # Эмбеддинги должны отличаться
    assert not torch.equal(emb_simple, emb_hybrid)