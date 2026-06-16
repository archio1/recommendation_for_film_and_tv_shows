import torch
import pandas as pd
import numpy as np

from recommendation_system.models.gnn.lightgcn import LightGCN
from recommendation_system.paths import MODELS_DIR, PROCESSED_DIR

# --- НАСТРОЙКИ ---
DATA_DIR = PROCESSED_DIR
MODEL_PATH = MODELS_DIR / 'lightgcn_best_v4.pt'


def prepare_content_features_for_inference(metadata, device, expected_num_genres):
    """Подготовка фичей для inference (без обучения)"""
    metadata = metadata.sort_values('item_id')

    # Собираем все жанры
    all_genres_in_data = set()
    for gs in metadata['genres']:
        if isinstance(gs, (list, np.ndarray)):
            all_genres_in_data.update(gs)

    # Сортируем по алфавиту (как при обучении)
    genre_list = sorted(list(all_genres_in_data))

    # 2. Подгоняем список под размер модели (expected_num_genres)
    if len(genre_list) > expected_num_genres:
        # Если жанров в данных больше, берем только первые N, на которых училась модель
        genre_list = genre_list[:expected_num_genres]
    elif len(genre_list) < expected_num_genres:
        # Если меньше, дополняем пустышками
        genre_list = genre_list + [f"dummy_{i}" for i in range(expected_num_genres - len(genre_list))]

    genre_map = {g: i for i, g in enumerate(genre_list)}

    print(f"  ✅ Жанры синхронизированы: {len(genre_list)} шт.")

    # Genre matrix
    genre_matrix = torch.zeros((len(metadata), expected_num_genres), device=device)
    for idx, row in metadata.iterrows():
        item_id = row['item_id']
        if item_id >= len(metadata):
            continue
        gs = row['genres']
        if isinstance(gs, (list, np.ndarray)):
            # Берем только те жанры, которые попали в наш синхронизированный список
            indices = [genre_map[g] for g in gs if g in genre_map]
            if indices:
                genre_matrix[item_id, indices] = 1.0

    # Year tensor (z-score)
    years = metadata['year'].fillna(2000).values
    year_mean = np.mean(years)
    year_std = np.std(years) + 1e-8
    years_normalized = (years - year_mean) / year_std
    year_tensor = torch.tensor(years_normalized, dtype=torch.float32, device=device).view(-1, 1)

    return (genre_matrix, year_tensor)


def load_resources():
    print(f"📂 Загрузка данных из: {DATA_DIR}")

    # 1. Загрузка чекпоинта ПЕРВЫМ делом, чтобы узнать размеры
    print(f"🧠 Загрузка модели: {MODEL_PATH.name}...")
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # ОПРЕДЕЛЯЕМ ПАРАМЕТРЫ ИЗ МОДЕЛИ
    num_users = state_dict['user_embedding.weight'].shape[0]
    embedding_dim = state_dict['user_embedding.weight'].shape[1]

    if 'item_id_embedding.weight' in state_dict:
        model_num_items = state_dict['item_id_embedding.weight'].shape[0]
    else:
        model_num_items = state_dict['item_embedding.weight'].shape[0]

    num_genres = 0
    if 'genre_encoder.weight' in state_dict:
        num_genres = state_dict['genre_encoder.weight'].shape[1]

    use_layer_weights = 'layer_weights' in state_dict

    print(f"  ⚙️ Модель обучена на: Users={num_users}, Items={model_num_items}, Genres={num_genres}")

    # 2. Метаданные (ФИЛЬТРУЕМ ПОД РАЗМЕР МОДЕЛИ)
    full_metadata = pd.read_parquet(DATA_DIR / 'items_metadata_final.parquet')

    # Оставляем только те айтемы, которые знает нейросеть
    metadata = full_metadata[full_metadata['item_id'] < model_num_items].copy()
    print(f"  📊 Метаданные обрезаны до {len(metadata)} записей (только trained айтемы)")

    # Создаем модель
    model = LightGCN(
        num_users=num_users,
        num_items=model_num_items,
        num_genres=num_genres,
        embedding_dim=embedding_dim,
        use_layer_weights=use_layer_weights
    )

    model.load_state_dict(state_dict)
    model.eval()

    # Подготовка фичей только для известных модели айтемов
    device = 'cpu'
    item_features = prepare_content_features_for_inference(metadata, device, num_genres)

    return model, metadata, item_features


def build_dummy_graph(num_users, num_items):
    """
    Создаёт минимальный граф для inference.

    Для поиска похожих фильмов нам не нужен полный граф —
    достаточно получить эмбеддинги через forward pass.
    Но forward требует edge_index, поэтому создаём пустой.
    """
    # Пустой граф (или можно загрузить реальный для более точных результатов)
    # Для similarity search между items важны только item эмбеддинги
    edge_index = torch.tensor([[], []], dtype=torch.long)
    return edge_index


def get_final_embeddings(model, item_features, num_users, num_items):
    """
    КРИТИЧЕСКИ ВАЖНО: получаем ФИНАЛЬНЫЕ эмбеддинги через forward pass,
    а не сырые веса!

    Однако для inference без графа мы используем get_item_embedding напрямую,
    так как GCN слои без рёбер не изменят эмбеддинги.
    """
    with torch.no_grad():
        # Вариант 1: Если есть граф — используем полный forward
        # user_emb, item_emb = model(edge_index, item_features, normalize_output=True)

        # Вариант 2: Без графа — используем только item embedding с фичами
        # Это даёт нам ID + Genre + Year но без GCN propagation
        item_emb = model.get_item_embedding(item_features)

        # Нормализуем для cosine similarity
        import torch.nn.functional as F
        item_emb = F.normalize(item_emb, p=2, dim=1)

    return item_emb


def find_similar(model, metadata, item_features, query, top_k=10):
    """Поиск похожих фильмов с ПРАВИЛЬНЫМИ эмбеддингами"""

    # Ищем фильм по названию
    mask = metadata['title'].str.contains(query, case=False, na=False)
    matches = metadata[mask]

    if len(matches) == 0:
        print(f"\n❌ Фильм '{query}' не найден.")
        return

    # Берем первый результат
    target_row = matches.iloc[0]
    target_id = target_row['item_id']
    print(f"\n🔎 Поиск похожих на: \033[1m{target_row['title']} ({int(target_row['year'])})\033[0m")

    # Жанры целевого фильма
    target_genres = target_row['genres']
    if isinstance(target_genres, (list, np.ndarray)):
        print(f"   Жанры: {', '.join(target_genres)}")

    # --- ПОЛУЧАЕМ ПРАВИЛЬНЫЕ ЭМБЕДДИНГИ ---
    item_weights = get_final_embeddings(
        model, item_features,
        model.num_users, model.num_items
    )

    target_vector = item_weights[target_id].unsqueeze(0)

    # Косинусное сходство
    cosine_sim = torch.nn.functional.cosine_similarity(target_vector, item_weights)

    # Топ-K
    scores, indices = torch.topk(cosine_sim, top_k + 1)

    print("-" * 70)
    print(f" {'Score':<8} | {'Title':<40} | {'Genres'}")
    print("-" * 70)

    for score, idx in zip(scores, indices):
        idx = idx.item()
        if idx == target_id:
            continue

        row = metadata[metadata['item_id'] == idx]
        if len(row) == 0:
            continue
        row = row.iloc[0]

        genres = row['genres']
        if isinstance(genres, (list, np.ndarray)):
            genres_str = ", ".join(genres[:4])  # Ограничиваем для читаемости
            if len(genres) > 4:
                genres_str += "..."
        else:
            genres_str = str(genres)

        title = f"{row['title']} ({int(row['year'])})"
        if len(title) > 38:
            title = title[:35] + "..."

        print(f" {score:.4f}  | {title:<40} | {genres_str}")

    print("-" * 70)


def diagnose_embeddings(model, item_features, metadata):
    """Диагностика качества эмбеддингов"""
    print("\n" + "=" * 70)
    print("📊 ДИАГНОСТИКА ЭМБЕДДИНГОВ")
    print("=" * 70)

    item_emb = get_final_embeddings(
        model, item_features,
        model.num_users, model.num_items
    )

    # 1. Variance
    variance = item_emb.var(dim=0).mean().item()
    print(f"Variance (по измерениям): {variance:.6f}")
    print(f"   {'🟢 OK' if variance > 0.01 else '🔴 НИЗКАЯ VARIANCE!'}")

    # 2. Средняя норма (после нормализации должна быть ≈ 1)
    norms = item_emb.norm(dim=1)
    print(f"Средняя норма: {norms.mean().item():.4f} ± {norms.std().item():.4f}")

    # 3. Среднее косинусное сходство (случайная выборка)
    sample_size = min(500, len(item_emb))
    indices = torch.randperm(len(item_emb))[:sample_size]
    sample = item_emb[indices]

    cosine_matrix = torch.matmul(sample, sample.t())
    mask = ~torch.eye(sample_size, dtype=torch.bool)
    cosine_values = cosine_matrix[mask]

    cosine_mean = cosine_values.mean().item()
    cosine_std = cosine_values.std().item()
    cosine_max = cosine_values.max().item()

    print(f"\nКосинусное сходство (random pairs):")
    print(f"   Mean: {cosine_mean:.4f}")
    print(f"   Std:  {cosine_std:.4f}")
    print(f"   Max:  {cosine_max:.4f}")

    if cosine_mean > 0.9:
        print("   🔴 EMBEDDING COLLAPSE DETECTED!")
        print("   Все эмбеддинги почти идентичны.")
    elif cosine_mean > 0.7:
        print("   🟡 WARNING: высокое среднее сходство")
    else:
        print("   🟢 OK: эмбеддинги хорошо разделены")

    # 4. PCA анализ (быстрый)
    print(f"\n📐 Эффективная размерность:")
    # SVD для оценки сколько компонент несут информацию
    U, S, V = torch.svd(item_emb[:1000])  # Выборка для скорости
    explained_var = (S ** 2) / (S ** 2).sum()
    cumsum = explained_var.cumsum(0)

    # Сколько компонент объясняют 90% дисперсии?
    dims_90 = (cumsum < 0.9).sum().item() + 1
    dims_99 = (cumsum < 0.99).sum().item() + 1

    print(f"   90% variance: {dims_90} dims (из {item_emb.shape[1]})")
    print(f"   99% variance: {dims_99} dims")

    if dims_90 < 10:
        print("   🔴 ПРОБЛЕМА: эмбеддинги лежат в очень узком подпространстве!")

    print("=" * 70)


if __name__ == "__main__":
    try:
        model, meta, item_features = load_resources()

        # Диагностика
        diagnose_embeddings(model, item_features, meta)

        # Тесты
        print("\n" + "=" * 70)
        print("🎬 ТЕСТЫ ПОИСКА ПОХОЖИХ ФИЛЬМОВ")
        print("=" * 70)

        find_similar(model, meta, item_features, "Shrek")
        find_similar(model, meta, item_features, "The Matrix")
        find_similar(model, meta, item_features, "The Dark Knight")
        find_similar(model, meta, item_features, "Inception")
        find_similar(model, meta, item_features, "Mean Girls")

        while True:
            q = input("\nВведите название фильма (или 'q' для выхода): ")
            if q.lower() == 'q':
                break
            find_similar(model, meta, item_features, q)

    except FileNotFoundError as e:
        print(f"❌ ОШИБКА: {e}")
