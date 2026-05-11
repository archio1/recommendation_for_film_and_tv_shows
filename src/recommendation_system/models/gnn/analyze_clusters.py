import torch
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from pathlib import Path
import json

# --- НАСТРОЙКИ ---
CLUSTERS_COUNT = 15  # На сколько групп делим фильмы
TOP_MOVIES_TO_ANALYZE = 5000  # Берем только популярные, чтобы не засорять график

# --- ПУТИ ---
BASE_DIR = Path(__file__).parent.parents[3]
DATA_DIR = BASE_DIR / 'data' / 'processed'
MODEL_PATH = BASE_DIR / 'models' / 'lightgcn_best.pt'
HTML_OUTPUT = BASE_DIR / 'reports' / 'figures' / 'interactive_map_v12k.html'


def analyze_and_plot():
    print("1. Загрузка данных...")
    # Загружаем метаданные
    metadata = pd.read_parquet(DATA_DIR / 'items_metadata_final.parquet')

    # Загружаем модель
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    # Получаем эмбеддинги (координаты) фильмов [N, 16]
    item_weights = checkpoint['model_state_dict']['item_embedding.weight'].numpy()

    # Ограничиваемся топ фильмами (они идут первыми в metadata, если id сортированы по популярности)
    # Или просто берем первые N, так как в вашем датасете они уже отсортированы при создании
    if len(item_weights) > TOP_MOVIES_TO_ANALYZE:
        subset_ids = np.arange(TOP_MOVIES_TO_ANALYZE)
    else:
        subset_ids = np.arange(len(item_weights))

    vectors = item_weights[subset_ids]

    # Фильтруем метаданные, чтобы совпадали с векторами
    df = metadata[metadata['item_id'].isin(subset_ids)].copy()
    # Убедимся, что порядок совпадает
    df = df.set_index('item_id').reindex(subset_ids).reset_index()

    print(f"2. Кластеризация (K-Means) на {len(df)} фильмов...")
    # Делим фильмы на группы на основе их векторов
    kmeans = KMeans(n_clusters=CLUSTERS_COUNT, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(vectors)

    # --- ОТЧЕТ В КОНСОЛЬ ---
    print("\n" + "=" * 60)
    print(f"ОПИСАНИЕ КЛАСТЕРОВ (Что выучила нейросеть)")
    print("=" * 60)

    # Для каждого кластера находим фильмы, ближайшие к центру этого кластера
    for cluster_id in range(CLUSTERS_COUNT):
        # Вектора этого кластера
        cluster_indices = df[df['cluster'] == cluster_id].index
        cluster_vectors = vectors[cluster_indices]
        center = kmeans.cluster_centers_[cluster_id]

        # Считаем расстояние до центра
        distances = np.linalg.norm(cluster_vectors - center, axis=1)

        # Берем топ-7 самых близких к центру (самых типичных представителей)
        top_idx_local = np.argsort(distances)[:7]
        top_movies_indices = cluster_indices[top_idx_local]

        movies = df.loc[top_movies_indices, ['title', 'year', 'genres']]

        print(f"\n📁 КЛАСТЕР {cluster_id}:")
        for _, row in movies.iterrows():
            genres = row['genres'] if isinstance(row['genres'], str) else str(row['genres'])
            print(f"   • {row['title']} ({row['year']}) | {genres}")

    print("\n" + "=" * 60)
    print("3. Генерация 2D карты (t-SNE)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    projections = tsne.fit_transform(vectors)

    df['x'] = projections[:, 0]
    df['y'] = projections[:, 1]

    # Преобразуем жанры в строку для красивого отображения при наведении
    df['genres_str'] = df['genres'].apply(lambda x: ", ".join(x) if isinstance(x, (list, np.ndarray)) else str(x))

    print(f"4. Создание интерактивного графика...")
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='cluster',
        hover_data=['title', 'year', 'genres_str'],
        title='Интерактивная карта интересов (LightGCN)',
        color_continuous_scale=px.colors.qualitative.G10  # Яркие цвета
    )

    # Убираем оси, чтобы было красивее
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    # Создаем папку если нет
    HTML_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(HTML_OUTPUT)
    print(f"✅ Готово! Откройте файл в браузере: \n{HTML_OUTPUT}")


if __name__ == "__main__":
    analyze_and_plot()