import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np

from recommendation_system.paths import MODELS_DIR, PROCESSED_DIR

# Пути (те же)
DATA_DIR = PROCESSED_DIR
MODEL_PATH = MODELS_DIR / 'lightgcn_best.pt'


def visualize():
    # 1. Загрузка
    metadata = pd.read_parquet(DATA_DIR / 'items_metadata_final.parquet')
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    item_weights = checkpoint['model_state_dict']['item_embedding.weight'].numpy()

    # 2. Фильтрация (берем только Top-500 популярных, чтобы не было каши на графике)
    # Предполагаем, что item_id отсортированы или у нас есть поле popularity
    # Для простоты возьмем топ-1000 фильмов из метаданных (они там обычно отсортированы)
    top_k = 1000

    # Берем ID и Жанры для топ-K
    # Важно: item_id в метаданных должны соответствовать индексам в матрице весов!
    # (В вашем коде это так, так что все ок)
    subset = metadata[metadata['item_id'] < len(item_weights)].head(top_k).copy()

    # Вытаскиваем главный жанр для раскраски
    def get_main_genre(genres):
        # genres может быть списком или строкой
        if isinstance(genres, np.ndarray) or isinstance(genres, list):
            return genres[0] if len(genres) > 0 else 'Other'
        return 'Other'

    subset['main_genre'] = subset['genres'].apply(get_main_genre)

    # Берем только популярные жанры для легенды
    top_genres = subset['main_genre'].value_counts().head(8).index
    subset.loc[~subset['main_genre'].isin(top_genres), 'main_genre'] = 'Other'

    # Эмбеддинги для этих фильмов
    indices = subset['item_id'].values
    vectors = item_weights[indices]

    print(f"Запуск t-SNE для {top_k} фильмов...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(vectors)

    # 3. Рисуем
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=subset['main_genre'],
        palette='tab10',
        s=60,
        alpha=0.8
    )

    # Подпишем несколько известных фильмов
    famous_movies = ["Shrek", "Matrix", "Titanic", "Godfather", "Avengers", "Inception"]

    for title in famous_movies:
        row = subset[subset['title'].str.contains(title, case=False)].head(1)
        if not row.empty:
            idx = subset.index.get_loc(row.index[0])
            plt.text(
                embeddings_2d[idx, 0] + 0.2,
                embeddings_2d[idx, 1] + 0.2,
                row.iloc[0]['title'],
                fontsize=9,
                weight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )

    plt.title('Карта фильмов (LightGCN Embeddings)', fontsize=16)
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend(title='Жанр')
    plt.grid(True, alpha=0.3)

    output_path = BASE_DIR / 'reports' / 'figures' / 'movie_map.png'
    plt.savefig(output_path)
    print(f"✅ График сохранен: {output_path}")
    plt.show()


if __name__ == "__main__":
    visualize()