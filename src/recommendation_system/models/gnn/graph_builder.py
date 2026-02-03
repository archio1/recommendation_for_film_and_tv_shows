import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path
from typing import Tuple, Dict
import json
import logging

logger = logging.getLogger(__name__)


class MovieGraphBuilder:
    """Строит биpartite граф user-item для LightGCN"""

    def __init__(self, data_dir: Path):
        self.data_processed = data_dir

        # Загружаем данные
        self.interactions = pd.read_parquet(self.data_processed / 'interactions_final.parquet')
        self.metadata = pd.read_parquet(self.data_processed / 'items_metadata_final.parquet')

        with open(self.data_processed / 'id_mapping.json') as f:
            self.mapping = json.load(f)

        self.num_users = self.mapping['num_users']
        self.num_items = self.mapping['num_items']

        logger.info(f"Загружено: {len(self.interactions):,} взаимодействий")
        logger.info(f"Users: {self.num_users:,}, Items: {self.num_items:,}")

    def build_graph(self) -> Data:
        """
        Строит undirected bipartite граф для LightGCN

        Структура:
        - Узлы: [user_0, user_1, ..., user_N, item_0, item_1, ..., item_M]
        - Рёбра: user -> item и item -> user (двунаправленные)

        Returns:
            PyG Data объект
        """
        logger.info("\nПостроение графа...")

        # User IDs: 0 до num_users-1
        user_ids = self.interactions['user_id'].values

        # Item IDs: сдвигаем на num_users чтобы не пересекались с user IDs
        # В графе: items будут иметь ID от num_users до num_users+num_items-1
        item_ids = self.interactions['item_id'].values + self.num_users

        # Создаём рёбра: user -> item
        edge_index = np.array([user_ids, item_ids])  # Преобразуем в numpy массив
        edge_index_user_to_item = torch.tensor(edge_index, dtype=torch.long)

        # Создаём рёбра: item -> user (обратные)
        edge_index_item_to_user = torch.tensor([item_ids, user_ids], dtype=torch.long)

        # Объединяем (undirected граф)
        edge_index = torch.cat([edge_index_user_to_item, edge_index_item_to_user], dim=1)

        # Создаём PyG Data объект
        data = Data(
            edge_index=edge_index,
            num_nodes=self.num_users + self.num_items
        )

        # Добавляем метаданные
        data.num_users = self.num_users
        data.num_items = self.num_items

        logger.info(f"✓ Граф построен:")
        logger.info(f"  Узлов: {data.num_nodes:,} ({self.num_users:,} users + {self.num_items:,} items)")
        logger.info(f"  Рёбер: {data.edge_index.shape[1]:,} ({data.edge_index.shape[1] // 2:,} взаимодействий × 2)")

        return data

    def train_test_split(
            self,
            test_size: float = 0.2,
            temporal: bool = False,
            random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        БЫСТРОЕ разделение на train/test

        Args:
            test_size: Доля тестовой выборки
            temporal: True = по времени, False = случайно
            random_state: Seed для воспроизводимости

        Returns:
            train_df, test_df, split_info
        """
        logger.info(f"\nРазделение train/test (test_size={test_size})...")

        if temporal and 'timestamp' in self.interactions.columns:
            # Temporal split (БЫСТРО)
            df_sorted = self.interactions.sort_values('timestamp')
            split_idx = int(len(df_sorted) * (1 - test_size))

            train_df = df_sorted.iloc[:split_idx].copy()
            test_df = df_sorted.iloc[split_idx:].copy()

            logger.info("  Метод: temporal split")
        else:
            # БЫСТРОЕ случайное разделение (векторизованное)
            np.random.seed(random_state)

            # Простое случайное разделение (намного быстрее!)
            mask = np.random.rand(len(self.interactions)) < (1 - test_size)
            train_df = self.interactions[mask].copy()
            test_df = self.interactions[~mask].copy()

            logger.info("  Метод: fast random split")

        split_info = {
            'train_size': len(train_df),
            'test_size': len(test_df),
            'train_users': int(train_df['user_id'].nunique()),
            'test_users': int(test_df['user_id'].nunique()),
            'train_items': int(train_df['item_id'].nunique()),
            'test_items': int(test_df['item_id'].nunique())
        }

        logger.info(f"✓ Train: {split_info['train_size']:,}, Test: {split_info['test_size']:,}")

        return train_df, test_df, split_info

    def build_user_item_matrix(self, interactions_df: pd.DataFrame):
        """
        Строит разреженную матрицу user-item для быстрого доступа

        Returns:
            scipy.sparse.csr_matrix
        """
        from scipy.sparse import csr_matrix

        users = interactions_df['user_id'].values
        items = interactions_df['item_id'].values
        data = np.ones(len(users), dtype=np.float32)

        matrix = csr_matrix(
            (data, (users, items)),
            shape=(self.num_users, self.num_items),
            dtype=np.float32
        )

        logger.info(
            f"  User-item матрица: {matrix.shape}, плотность: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.6%}")

        return matrix

    def prepare_for_training(
            self,
            test_size: float = 0.2,
            temporal: bool = False,
            random_state: int = 42
    ) -> Dict:
        """
        Полная подготовка данных для обучения

        Returns:
            dict с train_graph, test_data, train_matrix, metadata
        """
        logger.info("=" * 70)
        logger.info("ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
        logger.info("=" * 70)

        # 1. Train/test split
        train_df, test_df, split_info = self.train_test_split(
            test_size=test_size,
            temporal=temporal,
            random_state=random_state
        )

        # 2. Строим train граф
        logger.info("\nПостроение train графа...")
        # Временно подменяем interactions на train
        original_interactions = self.interactions
        self.interactions = train_df
        train_graph = self.build_graph()
        self.interactions = original_interactions

        # 3. Строим train матрицу (для negative sampling)
        logger.info("\nПостроение train матрицы...")
        train_matrix = self.build_user_item_matrix(train_df)

        # 4. Подготовка test данных
        logger.info("\nПодготовка test данных...")
        test_data = {
            'interactions': test_df,
            'user_ids': test_df['user_id'].values,
            'item_ids': test_df['item_id'].values,
            'num_interactions': len(test_df)
        }

        # 5. Собираем всё вместе
        result = {
            'train_graph': train_graph,
            'train_matrix': train_matrix,
            'test_data': test_data,
            'train_df': train_df,
            'test_df': test_df,
            'split_info': split_info,
            'num_users': self.num_users,
            'num_items': self.num_items,
            'metadata': self.metadata
        }

        logger.info("\n" + "=" * 70)
        logger.info("✅ ДАННЫЕ ГОТОВЫ ДЛЯ ОБУЧЕНИЯ")
        logger.info("=" * 70)

        return result


# Пример использования
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Путь к данным
    project_root = Path(__file__).resolve()
    while project_root.name != 'recommendation_for_film_and_tv_shows':
        project_root = project_root.parent
    data_dir = project_root / 'data'

    # Создаём builder
    builder = MovieGraphBuilder(data_dir)

    # Подготовка для обучения
    data = builder.prepare_for_training(
        test_size=0.2,
        temporal=False,
        random_state=42
    )

    print(f"\n✓ Train граф: {data['train_graph']}")
    print(f"✓ Train матрица: {data['train_matrix'].shape}")
    print(f"✓ Test взаимодействий: {data['test_data']['num_interactions']:,}")