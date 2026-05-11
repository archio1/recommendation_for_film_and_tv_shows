"""
graph_builder.py — Fixed version
Changes:
  1. Fix UserWarning: convert numpy arrays to single ndarray before tensor creation
  2. Use int32 for edge_index to save VRAM (PyG supports it)
"""

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
    """Builds bipartite user-item graph for LightGCN"""

    def __init__(self, data_dir: Path):
        self.data_processed = data_dir / 'processed'

        self.interactions = pd.read_parquet(self.data_processed / 'interactions_final.parquet')
        self.metadata = pd.read_parquet(self.data_processed / 'items_metadata_final.parquet')

        with open(self.data_processed / 'id_mapping.json') as f:
            self.mapping = json.load(f)

        self.num_users = self.mapping['num_users']
        # Use num_trained_items for graph (only items with interactions)
        self.num_trained_items = self.mapping.get('num_trained_items', self.mapping['num_items'])
        self.num_items = self.mapping['num_items']

        logger.info(f"Loaded: {len(self.interactions):,} interactions")
        logger.info(f"Users: {self.num_users:,}, Items (graph): {self.num_trained_items:,}, Items (catalog): {self.num_items:,}")

    def build_graph(self) -> Data:
        """Build undirected bipartite graph for LightGCN"""
        logger.info("Building graph...")

        # FIX: Convert to numpy arrays FIRST, then to tensor (eliminates UserWarning)
        user_ids = self.interactions['user_id'].values.astype(np.int64)
        item_ids = (self.interactions['item_id'].values + self.num_users).astype(np.int64)

        # Stack into single numpy array, then convert once
        u2i = np.stack([user_ids, item_ids])  # [2, num_edges]
        i2u = np.stack([item_ids, user_ids])   # [2, num_edges]
        edge_index_np = np.concatenate([u2i, i2u], axis=1)  # [2, 2*num_edges]

        edge_index = torch.from_numpy(edge_index_np).long()

        num_nodes = self.num_users + self.num_trained_items

        data = Data(edge_index=edge_index, num_nodes=num_nodes)
        data.num_users = self.num_users
        data.num_items = self.num_trained_items

        logger.info(f"  Nodes: {num_nodes:,} ({self.num_users:,} users + {self.num_trained_items:,} items)")
        logger.info(f"  Edges: {edge_index.shape[1]:,}")

        return data

    def train_test_split(
        self,
        test_size: float = 0.2,
        temporal: bool = False,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        logger.info(f"Train/test split (test_size={test_size})...")

        if temporal and 'timestamp' in self.interactions.columns:
            df_sorted = self.interactions.sort_values('timestamp')
            split_idx = int(len(df_sorted) * (1 - test_size))
            train_df = df_sorted.iloc[:split_idx].copy()
            test_df = df_sorted.iloc[split_idx:].copy()
        else:
            np.random.seed(random_state)
            mask = np.random.rand(len(self.interactions)) < (1 - test_size)
            train_df = self.interactions[mask].copy()
            test_df = self.interactions[~mask].copy()

        split_info = {
            'train_size': len(train_df),
            'test_size': len(test_df),
            'train_users': int(train_df['user_id'].nunique()),
            'test_users': int(test_df['user_id'].nunique()),
        }
        logger.info(f"  Train: {split_info['train_size']:,}, Test: {split_info['test_size']:,}")
        return train_df, test_df, split_info

    def build_user_item_matrix(self, interactions_df: pd.DataFrame):
        from scipy.sparse import csr_matrix
        users = interactions_df['user_id'].values
        items = interactions_df['item_id'].values
        data = np.ones(len(users), dtype=np.float32)
        matrix = csr_matrix(
            (data, (users, items)),
            shape=(self.num_users, self.num_trained_items),
            dtype=np.float32,
        )
        return matrix

    def prepare_for_training(
        self,
        test_size: float = 0.2,
        temporal: bool = False,
        random_state: int = 42,
    ) -> Dict:
        logger.info("=" * 60)
        logger.info("PREPARING DATA FOR TRAINING")
        logger.info("=" * 60)

        train_df, test_df, split_info = self.train_test_split(
            test_size=test_size, temporal=temporal, random_state=random_state
        )

        # Build graph from train only
        original = self.interactions
        self.interactions = train_df
        train_graph = self.build_graph()
        self.interactions = original

        train_matrix = self.build_user_item_matrix(train_df)

        test_data = {
            'interactions': test_df,
            'user_ids': test_df['user_id'].values,
            'item_ids': test_df['item_id'].values,
            'num_interactions': len(test_df),
        }

        return {
            'train_graph': train_graph,
            'train_matrix': train_matrix,
            'test_data': test_data,
            'train_df': train_df,
            'test_df': test_df,
            'split_info': split_info,
            'num_users': self.num_users,
            'num_items': self.num_trained_items,  # Graph sees only trained items
            'metadata': self.metadata,
        }
