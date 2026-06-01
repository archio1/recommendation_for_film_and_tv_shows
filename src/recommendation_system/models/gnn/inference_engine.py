"""
inference_engine.py — LightGCN model loader and per-domain inference helper.

Loads a trained LightGCN checkpoint, restores embedding dimensions from the
state_dict, clips metadata to trained items, and exposes the model + features
to higher-level consumers (`UniversalSearchEngine`, `DualDomainEngine`,
graph-neighbor visualization, tests).

Extracted from `trainer_gui.py` so the bot, tests, and visualization tools
no longer have to import GUI code.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Mirror the sibling-import pattern used by trainer.py / check_similarity.py:
# put the gnn/ directory on sys.path so `from lightgcn import LightGCN` works
# both when run as a script and when imported as a package.
sys.path.append(str(Path(__file__).parent))

from lightgcn import LightGCN  # noqa: E402


STOP_WORDS = {'the', 'a', 'an', 'in', 'of', 'and', 'to', 'for', 'my', 'is', 'on'}


class InferenceEngine:
    def __init__(self, data_dir: Path, model_path: Path, device: str = 'cpu') -> None:
        self.data_dir = data_dir
        self.model_path = model_path
        self.device = device
        self.model = None
        self.metadata = None
        self.id_mapping = None
        self.item_features = None  # Для гибридной модели
        self.is_loaded = False
        self._pop_penalty = None  # cached popularity-debias tensor (lazy)

    def _prepare_features(self, expected_genres):
        """Подготовка фичей, строго подогнанная под размер весов модели"""
        df = self.metadata.sort_values('item_id')

        all_genres_in_data = set()
        for genres in df['genres']:
            if isinstance(genres, (list, np.ndarray)):
                all_genres_in_data.update(genres)

        sorted_genres = sorted(list(all_genres_in_data))
        # Обрезаем/дополняем список жанров под модель
        final_genres = sorted_genres[:expected_genres]
        if len(final_genres) < expected_genres:
            final_genres += [f"dummy_{i}" for i in range(expected_genres - len(final_genres))]

        genre_map = {g: i for i, g in enumerate(final_genres)}

        genre_matrix = torch.zeros((len(df), expected_genres), device=self.device)
        for idx, row in df.iterrows():
            item_id = row['item_id']
            if item_id >= len(df):
                continue
            gs = row['genres']
            if isinstance(gs, (list, np.ndarray)):
                indices = [genre_map[g] for g in gs if g in genre_map]
                if indices:
                    genre_matrix[item_id, indices] = 1.0

        years = df['year'].fillna(2000).values
        years_norm = (years - 1990) / 30.0
        year_tensor = torch.tensor(years_norm, dtype=torch.float32, device=self.device).view(-1, 1)

        return (genre_matrix, year_tensor), expected_genres

    def load_resources(self):
        try:
            with open(self.data_dir / 'id_mapping.json', 'r') as f:
                self.id_mapping = json.load(f)

            full_metadata = pd.read_parquet(self.data_dir / 'items_metadata_final.parquet')

            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint['model_state_dict']

            if 'item_id_embedding.weight' in state_dict:
                model_num_items = state_dict['item_id_embedding.weight'].shape[0]
            else:
                model_num_items = state_dict['item_embedding.weight'].shape[0]

            # Оставляем только те записи, которые есть в весах модели
            self.metadata = full_metadata[full_metadata['item_id'] < model_num_items].copy()
            print(f"📊 InferenceEngine: метаданные ограничены до {model_num_items} (trained items)")

            embedding_dim = state_dict['user_embedding.weight'].shape[1]
            num_users = state_dict['user_embedding.weight'].shape[0]
            # genre_encoder есть только в гибридных моделях (num_genres > 0).
            # Чистые LightGCN (новый trainer.py) — без content-features.
            if 'genre_encoder.weight' in state_dict:
                num_genres = state_dict['genre_encoder.weight'].shape[1]
            else:
                num_genres = 0

            num_layers = 2
            if 'layer_weights' in state_dict:
                num_layers = state_dict['layer_weights'].shape[0] - 1

            if num_genres > 0:
                self.item_features, _ = self._prepare_features(num_genres)
            else:
                self.item_features = None

            self.model = LightGCN(
                num_users=num_users,
                num_items=model_num_items,
                num_genres=num_genres,
                embedding_dim=embedding_dim,
                num_layers=num_layers,
            )

            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            self.is_loaded = True
            version = self.model_path.stem.split('_')[-1]
            return True, f"Модель {version} ({num_layers} layers) загружена"
        except Exception as e:
            print(traceback.format_exc())
            return False, f"Ошибка загрузки: {e}"

    def search_movies(self, query: str, limit: int = 5):
        if not self.is_loaded or not query:
            return []

        query = query.lower().strip()

        mask = self.metadata['title'].str.lower().str.contains(query, na=False)
        matches = self.metadata[mask].copy()

        if matches.empty:
            return []

        if 'vote_count' in matches.columns:
            matches = matches.sort_values('vote_count', ascending=False)
        elif 'popularity' in matches.columns:
            matches = matches.sort_values('popularity', ascending=False)

        matches['exact_match'] = matches['title'].str.lower() == query
        matches = matches.sort_values(
            ['exact_match', 'vote_count' if 'vote_count' in matches.columns else 'popularity'],
            ascending=[False, False],
        )

        return matches.head(limit)[['title', 'year', 'item_id']].to_dict('records')

    def _popularity_penalty(self, n_items: int):
        """Per-item popularity penalty in [0, 1], aligned to item_id index.

        Normalized log1p(vote_count) — the same signal the popularity-bias
        tests treat as "popular". Subtracted from LightGCN cosine scores so
        globally-popular items (central in the co-watch graph, hence near
        every user vector) stop dominating every recommendation list.
        Cached after first build.
        """
        if self._pop_penalty is not None and self._pop_penalty.shape[0] == n_items:
            return self._pop_penalty
        vc = np.zeros(n_items, dtype=np.float64)
        if 'vote_count' in self.metadata.columns:
            ids = self.metadata['item_id'].to_numpy()
            counts = self.metadata['vote_count'].fillna(0).to_numpy()
            in_range = ids < n_items
            vc[ids[in_range]] = counts[in_range]
        pen = np.log1p(vc)
        mx = pen.max()
        if mx > 0:
            pen = pen / mx
        self._pop_penalty = torch.tensor(pen, dtype=torch.float32, device=self.device)
        return self._pop_penalty

    def get_recommendations(self, liked_item_ids: list, top_k: int = 8, popularity_debias: float = 0.0):
        if not self.is_loaded or not liked_item_ids:
            return []

        selected_titles = self.metadata[
            self.metadata['item_id'].isin(liked_item_ids)
        ]['title'].str.lower().tolist()
        item_emb = self.model.get_item_embedding(self.item_features).detach()

        selected_indices = torch.tensor(liked_item_ids).to(self.device)
        selected_vectors = item_emb[selected_indices]
        user_vector = torch.mean(selected_vectors, dim=0).unsqueeze(0)

        user_vector = F.normalize(user_vector, p=2, dim=1)
        item_emb_norm = F.normalize(item_emb, p=2, dim=1)

        scores = torch.matmul(user_vector, item_emb_norm.t()).squeeze(0)
        scores[selected_indices] = -float('inf')

        # Popularity de-bias: push down globally-popular items so the list
        # reflects "popular among users like you", not the IMDb top-250.
        if popularity_debias and popularity_debias > 0:
            scores = scores - popularity_debias * self._popularity_penalty(scores.shape[0])

        candidate_count = 100
        top_scores, top_indices = torch.topk(scores, min(candidate_count, len(scores)))
        top_indices = top_indices.cpu().numpy()

        recs = []
        for idx in top_indices:
            if len(recs) >= top_k:
                break

            row = self.metadata[self.metadata['item_id'] == idx].iloc[0]
            rec_title = str(row['title']).lower()

            # Длина >= 3 защищает от ложных срабатываний на коротких токенах ("saw" → "Saw II", и т.п.)
            words = rec_title.replace(':', ' ').replace('-', ' ').split()
            base_rec = None
            for word in words:
                if word not in STOP_WORDS and len(word) >= 3:
                    base_rec = word
                    break

            is_sequel = False
            if base_rec:
                for sel_title in selected_titles:
                    if base_rec in sel_title:
                        is_sequel = True
                        break

            if is_sequel:
                continue

            search_query = f"{row['title']} {int(row['year']) if row['year'] else ''}".replace(" ", "+")
            imdb_link = f"https://www.imdb.com/find?q={search_query}"

            recs.append({
                'item_id': int(idx),
                'title': row['title'],
                'year': row['year'],
                'genres': row['genres'],
                'imdb_url': imdb_link,
            })

        return recs
