"""
check_data.py — Dataset Validator v3.1 (Interaction Breakdown Edition)
"""

import json
import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import Counter

import numpy as np
import pandas as pd

from recommendation_system.paths import PROCESSED_DIR

# Жанры, которые мы ожидаем увидеть
VALID_GENRES: Set[str] = {
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
    "Romance", "Science Fiction", "Sci-Fi", "Thriller", "TV Movie", "War",
    "Western", "Action & Adventure", "Kids", "Sci-Fi & Fantasy", "War & Politics"
}

class DatasetValidator:
    def __init__(self, processed_dir: Path):
        self.dir = processed_dir
        self.errors = []
        self.warnings = []
        self.info = []

    def _err(self, msg: str): self.errors.append(msg); print(f"  ❌ {msg}")
    def _warn(self, msg: str): self.warnings.append(msg); print(f"  ⚠️  {msg}")
    def _ok(self, msg: str): self.info.append(msg); print(f"  ✅ {msg}")

    def run(self) -> bool:
        print(f"\n{'=' * 70}\nDATASET VALIDATOR v3.1 (Breakdown Edition)\n{'=' * 70}")

        # 0. Загрузка файлов
        try:
            with open(self.dir / 'id_mapping.json', 'r') as f:
                mapping = json.load(f)
            meta = pd.read_parquet(self.dir / 'items_metadata_final.parquet')
            ints = pd.read_parquet(self.dir / 'interactions_final.parquet')
        except Exception as e:
            self._err(f"Failed to load files: {e}")
            return False

        # 1. Проверка маппинга
        self._section("1. ID Mapping & Coverage")
        num_trained = mapping['num_trained_items']
        num_items = mapping['num_items']
        self._ok(f"Users: {mapping['num_users']:,}")
        self._ok(f"Items: {num_items:,} (Trained: {num_trained:,})")

        # 2. Проверка метаданных (Особенно TV и Keywords)
        self._section("2. Metadata & Content")

        # Проверка ключевых слов
        if 'keywords' in meta.columns:
            kw_filled = meta['keywords'].apply(lambda x: len(x) > 0).sum()
            kw_avg = meta['keywords'].apply(len).mean() if kw_filled > 0 else 0
            self._ok(f"Keywords: {kw_filled:,} items have keywords (Avg: {kw_avg:.1f} per item)")
            if kw_filled < len(meta) * 0.5:
                self._warn("More than 50% of items are missing keywords")
        else:
            self._err("Column 'keywords' is missing!")

        # Проверка типов в обучении
        trained_meta = meta[meta['item_id'] < num_trained]
        tv_trained_list = trained_meta[trained_meta['type'] == 'tv']['item_id'].unique()
        movie_trained_list = trained_meta[trained_meta['type'] == 'movie']['item_id'].unique()

        self._ok(f"Training set items: {len(movie_trained_list):,} movies + {len(tv_trained_list):,} TV shows")

        if len(tv_trained_list) == 0:
            self._err("ZERO TV shows in training set! Amazon integration failed?")

        # 3. Проверка взаимодействий (Breakdown)
        self._section("3. Interactions Breakdown")
        total_ints = len(ints)
        self._ok(f"Total interactions: {total_ints:,}")

        # Считаем взаимодействия отдельно для фильмов и сериалов
        # Мапим item_id на тип из метаданных
        type_map = meta.set_index('item_id')['type'].to_dict()
        ints['temp_type'] = ints['item_id'].map(type_map)

        counts = ints['temp_type'].value_counts().to_dict()
        movie_ints = counts.get('movie', 0)
        tv_ints = counts.get('tv', 0)

        self._ok(f"Movie interactions: {movie_ints:,} ({movie_ints/total_ints:.11%})")
        self._ok(f"TV interactions:    {tv_ints:,} ({tv_ints/total_ints:.11%})")

        if tv_ints > 0:
            avg_tv = tv_ints / len(tv_trained_list) if len(tv_trained_list) > 0 else 0
            self._ok(f"Avg interactions per trained TV show: {avg_tv:.1f}")
        else:
            self._warn("TV shows have no interactions in the final table!")

        # Проверка на дубликаты
        dupes = ints.duplicated(subset=['user_id', 'item_id']).sum()
        if dupes > 0:
            self._err(f"Found {dupes:,} duplicate user-item pairs!")
        else:
            self._ok("No duplicate interactions")

        # Проверка разброса рейтингов
        r_min, r_max = ints['rating'].min(), ints['rating'].max()
        self._ok(f"Rating range: {r_min} - {r_max}")

        # 4. Проверка консистентности
        self._section("4. Cross-Consistency")
        ints_items = set(ints['item_id'].unique())
        self._ok(f"Interactions cover {len(ints_items):,} unique items (trained pool: {num_trained:,})")

        # Итог
        print(f"\n{'=' * 70}\nSUMMARY: {len(self.errors)} Errors, {len(self.warnings)} Warnings\n{'=' * 70}")
        return len(self.errors) == 0

    def _section(self, name: str):
        print(f"\n--- {name} ---")

def main():
    # Путь к обработанным данным
    validator = DatasetValidator(PROCESSED_DIR)
    sys.exit(0 if validator.run() else 1)

if __name__ == "__main__":
    main()