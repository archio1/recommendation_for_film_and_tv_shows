import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import argparse
import logging
from datetime import datetime
import shutil
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MovieDatasetProcessor:
    """

    """

    DTYPE_INTERACTIONS = {
        'user_id': 'uint32',
        'item_id': 'uint32',
        'rating': 'float32',
        'timestamp': 'uint32'
    }

    def __init__(
            self,
            data_dir: Path,
            top_n_movies: int = 3000,  # УМЕНЬШЕНО с 10000!
            top_n_tv: int = 1000,  # УМЕНЬШЕНО с 5000!
            min_year: Optional[int] = None,
            languages: Optional[List[str]] = None,
            min_user_interactions: int = 30,  # УВЕЛИЧЕНО для фильтрации
            min_item_interactions: int = 50,  # УВЕЛИЧЕНО для популярных
            rating_threshold: float = 4.0,
            max_interactions: int = 3_000_000  # НОВЫЙ: лимит взаимодействий
    ):
        self.data_raw = data_dir / 'raw'
        self.data_processed = data_dir / 'processed'
        self.data_processed.mkdir(parents=True, exist_ok=True)

        # Параметры фильтрации
        self.top_n_movies = top_n_movies
        self.top_n_tv = top_n_tv
        self.min_year = min_year
        self.languages = languages or ['en']
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
        self.rating_threshold = rating_threshold
        self.max_interactions = max_interactions  # НОВЫЙ параметр

        logger.info("=" * 70)
        logger.info("MOVIE DATASET PROCESSOR (FAST VERSION)")
        logger.info("=" * 70)
        logger.info(f"Топ фильмов: {top_n_movies:,}, Топ сериалов: {top_n_tv:,}")
        logger.info(f"Языки: {languages}, Мин. год: {min_year or 'без ограничений'}")
        logger.info(f"Пороги: user≥{min_user_interactions}, item≥{min_item_interactions}, rating≥{rating_threshold}")
        logger.info(f"МАКС взаимодействий: {max_interactions:,}")

    # ========================================================================
    # ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
    # ========================================================================

    def _parse_genres(self, value) -> List[str]:
        """Парсит жанры из любого формата"""
        if pd.isna(value):
            return []

        try:
            value_str = str(value).strip()
            if not value_str or value_str in ['[]', 'nan', 'None']:
                return []

            # "Action, Sci-Fi, Drama"
            if ',' in value_str and '{' not in value_str:
                return [g.strip() for g in value_str.split(',') if g.strip() and g.strip().lower() != 'unknown']

            # "Action|Adventure"
            if '|' in value_str and '{' not in value_str:
                return [g.strip() for g in value_str.split('|') if
                        g.strip() and g.strip().lower() not in ['unknown', '(no genres listed)']]

            # [{"id": 28, "name": "Action"}]
            if value_str.startswith('[') and '{' in value_str:
                import ast
                parsed = ast.literal_eval(value_str)
                if isinstance(parsed, list):
                    return [g['name'] for g in parsed if isinstance(g, dict) and 'name' in g]

            # numpy array / list
            if isinstance(value, (list, np.ndarray)):
                return [str(g).strip() for g in value if str(g).strip() and str(g).lower() != 'unknown']

            # Одиночный жанр
            if value_str.lower() not in ['unknown', '(no genres listed)']:
                return [value_str]

            return []
        except:
            return []

    def _parse_tv_genres(self, row) -> List[str]:
        """Парсит жанры из TV Series (колонки genres[0].name, ...)"""
        genres = []
        for i in range(8):
            col = f'genres[{i}].name'
            if col in row.index and pd.notna(row[col]):
                genre = str(row[col]).strip()
                if genre and genre.lower() != 'unknown':
                    genres.append(genre)
        return genres

    def _ensure_list_genres(self, x):
        """Конвертирует genres в list если нужно"""
        if isinstance(x, (list, tuple)):
            return list(x)
        elif isinstance(x, np.ndarray):
            return x.tolist()
        elif isinstance(x, str):
            import ast
            try:
                return ast.literal_eval(x) if x.startswith('[') else [x]
            except:
                return []
        return []

    # ========================================================================
    # 1. ЗАГРУЗКА
    # ========================================================================

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Загружает сырые данные"""
        logger.info("\n" + "=" * 70)
        logger.info("1. ЗАГРУЗКА ДАННЫХ")
        logger.info("=" * 70)

        ml_ratings = pd.read_csv(
            self.data_raw / 'ml-32m/ratings.csv',
            dtype={'userId': 'uint32', 'movieId': 'uint32', 'rating': 'float32', 'timestamp': 'uint32'}
        )
        ml_movies = pd.read_csv(self.data_raw / 'ml-32m/movies.csv')
        ml_links = pd.read_csv(self.data_raw / 'ml-32m/links.csv')

        ml_ratings.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
        ml_movies.columns = ['movie_id', 'title', 'genres']
        ml_links.columns = ['movie_id', 'imdb_id', 'tmdb_id']

        logger.info(f"MovieLens: {len(ml_ratings):,} рейтингов, {len(ml_movies):,} фильмов")

        tmdb_movies = pd.read_csv(self.data_raw / 'TMDB_movie_dataset_v11.csv', low_memory=False)
        tv_series = pd.read_csv(self.data_raw / 'tv_series.csv', low_memory=False)

        logger.info(f"TMDB: {len(tmdb_movies):,}, TV: {len(tv_series):,}")

        return ml_ratings, ml_movies, ml_links, tmdb_movies, tv_series

    # ========================================================================
    # 2. ОЧИСТКА TMDB
    # ========================================================================

    def clean_tmdb_movies(self, tmdb: pd.DataFrame) -> pd.DataFrame:
        """Очистка TMDB фильмов с фильтрами качества"""
        logger.info("\n" + "=" * 70)
        logger.info("2. ОЧИСТКА TMDB MOVIES")
        logger.info("=" * 70)

        df = tmdb.copy()
        initial = len(df)

        # Переименовываем
        df = df.rename(columns={'id': 'tmdb_id'})

        # Фильтры
        if self.languages and 'original_language' in df.columns:
            df = df[df['original_language'].isin(self.languages)]
            logger.info(f"Язык {self.languages}: {len(df):,}")

        if 'overview' in df.columns:
            df = df[df['overview'].notna()]
            df = df[~df['overview'].str.lower().str.contains('no overview|no description', na=False, regex=True)]
            df = df[df['overview'].str.strip() != '']
            logger.info(f"С описанием: {len(df):,}")

        if 'genres' in df.columns:
            df['genres'] = df['genres'].apply(self._parse_genres)
            df = df[df['genres'].apply(lambda x: len(x) > 0)]
            logger.info(f"С жанрами: {len(df):,}")

        if 'vote_count' in df.columns and 'vote_average' in df.columns:
            df = df[(df['vote_count'] >= 100) & (df['vote_average'] > 0)]  # УВЕЛИЧЕНО с 50!
            logger.info(f"С оценками (≥100): {len(df):,}")

        if 'release_date' in df.columns:
            df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
            if self.min_year:
                df = df[df['year'] >= self.min_year]
            else:
                df = df[(df['year'] >= 1900) & (df['year'] <= 2025)]
            logger.info(f"Годы OK: {len(df):,}")

        # Дубликаты
        df = df.drop_duplicates(subset=['title', 'year'], keep='first')

        df['type'] = 'movie'

        logger.info(f"✅ Итого: {len(df):,} ({len(df) / initial * 100:.1f}%)")
        return df

    # ========================================================================
    # 3. ОЧИСТКА TV
    # ========================================================================

    def clean_tv_series(self, tv: pd.DataFrame) -> pd.DataFrame:
        """Очистка TV Series с топ-N выбором"""
        logger.info("\n" + "=" * 70)
        logger.info("3. ОЧИСТКА TV SERIES")
        logger.info("=" * 70)

        df = tv.copy()
        initial = len(df)

        # Переименовываем
        df = df.rename(columns={'id': 'tmdb_id', 'name': 'title'})

        # Описание
        overview_col = 'overview' if 'overview' in df.columns else 'description'
        if overview_col in df.columns:
            df = df[df[overview_col].notna()]
            df = df[~df[overview_col].str.lower().str.contains('no overview|no description', na=False, regex=True)]
            df = df[df[overview_col].str.strip() != '']
            df = df.rename(columns={overview_col: 'overview'})
            logger.info(f"С описанием: {len(df):,}")

        # Жанры
        df['genres'] = df.apply(self._parse_tv_genres, axis=1)
        df = df[df['genres'].apply(lambda x: len(x) > 0)]
        logger.info(f"С жанрами: {len(df):,}")

        # Оценки
        if 'vote_count' in df.columns and 'vote_average' in df.columns:
            df = df[(df['vote_count'] >= 50) & (df['vote_average'] > 0)]  # УВЕЛИЧЕНО с 20!
            logger.info(f"С оценками (≥50): {len(df):,}")

        # Год
        date_col = 'first_air_date' if 'first_air_date' in df.columns else 'release_date'
        if date_col in df.columns:
            df['year'] = pd.to_datetime(df[date_col], errors='coerce').dt.year
            if self.min_year:
                df = df[df['year'] >= self.min_year]
            else:
                df = df[(df['year'] >= 1950) & (df['year'] <= 2025)]

        df['type'] = 'tv'

        # Топ по популярности
        if 'popularity' in df.columns:
            df = df.sort_values('popularity', ascending=False)
        elif 'vote_count' in df.columns:
            df = df.sort_values('vote_count', ascending=False)

        df = df.head(self.top_n_tv)

        logger.info(f"✅ Топ {self.top_n_tv}: {len(df):,}")
        return df

    # ========================================================================
    # 4. ОБЪЕДИНЕНИЕ МЕТАДАННЫХ
    # ========================================================================

    def merge_metadata(self, tmdb: pd.DataFrame, tv: pd.DataFrame) -> pd.DataFrame:
        """Объединяет фильмы и сериалы в единый датасет"""
        logger.info("\n" + "=" * 70)
        logger.info("4. ОБЪЕДИНЕНИЕ МЕТАДАННЫХ")
        logger.info("=" * 70)

        # Выбираем общие колонки
        common_cols = ['tmdb_id', 'title', 'type', 'year', 'genres', 'overview',
                       'vote_average', 'vote_count', 'popularity']

        tmdb_select = tmdb[[c for c in common_cols if c in tmdb.columns]].copy()
        tv_select = tv[[c for c in common_cols if c in tv.columns]].copy()

        metadata = pd.concat([tmdb_select, tv_select], ignore_index=True)

        logger.info(f"Фильмы: {len(tmdb_select):,}, Сериалы: {len(tv_select):,}")
        logger.info(f"✅ Итого: {len(metadata):,}")

        return metadata

    # ========================================================================
    # 5. СВЯЗЫВАНИЕ С MOVIELENS
    # ========================================================================

    def link_to_movielens(
            self,
            ml_movies: pd.DataFrame,
            ml_links: pd.DataFrame,
            metadata: pd.DataFrame
    ) -> pd.DataFrame:
        """Связывает MovieLens с метаданными через TMDB ID"""
        logger.info("\n" + "=" * 70)
        logger.info("5. СВЯЗЫВАНИЕ С MOVIELENS")
        logger.info("=" * 70)

        # MovieLens + links
        ml_with_links = ml_movies.merge(ml_links[['movie_id', 'tmdb_id']], on='movie_id', how='left')

        # MovieLens + metadata
        linked = ml_with_links.merge(metadata, on='tmdb_id', how='inner', suffixes=('_ml', ''))

        # Берём название и жанры из метаданных
        if 'title' not in linked.columns and 'title_ml' in linked.columns:
            linked['title'] = linked['title_ml']

        logger.info(f"Совпадений: {len(linked):,} ({len(linked) / len(ml_movies) * 100:.1f}%)")
        logger.info(f"✅ MovieLens с метаданными: {len(linked):,}")

        return linked

    # ========================================================================
    # 6. ФИЛЬТРАЦИЯ ВЗАИМОДЕЙСТВИЙ (КЛЮЧЕВОЕ!)
    # ========================================================================

    def filter_interactions(
            self,
            ml_ratings: pd.DataFrame,
            ml_movies_linked: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """

        """
        logger.info("\n" + "=" * 70)
        logger.info("6. ФИЛЬТРАЦИЯ ВЗАИМОДЕЙСТВИЙ (АГРЕССИВНАЯ)")
        logger.info("=" * 70)

        # 1. Топ-N популярных фильмов
        logger.info(f"\n1) Топ {self.top_n_movies} популярных...")
        movie_counts = ml_ratings.groupby('movie_id').size().sort_values(ascending=False)
        top_movies = movie_counts.head(self.top_n_movies).index

        ratings = ml_ratings[ml_ratings['movie_id'].isin(top_movies)].copy()
        logger.info(f"   Рейтингов: {len(ratings):,} ({len(ratings) / len(ml_ratings) * 100:.1f}%)")

        # 2. Положительные
        logger.info(f"\n2) Фильтр положительных (≥{self.rating_threshold})...")
        ratings = ratings[ratings['rating'] >= self.rating_threshold]
        ratings = ratings.drop_duplicates(subset=['user_id', 'movie_id'], keep='last')
        logger.info(f"   Рейтингов: {len(ratings):,}")

        # 3. Cold start (АГРЕССИВНЫЙ)
        logger.info(f"\n3) Cold start (user≥{self.min_user_interactions}, item≥{self.min_item_interactions})...")
        for i in range(10):
            prev = len(ratings)

            user_counts = ratings.groupby('user_id').size()
            ratings = ratings[ratings['user_id'].isin(user_counts[user_counts >= self.min_user_interactions].index)]

            item_counts = ratings.groupby('movie_id').size()
            ratings = ratings[ratings['movie_id'].isin(item_counts[item_counts >= self.min_item_interactions].index)]

            if len(ratings) == prev:
                logger.info(f"   Сошлось за {i + 1} итераций: {len(ratings):,}")
                break

        # 4. НОВОЕ: Лимит на количество взаимодействий
        if len(ratings) > self.max_interactions:
            logger.info(f"\n4) Применение лимита {self.max_interactions:,}...")

            # Стратегия: сохраняем наиболее активных пользователей
            user_counts = ratings.groupby('user_id').size().sort_values(ascending=False)

            # Берём топ пользователей, пока не наберём лимит
            cumsum = user_counts.cumsum()
            num_users_to_keep = (cumsum <= self.max_interactions).sum()
            top_users = user_counts.head(num_users_to_keep).index

            ratings = ratings[ratings['user_id'].isin(top_users)]
            logger.info(f"   Оставлено: {len(ratings):,} взаимодействий от {len(top_users):,} пользователей")

        # 5. Синхронизация с метаданными
        logger.info(f"\n5) Синхронизация с метаданными...")
        used_movies = ratings['movie_id'].unique()
        metadata = ml_movies_linked[ml_movies_linked['movie_id'].isin(used_movies)].copy()

        # Убираем дубликаты по movie_id
        metadata = metadata.drop_duplicates(subset='movie_id', keep='first')

        # Убираем взаимодействия для фильмов без метаданных
        valid_movies = metadata['movie_id'].unique()
        ratings = ratings[ratings['movie_id'].isin(valid_movies)]

        # Сортируем metadata для удобства
        metadata = metadata.sort_values('movie_id').reset_index(drop=True)

        logger.info(f"\n✅ Результат:")
        logger.info(f"   Рейтинги: {len(ratings):,}")
        logger.info(f"   Пользователи: {ratings['user_id'].nunique():,}")
        logger.info(f"   Фильмы: {len(metadata):,}")

        # Проверка на целевой размер
        if len(ratings) > self.max_interactions * 1.2:
            logger.warning(f"⚠️  Превышен лимит! Нужна дополнительная фильтрация")

        return ratings, metadata

    # ========================================================================
    # 7. СОЗДАНИЕ UNIFIED ID
    # ========================================================================

    def create_unified_ids(
            self,
            interactions: pd.DataFrame,
            metadata: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Создаёт последовательные ID (0, 1, 2, ...) для users и items"""
        logger.info("\n" + "=" * 70)
        logger.info("7. СОЗДАНИЕ UNIFIED ID")
        logger.info("=" * 70)

        # Item ID: movie_id -> item_id (0, 1, 2, ...)
        unique_movies = sorted(metadata['movie_id'].unique())
        ml_to_item = {ml_id: idx for idx, ml_id in enumerate(unique_movies)}

        metadata['item_id'] = metadata['movie_id'].map(ml_to_item).astype('uint32')
        interactions['item_id'] = interactions['movie_id'].map(ml_to_item)

        # Убираем NaN (на всякий случай)
        interactions = interactions[interactions['item_id'].notna()].copy()
        interactions['item_id'] = interactions['item_id'].astype('uint32')

        logger.info(f"item_id: 0 → {len(unique_movies) - 1}")

        # User ID: старый user_id -> новый user_id (0, 1, 2, ...)
        unique_users = sorted(interactions['user_id'].unique())
        user_to_idx = {old_id: new_id for new_id, old_id in enumerate(unique_users)}

        interactions['user_id'] = interactions['user_id'].map(user_to_idx).astype('uint32')

        logger.info(f"user_id: 0 → {len(unique_users) - 1}")

        # Маппинг
        mapping = {
            'ml_to_item': {int(k): int(v) for k, v in ml_to_item.items()},
            'item_to_ml': {int(v): int(k) for k, v in ml_to_item.items()},
            'user_to_idx': {int(k): int(v) for k, v in user_to_idx.items()},
            'idx_to_user': {int(v): int(k) for k, v in user_to_idx.items()},
            'num_users': len(unique_users),
            'num_items': len(unique_movies)
        }

        logger.info(f"✅ {mapping['num_users']:,} users, {mapping['num_items']:,} items")

        return interactions, metadata, mapping

    # ========================================================================
    # 8. СОХРАНЕНИЕ С ВАЛИДАЦИЕЙ
    # ========================================================================

    def save(
            self,
            interactions: pd.DataFrame,
            metadata: pd.DataFrame,
            mapping: Dict
    ):
        """Сохраняет финальные данные с валидацией"""
        logger.info("\n" + "=" * 70)
        logger.info("8. ВАЛИДАЦИЯ И СОХРАНЕНИЕ")
        logger.info("=" * 70)

        # Валидация
        logger.info("\nВалидация...")

        # Проверка 1: ID последовательны
        assert interactions['user_id'].min() == 0
        assert interactions['user_id'].max() == interactions['user_id'].nunique() - 1
        assert interactions['item_id'].min() == 0
        assert interactions['item_id'].max() == interactions['item_id'].nunique() - 1
        assert metadata['item_id'].min() == 0
        assert metadata['item_id'].max() == metadata['item_id'].nunique() - 1
        logger.info("✓ ID последовательны")

        # Проверка 2: Нет дубликатов item_id в metadata
        assert metadata['item_id'].nunique() == len(metadata)
        logger.info("✓ Нет дубликатов в metadata")

        # Проверка 3: Полное соответствие item_id
        int_items = set(interactions['item_id'].unique())
        meta_items = set(metadata['item_id'].unique())
        assert int_items == meta_items
        logger.info("✓ item_id полностью совпадают")

        # Проверка 4: Genres в формате list
        if 'genres' in metadata.columns:
            metadata['genres'] = metadata['genres'].apply(self._ensure_list_genres)
            logger.info("✓ Genres конвертированы в list")

        # Выбираем финальные колонки
        essential_cols = ['item_id', 'movie_id', 'title', 'type', 'year', 'genres',
                          'overview', 'vote_average', 'vote_count', 'popularity', 'tmdb_id']
        metadata = metadata[[c for c in essential_cols if c in metadata.columns]].copy()

        # Резервная копия
        backup_dir = self.data_processed / 'backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir.mkdir(parents=True, exist_ok=True)

        for file in ['interactions_final.parquet', 'items_metadata_final.parquet', 'id_mapping.json']:
            src = self.data_processed / file
            if src.exists():
                shutil.copy2(src, backup_dir / file)

        logger.info(f"\n✓ Резервная копия: {backup_dir}")

        # Оптимизация типов
        for col, dtype in self.DTYPE_INTERACTIONS.items():
            if col in interactions.columns:
                interactions[col] = interactions[col].astype(dtype)

        # Сохранение
        interactions.to_parquet(self.data_processed / 'interactions_final.parquet', index=False)
        metadata.to_parquet(self.data_processed / 'items_metadata_final.parquet', index=False)

        with open(self.data_processed / 'id_mapping.json', 'w') as f:
            json.dump(mapping, f, indent=2)

        # Статистика
        density = len(interactions) / (mapping['num_users'] * mapping['num_items'])
        stats = {
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'top_n_movies': self.top_n_movies,
                'top_n_tv': self.top_n_tv,
                'min_year': self.min_year,
                'languages': self.languages,
                'min_user_interactions': self.min_user_interactions,
                'min_item_interactions': self.min_item_interactions,
                'rating_threshold': self.rating_threshold,
                'max_interactions': self.max_interactions
            },
            'interactions': {
                'total': len(interactions),
                'users': mapping['num_users'],
                'items': mapping['num_items'],
                'density': float(density)
            },
            'metadata': {
                'total': len(metadata),
                'movies': int((metadata['type'] == 'movie').sum()) if 'type' in metadata.columns else 0,
                'tv_shows': int((metadata['type'] == 'tv').sum()) if 'type' in metadata.columns else 0
            }
        }

        with open(self.data_processed / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info("\n" + "=" * 70)
        logger.info("ФИНАЛЬНАЯ СТАТИСТИКА")
        logger.info("=" * 70)
        logger.info(f"Взаимодействия: {stats['interactions']['total']:,}")
        logger.info(f"  Пользователи: {stats['interactions']['users']:,}")
        logger.info(f"  Айтемы: {stats['interactions']['items']:,}")
        logger.info(f"  Плотность: {stats['interactions']['density']:.6%}")
        logger.info(f"Метаданные: {stats['metadata']['total']:,}")
        logger.info(f"  Фильмы: {stats['metadata']['movies']:,}")
        logger.info(f"  Сериалы: {stats['metadata']['tv_shows']:,}")
        logger.info("=" * 70)
        logger.info("✅ ГОТОВО! Данные сохранены и готовы для обучения")

    # ========================================================================
    # ГЛАВНЫЙ PIPELINE
    # ========================================================================

    def process(self) -> bool:
        """Запускает полный pipeline обработки"""
        try:
            # Pipeline
            ml_ratings, ml_movies, ml_links, tmdb, tv = self.load_data()

            tmdb_clean = self.clean_tmdb_movies(tmdb)
            tv_clean = self.clean_tv_series(tv)

            metadata = self.merge_metadata(tmdb_clean, tv_clean)

            ml_linked = self.link_to_movielens(ml_movies, ml_links, metadata)

            interactions, metadata_final = self.filter_interactions(ml_ratings, ml_linked)

            interactions, metadata_final, mapping = self.create_unified_ids(interactions, metadata_final)

            self.save(interactions, metadata_final, mapping)

            return True

        except Exception as e:
            logger.error(f"\n❌ ОШИБКА: {e}", exc_info=True)
            return False


def main():
    """Главная функция"""
    data_dir = Path('../../../data').resolve()

    if not data_dir.exists():
        logger.error(f"Директория не найдена: {data_dir}")
        return 1

    processor = MovieDatasetProcessor(
        data_dir=data_dir,
        top_n_movies=12000,
        top_n_tv=2000,
        min_year=None,
        languages=['en'],
        min_user_interactions=10,
        min_item_interactions=10,
        rating_threshold=3.5,
        max_interactions=10_000_000
    )

    success = processor.process()
    exit(0 if success else 1)


if __name__ == "__main__":
    main()