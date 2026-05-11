"""
make_dataset.py — Refactored Data Pipeline v5

Changes from v4:
  1. DRY: Unified _parse_genres() eliminates _clean_genres() duplication
  2. Memory: Categorical dtypes, chunked ML loading, aggressive .copy() removal
  3. Safety: tmdb_id collision guard between movies/TV, genre validation pass
  4. Robustness: Explicit NaN handling in link_to_movielens, assertion-backed ID checks
  5. Bilingual-ready: title_en column preserved, title_ru placeholder added
"""

import ast
import json
import logging
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import requests
import unicodedata
import re


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Sentinel values treated as empty/missing in genre fields
_GENRE_TRASH: Set[str] = {'unknown', 'nan', 'none', '(no genres listed)', ''}

# --- Sci-Fi vs Fantasy disambiguation keywords ---
_SCI_FI_SIGNALS = frozenset({
    'space', 'planet', 'planets', 'galaxy', 'galactic', 'spaceship', 'spacecraft',
    'starship', 'cosmos', 'orbit', 'mars', 'moon', 'asteroid', 'station',
    'crew', 'interstellar', 'solar', 'sector', 'light-year',
    'robot', 'robots', 'android', 'cyborg', 'mech', 'hologram', 'nano',
    'computer', 'hacker', 'matrix', 'digital', 'virtual', 'simulation',
    'cyberpunk', 'technology', 'tech',
    'future', 'futuristic', 'century', 'dystopia', 'dystopian',
    'apocalyptic', 'post-apocalyptic', 'wasteland', 'bunker', 'underground',
    'silo', 'vault', 'survivors', 'remnants', 'ruins',
    'alien', 'aliens', 'clone', 'clones', 'mutation', 'mutations', 'mutant',
    'genetic', 'engineered', 'experiment', 'laboratory', 'scientist',
    'virus', 'cryogenic', 'cryosleep',
    'quantum', 'multiverse', 'dimension', 'warp', 'terraform',
    'parallel', 'time travel',
    'posthuman', 'transhuman', 'consciousness', 'transferred', 'uploaded',
    'nuclear', 'radioactive', 'reactor', 'radiation',
    'colony', 'colonies', 'colonist', 'colonists', 'settlement',
    'fleet', 'battleship', 'fighter', 'rebellion', 'empire',
    'bounty hunter', 'smuggler',
})

_FANTASY_SIGNALS = frozenset({
    'magic', 'magical', 'spell', 'spells', 'sorcerer', 'sorcery',
    'wizard', 'warlock', 'witch', 'witches', 'mage', 'arcane',
    'enchanted', 'enchantment', 'conjure', 'potion', 'potions',
    'wand', 'amulet', 'rune', 'runes', 'necromancer', 'druid',
    'dragon', 'dragons', 'elf', 'elves', 'dwarf', 'dwarves',
    'orc', 'orcs', 'troll', 'goblin', 'fairy', 'fairies', 'fae',
    'monster', 'monsters', 'beast', 'beasts', 'creature', 'creatures',
    'demon', 'demons',
    'sword', 'swords', 'knight', 'knights', 'castle', 'castles',
    'kingdom', 'kingdoms', 'realm', 'realms', 'throne', 'thrones',
    'princess', 'prince', 'medieval', 'feudal',
    'quest', 'prophecy', 'chosen one', 'destiny', 'destined',
    'ancient', 'legend', 'legendary', 'mythical', 'mythic', 'folklore',
    'curse', 'cursed',
    'dark lord', 'dark power', 'dark forces',
    'supernatural', 'mystical',
})


def _disambiguate_scifi_fantasy(overview: Optional[str]) -> List[str]:
    """Decide Sci-Fi, Fantasy, or both based on overview text."""
    if not overview or len(str(overview)) < 20:
        return ['Sci-Fi', 'Fantasy']
    text = str(overview).lower()
    sci = sum(1 for kw in _SCI_FI_SIGNALS if kw in text)
    fan = sum(1 for kw in _FANTASY_SIGNALS if kw in text)
    if sci > 0 and fan == 0:
        return ['Sci-Fi']
    if fan > 0 and sci == 0:
        return ['Fantasy']
    if sci > 0 and fan > 0:
        return ['Sci-Fi', 'Fantasy'] if sci >= fan else ['Fantasy', 'Sci-Fi']
    return ['Sci-Fi', 'Fantasy']

# Default columns expected in the final metadata output
_METADATA_COLS = [
    'item_id', 'tmdb_id', 'title', 'title_ru', 'type', 'year',
    'genres', 'keywords',
    'overview', 'overview_ru',
    'vote_average', 'vote_count', 'popularity',
    'has_embeddings', 'source',
]

_INTERACTION_DTYPES = {
    'user_id': 'uint32',
    'item_id': 'uint32',
    'rating': 'float32',
    'timestamp': 'uint32',
}


# =============================================================================
# PROCESSOR
# =============================================================================

class MovieDatasetProcessor:
    """
    End-to-end pipeline: raw MovieLens + TMDB + TV → unified Parquet files
    ready for LightGCN training and the Telegram bot.

    Output files (in data/processed/):
        - interactions_final.parquet  (user_id, item_id, rating, timestamp)
        - items_metadata_final.parquet  (item_id, tmdb_id, title, …)
        - id_mapping.json  (num_users, num_items, num_trained_items, tmdb_to_item)
        - dataset_stats.json  (human-readable summary)
    """

    def __init__(
            self,
            data_dir: Path,
            config_path: str = "config/settings.json",
            *,
            top_n_movies: int = 12_000,
            top_n_tv: int = 5_000,
            min_year: Optional[int] = None,
            languages: Optional[List[str]] = None,
            min_user_interactions: int = 10,
            min_item_interactions: int = 10,
            rating_threshold: float = 3.5,
            max_interactions: int = 10_000_000,
    ):
        self.data_raw = data_dir / 'raw'
        self.data_processed = data_dir / 'processed'
        self.data_processed.mkdir(parents=True, exist_ok=True)

        # Per-domain output dir, set by build_movie_dataset() / build_tv_dataset().
        # Defaults to data_processed for legacy flows.
        self.output_dir: Path = self.data_processed

        # Filtering params
        self.top_n_movies = top_n_movies
        self.top_n_tv = top_n_tv
        self.min_year = min_year
        self.languages = languages or ['en']
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
        self.rating_threshold = rating_threshold
        self.max_interactions = max_interactions

        # Genre mapping from external config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self._genre_mapping: Dict[str, Any] = config.get('genre_mapping', {})

        self._log_config()

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_config(self) -> None:
        logger.info("=" * 70)
        logger.info("MOVIE DATASET PROCESSOR v5")
        logger.info("=" * 70)
        logger.info(
            f"Movies: top {self.top_n_movies:,} | TV: top {self.top_n_tv:,} | "
            f"Languages: {self.languages}"
        )
        logger.info(
            f"Thresholds: user≥{self.min_user_interactions}, "
            f"item≥{self.min_item_interactions}, rating≥{self.rating_threshold}"
        )
        logger.info(f"Max interactions: {self.max_interactions:,}")

    @staticmethod
    def _section(title: str) -> None:
        logger.info(f"\n{'=' * 70}\n{title}\n{'=' * 70}")

    # =====================================================================
    # GENRE PARSING (single DRY implementation)
    # =====================================================================

    def _parse_genres(self, value, overview: Optional[str] = None) -> List[str]:
        """
        Universal genre parser. Supports multi-mapping (one tag → list of genres)
        with overview-based disambiguation for ambiguous compound tags.
        """
        # 1. Базовая проверка на физическое отсутствие
        if value is None:
            return []

        raw: List[str] = []

        # 2. Если пришел список или массив (как от сериалов)
        if isinstance(value, (list, np.ndarray)):
            raw = [str(v).strip() for v in value if pd.notna(v)]

        # 3. Если пришла строка (как от MovieLens или TMDB Movies)
        elif isinstance(value, str):
            s = value.strip()
            if not s or s.lower() in _GENRE_TRASH:
                return []
            try:
                if s.startswith('[') and '{' in s:
                    import ast
                    parsed = ast.literal_eval(s)
                    raw = [g['name'] for g in parsed if isinstance(g, dict) and 'name' in g]
                else:
                    s = s.replace('[', '').replace(']', '').replace("'", '').replace('"', '')
                    delimiter = '|' if '|' in s else ','
                    raw = [g.strip() for g in s.split(delimiter)]
            except (ValueError, SyntaxError):
                return []
        else:
            if pd.isna(value):
                return []
            raw = [str(value).strip()]

        # 4. МАППИНГ + ОЧИСТКА + ДЕДУПЛИКАЦИЯ + ДИЗАМБІГУАЦІЯ
        seen: Set[str] = set()
        result: List[str] = []

        for g in raw:
            if not g or g.lower() in _GENRE_TRASH:
                continue

            # Берем значение из нашего JSON-конфига
            mapped = self._genre_mapping.get(g, g)

            # --- МНОЖЕСТВЕННЫЙ МАППИНГ С ДИЗАМБІГУАЦІЄЮ ---
            if isinstance(mapped, list):
                # Проверяем, нужна ли дизамбігуація (Sci-Fi & Fantasy)
                if g == 'Sci-Fi & Fantasy' or mapped == ['Sci-Fi', 'Fantasy']:
                    sub_genres = _disambiguate_scifi_fantasy(overview)
                else:
                    sub_genres = mapped
                for sub_genre in sub_genres:
                    if sub_genre not in seen:
                        seen.add(sub_genre)
                        result.append(sub_genre)
            else:
                # Обычное поведение для строк
                if mapped not in seen:
                    seen.add(mapped)
                    result.append(mapped)

        return result

    def _parse_tv_genres(self, row: pd.Series) -> List[str]:
        """Parse genres from TV series row (handles different column layouts)."""
        overview = row.get('overview') if 'overview' in row.index else None
        if overview is not None and (not isinstance(overview, str) or pd.isna(overview)):
            overview = None

        # Try standard 'genres' column first
        if 'genres' in row.index and pd.notna(row.get('genres')):
            return self._parse_genres(row['genres'], overview)

        # Fall back to Kaggle-style genres[i].name columns
        found: List[str] = []
        for i in range(10):
            col = f'genres[{i}].name'
            if col in row.index and pd.notna(row[col]):
                found.append(str(row[col]))

        return self._parse_genres(found, overview) if found else []

    # =====================================================================
    # 1. DATA LOADING (memory-optimized)
    # =====================================================================

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame]:
        """Load raw datasets with explicit dtypes for memory efficiency."""
        self._section("1. LOADING DATA")

        ml_ratings = pd.read_csv(
            self.data_raw / 'ml-32m/ratings.csv',
            dtype={'userId': 'uint32', 'movieId': 'uint32',
                   'rating': 'float32', 'timestamp': 'uint32'},
        )
        ml_ratings.columns = ['user_id', 'movie_id', 'rating', 'timestamp']

        ml_movies = pd.read_csv(self.data_raw / 'ml-32m/movies.csv')
        ml_movies.columns = ['movie_id', 'title', 'genres']

        # tmdb_id can be NaN → read as float, convert later
        ml_links = pd.read_csv(
            self.data_raw / 'ml-32m/links.csv',
            dtype={'tmdbId': 'float64'},
        )
        ml_links.columns = ['movie_id', 'imdb_id', 'tmdb_id']

        # TMDB + TV (graceful if files missing)
        tmdb_path = self.data_raw / 'TMDB_movie_dataset_v11.csv'
        tv_path = self.data_raw / 'tv_series.csv'

        tmdb_movies = (
            pd.read_csv(tmdb_path, low_memory=False) if tmdb_path.exists()
            else pd.DataFrame()
        )
        tv_series = (
            pd.read_csv(tv_path, low_memory=False) if tv_path.exists()
            else pd.DataFrame()
        )

        logger.info(f"MovieLens: {len(ml_ratings):,} ratings, {len(ml_movies):,} movies")
        logger.info(f"TMDB: {len(tmdb_movies):,}, TV: {len(tv_series):,}")

        return ml_ratings, ml_movies, ml_links, tmdb_movies, tv_series

    def load_amazon_interactions(self, full_metadata: pd.DataFrame, amazon_dir: Path) -> pd.DataFrame:
        """
        Загрузка Amazon Reviews 2023 и маппинг на TMDB ID.
        Читает файлы с диска D построчно для экономии памяти.
        """
        self._section("4.7 LOAD AMAZON INTERACTIONS")

        meta_path = amazon_dir / "meta_Movies_and_TV.jsonl"
        reviews_path = amazon_dir / "Movies_and_TV.jsonl"

        if not meta_path.exists() or not reviews_path.exists():
            logger.warning(f"Amazon data not found in {amazon_dir} — skipping")
            return pd.DataFrame()

        # 1. Готовим словарь для маппинга по названиям
        # Ключ: очищенное название, Значение: tmdb_id
        title_to_tmdb = full_metadata.copy()
        title_to_tmdb['clean_title'] = title_to_tmdb['title'].str.lower().str.strip()
        # Важно: убираем дубликаты названий, оставляя самые популярные
        title_to_tmdb = title_to_tmdb.sort_values('popularity', ascending=False)
        mapping_dict = title_to_tmdb.set_index('clean_title')['tmdb_id'].to_dict()

        target_titles = set(mapping_dict.keys())
        asin_to_tmdb = {}

        # 2. Читаем метаданные Amazon (Маппинг ASIN -> TMDB)
        logger.info("Mapping Amazon ASINs to TMDB IDs...")
        trash_keywords = {'cable', 'remote control', 'bracket', 'mount', 'adapter', 'glasses', 'battery'}

        with open(meta_path, 'r', encoding='utf-8') as f:
            lines_read = 0
            lines_parsed = 0
            lines_with_title = 0
            errors = []

            for line in f:
                lines_read += 1
                try:
                    item = json.loads(line)
                    lines_parsed += 1
                    title = str(item.get('title', '')).lower()
                    if not title or title == 'none':
                        continue
                    lines_with_title += 1

                    # Очистка названия: "The Matrix [Blu-ray]" -> "the matrix"
                    clean_title = title.split('[')[0].split('(')[0].strip()

                    # TV-суффиксы вне скобок:
                    # "Breaking Bad: The Complete Series" -> "breaking bad"
                    # "The Wire - Complete Series"        -> "the wire"
                    clean_title = re.sub(
                        r'[:\-]\s*(the\s+)?(complete|entire|full|whole)\b.*',
                        '', clean_title, flags=re.IGNORECASE
                    ).strip()
                    # "Game of Thrones Season 1" -> "game of thrones"
                    clean_title = re.sub(
                        r'\s+(?:season|series|collection|vol\.?|part)\s+\d.*$',
                        '', clean_title, flags=re.IGNORECASE
                    ).strip()
                    # "Seinfeld: Season 1" -> "seinfeld"
                    clean_title = re.sub(
                        r'[:\-]\s*(?:season|series|episode|collection|vol\.?|part)\b.*',
                        '', clean_title, flags=re.IGNORECASE
                    ).strip()
                    # Убираем висящую пунктуацию: "The Wire," -> "the wire"
                    clean_title = clean_title.rstrip(':,-. ')

                    if clean_title in target_titles:
                        # Проверка на "мусор"
                        if any(trash in title for trash in trash_keywords):
                            continue
                        asin = item.get('parent_asin')
                        if asin:
                            asin_to_tmdb[asin] = mapping_dict[clean_title]
                except Exception as e:
                    if len(errors) < 5:
                        errors.append(f"Line {lines_read}: {type(e).__name__}: {e}")
                    continue

            if errors:
                for err in errors:
                    logger.warning(f"Amazon meta parse error: {err}")
            logger.info(f"Amazon meta: {lines_read:,} lines read, {lines_parsed:,} parsed, "
                        f"{lines_with_title:,} with titles")

        unique_tmdb = len(set(asin_to_tmdb.values()))
        logger.info(f"Successfully mapped {len(asin_to_tmdb):,} ASINs → {unique_tmdb:,} unique TMDB IDs")

        if not asin_to_tmdb:
            sample_targets = list(target_titles)[:5]
            logger.warning(f"0 ASINs matched! Sample target titles: {sample_targets}")
            logger.warning(f"Target titles count: {len(target_titles):,}")

        # 3. Читаем отзывы Amazon
        logger.info("Reading Amazon reviews...")
        amazon_ratings = []

        review_errors = []
        reviews_read = 0
        with open(reviews_path, 'r', encoding='utf-8') as f:
            for line in f:
                reviews_read += 1
                try:
                    rev = json.loads(line)
                    asin = rev.get('parent_asin')
                    if asin in asin_to_tmdb:
                        amazon_ratings.append({
                            'user_id': rev['user_id'],  # Пока строка
                            'tmdb_id': asin_to_tmdb[asin],
                            'rating': float(rev['rating']),
                            'timestamp': int(rev.get('timestamp', 0) / 1000)
                        })
                except Exception as e:
                    if len(review_errors) < 5:
                        review_errors.append(f"Line {reviews_read}: {type(e).__name__}: {e}")
                    continue

                if len(amazon_ratings) > 2000000:  # Лимит для стабильности
                    break

        if review_errors:
            for err in review_errors:
                logger.warning(f"Amazon review parse error: {err}")
        logger.info(f"Amazon reviews: {reviews_read:,} lines read")

        if not amazon_ratings:
            return pd.DataFrame()

        df_amazon = pd.DataFrame(amazon_ratings)

        # 4. Конвертируем строковые user_id Amazon в числовые
        # Используем хэш или факторизацию, чтобы не пересекаться с MovieLens
        # Но проще всего - категориальный маппинг
        unique_users = df_amazon['user_id'].unique()
        user_map = {uid: i for i, uid in enumerate(unique_users)}
        df_amazon['user_id'] = df_amazon['user_id'].map(user_map).astype('uint32')

        logger.info(f"Loaded {len(df_amazon):,} interactions from Amazon")
        return df_amazon

    # =====================================================================
    # 4.8 LOAD TRAKT (TV SHOWS + USER RATINGS)
    # =====================================================================

    def load_trakt_metadata(self) -> pd.DataFrame:
        """
        Загрузка метаданных TV-шоу, собранных через trakt_collector.py.

        Читает data/raw/trakt_shows.csv. Применяет TV_OFFSET=10_000_000 к tmdb_id
        в соответствии с конвенцией проекта (TV и Movie tmdb_id могут пересекаться
        в TMDB, поэтому TV offset'ится).

        Возвращает DataFrame в том же формате, что и clean_tv_series().
        """
        self._section("TRAKT TV METADATA")

        shows_path = self.data_raw / "trakt_shows.csv"
        if not shows_path.exists():
            logger.warning(f"Trakt shows CSV not found: {shows_path}")
            return pd.DataFrame()

        df = pd.read_csv(shows_path)
        initial = len(df)

        df = df.dropna(subset=['tmdb_id', 'title'])
        df['tmdb_id'] = df['tmdb_id'].astype(int) + 10_000_000
        df['year'] = pd.to_numeric(df.get('year'), errors='coerce')
        df = df.dropna(subset=['year'])
        df['year'] = df['year'].astype(int)

        df['genres'] = df['genres'].apply(
            lambda v: self._parse_genres(v) if pd.notna(v) else []
        )
        df = df[df['genres'].apply(len) > 0]

        df['title_ru'] = None
        df['type'] = 'tv'

        for col, default in (
            ('vote_average', 0.0),
            ('vote_count', 0),
            ('popularity', 0.0),
            ('overview', ''),
        ):
            if col not in df.columns:
                df[col] = default

        df = df[df['overview'].notna() & (df['overview'].astype(str).str.len() > 10)]

        if self.languages and 'language' in df.columns:
            is_target_lang = df['language'].isin(self.languages)
            is_popular_non_target = (~is_target_lang) & (df['vote_count'].fillna(0) >= 100)
            df = df[is_target_lang | is_popular_non_target]

        if 'popularity' in df.columns:
            df = df.sort_values('popularity', ascending=False)
        df = df.drop_duplicates(subset=['tmdb_id'], keep='first').head(self.top_n_tv)

        cols = ['tmdb_id', 'title', 'title_ru', 'year', 'genres', 'overview',
                'vote_average', 'vote_count', 'popularity', 'type']
        result = df[[c for c in cols if c in df.columns]].copy()

        logger.info(f"Trakt TV metadata: {initial:,} → {len(result):,}")
        return result

    def load_trakt_interactions(self, tv_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Загрузка user-item рейтингов из trakt_interactions.csv.

        CSV колонки: user_id, tmdb_id, rating (0.5–5.0), timestamp (epoch sec).
        К tmdb_id применяется TV_OFFSET=10_000_000 для согласования с метаданными.
        Возвращает DataFrame с колонками user_id, tmdb_id, rating, timestamp.
        """
        self._section("TRAKT INTERACTIONS")

        path = self.data_raw / "trakt_interactions.csv"
        if not path.exists():
            logger.warning(f"Trakt interactions CSV not found: {path}")
            return pd.DataFrame()

        df = pd.read_csv(
            path,
            dtype={'user_id': 'uint32', 'tmdb_id': 'uint32',
                   'rating': 'float32', 'timestamp': 'uint32'},
        )
        df['tmdb_id'] = df['tmdb_id'].astype('int64') + 10_000_000

        valid_tmdb = set(tv_metadata['tmdb_id'].astype(int))
        before = len(df)
        df = df[df['tmdb_id'].isin(valid_tmdb)].copy()
        logger.info(f"Trakt interactions: {before:,} → {len(df):,} (after metadata join)")

        return df

    # =====================================================================
    # 2. TMDB MOVIES CLEANING
    # =====================================================================

    def clean_tmdb_movies(self, tmdb: pd.DataFrame) -> pd.DataFrame:
        """Clean TMDB movie metadata. Returns standardized columns."""
        if tmdb.empty:
            return pd.DataFrame()

        self._section("2. CLEAN TMDB MOVIES")
        df = tmdb.rename(columns={'id': 'tmdb_id'})
        initial = len(df)

        # Sort by popularity early → keeps best entry when deduping
        if 'popularity' in df.columns:
            df = df.sort_values('popularity', ascending=False)

        # Language filter: target languages pass freely, others need vote_count >= 100
        if self.languages and 'original_language' in df.columns:
            is_target_lang = df['original_language'].isin(self.languages)
            is_popular_non_target = (~is_target_lang) & (df['vote_count'] >= 100)
            before_lang = len(df)
            df = df[is_target_lang | is_popular_non_target]
            logger.info(f"Language filter: {before_lang:,} → {len(df):,} "
                        f"(EN + non-EN with vote_count≥100)")

        # Genres (with overview-based disambiguation for Sci-Fi & Fantasy)
        df['genres'] = df.apply(
            lambda row: self._parse_genres(
                row['genres'],
                row.get('overview') if 'overview' in row.index else None,
            ),
            axis=1,
        )
        df = df[df['genres'].apply(len) > 0]

        # Year
        if 'release_date' in df.columns:
            df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
            df = df.dropna(subset=['year', 'title'])
            df['year'] = df['year'].astype(int)
            if self.min_year:
                df = df[df['year'] >= self.min_year]

        # Quality gates
        if 'vote_count' in df.columns:
            df = df[(df['vote_count'] >= 100) & (df['vote_average'] > 0)]
        if 'overview' in df.columns:
            df = df[df['overview'].notna() & (df['overview'].str.len() > 10)]

        # Dedup by title+year (keeps most popular via prior sort)
        df = df.drop_duplicates(subset=['title', 'year'], keep='first')
        df['type'] = 'movie'

        # Bilingual prep: rename title → title (EN), add placeholder for RU
        df['title_ru'] = None

        cols = ['tmdb_id', 'title', 'title_ru', 'year', 'genres', 'overview',
                'vote_average', 'vote_count', 'popularity', 'type']
        result = df[[c for c in cols if c in df.columns]].copy()

        logger.info(f"TMDB movies: {initial:,} → {len(result):,}")
        return result

    # =====================================================================
    # 3. TV SERIES CLEANING
    # =====================================================================

    def clean_tv_series(self, tv: pd.DataFrame) -> pd.DataFrame:
        """Clean TV series metadata. Returns standardized columns."""
        if tv.empty:
            return pd.DataFrame()

        self._section("3. CLEAN TV SERIES")
        df = tv.copy()
        df.columns = [c.lower() for c in df.columns]
        initial = len(df)

        # Resolve ID and title columns (different Kaggle datasets use different names)
        id_col = next((c for c in ('id', 'tmdb_id') if c in df.columns), None)
        name_col = next((c for c in ('name', 'title', 'original_name') if c in df.columns), None)

        if not id_col or not name_col:
            logger.error(f"Cannot find ID/Name columns. Available: {df.columns.tolist()}")
            return pd.DataFrame()

        df = df.rename(columns={id_col: 'tmdb_id', name_col: 'title'})

        # Year
        date_col = next((c for c in ('first_air_date', 'release_date') if c in df.columns), None)
        if date_col:
            df['year'] = pd.to_datetime(df[date_col], errors='coerce').dt.year
            df = df.dropna(subset=['year'])
            df['year'] = df['year'].astype(int)
            df = df[(df['year'] >= 1950) & (df['year'] <= 2025)]

        # Language filter: target languages pass freely, others need vote_count >= 100
        if self.languages and 'original_language' in df.columns:
            is_target_lang = df['original_language'].isin(self.languages)
            vote_counts = df['vote_count'] if 'vote_count' in df.columns else pd.Series(0, index=df.index)
            is_popular_non_target = (~is_target_lang) & (vote_counts.fillna(0) >= 100)
            before_lang = len(df)
            df = df[is_target_lang | is_popular_non_target]
            logger.info(f"TV language filter: {before_lang:,} → {len(df):,} "
                        f"(EN + non-EN with vote_count≥100)")

        # Genres (uses TV-specific parser)
        df['genres'] = df.apply(self._parse_tv_genres, axis=1)
        df = df[df['genres'].apply(len) > 0]

        # Sort by popularity, take top N
        if 'popularity' in df.columns:
            df = df.sort_values('popularity', ascending=False)

        df = df.head(self.top_n_tv)
        df['type'] = 'tv'
        df['title_ru'] = None

        cols = ['tmdb_id', 'title', 'title_ru', 'year', 'genres', 'overview',
                'vote_average', 'vote_count', 'popularity', 'type']
        result = df[[c for c in cols if c in df.columns]].copy()

        logger.info(f"TV series: {initial:,} → {len(result):,}")
        return result

    # =====================================================================
    # 4. MERGE METADATA (with tmdb_id collision detection)
    # =====================================================================

    def merge_metadata(
            self, tmdb: pd.DataFrame, tv: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge movie and TV metadata into a single catalog.

        SAFETY: Detects and resolves tmdb_id collisions between movies and TV.
        TMDB uses separate ID spaces for movies and TV, but our flat catalog
        needs unique IDs. If a collision occurs, we keep the more popular entry.
        """
        self._section("4. MERGE METADATA")

        tmdb = tmdb.dropna(subset=['tmdb_id']).copy()
        tv = tv.dropna(subset=['tmdb_id']).copy()
        tmdb['tmdb_id'] = tmdb['tmdb_id'].astype(int)
        tv['tmdb_id'] = tv['tmdb_id'].astype(int)

        # Detect ID collisions between movies and TV
        movie_ids = set(tmdb['tmdb_id'])
        tv_ids = set(tv['tmdb_id'])
        collisions = movie_ids & tv_ids

        if collisions:
            logger.warning(
                f"⚠️  {len(collisions)} tmdb_id collisions between movies and TV! "
                f"(TMDB uses separate ID spaces — this is expected)\n"
                f"   Strategy: prefixing TV IDs with offset 10_000_000 to avoid conflicts."
            )
            # CRITICAL FIX: TMDB movie IDs and TV IDs can overlap.
            # Prefix TV IDs with a large offset to make them globally unique.
            TV_ID_OFFSET = 10_000_000
            tv['tmdb_id'] = tv['tmdb_id'] + TV_ID_OFFSET
            logger.info(f"   TV tmdb_id range: {tv['tmdb_id'].min()} — {tv['tmdb_id'].max()}")

        # Standardize columns
        common = ['tmdb_id', 'title', 'title_ru', 'type', 'year', 'genres',
                  'overview', 'vote_average', 'vote_count', 'popularity']
        tmdb_cols = [c for c in common if c in tmdb.columns]
        tv_cols = [c for c in common if c in tv.columns]

        metadata = pd.concat([tmdb[tmdb_cols], tv[tv_cols]], ignore_index=True)

        # Final dedup (safety net)
        before = len(metadata)
        if 'popularity' in metadata.columns:
            metadata = metadata.sort_values('popularity', ascending=False)
        metadata = metadata.drop_duplicates(subset=['tmdb_id'], keep='first')

        if len(metadata) < before:
            logger.warning(f"Removed {before - len(metadata)} duplicate tmdb_ids after merge")

        logger.info(
            f"Movies: {len(tmdb):,} | TV: {len(tv):,} | "
            f"Merged (unique): {len(metadata):,}"
        )
        return metadata

    # =====================================================================
    # 4.5 FETCH KEYWORDS
    # =====================================================================

    def fetch_keywords(self, metadata: pd.DataFrame) -> pd.DataFrame:
        self._section("4.5 FETCH KEYWORDS")

        # 1. Сразу проверяем ключ
        api_key = os.getenv("TMDB_API_KEY")
        if not api_key:
            logger.warning("TMDB_API_KEY not set — using empty keywords")
            metadata['keywords'] = [[] for _ in range(len(metadata))]
            return metadata

        # 2. Инициализируем БД
        cache_path = self.data_processed / "cache" / "keywords.db"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(cache_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS keywords (
                tmdb_id INTEGER PRIMARY KEY,
                media_type TEXT,
                keywords TEXT,
                fetched_at INTEGER
            )
        """)

        # 3. Находим, что нужно докачать
        cached = {r[0] for r in conn.execute("SELECT tmdb_id FROM keywords").fetchall()}

        to_fetch = []
        for _, row in metadata.iterrows():
            tid = int(row['tmdb_id'])
            if tid not in cached:
                mt = 'tv' if row.get('type') == 'tv' else 'movie'
                api_id = tid - 10_000_000 if mt == 'tv' and tid >= 10_000_000 else tid
                to_fetch.append((tid, api_id, mt))

        logger.info(f"Keywords: {len(cached)} cached, {len(to_fetch)} to fetch")

        # 4. Качаем
        if to_fetch:
            session = requests.Session()
            for i, (tid, api_id, mt) in enumerate(to_fetch):
                # TMDB limit: 40 req / 10 sec
                if i > 0 and i % 35 == 0:
                    conn.commit()  # Сохраняем промежуточный результат
                    time.sleep(10)

                try:
                    url = f"https://api.themoviedb.org/3/{mt}/{api_id}/keywords"
                    resp = session.get(url, params={"api_key": api_key}, timeout=10)
                    if resp.status_code == 200:
                        data = resp.json()
                        kw_list = data.get('keywords') or data.get('results') or []
                        kw_names = [kw['name'] for kw in kw_list]
                        conn.execute(
                            "INSERT OR REPLACE INTO keywords VALUES (?,?,?,?)",
                            (tid, mt, json.dumps(kw_names), int(time.time()))
                        )
                    elif resp.status_code == 429:  # Too many requests
                        time.sleep(20)
                except Exception as e:
                    logger.warning(f"Keyword fetch failed for {tid}: {e}")
            conn.commit()

        # 5. Пришиваем данные
        all_kw = {r[0]: json.loads(r[1]) for r in conn.execute("SELECT tmdb_id, keywords FROM keywords").fetchall()}
        metadata['keywords'] = metadata['tmdb_id'].astype(int).map(lambda tid: all_kw.get(tid, []))

        conn.close()
        return metadata

    # =====================================================================
    # 5. LINK TO MOVIELENS
    # =====================================================================

    def link_to_movielens(
            self,
            ml_movies: pd.DataFrame,
            ml_links: pd.DataFrame,
            metadata: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Map MovieLens movie_ids → tmdb_ids using links.csv,
        then inner-join with our cleaned metadata catalog.

        Returns: DataFrame with (movie_id, tmdb_id, title, type).
        """
        self._section("5. LINK TO MOVIELENS")

        # Prepare links (drop movies without TMDB mapping)
        links = ml_links.dropna(subset=['tmdb_id']).copy()
        links['tmdb_id'] = links['tmdb_id'].astype(int)
        links['movie_id'] = links['movie_id'].astype(int)

        # ML movies → tmdb_id
        ml_with_tmdb = ml_movies[['movie_id', 'title']].merge(
            links[['movie_id', 'tmdb_id']], on='movie_id', how='inner'
        )

        # Inner join with metadata (only keep ML movies that passed quality filters)
        valid_tmdb = set(metadata['tmdb_id'])
        linked = ml_with_tmdb[ml_with_tmdb['tmdb_id'].isin(valid_tmdb)].copy()

        # Add media type from metadata
        type_map = metadata.set_index('tmdb_id')['type'].to_dict()
        linked['type'] = linked['tmdb_id'].map(type_map).fillna('movie')

        # Stats
        coverage = len(linked) / max(len(ml_movies), 1) * 100
        logger.info(f"MovieLens: {len(ml_movies):,} → Linked: {len(linked):,} ({coverage:.1f}%)")

        if coverage < 50:
            logger.error("⚠️  Low coverage! Check TMDB dataset compatibility.")

        return linked[['movie_id', 'tmdb_id', 'title', 'type']]

    # =====================================================================
    # 6. FILTER INTERACTIONS (k-core)
    # =====================================================================

    def filter_interactions(
            self,
            ml_ratings: pd.DataFrame,
            ml_movies_linked: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Iterative k-core filtering:
          1. Keep only ratings for linked movies
          2. Positive ratings only (≥ threshold)
          3. Iterative user/item minimum interaction pruning
          4. Max interactions cap (keeps most active users)
        """
        self._section("6. FILTER INTERACTIONS")

        valid_ml_ids = set(ml_movies_linked['movie_id'].unique())
        ratings = ml_ratings[ml_ratings['movie_id'].isin(valid_ml_ids)].copy()
        logger.info(f"Ratings for linked movies: {len(ratings):,}")

        # Positive only + dedup
        ratings = ratings[ratings['rating'] >= self.rating_threshold]
        ratings = ratings.drop_duplicates(subset=['user_id', 'movie_id'], keep='last')
        logger.info(f"Positive (≥{self.rating_threshold}): {len(ratings):,}")

        # Iterative k-core
        for i in range(10):
            prev = len(ratings)

            item_counts = ratings['movie_id'].value_counts()
            valid_items = item_counts[item_counts >= self.min_item_interactions].index
            ratings = ratings[ratings['movie_id'].isin(valid_items)]

            user_counts = ratings['user_id'].value_counts()
            valid_users = user_counts[user_counts >= self.min_user_interactions].index
            ratings = ratings[ratings['user_id'].isin(valid_users)]

            if len(ratings) == prev:
                logger.info(f"K-core converged in {i + 1} iterations: {len(ratings):,}")
                break

        # Max interactions cap
        if len(ratings) > self.max_interactions:
            user_counts = ratings['user_id'].value_counts()
            cumsum = user_counts.cumsum()
            n_users = max(1, (cumsum <= self.max_interactions).sum())
            top_users = user_counts.head(n_users).index
            ratings = ratings[ratings['user_id'].isin(top_users)]
            logger.info(f"Capped to {len(ratings):,} interactions ({n_users:,} users)")

        # Sync linked metadata to surviving movies
        surviving_movies = set(ratings['movie_id'].unique())
        trained_movies = ml_movies_linked[
            ml_movies_linked['movie_id'].isin(surviving_movies)
        ].copy()

        logger.info(
            f"Final: {len(ratings):,} ratings, "
            f"{ratings['user_id'].nunique():,} users, "
            f"{len(trained_movies):,} trained items"
        )
        return ratings, trained_movies

    # =====================================================================
    # 7. CREATE UNIFIED IDS (hybrid: trained + catalog)
    # =====================================================================

    def filter_interactions_unified(self, ratings: pd.DataFrame, full_metadata: pd.DataFrame):
        self._section("6. FILTER INTERACTIONS (UNIFIED + ASYMMETRIC K-CORE)")

        # 1. Оставляем только те, что есть в нашем каталоге
        valid_ids = set(full_metadata['tmdb_id'])
        ratings = ratings[ratings['tmdb_id'].isin(valid_ids)].copy()

        # 2. Фильтр по минимальному рейтингу (3.5+)
        ratings = ratings[ratings['rating'] >= self.rating_threshold]
        ratings = ratings.drop_duplicates(subset=['user_id', 'tmdb_id'], keep='last')

        # Константы для порогов
        MIN_MOVIE_REVIEWS = 10  # Оставляем 10 для стабильности фильмов
        MIN_TV_REVIEWS = 5  # Снижаем до 5 для захвата большего числа сериалов
        TV_OFFSET = 10_000_000

        # 3. Итеративный k-core процесс
        for i in range(5):
            prev_len = len(ratings)

            # --- Асимметричный фильтр айтемов ---
            it_counts = ratings['tmdb_id'].value_counts()

            # Разделяем на фильмы и сериалы по ID
            movie_ids_counts = it_counts[it_counts.index < TV_OFFSET]
            tv_ids_counts = it_counts[it_counts.index >= TV_OFFSET]

            # Выбираем выжившие фильмы (>=10) и сериалы (>=5)
            valid_movies = movie_ids_counts[movie_ids_counts >= MIN_MOVIE_REVIEWS].index
            valid_tv = tv_ids_counts[tv_ids_counts >= MIN_TV_REVIEWS].index

            # Объединяем "выживших"
            valid_it = valid_movies.union(valid_tv)
            ratings = ratings[ratings['tmdb_id'].isin(valid_it)]

            # --- Стандартный фильтр пользователей ---
            # Порог для юзеров оставляем 10, чтобы модель училась на опытных людях
            u_counts = ratings['user_id'].value_counts()
            valid_u = u_counts[u_counts >= self.min_user_interactions].index
            ratings = ratings[ratings['user_id'].isin(valid_u)]

            if len(ratings) == prev_len:
                logger.info(f"  K-core converged after {i + 1} iterations")
                break

        trained_tmdb_ids = set(ratings['tmdb_id'].unique())

        # Логируем результат
        tv_count = sum(1 for tid in trained_tmdb_ids if tid >= TV_OFFSET)
        movie_count = len(trained_tmdb_ids) - tv_count
        logger.info(f"  Final trained items: {movie_count} movies, {tv_count} TV shows")

        return ratings, trained_tmdb_ids

    def create_unified_ids(
            self,
            interactions: pd.DataFrame,
            trained_tmdb_ids: Set[int],
            all_metadata: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Создает последовательные ID для гибридного датасета (MovieLens + Amazon):
          - item_id 0..N-1   → Обученные айтемы (фильмы и СЕРИАЛЫ с рейтингами)
          - item_id N..M-1   → Только каталог (без достаточного кол-ва рейтингов)
          - user_id 0..U-1   → Активные пользователи (объединенные из двух источников)
        """
        self._section("7. CREATE UNIFIED IDs")

        # --- 1. Item IDs (Маппинг предметов) ---
        # Сначала идем по тем, кто прошел k-core фильтрацию (0..N-1)
        # Это критически важно для LightGCN: первые N эмбеддингов будут обучаемыми
        sorted_trained = sorted(list(trained_tmdb_ids))
        tmdb_to_item: Dict[int, int] = {int(tid): idx for idx, tid in enumerate(sorted_trained)}
        num_trained = len(sorted_trained)

        # Затем добавляем остальные айтемы из каталога (N..M-1)
        all_tmdb_ids = sorted(all_metadata['tmdb_id'].unique())
        next_id = num_trained
        for tid in all_tmdb_ids:
            tid_int = int(tid)
            if tid_int not in tmdb_to_item:
                tmdb_to_item[tid_int] = next_id
                next_id += 1

        # --- 2. Apply to metadata (Обновление метаданных) ---
        all_metadata = all_metadata.copy()
        all_metadata['item_id'] = all_metadata['tmdb_id'].astype(int).map(tmdb_to_item)

        # Проверка на пропущенные ID
        unmapped_meta = all_metadata['item_id'].isna().sum()
        if unmapped_meta > 0:
            logger.warning(f"⚠️ {unmapped_meta} metadata rows without item_id — dropping")
            all_metadata = all_metadata.dropna(subset=['item_id'])

        all_metadata['item_id'] = all_metadata['item_id'].astype('uint32')

        # Флаги для UniversalSearchEngine и бота
        all_metadata['has_embeddings'] = all_metadata['item_id'] < num_trained
        all_metadata['source'] = np.where(
            all_metadata['has_embeddings'], 'trained', 'catalog'
        )

        # --- 3. Apply to interactions (Обновление взаимодействий) ---
        interactions = interactions.copy()
        # Теперь interactions уже содержит tmdb_id (после объединения ML и Amazon)
        interactions['item_id'] = interactions['tmdb_id'].astype(int).map(tmdb_to_item)

        # Удаляем взаимодействия, которые не попали в маппинг (защита)
        before_count = len(interactions)
        interactions = interactions.dropna(subset=['item_id'])
        if len(interactions) < before_count:
            logger.warning(f"Dropped {before_count - len(interactions)} interactions with unmapped IDs")

        interactions['item_id'] = interactions['item_id'].astype('uint32')

        # --- 4. User IDs (Последовательные ID пользователей) ---
        # Так как мы объединили пользователей ML и Amazon, их ID нужно перенумеровать с 0
        unique_users = sorted(interactions['user_id'].unique())
        user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        interactions['user_id'] = interactions['user_id'].map(user_to_idx).astype('uint32')

        # --- 5. Сохранение маппинга для работы бота ---
        mapping = {
            'num_users': len(unique_users),
            'num_items': len(tmdb_to_item),
            'num_trained_items': num_trained,
            'tmdb_to_item': {int(k): int(v) for k, v in tmdb_to_item.items()},
        }

        # Анализируем, сколько сериалов попало в обучение
        trained_metadata = all_metadata[all_metadata['has_embeddings']]
        tv_in_training = (trained_metadata['type'] == 'tv').sum()
        movies_in_training = (trained_metadata['type'] == 'movie').sum()

        logger.info(
            f"Users: {mapping['num_users']:,} | "
            f"Items total: {mapping['num_items']:,} | "
            f"Trained: {num_trained:,} ({movies_in_training} movies + {tv_in_training} TV shows)"
        )

        final_interactions = interactions[['user_id', 'item_id', 'rating', 'timestamp']].copy()
        return final_interactions, all_metadata, mapping

    # =====================================================================
    # 8. VALIDATE & SAVE
    # =====================================================================

    def validate_and_save(
            self,
            interactions: pd.DataFrame,
            metadata: pd.DataFrame,
            mapping: Dict,
    ) -> None:
        """Validate data integrity, create backup, and save final files."""
        self._section("8. VALIDATE & SAVE")

        # --- КРОСС-ДЕФЕКТ: Удаление дублей (исправляет Warning из check_data) ---
        initial_int = len(interactions)
        interactions = interactions.drop_duplicates(subset=['user_id', 'item_id'], keep='last')
        if len(interactions) < initial_int:
            logger.info(f"✓ Удалено {initial_int - len(interactions)} дубликатов (user, item)")

        num_trained = mapping['num_trained_items']

        # --- Validation ---
        errors: List[str] = []

        # 1) User IDs are sequential 0..U-1
        u_min, u_max, u_nuniq = (
            interactions['user_id'].min(),
            interactions['user_id'].max(),
            interactions['user_id'].nunique(),
        )
        if u_min != 0 or u_max != u_nuniq - 1:
            errors.append(f"User IDs not sequential: min={u_min}, max={u_max}, nunique={u_nuniq}")

        # 2) Interaction item_ids are within trained range
        i_max = interactions['item_id'].max()
        if i_max >= num_trained:
            errors.append(
                f"Interaction item_id {i_max} ≥ num_trained {num_trained}. "
                f"Interactions should only reference trained items!"
            )

        # 3) Metadata item_ids are sequential 0..M-1
        m_min, m_max, m_nuniq = (
            metadata['item_id'].min(),
            metadata['item_id'].max(),
            metadata['item_id'].nunique(),
        )
        if m_min != 0 or m_max != m_nuniq - 1:
            errors.append(f"Metadata item_ids not sequential: min={m_min}, max={m_max}, nunique={m_nuniq}")

        # 4) No duplicate item_ids in metadata
        if m_nuniq != len(metadata):
            errors.append(f"Duplicate item_ids in metadata: {len(metadata)} rows but {m_nuniq} unique")

        # 5) All interaction items exist in metadata
        int_items = set(interactions['item_id'].unique())
        meta_items = set(metadata['item_id'].unique())
        orphans = int_items - meta_items
        if orphans:
            errors.append(f"{len(orphans)} interaction item_ids missing from metadata")

        # Принудительно конвертируем в list, если там numpy array
        if 'genres' in metadata.columns:
            metadata['genres'] = metadata['genres'].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else list(x) if x is not None else []
            )

        # 6) Genre format check (sample)
        if 'genres' in metadata.columns:
            sample = metadata['genres'].iloc[0]
            if not isinstance(sample, list):
                errors.append(f"Genres not list type: got {type(sample)}")

        # 7) num_items matches metadata length
        if mapping['num_items'] != len(metadata):
            errors.append(
                f"mapping.num_items ({mapping['num_items']}) ≠ len(metadata) ({len(metadata)})"
            )

        if errors:
            for e in errors:
                logger.error(f"❌ VALIDATION FAIL: {e}")
            raise ValueError(f"Dataset validation failed with {len(errors)} errors")

        logger.info("✓ All validation checks passed")

        # --- Ensure genres are clean lists ---
        metadata['genres'] = metadata['genres'].apply(self._parse_genres)

        # --- Select final columns ---
        final_cols = [c for c in _METADATA_COLS if c in metadata.columns]
        metadata = metadata[final_cols].copy()

        # --- Backup ---
        self.output_dir.mkdir(parents=True, exist_ok=True)
        backup_dir = self.output_dir / 'backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir.mkdir(parents=True, exist_ok=True)
        for fname in ('interactions_final.parquet', 'items_metadata_final.parquet', 'id_mapping.json'):
            src = self.output_dir / fname
            if src.exists():
                shutil.copy2(src, backup_dir / fname)
        logger.info(f"Backup: {backup_dir}")

        # --- Optimize dtypes ---
        interactions = interactions.copy()
        target_dtypes = {col: dtype for col, dtype in _INTERACTION_DTYPES.items() if col in interactions.columns}
        interactions = interactions.astype(target_dtypes)

        # --- Write ---
        interactions.to_parquet(self.output_dir / 'interactions_final.parquet', index=False)
        metadata.to_parquet(self.output_dir / 'items_metadata_final.parquet', index=False)

        with open(self.output_dir / 'id_mapping.json', 'w') as f:
            json.dump(mapping, f, indent=2)

        # --- Stats ---
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
                'max_interactions': self.max_interactions,
            },
            'results': {
                'interactions': len(interactions),
                'users': mapping['num_users'],
                'items_total': mapping['num_items'],
                'items_trained': mapping['num_trained_items'],
                'items_catalog': mapping['num_items'] - mapping['num_trained_items'],
                'density': float(density),
                'movies': int((metadata['type'] == 'movie').sum()) if 'type' in metadata.columns else 0,
                'tv_shows': int((metadata['type'] == 'tv').sum()) if 'type' in metadata.columns else 0,
            },
        }
        with open(self.output_dir / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"\n{'=' * 70}")
        logger.info("FINAL STATS")
        logger.info(f"{'=' * 70}")
        for k, v in stats['results'].items():
            logger.info(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
        logger.info("✅ Dataset saved successfully")

    # =====================================================================
    # MAIN PIPELINE
    # =====================================================================

    def build_movie_dataset(self, amazon_dir: Optional[Path] = None) -> bool:
        """
        Movie-only pipeline: MovieLens + Amazon(movies) → data/processed/movies/.

        Юзер/item пространства независимы от TV-ветки. tmdb_id для фильмов
        используется as-is (не смещается).
        """
        self.output_dir = self.data_processed / 'movies'
        try:
            ml_ratings, ml_movies, ml_links, tmdb_raw, _tv_raw = self.load_data()

            movie_metadata = self.clean_tmdb_movies(tmdb_raw)
            if movie_metadata.empty:
                logger.error("Movie metadata empty — aborting movie branch")
                return False

            movie_metadata['genres'] = movie_metadata['genres'].apply(self._parse_genres)
            movie_metadata = self.fetch_keywords(movie_metadata)

            ml_linked = self.link_to_movielens(ml_movies, ml_links, movie_metadata)
            ml_to_tmdb = ml_linked.set_index('movie_id')['tmdb_id'].to_dict()

            ml_ratings = ml_ratings.copy()
            ml_ratings['tmdb_id'] = ml_ratings['movie_id'].map(ml_to_tmdb)
            ml_ratings = ml_ratings.dropna(subset=['tmdb_id'])
            ml_ratings['tmdb_id'] = ml_ratings['tmdb_id'].astype(int)
            ml_interactions = ml_ratings[['user_id', 'tmdb_id', 'rating', 'timestamp']].copy()

            amazon_dir = amazon_dir or Path("D:/amazon_data")
            amazon_interactions = self.load_amazon_interactions(movie_metadata, amazon_dir)

            if not amazon_interactions.empty:
                user_offset = int(ml_interactions['user_id'].max()) + 1
                amazon_interactions['user_id'] += user_offset
                unified = pd.concat([ml_interactions, amazon_interactions], ignore_index=True)
                logger.info(f"Movie unified interactions: {len(unified):,}")
            else:
                unified = ml_interactions

            ratings, trained_tmdb_ids = self.filter_interactions_unified(unified, movie_metadata)
            interactions, metadata, mapping = self.create_unified_ids(
                ratings, trained_tmdb_ids, movie_metadata
            )
            self.validate_and_save(interactions, metadata, mapping)
            logger.info(f"✅ Movie dataset written to {self.output_dir}")
            return True

        except Exception as e:
            logger.error(f"\n❌ MOVIE PIPELINE ERROR: {e}", exc_info=True)
            return False

    def build_tv_dataset(self, amazon_dir: Optional[Path] = None) -> bool:
        """
        TV-only pipeline: Trakt (+ опционально Amazon TV) → data/processed/tv/.

        Собственные user_id/item_id пространства. tmdb_id смещены на +10_000_000
        (согласовано с HotCache и прочими частями системы).
        """
        self.output_dir = self.data_processed / 'tv'
        try:
            tv_metadata = self.load_trakt_metadata()
            if tv_metadata.empty:
                logger.error("Trakt TV metadata empty — aborting TV branch")
                return False

            tv_metadata['genres'] = tv_metadata['genres'].apply(self._parse_genres)
            tv_metadata = self.fetch_keywords(tv_metadata)

            trakt_interactions = self.load_trakt_interactions(tv_metadata)
            if trakt_interactions.empty:
                logger.error("Trakt interactions empty — aborting TV branch")
                return False

            amazon_dir = amazon_dir or Path("D:/amazon_data")
            amazon_interactions = self.load_amazon_interactions(tv_metadata, amazon_dir)

            if not amazon_interactions.empty:
                user_offset = int(trakt_interactions['user_id'].max()) + 1
                amazon_interactions['user_id'] += user_offset
                unified = pd.concat([trakt_interactions, amazon_interactions], ignore_index=True)
                logger.info(f"TV unified interactions: {len(unified):,}")
            else:
                unified = trakt_interactions

            ratings, trained_tmdb_ids = self.filter_interactions_unified(unified, tv_metadata)
            interactions, metadata, mapping = self.create_unified_ids(
                ratings, trained_tmdb_ids, tv_metadata
            )
            self.validate_and_save(interactions, metadata, mapping)
            logger.info(f"✅ TV dataset written to {self.output_dir}")
            return True

        except Exception as e:
            logger.error(f"\n❌ TV PIPELINE ERROR: {e}", exc_info=True)
            return False


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build recommendation datasets per domain.")
    parser.add_argument(
        '--domain', choices=['movies', 'tv', 'all'], default='all',
        help="Which dataset to build (default: all)"
    )
    args = parser.parse_args()

    current_file = Path(__file__).resolve()
    project_root = current_file.parents[3]
    data_dir = project_root / "data"
    config_path = (
            project_root / "src" / "recommendation_system"
            / "models" / "gnn" / "config" / "genre_map.json"
    )

    logger.info(f"Project Root: {project_root}")
    logger.info(f"Data Dir:     {data_dir}")
    logger.info(f"Config:       {config_path}")
    logger.info(f"Domain:       {args.domain}")

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    processor = MovieDatasetProcessor(
        data_dir=data_dir,
        config_path=str(config_path),
        top_n_movies=15_000,
        top_n_tv=10_000,
        min_year=None,
        languages=['en'],
        min_user_interactions=10,
        min_item_interactions=10,
        rating_threshold=3.5,
        max_interactions=15_000_000,
    )

    ok = True
    if args.domain in ('movies', 'all'):
        ok &= processor.build_movie_dataset()
    if args.domain in ('tv', 'all'):
        ok &= processor.build_tv_dataset()

    exit(0 if ok else 1)


if __name__ == "__main__":
    main()
