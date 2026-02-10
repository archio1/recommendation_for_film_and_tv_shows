import json
import sqlite3
import hashlib
import time
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from functools import lru_cache
import pandas as pd
import numpy as np

# Optional: TMDB API
try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("⚠️ requests not installed. Run: pip install requests")

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

GENRE_EN_TO_RU = {
    "Action": "Боевик",
    "Adventure": "Приключения",
    "Animation": "Мультфильм",
    "Comedy": "Комедия",
    "Crime": "Криминал",
    "Documentary": "Документальный",
    "Drama": "Драма",
    "Family": "Семейный",
    "Fantasy": "Фэнтези",
    "History": "Исторический",
    "Horror": "Ужасы",
    "Music": "Музыка",
    "Mystery": "Детектив",
    "Romance": "Мелодрама",
    "Science Fiction": "Научная фантастика",
    "Sci-Fi": "Фантастика",
    "Thriller": "Триллер",
    "TV Movie": "ТВ-фильм",
    "War": "Военный",
    "Western": "Вестерн",
}

GENRE_EMOJI = {
    "Action": "💥", "Adventure": "🗺️", "Animation": "🎨",
    "Comedy": "😂", "Crime": "🔫", "Documentary": "📹",
    "Drama": "🎭", "Family": "👨‍👩‍👧‍👦", "Fantasy": "🏰",
    "History": "📜", "Horror": "👻", "Music": "🎵",
    "Mystery": "🔍", "Romance": "💕", "Science Fiction": "🚀",
    "Sci-Fi": "🚀", "Thriller": "😱", "War": "⚔️", "Western": "🤠",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MovieInfo:
    """ """
    item_id: int
    title_en: str
    title_ru: str
    year: int
    genres_en: List[str]
    genres_ru: List[str]
    tmdb_id: Optional[int] = None
    imdb_id: Optional[str] = None
    overview_ru: Optional[str] = None
    poster_path: Optional[str] = None
    confidence: float = 1.0


@dataclass
class SearchResult:
    """ """
    query: str
    matches: List[MovieInfo]
    error: Optional[str] = None


# =============================================================================
# TMDB TRANSLATION CACHE (SQLite)
# =============================================================================

class TMDBTranslationCache:
    """

    """

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.conn = sqlite3.connect(str(cache_path), check_same_thread=False)
        self._init_db()

    def _init_db(self):
        """ """
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS translations (
                tmdb_id INTEGER PRIMARY KEY,
                title_ru TEXT,
                overview_ru TEXT,
                poster_path TEXT,
                updated_at INTEGER
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_updated 
            ON translations(updated_at)
        """)
        self.conn.commit()

    def get(self, tmdb_id: int) -> Optional[Dict]:
        """ """
        cursor = self.conn.execute(
            "SELECT title_ru, overview_ru, poster_path FROM translations WHERE tmdb_id = ?",
            (tmdb_id,)
        )
        row = cursor.fetchone()
        if row:
            return {
                'title_ru': row[0],
                'overview_ru': row[1],
                'poster_path': row[2]
            }
        return None

    def get_batch(self, tmdb_ids: List[int]) -> Dict[int, Dict]:
        """ """
        if not tmdb_ids:
            return {}

        placeholders = ','.join('?' * len(tmdb_ids))
        cursor = self.conn.execute(
            f"SELECT tmdb_id, title_ru, overview_ru, poster_path FROM translations WHERE tmdb_id IN ({placeholders})",
            tmdb_ids
        )

        result = {}
        for row in cursor:
            result[row[0]] = {
                'title_ru': row[1],
                'overview_ru': row[2],
                'poster_path': row[3]
            }
        return result

    def set(self, tmdb_id: int, title_ru: str, overview_ru: str = None, poster_path: str = None):
        """ """
        self.conn.execute("""
            INSERT OR REPLACE INTO translations (tmdb_id, title_ru, overview_ru, poster_path, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (tmdb_id, title_ru, overview_ru, poster_path, int(time.time())))
        self.conn.commit()

    def set_batch(self, translations: List[Tuple[int, str, str, str]]):
        """ """
        now = int(time.time())
        data = [(t[0], t[1], t[2], t[3], now) for t in translations]
        self.conn.executemany("""
            INSERT OR REPLACE INTO translations (tmdb_id, title_ru, overview_ru, poster_path, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, data)
        self.conn.commit()

    def stats(self) -> Dict:
        """ """
        cursor = self.conn.execute("SELECT COUNT(*), MIN(updated_at), MAX(updated_at) FROM translations")
        row = cursor.fetchone()
        return {
            'total': row[0],
            'oldest': row[1],
            'newest': row[2]
        }

    def close(self):
        self.conn.close()


# =============================================================================
# TMDB API CLIENT
# =============================================================================

class TMDBClient:
    """

    Rate limits: 40 requests / 10 seconds
    """

    BASE_URL = "https://api.themoviedb.org/3"

    def __init__(self, api_key: str, cache: TMDBTranslationCache):
        self.api_key = api_key
        self.cache = cache
        self.session = requests.Session() if HAS_REQUESTS else None
        self._request_times = []

    def _rate_limit(self):
        """ """
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 10]

        if len(self._request_times) >= 35:
            sleep_time = 10 - (now - self._request_times[0]) + 0.1
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._request_times.append(time.time())

    def get_movie_translation(self, tmdb_id: int, media_type: str = 'movie') -> Optional[Dict]:
        """

        Args:
            tmdb_id: TMDB ID
            media_type: 'movie' або 'tv'

        Returns:
            {'title_ru': '...', 'overview_ru': '...', 'poster_path': '...'} або None
        """
        # 1. Спочатку перевіряємо кеш
        cached = self.cache.get(tmdb_id)
        if cached:
            return cached

        if not self.session:
            return None

        self._rate_limit()

        try:
            url = f"{self.BASE_URL}/{media_type}/{tmdb_id}"
            params = {
                'api_key': self.api_key,
                'language': 'ru-RU'
            }

            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                result = {
                    'title_ru': data.get('title') or data.get('name', ''),
                    'overview_ru': data.get('overview', ''),
                    'poster_path': data.get('poster_path', '')
                }

                self.cache.set(
                    tmdb_id,
                    result['title_ru'],
                    result['overview_ru'],
                    result['poster_path']
                )

                return result

            elif response.status_code == 404:
                self.cache.set(tmdb_id, '', '', '')
                return None

            else:
                logger.warning(f"TMDB API error {response.status_code} for {tmdb_id}")
                return None

        except Exception as e:
            logger.error(f"TMDB API exception: {e}")
            return None

    def prefetch_translations(
            self,
            tmdb_ids: List[int],
            media_types: List[str] = None,
            progress_callback=None
    ) -> int:
        if media_types is None:
            media_types = ['movie'] * len(tmdb_ids)

        cached = self.cache.get_batch(tmdb_ids)
        to_fetch = [(id_, mt) for id_, mt in zip(tmdb_ids, media_types) if id_ not in cached]

        logger.info(f"Prefetch: {len(to_fetch)} нових з {len(tmdb_ids)} (в кеші: {len(cached)})")

        fetched = 0
        for i, (tmdb_id, media_type) in enumerate(to_fetch):
            result = self.get_movie_translation(tmdb_id, media_type)
            if result and result.get('title_ru'):
                fetched += 1

            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(i + 1, len(to_fetch))

        return fetched


# =============================================================================
# SCALABLE MOVIE INTELLIGENCE
# =============================================================================

class ScalableMovieIntelligence:
    """
    """

    def __init__(
            self,
            metadata: pd.DataFrame,
            cache_dir: Path = None,
            tmdb_api_key: str = None
    ):
        """

        """
        self.metadata = metadata.copy()

        # Prepare cache
        if cache_dir is None:
            cache_dir = Path.home() / '.movie_intelligence_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache = TMDBTranslationCache(cache_dir / 'translations.db')

        # TMDB client (optional)
        self.tmdb_client = None
        if tmdb_api_key and HAS_REQUESTS:
            self.tmdb_client = TMDBClient(tmdb_api_key, self.cache)

        # Build search indices
        self._build_indices()

        logger.info(
            f"ScalableMovieIntelligence: {len(metadata)} фільмів, кеш: {self.cache.stats()['total']} перекладів")

    def _build_indices(self):
        """ """
        self.title_to_items = {}  # normalized_title -> [item_ids]

        for _, row in self.metadata.iterrows():
            item_id = row['item_id']
            title = str(row.get('title', '')).lower()
            normalized = self._normalize_text(title)

            if normalized not in self.title_to_items:
                self.title_to_items[normalized] = []
            self.title_to_items[normalized].append(item_id)

            for prefix in ['the ', 'a ', 'an ']:
                if normalized.startswith(prefix):
                    alt = normalized[len(prefix):]
                    if alt not in self.title_to_items:
                        self.title_to_items[alt] = []
                    self.title_to_items[alt].append(item_id)

        # tmdb_id -> item_id mapping
        self.tmdb_to_item = {}
        if 'tmdb_id' in self.metadata.columns:
            for _, row in self.metadata.iterrows():
                tmdb_id = row.get('tmdb_id')
                if pd.notna(tmdb_id):
                    self.tmdb_to_item[int(tmdb_id)] = row['item_id']

    def _normalize_text(self, text: str) -> str:
        """ """
        text = text.lower().strip()
        text = re.sub(r'[^\w\s\d]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _is_cyrillic(self, text: str) -> bool:
        """ """
        return bool(re.search(r'[а-яёА-ЯЁ]', text))

    def _get_russian_title(self, item_id: int) -> str:
        """
        """
        row = self.metadata[self.metadata['item_id'] == item_id]
        if len(row) == 0:
            return "Unknown"

        row = row.iloc[0]
        title_en = str(row.get('title', 'Unknown'))
        tmdb_id = row.get('tmdb_id')
        media_type = 'tv' if row.get('type') == 'tv' else 'movie'

        if pd.notna(tmdb_id):
            cached = self.cache.get(int(tmdb_id))
            if cached and cached.get('title_ru'):
                return cached['title_ru']

            if self.tmdb_client:
                result = self.tmdb_client.get_movie_translation(int(tmdb_id), media_type)
                if result and result.get('title_ru'):
                    return result['title_ru']

        return title_en

    def _fuzzy_search(self, query: str, threshold: float = 0.5) -> List[Tuple[int, float]]:
        """ """
        query_normalized = self._normalize_text(query)
        results = []

        for title, item_ids in self.title_to_items.items():
            if title == query_normalized:
                for item_id in item_ids:
                    results.append((item_id, 1.0))
                continue

            if query_normalized in title or title in query_normalized:
                score = 0.8
                for item_id in item_ids:
                    results.append((item_id, score))
                continue

            ratio = SequenceMatcher(None, query_normalized, title).ratio()
            if ratio >= threshold:
                for item_id in item_ids:
                    results.append((item_id, ratio))

        results.sort(key=lambda x: -x[1])
        seen = set()
        unique = []
        for item_id, score in results:
            if item_id not in seen:
                seen.add(item_id)
                unique.append((item_id, score))

        return unique[:20]  # Top 20

    def _search_russian(self, query: str) -> List[Tuple[int, float]]:
        """

        """
        query_lower = self._normalize_text(query)
        results = []

        all_tmdb_ids = list(self.tmdb_to_item.keys())

        cached = self.cache.get_batch(all_tmdb_ids)

        for tmdb_id, translation in cached.items():
            title_ru = translation.get('title_ru', '')
            if not title_ru:
                continue

            title_ru_normalized = self._normalize_text(title_ru)

            if title_ru_normalized == query_lower:
                if tmdb_id in self.tmdb_to_item:
                    results.append((self.tmdb_to_item[tmdb_id], 1.0))
                continue

            if query_lower in title_ru_normalized or title_ru_normalized in query_lower:
                if tmdb_id in self.tmdb_to_item:
                    results.append((self.tmdb_to_item[tmdb_id], 0.8))
                continue

            ratio = SequenceMatcher(None, query_lower, title_ru_normalized).ratio()
            if ratio >= 0.6:
                if tmdb_id in self.tmdb_to_item:
                    results.append((self.tmdb_to_item[tmdb_id], ratio))

        results.sort(key=lambda x: -x[1])
        return results[:20]

    def search(self, query: str, limit: int = 8) -> SearchResult:
        """

        """
        results = []

        if self._is_cyrillic(query):
            matches = self._search_russian(query)

            if not matches:
                pass
        else:
            matches = self._fuzzy_search(query)

        for item_id, score in matches[:limit]:
            info = self.get_movie_info(item_id)
            if info:
                info.confidence = score
                results.append(info)

        error = None if results else f"По запросу «{query}» ничего не найдено."

        return SearchResult(query=query, matches=results, error=error)

    def get_movie_info(self, item_id: int) -> Optional[MovieInfo]:
        """ """
        row = self.metadata[self.metadata['item_id'] == item_id]
        if len(row) == 0:
            return None

        row = row.iloc[0]

        title_en = str(row.get('title', 'Unknown'))
        year = int(row.get('year', 2000))
        tmdb_id = row.get('tmdb_id')

        # Genres
        genres = row.get('genres', [])
        if isinstance(genres, str):
            # Если это строка типа "['Action', 'Horror']" или "Action, Horror"
            if '[' in genres:
                try:
                    import ast
                    genres = ast.literal_eval(genres)
                except:
                    genres = []
            else:
                genres = [g.strip() for g in genres.split(',') if g.strip()]

        if not isinstance(genres, (list, np.ndarray)):
            genres = []

        # Убираем возможные пустые значения
        genres = [g for g in genres if g and str(g).lower() != 'nan']

        # Russian title
        title_ru = self._get_russian_title(item_id)

        # Russian genres
        genres_ru = [GENRE_EN_TO_RU.get(g, g) for g in genres]

        # Overview
        overview_ru = None
        if pd.notna(tmdb_id):
            cached = self.cache.get(int(tmdb_id))
            if cached:
                overview_ru = cached.get('overview_ru')

        return MovieInfo(
            item_id=item_id,
            title_en=title_en,
            title_ru=title_ru,
            year=year,
            genres_en=genres,
            genres_ru=genres_ru,
            tmdb_id=int(tmdb_id) if pd.notna(tmdb_id) else None,
            overview_ru=overview_ru
        )

    def format_genres_bilingual(self, genres_en: List[str]) -> str:
        """ """
        genres_ru = [GENRE_EN_TO_RU.get(g, g) for g in genres_en]
        return f"{', '.join(genres_en)} — {', '.join(genres_ru)}"

    def get_genre_emojis(self, genres: List[str]) -> str:
        """ """
        emojis = [GENRE_EMOJI.get(g, '') for g in genres[:3] if g in GENRE_EMOJI]
        return ''.join(emojis) if emojis else "🎬"

    def prefetch_all_translations(self, progress_callback=None):
        """
        """
        if not self.tmdb_client:
            logger.error("TMDB client not configured. Set tmdb_api_key.")
            return

        tmdb_data = []
        for _, row in self.metadata.iterrows():
            tmdb_id = row.get('tmdb_id')
            if pd.notna(tmdb_id):
                media_type = 'tv' if row.get('type') == 'tv' else 'movie'
                tmdb_data.append((int(tmdb_id), media_type))

        tmdb_ids = [x[0] for x in tmdb_data]
        media_types = [x[1] for x in tmdb_data]

        fetched = self.tmdb_client.prefetch_translations(
            tmdb_ids,
            media_types,
            progress_callback
        )

        logger.info(f"Prefetch завершено: {fetched} нових перекладів")
        return fetched

    def close(self):
        self.cache.close()


# =============================================================================
# INTEGRATION WRAPPER
# =============================================================================

class ScalableBilingualEngine:
    """
    """

    def __init__(
            self,
            base_engine,  # InferenceEngine
            cache_dir: Path = None,
            tmdb_api_key: str = None
    ):
        self.engine = base_engine
        self.metadata = base_engine.metadata

        self.intelligence = ScalableMovieIntelligence(
            metadata=self.metadata,
            cache_dir=cache_dir,
            tmdb_api_key=tmdb_api_key
        )

    def search_movies_bilingual(self, query: str, limit: int = 8) -> List[Dict]:
        result = self.intelligence.search(query, limit=limit)

        return [
            {
                'title_ru': m.title_ru,
                'title_en': m.title_en,
                'year': m.year,
                'item_id': m.item_id,
                'confidence': m.confidence
            }
            for m in result.matches
        ]

    def get_recommendations_bilingual(self, liked_ids: List[int], top_k: int = 8) -> List[Dict]:
        base_recs = self.engine.get_recommendations(liked_ids, top_k=top_k)

        bilingual = []
        for rec in base_recs:
            item_id = rec.get('item_id')
            if item_id is None:
                matches = self.metadata[self.metadata['title'] == rec['title']]
                if len(matches) > 0:
                    item_id = matches.iloc[0]['item_id']
                else:
                    continue

            info = self.intelligence.get_movie_info(item_id)
            if info:
                bilingual.append({
                    'title_ru': info.title_ru,
                    'title_en': info.title_en,
                    'year': info.year,
                    'genres': self.intelligence.format_genres_bilingual(info.genres_en),
                    'emoji_icons': self.intelligence.get_genre_emojis(info.genres_en),
                    'reason_ru': self._generate_reason(info.genres_en),
                    'imdb_url': rec.get('imdb_url')
                })

        return bilingual

    def _generate_reason(self, genres: List[str]) -> str:
        reasons = {
            "Animation": "Отличный анимационный фильм с глубоким сюжетом.",
            "Action": "Захватывающий боевик с динамичным экшеном.",
            "Comedy": "Комедия, которая поднимет настроение.",
            "Drama": "Глубокая драма с сильными персонажами.",
            "Science Fiction": "Фантастика с интересными идеями о будущем.",
            "Fantasy": "Волшебный мир фэнтези и приключений.",
            "Horror": "Атмосферный ужастик для любителей острых ощущений.",
            "Romance": "Трогательная история о любви.",
            "Thriller": "Напряжённый триллер с неожиданными поворотами.",
        }

        for genre in genres:
            if genre in reasons:
                return reasons[genre]

        return "Рекомендуем к просмотру!"

    def prefetch_translations(self, progress_callback=None):
        return self.intelligence.prefetch_all_translations(progress_callback)


# =============================================================================
# STANDALONE PREFETCH SCRIPT
# =============================================================================

def prefetch_cli():
    """CLI для prefetch переводов"""
    import argparse

    parser = argparse.ArgumentParser(description='Prefetch Russian translations from TMDB')
    parser.add_argument('--metadata', required=True, help='Path to items_metadata_final.parquet')
    parser.add_argument('--api-key', required=True, help='TMDB API key')
    parser.add_argument('--cache-dir', default=None, help='Cache directory')

    args = parser.parse_args()

    metadata = pd.read_parquet(args.metadata)

    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    intel = ScalableMovieIntelligence(
        metadata=metadata,
        cache_dir=cache_dir,
        tmdb_api_key=args.api_key
    )

    def progress(current, total):
        print(f"\rProgress: {current}/{total} ({100 * current / total:.1f}%)", end='', flush=True)

    intel.prefetch_all_translations(progress_callback=progress)
    print("\n✅ Done!")

    intel.close()


if __name__ == "__main__":
    # Quick test
    print("ScalableMovieIntelligence — Test Mode")
    print("=" * 50)

    # Create dummy metadata
    dummy_data = pd.DataFrame({
        'item_id': [0, 1, 2, 3],
        'title': ['The Matrix', 'Shrek', 'Inception', 'Frozen'],
        'year': [1999, 2001, 2010, 2013],
        'genres': [['Action', 'Sci-Fi'], ['Animation', 'Comedy'], ['Sci-Fi', 'Thriller'], ['Animation', 'Family']],
        'tmdb_id': [603, 808, 27205, 109445],
        'type': ['movie', 'movie', 'movie', 'movie']
    })

    intel = ScalableMovieIntelligence(
        metadata=dummy_data,
        cache_dir=Path('/tmp/movie_cache')
    )

    # Test search
    for query in ['matrix', 'шрек', 'inception']:
        result = intel.search(query)
        print(f"\nQuery: '{query}'")
        for m in result.matches:
            print(f"  → {m.title_ru} / {m.title_en} ({m.year}) [conf: {m.confidence:.2f}]")

    intel.close()
    print("\n✅ Test passed!")
