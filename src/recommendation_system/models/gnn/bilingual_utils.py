import json
import sqlite3
import hashlib
import time
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from functools import lru_cache
import pandas as pd
import numpy as np
from recommendation_system.models.gnn.universal_search import ensure_genres, expand_compound_genres


# Optional: TMDB API
try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("requests not installed. Run: pip install requests")

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
    "Fantasy": "Фэнтези", # Теперь GoT будет попадать сюда
    "History": "Исторический",
    "Horror": "Ужасы",
    "Music": "Музыка",
    "Musical": "Мюзикл",
    "Mystery": "Детектив",
    "Romance": "Мелодрама",
    "Science Fiction": "Научная фантастика",
    "Sci-Fi": "Фантастика",
    "Thriller": "Триллер",
    "TV Movie": "ТВ-фильм",
    "War": "Военный",
    "Western": "Вестерн",
    "Children": "Детский",
    # Compound TMDB tags are now expanded BEFORE translation:
    # "Sci-Fi & Fantasy" → ["Sci-Fi", "Fantasy"] (by expand_compound_genres)
    # "Action & Adventure" → ["Action", "Adventure"]
    # So compound entries here are no longer needed.
}

GENRE_EN_TO_UK = {
    "Action": "Бойовик",
    "Adventure": "Пригоди",
    "Animation": "Мультфільм",
    "Comedy": "Комедія",
    "Crime": "Кримінал",
    "Documentary": "Документальний",
    "Drama": "Драма",
    "Family": "Сімейний",
    "Fantasy": "Фентезі",
    "History": "Історичний",
    "Horror": "Жахи",
    "Music": "Музика",
    "Musical": "Мюзикл",
    "Mystery": "Детектив",
    "Romance": "Мелодрама",
    "Science Fiction": "Наукова фантастика",
    "Sci-Fi": "Фантастика",
    "Thriller": "Трилер",
    "TV Movie": "Телефільм",
    "War": "Військовий",
    "Western": "Вестерн",
    "Children": "Дитячий",
}

GENRE_EMOJI = {
    "Action": "💥", "Adventure": "🗺️", "Animation": "🎨",
    "Comedy": "😂", "Crime": "🔫", "Documentary": "📹",
    "Drama": "🎭", "Family": "👨‍👩‍👧‍👦", "Fantasy": "🏰",
    "History": "📜", "Horror": "👻", "Music": "🎵",
    "Musical": "🎵", "Mystery": "🔍", "Romance": "💕",
    "Science Fiction": "🚀", "Sci-Fi": "🚀", "Thriller": "😱",
    "War": "⚔️", "Western": "🤠", "Children": "🧸",
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
    title_uk: str = ""
    overview_uk: Optional[str] = None
    genres_uk: List[str] = field(default_factory=list)


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
        # Composite PK (tmdb_id, media_type) — TMDB id-spaces for movies and
        # TV are independent, so a single INTEGER PRIMARY KEY collides
        # whenever a movie and a TV show share the same raw tmdb_id (e.g.
        # movie 1705 = "Battle for the Planet of the Apes", TV 1705 = Fringe).
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='translations'"
        )
        table_exists = cursor.fetchone() is not None

        if table_exists:
            cursor = self.conn.execute("PRAGMA table_info(translations)")
            columns = {row[1] for row in cursor.fetchall()}
            if 'media_type' not in columns:
                # Legacy schema — migrate by copying every row as media_type='movie'.
                # This preserves the data; TV-side rows that may have been
                # poisoned by the collision will get rewritten by the next
                # backfill pass against the (tmdb_id, 'tv') key.
                self.conn.execute("""
                    CREATE TABLE translations_new (
                        tmdb_id INTEGER NOT NULL,
                        media_type TEXT NOT NULL,
                        title_ru TEXT,
                        overview_ru TEXT,
                        poster_path TEXT,
                        title_uk TEXT,
                        overview_uk TEXT,
                        updated_at INTEGER,
                        PRIMARY KEY (tmdb_id, media_type)
                    )
                """)
                self.conn.execute("""
                    INSERT INTO translations_new
                        (tmdb_id, media_type, title_ru, overview_ru, poster_path,
                         title_uk, overview_uk, updated_at)
                    SELECT tmdb_id, 'movie', title_ru, overview_ru, poster_path,
                           title_uk, overview_uk, updated_at
                    FROM translations
                """)
                self.conn.execute("DROP TABLE translations")
                self.conn.execute("ALTER TABLE translations_new RENAME TO translations")
        else:
            self.conn.execute("""
                CREATE TABLE translations (
                    tmdb_id INTEGER NOT NULL,
                    media_type TEXT NOT NULL,
                    title_ru TEXT,
                    overview_ru TEXT,
                    poster_path TEXT,
                    title_uk TEXT,
                    overview_uk TEXT,
                    updated_at INTEGER,
                    PRIMARY KEY (tmdb_id, media_type)
                )
            """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_updated
            ON translations(updated_at)
        """)
        self.conn.commit()

    def get(self, tmdb_id: int, media_type: str) -> Optional[Dict]:
        """ """
        cursor = self.conn.execute(
            "SELECT title_ru, overview_ru, poster_path, title_uk, overview_uk "
            "FROM translations WHERE tmdb_id = ? AND media_type = ?",
            (tmdb_id, media_type)
        )
        row = cursor.fetchone()
        if row:
            return {
                'title_ru': row[0],
                'overview_ru': row[1],
                'poster_path': row[2],
                'title_uk': row[3],
                'overview_uk': row[4],
            }
        return None

    def get_batch(
        self,
        tmdb_ids: List[int],
        media_type: str,
    ) -> Dict[int, Dict]:
        """Bulk lookup for one media_type (movie or tv).

        media_type is required — without it movie/tv rows with the same raw
        tmdb_id can no longer be distinguished and the whole point of the
        composite PK collapses. Callers that need to mix domains should call
        this twice (once per type).
        """
        if not tmdb_ids:
            return {}

        placeholders = ','.join('?' * len(tmdb_ids))
        cursor = self.conn.execute(
            f"SELECT tmdb_id, title_ru, overview_ru, poster_path, title_uk, overview_uk "
            f"FROM translations WHERE media_type = ? AND tmdb_id IN ({placeholders})",
            [media_type, *tmdb_ids]
        )

        result = {}
        for row in cursor:
            result[row[0]] = {
                'title_ru': row[1],
                'overview_ru': row[2],
                'poster_path': row[3],
                'title_uk': row[4],
                'overview_uk': row[5],
            }
        return result

    def set(
        self,
        tmdb_id: int,
        media_type: str,
        title_ru: str,
        overview_ru: str = None,
        poster_path: str = None,
    ):
        """ """
        self.conn.execute("""
            INSERT OR REPLACE INTO translations
                (tmdb_id, media_type, title_ru, overview_ru, poster_path, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (tmdb_id, media_type, title_ru, overview_ru, poster_path, int(time.time())))
        self.conn.commit()

    def set_batch(
        self,
        translations: List[Tuple[int, str, str, str, str]],
    ):
        """Each tuple: (tmdb_id, media_type, title_ru, overview_ru, poster_path)."""
        now = int(time.time())
        data = [(t[0], t[1], t[2], t[3], t[4], now) for t in translations]
        self.conn.executemany("""
            INSERT OR REPLACE INTO translations
                (tmdb_id, media_type, title_ru, overview_ru, poster_path, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, data)
        self.conn.commit()

    def set_ru(
        self,
        tmdb_id: int,
        media_type: str,
        title_ru: str,
        overview_ru: Optional[str] = None,
    ) -> None:
        """Upsert just the ru columns without touching uk / poster_path.

        `set` uses INSERT OR REPLACE which clobbers any uk row written
        earlier; the ru backfill must not lose UK data, so it goes through
        this UPSERT path the same way set_uk does for the mirror case.
        """
        self.conn.execute(
            """
            INSERT INTO translations (tmdb_id, media_type, title_ru, overview_ru, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(tmdb_id, media_type) DO UPDATE SET
                title_ru = excluded.title_ru,
                overview_ru = excluded.overview_ru,
                updated_at = excluded.updated_at
            """,
            (tmdb_id, media_type, title_ru, overview_ru, int(time.time())),
        )
        self.conn.commit()

    def set_uk(
        self,
        tmdb_id: int,
        media_type: str,
        title_uk: str,
        overview_uk: Optional[str] = None,
    ) -> None:
        """Upsert just the uk columns without touching the ru ones.

        Used by the uk backfill script and the TMDB uk-fetch path so a
        first-time uk write doesn't blank existing ru data on the row.
        """
        self.conn.execute(
            """
            INSERT INTO translations (tmdb_id, media_type, title_uk, overview_uk, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(tmdb_id, media_type) DO UPDATE SET
                title_uk = excluded.title_uk,
                overview_uk = excluded.overview_uk,
                updated_at = excluded.updated_at
            """,
            (tmdb_id, media_type, title_uk, overview_uk, int(time.time())),
        )
        self.conn.commit()

    def set_uk_batch(
        self,
        translations: List[Tuple[int, str, str, Optional[str]]],
    ) -> None:
        """Each tuple: (tmdb_id, media_type, title_uk, overview_uk)."""
        now = int(time.time())
        data = [(t[0], t[1], t[2], t[3], now) for t in translations]
        self.conn.executemany(
            """
            INSERT INTO translations (tmdb_id, media_type, title_uk, overview_uk, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(tmdb_id, media_type) DO UPDATE SET
                title_uk = excluded.title_uk,
                overview_uk = excluded.overview_uk,
                updated_at = excluded.updated_at
            """,
            data,
        )
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
        cached = self.cache.get(tmdb_id, media_type)
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
                    media_type,
                    result['title_ru'],
                    result['overview_ru'],
                    result['poster_path']
                )

                return result

            elif response.status_code == 404:
                self.cache.set(tmdb_id, media_type, '', '', '')
                return None

            else:
                logger.warning(f"TMDB API error {response.status_code} for {tmdb_id}")
                return None

        except Exception as e:
            logger.error(f"TMDB API exception: {e}")
            return None

    def get_russian_translation(
        self, tmdb_id: int, media_type: str = 'movie'
    ) -> Optional[Dict]:
        """Fetch Russian title/overview for one tmdb_id.

        Mirrors get_ukrainian_translation but with language='ru-RU' and
        writes only the ru columns via cache.set_ru — so calling this does
        not clobber uk data already stored on the row.

        Note: get_movie_translation (legacy) returns the cached row as-is,
        which after the uk backfill means title_ru=NULL hits the cache and
        TMDB is never called. This method only treats the row as cached
        when title_ru itself has been populated.
        """
        cached = self.cache.get(tmdb_id, media_type)
        if cached and (cached.get('title_ru') is not None):
            return {
                'title_ru': cached.get('title_ru') or '',
                'overview_ru': cached.get('overview_ru') or '',
            }

        if not self.session:
            return None

        self._rate_limit()

        try:
            url = f"{self.BASE_URL}/{media_type}/{tmdb_id}"
            params = {'api_key': self.api_key, 'language': 'ru-RU'}
            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                title_ru = data.get('title') or data.get('name', '') or ''
                overview_ru = data.get('overview', '') or ''
                self.cache.set_ru(tmdb_id, media_type, title_ru, overview_ru)
                return {'title_ru': title_ru, 'overview_ru': overview_ru}

            if response.status_code == 404:
                self.cache.set_ru(tmdb_id, media_type, '', '')
                return {'title_ru': '', 'overview_ru': ''}

            logger.warning(
                f"TMDB ru API error {response.status_code} for {tmdb_id}"
            )
            return None

        except Exception as e:
            logger.error(f"TMDB ru API exception for {tmdb_id}: {e}")
            return None

    def get_ukrainian_translation(
        self, tmdb_id: int, media_type: str = 'movie'
    ) -> Optional[Dict]:
        """Fetch Ukrainian title/overview for one tmdb_id.

        Mirrors get_movie_translation but with language='uk-UA' and writes
        only the uk columns via cache.set_uk — so calling this does not
        clobber an existing ru row.

        Returns dict with 'title_uk', 'overview_uk' or None on hard failure.
        Empty/missing uk on TMDB returns dict with empty strings (cached as
        a negative result so we don't hit TMDB again on the next pass).
        """
        cached = self.cache.get(tmdb_id, media_type)
        if cached and (cached.get('title_uk') is not None):
            return {
                'title_uk': cached.get('title_uk') or '',
                'overview_uk': cached.get('overview_uk') or '',
            }

        if not self.session:
            return None

        self._rate_limit()

        try:
            url = f"{self.BASE_URL}/{media_type}/{tmdb_id}"
            params = {'api_key': self.api_key, 'language': 'uk-UA'}
            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                title_uk = data.get('title') or data.get('name', '') or ''
                overview_uk = data.get('overview', '') or ''
                # If TMDB returned uk-locale content that's actually English
                # (no uk translation available), it leaks through as the
                # original title. We still cache as empty to mark "checked".
                self.cache.set_uk(tmdb_id, media_type, title_uk, overview_uk)
                return {'title_uk': title_uk, 'overview_uk': overview_uk}

            if response.status_code == 404:
                self.cache.set_uk(tmdb_id, media_type, '', '')
                return {'title_uk': '', 'overview_uk': ''}

            logger.warning(
                f"TMDB uk API error {response.status_code} for {tmdb_id}"
            )
            return None

        except Exception as e:
            logger.error(f"TMDB uk API exception for {tmdb_id}: {e}")
            return None

    def prefetch_translations(
            self,
            tmdb_ids: List[int],
            media_types: List[str] = None,
            progress_callback=None
    ) -> int:
        if media_types is None:
            media_types = ['movie'] * len(tmdb_ids)

        movie_ids = [id_ for id_, mt in zip(tmdb_ids, media_types) if mt == 'movie']
        tv_ids = [id_ for id_, mt in zip(tmdb_ids, media_types) if mt == 'tv']
        cached_movie = self.cache.get_batch(movie_ids, 'movie') if movie_ids else {}
        cached_tv = self.cache.get_batch(tv_ids, 'tv') if tv_ids else {}
        to_fetch = [
            (id_, mt) for id_, mt in zip(tmdb_ids, media_types)
            if (mt == 'movie' and id_ not in cached_movie)
            or (mt == 'tv' and id_ not in cached_tv)
            or mt not in ('movie', 'tv')
        ]
        cached_total = len(cached_movie) + len(cached_tv)

        logger.info(f"Prefetch: {len(to_fetch)} нових з {len(tmdb_ids)} (в кеші: {cached_total})")

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
            cached = self.cache.get(int(tmdb_id), media_type)
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

        # Partition known tmdb_ids by media_type so the per-domain composite
        # PK lookup hits the right rows. We then merge — collisions across
        # types can no longer happen because each row is keyed by both.
        movie_ids: List[int] = []
        tv_ids: List[int] = []
        if 'type' in self.metadata.columns and 'tmdb_id' in self.metadata.columns:
            for _, row in self.metadata[['tmdb_id', 'type']].dropna(subset=['tmdb_id']).iterrows():
                tmdb_id = int(row['tmdb_id'])
                if tmdb_id not in self.tmdb_to_item:
                    continue
                if str(row['type']) == 'tv':
                    tv_ids.append(tmdb_id)
                else:
                    movie_ids.append(tmdb_id)
        else:
            movie_ids = list(self.tmdb_to_item.keys())

        cached: Dict[int, Dict] = {}
        if movie_ids:
            cached.update(self.cache.get_batch(movie_ids, 'movie'))
        if tv_ids:
            cached.update(self.cache.get_batch(tv_ids, 'tv'))

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

    def get_movie_info(self, item_id: int):
        """
        Fixed version: uses ensure_genres() for reliable genre parsing.
        Replaces the old ad-hoc ast.literal_eval / string splitting approach.
        """
        row = self.metadata[self.metadata['item_id'] == item_id]
        if len(row) == 0:
            return None

        row = row.iloc[0]

        title_en = str(row.get('title', 'Unknown'))
        year = int(row.get('year', 2000))
        tmdb_id = row.get('tmdb_id')

        # ---- GENRE FIX: single source of truth ----
        # Before: 15 lines of fragile ad-hoc parsing
        # After:  1 line + expand compounds
        from recommendation_system.models.gnn.universal_search import (
            ensure_genres,
            expand_compound_genres,
        )
        genres = ensure_genres(row.get('genres', []))

        # Russian title
        title_ru = self._get_russian_title(item_id)

        # Russian genres (compounds already expanded by ensure_genres)
        genres_ru = [GENRE_EN_TO_RU.get(g, g) for g in genres]

        # Overview
        overview_ru = None
        if pd.notna(tmdb_id):
            media_type = 'tv' if row.get('type') == 'tv' else 'movie'
            cached = self.cache.get(int(tmdb_id), media_type)
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
        """Format genres bilingually, expanding compound tags first."""
        expanded = expand_compound_genres(genres_en)
        genres_ru = [GENRE_EN_TO_RU.get(g, g) for g in expanded]
        return f"{', '.join(expanded)} — {', '.join(genres_ru)}"

    def get_genre_emojis(self, genres: List[str]) -> str:
        """Get emoji string for genres, expanding compound TMDB tags first."""
        expanded = expand_compound_genres(genres)
        emojis = [GENRE_EMOJI.get(g, '') for g in expanded[:5] if g in GENRE_EMOJI]
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
