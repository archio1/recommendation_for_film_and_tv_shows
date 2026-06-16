"""
trakt_collector.py — Multi-phase TV show dataset builder from trakt.tv API

Collects TV show metadata and user-item interactions for the recommendation system.
Designed for multi-day collection with safe shutdown/resume via SQLite checkpoints.

Usage (m = python -m recommendation_system.data.trakt_collector):
    m                     # run all phases (auto-resume)
    m --phase discover    # only phase 1
    m --phase enrich      # only phase 2
    m --phase network     # only phase 2.5 (follower crawl)
    m --phase users       # only phase 3
    m --phase export      # only phase 4
    m --max-shows 5000    # limit discovered shows
    m --max-users 80000   # limit users to collect
    m --comment-pages 2   # comment pages per show
    m --reset-phase users # reset a phase to re-run it
"""

import argparse
import collections
import csv
import json
import logging
import os
import random
import sqlite3
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from recommendation_system.paths import ENV_FILE, RAW_DIR

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trakt_collector.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAKT_BASE_URL = "https://api.trakt.tv"
TRAKT_API_VERSION = "2"
RATE_LIMIT_MAX = 950          # stay under 1000/5min hard cap
RATE_LIMIT_WINDOW = 300       # 5 minutes in seconds
REQUEST_TIMEOUT = 30          # seconds per request
BATCH_COMMIT_SIZE = 50        # commit to DB every N items


# ===========================================================================
# Rate Limiter
# ===========================================================================
class TraktRateLimiter:
    """Token-bucket rate limiter: max RATE_LIMIT_MAX requests per RATE_LIMIT_WINDOW seconds."""

    def __init__(self):
        self._timestamps: collections.deque = collections.deque()

    def wait_if_needed(self):
        now = time.monotonic()
        # Purge timestamps outside the window
        while self._timestamps and self._timestamps[0] < now - RATE_LIMIT_WINDOW:
            self._timestamps.popleft()
        if len(self._timestamps) >= RATE_LIMIT_MAX:
            sleep_for = self._timestamps[0] + RATE_LIMIT_WINDOW - now + 0.5
            if sleep_for > 0:
                logger.info(f"Rate limit: sleeping {sleep_for:.1f}s")
                time.sleep(sleep_for)
        self._timestamps.append(time.monotonic())


# ===========================================================================
# Trakt API Client
# ===========================================================================
class TraktAPI:
    """Low-level HTTP client with rate limiting, retries, and auth headers."""

    def __init__(self, client_id: str):
        self.client_id = client_id
        self.limiter = TraktRateLimiter()
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "trakt-api-version": TRAKT_API_VERSION,
            "trakt-api-key": self.client_id,
        })
        self._total_requests = 0

    @property
    def total_requests(self) -> int:
        return self._total_requests

    def _request(self, path: str, params: dict | None = None) -> dict | list | None:
        """Execute a GET request with rate limiting and retries."""

        url = f"{TRAKT_BASE_URL}{path}"
        retries = [5, 15, 45]

        for attempt in range(len(retries) + 1):
            self.limiter.wait_if_needed()
            try:
                resp = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                self._total_requests += 1
            except requests.ConnectionError:
                logger.warning(f"ConnectionError: {path} — retry in 30s")
                time.sleep(30)
                continue
            except requests.Timeout:
                logger.warning(f"Timeout: {path} — retry once")
                continue

            if resp.status_code == 200:
                try:
                    data = resp.json()

                    # Защита от случаев, когда API возвращает строку вместо JSON
                    if isinstance(data, str):
                        logger.warning(f"API returned string instead of JSON for {path}")
                        return None

                    return data

                except json.JSONDecodeError:
                    logger.error(f"JSON decode error on {path}")
                    logger.debug(f"Response text (first 500 chars): {resp.text[:500]}")
                    return None

            if resp.status_code == 204:
                return []

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 10))
                logger.warning(f"429 Rate limited: sleeping {retry_after}s")
                time.sleep(retry_after)
                continue

            if resp.status_code in (401, 404, 403):
                logger.debug(f"Expected error {resp.status_code} on {path} (user may be private/deleted)")
                return None

            if resp.status_code >= 500:
                if attempt < len(retries):
                    logger.warning(f"HTTP {resp.status_code} on {path} — retry in {retries[attempt]}s")
                    time.sleep(retries[attempt])
                    continue

                logger.error(f"HTTP {resp.status_code} on {path} — giving up")
                return None

                # Неожиданные статус-коды (405, 400, 406 и т.д.)
            logger.warning(f"Unexpected HTTP {resp.status_code} on {path}")
            logger.debug(f"Response: {resp.text[:400]}")  # ← на DEBUG, чтобы не засорять
            return None
        return None

    # --- Show list endpoints (paginated) ---
    def get_show_list(self, list_name: str, page: int = 1, limit: int = 100) -> list | None:
        """Fetch a paginated list of shows (popular, watched, trending, played, collected)."""
        return self._request(f"/shows/{list_name}", {"page": page, "limit": limit, "extended": "full"})

    # --- Single show endpoints ---
    def get_show_full(self, slug: str) -> dict | None:
        return self._request(f"/shows/{slug}", {"extended": "full"})

    def get_show_stats(self, slug: str) -> dict | None:
        return self._request(f"/shows/{slug}/stats")

    def get_show_comments(self, slug: str, page: int = 1, limit: int = 100) -> list | None:
        return self._request(f"/shows/{slug}/comments/newest", {"page": page, "limit": limit})

    def get_show_lists(self, slug: str, page: int = 1, limit: int = 100) -> list | None:
        return self._request(f"/shows/{slug}/lists/personal", {"page": page, "limit": limit})

    def get_show_related(self, slug: str) -> list | None:
        return self._request(f"/shows/{slug}/related", {"limit": 10})

    # --- User endpoints ---
    def get_user_ratings(self, username: str) -> list | None:
        safe_username = urllib.parse.quote(username)
        return self._request(f"/users/{safe_username}/ratings/shows")

    def get_user_followers(self, username: str) -> list | None:
        safe_username = urllib.parse.quote(username)
        return self._request(f"/users/{safe_username}/followers")

    def get_user_following(self, username: str) -> list | None:
        safe_username = urllib.parse.quote(username)
        return self._request(f"/users/{safe_username}/following")

# ===========================================================================
# Checkpoint DB
# ===========================================================================
class CheckpointDB:
    """SQLite checkpoint and staging database with WAL mode for crash safety."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS progress (
        phase       TEXT PRIMARY KEY,
        status      TEXT DEFAULT 'pending',
        cursor      TEXT,
        updated_at  TEXT
    );
    CREATE TABLE IF NOT EXISTS shows (
        trakt_id    INTEGER PRIMARY KEY,
        slug        TEXT,
        tmdb_id     INTEGER,
        imdb_id     TEXT,
        title       TEXT,
        year        INTEGER,
        source      TEXT,
        metadata_fetched INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS show_metadata (
        trakt_id    INTEGER PRIMARY KEY,
        tmdb_id     INTEGER,
        title       TEXT,
        year        INTEGER,
        overview    TEXT,
        genres      TEXT,
        network     TEXT,
        status      TEXT,
        rating      REAL,
        votes       INTEGER,
        watchers    INTEGER,
        plays       INTEGER,
        language    TEXT,
        country     TEXT,
        runtime     INTEGER
    );
    CREATE TABLE IF NOT EXISTS users (
        username        TEXT PRIMARY KEY,
        source          TEXT,
        ratings_fetched INTEGER DEFAULT 0,
        is_private      INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS user_ratings (
        username    TEXT,
        trakt_id    INTEGER,
        tmdb_id     INTEGER,
        rating      INTEGER,
        rated_at    TEXT,
        UNIQUE(username, trakt_id)
    );
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(self.SCHEMA)
        # Migration: add network_crawled column for Phase 2.5
        try:
            self.conn.execute("ALTER TABLE users ADD COLUMN network_crawled INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # column already exists
        self.conn.commit()

    def close(self):
        self.conn.close()

    # --- Progress tracking ---
    def get_phase(self, phase: str) -> tuple[str, str | None]:
        row = self.conn.execute(
            "SELECT status, cursor FROM progress WHERE phase = ?", (phase,)
        ).fetchone()
        if row:
            return row[0], row[1]
        return "pending", None

    def set_phase(self, phase: str, status: str, cursor: str | None = None):
        self.conn.execute(
            "INSERT OR REPLACE INTO progress (phase, status, cursor, updated_at) VALUES (?, ?, ?, ?)",
            (phase, status, cursor, datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    # --- Shows ---
    def upsert_show(self, trakt_id: int, slug: str, tmdb_id: int | None,
                    imdb_id: str | None, title: str, year: int | None, source: str):
        self.conn.execute(
            """INSERT OR IGNORE INTO shows (trakt_id, slug, tmdb_id, imdb_id, title, year, source)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (trakt_id, slug, tmdb_id, imdb_id, title, year, source),
        )

    def get_show_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM shows").fetchone()[0]

    def get_shows_needing_metadata(self, limit: int = 0) -> list[tuple]:
        q = "SELECT trakt_id, slug FROM shows WHERE metadata_fetched = 0 ORDER BY trakt_id"
        if limit > 0:
            q += f" LIMIT {limit}"
        return self.conn.execute(q).fetchall()

    def mark_show_metadata_done(self, trakt_id: int):
        self.conn.execute("UPDATE shows SET metadata_fetched = 1 WHERE trakt_id = ?", (trakt_id,))

    def upsert_show_metadata(self, trakt_id: int, tmdb_id: int | None, title: str,
                             year: int | None, overview: str | None, genres: str,
                             network: str | None, status: str | None, rating: float | None,
                             votes: int | None, watchers: int | None, plays: int | None,
                             language: str | None, country: str | None, runtime: int | None):
        self.conn.execute(
            """INSERT OR REPLACE INTO show_metadata
               (trakt_id, tmdb_id, title, year, overview, genres, network, status,
                rating, votes, watchers, plays, language, country, runtime)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (trakt_id, tmdb_id, title, year, overview, genres, network, status,
             rating, votes, watchers, plays, language, country, runtime),
        )

    # --- Users ---
    def upsert_user(self, username: str, source: str):
        self.conn.execute(
            "INSERT OR IGNORE INTO users (username, source) VALUES (?, ?)",
            (username, source),
        )

    def get_user_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def get_users_needing_ratings(self, limit: int = 0) -> list[tuple]:
        q = ("SELECT username FROM users "
             "WHERE ratings_fetched = 0 AND is_private = 0 "
             "ORDER BY username")
        if limit > 0:
            q += f" LIMIT {limit}"
        return self.conn.execute(q).fetchall()

    def mark_user_done(self, username: str):
        self.conn.execute("UPDATE users SET ratings_fetched = 1 WHERE username = ?", (username,))

    def mark_user_private(self, username: str):
        self.conn.execute(
            "UPDATE users SET is_private = 1, ratings_fetched = 1 WHERE username = ?",
            (username,),
        )

    # --- Network crawl ---
    def get_users_for_network_crawl(self, min_ratings: int = 10) -> list[tuple]:
        """Get active users whose followers/following haven't been crawled yet."""
        return self.conn.execute(
            """SELECT u.username FROM users u
               JOIN (SELECT username, COUNT(*) as cnt FROM user_ratings
                     GROUP BY username HAVING cnt >= ?) r
               ON u.username = r.username
               WHERE u.network_crawled = 0 AND u.is_private = 0
               ORDER BY u.username""",
            (min_ratings,),
        ).fetchall()

    def mark_user_network_crawled(self, username: str):
        self.conn.execute(
            "UPDATE users SET network_crawled = 1 WHERE username = ?", (username,),
        )

    # --- User ratings ---
    def upsert_user_rating(self, username: str, trakt_id: int, tmdb_id: int | None,
                           rating: int, rated_at: str | None):
        self.conn.execute(
            """INSERT OR IGNORE INTO user_ratings (username, trakt_id, tmdb_id, rating, rated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (username, trakt_id, tmdb_id, rating, rated_at),
        )

    # --- Stats ---
    def get_stats(self) -> dict:
        shows = self.conn.execute("SELECT COUNT(*) FROM shows").fetchone()[0]
        meta = self.conn.execute("SELECT COUNT(*) FROM show_metadata").fetchone()[0]
        users = self.conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        users_done = self.conn.execute(
            "SELECT COUNT(*) FROM users WHERE ratings_fetched = 1"
        ).fetchone()[0]
        users_private = self.conn.execute(
            "SELECT COUNT(*) FROM users WHERE is_private = 1"
        ).fetchone()[0]
        ratings = self.conn.execute("SELECT COUNT(*) FROM user_ratings").fetchone()[0]
        return {
            "shows_discovered": shows,
            "shows_enriched": meta,
            "users_discovered": users,
            "users_processed": users_done,
            "users_private": users_private,
            "total_ratings": ratings,
        }

    def commit(self):
        self.conn.commit()


# ===========================================================================
# Helper: extract show IDs from API response
# ===========================================================================
def _extract_show(item: dict, source: str) -> dict | None:
    """Extract show info from various API response formats."""
    # trending/watched/played wrap the show in a "show" key
    show = item.get("show", item)
    ids = show.get("ids", {})
    trakt_id = ids.get("trakt")
    if not trakt_id:
        return None
    return {
        "trakt_id": trakt_id,
        "slug": ids.get("slug", ""),
        "tmdb_id": ids.get("tmdb"),
        "imdb_id": ids.get("imdb"),
        "title": show.get("title", ""),
        "year": show.get("year"),
        "source": source,
    }


# ===========================================================================
# Collector
# ===========================================================================
class TraktCollector:
    """Multi-phase orchestrator for trakt.tv data collection."""

    # Show list endpoints to iterate
    SHOW_LISTS = ["popular", "watched", "played", "trending", "collected", "anticipated"]

    def __init__(self, db: CheckpointDB, api: TraktAPI,
                 max_shows: int = 5000, max_users: int = 80000,
                 comment_pages: int = 2, list_pages: int = 1,
                 network_min_ratings: int = 10):
        self.db = db
        self.api = api
        self.max_shows = max_shows
        self.max_users = max_users
        self.comment_pages = comment_pages
        self.list_pages = list_pages
        self.network_min_ratings = network_min_ratings

    def run(self, phase: str = "all"):
        """Run specified phase or all phases sequentially with resume."""
        phases = {
            "discover": self.phase_discover_shows,
            "enrich": self.phase_enrich_shows,
            "network": self.phase_network_crawl,
            "users": self.phase_collect_user_ratings,
            "export": self.phase_export,
        }
        if phase == "all":
            for name, func in phases.items():
                status, _ = self.db.get_phase(name)
                if status == "done":
                    logger.info(f"Phase '{name}' already done — skipping")
                    continue
                func()
        elif phase in phases:
            phases[phase]()
        else:
            logger.error(f"Unknown phase: {phase}")

        stats = self.db.get_stats()
        logger.info(f"Final stats: {json.dumps(stats, indent=2)}")
        logger.info(f"Total API requests made this session: {self.api.total_requests}")

    # ------------------------------------------------------------------
    # Phase 1: Discover Shows
    # ------------------------------------------------------------------
    def phase_discover_shows(self):
        logger.info("=" * 60)
        logger.info("PHASE 1: Discover Shows")
        logger.info("=" * 60)

        status, cursor_json = self.db.get_phase("discover")
        if status == "done":
            return

        # Parse resume cursor
        cursor = json.loads(cursor_json) if cursor_json else {"list_idx": 0, "page": 1}
        start_list_idx = cursor["list_idx"]
        start_page = cursor["page"]

        for list_idx in range(start_list_idx, len(self.SHOW_LISTS)):
            list_name = self.SHOW_LISTS[list_idx]
            page = start_page if list_idx == start_list_idx else 1

            while True:
                if self.db.get_show_count() >= self.max_shows:
                    logger.info(f"Reached max_shows={self.max_shows}")
                    break

                logger.info(f"  Fetching /shows/{list_name} page {page}...")
                data = self.api.get_show_list(list_name, page=page)
                if not data:
                    break

                count = 0
                for item in data:
                    show = _extract_show(item, list_name)
                    if show:
                        self.db.upsert_show(**show)
                        count += 1

                self.db.set_phase("discover", "in_progress",
                                  json.dumps({"list_idx": list_idx, "page": page + 1}))
                self.db.commit()
                logger.info(f"    page {page}: {count} shows (total: {self.db.get_show_count()})")

                if len(data) < 100:
                    break
                page += 1

            if self.db.get_show_count() >= self.max_shows:
                break

        # Discover related shows for top shows (by votes)
        logger.info("  Fetching related shows for top entries...")
        top_shows = self.db.conn.execute(
            "SELECT slug FROM shows ORDER BY trakt_id LIMIT 200"
        ).fetchall()
        for (slug,) in top_shows:
            if self.db.get_show_count() >= self.max_shows:
                break
            related = self.api.get_show_related(slug)
            if related:
                for item in related:
                    show = _extract_show(item, "related")
                    if show:
                        self.db.upsert_show(**show)
        self.db.commit()

        self.db.set_phase("discover", "done")
        logger.info(f"Phase 1 done: {self.db.get_show_count()} shows discovered")

    # ------------------------------------------------------------------
    # Phase 2: Enrich Shows (metadata + stats + discover users)
    # ------------------------------------------------------------------
    def phase_enrich_shows(self):
        logger.info("=" * 60)
        logger.info("PHASE 2: Enrich Shows + Discover Users")
        logger.info("=" * 60)

        status, _ = self.db.get_phase("enrich")
        if status == "done":
            return

        self.db.set_phase("enrich", "in_progress")
        pending = self.db.get_shows_needing_metadata()
        total = len(pending)
        logger.info(f"Shows to enrich: {total}")

        for idx, (trakt_id, slug) in enumerate(pending):
            # 1. Full metadata
            show_data = self.api.get_show_full(slug)
            if not show_data:
                self.db.mark_show_metadata_done(trakt_id)
                if (idx + 1) % BATCH_COMMIT_SIZE == 0:
                    self.db.commit()
                continue

            ids = show_data.get("ids", {})
            genres = json.dumps(show_data.get("genres", []))

            # 2. Stats
            stats = self.api.get_show_stats(slug)
            watchers = stats.get("watchers", 0) if stats else 0
            plays = stats.get("plays", 0) if stats else 0

            self.db.upsert_show_metadata(
                trakt_id=trakt_id,
                tmdb_id=ids.get("tmdb"),
                title=show_data.get("title", ""),
                year=show_data.get("year"),
                overview=show_data.get("overview"),
                genres=genres,
                network=show_data.get("network"),
                status=show_data.get("status"),
                rating=show_data.get("rating"),
                votes=show_data.get("votes"),
                watchers=watchers,
                plays=plays,
                language=show_data.get("language"),
                country=show_data.get("country"),
                runtime=show_data.get("runtime"),
            )

            # 3. Discover users from comments
            for cp in range(1, self.comment_pages + 1):
                comments = self.api.get_show_comments(slug, page=cp)
                if not comments:
                    break
                for comment in comments:
                    user = comment.get("user", {})
                    username = user.get("username")
                    if username:
                        self.db.upsert_user(username, f"comment:{slug}")
                if len(comments) < 100:
                    break

            # 4. Discover users from personal lists
            for lp in range(1, self.list_pages + 1):
                lists = self.api.get_show_lists(slug, page=lp)
                if not lists:
                    break
                for lst in lists:
                    user = lst.get("user", {})
                    username = user.get("username")
                    if username:
                        self.db.upsert_user(username, f"list:{slug}")
                if len(lists) < 100:
                    break

            self.db.mark_show_metadata_done(trakt_id)

            if (idx + 1) % BATCH_COMMIT_SIZE == 0:
                self.db.commit()
                stats_db = self.db.get_stats()
                logger.info(
                    f"  Progress: {idx + 1}/{total} shows | "
                    f"{stats_db['users_discovered']} users | "
                    f"API calls: {self.api.total_requests}"
                )

        self.db.commit()
        self.db.set_phase("enrich", "done")
        stats = self.db.get_stats()
        logger.info(f"Phase 2 done: {stats['shows_enriched']} enriched, {stats['users_discovered']} users found")

    # ------------------------------------------------------------------
    # Phase 2.5: Network Crawl (Followers/Following)
    # ------------------------------------------------------------------
    def phase_network_crawl(self):
        """Discover new users by crawling followers/following of active raters."""
        logger.info("=" * 60)
        logger.info("PHASE 2.5: Network Crawl (Followers/Following)")
        logger.info("=" * 60)

        status, _ = self.db.get_phase("network")
        if status == "done":
            return

        self.db.set_phase("network", "in_progress")

        seeds = self.db.get_users_for_network_crawl(
            min_ratings=self.network_min_ratings,
        )
        # Shuffle seeds for genre/taste diversity
        random.shuffle(seeds)
        total = len(seeds)
        logger.info(f"Seed users for network crawl: {total} "
                    f"(min_ratings={self.network_min_ratings})")

        new_users_before = self.db.get_user_count()

        for idx, (username,) in enumerate(seeds):
            if self.db.get_user_count() >= self.max_users:
                logger.info(f"Reached max_users={self.max_users} — stopping network crawl")
                break

            # Fetch followers
            followers = self.api.get_user_followers(username)
            if followers and isinstance(followers, list):
                for item in followers:
                    if not isinstance(item, dict):
                        continue
                    user = item.get("user", {})
                    uname = user.get("username") if isinstance(user, dict) else None
                    if uname:
                        self.db.upsert_user(uname, f"follower:{username}")

            # Fetch following
            following = self.api.get_user_following(username)
            if following and isinstance(following, list):
                for item in following:
                    if not isinstance(item, dict):
                        continue
                    user = item.get("user", {})
                    uname = user.get("username") if isinstance(user, dict) else None
                    if uname:
                        self.db.upsert_user(uname, f"following:{username}")

            self.db.mark_user_network_crawled(username)

            if (idx + 1) % BATCH_COMMIT_SIZE == 0:
                self.db.commit()
                current_users = self.db.get_user_count()
                logger.info(
                    f"  Progress: {idx + 1}/{total} seeds crawled | "
                    f"{current_users} total users (+{current_users - new_users_before} new) | "
                    f"API calls: {self.api.total_requests}"
                )

        self.db.commit()
        self.db.set_phase("network", "done")
        new_users_after = self.db.get_user_count()
        logger.info(
            f"Phase 2.5 done: {new_users_after - new_users_before} new users discovered "
            f"({new_users_after} total)"
        )

    # ------------------------------------------------------------------
    # Phase 3: Collect User Ratings
    # ------------------------------------------------------------------
    def phase_collect_user_ratings(self):
        logger.info("=" * 60)
        logger.info("PHASE 3: Collect User Ratings")
        logger.info("=" * 60)

        status, _ = self.db.get_phase("users")
        if status == "done":
            return

        self.db.set_phase("users", "in_progress")
        pending = self.db.get_users_needing_ratings(limit=self.max_users)
        total = len(pending)
        logger.info(f"Users to process: {total}")

        processed = 0
        ratings_collected = 0

        for idx, (username,) in enumerate(pending):
            data = self.api.get_user_ratings(username)

            if data is None:
                self.db.mark_user_private(username)
                processed += 1
                continue

            # === НОВАЯ ЗАЩИТА ===
            if not isinstance(data, list):
                logger.warning(f"Unexpected response for user '{username}': {type(data)} — {str(data)[:200]}")
                self.db.mark_user_private(username)  # или просто пропускаем
                processed += 1
                continue

            for item in data:
                # Защита от строк и других странных типов
                if not isinstance(item, dict):
                    logger.debug(f"Skipping non-dict item for user {username}: {type(item)}")
                    continue

                show = item.get("show", {})
                if not isinstance(show, dict):
                    continue

                ids = show.get("ids", {})
                trakt_id = ids.get("trakt")
                if not trakt_id:
                    continue

                rating = item.get("rating")
                if not rating:
                    continue

                self.db.upsert_user_rating(
                    username=username,
                    trakt_id=trakt_id,
                    tmdb_id=ids.get("tmdb"),
                    rating=rating,
                    rated_at=item.get("rated_at"),
                )
                ratings_collected += 1

            self.db.mark_user_done(username)
            processed += 1

            if processed % BATCH_COMMIT_SIZE == 0:
                self.db.commit()
                stats = self.db.get_stats()
                logger.info(
                    f"  Progress: {processed}/{total} users | "
                    f"{stats['users_private']} private | "
                    f"{stats['total_ratings']} ratings | "
                    f"API calls: {self.api.total_requests}"
                )

        self.db.commit()
        self.db.set_phase("users", "done")
        stats = self.db.get_stats()
        logger.info(
            f"Phase 3 done: {stats['users_processed']} users processed, "
            f"{stats['users_private']} private, {stats['total_ratings']} ratings"
        )

    # ------------------------------------------------------------------
    # Phase 4: Export to CSV
    # ------------------------------------------------------------------
    def phase_export(self):
        logger.info("=" * 60)
        logger.info("PHASE 4: Export to CSV")
        logger.info("=" * 60)

        raw_dir = RAW_DIR
        raw_dir.mkdir(parents=True, exist_ok=True)

        # --- Export show metadata ---
        shows_path = raw_dir / "trakt_shows.csv"
        rows = self.db.conn.execute(
            """SELECT tmdb_id, title, year, overview, genres, network, status,
                      rating, votes, watchers, language, country, runtime
               FROM show_metadata
               WHERE tmdb_id IS NOT NULL"""
        ).fetchall()

        with open(shows_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "tmdb_id", "title", "year", "overview", "genres", "network",
                "status", "vote_average", "vote_count", "popularity",
                "language", "country", "runtime",
            ])
            for row in rows:
                # genres is JSON string — pass through
                # rating is 0-10, keep as-is for metadata (vote_average)
                # watchers = popularity proxy
                writer.writerow(row)

        logger.info(f"Exported {len(rows)} shows to {shows_path}")

        # --- Export user interactions ---
        interactions_path = raw_dir / "trakt_interactions.csv"

        # Build username -> numeric user_id mapping
        usernames = self.db.conn.execute(
            "SELECT DISTINCT username FROM user_ratings ORDER BY username"
        ).fetchall()
        user_map = {name: uid for uid, (name,) in enumerate(usernames)}

        ratings = self.db.conn.execute(
            """SELECT ur.username, ur.tmdb_id, ur.rating, ur.rated_at
               FROM user_ratings ur
               WHERE ur.tmdb_id IS NOT NULL AND ur.rating IS NOT NULL"""
        ).fetchall()

        with open(interactions_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "tmdb_id", "rating", "timestamp"])
            for username, tmdb_id, rating, rated_at in ratings:
                user_id = user_map.get(username)
                if user_id is None:
                    continue
                # Convert rating: trakt 1-10 -> 0.5-5.0 (MovieLens scale)
                ml_rating = round(rating / 2.0, 1)
                # Convert ISO timestamp to epoch seconds
                ts = 0
                if rated_at:
                    try:
                        dt = datetime.fromisoformat(rated_at.replace("Z", "+00:00"))
                        ts = int(dt.timestamp())
                    except (ValueError, OSError):
                        pass
                writer.writerow([user_id, tmdb_id, ml_rating, ts])

        logger.info(f"Exported {len(ratings)} interactions ({len(user_map)} users) to {interactions_path}")

        # --- Summary ---
        # Filter: users with < 3 ratings
        user_rating_counts = self.db.conn.execute(
            "SELECT username, COUNT(*) as cnt FROM user_ratings GROUP BY username HAVING cnt >= 3"
        ).fetchall()
        logger.info(f"Users with >= 3 ratings: {len(user_rating_counts)}")

        self.db.set_phase("export", "done")
        logger.info("Phase 4 done: export complete")


# ===========================================================================
# CLI Entry Point
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Collect TV show data from trakt.tv")
    parser.add_argument(
        "--phase",
        choices=["all", "discover", "enrich", "network", "users", "export"],
        default="all",
        help="Run a specific phase or all (default: all)",
    )
    parser.add_argument("--db", default=None, help="Path to checkpoint database")
    parser.add_argument("--max-shows", type=int, default=5000, help="Max shows to discover")
    parser.add_argument("--max-users", type=int, default=80000, help="Max users to collect")
    parser.add_argument("--comment-pages", type=int, default=2, help="Comment pages per show")
    parser.add_argument("--list-pages", type=int, default=1, help="List pages per show")
    parser.add_argument("--network-min-ratings", type=int, default=10,
                        help="Min ratings for a user to be used as network crawl seed")
    parser.add_argument("--reset-phase", type=str, default=None,
                        help="Reset a phase to 'pending' to allow re-run (e.g. --reset-phase users)")
    args = parser.parse_args()

    # Load .env
    load_dotenv(ENV_FILE)

    client_id = os.getenv("TRAKT_CLIENT_ID")
    if not client_id:
        logger.error("TRAKT_CLIENT_ID not set in .env — aborting")
        sys.exit(1)

    # DB path
    if args.db:
        db_path = Path(args.db)
    else:
        db_path = RAW_DIR / "trakt_collector.db"

    logger.info(f"Raw data dir:  {RAW_DIR}")
    logger.info(f"Checkpoint DB: {db_path}")
    logger.info(f"Settings: max_shows={args.max_shows}, max_users={args.max_users}, "
                f"comment_pages={args.comment_pages}, list_pages={args.list_pages}, "
                f"network_min_ratings={args.network_min_ratings}")

    api = TraktAPI(client_id)
    db = CheckpointDB(db_path)

    # Reset a phase if requested (e.g. --reset-phase users)
    if args.reset_phase:
        db.set_phase(args.reset_phase, "pending")
        logger.info(f"Reset phase '{args.reset_phase}' to pending")

    try:
        collector = TraktCollector(
            db=db,
            api=api,
            max_shows=args.max_shows,
            max_users=args.max_users,
            comment_pages=args.comment_pages,
            list_pages=args.list_pages,
            network_min_ratings=args.network_min_ratings,
        )
        collector.run(phase=args.phase)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user — progress saved. Resume by running again.")
    finally:
        db.commit()
        db.close()


if __name__ == "__main__":
    main()
