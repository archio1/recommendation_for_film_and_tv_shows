"""
SQLite-backed persistence for per-user liked lists.

Replaces the in-memory `user_sessions` dict so that restarting the bot
doesn't wipe every user's list. Schema is minimal on purpose — the only
state we track is (user_id, tmdb_id, added_at). TV tmdb_ids are stored
with the +TV_OFFSET offset already applied, matching what we carry in
the rest of the pipeline.

Single shared connection with `check_same_thread=False` — handlers can
run on aiogram's default thread, and short SQLite writes don't need to
hop to a worker. Writes are serialized by SQLite's own locking.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


class SessionStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_db()

    SUPPORTED_LANGS: tuple[str, ...] = ("en", "ru", "uk")

    def _init_db(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_likes (
                user_id INTEGER NOT NULL,
                tmdb_id INTEGER NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, tmdb_id)
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_likes_user "
            "ON user_likes(user_id)"
        )
        # Per-user UI-language preference. Stored separately from the
        # likes table so writes don't contend on the same rows. Default
        # 'en' is the fallback for users we've never seen — autodetect
        # logic in the bot replaces it on the first message.
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_prefs (
                user_id INTEGER PRIMARY KEY,
                lang TEXT NOT NULL DEFAULT 'en',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def add(self, user_id: int, tmdb_id: int) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO user_likes (user_id, tmdb_id) VALUES (?, ?)",
            (user_id, tmdb_id),
        )
        self.conn.commit()

    def remove(self, user_id: int, tmdb_id: int) -> None:
        self.conn.execute(
            "DELETE FROM user_likes WHERE user_id = ? AND tmdb_id = ?",
            (user_id, tmdb_id),
        )
        self.conn.commit()

    def clear(self, user_id: int) -> None:
        self.conn.execute(
            "DELETE FROM user_likes WHERE user_id = ?", (user_id,)
        )
        self.conn.commit()

    def get_tmdb_ids(self, user_id: int) -> list[int]:
        rows = self.conn.execute(
            "SELECT tmdb_id FROM user_likes WHERE user_id = ? "
            "ORDER BY added_at, tmdb_id",
            (user_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_lang(self, user_id: int) -> str | None:
        """Return stored language code or None if user has no row yet.

        Returning None (rather than the default 'en') lets callers
        distinguish "never seen this user" from "user explicitly chose
        en" — autodetect from Telegram language_code only fires for the
        former.
        """
        row = self.conn.execute(
            "SELECT lang FROM user_prefs WHERE user_id = ?", (user_id,)
        ).fetchone()
        return row[0] if row else None

    def set_lang(self, user_id: int, lang: str) -> None:
        if lang not in self.SUPPORTED_LANGS:
            raise ValueError(f"unsupported lang: {lang!r}")
        self.conn.execute(
            """
            INSERT INTO user_prefs (user_id, lang, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                lang = excluded.lang,
                updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, lang),
        )
        self.conn.commit()

    # -- read-only admin views ----------------------------------------------

    def get_all_users(self) -> list[tuple[int, str, int, str | None]]:
        """Return (user_id, lang, likes_count, last_added_at) per user.

        Driven by user_prefs (every user that ever interacted with the bot
        has a row there); user_likes is LEFT JOIN-ed so users with no
        likes still appear with likes_count=0 and last_added_at=None.
        """
        rows = self.conn.execute(
            """
            SELECT p.user_id, p.lang,
                   COUNT(l.tmdb_id) AS likes_count,
                   MAX(l.added_at)  AS last_added_at
            FROM user_prefs p
            LEFT JOIN user_likes l ON l.user_id = p.user_id
            GROUP BY p.user_id, p.lang
            ORDER BY p.user_id
            """
        ).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    def get_user_likes(self, user_id: int) -> list[tuple[int, str]]:
        """Return [(tmdb_id, added_at), ...] sorted by added_at DESC."""
        rows = self.conn.execute(
            "SELECT tmdb_id, added_at FROM user_likes "
            "WHERE user_id = ? ORDER BY added_at DESC, tmdb_id DESC",
            (user_id,),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def get_top_liked(self, n: int = 20) -> list[tuple[int, int]]:
        """Return [(tmdb_id, count), ...] for the N most-liked items."""
        rows = self.conn.execute(
            "SELECT tmdb_id, COUNT(*) AS cnt FROM user_likes "
            "GROUP BY tmdb_id ORDER BY cnt DESC, tmdb_id ASC LIMIT ?",
            (n,),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]
