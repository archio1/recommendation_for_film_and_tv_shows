"""
Read-only CLI viewer for `data/processed/cache/user_sessions.db`.

Three subcommands:
    --list           every user with (user_id, lang, likes, last_added_at)
    --user <id>      detailed likes for one user
    --top <N>        N most-liked tmdb_ids globally

Title resolution: loads `data/processed/{movies,tv}/items_metadata_final.parquet`
once and indexes by tmdb_id (TV rows already carry the +TV_OFFSET offset,
matching what `user_likes` stores). Items missing from parquet print as
`tmdb_id=N (no title)`.

Strictly read-only — no add/remove/clear methods are imported.

Usage (package must be installed: pip install -e .):
    python scripts/admin_users.py --list
    python scripts/admin_users.py --user 12345
    python scripts/admin_users.py --top 10
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import pandas as pd

from recommendation_system.models.gnn.bot.session_store import SessionStore
from recommendation_system.paths import CACHE_DIR, MOVIES_DIR, TV_DIR

SESSION_DB = CACHE_DIR / "user_sessions.db"
MOVIES_PARQUET = MOVIES_DIR / "items_metadata_final.parquet"
TV_PARQUET = TV_DIR / "items_metadata_final.parquet"


# ---------------------------------------------------------------------------
# title resolution
# ---------------------------------------------------------------------------


def _load_titles() -> dict[int, tuple[str, Optional[str], str, int]]:
    """tmdb_id -> (title, title_ru, media_type, year). Empty if parquets missing."""
    frames = []
    for path in (MOVIES_PARQUET, TV_PARQUET):
        if path.exists():
            df = pd.read_parquet(path, columns=["tmdb_id", "title", "title_ru", "type", "year"])
            frames.append(df)
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    out: dict[int, tuple[str, Optional[str], str, int]] = {}
    for tmdb_id, title, title_ru, mtype, year in df.itertuples(index=False, name=None):
        ru = title_ru if isinstance(title_ru, str) and title_ru else None
        out[int(tmdb_id)] = (str(title) if title else "", ru, str(mtype), int(year) if pd.notna(year) else 0)
    return out


def _fmt_title(tmdb_id: int, titles: dict[int, tuple[str, Optional[str], str, int]]) -> tuple[str, str]:
    """Return (title_str, media_type). title_str: 'Title (year) / Title-RU' or fallback."""
    rec = titles.get(int(tmdb_id))
    if rec is None:
        return (f"tmdb_id={tmdb_id} (no title)", "?")
    title, title_ru, mtype, year = rec
    base = f"{title} ({year})" if year else title
    if title_ru and title_ru != title:
        base = f"{base} / {title_ru}"
    return (base, mtype)


# ---------------------------------------------------------------------------
# subcommand handlers
# ---------------------------------------------------------------------------


def cmd_list(store: SessionStore) -> None:
    rows = store.get_all_users()
    if not rows:
        print("no users yet")
        return
    header = f"{'user_id':<12} {'lang':<4} {'likes':<6} last_added_at"
    print(header)
    print("-" * len(header))
    for user_id, lang, likes, last_added in rows:
        last_added_str = last_added if last_added else "-"
        print(f"{user_id:<12} {lang:<4} {likes:<6} {last_added_str}")


def cmd_user(store: SessionStore, user_id: int, titles: dict) -> None:
    lang = store.get_lang(user_id)
    likes = store.get_user_likes(user_id)
    if lang is None and not likes:
        print(f"unknown user_id: {user_id}")
        return
    print(f"user_id={user_id}, lang={lang or '-'}, likes={len(likes)}")
    if not likes:
        return
    header = f"  {'tmdb_id':<10} {'media':<6} {'added_at':<22} title"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for tmdb_id, added_at in likes:
        title_str, mtype = _fmt_title(tmdb_id, titles)
        print(f"  {tmdb_id:<10} {mtype:<6} {added_at:<22} {title_str}")


def cmd_top(store: SessionStore, n: int, titles: dict) -> None:
    rows = store.get_top_liked(n=n)
    if not rows:
        print("no likes yet")
        return
    header = f"{'tmdb_id':<10} {'count':<6} {'media':<6} title"
    print(header)
    print("-" * len(header))
    for tmdb_id, count in rows:
        title_str, mtype = _fmt_title(tmdb_id, titles)
        print(f"{tmdb_id:<10} {count:<6} {mtype:<6} {title_str}")


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="admin_users",
        description="Read-only viewer for user_sessions.db.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="list every user")
    group.add_argument("--user", type=int, metavar="ID", help="show one user's likes")
    group.add_argument("--top", type=int, metavar="N", help="top-N most-liked tmdb_ids")
    args = parser.parse_args(argv)

    if not SESSION_DB.exists():
        print(f"session DB not found: {SESSION_DB}")
        return 0

    store = SessionStore(SESSION_DB)

    if args.list:
        cmd_list(store)
    elif args.user is not None:
        cmd_user(store, args.user, _load_titles())
    elif args.top is not None:
        cmd_top(store, args.top, _load_titles())
    return 0


if __name__ == "__main__":
    sys.exit(main())
