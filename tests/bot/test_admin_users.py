"""
Read-only admin views on SessionStore.

These methods power `scripts/admin_users.py`. They never mutate state, so
the tests exercise both happy paths and edge cases (empty DB, unknown
user_id, ties on count/timestamp). Direct INSERTs are used where we need
to control `added_at` precisely — `SessionStore.add` resolves to
CURRENT_TIMESTAMP with 1-second granularity, which would make ordering
assertions flaky.
"""

from __future__ import annotations

import pytest

from recommendation_system.models.gnn.bot.session_store import SessionStore


@pytest.fixture
def store(tmp_path) -> SessionStore:
    return SessionStore(tmp_path / "sessions.db")


def _insert_like(store: SessionStore, user_id: int, tmdb_id: int, added_at: str) -> None:
    """Direct INSERT to pin added_at — `add()` only writes CURRENT_TIMESTAMP."""
    store.conn.execute(
        "INSERT INTO user_likes (user_id, tmdb_id, added_at) VALUES (?, ?, ?)",
        (user_id, tmdb_id, added_at),
    )
    store.conn.commit()


# ---------------------------------------------------------------------------
# get_all_users
# ---------------------------------------------------------------------------


def test_get_all_users_empty_db(store):
    assert store.get_all_users() == []


def test_get_all_users_aggregates_likes(store):
    # User 1: ru, 3 likes; user 2: uk, 1 like; user 3: en, 0 likes.
    store.set_lang(1, "ru")
    store.set_lang(2, "uk")
    store.set_lang(3, "en")
    _insert_like(store, 1, 100, "2026-05-01 10:00:00")
    _insert_like(store, 1, 200, "2026-05-02 10:00:00")
    _insert_like(store, 1, 300, "2026-05-03 10:00:00")
    _insert_like(store, 2, 200, "2026-05-01 09:00:00")

    rows = store.get_all_users()
    assert rows == [
        (1, "ru", 3, "2026-05-03 10:00:00"),
        (2, "uk", 1, "2026-05-01 09:00:00"),
        (3, "en", 0, None),
    ]


def test_get_all_users_includes_zero_like_users(store):
    """A user who only ran /lang (no likes yet) still shows up — spec §3."""
    store.set_lang(42, "uk")
    rows = store.get_all_users()
    assert rows == [(42, "uk", 0, None)]


def test_get_all_users_skips_likes_only_user(store):
    """Driven by user_prefs (LEFT JOIN), so a user that managed to like
    something without any user_prefs row is invisible. In practice the
    bot writes a prefs row on first interaction, so this can't happen —
    but pinning the contract here keeps future schema changes honest."""
    _insert_like(store, 999, 100, "2026-05-01 10:00:00")
    assert store.get_all_users() == []


# ---------------------------------------------------------------------------
# get_user_likes
# ---------------------------------------------------------------------------


def test_get_user_likes_empty_for_unknown_user(store):
    assert store.get_user_likes(user_id=12345) == []


def test_get_user_likes_sorted_desc_by_added_at(store):
    _insert_like(store, 1, 100, "2026-05-01 10:00:00")
    _insert_like(store, 1, 200, "2026-05-03 10:00:00")
    _insert_like(store, 1, 300, "2026-05-02 10:00:00")

    assert store.get_user_likes(1) == [
        (200, "2026-05-03 10:00:00"),
        (300, "2026-05-02 10:00:00"),
        (100, "2026-05-01 10:00:00"),
    ]


def test_get_user_likes_breaks_ties_by_tmdb_id_desc(store):
    """Same added_at → larger tmdb_id first. Deterministic order is what
    matters; any stable rule will do, this one matches the SQL."""
    ts = "2026-05-01 10:00:00"
    _insert_like(store, 1, 100, ts)
    _insert_like(store, 1, 300, ts)
    _insert_like(store, 1, 200, ts)
    assert store.get_user_likes(1) == [(300, ts), (200, ts), (100, ts)]


def test_get_user_likes_isolates_users(store):
    _insert_like(store, 1, 100, "2026-05-01 10:00:00")
    _insert_like(store, 2, 200, "2026-05-01 10:00:00")
    assert store.get_user_likes(1) == [(100, "2026-05-01 10:00:00")]
    assert store.get_user_likes(2) == [(200, "2026-05-01 10:00:00")]


# ---------------------------------------------------------------------------
# get_top_liked
# ---------------------------------------------------------------------------


def test_get_top_liked_empty(store):
    assert store.get_top_liked(n=10) == []


def test_get_top_liked_orders_by_count_desc(store):
    # tmdb 200: 3 likes; 100: 2 likes; 300: 1 like.
    _insert_like(store, 1, 100, "2026-05-01 10:00:00")
    _insert_like(store, 2, 100, "2026-05-01 10:00:00")
    _insert_like(store, 1, 200, "2026-05-01 10:00:00")
    _insert_like(store, 2, 200, "2026-05-01 10:00:00")
    _insert_like(store, 3, 200, "2026-05-01 10:00:00")
    _insert_like(store, 1, 300, "2026-05-01 10:00:00")

    assert store.get_top_liked(n=3) == [(200, 3), (100, 2), (300, 1)]


def test_get_top_liked_breaks_count_ties_by_tmdb_id_asc(store):
    """tmdb 100 and 300 both have 1 like → smaller id first (deterministic)."""
    _insert_like(store, 1, 300, "2026-05-01 10:00:00")
    _insert_like(store, 2, 100, "2026-05-01 10:00:00")
    _insert_like(store, 3, 200, "2026-05-01 10:00:00")
    assert store.get_top_liked(n=3) == [(100, 1), (200, 1), (300, 1)]


def test_get_top_liked_respects_limit(store):
    for uid, tid in [(1, 100), (2, 100), (1, 200), (1, 300)]:
        _insert_like(store, uid, tid, "2026-05-01 10:00:00")
    top = store.get_top_liked(n=2)
    assert len(top) == 2
    assert top[0] == (100, 2)


def test_get_top_liked_n_larger_than_distinct_count(store):
    """Ask for 100, only 2 distinct tmdb_ids exist → return both, no error."""
    _insert_like(store, 1, 100, "2026-05-01 10:00:00")
    _insert_like(store, 2, 200, "2026-05-01 10:00:00")
    assert store.get_top_liked(n=100) == [(100, 1), (200, 1)]
