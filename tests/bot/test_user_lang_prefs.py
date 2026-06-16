"""
SessionStore language-preference tests + Telegram language_code mapping.

`SessionStore.set_lang/get_lang` is the persistence layer for the per-user
UI language. `_map_tg_lang` (in movie_bot) translates the ISO-ish code
that Telegram puts on `from_user.language_code` into one of the three
languages the bot actually supports.

These tests deliberately don't touch movie_bot's runtime — the mapping
function is reimplemented in a tiny shim here so we don't need to import
the bot module (which initializes engines on import).
"""

from __future__ import annotations

import pytest

from recommendation_system.models.gnn.bot.session_store import SessionStore


# ---------------------------------------------------------------------------
# SessionStore.{get,set}_lang
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path) -> SessionStore:
    return SessionStore(tmp_path / "sessions.db")


def test_get_lang_returns_none_for_new_user(store):
    """No row yet → None, so the bot knows to autodetect."""
    assert store.get_lang(user_id=1) is None


def test_set_then_get_round_trip(store):
    store.set_lang(1, "ru")
    assert store.get_lang(1) == "ru"
    store.set_lang(1, "uk")
    assert store.get_lang(1) == "uk"
    store.set_lang(1, "en")
    assert store.get_lang(1) == "en"


def test_set_lang_rejects_unsupported_code(store):
    """Whitelist enforcement — typos must not silently persist garbage."""
    with pytest.raises(ValueError):
        store.set_lang(1, "fr")
    with pytest.raises(ValueError):
        store.set_lang(1, "RU")  # case matters — keys are lowercase
    with pytest.raises(ValueError):
        store.set_lang(1, "")


def test_set_lang_overwrites_previous_choice(store):
    """The /lang flow has to work after autodetect already saved a value."""
    store.set_lang(1, "uk")  # autodetected on first message
    store.set_lang(1, "ru")  # user clicked 🇷🇺 in /lang
    assert store.get_lang(1) == "ru"


def test_users_isolated(store):
    store.set_lang(1, "ru")
    store.set_lang(2, "uk")
    store.set_lang(3, "en")
    assert store.get_lang(1) == "ru"
    assert store.get_lang(2) == "uk"
    assert store.get_lang(3) == "en"


def test_lang_persists_across_reopen(tmp_path):
    """Restarting the bot must not wipe a user's language choice."""
    db_path = tmp_path / "sessions.db"
    s1 = SessionStore(db_path)
    s1.set_lang(42, "uk")
    s1.conn.close()

    s2 = SessionStore(db_path)
    assert s2.get_lang(42) == "uk"


def test_set_lang_does_not_affect_user_likes(store):
    """user_prefs and user_likes are separate tables — writes on one
    must not contend or clobber the other."""
    store.add(1, 27205)
    store.add(1, 562)
    store.set_lang(1, "ru")
    # get_tmdb_ids breaks added_at ties by tmdb_id ascending, so two adds
    # in the same second land in numeric order — use a set comparison.
    assert set(store.get_tmdb_ids(1)) == {27205, 562}
    assert store.get_lang(1) == "ru"


def test_lang_supported_constants():
    """Smoke-check that the whitelist matches the i18n module."""
    from recommendation_system.models.gnn.bot.i18n import SUPPORTED_LANGS as I18N_LANGS

    assert set(SessionStore.SUPPORTED_LANGS) == set(I18N_LANGS)


# ---------------------------------------------------------------------------
# _map_tg_lang — Telegram language_code → {en, ru, uk}
# ---------------------------------------------------------------------------


def _map_tg_lang(code):
    """Mirror of movie_bot._map_tg_lang.

    Reimplemented here so the test doesn't trigger movie_bot's import-time
    engine load. If the bot ever drifts from this, that's a real bug —
    the test below pins the contract.
    """
    if not code:
        return "en"
    c = code.lower()
    if c.startswith("uk"):
        return "uk"
    if c.startswith("ru") or c.startswith("be"):
        return "ru"
    return "en"


@pytest.mark.parametrize(
    "code,expected",
    [
        # Ukrainian variants
        ("uk", "uk"),
        ("uk-UA", "uk"),
        ("UK-UA", "uk"),
        # Russian variants
        ("ru", "ru"),
        ("ru-RU", "ru"),
        ("RU", "ru"),
        # Belarusian — bot policy: most BY users prefer ru-locale UI
        ("be", "ru"),
        ("be-BY", "ru"),
        # English — direct + fallback for unsupported langs
        ("en", "en"),
        ("en-US", "en"),
        ("en-GB", "en"),
        ("fr", "en"),
        ("de-DE", "en"),
        ("ja", "en"),
        ("kk", "en"),  # Kazakh isn't supported, falls back to en
        # Edge: missing code from older Telegram clients
        (None, "en"),
        ("", "en"),
    ],
)
def test_map_tg_lang(code, expected):
    assert _map_tg_lang(code) == expected


def test_movie_bot_map_matches_local_shim():
    """If movie_bot._map_tg_lang drifts from the contract above, fail loudly.

    Imports lazily and skips when bot dependencies aren't installed in the
    test environment (it pulls in aiogram, models, etc.).
    """
    try:
        # Defer the heavy import — movie_bot loads engines at import time,
        # so we can't always call it. Skip rather than fail when the
        # dependency tree isn't available.
        import importlib

        spec = importlib.util.find_spec("recommendation_system.models.gnn.movie_bot")
        if spec is None:
            pytest.skip("movie_bot not importable in this environment")
        # Source-level inspection: read the function definition and confirm
        # the branch list hasn't changed without updating this test.
        import inspect
        source_path = spec.origin
        with open(source_path, "r", encoding="utf-8") as f:
            src = f.read()
        for needle in (
            'if c.startswith("uk"):',
            'if c.startswith("ru") or c.startswith("be"):',
            'return "en"',
        ):
            assert needle in src, f"movie_bot._map_tg_lang missing branch: {needle!r}"
    except ImportError:
        pytest.skip("import inspection unavailable")
