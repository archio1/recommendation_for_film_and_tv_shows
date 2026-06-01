"""
movie_bot.py — Telegram bot front-end for the dual-domain recommender.

Two per-domain UniversalSearchEngine instances (movies and tv) sit behind
a single DualDomainEngine router. A ColdStartIngestor handles freshly-
released items that neither LightGCN knows about yet.

User session state is a flat list of tmdb_ids — movie ids raw, tv ids
with +TV_OFFSET baked in (matching the parquet convention). We never
store internal item_ids in the session, because the two domains have
disjoint item_id spaces.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from aiogram import Bot, Dispatcher, F, types
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
from dotenv import load_dotenv

from bilingual_utils import GENRE_EN_TO_RU, GENRE_EN_TO_UK
from bot.i18n import T
from bot.keyboards import (
    BTN_CLEAR_ALL,
    BTN_LIST_ALL,
    BTN_MOVIES_ALL,
    BTN_SEARCH_ALL,
    BTN_TRENDING_ALL,
    BTN_TV_ALL,
    main_reply_kb,
)
from bot.search import get_query, merged_search, store_query
from bot.session_store import SessionStore
from cold_start import ColdStartIngestor
from dual_domain_engine import DualDomainEngine
from faiss_bridge import TV_OFFSET, FaissCatalog
from inference_engine import InferenceEngine
from universal_search import (
    TrendingUpdater,
    UniversalMediaItem,
    UniversalSearchEngine,
)


# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[4]
API_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MOVIES_DIR = DATA_DIR / "movies"
TV_DIR = DATA_DIR / "tv"
CACHE_DIR = DATA_DIR / "cache"

FAISS_DIR = PROJECT_ROOT / "src" / "recommendation_system" / "faiss_index"
FAISS_INDEX = FAISS_DIR / "catalog.faiss"
FAISS_META = FAISS_DIR / "catalog_meta.json"

MOVIES_CHECKPOINT = PROJECT_ROOT / "models" / "movies" / "lightgcn_movies_best_v4.pt"
TV_CHECKPOINT = PROJECT_ROOT / "models" / "tv" / "lightgcn_tv_best_v4.pt"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

def _build_engine(
    dataset_dir: Path,
    checkpoint: Path,
    cache_dir: Path,
) -> UniversalSearchEngine:
    infer = InferenceEngine(dataset_dir, checkpoint, device="cpu")
    ok, msg = infer.load_resources()
    if not ok:
        raise RuntimeError(f"{dataset_dir.name}: {msg}")
    logger.info(f"{dataset_dir.name}: {msg}")

    metadata = pd.read_parquet(dataset_dir / "items_metadata_final.parquet")
    return UniversalSearchEngine(
        metadata=metadata,
        cache_dir=cache_dir,
        tmdb_api_key=TMDB_API_KEY,
        inference_engine=infer,
        embeddings_path=dataset_dir / "overview_embeddings.npy",
        model_num_items=infer.model.num_items,
    )


CACHE_DIR.mkdir(parents=True, exist_ok=True)
movies_engine = _build_engine(MOVIES_DIR, MOVIES_CHECKPOINT, CACHE_DIR)
# Movies LightGCN collapses toward globally-popular titles (top-IMDb to
# everyone); a popularity penalty restores genre diversity. TV stays at 0 —
# its small catalog shows no such collapse. See UniversalSearchEngine.
movies_engine.popularity_debias = 0.5
tv_engine = _build_engine(TV_DIR, TV_CHECKPOINT, CACHE_DIR)

if FAISS_INDEX.exists() and FAISS_META.exists():
    faiss_catalog = FaissCatalog.load(FAISS_INDEX, FAISS_META)
    logger.info(f"FAISS catalog loaded: {faiss_catalog.size:,} items")
else:
    raise RuntimeError(
        f"FAISS catalog not found at {FAISS_INDEX}. "
        "Run: python -m recommendation_system.models.gnn.compute_embeddings --to-faiss"
    )

cold_start: Optional[ColdStartIngestor] = None
if movies_engine.tmdb_client is not None:
    cold_start = ColdStartIngestor(
        tmdb_client=movies_engine.tmdb_client,
        faiss_catalog=faiss_catalog,
        movies_hot_cache=movies_engine.hot_cache,
        tv_hot_cache=tv_engine.hot_cache,
        faiss_index_path=FAISS_INDEX,
        faiss_meta_path=FAISS_META,
    )

router = DualDomainEngine(
    movies_engine=movies_engine,
    tv_engine=tv_engine,
    faiss_catalog=faiss_catalog,
    cold_start=cold_start,
)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()


# ---------------------------------------------------------------------------
# Session — SQLite-backed, survives bot restarts
# ---------------------------------------------------------------------------

SESSION_DB = CACHE_DIR / "user_sessions.db"
session_store = SessionStore(SESSION_DB)

LIST_PAGE_SIZE = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _map_tg_lang(code: Optional[str]) -> str:
    """Map a Telegram language_code (ISO-ish) to one of our 3 supported langs.

    'be' (Belarusian) → 'ru' since most Belarusian Telegram users keep a
    Russian-locale UI. Anything else outside ru/uk family falls back to
    English — better than guessing.
    """
    if not code:
        return "en"
    c = code.lower()
    if c.startswith("uk"):
        return "uk"
    if c.startswith("ru") or c.startswith("be"):
        return "ru"
    return "en"


def _resolve_lang(user) -> str:
    """Lookup stored lang or autodetect from Telegram language_code.

    `user` is `types.User` (message.from_user / callback.from_user).
    Stores the autodetected value so subsequent messages don't pay the
    Telegram-attribute lookup again.
    """
    stored = session_store.get_lang(user.id)
    if stored:
        return stored
    detected = _map_tg_lang(getattr(user, "language_code", None))
    session_store.set_lang(user.id, detected)
    return detected


def _engine_for(tmdb_id: int) -> UniversalSearchEngine:
    return tv_engine if tmdb_id >= TV_OFFSET else movies_engine


def _lookup_item(tmdb_id: int) -> Optional[UniversalMediaItem]:
    """parquet metadata → HotCache → None. The router hits TMDb live
    separately via cold_start when it actually needs the embedding."""
    engine = _engine_for(tmdb_id)
    row = engine.metadata[engine.metadata["tmdb_id"] == tmdb_id]
    if not row.empty:
        return engine._row_to_universal(row.iloc[0])
    return engine.hot_cache.get_by_tmdb_id(tmdb_id)


def _media_word(media_type: str, lang: str = "ru") -> str:
    return T("media_tv" if media_type == "tv" else "media_movie", lang)


_GENRE_TABLES: dict[str, dict[str, str]] = {
    "ru": GENRE_EN_TO_RU,
    "uk": GENRE_EN_TO_UK,
    "en": {},  # english is the source language — pass through as-is
}


def _format_genres(genres: list[str], lang: str = "ru", limit: int = 3) -> str:
    if not genres:
        return ""
    table = _GENRE_TABLES.get(lang, _GENRE_TABLES["ru"])
    return ", ".join(table.get(g, g) for g in genres[:limit])


def _pick_localized(item: UniversalMediaItem, lang: str) -> str:
    """Pick a title from the available localized fields.

    Fallback chains favour the closest related language so a missing uk
    title shows ru rather than skipping straight to en — uk and ru are
    close enough that a Russian title beats an English one for a uk user.
    """
    if lang == "uk":
        return item.title_uk or item.title_ru or item.title
    if lang == "ru":
        return item.title_ru or item.title_uk or item.title
    return item.title or item.title_ru or item.title_uk or ""


def _format_title(
    item: UniversalMediaItem, lang: str = "ru", *, link: bool = True
) -> str:
    """Title (+ year) for display.

    HTML mode wraps the visible label in an <a href> pointing at IMDb (or
    TMDB if imdb_id is missing — common for some TV items and older art-
    house). Telegram requires <b> outside <a>, so callers nest as
    `<b>{_format_title(...)}</b>`. Pass link=False from inline-button
    label callers, where HTML isn't rendered.
    """
    title = _pick_localized(item, lang)
    year = f" ({item.year})" if item.year else ""
    if not link:
        return f"{title}{year}"
    if item.imdb_id:
        url = f"https://www.imdb.com/title/{item.imdb_id}/"
    else:
        # item.tmdb_id carries +TV_OFFSET for tv rows — strip it for the
        # public TMDB URL, which uses the raw id.
        raw = (
            item.tmdb_id - TV_OFFSET
            if item.media_type == "tv" and item.tmdb_id >= TV_OFFSET
            else item.tmdb_id
        )
        url = f"https://www.themoviedb.org/{item.media_type}/{raw}"
    return f'<a href="{url}">{title}{year}</a>'


def _format_rec_line(idx: int, item: UniversalMediaItem, lang: str = "ru") -> str:
    title = _format_title(item, lang)
    media = _media_word(item.media_type, lang)
    genres = _format_genres(item.genres, lang)
    if genres:
        return f"{idx}. <b>{title}</b> · {media}\n   <i>{genres}</i>"
    return f"{idx}. <b>{title}</b> · {media}"


def _format_item_line(idx: int, item: UniversalMediaItem, lang: str = "ru") -> str:
    """Like _format_rec_line but without the media-word (used inside a
    type-grouped section where the header already conveys the type)."""
    title = _format_title(item, lang)
    genres = _format_genres(item.genres, lang)
    if genres:
        return f"{idx}. <b>{title}</b>\n   <i>{genres}</i>"
    return f"{idx}. <b>{title}</b>"


def _format_grouped(items: list[UniversalMediaItem], lang: str = "ru") -> str:
    """Split into 🎬 Movies / 📺 Shows sections with per-section numbering."""
    movies = [i for i in items if i.media_type == "movie"]
    tv = [i for i in items if i.media_type == "tv"]

    parts: list[str] = []
    if movies:
        parts.append(T("section_movies", lang))
        for idx, item in enumerate(movies, 1):
            parts.append(_format_item_line(idx, item, lang))
    if tv:
        if parts:
            parts.append("")
        parts.append(T("section_tv", lang))
        for idx, item in enumerate(tv, 1):
            parts.append(_format_item_line(idx, item, lang))
    return "\n".join(parts)


def _recs_keyboard(has_movies: bool, has_tv: bool, lang: str = "ru") -> types.InlineKeyboardMarkup:
    """Bottom keyboard under a recommendation message.

    Cross-domain buttons only appear when the user's liked list contains
    items of the opposite domain to the cross target — otherwise they
    produce empty results.
    """
    kb = InlineKeyboardBuilder()
    kb.row(
        types.InlineKeyboardButton(text=T("ikbtn_my_list", lang), callback_data="view_list"),
    )
    cross_buttons = []
    if has_movies:
        cross_buttons.append(
            types.InlineKeyboardButton(
                text=T("ikbtn_cross_tv", lang), callback_data="cross_tv"
            )
        )
    if has_tv:
        cross_buttons.append(
            types.InlineKeyboardButton(
                text=T("ikbtn_cross_movie", lang), callback_data="cross_movie"
            )
        )
    if cross_buttons:
        kb.row(*cross_buttons)
    return kb.as_markup()


def _list_keyboard(
    items: list[UniversalMediaItem],
    has_movies: bool,
    has_tv: bool,
    page: int,
    max_page: int,
    lang: str = "ru",
) -> types.InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    # Rec buttons at the top — the primary action on this screen.
    row = []
    if has_movies:
        row.append(
            types.InlineKeyboardButton(text=T("ikbtn_recs_movie", lang), callback_data="recs_movie")
        )
    if has_tv:
        row.append(
            types.InlineKeyboardButton(text=T("ikbtn_recs_tv", lang), callback_data="recs_tv")
        )
    if has_movies and has_tv:
        row.append(
            types.InlineKeyboardButton(text=T("ikbtn_recs_all", lang), callback_data="recs_all")
        )
    if row:
        kb.row(*row)

    # Pagination row (only if list spans more than one page).
    if max_page > 0:
        nav = []
        if page > 0:
            nav.append(
                types.InlineKeyboardButton(
                    text="◀", callback_data=f"list_page:{page - 1}"
                )
            )
        nav.append(
            types.InlineKeyboardButton(
                text=f"{page + 1}/{max_page + 1}", callback_data="list_noop"
            )
        )
        if page < max_page:
            nav.append(
                types.InlineKeyboardButton(
                    text="▶", callback_data=f"list_page:{page + 1}"
                )
            )
        kb.row(*nav)

    # One delete button per item on the current page. Page is baked into
    # callback_data so removal stays on the same page.
    rm_label = T("ikbtn_remove", lang)
    for item in items:
        title = _format_title(item, lang, link=False)[:28]
        kb.row(
            types.InlineKeyboardButton(
                text=f"{rm_label}: {title}",
                callback_data=f"rm_{item.tmdb_id}_{page}",
            )
        )

    if items:
        kb.row(
            types.InlineKeyboardButton(text=T("ikbtn_clear_list", lang), callback_data="clear")
        )
    return kb.as_markup()


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

@dp.message(Command("start"))
async def cmd_start(m: types.Message):
    # Persistent sessions — do NOT wipe the list on /start. /clear does that.
    lang = _resolve_lang(m.from_user)
    await m.answer(T("help_text", lang), reply_markup=main_reply_kb(lang))


@dp.message(Command("list"))
async def cmd_list(m: types.Message):
    await render_list(m.from_user.id, m)


@dp.message(Command("clear"))
async def cmd_clear(m: types.Message):
    lang = _resolve_lang(m.from_user)
    session_store.clear(m.from_user.id)
    await m.answer(T("list_cleared", lang), reply_markup=main_reply_kb(lang))


@dp.message(Command("movies"))
async def cmd_movies(m: types.Message):
    await _send_recs(m, m.from_user.id, "movie")


@dp.message(Command("tv"))
async def cmd_tv(m: types.Message):
    await _send_recs(m, m.from_user.id, "tv")


@dp.message(Command("trending"))
async def cmd_trending(m: types.Message):
    await _send_trending(m)


def _lang_keyboard() -> types.InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.row(
        types.InlineKeyboardButton(text="🇺🇦 Українська", callback_data="lang_uk"),
        types.InlineKeyboardButton(text="🇷🇺 Русский", callback_data="lang_ru"),
        types.InlineKeyboardButton(text="🇬🇧 English", callback_data="lang_en"),
    )
    return kb.as_markup()


@dp.message(Command("lang"))
async def cmd_lang(m: types.Message):
    lang = _resolve_lang(m.from_user)
    await m.answer(T("lang_prompt", lang), reply_markup=_lang_keyboard())


@dp.callback_query(F.data.startswith("lang_"))
async def cb_lang(c: types.CallbackQuery):
    new_lang = c.data.split("_", 1)[1]
    if new_lang not in ("en", "ru", "uk"):
        await c.answer()
        return
    session_store.set_lang(c.from_user.id, new_lang)
    # Confirmation rendered in the *new* language so the switch is
    # visible immediately. Refresh the persistent reply keyboard too —
    # otherwise the user keeps seeing the old language until /start.
    await c.message.edit_text(T(f"lang_confirm_{new_lang}", new_lang))
    await c.message.answer(
        T("help_text", new_lang).split("\n")[0],
        reply_markup=main_reply_kb(new_lang),
    )
    await c.answer()


async def _send_trending(target) -> None:
    lang = _resolve_lang(target.from_user)
    try:
        # 10🎬 + 10📺 fits the Telegram 4096-char message limit and gives
        # the user enough breadth without pagination. min_vote_count
        # default (100) is applied inside HotCache.get_trending.
        movies = movies_engine.hot_cache.get_trending(media_type="movie", limit=10)
        tv = movies_engine.hot_cache.get_trending(media_type="tv", limit=10)
    except Exception:
        logger.exception("Trending fetch failed")
        await target.answer(T("trending_failed", lang))
        return

    if not movies and not tv:
        await target.answer(T("trending_empty", lang))
        return

    text = T("header_trending", lang) + "\n\n" + _format_grouped(movies + tv, lang)
    await target.answer(text, parse_mode=ParseMode.HTML)


# ---- Reply-keyboard button handlers ---------------------------------------
# These match the exact label strings the reply keyboard sends. Registering
# them before the generic F.text search handler means taps on the persistent
# keyboard are routed to the right action instead of being treated as queries.


@dp.message(F.text.in_(BTN_SEARCH_ALL))
async def btn_search_prompt(m: types.Message):
    lang = _resolve_lang(m.from_user)
    await m.answer(T("search_prompt", lang))


@dp.message(F.text.in_(BTN_LIST_ALL))
async def btn_list(m: types.Message):
    await render_list(m.from_user.id, m)


@dp.message(F.text.in_(BTN_MOVIES_ALL))
async def btn_movies(m: types.Message):
    await _send_recs(m, m.from_user.id, "movie")


@dp.message(F.text.in_(BTN_TV_ALL))
async def btn_tv(m: types.Message):
    await _send_recs(m, m.from_user.id, "tv")


@dp.message(F.text.in_(BTN_TRENDING_ALL))
async def btn_trending(m: types.Message):
    await _send_trending(m)


@dp.message(F.text.in_(BTN_CLEAR_ALL))
async def btn_clear(m: types.Message):
    lang = _resolve_lang(m.from_user)
    session_store.clear(m.from_user.id)
    await m.answer(T("list_cleared", lang), reply_markup=main_reply_kb(lang))


def _search_keyboard(
    items: list[UniversalMediaItem],
    query_hash: str,
    media_filter: Optional[str],
    lang: str = "ru",
) -> types.InlineKeyboardMarkup:
    """Add buttons per item + filter row re-running the search for one domain."""
    kb = InlineKeyboardBuilder()
    for item in items:
        tid = item.tmdb_id
        if item.media_type == "tv" and tid < TV_OFFSET:
            tid += TV_OFFSET
        emoji = "📺" if item.media_type == "tv" else "🎬"
        label = f"{emoji} {_format_title(item, lang, link=False)}"
        kb.row(
            types.InlineKeyboardButton(
                text=label[:60], callback_data=f"add_{tid}"
            )
        )

    filter_row = []
    if media_filter != "movie":
        filter_row.append(
            types.InlineKeyboardButton(
                text=T("ikbtn_filter_only_movies", lang),
                callback_data=f"sf_movie:{query_hash}",
            )
        )
    if media_filter != "tv":
        filter_row.append(
            types.InlineKeyboardButton(
                text=T("ikbtn_filter_only_tv", lang),
                callback_data=f"sf_tv:{query_hash}",
            )
        )
    if media_filter is not None:
        filter_row.append(
            types.InlineKeyboardButton(
                text=T("ikbtn_filter_all", lang), callback_data=f"sf_all:{query_hash}"
            )
        )
    if filter_row:
        kb.row(*filter_row)
    return kb.as_markup()


async def _render_search(
    reply_target,
    query: str,
    media_filter: Optional[str],
) -> None:
    """Run `merged_search` with optional domain filter and post the reply.

    `reply_target` is a Message (fresh search) or CallbackQuery (filter tap).
    On callback we edit the existing message; on a fresh search we send new.
    """
    is_cb = isinstance(reply_target, types.CallbackQuery)
    lang = _resolve_lang(reply_target.from_user)

    results = await asyncio.to_thread(
        merged_search,
        movies_engine,
        tv_engine,
        query,
        limit=8,
        media_type=media_filter,
    )

    if not results:
        msg = T("search_no_results", lang)
        if is_cb:
            await reply_target.message.edit_text(msg)
            await reply_target.answer()
        else:
            await reply_target.answer(msg)
        return

    query_hash = store_query(query)

    if media_filter == "movie":
        header = T("header_search_movie", lang).format(q=query) + "\n\n"
    elif media_filter == "tv":
        header = T("header_search_tv", lang).format(q=query) + "\n\n"
    else:
        header = T("header_search", lang).format(q=query) + "\n\n"

    text = header + _format_grouped(results, lang)
    kb = _search_keyboard(results, query_hash, media_filter, lang)

    if is_cb:
        await reply_target.message.edit_text(
            text, reply_markup=kb, parse_mode=ParseMode.HTML
        )
        await reply_target.answer()
    else:
        await reply_target.answer(
            text, reply_markup=kb, parse_mode=ParseMode.HTML
        )


@dp.message(F.text)
async def handle_search(message: types.Message):
    query = message.text.strip()
    if len(query) < 2:
        return
    await _render_search(message, query, media_filter=None)


@dp.callback_query(F.data.startswith("sf_"))
async def cb_search_filter(c: types.CallbackQuery):
    # callback_data: "sf_<movie|tv|all>:<query_hash>"
    try:
        prefix, query_hash = c.data.split(":", 1)
        _, domain = prefix.split("_", 1)
    except ValueError:
        await c.answer()
        return

    query = get_query(query_hash)
    if query is None:
        lang = _resolve_lang(c.from_user)
        await c.answer(T("query_expired", lang), show_alert=True)
        return

    media_filter = None if domain == "all" else domain
    await _render_search(c, query, media_filter=media_filter)


@dp.callback_query(F.data.startswith("add_"))
async def cb_add(c: types.CallbackQuery):
    tid = int(c.data.split("_", 1)[1])
    lang = _resolve_lang(c.from_user)
    session_store.add(c.from_user.id, tid)

    count = len(session_store.get_tmdb_ids(c.from_user.id))
    kb = InlineKeyboardBuilder()
    kb.row(
        types.InlineKeyboardButton(text=T("ikbtn_my_list", lang), callback_data="view_list"),
        types.InlineKeyboardButton(text=T("ikbtn_recs", lang), callback_data="recs_all"),
    )
    await c.message.answer(
        T("added", lang).format(n=count), reply_markup=kb.as_markup()
    )
    await c.answer()


@dp.callback_query(F.data == "view_list")
async def cb_view_list(c: types.CallbackQuery):
    await render_list(c.from_user.id, c)
    await c.answer()


@dp.callback_query(F.data.startswith("rm_"))
async def cb_remove(c: types.CallbackQuery):
    # callback_data: "rm_<tid>_<page>" (legacy "rm_<tid>" falls back to page 0)
    lang = _resolve_lang(c.from_user)
    parts = c.data.split("_")
    tid = int(parts[1])
    page = int(parts[2]) if len(parts) > 2 else 0
    session_store.remove(c.from_user.id, tid)
    await render_list(c.from_user.id, c, page=page)
    await c.answer(T("removed", lang))


@dp.callback_query(F.data == "clear")
async def cb_clear(c: types.CallbackQuery):
    lang = _resolve_lang(c.from_user)
    session_store.clear(c.from_user.id)
    await c.message.edit_text(T("list_cleared", lang))
    await c.answer()


@dp.callback_query(F.data.startswith("list_page:"))
async def cb_list_page(c: types.CallbackQuery):
    try:
        page = int(c.data.split(":", 1)[1])
    except ValueError:
        await c.answer()
        return
    await render_list(c.from_user.id, c, page=page)
    await c.answer()


@dp.callback_query(F.data == "list_noop")
async def cb_list_noop(c: types.CallbackQuery):
    # Page indicator "N/M" — inert, but Telegram requires a callback answer.
    await c.answer()


async def render_list(uid: int, target, page: int = 0) -> None:
    """Render the user's liked list. `target` is a Message or CallbackQuery.

    Pagination: LIST_PAGE_SIZE per page. `has_movies`/`has_tv` reflect the
    full list (not the page) so rec buttons don't flicker across pages.
    """
    lang = _resolve_lang(target.from_user)
    tmdb_ids = session_store.get_tmdb_ids(uid)
    total = len(tmdb_ids)

    if not tmdb_ids:
        text = T("list_empty", lang)
        if isinstance(target, types.CallbackQuery):
            await target.message.edit_text(text)
        else:
            await target.answer(text)
        return

    max_page = (total - 1) // LIST_PAGE_SIZE
    page = max(0, min(page, max_page))
    start = page * LIST_PAGE_SIZE
    page_ids = tmdb_ids[start : start + LIST_PAGE_SIZE]

    items: list[UniversalMediaItem] = []
    for tid in page_ids:
        item = _lookup_item(tid)
        if item is not None:
            items.append(item)

    has_movies = any(t < TV_OFFSET for t in tmdb_ids)
    has_tv = any(t >= TV_OFFSET for t in tmdb_ids)

    if max_page > 0:
        lines = [
            T("header_list_paged", lang).format(n=total, p=page + 1, pp=max_page + 1) + "\n"
        ]
    else:
        lines = [T("header_list", lang).format(n=total) + "\n"]
    for offset, item in enumerate(items, 1):
        idx = start + offset
        lines.append(
            f"{idx}. {_format_title(item, lang)} · {_media_word(item.media_type, lang)}"
        )
    text = "\n".join(lines)
    kb = _list_keyboard(items, has_movies, has_tv, page, max_page, lang)

    if isinstance(target, types.CallbackQuery):
        await target.message.edit_text(
            text, reply_markup=kb, parse_mode=ParseMode.HTML
        )
    else:
        await target.answer(text, reply_markup=kb, parse_mode=ParseMode.HTML)


async def _send_recs(reply_target, uid: int, mode: str) -> None:
    """Fetch and render recs. `reply_target` is a Message or CallbackQuery.

    mode: "movie" | "tv" | "all". For "all" we group by media_type with
    section headers; single-domain modes stay flat.
    """
    lang = _resolve_lang(reply_target.from_user)
    tmdb_ids = session_store.get_tmdb_ids(uid)
    is_cb = isinstance(reply_target, types.CallbackQuery)

    if not tmdb_ids:
        if is_cb:
            await reply_target.answer(T("list_empty_short", lang), show_alert=True)
        else:
            await reply_target.answer(T("list_empty", lang))
        return

    reply_src = reply_target.message if is_cb else reply_target
    status = await reply_src.answer(T("loading_recs", lang))
    try:
        if mode == "movie":
            recs = await asyncio.to_thread(
                router.recs_movie, tmdb_ids, top_k=8
            )
            header = T("header_recs_movies", lang)
        elif mode == "tv":
            recs = await asyncio.to_thread(router.recs_tv, tmdb_ids, top_k=8)
            header = T("header_recs_tv", lang)
        else:
            recs = await asyncio.to_thread(router.recs_all, tmdb_ids, top_k=8)
            header = T("header_recs", lang)
    except Exception:
        logger.exception("Recommendation failure")
        await status.edit_text(T("rec_failed", lang))
        if is_cb:
            await reply_target.answer()
        return

    if not recs:
        if mode == "movie":
            msg = T("rec_empty_movies", lang)
        elif mode == "tv":
            msg = T("rec_empty_tv", lang)
        else:
            msg = T("rec_empty_all", lang)
        await status.edit_text(msg)
        if is_cb:
            await reply_target.answer()
        return

    has_movies = any(t < TV_OFFSET for t in tmdb_ids)
    has_tv = any(t >= TV_OFFSET for t in tmdb_ids)

    if mode == "all":
        text = f"<b>{header}</b>\n\n" + _format_grouped(recs, lang)
    else:
        lines = [f"<b>{header}</b>\n"]
        for idx, item in enumerate(recs, 1):
            lines.append(_format_rec_line(idx, item, lang))
        text = "\n".join(lines)

    await status.edit_text(
        text,
        reply_markup=_recs_keyboard(has_movies, has_tv, lang),
        parse_mode=ParseMode.HTML,
    )
    if is_cb:
        await reply_target.answer()


@dp.callback_query(F.data.in_({"recs_movie", "recs_tv", "recs_all"}))
async def cb_recs(c: types.CallbackQuery):
    mode = c.data.split("_", 1)[1]
    await _send_recs(c, c.from_user.id, mode)


@dp.callback_query(F.data.in_({"cross_movie", "cross_tv"}))
async def cb_cross(c: types.CallbackQuery):
    lang = _resolve_lang(c.from_user)
    target = c.data.split("_", 1)[1]  # "movie" or "tv"
    tmdb_ids = session_store.get_tmdb_ids(c.from_user.id)

    if not tmdb_ids:
        await c.answer(T("list_empty_short", lang), show_alert=True)
        return

    status = await c.message.answer(T("loading_cross", lang))
    try:
        recs = await asyncio.to_thread(
            router.recs_cross,
            liked_tmdb_ids=tmdb_ids,
            target_media_type=target,
            top_k=8,
        )
    except Exception:
        logger.exception("Cross recommendation failure")
        await status.edit_text(T("rec_failed", lang))
        await c.answer()
        return

    if not recs:
        await status.edit_text(T("cross_no_results", lang))
        await c.answer()
        return

    header = T("header_cross_tv" if target == "tv" else "header_cross_movie", lang)
    lines = [f"<b>{header}</b>\n"]
    for idx, item in enumerate(recs, 1):
        lines.append(_format_rec_line(idx, item, lang))

    has_movies = any(t < TV_OFFSET for t in tmdb_ids)
    has_tv = any(t >= TV_OFFSET for t in tmdb_ids)
    await status.edit_text(
        "\n".join(lines),
        reply_markup=_recs_keyboard(has_movies, has_tv, lang),
        parse_mode=ParseMode.HTML,
    )
    await c.answer()


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

async def main():
    print("--- ЗАПУСК БОТА ---")

    # 1. Запуск фоновой задачи трендов
    print("1. Запуск обновления трендов...")
    asyncio.create_task(TrendingUpdater(movies_engine).start())

    # 2. Установка команд (МЫ МОЖЕМ ЕЁ ПРОПУСТИТЬ, ЕСЛИ ОНА ТОРМОЗИТ)
    print("2. Попытка установить меню команд...")
    try:
        # Ставим короткий тайм-аут (5 секунд), чтобы не висеть вечно
        await asyncio.wait_for(dp.set_my_commands([
            types.BotCommand(command="start", description="Начать"),
            types.BotCommand(command="list", description="Мой список"),
            types.BotCommand(command="movies", description="Рекомендации фильмов"),
            types.BotCommand(command="tv", description="Рекомендации сериалов"),
            types.BotCommand(command="trending", description="Что сейчас смотрят"),
            types.BotCommand(command="clear", description="Очистить список"),
            types.BotCommand(command="lang", description="Сменить язык / Change language"),
        ]), timeout=5.0)
        print("✅ Меню команд установлено.")
    except Exception as e:
        print(f"⚠️ Не удалось установить меню команд (возможно, сеть): {e}")
        print("Пропускаем этот шаг и идем дальше...")

    # 3. Финальный лог
    logger.info("Bot ready, starting polling…")
    print("🚀 БОТ ЗАПУЩЕН И ГОТОВ К РАБОТЕ!")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
