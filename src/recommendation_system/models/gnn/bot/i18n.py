"""
UI-string registry and lookup for the Telegram bot.

`I18N` keys map to {en, ru, uk} variants. `T(key, lang)` looks up the
right variant with a uk → ru → en fallback chain (uk and ru are close
enough that ru is the better fallback when a uk string is missing).

Keep this module data-only — no aiogram imports — so it can be unit-
tested without touching the bot runtime.
"""

from __future__ import annotations

from typing import Dict

SUPPORTED_LANGS: tuple[str, ...] = ("en", "ru", "uk")

_FALLBACK: Dict[str, tuple[str, ...]] = {
    "uk": ("ru", "en"),
    "ru": ("uk", "en"),
    "en": ("ru", "uk"),
}


I18N: Dict[str, Dict[str, str]] = {
    # ---------------- Buttons (reply keyboard) ----------------
    "btn_search":   {"en": "🔍 Search",       "ru": "🔍 Поиск",      "uk": "🔍 Пошук"},
    "btn_list":     {"en": "📋 My list",      "ru": "📋 Мой список", "uk": "📋 Мій список"},
    "btn_movies":   {"en": "🎬 Movies",       "ru": "🎬 Фильмы",     "uk": "🎬 Фільми"},
    "btn_tv":       {"en": "📺 Shows",        "ru": "📺 Сериалы",    "uk": "📺 Серіали"},
    "btn_trending": {"en": "🔥 Trending",     "ru": "🔥 Тренды",     "uk": "🔥 Тренди"},
    "btn_clear":    {"en": "🗑 Clear",        "ru": "🗑 Очистить",   "uk": "🗑 Очистити"},

    # ---------------- Inline keyboard buttons ----------------
    "ikbtn_my_list":          {"en": "My list",            "ru": "Мой список",      "uk": "Мій список"},
    "ikbtn_recs":             {"en": "Recommendations",    "ru": "Рекомендации",    "uk": "Рекомендації"},
    "ikbtn_recs_movie":       {"en": "Movies",             "ru": "Фильмы",          "uk": "Фільми"},
    "ikbtn_recs_tv":          {"en": "Shows",              "ru": "Сериалы",         "uk": "Серіали"},
    "ikbtn_recs_all":         {"en": "Mixed",              "ru": "Всё вместе",      "uk": "Все разом"},
    "ikbtn_cross_tv":         {"en": "Similar shows",      "ru": "Похожие сериалы", "uk": "Схожі серіали"},
    "ikbtn_cross_movie":      {"en": "Similar movies",     "ru": "Похожие фильмы",  "uk": "Схожі фільми"},
    "ikbtn_remove":           {"en": "Remove",             "ru": "Убрать",          "uk": "Прибрати"},
    "ikbtn_clear_list":       {"en": "Clear list",         "ru": "Очистить список", "uk": "Очистити список"},
    "ikbtn_filter_only_movies": {"en": "🎬 Movies only",   "ru": "🎬 Только фильмы","uk": "🎬 Лише фільми"},
    "ikbtn_filter_only_tv":     {"en": "📺 Shows only",    "ru": "📺 Только сериалы","uk": "📺 Лише серіали"},
    "ikbtn_filter_all":         {"en": "🔁 All",           "ru": "🔁 Всё",          "uk": "🔁 Усе"},

    # ---------------- Section headers ----------------
    "section_movies": {"en": "🎬 <b>Movies:</b>", "ru": "🎬 <b>Фильмы:</b>", "uk": "🎬 <b>Фільми:</b>"},
    "section_tv":     {"en": "📺 <b>Shows:</b>",  "ru": "📺 <b>Сериалы:</b>", "uk": "📺 <b>Серіали:</b>"},

    # ---------------- Static command responses ----------------
    "help_text": {
        "en": (
            "Hi! Send me a movie or show title — I'll suggest similar.\n\n"
            "Commands:\n"
            "/list — your list\n"
            "/movies — movie recommendations\n"
            "/tv — show recommendations\n"
            "/trending — what's hot now\n"
            "/clear — clear list\n"
            "/lang — change language\n"
        ),
        "ru": (
            "Привет! Напиши название фильма или сериала — подберу похожее.\n\n"
            "Команды:\n"
            "/list — твой список\n"
            "/movies — рекомендации фильмов\n"
            "/tv — рекомендации сериалов\n"
            "/trending — что сейчас смотрят\n"
            "/clear — очистить список\n"
            "/lang — сменить язык\n"
        ),
        "uk": (
            "Привіт! Напиши назву фільму чи серіалу — підберу схоже.\n\n"
            "Команди:\n"
            "/list — твій список\n"
            "/movies — рекомендації фільмів\n"
            "/tv — рекомендації серіалів\n"
            "/trending — що зараз дивляться\n"
            "/clear — очистити список\n"
            "/lang — змінити мову\n"
        ),
    },

    "list_cleared":    {"en": "List cleared.", "ru": "Список очищен.", "uk": "Список очищено."},
    "loading_recs":    {"en": "Picking recommendations…", "ru": "Подбираю рекомендации…", "uk": "Підбираю рекомендації…"},
    "loading_cross":   {"en": "Looking for matches in another genre…", "ru": "Ищу похожее в другом жанре…", "uk": "Шукаю схоже в іншому жанрі…"},
    "search_prompt":   {"en": "Send a movie or show title — I'll find it.", "ru": "Напиши название фильма или сериала — я найду.", "uk": "Напиши назву фільму чи серіалу — я знайду."},
    "list_empty":      {"en": "List is empty. Send a movie or show title to add it.", "ru": "Список пуст. Напиши название фильма или сериала, чтобы добавить.", "uk": "Список порожній. Напиши назву фільму чи серіалу, щоб додати."},
    "list_empty_short":{"en": "List is empty.", "ru": "Список пуст.", "uk": "Список порожній."},
    "search_no_results": {"en": "Nothing found. Try another title.", "ru": "Ничего не нашёл. Попробуй другое название.", "uk": "Нічого не знайшов. Спробуй іншу назву."},
    "rec_failed":      {"en": "Couldn't pick anything. Try later.", "ru": "Не удалось подобрать. Попробуй позже.", "uk": "Не вдалося підібрати. Спробуй пізніше."},
    "rec_empty_movies":{"en": "No movies in your list — add some and I'll pick.", "ru": "В списке нет фильмов — добавь что-нибудь, и подберу.", "uk": "У списку немає фільмів — додай щось, і підберу."},
    "rec_empty_tv":    {"en": "No shows in your list — add some and I'll pick.", "ru": "В списке нет сериалов — добавь что-нибудь, и подберу.", "uk": "У списку немає серіалів — додай щось, і підберу."},
    "rec_empty_all":   {"en": "Nothing matched. Add a couple more movies or shows.", "ru": "Ничего не подошло. Добавь ещё пару фильмов или сериалов.", "uk": "Нічого не підійшло. Додай ще пару фільмів чи серіалів."},
    "cross_no_results":{"en": "Nothing fitting found.", "ru": "Ничего подходящего не нашёл.", "uk": "Нічого відповідного не знайшов."},
    "trending_failed": {"en": "Couldn't fetch trending. Try later.", "ru": "Не удалось получить тренды. Попробуй позже.", "uk": "Не вдалося отримати тренди. Спробуй пізніше."},
    "trending_empty":  {"en": "No trending yet — the cache is still warming up.", "ru": "Трендов пока нет — TrendingUpdater ещё не наполнил кэш.", "uk": "Трендів поки немає — кеш ще наповнюється."},
    "query_expired":   {"en": "Query expired — type the title again.", "ru": "Запрос уже истёк — введи название заново.", "uk": "Запит вже застарів — введи назву ще раз."},
    "added":           {"en": "Added. List size: {n}.", "ru": "Добавлено. В списке сейчас: {n}.", "uk": "Додано. У списку зараз: {n}."},
    "removed":         {"en": "Removed.", "ru": "Убрал.", "uk": "Прибрав."},

    # ---------------- Headers ----------------
    "header_recs":         {"en": "Recommendations:", "ru": "Рекомендации:", "uk": "Рекомендації:"},
    "header_recs_movies":  {"en": "Recommendations (movies):", "ru": "Рекомендации (фильмы):", "uk": "Рекомендації (фільми):"},
    "header_recs_tv":      {"en": "Recommendations (shows):", "ru": "Рекомендации (сериалы):", "uk": "Рекомендації (серіали):"},
    "header_cross_tv":     {"en": "Shows similar to your list:", "ru": "Похожие сериалы по твоему списку:", "uk": "Схожі серіали за твоїм списком:"},
    "header_cross_movie":  {"en": "Movies similar to your list:", "ru": "Похожие фильмы по твоему списку:", "uk": "Схожі фільми за твоїм списком:"},
    "header_trending":     {"en": "<b>🔥 Trending this week:</b>", "ru": "<b>🔥 Тренды недели:</b>", "uk": "<b>🔥 Тренди тижня:</b>"},
    "header_search":       {"en": "<b>For «{q}»:</b>", "ru": "<b>По запросу «{q}»:</b>", "uk": "<b>За запитом «{q}»:</b>"},
    "header_search_movie": {"en": "<b>For «{q}» — movies only:</b>", "ru": "<b>По запросу «{q}» — только фильмы:</b>", "uk": "<b>За запитом «{q}» — лише фільми:</b>"},
    "header_search_tv":    {"en": "<b>For «{q}» — shows only:</b>", "ru": "<b>По запросу «{q}» — только сериалы:</b>", "uk": "<b>За запитом «{q}» — лише серіали:</b>"},
    "header_list":         {"en": "<b>Your list ({n}):</b>", "ru": "<b>Твой список ({n}):</b>", "uk": "<b>Твій список ({n}):</b>"},
    "header_list_paged":   {"en": "<b>Your list ({n}):</b> page {p}/{pp}", "ru": "<b>Твой список ({n}):</b> страница {p}/{pp}", "uk": "<b>Твій список ({n}):</b> сторінка {p}/{pp}"},

    # ---------------- Media-type words ----------------
    "media_movie": {"en": "movie", "ru": "фильм", "uk": "фільм"},
    "media_tv":    {"en": "show",  "ru": "сериал", "uk": "серіал"},

    # ---------------- /lang command ----------------
    "lang_prompt":  {"en": "Choose your language:", "ru": "Выберите язык:", "uk": "Оберіть мову:"},
    "lang_confirm_en": {"en": "Language switched to English.", "ru": "Язык переключён на английский.", "uk": "Мову перемкнуто на англійську."},
    "lang_confirm_ru": {"en": "Language switched to Russian.", "ru": "Язык переключён на русский.", "uk": "Мову перемкнуто на російську."},
    "lang_confirm_uk": {"en": "Language switched to Ukrainian.", "ru": "Язык переключён на украинский.", "uk": "Мову перемкнуто на українську."},

    # ---------------- Bot command descriptions ----------------
    "cmd_desc_start":    {"en": "Start", "ru": "Начать", "uk": "Почати"},
    "cmd_desc_list":     {"en": "My list", "ru": "Мой список", "uk": "Мій список"},
    "cmd_desc_movies":   {"en": "Movie recommendations", "ru": "Рекомендации фильмов", "uk": "Рекомендації фільмів"},
    "cmd_desc_tv":       {"en": "Show recommendations", "ru": "Рекомендации сериалов", "uk": "Рекомендації серіалів"},
    "cmd_desc_trending": {"en": "What's trending", "ru": "Что сейчас смотрят", "uk": "Що зараз дивляться"},
    "cmd_desc_clear":    {"en": "Clear list", "ru": "Очистить список", "uk": "Очистити список"},
    "cmd_desc_lang":     {"en": "Change language", "ru": "Сменить язык", "uk": "Змінити мову"},
}


def T(key: str, lang: str) -> str:
    """Lookup a string by (key, lang) with uk→ru→en fallback.

    Returns the key itself as a last resort — bad enough to surface in
    QA but doesn't crash the handler.
    """
    bucket = I18N.get(key)
    if not bucket:
        return key
    if lang in bucket:
        return bucket[lang]
    for candidate in _FALLBACK.get(lang, ("en",)):
        if candidate in bucket:
            return bucket[candidate]
    return key


def all_button_texts(button_key: str) -> set[str]:
    """Return the union of a button's labels across all supported langs.

    Reply-keyboard handlers use this to keep working when a user
    switches language but keeps tapping their old keyboard.
    """
    bucket = I18N.get(button_key, {})
    return {bucket[l] for l in SUPPORTED_LANGS if l in bucket}
