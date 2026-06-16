"""
Persistent bottom reply-keyboard.

The reply keyboard stays visible across messages (Telegram persists it on
the client until we send a ReplyKeyboardRemove). Per-user language picks
which labels render — but reply-handlers in movie_bot.py match against
the union of labels across all 3 supported languages, so a user who
flips language keeps a working keyboard until their next /start.
"""

from __future__ import annotations

from aiogram.types import KeyboardButton, ReplyKeyboardMarkup

from .i18n import T, all_button_texts


def main_reply_kb(lang: str = "en") -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text=T("btn_search", lang)),
                KeyboardButton(text=T("btn_list", lang)),
            ],
            [
                KeyboardButton(text=T("btn_movies", lang)),
                KeyboardButton(text=T("btn_tv", lang)),
            ],
            [
                KeyboardButton(text=T("btn_trending", lang)),
                KeyboardButton(text=T("btn_clear", lang)),
            ],
        ],
        resize_keyboard=True,
        is_persistent=True,
    )


# Multi-language button label sets — handlers use these with F.text.in_(...)
# so a user who switched language and is still tapping the old keyboard
# still gets routed to the right action.
BTN_SEARCH_ALL = all_button_texts("btn_search")
BTN_LIST_ALL = all_button_texts("btn_list")
BTN_MOVIES_ALL = all_button_texts("btn_movies")
BTN_TV_ALL = all_button_texts("btn_tv")
BTN_TRENDING_ALL = all_button_texts("btn_trending")
BTN_CLEAR_ALL = all_button_texts("btn_clear")
