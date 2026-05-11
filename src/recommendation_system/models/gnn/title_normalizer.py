"""
Numeric query normalization for title search.

Catalog titles use digits in place of letters as a stylistic choice
(Se7en, M3GAN, 8MM, V/H/S, 1917, 21, 300). A user typing the natural
spelling — "seven", "megan", "eight mm" — would otherwise miss them
because LIKE does not know that "7" and "seven" stand for the same
concept. The reverse is also a problem: querying "se7en" against a
catalog row that only has the spelled-out title misses too.

`normalize_for_search` returns a small set of lowered string variants
to feed into an OR'd LIKE mask. The set always contains the original
lowered query plus, when applicable, variants where digits have been
replaced with their word in en/ru/uk and where number words have been
replaced with the corresponding digit. Variants that collapse to less
than 2 characters are dropped to avoid degenerate matches against
single-digit substrings.

The cyrillic mappings are speculative — Russian/Ukrainian catalog
titles rarely use digit-as-letter — but kept on by design: localized
title fields can carry surprising spellings (TMDB editorial choices),
and the cost of an extra LIKE clause per variant is negligible.
"""

from __future__ import annotations

import re
from typing import Mapping

EN_DIGIT_TO_WORD: Mapping[str, str] = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}
RU_DIGIT_TO_WORD: Mapping[str, str] = {
    "0": "ноль", "1": "один", "2": "два", "3": "три", "4": "четыре",
    "5": "пять", "6": "шесть", "7": "семь", "8": "восемь", "9": "девять",
}
UK_DIGIT_TO_WORD: Mapping[str, str] = {
    "0": "нуль", "1": "один", "2": "два", "3": "три", "4": "чотири",
    "5": "п'ять", "6": "шість", "7": "сім", "8": "вісім", "9": "дев'ять",
}

_LANG_TABLES = (EN_DIGIT_TO_WORD, RU_DIGIT_TO_WORD, UK_DIGIT_TO_WORD)

# Reverse map: every spelled-out digit across all three languages → digit.
# A word like "три" appears in both ru and uk with the same value, so the
# overwrite in the dict comprehension is a no-op. If a future locale ever
# introduces a collision (same word, different digit), the loader below
# would need explicit conflict handling — none today.
WORD_TO_DIGIT: dict[str, str] = {}
for _table in _LANG_TABLES:
    for _digit, _word in _table.items():
        WORD_TO_DIGIT[_word] = _digit

# Word-boundary regex over all known number words. Sorted by length DESC
# so longer words match before shorter prefixes (defensive — none of the
# current entries are prefixes of each other, but a future "ten"/"twenty"
# extension would need it).
_WORD_PATTERN = re.compile(
    r"(?<!\w)(" + "|".join(
        re.escape(w) for w in sorted(WORD_TO_DIGIT, key=len, reverse=True)
    ) + r")(?!\w)",
    re.UNICODE,
)

_MIN_VARIANT_LEN = 2


def normalize_for_search(text: str) -> list[str]:
    """Return distinct lowered variants of `text` for LIKE-mask matching.

    Always includes the original lowered query. Adds:
      * one digit→word variant per language whenever `text` has digits;
      * one word→digit variant whenever `text` has a recognised number
        word (any language) on a word boundary.

    Variants shorter than `_MIN_VARIANT_LEN` characters are dropped —
    a query that collapses to "7" alone would substring-match almost
    every title with a year in it.

    Order is stable (sorted) so callers that hash or log the variants
    get reproducible output.
    """
    if not text:
        return []
    base = text.lower().strip()
    if not base:
        return []

    variants: set[str] = {base}

    if any(ch.isdigit() for ch in base):
        for table in _LANG_TABLES:
            converted = "".join(table.get(ch, ch) for ch in base)
            if converted != base and len(converted) >= _MIN_VARIANT_LEN:
                variants.add(converted)

    if _WORD_PATTERN.search(base):
        converted = _WORD_PATTERN.sub(
            lambda m: WORD_TO_DIGIT[m.group(1)], base
        )
        if converted != base and len(converted) >= _MIN_VARIANT_LEN:
            variants.add(converted)

    return sorted(variants)
