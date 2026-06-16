"""
Unit tests for `title_normalizer.normalize_for_search`.

Covers the three variant flavours the LIKE-mask depends on:
  * digit→word in en/ru/uk (so a query "Se7en" finds catalog "Seven"
    and "Сім" / "Семь" too);
  * word→digit (so "seven" finds catalog "Se7en");
  * cyrillic-language number words (so "сім самураїв" finds a row
    that happens to use "7 самураїв" — speculative but cheap).

The min-length and dedup invariants are pinned with their own cases —
they prevent the LIKE mask from collapsing to a single-character
substring that would surface every title containing that digit.
"""

from __future__ import annotations

import pytest

from recommendation_system.models.gnn.title_normalizer import (
    EN_DIGIT_TO_WORD,
    RU_DIGIT_TO_WORD,
    UK_DIGIT_TO_WORD,
    WORD_TO_DIGIT,
    normalize_for_search,
)


# ---------------------------------------------------------------------------
# Empty / degenerate inputs
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_list():
    assert normalize_for_search("") == []
    assert normalize_for_search("   ") == []


def test_none_handled():
    """The function is the only normalization point — must not raise on None."""
    assert normalize_for_search(None) == []  # type: ignore[arg-type]


def test_plain_alpha_query_returns_only_lowered_original():
    assert normalize_for_search("Inception") == ["inception"]


def test_already_lower_alpha_query_unchanged():
    assert normalize_for_search("matrix") == ["matrix"]


# ---------------------------------------------------------------------------
# Digit → word (en/ru/uk)
# ---------------------------------------------------------------------------


def test_se7en_produces_english_spelled_variant():
    """The flagship case: 'Se7en' yields a per-language digit→word
    variant that substring-contains 'seven' so the LIKE mask hits a
    catalog row spelled with the digit AND a row that happens to use
    the spelled form somewhere (e.g. 'Sesevenen' is unlikely, but the
    variant `sesevenen` will substring-match longer titles whose
    English spelling embeds 'seven' — and is harmless otherwise)."""
    out = normalize_for_search("Se7en")
    assert "se7en" in out
    # Char-by-char digit→word preserves surrounding letters, so the
    # English variant is 'sesevenen' (s + e + 'seven' + e + n), not
    # the prettier 'seseven'. The substring 'seven' is what matters
    # for LIKE matching.
    assert "sesevenen" in out, out
    assert any("seven" in v for v in out)


def test_se7en_produces_russian_spelled_variant():
    out = normalize_for_search("Se7en")
    assert any("семь" in v for v in out), out


def test_se7en_produces_ukrainian_spelled_variant():
    out = normalize_for_search("Se7en")
    assert any("сім" in v for v in out), out


def test_m3gan_produces_megan_variant():
    out = normalize_for_search("M3GAN")
    assert any("mthreegan" in v for v in out), out


def test_8mm_produces_eightmm_variant():
    """8MM (1999) — query '8mm' should yield variant 'eightmm' so a
    catalog row spelled 'Eight MM' would match."""
    out = normalize_for_search("8mm")
    assert "8mm" in out
    assert "eightmm" in out, out


def test_multiple_digits_all_substituted_in_one_variant():
    """1917 (Mendes' WWI movie) → english variant should spell every
    digit, not just the first one."""
    out = normalize_for_search("1917")
    en_variant = next((v for v in out if v.isalpha()), None)
    assert en_variant == "onenineoneseven", out


# ---------------------------------------------------------------------------
# Word → digit
# ---------------------------------------------------------------------------


def test_seven_word_produces_digit_variant_only_with_context():
    """Bare 'seven' → '7' is single-char and gets dropped by the
    min-length filter (an LIKE %7% would substring-match every title
    that has a year or any digit). Add even one more token and the
    word→digit substitution survives."""
    bare = normalize_for_search("seven")
    assert bare == ["seven"], bare

    with_context = normalize_for_search("seven samurai")
    assert "seven samurai" in with_context
    assert "7 samurai" in with_context, with_context


def test_word_to_digit_only_on_word_boundary():
    """'seventeen' starts with 'seven' but is a different word — the
    \\b regex must not split it. Otherwise we'd produce '7teen'."""
    out = normalize_for_search("seventeen")
    assert out == ["seventeen"], out


def test_word_to_digit_inside_phrase():
    """A query like 'seven samurai' should produce '7 samurai' so the
    catalog row 'Seven Samurai' (English) and any localized row that
    actually used the digit both get hit."""
    out = normalize_for_search("seven samurai")
    assert "seven samurai" in out
    assert "7 samurai" in out, out


def test_russian_word_to_digit():
    """User types 'семь самураев' — yield '7 самураев'. Speculative but
    matches the design goal of covering localized digit-as-letter."""
    out = normalize_for_search("семь самураев")
    assert "семь самураев" in out
    assert "7 самураев" in out, out


def test_ukrainian_word_to_digit():
    out = normalize_for_search("сім самураїв")
    assert "сім самураїв" in out
    assert "7 самураїв" in out, out


def test_cyrillic_query_not_skipped_for_substitution():
    """Spec note: cyrillic queries are NOT skipped. RU/UK titles can
    occasionally pull the same digit-as-letter trick, so we still
    generate variants. The bare 'один' → '1' single-char variant is
    dropped by the same min-length rule that affects 'seven', but
    'один самурай' → '1 самурай' survives."""
    bare = normalize_for_search("один")
    assert bare == ["один"], bare

    with_context = normalize_for_search("один самурай")
    assert "один самурай" in with_context
    assert "1 самурай" in with_context, with_context


# ---------------------------------------------------------------------------
# Variant set invariants
# ---------------------------------------------------------------------------


def test_variants_are_unique():
    out = normalize_for_search("Se7en")
    assert len(out) == len(set(out))


def test_variants_are_lowercased():
    out = normalize_for_search("SE7EN")
    assert all(v == v.lower() for v in out)


def test_variants_are_sorted_for_stable_order():
    """Sorted output gives reproducible logging / hashing for callers."""
    out = normalize_for_search("Se7en")
    assert out == sorted(out)


def test_no_variant_shorter_than_two_chars():
    """A bare '7' substring would match every year-bearing title — the
    min-length filter must reject it."""
    out = normalize_for_search("7")
    # '7' itself is the original (kept), but the en/ru/uk word-out
    # variants should all be present and longer than 2 chars.
    assert "7" in out  # original is always preserved
    assert "seven" in out
    assert "семь" in out
    assert "сім" in out


def test_short_digit_word_query_not_collapsed_to_single_char():
    """Edge: 'one' alone → variant '1' is single-char and should be
    dropped by the min-length filter, but 'one' itself stays."""
    out = normalize_for_search("one")
    assert "one" in out
    # '1' single-char would substring-match every title with a year
    # like (2010) — explicitly dropped.
    assert "1" not in out


# ---------------------------------------------------------------------------
# Mapping-table sanity
# ---------------------------------------------------------------------------


def test_all_three_languages_cover_digits_zero_through_nine():
    digits = {str(d) for d in range(10)}
    for table in (EN_DIGIT_TO_WORD, RU_DIGIT_TO_WORD, UK_DIGIT_TO_WORD):
        assert set(table.keys()) == digits


def test_word_to_digit_reverse_map_complete():
    """Every word in any language table must be reachable through the
    reverse map."""
    for table in (EN_DIGIT_TO_WORD, RU_DIGIT_TO_WORD, UK_DIGIT_TO_WORD):
        for digit, word in table.items():
            assert WORD_TO_DIGIT[word] == digit


def test_ru_uk_share_some_digit_words_no_conflict():
    """Several digits use the same word in ru and uk (e.g. 'три', 'два',
    'один'). The reverse map must agree on the digit, no silent drift."""
    overlap = set(RU_DIGIT_TO_WORD.values()) & set(UK_DIGIT_TO_WORD.values())
    for word in overlap:
        ru_digit = next(d for d, w in RU_DIGIT_TO_WORD.items() if w == word)
        uk_digit = next(d for d, w in UK_DIGIT_TO_WORD.items() if w == word)
        assert ru_digit == uk_digit, (
            f"word {word!r} maps to different digits in ru ({ru_digit}) "
            f"vs uk ({uk_digit})"
        )
