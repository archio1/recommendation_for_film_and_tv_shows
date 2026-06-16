"""
Backfill localized title/overview from TMDB for an entire metadata parquet.

One script for every supported language (`--lang ru|uk`); replaces the old
near-identical scripts/backfill_ru_translations.py and
scripts/backfill_uk_translations.py. Iterates over `tmdb_id` in
`data/processed/<domain>/items_metadata_final.parquet`, fetches the
translation through the language-specific TMDBClient method (which caches
into `TMDBTranslationCache` without clobbering the other language's rows),
and writes `title_<lang>` / `overview_<lang>` back to the parquet.

The TMDBClient already enforces the 40-req/10s rate limit. With ~14k movies
or ~5k TV shows and ~100ms per uncached request, a full run takes ~25 min
for movies and ~10 min for TV. The cache means re-runs are essentially free.

Resumable: a per-language, per-domain JSON checkpoint stores processed
tmdb_ids, so an interrupted run picks up where it left off via --resume.
Checkpoint file names match the old scripts (`backfill_<lang>_<domain>.
checkpoint.json`), so existing checkpoints keep working.

Usage:
    recsys-backfill --lang ru --domain movies
    recsys-backfill --lang uk --domain tv --resume

Adding another language is NOT just a new entry in LANG_CONFIG: the
translations SQLite cache has fixed per-language columns, TMDBClient has
per-language fetch methods, and the search/bot layers hardcode the
title_ru/title_uk fallback chains. See ARCHITECTURE.md before extending.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from recommendation_system.models.gnn.bilingual_utils import TMDBClient, TMDBTranslationCache
from recommendation_system.paths import CACHE_DIR, ENV_FILE, PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)

DATA_DIR = PROCESSED_DIR
CHECKPOINT_EVERY = 500


@dataclass(frozen=True)
class LangSpec:
    client_method: str   # TMDBClient method that fetches+caches this language
    title_col: str
    overview_col: str
    checkpoint_prefix: str


LANG_CONFIG: dict[str, LangSpec] = {
    "ru": LangSpec(
        client_method="get_russian_translation",
        title_col="title_ru",
        overview_col="overview_ru",
        checkpoint_prefix="backfill_ru",
    ),
    "uk": LangSpec(
        client_method="get_ukrainian_translation",
        title_col="title_uk",
        overview_col="overview_uk",
        checkpoint_prefix="backfill_uk",
    ),
}


def _checkpoint_path(spec: LangSpec, domain: str) -> Path:
    return CACHE_DIR / f"{spec.checkpoint_prefix}_{domain}.checkpoint.json"


def _load_checkpoint(spec: LangSpec, domain: str, logger: logging.Logger) -> set[int]:
    p = _checkpoint_path(spec, domain)
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return set(int(x) for x in data.get("done", []))
    except Exception as e:
        logger.warning("Checkpoint parse failed (%s) — starting fresh", e)
        return set()


def _save_checkpoint(spec: LangSpec, domain: str, done: set[int]) -> None:
    p = _checkpoint_path(spec, domain)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"done": sorted(done)}, ensure_ascii=False),
        encoding="utf-8",
    )


def backfill(lang: str, domain: str, resume: bool, limit: int | None) -> None:
    spec = LANG_CONFIG[lang]
    logger = logging.getLogger(f"backfill_{lang}")

    parquet_path = DATA_DIR / domain / "items_metadata_final.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        raise RuntimeError("TMDB_API_KEY not set in env")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = TMDBTranslationCache(CACHE_DIR / "translations.db")
    client = TMDBClient(api_key, cache)
    fetch = getattr(client, spec.client_method)

    df = pd.read_parquet(parquet_path)
    # Ensure both columns exist before the first write (historically some
    # parquets carried only the title placeholder, or neither column).
    for col in (spec.title_col, spec.overview_col):
        if col not in df.columns:
            df[col] = pd.NA

    media_type = "tv" if domain == "tv" else "movie"
    done = _load_checkpoint(spec, domain, logger) if resume else set()

    # tmdb_id may live above TV_OFFSET in the TV parquet, but TMDB itself
    # doesn't know about the offset — strip it for API calls.
    from recommendation_system.models.gnn.faiss_bridge import TV_OFFSET  # local import: avoid heavy deps unless run
    rows = df[df["tmdb_id"].notna()].copy()
    rows["raw_tmdb_id"] = rows["tmdb_id"].astype(int).map(
        lambda x: x - TV_OFFSET if x >= TV_OFFSET else x
    )
    pending = rows[~rows["raw_tmdb_id"].isin(done)]
    if limit is not None:
        pending = pending.head(limit)

    total = len(pending)
    logger.info(
        "Lang=%s domain=%s parquet=%s rows=%d done=%d pending=%d",
        lang, domain, parquet_path.name, len(df), len(done), total,
    )
    if total == 0:
        logger.info("Nothing to do.")
        cache.close()
        return

    started = time.time()
    fetched = 0

    for n, (_, row) in enumerate(pending.iterrows(), start=1):
        raw_tmdb = int(row["raw_tmdb_id"])
        full_tmdb = int(row["tmdb_id"])
        try:
            result = fetch(raw_tmdb, media_type)
        except Exception as e:
            logger.warning("Fetch failed tmdb=%s: %s", raw_tmdb, e)
            continue

        if result is None:
            # Hard failure (5xx, network) — don't mark done so a --resume
            # rerun retries this row.
            continue

        title = result.get(spec.title_col, "") or ""
        overview = result.get(spec.overview_col, "") or ""

        # Update parquet df via the original (offset-included) tmdb_id —
        # multiple parquet rows can never share a tmdb_id, so this is safe.
        df.loc[df["tmdb_id"] == full_tmdb, spec.title_col] = title or pd.NA
        df.loc[df["tmdb_id"] == full_tmdb, spec.overview_col] = overview or pd.NA

        if title:
            fetched += 1
        done.add(raw_tmdb)

        if n % CHECKPOINT_EVERY == 0:
            _save_checkpoint(spec, domain, done)
            elapsed = time.time() - started
            rate = n / max(elapsed, 1e-6)
            eta = (total - n) / max(rate, 1e-6)
            logger.info(
                "Progress %d/%d (%.1f%%) fetched=%d rate=%.1f/s eta=%.0fs",
                n, total, 100 * n / total, fetched, rate, eta,
            )
            # Flush parquet in-place periodically so a crash doesn't lose work.
            df.to_parquet(parquet_path, index=False)

    _save_checkpoint(spec, domain, done)
    df.to_parquet(parquet_path, index=False)

    elapsed = time.time() - started
    logger.info(
        "DONE lang=%s domain=%s processed=%d fetched=%d elapsed=%.0fs",
        lang, domain, total, fetched, elapsed,
    )
    cache.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill localized title/overview columns from TMDB."
    )
    parser.add_argument("--lang", choices=sorted(LANG_CONFIG), required=True)
    parser.add_argument("--domain", choices=["movies", "tv"], required=True)
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip tmdb_ids already in the checkpoint file.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N rows (debugging / first-pass smoke).",
    )
    args = parser.parse_args()
    load_dotenv(ENV_FILE)
    backfill(args.lang, args.domain, args.resume, args.limit)


if __name__ == "__main__":
    main()
