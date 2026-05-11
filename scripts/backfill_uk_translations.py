"""
Backfill Ukrainian title/overview from TMDB for an entire metadata parquet.

Iterates over `tmdb_id` in `data/processed/<domain>/items_metadata_final.parquet`,
fetches uk-UA translation through `TMDBClient.get_ukrainian_translation`
(which already caches into `TMDBTranslationCache` and skips rows that
already have uk data), and writes title_uk / overview_uk back to the parquet.

The TMDBClient already enforces the 40-req/10s rate limit. With ~14k movies
or ~5k TV shows and ~100ms per uncached request, a full run takes 25 min
for movies and ~10 min for TV. The cache means re-runs are essentially free.

Resumable: a JSON checkpoint stores the last processed tmdb_id and a flag
per row, so an interrupted run picks up where it left off via --resume.
Pass --domain {movies,tv} to pick a parquet, or run twice for both.

Usage:
    python -m scripts.backfill_uk_translations --domain movies
    python -m scripts.backfill_uk_translations --domain tv --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "recommendation_system" / "models" / "gnn"))

from src.recommendation_system.models.gnn.bilingual_utils import TMDBClient, TMDBTranslationCache  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("backfill_uk")

DATA_DIR = PROJECT_ROOT / "data" / "processed"
CACHE_DIR = DATA_DIR / "cache"
CHECKPOINT_EVERY = 500


def _checkpoint_path(domain: str) -> Path:
    return CACHE_DIR / f"backfill_uk_{domain}.checkpoint.json"


def _load_checkpoint(domain: str) -> set[int]:
    p = _checkpoint_path(domain)
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return set(int(x) for x in data.get("done", []))
    except Exception as e:
        logger.warning("Checkpoint parse failed (%s) — starting fresh", e)
        return set()


def _save_checkpoint(domain: str, done: set[int]) -> None:
    p = _checkpoint_path(domain)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"done": sorted(done)}, ensure_ascii=False),
        encoding="utf-8",
    )


def backfill(domain: str, resume: bool, limit: int | None) -> None:
    parquet_path = DATA_DIR / domain / "items_metadata_final.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        raise RuntimeError("TMDB_API_KEY not set in env")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = TMDBTranslationCache(CACHE_DIR / "translations.db")
    client = TMDBClient(api_key, cache)

    df = pd.read_parquet(parquet_path)
    if "title_uk" not in df.columns:
        df["title_uk"] = pd.NA
    if "overview_uk" not in df.columns:
        df["overview_uk"] = pd.NA

    media_type = "tv" if domain == "tv" else "movie"
    done = _load_checkpoint(domain) if resume else set()

    # Pick rows that still need fetching: missing uk title AND not already
    # marked done in checkpoint. tmdb_id may live above TV_OFFSET in TV
    # parquet, but TMDB itself doesn't know about the offset — strip it.
    from src.recommendation_system.models.gnn.faiss_bridge import TV_OFFSET  # local import: avoid heavy deps unless run
    rows = df[df["tmdb_id"].notna()].copy()
    rows["raw_tmdb_id"] = rows["tmdb_id"].astype(int).map(
        lambda x: x - TV_OFFSET if x >= TV_OFFSET else x
    )
    pending = rows[~rows["raw_tmdb_id"].isin(done)]
    if limit is not None:
        pending = pending.head(limit)

    total = len(pending)
    logger.info(
        "Domain=%s parquet=%s rows=%d done=%d pending=%d",
        domain, parquet_path.name, len(df), len(done), total,
    )
    if total == 0:
        logger.info("Nothing to do.")
        cache.close()
        return

    started = time.time()
    fetched = 0
    pending_writes: list[tuple[int, str, str | None]] = []

    for n, (_, row) in enumerate(pending.iterrows(), start=1):
        raw_tmdb = int(row["raw_tmdb_id"])
        full_tmdb = int(row["tmdb_id"])
        try:
            result = client.get_ukrainian_translation(raw_tmdb, media_type)
        except Exception as e:
            logger.warning("Fetch failed tmdb=%s: %s", raw_tmdb, e)
            continue

        if result is None:
            # Hard failure (5xx, network) — don't mark done so a --resume
            # rerun retries this row.
            continue

        title_uk = result.get("title_uk", "") or ""
        overview_uk = result.get("overview_uk", "") or ""

        # Update parquet df via the original (offset-included) tmdb_id —
        # multiple parquet rows can never share a tmdb_id, so this is safe.
        df.loc[df["tmdb_id"] == full_tmdb, "title_uk"] = title_uk or pd.NA
        df.loc[df["tmdb_id"] == full_tmdb, "overview_uk"] = overview_uk or pd.NA

        if title_uk:
            fetched += 1
        done.add(raw_tmdb)
        pending_writes.append((raw_tmdb, title_uk, overview_uk or None))

        if n % CHECKPOINT_EVERY == 0:
            _save_checkpoint(domain, done)
            elapsed = time.time() - started
            rate = n / max(elapsed, 1e-6)
            eta = (total - n) / max(rate, 1e-6)
            logger.info(
                "Progress %d/%d (%.1f%%) fetched=%d rate=%.1f/s eta=%.0fs",
                n, total, 100 * n / total, fetched, rate, eta,
            )
            # Flush parquet in-place periodically so a crash doesn't lose work.
            df.to_parquet(parquet_path, index=False)

    _save_checkpoint(domain, done)
    df.to_parquet(parquet_path, index=False)

    elapsed = time.time() - started
    logger.info(
        "DONE domain=%s processed=%d fetched=%d elapsed=%.0fs",
        domain, total, fetched, elapsed,
    )
    cache.close()


def main() -> None:
    parser = argparse.ArgumentParser()
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
    load_dotenv(PROJECT_ROOT / ".env")
    backfill(args.domain, args.resume, args.limit)


if __name__ == "__main__":
    main()
