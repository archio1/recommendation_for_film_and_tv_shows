"""
Cross-domain search merge + short-lived query cache for callback_data.

Each per-domain `UniversalSearchEngine.search()` returns items already
sorted by its own relevance heuristic (exact → prefix → popularity →
title length). We don't have a shared numeric score, so we merge by
rank: alternate the best-ranked item from each domain, then the second,
and so on. This guarantees each domain gets fair representation in the
top-N cut while preserving per-domain ordering.

Telegram `callback_data` is capped at 64 bytes, which is not enough for
arbitrary user queries. The filter buttons under search results need to
re-run the query for a single domain, so we cache the query string by a
short hash and pass only the hash in callback_data.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from recommendation_system.models.gnn.universal_search import (
        UniversalMediaItem,
        UniversalSearchEngine,
    )


_TTL_SECONDS = 10 * 60
_MAX_ENTRIES = 500
_query_cache: "OrderedDict[str, tuple[str, float]]" = OrderedDict()


def store_query(query: str) -> str:
    """Cache query; return a 10-char hash safe for callback_data."""
    now = time.time()
    _evict_expired(now)
    while len(_query_cache) >= _MAX_ENTRIES:
        _query_cache.popitem(last=False)
    h = hashlib.md5(query.encode("utf-8")).hexdigest()[:10]
    _query_cache[h] = (query, now)
    _query_cache.move_to_end(h)
    return h


def get_query(h: str) -> Optional[str]:
    rec = _query_cache.get(h)
    if rec is None:
        return None
    query, ts = rec
    if time.time() - ts > _TTL_SECONDS:
        _query_cache.pop(h, None)
        return None
    return query


def _evict_expired(now: float) -> None:
    stale = [h for h, (_, ts) in _query_cache.items() if now - ts > _TTL_SECONDS]
    for h in stale:
        _query_cache.pop(h, None)


def merged_search(
    movies_engine: "UniversalSearchEngine",
    tv_engine: "UniversalSearchEngine",
    query: str,
    limit: int = 8,
    media_type: Optional[str] = None,
) -> list["UniversalMediaItem"]:
    """
    Single-domain search when media_type is set; otherwise alternate-merge
    results from both domains by rank, deduped on tmdb_id.
    """
    if media_type == "movie":
        return movies_engine.search(query, limit=limit).results[:limit]
    if media_type == "tv":
        return tv_engine.search(query, limit=limit).results[:limit]

    movie_results = movies_engine.search(query, limit=limit).results
    tv_results = tv_engine.search(query, limit=limit).results

    merged: list = []
    seen: set[int] = set()
    for rank in range(max(len(movie_results), len(tv_results))):
        if rank < len(movie_results):
            item = movie_results[rank]
            if item.tmdb_id not in seen:
                merged.append(item)
                seen.add(item.tmdb_id)
                if len(merged) >= limit:
                    break
        if rank < len(tv_results):
            item = tv_results[rank]
            if item.tmdb_id not in seen:
                merged.append(item)
                seen.add(item.tmdb_id)
                if len(merged) >= limit:
                    break
    return merged
