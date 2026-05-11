"""
Pure functions for recommendation quality metrics.

Used by:
- TestQualityTrajectory (N=1..5 growth analysis)
- TestGoldenStandard (must-contain IDs recall)
- TestMediaDNA (TMDB keyword overlap)
- TestPopularityBias (top-10% by vote_count)
- TestGraphOverlap (LightGCN-neighbor ⊆ recommendations)

All functions are side-effect-free: they take data in, return numbers/sets out.
"""
from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


# --------------------------------------------------------------------------
# Genres (moved from test_recommendation_scenarios.py, signatures preserved)
# --------------------------------------------------------------------------

def normalize_genres(genres) -> set[str]:
    if genres is None:
        return set()
    out: set[str] = set()
    for g in genres:
        if g is None:
            continue
        s = str(g).lower().strip()
        out.add(s)
        if s == "sci-fi":
            out.add("science-fiction")
        if s == "science-fiction":
            out.add("sci-fi")
    return out


def genre_overlap_pct(recs, expected: set[str]) -> float:
    if not recs:
        return 0.0
    expected_lower = {e.lower() for e in expected}
    hits = sum(
        1 for r in recs
        if normalize_genres(getattr(r, "genres", []) or []) & expected_lower
    )
    return 100.0 * hits / len(recs)


# --------------------------------------------------------------------------
# Keywords (Media DNA)
# --------------------------------------------------------------------------

def _ensure_str_list(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(x).lower().strip() for x in raw if x is not None and str(x).strip()]
    # pandas may store keywords as numpy array or stringified list.
    try:
        import numpy as np
        if isinstance(raw, np.ndarray):
            return [str(x).lower().strip() for x in raw.tolist() if x is not None and str(x).strip()]
    except Exception:
        pass
    return []


def collect_liked_keywords(metadata_df: pd.DataFrame, liked_tmdb_ids: Iterable[int]) -> set[str]:
    """Union of lowercase keywords across all liked items that exist in metadata."""
    liked = set(int(x) for x in liked_tmdb_ids)
    if "keywords" not in metadata_df.columns or not liked:
        return set()
    rows = metadata_df[metadata_df["tmdb_id"].isin(liked)]
    out: set[str] = set()
    for kw in rows["keywords"]:
        out.update(_ensure_str_list(kw))
    return out


def keyword_overlap_pct(recs, liked_keywords: set[str]) -> float:
    """Percentage of recommendations that share at least one keyword with the liked set."""
    if not recs or not liked_keywords:
        return 0.0
    hits = 0
    for r in recs:
        rec_kw = set(_ensure_str_list(getattr(r, "keywords", []) or []))
        if rec_kw & liked_keywords:
            hits += 1
    return 100.0 * hits / len(recs)


# --------------------------------------------------------------------------
# Popularity Swamp
# --------------------------------------------------------------------------

def build_popular_set(metadata_df: pd.DataFrame, pct: float = 0.10) -> set[int]:
    """Return the set of tmdb_ids that fall in the top-`pct` of the catalog by vote_count."""
    if "vote_count" not in metadata_df.columns or metadata_df.empty:
        return set()
    threshold = metadata_df["vote_count"].quantile(1.0 - pct)
    top = metadata_df[metadata_df["vote_count"] >= threshold]
    return set(int(v) for v in top["tmdb_id"].dropna().tolist())


def popular_share(recs, popular_ids: set[int]) -> float:
    """Fraction (0..1) of recommendations that are in the popular set."""
    if not recs:
        return 0.0
    hits = sum(1 for r in recs if int(getattr(r, "tmdb_id", -1)) in popular_ids)
    return hits / len(recs)


# --------------------------------------------------------------------------
# Golden Standard
# --------------------------------------------------------------------------

def recall_at_k(recs, must_contain_ids: set[int], k: Optional[int] = None) -> float:
    """Fraction (0..1) of `must_contain_ids` that appear in the first `k` recommendations."""
    if not must_contain_ids:
        return 1.0
    if not recs:
        return 0.0
    pool = recs[:k] if k is not None else recs
    found = {int(getattr(r, "tmdb_id", -1)) for r in pool} & must_contain_ids
    return len(found) / len(must_contain_ids)


# --------------------------------------------------------------------------
# Graph overlap (LightGCN-neighbor prediction)
# --------------------------------------------------------------------------

def compute_graph_neighbors(
    inference_engine,
    search_engine,
    liked_tmdb_ids: Iterable[int],
    k: int = 50,
) -> set[int]:
    """
    For each liked tmdb_id that sits in the LightGCN graph, fetch the top-k
    nearest items in embedding space via the engine's own ranker, and return
    the union of their tmdb_ids (excluding the liked inputs themselves).

    Returns an empty set if none of the liked ids are in-graph.
    """
    liked = [int(x) for x in liked_tmdb_ids]
    if not liked or inference_engine is None or search_engine is None:
        return set()

    tmdb_to_item = search_engine.tmdb_to_item_id
    item_to_tmdb: dict[int, int] = {}
    metadata = search_engine.metadata
    for _, row in metadata[["tmdb_id", "item_id"]].dropna().iterrows():
        item_to_tmdb[int(row["item_id"])] = int(row["tmdb_id"])

    union: set[int] = set()
    for tid in liked:
        iid = tmdb_to_item.get(tid)
        if iid is None:
            continue
        raw = inference_engine.get_recommendations(liked_item_ids=[iid], top_k=k)
        for entry in raw or []:
            rec_iid = entry.get("item_id")
            if rec_iid is None:
                continue
            mapped = item_to_tmdb.get(int(rec_iid))
            if mapped is not None:
                union.add(mapped)

    union.difference_update(liked)
    return union
