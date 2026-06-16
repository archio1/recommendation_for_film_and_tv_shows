"""
dual_domain_engine.py — router over two per-domain LightGCN engines + a
shared FAISS content-bridge.

Delegates to `UniversalSearchEngine` (one per domain) for collaborative
recommendations, and to `FaissCatalog` for cross-domain / cold-start
content search. User-facing surface: /recs_movie, /recs_tv, /recs_cross,
/recs_all.

tmdb_id is the only identifier accepted at this layer — item_id spaces
are disjoint between movies and tv and can't safely be mixed. Movie
tmdb_ids are raw; tv tmdb_ids carry the +TV_OFFSET offset that was baked
into the parquet metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np

from recommendation_system.models.gnn.faiss_bridge import TV_OFFSET, FaissCatalog

if TYPE_CHECKING:
    from recommendation_system.models.gnn.cold_start import ColdStartIngestor
    from recommendation_system.models.gnn.universal_search import (
        UniversalMediaItem,
        UniversalSearchEngine,
    )


class DualDomainEngine:
    """Thin coordinator over two domain-specific engines + shared FAISS."""

    def __init__(
        self,
        movies_engine: "UniversalSearchEngine",
        tv_engine: "UniversalSearchEngine",
        faiss_catalog: "FaissCatalog",
        cold_start: "ColdStartIngestor | None" = None,
    ):
        self.movies = movies_engine
        self.tv = tv_engine
        self.faiss = faiss_catalog
        self.cold_start = cold_start

    @staticmethod
    def _split_by_domain(
        liked_tmdb_ids: Iterable[int] | None,
    ) -> tuple[list[int], list[int]]:
        """Split tmdb_ids into (movie_ids, tv_ids) by TV_OFFSET convention."""
        if not liked_tmdb_ids:
            return [], []
        movie_ids: list[int] = []
        tv_ids: list[int] = []
        for tid in liked_tmdb_ids:
            if tid is None:
                continue
            tid = int(tid)
            if tid >= TV_OFFSET:
                tv_ids.append(tid)
            else:
                movie_ids.append(tid)
        return movie_ids, tv_ids

    @staticmethod
    def _to_item_ids(
        tmdb_ids: list[int],
        engine: "UniversalSearchEngine",
    ) -> tuple[list[int], list[int]]:
        """
        Split tmdb_ids into (item_ids_known_to_graph, tmdb_ids_unknown).
        Known ones go through the collaborative (LightGCN) path; unknown
        ones will later be routed through FAISS cold-start.
        """
        known: list[int] = []
        unknown: list[int] = []
        for tid in tmdb_ids:
            iid = engine.tmdb_to_item_id.get(int(tid))
            if iid is not None:
                known.append(int(iid))
            else:
                unknown.append(int(tid))
        return known, unknown

    def recs_movie(
        self,
        liked_tmdb_ids: Iterable[int] | None = None,
        top_k: int = 10,
        popularity_debias: float | None = None,
    ) -> list["UniversalMediaItem"]:
        """
        LightGCN-movies recommendations. TV tmdb_ids are silently ignored —
        for cross-domain output use recs_cross or recs_all.

        popularity_debias overrides the engine default per call (None = use the
        engine's own setting). The bot passes a per-user value here; mutating the
        shared engine attribute would race across concurrent users.
        """
        movie_ids, _ = self._split_by_domain(liked_tmdb_ids)
        if not movie_ids:
            return []
        known_iids, unknown_tids = self._to_item_ids(movie_ids, self.movies)
        if not known_iids and not unknown_tids:
            return []
        return self.movies.get_recommendations(
            liked_item_ids=known_iids or None,
            liked_tmdb_ids=unknown_tids or None,
            top_k=top_k,
            media_type="movie",
            popularity_debias=popularity_debias,
        )

    def recs_tv(
        self,
        liked_tmdb_ids: Iterable[int] | None = None,
        top_k: int = 10,
    ) -> list["UniversalMediaItem"]:
        """LightGCN-tv recommendations. Movie tmdb_ids are silently ignored."""
        _, tv_ids = self._split_by_domain(liked_tmdb_ids)
        if not tv_ids:
            return []
        known_iids, unknown_tids = self._to_item_ids(tv_ids, self.tv)
        if not known_iids and not unknown_tids:
            return []
        return self.tv.get_recommendations(
            liked_item_ids=known_iids or None,
            liked_tmdb_ids=unknown_tids or None,
            top_k=top_k,
            media_type="tv",
        )

    def _reconstruct_query(self, liked_tmdb_ids: list[int]) -> np.ndarray | None:
        """
        Pull stored embeddings for each liked tmdb_id out of FAISS and
        average them. For ids not yet in the index, ask the cold-start
        ingestor to fetch+encode+upsert on the fly (if one is wired up);
        otherwise they're silently skipped.
        """
        vecs: list[np.ndarray] = []
        for tid in liked_tmdb_ids:
            media_type = "tv" if tid >= TV_OFFSET else "movie"
            fid = FaissCatalog._faiss_id(tid, media_type)
            if fid not in self.faiss.mapping and self.cold_start is not None:
                self.cold_start.ensure(tid, media_type)
            if fid in self.faiss.mapping:
                vecs.append(self.faiss.index.reconstruct(fid))
        if not vecs:
            return None
        return np.mean(np.stack(vecs), axis=0)

    def _result_to_item(self, tmdb_id_raw: int, media_type: str):
        """
        Map a FaissSearchResult back to a UniversalMediaItem.

        Lookup order: per-domain parquet metadata first (in_graph=True),
        then the per-domain HotCache (cold-started items that never made
        it into a retrain). Returns None if neither knows it.
        """
        engine = self.tv if media_type == "tv" else self.movies
        lookup_tmdb = tmdb_id_raw + TV_OFFSET if media_type == "tv" else tmdb_id_raw
        row = engine.metadata[engine.metadata["tmdb_id"] == lookup_tmdb]
        if not row.empty:
            return engine._row_to_universal(row.iloc[0])
        return engine.hot_cache.get_by_tmdb_id(lookup_tmdb)

    _RRF_K = 60  # standard Reciprocal Rank Fusion constant

    @staticmethod
    def _split_quota(top_k: int, n_movie: int, n_tv: int) -> tuple[int, int]:
        """
        Divide top_k between domains proportionally to the number of liked
        items in each. If the user liked items in both domains, guarantee
        each side gets at least 1 slot (so the mixed feed stays mixed).
        """
        if n_movie and not n_tv:
            return top_k, 0
        if n_tv and not n_movie:
            return 0, top_k
        total = n_movie + n_tv
        q_movie = round(top_k * n_movie / total)
        q_movie = max(1, min(q_movie, top_k - 1))
        return q_movie, top_k - q_movie

    def recs_all(
        self,
        liked_tmdb_ids: Iterable[int] | None = None,
        top_k: int = 10,
        popularity_debias: float | None = None,
    ) -> list["UniversalMediaItem"]:
        """
        Combined feed from both LightGCN models. Each domain is queried
        independently with a quota proportional to the user's liked-items
        split, then merged by Reciprocal Rank Fusion so items ranked high
        in either model surface to the top.

        Items that appear in both domain outputs would have been filtered
        by domain anyway (disjoint tmdb_id spaces via TV_OFFSET), so dedup
        is a safety net only.
        """
        if not liked_tmdb_ids:
            return []

        movie_likes, tv_likes = self._split_by_domain(liked_tmdb_ids)
        if not movie_likes and not tv_likes:
            return []

        q_movie, q_tv = self._split_quota(top_k, len(movie_likes), len(tv_likes))

        # Oversample 2x per side — absorbs items dropped during merge (e.g.
        # overlap with likes already handled inside each engine) and leaves
        # RRF enough material to reorder.
        movie_recs = (
            self.recs_movie(
                liked_tmdb_ids=movie_likes, top_k=q_movie * 2,
                popularity_debias=popularity_debias,
            )
            if q_movie else []
        )
        tv_recs = (
            self.recs_tv(liked_tmdb_ids=tv_likes, top_k=q_tv * 2) if q_tv else []
        )

        scored: dict[int, tuple[float, "UniversalMediaItem"]] = {}
        for recs in (movie_recs, tv_recs):
            for rank, item in enumerate(recs):
                tid = int(item.tmdb_id)
                contribution = 1.0 / (rank + self._RRF_K)
                if tid in scored:
                    scored[tid] = (scored[tid][0] + contribution, scored[tid][1])
                else:
                    scored[tid] = (contribution, item)

        # Pick top_k globally by RRF score, but respect the domain quotas
        # so a dominant single-domain signal doesn't starve the other side
        # when the user clearly liked both.
        if q_movie and q_tv:
            ranked_movie = sorted(
                (p for p in scored.values() if p[1].media_type == "movie"),
                key=lambda x: -x[0],
            )[:q_movie]
            ranked_tv = sorted(
                (p for p in scored.values() if p[1].media_type == "tv"),
                key=lambda x: -x[0],
            )[:q_tv]
            merged = sorted(ranked_movie + ranked_tv, key=lambda x: -x[0])
        else:
            merged = sorted(scored.values(), key=lambda x: -x[0])

        return [item for _, item in merged[:top_k]]

    def recs_cross(
        self,
        liked_tmdb_ids: Iterable[int] | None,
        target_media_type: str,
        top_k: int = 10,
    ) -> list["UniversalMediaItem"]:
        """
        Content-bridge recommendations: average liked embeddings and search
        the shared FAISS index, filtered to `target_media_type`. LightGCN
        is not involved — this is pure semantic similarity over overview +
        genres, which is what makes cross-domain possible at all.

        `liked_tmdb_ids` can mix movie and tv ids; their embeddings are all
        averaged together into a single taste vector. The liked items
        themselves are excluded from results.
        """
        if target_media_type not in {"movie", "tv"}:
            raise ValueError(
                f"target_media_type must be 'movie' or 'tv', got {target_media_type!r}"
            )
        if not liked_tmdb_ids:
            return []

        liked_list = [int(t) for t in liked_tmdb_ids if t is not None]
        query = self._reconstruct_query(liked_list)
        if query is None:
            return []

        liked_fids = {
            FaissCatalog._faiss_id(t, "tv" if t >= TV_OFFSET else "movie")
            for t in liked_list
        }

        # Oversample so we can drop liked items and still hit top_k.
        raw = self.faiss.search(
            query, top_k=top_k + len(liked_fids), media_type_filter=target_media_type
        )

        out: list["UniversalMediaItem"] = []
        for r in raw:
            fid = FaissCatalog._faiss_id(r.tmdb_id, r.media_type)
            if fid in liked_fids:
                continue
            item = self._result_to_item(r.tmdb_id, r.media_type)
            if item is not None:
                out.append(item)
                if len(out) >= top_k:
                    break
        return out
