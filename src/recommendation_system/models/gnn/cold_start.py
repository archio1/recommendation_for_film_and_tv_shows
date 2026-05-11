"""
cold_start.py — TMDb → HotCache → SBERT → FAISS pipeline for items that
neither LightGCN-movies nor LightGCN-tv knows about yet.

When a user pastes a freshly-released movie or tv show, the graph models
can't recommend it (not retrained) but content-bridge still should: we
pull metadata from TMDb, persist it in the per-domain HotCache, encode
the overview+genres with the same SBERT model used at build time, and
upsert into the shared FaissCatalog. Next request for "similar" hits the
index instantly.

The ingestor is idempotent: if the tmdb_id is already in FAISS, it
returns the cached UniversalMediaItem without touching TMDb.

Works symmetrically for movies and tv. For tv, TMDb returns a raw
tmdb_id; the ingestor applies +TV_OFFSET before storing in HotCache /
UniversalMediaItem so the id matches what the parquet metadata uses.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

try:
    from .compute_embeddings import build_text_for_embedding
    from .faiss_bridge import TV_OFFSET, FaissCatalog
except ImportError:
    from compute_embeddings import build_text_for_embedding
    from faiss_bridge import TV_OFFSET, FaissCatalog

if TYPE_CHECKING:
    from universal_search import (
        HotCache,
        TMDBLiveClient,
        UniversalMediaItem,
    )

logger = logging.getLogger(__name__)


class ColdStartIngestor:
    """
    Orchestrates cold-start for unknown tmdb_ids across both domains.

    One instance per running bot: shares a single SBERT encoder (lazy-
    loaded on first call), a single TMDb client, and the single FAISS
    catalog. Routes HotCache writes to the correct per-domain cache by
    media_type.
    """

    def __init__(
        self,
        tmdb_client: "TMDBLiveClient",
        faiss_catalog: FaissCatalog,
        movies_hot_cache: "HotCache",
        tv_hot_cache: "HotCache",
        faiss_index_path: Path,
        faiss_meta_path: Path,
        model_name: Optional[str] = None,
    ):
        self.tmdb = tmdb_client
        self.faiss = faiss_catalog
        self.movies_hot_cache = movies_hot_cache
        self.tv_hot_cache = tv_hot_cache
        self.faiss_index_path = Path(faiss_index_path)
        self.faiss_meta_path = Path(faiss_meta_path)
        self.model_name = model_name or faiss_catalog.model_name
        self._encoder = None  # lazy

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"ColdStartIngestor: loading {self.model_name} on {device}")
            self._encoder = SentenceTransformer(self.model_name, device=device)
        return self._encoder

    def _hot_cache_for(self, media_type: str) -> "HotCache":
        if media_type == "movie":
            return self.movies_hot_cache
        if media_type == "tv":
            return self.tv_hot_cache
        raise ValueError(f"media_type must be 'movie' or 'tv', got {media_type!r}")

    @staticmethod
    def _canonical_tmdb(tmdb_id: int, media_type: str) -> tuple[int, int]:
        """
        Return (raw_tmdb, stored_tmdb).

        raw_tmdb   — what TMDb API expects in /{mt}/{id} (no offset)
        stored_tmdb — what we write to HotCache / UniversalMediaItem.tmdb_id
                     (tv has +TV_OFFSET baked in to match parquet metadata)

        Accepts either form on input: a tv id may come in with or without
        the offset already applied.
        """
        tid = int(tmdb_id)
        if media_type == "movie":
            return tid, tid
        if media_type == "tv":
            if tid >= TV_OFFSET:
                return tid - TV_OFFSET, tid
            return tid, tid + TV_OFFSET
        raise ValueError(f"media_type must be 'movie' or 'tv', got {media_type!r}")

    def ensure(
        self,
        tmdb_id: int,
        media_type: str,
    ) -> Optional["UniversalMediaItem"]:
        """
        Guarantee that (tmdb_id, media_type) is in FAISS and HotCache.

        Returns the resulting UniversalMediaItem, or None when TMDb has
        no record for this id (404 or transient failure). Idempotent:
        repeated calls for the same id don't re-hit TMDb and don't
        re-persist FAISS.
        """
        raw_tmdb, stored_tmdb = self._canonical_tmdb(tmdb_id, media_type)
        fid = FaissCatalog._faiss_id(raw_tmdb, media_type)
        hot_cache = self._hot_cache_for(media_type)

        if fid in self.faiss.mapping:
            cached = hot_cache.get_by_tmdb_id(stored_tmdb)
            if cached is not None:
                return cached
            # FAISS knows it, HotCache doesn't — rare, but keep going
            # to repopulate the cache without re-persisting FAISS.

        # Prefer HotCache when it already has this item — /search_multi
        # populates it on-line, so we can skip a duplicate TMDb call.
        item = hot_cache.get_by_tmdb_id(stored_tmdb)
        if item is None:
            item = self.tmdb.get_details(raw_tmdb, media_type)
            if item is None:
                logger.info(
                    f"ColdStart: tmdb returned no details for {media_type} id={raw_tmdb}"
                )
                return None

        # Normalize in-place: ensure media_type and tmdb_id match our
        # on-disk conventions regardless of where `item` came from.
        item.media_type = media_type
        item.tmdb_id = stored_tmdb
        item.source = "tmdb_live"
        item.has_embeddings = False
        item.in_graph = False

        hot_cache.upsert_batch([item])

        if fid not in self.faiss.mapping:
            text = build_text_for_embedding(
                {
                    "overview": item.overview,
                    "overview_ru": getattr(item, "overview_ru", None),
                    "genres": item.genres,
                    "title": item.title,
                }
            )
            if not text.strip():
                text = "Unknown media"
            encoder = self._get_encoder()
            vec = encoder.encode(text, normalize_embeddings=True)
            self.faiss.add(raw_tmdb, media_type, np.asarray(vec, dtype=np.float32))
            self.faiss.persist(self.faiss_index_path, self.faiss_meta_path)

        return item
