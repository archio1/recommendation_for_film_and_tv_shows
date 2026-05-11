"""
faiss_bridge.py — единый FAISS-каталог для cross-domain content-search.

Хранит эмбеддинги всех items (movies ∪ tv ∪ TMDb-cache) в одном индексе.
Поиск возвращает (tmdb_id, media_type, score). Используется как:
  - content-bridge для /recs_cross (похожее в другом домене)
  - cold-start для новых items через TMDb + SBERT-upsert
  - fallback для items, которых нет в LightGCN
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np


TV_OFFSET = 10_000_000

DEFAULT_EMBEDDING_DIM = 384
DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class FaissSearchResult:
    tmdb_id: int
    media_type: str
    score: float


class FaissCatalog:
    """
    FAISS-индекс над нормализованными эмбеддингами + словарь метаданных.

    Индекс: IndexIDMap2(IndexFlatIP) — cosine через dot product (эмбеддинги
    пре-нормализованы), поддерживает remove/upsert по ID. На 10-20k items
    FlatIP даёт exact search без компромиссов; если каталог вырастет >100k —
    заменить на IndexIVFFlat без изменения API.

    faiss_id = tmdb_id (для movie) или tmdb_id + TV_OFFSET (для tv).
    Использование смещённого tmdb_id в качестве faiss_id делает пространство
    ID сквозным и согласованным с тем, что уже хранится в parquet-метаданных.
    """

    def __init__(
        self,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        base = faiss.IndexFlatIP(embedding_dim)
        self.index = faiss.IndexIDMap2(base)
        self.mapping: dict[int, tuple[int, str]] = {}

    @property
    def size(self) -> int:
        return self.index.ntotal

    @staticmethod
    def _faiss_id(tmdb_id: int, media_type: str) -> int:
        if media_type == "movie":
            return int(tmdb_id)
        if media_type == "tv":
            tid = int(tmdb_id)
            return tid if tid >= TV_OFFSET else tid + TV_OFFSET
        raise ValueError(f"media_type must be 'movie' or 'tv', got {media_type!r}")

    @staticmethod
    def _strip_offset(faiss_id: int, media_type: str) -> int:
        if media_type == "tv" and faiss_id >= TV_OFFSET:
            return faiss_id - TV_OFFSET
        return faiss_id

    @staticmethod
    def _prepare(embedding: np.ndarray, expected_dim: int) -> np.ndarray:
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if emb.shape[0] != expected_dim:
            raise ValueError(
                f"embedding dim {emb.shape[0]} != expected {expected_dim}"
            )
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        return emb.reshape(1, -1)

    def add(self, tmdb_id: int, media_type: str, embedding: np.ndarray) -> int:
        """Add or upsert an item. Returns faiss_id."""
        fid = self._faiss_id(tmdb_id, media_type)
        emb = self._prepare(embedding, self.embedding_dim)

        if fid in self.mapping:
            self.index.remove_ids(np.array([fid], dtype=np.int64))
        self.index.add_with_ids(emb, np.array([fid], dtype=np.int64))
        self.mapping[fid] = (self._strip_offset(fid, media_type), media_type)
        return fid

    def add_batch(
        self,
        items: Iterable[tuple[int, str, np.ndarray]],
    ) -> int:
        """
        Bulk add. Faster than repeated `add` when loading the full catalog
        (single add_with_ids call instead of N).

        Items already present are upserted (removed + re-added).
        """
        items = list(items)
        if not items:
            return 0

        to_remove = [
            self._faiss_id(tid, mt)
            for tid, mt, _ in items
            if self._faiss_id(tid, mt) in self.mapping
        ]
        if to_remove:
            self.index.remove_ids(np.array(to_remove, dtype=np.int64))

        embs = np.vstack([self._prepare(e, self.embedding_dim) for _, _, e in items])
        fids = np.array(
            [self._faiss_id(tid, mt) for tid, mt, _ in items],
            dtype=np.int64,
        )
        self.index.add_with_ids(embs, fids)

        for (tid, mt, _), fid in zip(items, fids):
            self.mapping[int(fid)] = (self._strip_offset(int(fid), mt), mt)
        return len(items)

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        media_type_filter: str | None = None,
    ) -> list[FaissSearchResult]:
        """
        Return top_k nearest items by cosine. If `media_type_filter` is set,
        results are filtered to that type (oversamples internally, so you
        still get top_k unless the catalog is smaller than that).
        """
        if self.size == 0:
            return []

        if media_type_filter is not None and media_type_filter not in {"movie", "tv"}:
            raise ValueError(
                f"media_type_filter must be 'movie', 'tv' or None, "
                f"got {media_type_filter!r}"
            )

        q = self._prepare(query, self.embedding_dim)
        k = min(top_k * 5 if media_type_filter else top_k, self.size)
        scores, ids = self.index.search(q, k)

        out: list[FaissSearchResult] = []
        for score, fid in zip(scores[0], ids[0]):
            if fid == -1:
                continue
            entry = self.mapping.get(int(fid))
            if entry is None:
                continue
            tmdb_id, media_type = entry
            if media_type_filter and media_type != media_type_filter:
                continue
            out.append(
                FaissSearchResult(
                    tmdb_id=tmdb_id,
                    media_type=media_type,
                    score=float(score),
                )
            )
            if len(out) >= top_k:
                break
        return out

    def persist(self, index_path: Path, meta_path: Path) -> None:
        index_path = Path(index_path)
        meta_path = Path(meta_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))

        meta = {
            "embedding_dim": self.embedding_dim,
            "model_name": self.model_name,
            "num_items": self.size,
            "tv_offset": TV_OFFSET,
            "mapping": {
                str(fid): {"tmdb_id": tid, "media_type": mt}
                for fid, (tid, mt) in self.mapping.items()
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, index_path: Path, meta_path: Path) -> "FaissCatalog":
        index_path = Path(index_path)
        meta_path = Path(meta_path)
        meta = json.loads(meta_path.read_text())

        cat = cls(
            embedding_dim=meta["embedding_dim"],
            model_name=meta.get("model_name", DEFAULT_MODEL_NAME),
        )
        cat.index = faiss.read_index(str(index_path))
        cat.mapping = {
            int(fid): (int(entry["tmdb_id"]), entry["media_type"])
            for fid, entry in meta["mapping"].items()
        }
        if cat.index.ntotal != len(cat.mapping):
            raise RuntimeError(
                f"FAISS index size {cat.index.ntotal} != mapping size "
                f"{len(cat.mapping)} — index and meta are out of sync"
            )
        return cat
