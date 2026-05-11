"""
Sanity tests on the real trained checkpoints.

These are integration tests, not unit tests: they actually load
`models/movies/lightgcn_movies_best_v4.pt` and `models/tv/lightgcn_tv_best_v4.pt`,
plus the persisted FAISS catalog. Each fixture skips cleanly if the
artifact is missing, so a clean checkout still passes.

What we check (in DS terms — see plan):
- **dimension consistency**  — checkpoint shapes match `id_mapping.json`
- **embedding health**       — no NaN / Inf, no all-zero rows, finite norms
- **end-to-end inference**   — a popular liked item produces non-empty,
                                correctly-typed recommendations
- **domain isolation**       — `recs_movie` never returns tv (and vice versa)
- **FAISS coverage**         — persisted catalog actually contains both
                                domains, not just one
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Stable, well-known tmdb_ids that should always be in the trained graph.
# Movie ids are raw; TV ids carry the +TV_OFFSET (10_000_000) baked in.
INCEPTION_TMDB = 27205
FRIENDS_TMDB = 10_001_668


# --------------------------------------------------------------------------
# Checkpoint shapes match id_mapping.json
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "domain,ckpt_path,mapping_path",
    [
        (
            "movies",
            PROJECT_ROOT / "models" / "movies" / "lightgcn_movies_best_v4.pt",
            PROJECT_ROOT / "data" / "processed" / "movies" / "id_mapping.json",
        ),
        (
            "tv",
            PROJECT_ROOT / "models" / "tv" / "lightgcn_tv_best_v4.pt",
            PROJECT_ROOT / "data" / "processed" / "tv" / "id_mapping.json",
        ),
    ],
)
def test_checkpoint_shapes_match_id_mapping(domain, ckpt_path, mapping_path):
    if not ckpt_path.exists() or not mapping_path.exists():
        pytest.skip(f"missing artifact for {domain}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    user_w = state["user_embedding.weight"]
    item_w = state["item_id_embedding.weight"]

    with open(mapping_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert user_w.shape[0] == meta["num_users"], (
        f"{domain}: checkpoint num_users {user_w.shape[0]} "
        f"!= id_mapping.num_users {meta['num_users']}"
    )
    assert item_w.shape[0] == meta["num_trained_items"], (
        f"{domain}: checkpoint num_items {item_w.shape[0]} "
        f"!= id_mapping.num_trained_items {meta['num_trained_items']}"
    )
    assert user_w.shape[1] == item_w.shape[1], (
        "user and item embedding dims differ in checkpoint"
    )


# --------------------------------------------------------------------------
# Embedding health
# --------------------------------------------------------------------------

def _embeddings_healthy(emb: torch.Tensor) -> None:
    arr = emb.detach().cpu().numpy()
    assert np.isfinite(arr).all(), "non-finite values in embeddings"
    norms = np.linalg.norm(arr, axis=1)
    assert (norms > 0).all(), (
        f"{(norms == 0).sum()} embedding rows are all-zero"
    )


def test_movies_item_embeddings_healthy(movies_engine_real):
    emb = movies_engine_real.inference_engine.model.get_item_embedding(
        movies_engine_real.inference_engine.item_features
    )
    _embeddings_healthy(emb)


def test_tv_item_embeddings_healthy(tv_engine_real):
    emb = tv_engine_real.inference_engine.model.get_item_embedding(
        tv_engine_real.inference_engine.item_features
    )
    _embeddings_healthy(emb)


# --------------------------------------------------------------------------
# End-to-end inference returns sensible results
# --------------------------------------------------------------------------

def test_movies_engine_recs_for_known_popular(dual_engine_real):
    """A famously-popular liked movie must yield a non-trivial movie list."""
    if INCEPTION_TMDB not in dual_engine_real.movies.tmdb_to_item_id:
        pytest.skip(f"Inception ({INCEPTION_TMDB}) not in movies graph")
    recs = dual_engine_real.recs_movie(liked_tmdb_ids=[INCEPTION_TMDB], top_k=10)
    assert len(recs) >= 5, f"expected >=5 recs, got {len(recs)}"
    assert all(r.media_type == "movie" for r in recs)
    assert INCEPTION_TMDB not in {r.tmdb_id for r in recs}, (
        "the liked item must not appear in its own recommendations"
    )


def test_tv_engine_recs_for_known_popular(dual_engine_real):
    """A famously-popular liked tv show must yield a non-trivial tv list."""
    if FRIENDS_TMDB not in dual_engine_real.tv.tmdb_to_item_id:
        pytest.skip(f"Friends ({FRIENDS_TMDB}) not in tv graph")
    recs = dual_engine_real.recs_tv(liked_tmdb_ids=[FRIENDS_TMDB], top_k=10)
    assert len(recs) >= 5, f"expected >=5 recs, got {len(recs)}"
    assert all(r.media_type == "tv" for r in recs)
    assert FRIENDS_TMDB not in {r.tmdb_id for r in recs}


# --------------------------------------------------------------------------
# Domain isolation
# --------------------------------------------------------------------------

def test_recs_movie_never_returns_tv(dual_engine_real):
    if INCEPTION_TMDB not in dual_engine_real.movies.tmdb_to_item_id:
        pytest.skip("Inception not in movies graph")
    recs = dual_engine_real.recs_movie(liked_tmdb_ids=[INCEPTION_TMDB], top_k=20)
    bad = [r for r in recs if r.media_type != "movie"]
    assert not bad, f"recs_movie leaked tv: {[(r.title, r.media_type) for r in bad]}"


def test_recs_tv_never_returns_movies(dual_engine_real):
    if FRIENDS_TMDB not in dual_engine_real.tv.tmdb_to_item_id:
        pytest.skip("Friends not in tv graph")
    recs = dual_engine_real.recs_tv(liked_tmdb_ids=[FRIENDS_TMDB], top_k=20)
    bad = [r for r in recs if r.media_type != "tv"]
    assert not bad, f"recs_tv leaked movies: {[(r.title, r.media_type) for r in bad]}"


# --------------------------------------------------------------------------
# FAISS catalog covers both domains
# --------------------------------------------------------------------------

def test_faiss_catalog_has_both_domains(faiss_catalog_real):
    movies = sum(1 for _, mt in faiss_catalog_real.mapping.values() if mt == "movie")
    tv = sum(1 for _, mt in faiss_catalog_real.mapping.values() if mt == "tv")
    assert movies > 0, "FAISS catalog has no movie items"
    assert tv > 0, "FAISS catalog has no tv items"
    # Sanity bounds: should match parquet ballparks (movies » tv).
    assert movies > tv, (
        f"unexpectedly tv >= movies in FAISS ({movies=}, {tv=})"
    )


def test_faiss_catalog_size_matches_mapping(faiss_catalog_real):
    """The on-disk index and the JSON mapping must agree on size."""
    assert faiss_catalog_real.size == len(faiss_catalog_real.mapping)
