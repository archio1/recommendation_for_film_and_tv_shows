"""
compute_embeddings.py — generate semantic embeddings for catalog items.

Runs per-domain (movies / tv) so each LightGCN domain has its own
`overview_embeddings.npy` aligned with its own `item_id` space.
Optionally populates the shared FAISS content-bridge used for cross-domain
search and TMDb cold-start.

Outputs per domain (in `data/processed/{domain}/`):
  - overview_embeddings.npy  (float16, aligned with sorted item_id)
  - embedding_meta.json      (item_id → embedding index)

Optional FAISS output (with --to-faiss):
  - src/recommendation_system/faiss_index/catalog.faiss
  - src/recommendation_system/faiss_index/catalog_meta.json

Model: paraphrase-multilingual-MiniLM-L12-v2 (384-d, EN+RU).

Usage:
  python -m recommendation_system.models.gnn.compute_embeddings                    # both domains, no FAISS
  python -m recommendation_system.models.gnn.compute_embeddings --to-faiss         # both + FAISS
  python -m recommendation_system.models.gnn.compute_embeddings --domain movies    # single domain
  python -m recommendation_system.models.gnn.compute_embeddings --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from recommendation_system.paths import FAISS_DIR, PROCESSED_DIR, PROJECT_ROOT

DEFAULT_DATA_DIR = PROCESSED_DIR
DEFAULT_FAISS_DIR = FAISS_DIR
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
METADATA_FILENAME = "items_metadata_final.parquet"
EMBEDDINGS_FILENAME = "overview_embeddings.npy"
EMBEDDING_META_FILENAME = "embedding_meta.json"
FAISS_INDEX_FILENAME = "catalog.faiss"
FAISS_META_FILENAME = "catalog_meta.json"


def build_text_for_embedding(row) -> str:
    """
    Combine overview text + genres into a single sentence for SBERT.

    The model (paraphrase-multilingual-MiniLM-L12-v2) is multilingual,
    so concatenating en/ru/uk overviews gives it more signal than a
    single-language fallback chain — equivalent terms in different
    languages reinforce each other in the embedding space.

    Genres are appended when present. Falls back to title only if every
    overview field is missing/empty.
    """
    parts: list[str] = []

    for field in ("overview", "overview_ru", "overview_uk"):
        v = row.get(field)
        if pd.notna(v) and len(str(v).strip()) > 15:
            parts.append(str(v).strip())

    genres = row.get("genres", [])
    if isinstance(genres, (list, np.ndarray)) and len(genres) > 0:
        genre_str = ", ".join(str(g) for g in genres if pd.notna(g))
        if genre_str:
            parts.append(f"Genres: {genre_str}")
    elif isinstance(genres, str) and genres.strip():
        parts.append(f"Genres: {genres}")

    if not parts:
        title = row.get("title", "")
        if pd.notna(title) and str(title).strip():
            parts.append(str(title).strip())

    return " ".join(parts)


def _load_model(model_name: str, device: str, batch_size: int):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ sentence-transformers not installed.")
        print("   pip install sentence-transformers")
        sys.exit(1)

    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = "cpu"
        print("   Device: CPU")
        if batch_size > 64:
            batch_size = 64
            print(f"   Reduced batch_size to {batch_size} for CPU")

    print(f"🤖 Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    print(f"   Model loaded ({model.get_sentence_embedding_dimension()} dims)")
    return model, batch_size


def _encode_domain(
    domain: str,
    data_dir: Path,
    model,
    batch_size: int,
    model_name: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Encode a single domain. Returns (sorted metadata, float32 L2-normalized
    embeddings aligned row-by-row with the metadata).
    """
    domain_dir = data_dir / domain
    meta_path = domain_dir / METADATA_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found for domain '{domain}': {meta_path}")

    print(f"\n=== Domain: {domain} ===")
    print(f"📄 {meta_path}")

    metadata = pd.read_parquet(meta_path)
    metadata = metadata.sort_values("item_id").reset_index(drop=True)
    num_items = len(metadata)
    print(f"📊 Items: {num_items:,}")

    texts = metadata.apply(build_text_for_embedding, axis=1).tolist()
    empty_count = sum(1 for t in texts if not t.strip())
    print(f"   Text non-empty: {num_items - empty_count:,}, empty: {empty_count}")
    texts = [t if t.strip() else "Unknown media" for t in texts]

    print(f"⚡ Encoding (batch_size={batch_size})...")
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    elapsed = time.time() - start
    print(f"✅ {elapsed:.1f}s ({num_items / elapsed:.0f} items/sec)")
    print(f"   Shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    emb_path = domain_dir / EMBEDDINGS_FILENAME
    np.save(emb_path, embeddings.astype(np.float16))
    size_mb = emb_path.stat().st_size / 1e6
    print(f"💾 {emb_path.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB)")

    meta_out = {
        "domain": domain,
        "item_ids": metadata["item_id"].astype(int).tolist(),
        "num_items": num_items,
        "model": model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "computed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_out_path = domain_dir / EMBEDDING_META_FILENAME
    meta_out_path.write_text(json.dumps(meta_out, indent=2))
    print(f"💾 {meta_out_path.relative_to(PROJECT_ROOT)}")

    return metadata, embeddings.astype(np.float32)


def _sanity_check(metadata: pd.DataFrame, embeddings: np.ndarray, domain: str):
    import random

    print(f"\n🔍 Sanity check [{domain}] — nearest neighbors for 3 random items:")
    sample_indices = random.sample(range(len(metadata)), min(3, len(metadata)))
    for idx in sample_indices:
        title = metadata.iloc[idx].get("title", "?")
        sims = embeddings @ embeddings[idx]
        sims[idx] = -1.0
        top3 = np.argsort(sims)[-3:][::-1]
        neighbors = [
            f"{metadata.iloc[j].get('title', '?')} ({sims[j]:.3f})" for j in top3
        ]
        print(f"   {title} → {', '.join(neighbors)}")


def _populate_faiss(
    per_domain: dict[str, tuple[pd.DataFrame, np.ndarray]],
    faiss_dir: Path,
    model_name: str,
) -> None:
    """
    Upsert encoded items into the shared FAISS catalog. Loads the existing
    index when present (so running --domain movies doesn't drop tv items),
    creates a fresh one otherwise.
    """
    from recommendation_system.models.gnn.faiss_bridge import FaissCatalog

    index_path = faiss_dir / FAISS_INDEX_FILENAME
    meta_path = faiss_dir / FAISS_META_FILENAME

    any_df = next(iter(per_domain.values()))[1]
    embedding_dim = int(any_df.shape[1])

    if index_path.exists() and meta_path.exists():
        print(f"\n📦 Loading existing FAISS catalog: {index_path.relative_to(PROJECT_ROOT)}")
        catalog = FaissCatalog.load(index_path, meta_path)
        if catalog.embedding_dim != embedding_dim:
            raise RuntimeError(
                f"Existing FAISS catalog dim {catalog.embedding_dim} "
                f"!= new embeddings dim {embedding_dim}. Delete the old index "
                f"to rebuild from scratch."
            )
    else:
        print(f"\n📦 Creating new FAISS catalog (dim={embedding_dim})")
        catalog = FaissCatalog(embedding_dim=embedding_dim, model_name=model_name)

    for domain, (metadata, embeddings) in per_domain.items():
        media_type = "tv" if domain == "tv" else "movie"
        if "tmdb_id" not in metadata.columns:
            raise RuntimeError(f"tmdb_id column missing in {domain} metadata")

        tmdb_ids = metadata["tmdb_id"].astype(np.int64).tolist()
        items = [
            (tid, media_type, emb)
            for tid, emb in zip(tmdb_ids, embeddings)
            if tid is not None and tid > 0
        ]
        print(f"   + {domain}: upserting {len(items):,} items")
        catalog.add_batch(items)

    catalog.persist(index_path, meta_path)
    size_mb = index_path.stat().st_size / 1e6
    print(f"💾 {index_path.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB, {catalog.size:,} items)")
    print(f"💾 {meta_path.relative_to(PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Compute per-domain semantic embeddings")
    parser.add_argument(
        "--domain",
        choices=["movies", "tv", "all"],
        default="all",
        help="Which domain(s) to encode (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Processed data root (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--faiss-dir",
        type=Path,
        default=DEFAULT_FAISS_DIR,
        help=f"FAISS catalog directory (default: {DEFAULT_FAISS_DIR})",
    )
    parser.add_argument(
        "--to-faiss",
        action="store_true",
        help="Upsert encoded items into the shared FAISS catalog",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--no-sanity-check",
        action="store_true",
        help="Skip the post-encoding nearest-neighbor sanity print",
    )
    args = parser.parse_args()

    domains = ["movies", "tv"] if args.domain == "all" else [args.domain]

    for d in domains:
        if not (args.data_dir / d / METADATA_FILENAME).exists():
            print(f"❌ Metadata missing for '{d}': {args.data_dir / d / METADATA_FILENAME}")
            sys.exit(1)

    model, batch_size = _load_model(args.model, args.device, args.batch_size)

    per_domain: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for d in domains:
        metadata, embeddings = _encode_domain(d, args.data_dir, model, batch_size, args.model)
        per_domain[d] = (metadata, embeddings)
        if not args.no_sanity_check:
            _sanity_check(metadata, embeddings, d)

    if args.to_faiss:
        _populate_faiss(per_domain, args.faiss_dir, args.model)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
