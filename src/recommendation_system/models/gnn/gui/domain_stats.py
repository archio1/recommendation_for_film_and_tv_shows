"""Сбор статистики по датасетам и моделям (общий слой для DataTab + InferenceTab)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from recommendation_system.paths import PROJECT_ROOT


@dataclass
class DomainStats:
    domain: str
    dataset_dir: Path
    models_dir: Path
    dataset_exists: bool
    interactions_mtime: Optional[datetime] = None
    items_mtime: Optional[datetime] = None
    num_users: Optional[int] = None
    num_items: Optional[int] = None
    num_interactions: Optional[int] = None
    last_train_at: Optional[str] = None
    last_train_recall: Optional[float] = None
    last_train_checkpoint: Optional[str] = None
    last_train_sanity_ok: Optional[bool] = None
    model_mtime: Optional[datetime] = None
    model_size_mb: Optional[float] = None


def _collect_dataset_stats(dataset_dir: Path) -> Optional[dict]:
    """Read interactions/items/id_mapping from any folder.
    Returns None if any of the three required files is missing."""
    interactions_path = dataset_dir / "interactions_final.parquet"
    items_path = dataset_dir / "items_metadata_final.parquet"
    mapping_path = dataset_dir / "id_mapping.json"

    if not (interactions_path.exists() and items_path.exists() and mapping_path.exists()):
        return None

    interactions_mtime = datetime.fromtimestamp(interactions_path.stat().st_mtime)
    items_mtime = datetime.fromtimestamp(items_path.stat().st_mtime)

    num_users = None
    num_items = None
    try:
        with open(mapping_path, encoding="utf-8") as f:
            mapping = json.load(f)
        num_users = mapping.get("num_users")
        # Prefer total catalog count; fall back to trained subset.
        num_items = mapping.get("num_items") or mapping.get("num_trained_items")
    except (json.JSONDecodeError, OSError):
        pass

    num_interactions = None
    try:
        import pyarrow.parquet as pq
        num_interactions = pq.ParquetFile(interactions_path).metadata.num_rows
    except Exception:
        try:
            import pandas as pd
            num_interactions = len(pd.read_parquet(interactions_path, columns=["user_id"]))
        except Exception:
            pass

    return {
        "interactions_mtime": interactions_mtime,
        "items_mtime": items_mtime,
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions": num_interactions,
    }


def _collect_model_stats(model_path: Path) -> dict:
    """Read sidecar .json next to a .pt checkpoint. If sidecar is missing,
    return only filename + mtime + size_mb. Always populates last_train_checkpoint."""
    result: dict = {
        "last_train_checkpoint": model_path.name,
        "model_mtime": datetime.fromtimestamp(model_path.stat().st_mtime),
        "model_size_mb": model_path.stat().st_size / (1024 * 1024),
    }

    sidecar = model_path.with_suffix(".json")
    if not sidecar.exists():
        return result

    try:
        with open(sidecar, encoding="utf-8") as f:
            s = json.load(f)
    except (json.JSONDecodeError, OSError):
        return result

    result["last_train_at"] = s.get("trained_at")
    metrics = s.get("metrics") or {}
    recall = metrics.get("recall@10")
    if isinstance(recall, (int, float)):
        result["last_train_recall"] = float(recall)
    # Sidecar's checkpoint name wins if present (handles renamed .pt files).
    sidecar_ckpt = s.get("checkpoint")
    if sidecar_ckpt:
        result["last_train_checkpoint"] = sidecar_ckpt
    sanity = s.get("sanity_check") or {}
    sanity_passed = sanity.get("passed")
    if isinstance(sanity_passed, bool):
        result["last_train_sanity_ok"] = sanity_passed
    return result


def _find_latest_model(models_dir: Path, domain: str) -> Optional[Path]:
    """Find the most recent .pt in models_dir. If a sidecar exists and its
    domain mismatches, skip; .pt files without a sidecar are still considered
    (covers old Colab v4 checkpoints)."""
    if not models_dir.exists():
        return None

    candidates: list[Path] = []
    for pt in models_dir.glob("*.pt"):
        sidecar = pt.with_suffix(".json")
        if sidecar.exists():
            try:
                with open(sidecar, encoding="utf-8") as f:
                    s = json.load(f)
                if s.get("domain") and s.get("domain") != domain:
                    continue
            except (json.JSONDecodeError, OSError):
                pass
        candidates.append(pt)

    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _collect_domain_stats(
    domain: str,
    dataset_override: Optional[Path] = None,
    model_override: Optional[Path] = None,
) -> DomainStats:
    dataset_dir = dataset_override or (PROJECT_ROOT / "data" / "processed" / domain)
    models_dir = PROJECT_ROOT / "models" / domain

    ds = _collect_dataset_stats(dataset_dir)

    if model_override and model_override.exists():
        ms = _collect_model_stats(model_override)
    else:
        latest = _find_latest_model(models_dir, domain)
        ms = _collect_model_stats(latest) if latest else {}

    return DomainStats(
        domain=domain,
        dataset_dir=dataset_dir,
        models_dir=models_dir,
        dataset_exists=ds is not None,
        interactions_mtime=ds.get("interactions_mtime") if ds else None,
        items_mtime=ds.get("items_mtime") if ds else None,
        num_users=ds.get("num_users") if ds else None,
        num_items=ds.get("num_items") if ds else None,
        num_interactions=ds.get("num_interactions") if ds else None,
        last_train_at=ms.get("last_train_at"),
        last_train_recall=ms.get("last_train_recall"),
        last_train_checkpoint=ms.get("last_train_checkpoint"),
        last_train_sanity_ok=ms.get("last_train_sanity_ok"),
        model_mtime=ms.get("model_mtime"),
        model_size_mb=ms.get("model_size_mb"),
    )
