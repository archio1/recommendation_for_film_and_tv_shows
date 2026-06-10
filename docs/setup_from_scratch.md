# Setup from Scratch — Build All Artifacts, Then Run

> **Just want the commands?** The copy-paste sequence is in
> [README → Quick start](../README.md#quick-start). This guide explains what each
> step does and what it produces.

This guide takes a **clean checkout** all the way to a running bot / GUI. It is the
end-to-end, ordered version of the scattered commands in the other docs: what to run,
**in what order**, what each step produces, and which files are *libraries* you never
run by hand.

Trained models, processed parquet, the FAISS catalog, and SQLite caches all live
**outside git** — a fresh clone has none of them. You generate them here.

> **Already have artifacts?** If `models/`, `data/processed/`, and
> `faiss_index/catalog.faiss` are already populated, skip to [step 6](#6-run).
> See [README → Getting started](../README.md#getting-started) for the short path.

---

## How commands are invoked

One style for everything. Install the package editable once:

```powershell
pip install -e .      # or: uv sync
```

This also installs short console commands into the environment — they are the
canonical way to run everything:

| Command | Module behind it |
|---|---|
| `recsys-dataset` | `recommendation_system.data.make_dataset` |
| `recsys-train` | `recommendation_system.models.gnn.trainer` |
| `recsys-embed` | `recommendation_system.models.gnn.compute_embeddings` |
| `recsys-bot` | `recommendation_system.models.gnn.movie_bot` |
| `recsys-gui` | `recommendation_system.models.gnn.trainer_gui` |
| `recsys-fill-cache` | `recommendation_system.models.gnn.fill_cache` |
| `recsys-backfill` | `recommendation_system.data.backfill_translations` |
| `recsys-check-data` | `recommendation_system.data.check_data` |
| `recsys-trakt` | `recommendation_system.data.trakt_collector` |

The long form `python -m recommendation_system.…` works identically.

All filesystem paths are resolved by `recommendation_system/paths.py` (it finds the
repo root by walking up to `pyproject.toml`), so commands work from any current
directory. To point data/models/reports somewhere else, set `RECSYS_PROJECT_ROOT`.

> PyCharm note: run configurations must use **module** mode
> (`recommendation_system.models.gnn.movie_bot`), not script-path mode —
> running the `.py` files directly is no longer supported.

### Linux / WSL notes

Everything except the GUI runs as-is (paths are `pathlib`-based; torch and
faiss-cpu ship Linux wheels; the Windows console-emoji issues don't exist there).
Specifics:

- `recsys-gui` (Flet desktop) needs a display: on WSL2 that's WSLg (Windows 11)
  plus system libs — `sudo apt install libgtk-3-0 libmpv2` (older distros: `libmpv1`).
- On Linux `pip install -e .` pulls the **CUDA** torch build by default (several GB).
  For a lean CPU-only install run
  `pip install torch --index-url https://download.pytorch.org/whl/cpu` first.
- GPU **training** works in WSL2 out of the box — only the Windows-side NVIDIA
  driver is required.

---

## Pipeline at a glance

```
raw data ──▶ make_dataset ──▶ trainer (×2 domains) ──▶ compute_embeddings --to-faiss ──▶ [fill_cache] ──▶ bot / GUI
 (1)            (2)                  (3)                        (4)                          (5)            (6)
```

| # | Command | Produces |
|---|---------|----------|
| 1 | (manual download) + optional `recsys-trakt` | `data/raw/…` raw inputs |
| 2 | `recsys-dataset --domain all` | `data/processed/{movies,tv}/*_final.parquet` + `id_mapping.json` |
| 3 | `recsys-train --domain movies` and `--domain tv` | `models/{domain}/lightgcn_{domain}_best_v{N}.pt` (+ sidecar `.json`) |
| 4 | `recsys-embed --domain all --to-faiss` | per-domain `overview_embeddings.npy` **and** the shared FAISS catalog |
| 5 | *(optional)* `recsys-fill-cache` | warmed translation cache under `data/processed/cache/` |
| 6 | `recsys-bot` / `recsys-gui` | the running app |

---

## 0. Prerequisites

```
# .env in the repo root
TMDB_API_KEY=...          # cold-start, translations, live search
TELEGRAM_BOT_TOKEN=...    # only needed to run the Telegram bot
```

```powershell
pip install -e .     # or: uv sync — installs dependencies AND the package itself
```

Python ≥ 3.10 (developed on 3.13).

---

## 1. Get raw data

Put the source files under `data/raw/`. Which sources each model needs, where to
download them, and the exact CSV schemas are in
**[data_sources.md](data_sources.md)** — follow it first.

- **Movies** require **MovieLens 32M** (structural dependency via `links.csv`).
- **TV** requires `trakt_shows.csv` + `trakt_interactions.csv`. Either collect them
  yourself (multi-day, resumable) or supply any CSV matching the schema contract:

  ```powershell
  recsys-trakt
  ```

  Checkpoints in `data/raw/trakt_collector.db` — safe to interrupt and resume.
- TMDB metadata and Amazon Reviews are optional enrichment.

---

## 2. Build the datasets

```powershell
recsys-dataset --domain all   # movies + tv
# or per domain: --domain movies   /   --domain tv
```

Writes, per domain, to `data/processed/{movies,tv}/`:

- `interactions_final.parquet` — `user_id, item_id, rating, timestamp` (trained items only)
- `items_metadata_final.parquet` — catalog (title/genres/overview/vote_count/… ; TV
  ids carry the `+10_000_000` offset)
- `id_mapping.json` — user/item counts and the `tmdb → item_id` map

You can also do this from the GUI **Dataset** tab — see
[trainer_gui_guide.md](trainer_gui_guide.md).

---

## 3. Train the two models

Two independent LightGCN models, one per domain:

```powershell
recsys-train --domain movies --epochs 30
recsys-train --domain tv     --epochs 30
```

Each writes an **auto-versioned** checkpoint `models/{domain}/lightgcn_{domain}_best_v{N}.pt`
plus a sidecar `.json` (metrics, dataset shape, hyperparameters). Older versions are
kept, not overwritten. The bot/GUI pick up the latest via the sidecar chain (bot
default fallback: `*_best_v4.pt`). Training is also available in the GUI **Train**
tab with live metrics.

---

## 4. Compute embeddings and build the FAISS catalog

This is the step that **creates the FAISS content-bridge** — the cross-domain /
cold-start index. The `--to-faiss` flag is what actually populates the catalog;
without it you only get the `.npy` files.

```powershell
recsys-embed --domain all --to-faiss
```

Produces:

- `data/processed/{movies,tv}/overview_embeddings.npy` (+ `embedding_meta.json`)
- `src/recommendation_system/faiss_index/catalog.faiss`
- `src/recommendation_system/faiss_index/catalog_meta.json`

Re-run after any dataset rebuild if you want the bridge to reflect new titles. It is
idempotent (upsert by id).

---

## 5. (Optional) Warm the translation cache

```powershell
recsys-fill-cache
```

Pre-fetches RU/UK TMDb translations into the SQLite cache under
`data/processed/cache/` so the first user requests don't pay live API latency. You
can **skip this** — cold-start fills the cache lazily on demand (see below).

> `recsys-backfill --lang ru|uk --domain movies|tv` is a separate, resumable way
> to add RU/UK columns directly into the metadata parquet. `scripts/admin_users.py`
> is a read-only viewer for the bot's user-session DB.

---

## 6. Run

```powershell
# Telegram bot
recsys-bot

# or the desktop admin GUI (dataset build / train / inference)
recsys-gui
```

The GUI **Inference** tab has a **Popularity de-bias** toggle (default **on**, λ=0.5
for movies) so its recommendations match the bot; turn it off to inspect the raw
LightGCN ranking. TV is never de-biased (its small catalog doesn't collapse). In the
bot, each user controls the same de-bias with **`/debias`** (default on, persisted
per user).

### Artifact checklist before launch

```
data/processed/movies/{interactions_final,items_metadata_final}.parquet  +  id_mapping.json
data/processed/tv/{interactions_final,items_metadata_final}.parquet       +  id_mapping.json
models/movies/lightgcn_movies_best_v*.pt   (+ sidecar .json)
models/tv/lightgcn_tv_best_v*.pt           (+ sidecar .json)
data/processed/{movies,tv}/overview_embeddings.npy
src/recommendation_system/faiss_index/catalog.faiss  +  catalog_meta.json
```

---

## What you never run by hand

Two modules look like scripts but are **libraries** — there is no setup step for them:

- **`faiss_bridge.py`** — the `FaissCatalog` class (add / search / persist / load).
  The catalog file is *built* by `compute_embeddings.py --to-faiss` (step 4) and
  *read* by the bot, GUI, and cold-start. You never invoke `faiss_bridge.py`.
- **`cold_start.py`** — the live ingestor. When a user mentions a title that isn't in
  any graph, it runs **per request** inside the bot/GUI: fetch from TMDb → cache in
  SQLite → encode with SBERT → upsert into the FAISS catalog. There is no batch
  "cold-start" command to run.

---

## See also

- [data_sources.md](data_sources.md) — raw inputs, where to get them, CSV schemas
- [trainer_gui_guide.md](trainer_gui_guide.md) — the GUI alternative to the CLI
- [ARCHITECTURE.md](../ARCHITECTURE.md) — how the pieces fit together
