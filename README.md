# Film & TV Recommendation System

A real-time, **dual-domain** recommender for movies and TV shows built on
**LightGCN** graph neural networks, with a **FAISS** content-bridge for
cross-domain suggestions and cold-start, served through a trilingual
(RU / UK / EN) **Telegram bot**.

> **Status:** v0.9 — two-model architecture in production; local training (CLI +
> desktop GUI) implemented. Trained on ~2M real interactions.

---

## Core idea

Two **independent LightGCN models** — one for movies, one for TV — each with its
own user/item embedding space. Keeping the domains separate preserves signal
quality inside each (a movie graph and a TV graph have very different structure),
while a shared **FAISS index over multilingual text embeddings** bridges them for
cross-domain recommendations ("you liked this show → here's a film") and
cold-start items that aren't in either graph yet.

```mermaid
graph LR
    subgraph Data
        ML[MovieLens 32M]
        AMZ[Amazon Reviews 2023]
        TRK[Trakt.tv ~2M ratings]
        TMDB[TMDb API]
    end
    Data --> ETL[make_dataset.py]
    ETL --> PM[(processed/movies)]
    ETL --> PT[(processed/tv)]
    PM --> LGM[LightGCN · movies]
    PT --> LGT[LightGCN · tv]
    LGM --> DDE[DualDomainEngine]
    LGT --> DDE
    FAISS[(FAISS content-bridge)] --> DDE
    TMDB --> CS[Cold-start ingestor] --> FAISS
    DDE --> BOT[Telegram bot]
    DDE --> GUI[Admin GUI]
```

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the full design and data flow.

---

## Features

- **Graph-based recommendations** — LightGCN (PyTorch Geometric) learns
  collaborative signal from a bipartite user–item graph per domain.
- **Cross-domain bridge** — FAISS `IndexFlatIP` over normalized SBERT
  (`paraphrase-multilingual-MiniLM-L12-v2`) embeddings links movies ↔ TV.
- **Cold-start** — unknown titles are fetched live from TMDb, cached in SQLite,
  encoded, and upserted into the FAISS catalog on the fly.
- **Hybrid search** — semantic (SBERT) + keyword/genre (IDF) + title
  normalization (e.g. `Se7en` ↔ `seven`, Cyrillic queries), independent of the graph.
- **Popularity de-bias** — a tunable penalty keeps the movie list from
  collapsing into the global IMDb top-250 (reflects "popular among users like
  you", not "popular overall"). Toggleable per user: `/debias` in the bot, a
  switch in the admin GUI.
- **Trilingual** — RU / UK / EN interface and per-user language preference.
- **Two ways to drive it** — an Aiogram Telegram bot for end users, and a Flet
  desktop admin GUI for dataset building, local training, and offline testing.

---

## Tech stack

| Layer | Tech |
|---|---|
| Models | PyTorch, PyTorch Geometric — two LightGCN (hybrid ID + content embeddings) |
| Content-bridge | FAISS `IndexIDMap2(IndexFlatIP)` on multilingual SBERT vectors |
| Data | MovieLens 32M · Amazon Reviews 2023 · Trakt.tv API · TMDb API |
| Search | SBERT (semantic) + IDF (keywords/genres) + title normalizer |
| Bot | Aiogram (Telegram) + SQLite session store |
| Admin GUI | Flet (desktop) |

---

## Repository structure

```
src/recommendation_system/
├── data/                 # ETL: MovieLens / Amazon / Trakt → processed parquet
│   ├── make_dataset.py   #   per-domain dataset builders (--domain movies|tv|all)
│   └── trakt_collector.py#   multi-phase Trakt.tv scraper (checkpoint + rate-limit)
├── models/gnn/           # the recommendation engine
│   ├── lightgcn.py       #   LightGCN model (graph conv, BPR + InfoNCE losses)
│   ├── trainer.py        #   training CLI (--domain, --epochs, …)
│   ├── inference_engine.py#  per-domain inference + popularity de-bias
│   ├── universal_search.py# hybrid search, intent clustering, media-DNA, hot cache
│   ├── dual_domain_engine.py# router over both models + FAISS bridge
│   ├── faiss_bridge.py   #   FAISS catalog (add / search / persist)
│   ├── cold_start.py     #   TMDb → cache → SBERT → FAISS ingestion
│   ├── compute_embeddings.py # SBERT embedding + FAISS build CLI
│   ├── movie_bot.py      #   Telegram bot (Aiogram)
│   ├── bot/              #   bot helpers (i18n, session store, search, keyboards)
│   └── gui/              #   Flet admin GUI (training / dataset / inference / data tabs)
├── data/ (processed)     # generated parquet + id mappings (git-ignored)
└── ...
tests/                    # pytest suite (unit + behavioral quality gates)
docs/                     # data_sources.md, trainer_gui_guide.md, …
```

---

## Getting started

> Requires Python 3.13, a TMDb API key (for cold-start/translations), and a
> Telegram bot token (to run the bot). Trained model artifacts and processed
> data live outside the repo.

```bash
pip install -r requirements.txt          # or: uv sync

# configure secrets in .env
TMDB_API_KEY=...
TELEGRAM_BOT_TOKEN=...

# run the Telegram bot — run the file directly (its modules use sibling imports,
# so the `-m` form does not resolve them)
python src/recommendation_system/models/gnn/movie_bot.py

# or launch the desktop admin GUI (dataset build / train / test)
# PowerShell: PYTHONPATH must include src/
$env:PYTHONPATH = "src"; python -m recommendation_system.models.gnn.trainer_gui
```

> **GPU note:** dependencies (incl. `torch` + PyTorch Geometric) are declared in
> `pyproject.toml`; `uv sync` / `pip install -e .` installs the **CPU** torch build,
> which is enough to serve the bot and GUI. For GPU **training**, install the CUDA
> build matching your toolkit instead, e.g.
> `uv pip install torch==2.6.0+cu124 --index https://download.pytorch.org/whl/cu124`.

Starting from a clean checkout (no models or processed data yet)? Follow
**[docs/setup_from_scratch.md](docs/setup_from_scratch.md)** — the ordered pipeline
that builds every artifact (datasets → models → FAISS catalog → caches) before you
run the bot. The admin GUI is documented in
**[docs/trainer_gui_guide.md](docs/trainer_gui_guide.md)**; the data pipeline in
**[docs/data_sources.md](docs/data_sources.md)**.

---

## Testing

```bash
pytest -q
```

The suite mixes plain unit tests with **behavioral quality gates** that run the
real models on curated scenarios (genre fidelity, popularity de-bias, graph
overlap, search quality). Tests that need model/parquet artifacts **skip
cleanly** when those are absent, so a fresh checkout stays green.

---

## Data at a glance

- **MovieLens 32M** — dense movie collaborative signal.
- **Amazon Reviews 2023** ("Movies and TV") — extra interactions, title-matched.
- **Trakt.tv** — purpose-collected TV signal: **5,020 shows × ~83k users ×
  ~1.95M ratings** (fixes the historical 99% movie / 1% TV imbalance).
- **TMDb** — metadata, posters, translations, and live cold-start.

---

## Roadmap

- [x] Dual-domain LightGCN + FAISS content-bridge
- [x] Trilingual Telegram bot + Flet admin GUI
- [x] Local training (CLI + GUI)
- [x] Popularity de-bias for the movie ranker
- [ ] Periodic retrain pipeline (`scripts/retrain.py`, see `spec/retrain-pipeline.md`)
- [ ] Taste-relative de-bias (penalize popularity relative to user profile)

---

## License

See [LICENSE](LICENSE).
