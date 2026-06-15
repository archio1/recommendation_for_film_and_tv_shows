# Data Sources — Setup & Schema Reference

All datasets live under `data/raw/`. This guide explains which sources each
model needs, where to get them, and the CSV format the `make_dataset.py`
pipeline expects.

When building through the GUI (`recsys-gui` → "Dataset" tab), a missing
**required** source is blocked by a pre-flight check that links back to this
document.

> **Note on the TV dataset:** the Trakt.tv signal used by the TV model was
> **collected from scratch for this project** — see
> [Trakt CSV](#trakt-csv--required). It is not an off-the-shelf download.

---

## TL;DR — what you need

| Source | Model | Required? | Size | Where to get it |
|---|---|---|---|---|
| MovieLens 32M | movies | **Required** | ~1.1 GB | [grouplens.org/datasets/movielens/32m](https://grouplens.org/datasets/movielens/32m/) |
| TMDB metadata v11 | movies | Optional | ~400 MB | [Kaggle: TMDB Movies Dataset (930K)](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies) |
| Trakt shows CSV | tv | **Required** | ~1.7 MB | [Release `data-v1`](https://github.com/archio1/recommendation_for_film_and_tv_shows/releases/tag/data-v1) (prebuilt) · or self-collect / third-party CSV |
| Trakt interactions CSV | tv | **Required** | ~25 MB | same |
| Amazon Reviews 2023 | movies/tv | Optional | ~100 GB | [amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io/) |

### Can I use my own / other datasets?

- **Movies — no (must be MovieLens 32M).** The movie pipeline depends structurally
  on MovieLens `links.csv` for the MovieID → TMDB ID mapping; it can't be swapped
  without rewriting `load_data()` / `link_to_movielens()`.
- **TV — yes, any CSV that matches the schema contract.** You don't have to run the
  Trakt crawler — a Kaggle dump, an IMDb export, or any other TV ratings source
  works as long as `trakt_shows.csv` and `trakt_interactions.csv` follow the columns
  and types in [the contract below](#c-plug-in-an-existing-csv--schema-contract).
- **TMDB metadata / Amazon Reviews — optional enrichment.** Drop them in to improve
  metadata/coverage, or leave them out; the pipeline degrades gracefully.

All inputs are **CSV** (or JSONL for Amazon). The exact columns, types, and
required/optional status per source are specified in the sections below.

---

## Movies pipeline

### MovieLens 32M — **Required**

GroupLens MovieLens 32M provides the core user–item matrix for the movie model.
Without it, `build_movie_dataset()` fails while reading `ratings.csv`
(see [`make_dataset.py:load_data`](../src/recommendation_system/data/make_dataset.py) — read with no `exists()` guard).

**How to install:**

1. Download `ml-32m.zip` (~250 MB) from https://grouplens.org/datasets/movielens/32m/.
2. Unpack into `data/raw/ml-32m/`. Expected layout:
   ```
   data/raw/ml-32m/
     ├── ratings.csv     (~870 MB, ~32M rows)
     ├── movies.csv      (~3 MB, ~87K movies)
     ├── links.csv       (~1.6 MB, movieId ↔ tmdbId)
     ├── tags.csv        (optional, used if present)
     ├── README.txt
     └── checksums.txt
   ```
3. Verify (GUI → Dataset tab → Sources → MovieLens dir): the indicator must be a green ✓.

**Schema (defined by GroupLens — do not change):**

- `ratings.csv`: `userId`, `movieId`, `rating` (0.5–5.0), `timestamp` (epoch).
- `movies.csv`: `movieId`, `title`, `genres` (pipe-separated).
- `links.csv`: `movieId`, `imdbId`, `tmdbId` (may be NaN).

### TMDB metadata v11 — Optional

Enriches movie metadata (overview, keywords, popularity, vote_count) and improves
genre classification. The movie pipeline works without it, just with leaner metadata.

**How to install:** download from Kaggle —
[TMDB Movies Dataset 2023 (930K movies)](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies) →
place the CSV at `data/raw/TMDB_movie_dataset_v11.csv`.

**Verify you got the right file:**
- Filename: `TMDB_movie_dataset_v11.csv` (~1.2 GB, ~930K rows).
- Has the columns the pipeline reads: `id`, `title`, `overview`, `genres`,
  `keywords`, `popularity`, `vote_count`, `vote_average`.
- GUI → Dataset tab → Sources → TMDB CSV: the indicator must be a green ✓.

Reference: [`make_dataset.py:clean_tmdb_movies`](../src/recommendation_system/data/make_dataset.py).

---

## TV pipeline

### Trakt CSV — **Required**

Trakt is the core user–item matrix for the TV model. **This dataset was
self-collected for the project** — Movie collaborative signal is abundant
(MovieLens), but a comparably large *TV* signal is not publicly available, so it
was crawled directly from the Trakt.tv API. The full collection yielded
**~5,020 shows × ~83K users × ~1.95M ratings** and fixed the historical
movie/TV imbalance (previously ~99% / ~1%).

Three ways to obtain the CSVs:

#### (a) Download the prebuilt CSVs — **recommended**

The exact dataset the project was trained on is published as GitHub Release
[`data-v1`](https://github.com/archio1/recommendation_for_film_and_tv_shows/releases/tag/data-v1).
Download both files into `data/raw/`:

- [`trakt_shows.csv`](https://github.com/archio1/recommendation_for_film_and_tv_shows/releases/download/data-v1/trakt_shows.csv) (~1.7 MB)
- [`trakt_interactions.csv`](https://github.com/archio1/recommendation_for_film_and_tv_shows/releases/download/data-v1/trakt_interactions.csv) (~25 MB)

**Verify the download (SHA-256):**

| File | SHA-256 |
|---|---|
| `trakt_shows.csv` | `bbaa413e25dd057414bb4983e3760e2734315ae6323bd98e7be517971d02010e` |
| `trakt_interactions.csv` | `c28c7b50f15e4cd571559317ec30a1f254f7a719a73044531a7823c2b3da0d6c` |

```bash
# from data/raw/ — should print the hashes above
sha256sum trakt_shows.csv trakt_interactions.csv          # Linux/macOS
certutil -hashfile trakt_shows.csv SHA256                 # Windows
```

#### (b) Self-collect via `trakt_collector.py` (≈2 days)

**Prerequisite — a Trakt API client id.** Create a Trakt account → Settings →
[Developer → Your API Apps](https://trakt.tv/oauth/applications) → "New Application",
then put the generated **Client ID** in your `.env` at the repo root:

```bash
TRAKT_CLIENT_ID=...        # required; recsys-trakt aborts without it
```

```bash
recsys-trakt
```

The script walks the Trakt API under rate limits; it discovers popular shows and
collects ratings from ~80K users. Progress is checkpointed in
`data/raw/trakt_collector.db`, so the crawl can be interrupted and resumed.

Output — two files in `data/raw/`:
- `trakt_shows.csv`
- `trakt_interactions.csv`

Use this path to reproduce the collection from scratch; for a quick start prefer
the prebuilt download in (a).

#### (c) Plug in an existing CSV — schema contract

If you already have a TV dataset (Kaggle, an IMDb dump, someone else's Trakt
snapshot), attach it via GUI → Dataset tab → "Sources" expander → file picker on
the `Trakt shows CSV` and `Trakt interactions CSV` fields.

The CSV must match the contract below, or the pipeline fails / returns an empty
DataFrame.

##### `trakt_shows.csv`

Reference: [`make_dataset.py:load_trakt_metadata`](../src/recommendation_system/data/make_dataset.py).

| Column | Type | Required? | Notes |
|---|---|---|---|
| `tmdb_id` | int | **Required** | TMDB show ID, **without** offset (+10M is added automatically) |
| `title` | str | **Required** | original title |
| `year` | int | **Required** | premiere year; NaN → row dropped |
| `genres` | str | **Required** | genre list: comma-separated (`"Drama, Sci-Fi"`) or JSON list (`'["Drama"]'`); empty → dropped |
| `overview` | str | **Required** | description; shorter than 10 chars → row dropped |
| `language` | str (ISO 639-1) | used if present | the GUI `languages` filter applies only if this column exists |
| `vote_average` | float | Optional (default 0.0) | TMDB-style 0–10 rating |
| `vote_count` | int | Optional (default 0) | used by the "non-target language ≥ 100 votes" filter |
| `popularity` | float | Optional (default 0.0) | used for top-N sorting |
| `title_ru` | str | Optional (default NULL) | RU title translation for the bilingual UI |

Minimal valid example:
```csv
tmdb_id,title,year,genres,overview,language,vote_average,vote_count,popularity
1399,Game of Thrones,2011,"Drama, Fantasy","Seven noble families fight for control of the mythical land of Westeros.",en,8.4,21000,500.0
1396,Breaking Bad,2008,"Drama, Crime","A high school chemistry teacher diagnosed with cancer turns to manufacturing meth.",en,8.9,11000,400.0
```

##### `trakt_interactions.csv`

Reference: [`make_dataset.py:load_trakt_interactions`](../src/recommendation_system/data/make_dataset.py).

| Column | Type | Required? | Notes |
|---|---|---|---|
| `user_id` | uint32 | **Required** | arbitrary non-negative ints; uniqueness not required (one row per rating) |
| `tmdb_id` | uint32 | **Required** | **without** +10M offset — added in the pipeline |
| `rating` | float32 | **Required** | 0.5–5.0 scale (MovieLens-compatible) |
| `timestamp` | uint32 | **Required** | unix epoch in seconds |

Minimal valid example:
```csv
user_id,tmdb_id,rating,timestamp
1,1399,5.0,1604188800
1,1396,4.5,1604189000
2,1399,4.0,1604190000
```

Rows whose `tmdb_id` is absent from `trakt_shows.csv` are dropped on join
(see [`make_dataset.py:load_trakt_interactions`](../src/recommendation_system/data/make_dataset.py)).

---

## Amazon Reviews 2023 — Optional (cross-domain)

A very large dataset (~100 GB) of user–item–rating interactions for movies and TV
from Amazon Prime. When attached, it adds tens of millions of interactions on top
of MovieLens / Trakt.

**How to use:**
1. Download [Movies_and_TV](https://amazon-reviews-2023.github.io/) (metadata + reviews) and unpack.
2. Folder layout:
   ```
   <amazon_dir>/
     ├── meta_Movies_and_TV.jsonl     (~3 GB)
     └── Movies_and_TV.jsonl          (~90 GB)
   ```
3. Point the pipeline at the folder:
   - **CLI:** `recsys-dataset --domain all --amazon-dir <amazon_dir>` (omit the flag
     to skip Amazon).
   - **GUI:** Dataset tab → "Sources" expander → set the "Amazon dir" field to the
     folder path.

If the files are absent, the pipeline does a graceful fallback (it simply skips
Amazon, [`make_dataset.py:load_amazon_interactions`](../src/recommendation_system/data/make_dataset.py)).
The source icon shows an amber ⚠ (optional missing).

ASIN → tmdb_id mapping is done by cleaned title (see
[`make_dataset.py:load_amazon_interactions`](../src/recommendation_system/data/make_dataset.py)).

---

## Overriding sources from the GUI

Any source can be overridden from the GUI without editing `make_dataset.py`:

1. Run `recsys-gui`.
2. Go to the "Dataset" tab.
3. Expand the "Sources" expander.
4. For the field you want, click the folder/file icon → file picker.
5. The icon on the right (`✓` / `✗` / `⚠`) confirms presence instantly.

An empty field = use the default under `data/raw/...`. An overridden path is kept
only for the current build run (not written to a settings file).

---

## Smoke vs Full preset

A full movies+tv build is memory- and time-heavy. On local hardware with limited
RAM/VRAM, pick a preset in the "Parameters" expander:

| Preset | top_n_movies | top_n_tv | min_year | rating_threshold | max_interactions | Build time | Good for |
|---|---|---|---|---|---|---|---|
| **Smoke** | 500 | 200 | 2010 | 3.5 | (no limit) | <5 min | pipeline check, code debugging |
| **Default** | 15,000 | 10,000 | (none) | 3.5 | 15,000,000 | 20–40 min | production build (current hardcodes) |
| **Full** | (all) | (all) | (none) | (all) | (all) | 1–3 h + ≥16 GB RAM | maximum coverage |
| **Custom** | — | — | — | — | — | — | manual tuning |

**Limited hardware** (e.g. a consumer GPU with ~8GB VRAM): use the Smoke preset —
the full movie dataset (~16M edges) is HW-bound on such a GPU (~10 h/epoch), while
Smoke gives ~1 min/epoch. Full training runs need a larger GPU / more RAM (e.g. a
cloud GPU host).

After you change any of the 8 parameters manually, the preset switches to `Custom`.

---

## Non-goals (explicitly out of scope)

- **Auto-download** of MovieLens/TMDB — instructions and links only, in this file.
- **Swapping MovieLens** for a "custom source" in the movie pipeline — structurally
  impossible without rewriting `load_data()` and `link_to_movielens()`. MovieLens
  is also required because `links.csv` provides the MovieID → TMDBID mapping.
- **A "run `trakt_collector.py`" button in the GUI** — the crawl takes ~2 days under
  Trakt API rate limits, which doesn't fit a GUI interaction.
- **Schema-contract validation on CSV override** — the user is responsible for
  conformance; errors surface in the build logs ("Dataset" tab → "Logs" panel).
