# `trainer_gui` Guide — Admin GUI for the dual-LightGCN stack

> **Version:** 2026-05-12 (after local training was implemented)
> **Files:** entry point `src/recommendation_system/models/gnn/trainer_gui.py`
> (thin shim); implementation in the `src/recommendation_system/models/gnn/gui/`
> subpackage (`theme`, `common`, `domain_stats`, `training_tab`, `dataset_tab`,
> `inference_tab`, `data_tab`, `app`)
> **Audience:** system developers/admins. This is **not** an end-user UI — end
> users have the Telegram bot (`movie_bot.py`).

## What it is and why

A Flet desktop GUI with 4 tabs. One process = one stack installation. Used for:

- training LightGCN models locally (for hyperparameter iteration / fixing regressions);
- building datasets through the UI instead of the CLI;
- offline recommendation testing (the same `DualDomainEngine` as the bot, but without Telegram);
- a quick look at dataset and model state.

**What the GUI does NOT do:**

- it does not run the Trakt collector (that's a ~2-day crawl, a separate script);
- it does not push models to the bot / production — that's a manual step;
- it does not do a hyperparameter sweep — tune parameters by hand;
- no model hot-swap: changing device or checkpoint requires a GUI restart;
- it does not cancel a dataset build (`make_dataset` has no cancel flag).

---

## First run

On a fresh checkout you have **no datasets and no models** yet. The order is:

1. **Build a dataset** — Tab 2 ("Dataset build"). You need the raw sources in place
   first; see **[data_sources.md](data_sources.md)** for exactly which datasets are
   **required**, which are **optional**, whether you can plug in your own, and the
   expected formats.
2. **Train a model** — Tab 1 ("Training") on that dataset.
3. **Test it** — Tab 3 ("Testing") before rolling it out to the bot.

Tab 4 ("Data") shows what you currently have. If a tab complains about missing
parquet/checkpoints, you've skipped a step above.

---

## Launch

```powershell
# Package installed via `pip install -e .` / `uv sync`.
recsys-gui      # long form: python -m recommendation_system.models.gnn.trainer_gui
```

On startup the GUI:

1. Imports `flet`, `torch` (5–10s cold start).
2. Creates 4 tabs. **Model engines are NOT loaded** — that happens lazily in Tab 3 on the first request.
3. Tab 4 (Data) immediately reads parquet/sidecar and shows statistics.

The window opens ~10–15s after the command. If it doesn't, check for the `python`
process in `tasklist`.

---

## Global header

Above the tabs are two global controls:

### Domain switcher (Dropdown)

`movies` / `tv`. Used **only** by Tab 1 (Training) — it decides which dataset/model
is built. Tab 2 (Dataset) and Tab 3 (Inference) have **their own** local dropdowns.
Tab 4 (Data) always shows both domains.

### Device selector (SegmentedButton)

CPU / GPU. The GPU segment is disabled if `torch.cuda.is_available() = False`.

- **Tab 1:** the value is passed to `trainer.main(--device=...)` with no override. CPU = ~10× slower than GPU.
- **Tab 3:** the value is fixed at the **first** engine load. Change it afterwards → restart the GUI.
- Tab 2 / Tab 4: device is not used.

---

## Tab 1 "Training"

A wrapper around `trainer.main()`. Implementation: `gui/training_tab.py:59` (`TrainingTab`).

### When to use

- Train a new model version locally.
- Tune hyperparameters.
- Fix a regression: "after editing `lightgcn.py` the old checkpoint still works, the new one doesn't".

### Hyperparameter fields

| Field | Default | What it changes |
|---|---|---|
| **Epochs** | 30 | Max epochs. Early stopping (`patience`) usually cuts it shorter. |
| **Batch** | 2048 | Minibatch size. On CPU use 512; on a 12GB GPU 2048 is fine; 24GB → 4096. |
| **LR** | 0.002 | Learning rate. Standard Adam; don't touch unless you know why. |
| **Embedding** | 32 | user/item vector dimension. 16–64 is sensible; 128 → overfits at our scale. |
| **GCN layers** | 2 | LightGCN depth. 1–3 sensible; >4 → overshoot, signal washout. |
| **Patience** | 3 | How many epochs to wait for recall@10 improvement before early-stop. |
| **Eval every** | 5 | Compute recall/ndcg on val every N epochs. More expensive → less often. |

Field tooltips show the defaults; when in doubt, leave them as-is.

### Start / Stop

- **Start** runs `trainer.main(argv, on_epoch_end=..., stop_flag=...)` in a `threading.Thread`. The UI stays responsive.
- **Stop** sets a flag → the next epoch aborts. Already-finished epochs are saved to the sidecar.

### What the metrics and chart show

| Metric | What it is |
|---|---|
| **Epoch** | current / total. Progress bar over epochs. |
| **Best epoch** | epoch with max recall@10. The checkpoint is saved from this epoch. |
| **Loss** | BPR loss on the batch. Should fall monotonically; a rise = LR too high. |
| **Recall@10 / NDCG@10** | measured every `eval_every` epochs on the validation split. |
| **Loss chart** | each point = an epoch. Flat after the first 5 → the model isn't learning (LR too low / broken graph). |

### Sanity check and result banner

After training, `trainer.main` validates the model:

- `recall@10 > 0.05` (low threshold; real models > 0.2).
- `||user_emb||.mean() > 0.01` (embeddings aren't all zeros).
- `||item_emb||.mean() > 0.01`.

Result:

- **Green** "sanity-check passed" → model OK, sidecar written, auto-bump points to the new checkpoint.
- **Red** "sanity-check FAILED (see sidecar)" → exit code 1, sidecar **is** written with `passed=false` and a failure list. Do not use such a model in the bot.
- **Red** "Training error (rc=...)" → exception. Check the log at the bottom.

### Sidecar JSON

After each train, a JSON sibling is written next to the `.pt` —
`lightgcn_{domain}_best_v{N}.json`:

```json
{
  "checkpoint": "lightgcn_movies_best_v5.pt",
  "domain": "movies",
  "device": "cuda",
  "trained_at": "2026-05-12T14:30:00Z",
  "epochs_run": 28,
  "best_epoch": 22,
  "metrics": {"recall@10": 0.234, "ndcg@10": 0.187},
  "dataset": {"users": 145032, "items": 18124, "interactions": 23900000},
  "hyperparameters": {"lr": 0.002, "batch_size": 2048, ...},
  "sanity_check": {"passed": true, "failures": []}
}
```

The "Open sidecar.json" button opens it in the system default app (Notepad / VS
Code). Use it to compare hyperparameters across runs.

### Version auto-bump

`trainer.py` scans `models/{domain}/lightgcn_{domain}_best_v*.pt` and picks the
next `vN+1`. Old versions are **not** deleted — clean them up by hand when disk runs out.

### Logs

At the bottom of the tab — a window with `trainer.py` logs via a `logging.Handler`.
Errors red, warnings yellow. Up to 500 lines (older ones auto-trimmed).

---

## Tab 2 "Dataset build"

A wrapper around `MovieDatasetProcessor.build_movie_dataset()` /
`.build_tv_dataset()` (`make_dataset.py:1238, 1289`). Implementation:
`gui/dataset_tab.py:170` (`DatasetTab`).

### When to use

- After collecting fresh Trakt data (`trakt_*.csv` updated).
- After changing the parquet schema (new columns, bilingual fields).
- On first deployment to a new machine.

A typical iteration is weekly, not daily.

### Sources (readonly)

Shows where the raw data lives:

- **Movies:** `data/raw/ml-32m/` (MovieLens 32M) + `data/raw/TMDB_movie_dataset_v11.csv`
- **TV:** `data/raw/trakt_shows.csv` + `data/raw/trakt_interactions.csv`
- **Amazon (optional):** see the field below.

### Domain dropdown (local)

`movies` / `tv` / `all`. **Does not** use the global header domain (which only has movies/tv).

- `all` → sequentially: movies first, then tv. If movies fails, tv won't run.

### Amazon dir

Default `D:/amazon_data`. If the folder is absent, the pipeline does a graceful
fallback (no Amazon interactions, just a `warning` log). To disable it entirely,
leave the field empty.

### Progress and logs

- **ProgressBar — indeterminate** (running bar). `make_dataset` gives no numeric callbacks, so a % can't be computed.
- **Status text** — a short status (last "stage": `Building movies...` / `Building tv...`).
- **Logs** — real `make_dataset.logger` output. Progress is visible via INFO messages.

A full movie build takes ~30–60 minutes (TMDB API hits for keywords + filtering +
k-core). TV is faster. The "Stop" button is decorative — `make_dataset` has no
cancel flag. **If you start it by mistake, you must wait or kill the whole process.**

### Post-build sanity check

After a successful `True` return, `_dataset_sanity(domain)` runs (`gui/dataset_tab.py:34`):

- `interactions_final.parquet`, `items_metadata_final.parquet`, `id_mapping.json` exist.
- `interactions_final.parquet`: ≥ 100k rows (otherwise a "smoke run?" warning).
- `items_metadata_final.parquet`: `tmdb_id`, `title`, `genres` columns present; no duplicate `tmdb_id`.
- `id_mapping.json`: `num_users` and `num_items` (or `num_trained_items`) — int > 0.

Result:

- **Green** "✅ Sanity-check passed" → use the dataset.
- **Red** "⚠ Sanity-check FAILED" with a concrete failure list → don't trust the dataset, investigate.

### Tab 4 auto-refresh

On success, `data_tab.refresh()` is called automatically — switch to Tab 4 to see
fresh mtimes/numbers.

---

## Tab 3 "Testing (Inference)"

An offline analog of the bot: the same `DualDomainEngine` + `UniversalSearchEngine`,
without Telegram. Implementation: `gui/inference_tab.py:57` (`InferenceTab`).

### When to use

- After training a new model — check that recommendations aren't broken.
- When debugging `DualDomainEngine` (cross-domain, FAISS, cold-start).
- For a bilingual check: "did search find Володар Перснів".
- For a smoke test before deploying to the bot.

### Layout

Left column (≈40%): **Search + Favorites (seed)**.
Right (≈60%): **Recommendations + 4 buttons + status**.

### Lazy engine init

Most important: when the tab is created, engines are **NOT** loaded (that's 30–60s
on CPU, ~30s on GPU). Loading starts on the first press of:

- any `/recs_*` button (if not yet loaded), or
- Search (if the active domain isn't loaded yet).

In the bottom-right is the `router_status`:

- `⏳ Engines not loaded (load on first /recs_*)` — initial state.
- `⏳ Movies: loading checkpoint...` → `⏳ TV: ...` → `⏳ FAISS: ...` — progress.
- `✅ Engines loaded` — ready.
- `❌ Engine load error` — traceback in the recommendations area.

All 4 buttons are disabled during loading. Device is taken from the header **at the
moment** of loading.

### Checkpoint selection

`InferenceTab` calls `_collect_domain_stats(domain).last_train_checkpoint` (see Tab 4)
— it takes the freshest sidecar JSON and reads the `.pt` name from it. Fallback is
the hardcoded `lightgcn_{domain}_best_v4.pt` (the same one `movie_bot.py` uses).

For InferenceTab to pick up a new model: train on Tab 1 → close the GUI → reopen it.
Hot-swap is not supported.

### Search (bilingual)

- **"Catalog" dropdown:** `movies` or `tv` — which catalog to search.
- **Search field:** EN / RU / UK, any string.
- **🔍 button / Enter** → `engine.search(query, limit=20)`.

`UniversalSearchEngine.search` already searches `title`, `title_ru`, `title_uk` in
parallel (`universal_search.py:627-...`). So "Володар Перснів", "Властелин колец",
"Lord of the Rings" all find the same film.

Results — a list of rows with emoji (🎬 movie / 📺 tv), the ★ button adds to favorites.

### Favorites (seed)

A list of tmdb_ids with title and media_type. Used as the **seed** for all `/recs_*`
buttons. Duplicates are dropped. The × button removes an item.

Held in process memory only — reset on GUI restart (intentional: this is an admin
tool, not a user session).

### 4 recommendation buttons

| Button | What it does | Needed in favorites |
|---|---|---|
| **/recs_movie** | `router.recs_movie(favorites, top_k=8)` | at least 1 movie |
| **/recs_tv** | `router.recs_tv(favorites, top_k=8)` | at least 1 show |
| **/recs_all** | `router.recs_all(favorites, top_k=8)` | anything |
| **/recs_cross** | `router.recs_cross(favorites, target_media_type=...)` | anything + the "Cross target" dropdown (`movie` / `tv`) |

The semantics are **identical** to the Telegram bot commands (`movie_bot.py:790-862`).
If the result differs from the bot, the bug is in the bot, not the GUI.

### Cross-domain (`/recs_cross`)

Requires the FAISS index:

- `src/recommendation_system/faiss_index/catalog.faiss`
- `src/recommendation_system/faiss_index/catalog_meta.json`

If the files are missing:

- On Tab 3 startup a **yellow banner** with instructions is shown.
- The `/recs_cross` button is disabled (the tooltip explains why).

Build the index:

```powershell
recsys-embed --to-faiss
```

After building, **restart the GUI** (Tab 3 only checks for the index at creation time).

### Lang switcher

The "Title language" dropdown: `en` / `ru` / `uk`. Switches the field shown in
results and favorites:

- `en` → `item.title`
- `ru` → `item.title_ru` (fallback to `title` if empty)
- `uk` → `item.title_uk` (fallback to `title` if empty)

Search works across **all languages simultaneously**, regardless of the switcher —
the switcher only controls display.

The search list is **not** re-rendered on language change (we don't keep source
items). To see the switch in search, repeat the query.

### Result card

- Title + year (in the selected language).
- Badge: 🎬 Movies / 📺 TV.
- Genres (first 5).
- TMDB rating + source (`trained` / `catalog` / `cold_start`).

### Common pitfalls

- **"Add items to favorites first"** → empty seed.
- **"Got 0 recommendations"** → the favorites' tmdb_ids aren't known to the relevant domain engine (e.g. `/recs_movie` with TV seeds).
- **"Engine load error"** → `checkpoint not found` (`v4.pt` missing) or `dataset not built` (no parquet). Go back to Tab 1 or Tab 2.

---

## Tab 4 "Data"

A readonly dashboard. Implementation: `gui/data_tab.py:26` (`DataTab`) +
`gui/domain_stats.py:138` (`_collect_domain_stats`).

### What it shows

Two cards side by side: 🎬 Movies and 📺 TV. Each:

**"Dataset" block:**
- Users / Items / Interactions (from `id_mapping.json` + parquet metadata).
- `interactions mtime` / `items mtime` (when the parquet was last rewritten).

**"Last training" block** (reads the freshest sidecar JSON from `models/{domain}/`):
- `trained at` (ISO timestamp).
- `checkpoint` (the `.pt` name).
- `recall@10`.
- Chip: **sanity OK** / **sanity FAIL**.

If there's no sidecar (the model predates the `local-training` infra) — "Sidecar
not found — model wasn't trained locally". This is normal for historical v1–v4.

**Buttons:**
- "Dataset folder" / "Models folder" → open in Explorer.

**Refresh (↻ at the top):** refreshes everything. Tab 2 calls refresh automatically
after a successful build — the button is only needed if you change something externally.

### If the dataset isn't built

The card shows "⚠ Dataset not built" + the expected path + a "Build it on the
'Dataset build' tab" hint. Build via Tab 2.

---

## End-to-end scenarios

### Scenario 1: "Trained a new model — want to check it"

1. **Tab 1**: domain=movies, epochs=30, device=GPU → Start.
2. Wait for the green banner → note the `.pt` path (or open the sidecar).
3. **Close the GUI and reopen it** (Tab 3 loads the checkpoint only on first launch).
4. **Tab 4**: check the Movies card → "Last training" = fresh, chip = sanity OK.
5. **Tab 3**: add 3–5 familiar movies to favorites → /recs_movie → eyeball them vs what the old model used to return.
6. If OK — roll it out to the bot (a separate step, not from the GUI).

### Scenario 2: "Fresh Trakt collect — rebuilding the TV dataset"

1. **Tab 2**: domain=tv, Amazon dir = default → Build.
2. Wait (TV ~10–20 min). Green "Sanity OK" banner.
3. **Tab 4**: the Movies card is unchanged; the TV card shows fresh numbers.
4. **Tab 1**: domain=tv, device=GPU → Start (a new TV checkpoint is needed because the catalog changed).
5. (Optional) **Tab 2**: domain=movies → rebuild, then Tab 1 movies. Do this **only** if the structure changed.
6. Restart the GUI and verify via Tab 3 (see Scenario 1).

### Scenario 3: "Compare two model versions"

The GUI doesn't do this directly. Workflow:

1. Do the first train (v5) on Tab 1.
2. Copy the v5 sidecar → notepad: note `recall@10`.
3. Change hyperparameters → second train (v6).
4. Open the v6 sidecar → compare by eye.
5. **Tab 3** shows recommendations only from the **latest** model (latest sidecar). To test v5, temporarily delete the v6 sidecar.

A full comparison → `spec/retrain-pipeline.md` or a custom script.

### Scenario 4: "GUI opened, but Tab 3 won't load engines"

Symptom: a red "Engine load error", a traceback in the log.

- **`FileNotFoundError: checkpoint not found`** → no `lightgcn_{domain}_best_v4.pt` OR the latest sidecar points to a missing file. Delete the broken sidecar / put `v4.pt` in place.
- **`FileNotFoundError: dataset not built`** → `data/processed/{domain}/` is empty. Tab 2 → Build.
- **`RuntimeError: InferenceEngine.movies: <error>`** → checkpoint incompatible with the current `lightgcn.py` (model architecture changed). Compare state_dict keys; rebuild the model with a fresh train.
- **`FaissCatalog not found`** → run `compute_embeddings --to-faiss`, restart the GUI.

---

## Limitations and risks

1. **Model hot-swap is not supported.** For InferenceTab to pick up a fresh checkpoint — close and reopen the GUI.
2. **Device is fixed at the first lazy-load in Tab 3.** Change the header device afterwards and the engine actually stays on the old one.
3. **Dataset build cannot be interrupted** (Tab 2). Start it by mistake → wait or kill the process.
4. **Concurrency:** don't run Tab 1 (train) and Tab 2 (build) at once — both are heavy. Tab 3 (inference) contends with train for GPU, with build for disk.
5. **Logs live only in the GUI window** (Tab 1, Tab 2). On a window crash they're lost. For archival use the sidecar (Tab 1) or `make_dataset.log` (Tab 2).

---

## Code references

| What | File / lines |
|---|---|
| GUI entry point | `gui/app.py:144 main()` (+ shim `trainer_gui.py`) |
| Design tokens / button factories | `gui/theme.py` (`COLORS`, `BUTTON_HEIGHT`, `CONTENT_MAX_WIDTH`, `primary/danger/accent/neutral/secondary_button`) |
| Tab 1 "Training" | `gui/training_tab.py:59 TrainingTab` |
| Tab 2 "Dataset build" | `gui/dataset_tab.py:170 DatasetTab` |
| Tab 3 "Testing" | `gui/inference_tab.py:57 InferenceTab` |
| Tab 4 "Data" | `gui/data_tab.py:26 DataTab` |
| Dataset sanity check | `gui/dataset_tab.py:34 _dataset_sanity()` |
| Stat collector for Tab 4 | `gui/domain_stats.py:138 _collect_domain_stats()` |
| Logs → UI bridge | `gui/common.py:33 _QueueLogHandler` |
| Backend CLI | `trainer.py:445-635 main()` |
| Backend dataset | `make_dataset.py:127 MovieDatasetProcessor` |
| Inference loader | `inference_engine.py:31 InferenceEngine` |
| Router | `dual_domain_engine.py:32 DualDomainEngine` |

## See also

- `spec/two-model-architecture.md` — why movies and tv are separated.
- `spec/retrain-pipeline.md` (future) — what will replace the manual train in Tab 1.
- `movie_bot.py` — the production variant of the same `DualDomainEngine`.
