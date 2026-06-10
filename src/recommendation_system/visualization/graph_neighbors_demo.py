"""
Graph neighbors visualization demo (visual companion to TestGraphOverlap).

For each curated seed (movie or TV show), shows side-by-side:
    LEFT  — top-10 nearest items in LightGCN embedding space (raw graph neighbors)
    RIGHT — top-15 production recommendations from DualDomainEngine
Items appearing in both panels are highlighted with a green border, making the
overlap visible to a non-technical viewer.

Output: a single offline HTML at reports/figures/graph_neighbors_demo.html.
"""
from __future__ import annotations

import math
import sys
from datetime import datetime

import pandas as pd

from recommendation_system.paths import (
    CACHE_DIR,
    FAISS_INDEX,
    FAISS_META,
    FIGURES_DIR,
    MOVIES_CHECKPOINT,
    MOVIES_DIR,
    PROJECT_ROOT,
    TV_CHECKPOINT,
    TV_DIR,
)

# Корень репо нужен в sys.path ради `from tests._quality_helpers import ...`.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_HTML = FIGURES_DIR / "graph_neighbors_demo.html"

NEIGHBORS_K = 10
TOP_K_RECS = 15
MIN_NEIGHBORS = 5

TV_OFFSET = 10_000_000

# (display_name, raw_tmdb_id) — TV ids get TV_OFFSET added at lookup time.
MOVIE_SEEDS_PRIMARY = [
    ("Inception", 27205),
    ("The Dark Knight", 155),
    ("The Matrix", 603),
    ("Interstellar", 157336),
    ("Avatar", 19995),
]
MOVIE_SEEDS_FALLBACK = [
    ("Pulp Fiction", 680),
    ("Forrest Gump", 13),
    ("The Shawshank Redemption", 278),
    ("Fight Club", 550),
    ("The Godfather", 238),
]

TV_SEEDS_PRIMARY = [
    ("Breaking Bad", 1396),
    ("Game of Thrones", 1399),
    ("Stranger Things", 66732),
    ("The Office (US)", 2316),
    ("Friends", 1668),
]
TV_SEEDS_FALLBACK = [
    ("Chernobyl", 87108),
    ("Sherlock", 19885),
    ("True Detective", 46648),
    ("Westworld", 63247),
    ("Black Mirror", 42009),
]

# ФИЛЬМЫ: (Имя для графика, raw_tmdb_id)
# MOVIE_SEEDS_PRIMARY = [
#     ("Inception", 27205),           # Проверка на "IMDb Top 250" эффект
#     ("Toy Story", 862),             # Чистая анимация
#     ("John Wick", 245891),          # Чистый экшен
#     ("Saw", 176),                   # Чистый хоррор
#     ("Dune: Part Two", 693134),     # Тест Cold Start (если его нет в графе, соседей будет 0 или мусор)
# ]
#
# MOVIE_SEEDS_FALLBACK = [
#     ("The Matrix", 603),
#     ("Interstellar", 157336),
#     ("Pulp Fiction", 680),
#     ("The Dark Knight", 155),
#     ("Gladiator", 98),
# ]

# СЕРИАЛЫ: (Имя для графика, raw_tmdb_id)
# Скрипт сам прибавит +10000000 для поиска в твоих .parquet файлах
# TV_SEEDS_PRIMARY = [
#     ("Breaking Bad", 1396),         # Главный хит
#     ("Game of Thrones", 1399),      # Эпик фэнтези
#     ("Stranger Things", 66732),     # Поп-культура / Мистика
#     ("Arcane", 94605),              # Анимация (проверим, не предложит ли он фильмы Нолана вместо аниме)
#     ("The Bear", 136315),           # Современная драма (тест на новизну)
# ]
#
# TV_SEEDS_FALLBACK = [
#     ("The Office (US)", 2316),
#     ("The Last of Us", 189132),
#     ("Succession", 76331),
#     ("Better Call Saul", 60735),
#     ("Friends", 1668),
# ]


def _short(text: str, limit: int = 22) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _genres_str(genres) -> str:
    if genres is None:
        return ""
    if isinstance(genres, (list, tuple)):
        return ", ".join(str(g) for g in genres if g)
    try:
        import numpy as np

        if isinstance(genres, np.ndarray):
            return ", ".join(str(g) for g in genres.tolist() if g)
    except Exception:
        pass
    return str(genres)


def _hover(title: str, year, genres) -> str:
    g = _genres_str(genres)
    parts = [f"<b>{title}</b>"]
    if year:
        try:
            parts[-1] += f" ({int(year)})"
        except (TypeError, ValueError):
            pass
    if g:
        parts.append(f"<i>{g}</i>")
    return "<br>".join(parts)


def _build_engine(dataset_dir: Path, checkpoint: Path):
    from recommendation_system.models.gnn.inference_engine import InferenceEngine
    from recommendation_system.models.gnn.universal_search import UniversalSearchEngine

    for required in (dataset_dir / "items_metadata_final.parquet", checkpoint):
        if not required.exists():
            raise FileNotFoundError(f"required artifact missing: {required}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    infer = InferenceEngine(dataset_dir, checkpoint, device="cpu")
    ok, msg = infer.load_resources()
    if not ok:
        raise RuntimeError(f"InferenceEngine failed to load {dataset_dir.name}: {msg}")

    metadata = pd.read_parquet(dataset_dir / "items_metadata_final.parquet")
    embeddings = dataset_dir / "overview_embeddings.npy"
    return UniversalSearchEngine(
        metadata=metadata,
        cache_dir=CACHE_DIR,
        tmdb_api_key=None,
        inference_engine=infer,
        embeddings_path=embeddings if embeddings.exists() else None,
        model_num_items=infer.model.num_items,
    )


def _build_dual_engine(movies_engine, tv_engine):
    from recommendation_system.models.gnn.dual_domain_engine import DualDomainEngine
    from recommendation_system.models.gnn.faiss_bridge import FaissCatalog

    if not (FAISS_INDEX.exists() and FAISS_META.exists()):
        raise FileNotFoundError(f"FAISS index missing: {FAISS_INDEX} / {FAISS_META}")
    faiss = FaissCatalog.load(FAISS_INDEX, FAISS_META)
    return DualDomainEngine(
        movies_engine=movies_engine,
        tv_engine=tv_engine,
        faiss_catalog=faiss,
    )


def _resolve_to_offset_tmdb(raw_tmdb: int, domain: str) -> int:
    return int(raw_tmdb) + TV_OFFSET if domain == "tv" else int(raw_tmdb)


def _row_for_tmdb(metadata: pd.DataFrame, tmdb_id: int):
    row = metadata[metadata["tmdb_id"] == int(tmdb_id)]
    return None if row.empty else row.iloc[0]


def _enrich(metadata: pd.DataFrame, tmdb_id: int) -> dict | None:
    row = _row_for_tmdb(metadata, tmdb_id)
    if row is None:
        return None
    return {
        "tmdb_id": int(row["tmdb_id"]),
        "item_id": int(row["item_id"]),
        "title": str(row["title"]),
        "year": int(row["year"]) if pd.notna(row["year"]) else None,
        "genres": row["genres"],
    }


def _seed_data(engine, dual_engine, domain: str, raw_tmdb: int) -> dict | None:
    """Fetch graph neighbors + production recs + overlap for one seed."""
    from tests._quality_helpers import compute_graph_neighbors

    offset_tmdb = _resolve_to_offset_tmdb(raw_tmdb, domain)
    seed = _enrich(engine.metadata, offset_tmdb)
    if seed is None:
        return None

    neighbor_ids = compute_graph_neighbors(
        inference_engine=engine.inference_engine,
        search_engine=engine,
        liked_tmdb_ids=[offset_tmdb],
        k=NEIGHBORS_K,
    )
    neighbors = [
        n for n in (_enrich(engine.metadata, t) for t in neighbor_ids) if n is not None
    ]
    neighbors.sort(key=lambda x: x["title"].lower())
    neighbors = neighbors[:NEIGHBORS_K]

    if domain == "movie":
        rec_items = dual_engine.recs_movie([offset_tmdb], top_k=TOP_K_RECS)
    else:
        rec_items = dual_engine.recs_tv([offset_tmdb], top_k=TOP_K_RECS)

    recs: list[dict] = []
    for r in rec_items[:TOP_K_RECS]:
        recs.append({
            "tmdb_id": int(r.tmdb_id),
            "item_id": int(getattr(r, "item_id", -1)),
            "title": str(r.title),
            "year": int(r.year) if r.year else None,
            "genres": list(r.genres) if r.genres else [],
        })

    overlap = {n["tmdb_id"] for n in neighbors} & {r["tmdb_id"] for r in recs}
    return {
        "seed": seed,
        "neighbors": neighbors,
        "recs": recs,
        "overlap": overlap,
        "domain": domain,
    }


def _seeds_with_fallback(engine, dual_engine, domain, primary, fallback):
    out = []
    used = set()
    for name, tid in primary:
        data = _seed_data(engine, dual_engine, domain, tid)
        if data is None:
            print(f"  ! skip {name} (tmdb={tid}): not in {domain} metadata")
            continue
        if len(data["neighbors"]) < MIN_NEIGHBORS:
            print(f"  ! {name}: only {len(data['neighbors'])} neighbors, trying fallback")
            continue
        out.append(data)
        used.add(tid)
    if len(out) < len(primary):
        for name, tid in fallback:
            if tid in used or len(out) >= len(primary):
                break
            data = _seed_data(engine, dual_engine, domain, tid)
            if data and len(data["neighbors"]) >= MIN_NEIGHBORS:
                print(f"  + fallback added: {name}")
                out.append(data)
    return out


def _build_seed_figure(data: dict):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    seed = data["seed"]
    neighbors = data["neighbors"]
    recs = data["recs"]
    overlap = data["overlap"]

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.42, 0.58],
        subplot_titles=(
            f"<b>Соседи по графу LightGCN</b>  (топ-{len(neighbors)})",
            f"<b>Что рекомендует модель</b>  (топ-{len(recs)})",
        ),
        horizontal_spacing=0.06,
    )

    # ---- LEFT: graph (seed in center, neighbors radial) ----
    n = max(1, len(neighbors))
    angle_step = 2 * math.pi / n
    radius = 1.6
    edge_x: list = []
    edge_y: list = []
    nb_x: list = []
    nb_y: list = []
    nb_text: list = []
    nb_hover: list = []
    nb_border: list = []
    for i, nb in enumerate(neighbors):
        angle = -math.pi / 2 + i * angle_step
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        edge_x.extend([0.0, x, None])
        edge_y.extend([0.0, y, None])
        nb_x.append(x)
        nb_y.append(y)
        nb_text.append(_short(nb["title"], 18))
        nb_hover.append(_hover(nb["title"], nb["year"], nb["genres"]))
        nb_border.append("#06d6a0" if nb["tmdb_id"] in overlap else "#457b9d")

    fig.add_trace(
        go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(color="#cfd8dc", width=1.5),
            hoverinfo="skip", showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0.0], y=[0.0], mode="markers+text",
            marker=dict(size=85, color="#ffd166",
                        line=dict(color="#e63946", width=3)),
            text=[_short(seed["title"], 18)],
            textposition="middle center",
            textfont=dict(size=11, color="#1d3557"),
            hovertext=[_hover(seed["title"], seed["year"], seed["genres"])],
            hoverinfo="text",
            showlegend=False,
        ),
        row=1, col=1,
    )
    if neighbors:
        fig.add_trace(
            go.Scatter(
                x=nb_x, y=nb_y, mode="markers+text",
                marker=dict(size=65, color="#a8dadc",
                            line=dict(color=nb_border, width=3)),
                text=nb_text,
                textposition="middle center",
                textfont=dict(size=9, color="#1d3557"),
                hovertext=nb_hover, hoverinfo="text",
                showlegend=False,
            ),
            row=1, col=1,
        )

    # ---- RIGHT: 3x5 card grid ----
    cols, rows = 3, 5
    rec_x: list = []
    rec_y: list = []
    rec_text: list = []
    rec_hover: list = []
    rec_fill: list = []
    rec_border: list = []
    for i, rec in enumerate(recs):
        cx = i % cols
        cy = rows - 1 - (i // cols)
        rec_x.append(cx)
        rec_y.append(cy)
        rec_text.append(f"{i+1}. {_short(rec['title'], 18)}")
        rec_hover.append(_hover(rec["title"], rec["year"], rec["genres"]))
        if rec["tmdb_id"] in overlap:
            rec_fill.append("#caffbf")
            rec_border.append("#06d6a0")
        else:
            rec_fill.append("#f1f3f5")
            rec_border.append("#adb5bd")

    if recs:
        fig.add_trace(
            go.Scatter(
                x=rec_x, y=rec_y, mode="markers+text",
                marker=dict(symbol="square", size=95,
                            color=rec_fill,
                            line=dict(color=rec_border, width=4)),
                text=rec_text,
                textposition="middle center",
                textfont=dict(size=9, color="#1d3557"),
                hovertext=rec_hover, hoverinfo="text",
                showlegend=False,
            ),
            row=1, col=2,
        )

    fig.update_xaxes(visible=False, range=[-2.4, 2.4], row=1, col=1)
    fig.update_yaxes(visible=False, range=[-2.4, 2.4], scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(visible=False, range=[-0.6, cols - 0.4], row=1, col=2)
    fig.update_yaxes(visible=False, range=[-0.6, rows - 0.4], row=1, col=2)

    overlap_count = sum(1 for r in recs if r["tmdb_id"] in overlap)
    n_recs = max(1, len(recs))
    pct = round(100.0 * overlap_count / n_recs)
    seed_year = f"({seed['year']})" if seed["year"] else ""
    title_text = (
        f"<b>{seed['title']}</b> {seed_year}  —  "
        f"<span style='color:#06d6a0'>{overlap_count} из {len(recs)}</span> "
        f"рекомендаций совпадают с соседями по графу ({pct}%)"
    )

    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor="center", font=dict(size=15)),
        height=620,
        margin=dict(t=80, b=20, l=20, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Segoe UI, Arial, sans-serif"),
    )
    return fig, overlap_count


_LEGEND_HTML = """
<div style="background:#f8f9fa;border-left:4px solid #06d6a0;padding:14px 18px;margin:18px 0;border-radius:4px;font-family:Segoe UI,Arial,sans-serif">
  <b>Как читать эту страницу:</b><br>
  Слева — соседи seed-айтема в графе LightGCN (что модель считает «похожим» на него по эмбеддингам).<br>
  Справа — то, что выдаёт <b>продакшн-пайплайн рекомендаций</b> (с интент-кластеризацией и контентным движком).<br>
  <b style="color:#06d6a0">Зелёная рамка</b> = айтем встречается одновременно и среди соседей, и среди рекомендаций. Чем больше зелёного — тем сильнее рекомендации опираются на структуру графа.
</div>
""".strip()

_TAB_CSS = """
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; background:#f5f7fa; margin:0; padding:24px; color:#1d3557; }
  h1 { text-align:center; margin:0 0 6px 0; }
  .subtitle { text-align:center; color:#6c757d; margin-bottom:18px; font-size:14px; }
  .tabs { display:flex; gap:6px; justify-content:center; margin:18px 0 8px 0; }
  .tab { padding:10px 22px; border:1px solid #adb5bd; background:#fff; cursor:pointer; border-radius:6px 6px 0 0; font-size:15px; font-family:inherit; }
  .tab.active { background:#1d3557; color:#fff; border-color:#1d3557; }
  .tabcontent { background:#fff; border-radius:8px; padding:14px; box-shadow:0 1px 4px rgba(0,0,0,0.06); }
  .seed-section { margin:6px 0 10px 0; }
  footer { text-align:center; color:#6c757d; font-size:12px; margin-top:30px; }
</style>
""".strip()

_TAB_JS = """
<script>
function showTab(name, btn) {
  document.querySelectorAll('.tabcontent').forEach(d => d.style.display='none');
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById(name).style.display='block';
  btn.classList.add('active');
}
</script>
""".strip()


def _render_html(movie_results, tv_results, output_path: Path) -> None:
    from plotly import io as pio

    movie_divs: list[str] = []
    tv_divs: list[str] = []
    first = True

    def to_div(fig) -> str:
        nonlocal first
        include = "inline" if first else False
        first = False
        return pio.to_html(
            fig,
            include_plotlyjs=include,
            full_html=False,
            config={"displayModeBar": False, "responsive": True},
        )

    for data in movie_results:
        fig, _ = _build_seed_figure(data)
        movie_divs.append(f'<div class="seed-section">{to_div(fig)}</div>')
    for data in tv_results:
        fig, _ = _build_seed_figure(data)
        tv_divs.append(f'<div class="seed-section">{to_div(fig)}</div>')

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    movie_count = len(movie_results)
    tv_count = len(tv_results)
    total_overlap = sum(len(d["overlap"]) for d in movie_results + tv_results)
    total_recs = sum(len(d["recs"]) for d in movie_results + tv_results) or 1
    avg_pct = round(100.0 * total_overlap / total_recs)

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Граф соседей vs. рекомендации модели</title>
  {_TAB_CSS}
</head>
<body>
  <h1>Граф соседей LightGCN vs. рекомендации модели</h1>
  <div class="subtitle">Визуальный компаньон теста <code>TestGraphOverlap</code>. Демо seed-айтемов: {movie_count} фильмов и {tv_count} сериалов. Среднее пересечение: <b>{avg_pct}%</b>.</div>
  {_LEGEND_HTML}
  <div class="tabs">
    <button class="tab active" onclick="showTab('movies', this)">🎬 Фильмы</button>
    <button class="tab" onclick="showTab('tv', this)">📺 Сериалы</button>
  </div>
  <div id="movies" class="tabcontent">
    {''.join(movie_divs) if movie_divs else '<p style="text-align:center;color:#6c757d">Нет данных по фильмам</p>'}
  </div>
  <div id="tv" class="tabcontent" style="display:none">
    {''.join(tv_divs) if tv_divs else '<p style="text-align:center;color:#6c757d">Нет данных по сериалам</p>'}
  </div>
  <footer>Сгенерировано {timestamp}</footer>
  {_TAB_JS}
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    print("1/4  Загрузка movies engine…")
    movies_engine = _build_engine(MOVIES_DIR, MOVIES_CHECKPOINT)
    print("2/4  Загрузка tv engine…")
    tv_engine = _build_engine(TV_DIR, TV_CHECKPOINT)
    print("3/4  Сбор DualDomainEngine + FAISS…")
    dual = _build_dual_engine(movies_engine, tv_engine)

    print("4/4  Расчёт соседей и рекомендаций по seed-айтемам:")
    print(" Фильмы:")
    movie_results = _seeds_with_fallback(
        movies_engine, dual, "movie", MOVIE_SEEDS_PRIMARY, MOVIE_SEEDS_FALLBACK
    )
    print(" Сериалы:")
    tv_results = _seeds_with_fallback(
        tv_engine, dual, "tv", TV_SEEDS_PRIMARY, TV_SEEDS_FALLBACK
    )

    if not movie_results and not tv_results:
        raise RuntimeError("ни один seed не дал валидных данных — проверьте датасет")

    print(f"   ↳ итог: {len(movie_results)} фильмов, {len(tv_results)} сериалов")
    print(f"   рендер HTML → {OUTPUT_HTML}")
    _render_html(movie_results, tv_results, OUTPUT_HTML)

    size_kb = OUTPUT_HTML.stat().st_size / 1024
    print(f"✅ Готово. Размер: {size_kb:.0f} KB. Откройте в браузере: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
