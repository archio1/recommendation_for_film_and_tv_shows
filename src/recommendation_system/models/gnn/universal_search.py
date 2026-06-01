"""
universal_search.py — Hybrid Search & Recommendation Engine v4
"""

import ast
import json
import time
import sqlite3
import logging
import asyncio
import math
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from title_normalizer import normalize_for_search

logger = logging.getLogger(__name__)

# =============================================================================
# GENRE & KEYWORD NORMALIZATION
# =============================================================================

GENRE_NORMALIZE: Dict[str, List[str]] = {
    "Action & Adventure": ["Action", "Adventure"],
    "Sci-Fi & Fantasy": ["Science Fiction", "Fantasy"],
    "War & Politics": ["War"],
    "Kids": ["Family"],
    "Sci-Fi": ["Science Fiction"],
}

def normalize_genres(genres: List[str]) -> List[str]:
    """Expand compound TMDB tags into standard atomic genres."""
    result: List[str] = []
    seen: Set[str] = set()
    for g in genres:
        expanded = GENRE_NORMALIZE.get(g, [g])
        for sub in expanded:
            if sub not in seen:
                seen.add(sub)
                result.append(sub)
    return result

expand_compound_genres = normalize_genres

def ensure_list(raw) -> List[str]:
    """Convert any format to clean List[str]."""
    if raw is None: return []
    items = []
    if isinstance(raw, np.ndarray):
        items = [str(g) for g in raw if pd.notna(g) and str(g).strip()]
    elif isinstance(raw, list):
        items = [str(g) for g in raw if g is not None and str(g).strip() and str(g).lower() != 'nan']
    elif isinstance(raw, str):
        s = raw.strip()
        if not s or s.lower() in ('nan', 'none', '[]'): return []
        if s.startswith('['):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list): items = [str(g) for g in parsed if g]
            except: pass
        if not items:
            delimiter = '|' if '|' in s else ','
            items = [g.strip() for g in s.split(delimiter) if g.strip()]
    return items

def ensure_genres(raw) -> List[str]:
    return normalize_genres(ensure_list(raw))

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class UniversalMediaItem:
    tmdb_id: int
    media_type: str
    title: str
    year: int
    genres: List[str]
    keywords: List[str] = field(default_factory=list)
    source: str = 'trained'
    has_embeddings: bool = False
    in_graph: bool = False
    item_id: Optional[int] = None
    title_ru: Optional[str] = None
    overview: Optional[str] = None
    overview_ru: Optional[str] = None
    title_uk: Optional[str] = None
    overview_uk: Optional[str] = None
    vote_average: float = 0.0
    vote_count: int = 0
    popularity: float = 0.0
    poster_path: Optional[str] = None
    imdb_id: Optional[str] = None
    last_updated: Optional[str] = None

@dataclass
class SearchResult:
    query: str
    results: List[UniversalMediaItem] = field(default_factory=list)
    source_breakdown: Dict[str, int] = field(default_factory=dict)

# =============================================================================
# MEDIA DNA & INTENT CLUSTERING
# =============================================================================

MEDIA_DNA_TAGS = {
    'anime':           {'anime', 'japanese animation', 'crunchyroll', 'manga', 'shonen', 'seinen'},
    'animation_west':  {'western animation', 'cartoon', 'animated series'},
    'gritty':          {'dark', 'violent', 'mature', 'blood', 'brutal', 'gore', 'graphic'},
    'cerebral':        {'psychological', 'philosophical', 'mind-bending', 'complex',
                        'existential', 'introspective', 'cerebral', 'thought-provoking',
                        'intellectual', 'puzzle', 'unreliable narrator', 'nonlinear'},
    'campy':           {'campy', 'tongue-in-cheek', 'over-the-top', 'satire'},
    'satirical':       {'satire', 'satirical', 'parody', 'social commentary',
                        'dark comedy', 'black comedy', 'subversive', 'deconstruction',
                        'anti-hero', 'antihero', 'irreverent'},
    'anthology':       {'anthology', 'standalone episodes'},
    'serial':          {'serialized', 'season-long arc', 'cliffhanger'},
    'procedural':      {'case of the week', 'procedural', 'self-contained',
                        'investigation', 'detective', 'forensic', 'law enforcement'},
    'korean':          {'k-drama', 'korean', 'south korea'},
    'british':         {'bbc', 'british', 'uk production', 'itv', 'channel 4'},
    'hbo_prestige':    {'hbo', 'prestige drama', 'prestige television'},
    'netflix':         {'netflix'},
    'apple_tv':        {'apple tv+', 'apple tv'},
    'prime_video':     {'amazon', 'prime video', 'amazon studios'},
    'epic_fantasy':    {'epic fantasy', 'high fantasy', 'sword and sorcery',
                        'dragon', 'kingdom', 'quest', 'prophecy', 'magic'},
    'superhero':       {'superhero', 'superpower', 'super powers', 'vigilante',
                        'comic book', 'marvel', 'dc comics', 'cape'},
    'slow_burn':       {'slow burn', 'atmospheric', 'meditative', 'contemplative',
                        'character study', 'minimalist'},
}

TONE_TAGS = set(MEDIA_DNA_TAGS.keys()) - {'anime', 'animation_west', 'live_action'}

def infer_media_dna(item: UniversalMediaItem) -> Set[str]:
    dna = set()
    all_signals = set(item.genres + item.keywords)
    overview_lower = (item.overview or '').lower()
    
    if 'Animation' in item.genres or 'animation' in all_signals:
        if any(kw in all_signals for kw in ['anime', 'manga', 'shonen', 'seinen']) or \
           any(kw in overview_lower for kw in ['japan', 'tokyo', 'samurai', 'anime']):
            dna.add('anime')
        else:
            dna.add('animation_west')
    else:
        dna.add('live_action')
    
    for tag, signals in MEDIA_DNA_TAGS.items():
        if signals and any(s in overview_lower or s in all_signals for s in signals):
            dna.add(tag)
            
    return dna

def detect_user_intents(
    liked_items: List[UniversalMediaItem],
    get_embedding_fn: Callable,
    similarity_threshold: float = 0.5
) -> List[List[UniversalMediaItem]]:
    """Group liked items into coherent intent clusters."""
    if len(liked_items) <= 2:
        return [liked_items]
    
    embeddings = []
    valid_items = []
    for item in liked_items:
        emb = get_embedding_fn(item)
        if emb is not None:
            embeddings.append(emb)
            valid_items.append(item)
            
    if len(valid_items) <= 2:
        return [liked_items]
        
    emb_matrix = np.stack(embeddings)
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.0 - similarity_threshold,
        metric='cosine',
        linkage='average'
    )
    labels = clustering.fit_predict(emb_matrix)
    
    clusters = {}
    for item, label in zip(valid_items, labels):
        clusters.setdefault(label, []).append(item)
        
    # Add back items without embeddings to the largest cluster
    unembedded = [item for item in liked_items if item not in valid_items]
    if unembedded:
        if clusters:
            largest_cluster = max(clusters.values(), key=len)
            largest_cluster.extend(unembedded)
        else:
            clusters[0] = unembedded
            
    return list(clusters.values())

# =============================================================================
# HOT CACHE & API CLIENT
# =============================================================================

class HotCache:
    """SQLite кэш для новинок и трендов (17 колонок, включая keywords)"""

    # Canonical SELECT column order shared by search/get_trending/get_by_tmdb_id.
    # Hardcoding this list keeps _row_to_item stable even if ALTER TABLE
    # appends new columns at the end on older databases.
    _SELECT_COLS = (
        "tmdb_id, media_type, title, title_ru, year, genres, keywords, "
        "overview, overview_ru, vote_average, vote_count, popularity, "
        "poster_path, imdb_id, source, added_at, last_updated, "
        "title_uk, overview_uk"
    )

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.conn = sqlite3.connect(str(cache_path), check_same_thread=False)
        # SQLite's built-in LOWER() and LIKE only fold ASCII case — Cyrillic
        # like "Крепкий" stays uppercase, so a query "крепкий" silently
        # misses the row. Register a Python UDF so search() can fold both
        # sides via str.lower(), which handles Unicode correctly.
        self.conn.create_function(
            "PY_LOWER", 1, lambda s: s.lower() if s else ""
        )
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS hot_items (
                tmdb_id INTEGER PRIMARY KEY,
                media_type TEXT NOT NULL,
                title TEXT NOT NULL,
                title_ru TEXT,
                year INTEGER,
                genres TEXT,
                keywords TEXT,
                overview TEXT,
                overview_ru TEXT,
                vote_average REAL,
                vote_count INTEGER,
                popularity REAL,
                poster_path TEXT,
                imdb_id TEXT,
                source TEXT DEFAULT 'tmdb_live',
                added_at INTEGER,
                last_updated INTEGER
            )
        """)
        try:
            self.conn.execute("ALTER TABLE hot_items ADD COLUMN keywords TEXT DEFAULT '[]'")
        except sqlite3.OperationalError:
            pass
        # Idempotent uk migration. Existing rows get NULL — fallback chain
        # in the bot picks ru/en for those.
        for col in ("title_uk", "overview_uk"):
            try:
                self.conn.execute(f"ALTER TABLE hot_items ADD COLUMN {col} TEXT")
            except sqlite3.OperationalError:
                pass

        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_hot_title ON hot_items(title COLLATE NOCASE)")
        self.conn.commit()

    def upsert_batch(self, items: List[UniversalMediaItem]):
        now = int(time.time())
        data = []
        for i in items:
            data.append((
                i.tmdb_id, i.media_type, i.title, i.title_ru, i.year,
                json.dumps(i.genres, ensure_ascii=False),
                json.dumps(i.keywords, ensure_ascii=False),
                i.overview, i.overview_ru if hasattr(i, 'overview_ru') else None,
                i.vote_average, i.vote_count if hasattr(i, 'vote_count') else 0,
                i.popularity, i.poster_path if hasattr(i, 'poster_path') else None,
                i.imdb_id if hasattr(i, 'imdb_id') else None,
                i.source, now, now,
                getattr(i, 'title_uk', None), getattr(i, 'overview_uk', None),
            ))

        # Explicit column list — the table now has 19 logical columns
        # (added title_uk/overview_uk via migration), but old DBs might
        # ALTER them in at the end, so we name the columns rather than
        # rely on positional VALUES order.
        self.conn.executemany(
            """INSERT OR REPLACE INTO hot_items
               (tmdb_id, media_type, title, title_ru, year, genres, keywords,
                overview, overview_ru, vote_average, vote_count, popularity,
                poster_path, imdb_id, source, added_at, last_updated,
                title_uk, overview_uk)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            data
        )
        self.conn.commit()

    def search(self, query, media_type=None, limit=10):
        # Fold both sides through PY_LOWER so Cyrillic case-insensitivity
        # matches the pandas .str.lower() path in UniversalSearchEngine.
        # Numeric variants (Se7en ↔ seven) come from title_normalizer —
        # each variant becomes one OR-clause across the three title cols.
        variants = normalize_for_search(query) or [(query or "").lower()]
        clauses = []
        params: list = []
        for v in variants:
            like = f"%{v}%"
            clauses.append(
                "(PY_LOWER(title) LIKE ? "
                "OR PY_LOWER(title_ru) LIKE ? "
                "OR PY_LOWER(title_uk) LIKE ?)"
            )
            params.extend([like, like, like])
        sql = (
            f"SELECT {self._SELECT_COLS} FROM hot_items "
            f"WHERE ({' OR '.join(clauses)})"
        )
        if media_type:
            sql += " AND media_type = ?"
            params.append(media_type)
        sql += f" ORDER BY popularity DESC LIMIT {limit}"
        return [self._row_to_item(r) for r in self.conn.execute(sql, params).fetchall()]

    def get_trending(self, media_type=None, limit=20, min_vote_count=100):
        # min_vote_count cuts TMDB-noise: items with high `popularity`
        # (search/view buzz) but very few actual ratings — typically
        # marketing-heavy releases that nobody has rated yet. A real
        # trending hit has tens of thousands of votes; even a brand-new
        # theatrical release breaks 100 within ~2 weeks.
        sql = f"SELECT {self._SELECT_COLS} FROM hot_items WHERE source = 'trending' AND vote_count >= ?"
        params = [min_vote_count]
        if media_type:
            sql += " AND media_type = ?"
            params.append(media_type)
        sql += " ORDER BY popularity DESC LIMIT ?"
        params.append(limit)
        return [self._row_to_item(r) for r in self.conn.execute(sql, params).fetchall()]

    def get_by_tmdb_id(self, tmdb_id: int):
        r = self.conn.execute(
            f"SELECT {self._SELECT_COLS} FROM hot_items WHERE tmdb_id = ?",
            (tmdb_id,),
        ).fetchone()
        return self._row_to_item(r) if r else None

    def _row_to_item(self, r):
        # Index order matches HotCache._SELECT_COLS exactly.
        return UniversalMediaItem(
            tmdb_id=r[0], media_type=r[1], title=r[2], title_ru=r[3],
            year=r[4], genres=json.loads(r[5]), keywords=json.loads(r[6] or '[]'),
            overview=r[7], overview_ru=r[8], vote_average=r[9], vote_count=r[10],
            popularity=r[11], poster_path=r[12], imdb_id=r[13],
            source=r[14], has_embeddings=False, last_updated=str(r[16]),
            title_uk=r[17], overview_uk=r[18],
        )

# =============================================================================
# ENHANCED CONTENT ENGINE (IDF + Semantic + Keywords + DNA)
# =============================================================================

class EnhancedContentEngine:
    """Движок контентного сходства: IDF Жанры + IDF Ключевые слова + Семантические векторы + Штрафы"""

    def __init__(self, genre_list, keyword_list, metadata, embeddings_path=None):
        self.genre_list = genre_list
        self.genre_to_idx = {g: i for i, g in enumerate(genre_list)}
        self.num_genres = len(genre_list)
        self.idf_weights = self._compute_idf(metadata)
        
        self.keyword_list = keyword_list
        self.keyword_to_idx = {k: i for i, k in enumerate(keyword_list)}
        self.num_keywords = len(keyword_list)
        self.keyword_idf_weights = self._compute_keyword_idf(metadata)

        self.embeddings = None
        self.item_id_to_emb_idx = {}
        self._live_encoder = None

        if embeddings_path and Path(embeddings_path).exists():
            try:
                self.embeddings = np.load(embeddings_path).astype(np.float32)
                sorted_meta = metadata.sort_values('item_id')
                for i, row in enumerate(sorted_meta.itertuples()):
                    self.item_id_to_emb_idx[int(row.item_id)] = i
                logger.info(f"EnhancedContentEngine: загружено {len(self.embeddings)} векторов.")
            except Exception as e:
                logger.error(f"Ошибка загрузки векторов: {e}")

    def _compute_idf(self, metadata):
        N = len(metadata)
        df = np.zeros(self.num_genres)
        for g_list in metadata['genres']:
            for g in ensure_genres(g_list):
                if g in self.genre_to_idx: df[self.genre_to_idx[g]] += 1
        return np.log(N / (df + 1)).astype(np.float32)

    def _compute_keyword_idf(self, metadata):
        N = len(metadata)
        df = np.zeros(self.num_keywords)
        if 'keywords' in metadata.columns:
            for k_list in metadata['keywords']:
                for k in ensure_list(k_list):
                    if k in self.keyword_to_idx: df[self.keyword_to_idx[k]] += 1
        return np.log(N / (df + 1)).astype(np.float32)

    def _encode_genres_idf(self, genres):
        v = np.zeros(self.num_genres)
        for g in genres:
            if g in self.genre_to_idx:
                v[self.genre_to_idx[g]] = self.idf_weights[self.genre_to_idx[g]]
        return v

    def _encode_keywords_idf(self, keywords):
        v = np.zeros(self.num_keywords)
        for k in keywords:
            if k in self.keyword_to_idx:
                v[self.keyword_to_idx[k]] = self.keyword_idf_weights[self.keyword_to_idx[k]]
        return v

    def _genre_idf_cosine(self, u_prof, c_genres):
        cv = self._encode_genres_idf(c_genres)
        un = np.linalg.norm(u_prof)
        cn = np.linalg.norm(cv)
        if un < 1e-8 or cn < 1e-8: return 0.0
        return np.dot(u_prof, cv) / (un * cn)

    def _get_live_encoder(self):
        if self._live_encoder is None:
            from sentence_transformers import SentenceTransformer
            self._live_encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
        return self._live_encoder

    def _get_embedding(self, item: UniversalMediaItem) -> Optional[np.ndarray]:
        if item.item_id is not None and self.embeddings is not None:
            idx = self.item_id_to_emb_idx.get(item.item_id)
            if idx is not None:
                return self.embeddings[idx]
        
        text = item.overview or item.title or ""
        if len(text.strip()) < 10:
            return None
        
        # Ленивая загрузка модели SBERT
        if self._live_encoder is None:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Загрузка живого энкодера на {device}...")
            self._live_encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

        # Кодируем текст (это очень быстро)
        # Мы нормализуем вектор сразу, чтобы потом просто делать dot product
        vec = self._live_encoder.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)

    def select_candidates_by_semantics(self, liked, candidate_meta, top_n=3000):
        """Pick the top-N most semantically-relevant candidate rows.

        A popularity-sorted head() acts as a popularity *filter* once the
        catalog outgrows top_n (e.g. 18k movies → only the most-popular ~17%
        ever enter the pool), which collapses recommendations toward
        blockbusters. Ranking the masked candidates by cosine to the user's
        semantic profile lets genre-relevant but less-popular items reach the
        ranker — the original intent of "find hidden gems similar by meaning".

        Popularity remains the tie-breaker / fallback ordering, so when no
        usable embeddings exist the behaviour is identical to the old head().
        """
        base = candidate_meta.sort_values('popularity', ascending=False)
        if base.empty or self.embeddings is None:
            return base.head(top_n)

        vecs = [v for v in (self._get_embedding(i) for i in liked) if v is not None]
        if not vecs:
            return base.head(top_n)
        prof = np.mean(vecs, axis=0)
        pnorm = np.linalg.norm(prof)
        if pnorm < 1e-8:
            return base.head(top_n)
        prof = prof / pnorm

        item_ids = base['item_id'].to_numpy()
        emb_idx = np.fromiter(
            (self.item_id_to_emb_idx.get(int(i), -1) for i in item_ids),
            dtype=np.int64, count=len(item_ids),
        )
        sims = np.full(len(item_ids), -np.inf, dtype=np.float32)
        has = emb_idx >= 0
        if has.any():
            mat = self.embeddings[emb_idx[has]]
            norms = np.linalg.norm(mat, axis=1) + 1e-8
            sims[has] = (mat @ prof) / norms
        # Relevance primary; stable sort keeps popularity order within ties
        # and for any item lacking an embedding (sim = -inf sinks to the end).
        order = np.argsort(-sims, kind="stable")
        return base.iloc[order].head(top_n)

    def compute_similarity(self, liked: List[UniversalMediaItem], candidates: List[UniversalMediaItem]) -> List[Tuple[UniversalMediaItem, float]]:
        if not liked or not candidates: return []

        user_allowed_genres = set()
        user_dna = set()
        for item in liked:
            user_allowed_genres.update(item.genres)
            user_dna.update(infer_media_dna(item))

        # 1. Semantic profile
        semantic_vecs = []
        for item in liked:
            emb = self._get_embedding(item)
            if emb is not None:
                semantic_vecs.append(emb)

        u_s_prof = np.mean(semantic_vecs, axis=0) if semantic_vecs else None

        # 2. Genre profile
        genre_vecs = [self._encode_genres_idf(item.genres) for item in liked]
        u_g_prof = np.mean(genre_vecs, axis=0)
        
        # 3. Keyword profile
        keyword_vecs = [self._encode_keywords_idf(item.keywords) for item in liked]
        u_k_prof = np.mean(keyword_vecs, axis=0) if keyword_vecs else np.zeros(self.num_keywords)

        scored = []
        for c in candidates:
            # A. Genre similarity
            g_sim = self._genre_idf_cosine(u_g_prof, c.genres)
            
            # B. Keyword similarity
            k_sim = 0.0
            if np.linalg.norm(u_k_prof) > 0:
                c_k_vec = self._encode_keywords_idf(c.keywords)
                cn = np.linalg.norm(c_k_vec)
                if cn > 1e-8:
                    k_sim = np.dot(u_k_prof, c_k_vec) / (np.linalg.norm(u_k_prof) * cn)

            # C. Semantic similarity
            s_sim = 0.0
            if u_s_prof is not None:
                c_emb = self._get_embedding(c)
                if c_emb is not None:
                    s_sim = np.dot(u_s_prof, c_emb) / (np.linalg.norm(u_s_prof) * np.linalg.norm(c_emb) + 1e-8)

            # D. Penalties & Dampening
            cand_genres = set(c.genres)
            extra_genres = cand_genres - user_allowed_genres
            genre_penalty = 0.15 * len(extra_genres)
            
            pop_dampening = 0.01 * math.log1p(c.popularity)
            quality = c.vote_average / 10.0

            # FUSION FORMULA v2
            final_score = (
                0.40 * max(0, s_sim) +
                0.35 * max(0, k_sim) +
                0.15 * g_sim +
                0.10 * quality -
                genre_penalty -
                pop_dampening
            )
            
            # E. Media DNA Adjustments
            candidate_dna = infer_media_dna(c)
            
            # Hard filter: penalize anime if user never liked it
            if 'anime' not in user_dna and 'anime' in candidate_dna:
                final_score *= 0.3
                
            # Soft boost: matching tone tags
            tone_overlap = len(user_dna & candidate_dna & TONE_TAGS)
            final_score += 0.05 * tone_overlap

            scored.append((c, max(0.0, float(final_score))))

        return sorted(scored, key=lambda x: -x[1])

# =============================================================================
# UNIVERSAL SEARCH ENGINE
# =============================================================================

class UniversalSearchEngine:
    TV_OFFSET = 10000000

    def __init__(self, metadata, cache_dir, tmdb_api_key=None, inference_engine=None, embeddings_path=None, model_num_items=0):
        self.metadata = metadata
        self.inference_engine = inference_engine
        self.model_num_items = model_num_items
        # LightGCN popularity de-bias strength (0 = off). Tunable per engine;
        # see InferenceEngine._popularity_penalty.
        self.popularity_debias = 0.0

        self.tmdb_to_item_id = {int(r['tmdb_id']): int(r['item_id']) for _, r in metadata.dropna(subset=['tmdb_id']).iterrows()}

        self.hot_cache = HotCache(cache_dir / 'hot_items.db')
        self.tmdb_client = TMDBLiveClient(tmdb_api_key) if tmdb_api_key else None

        all_gs = set()
        all_ks = set()
        for g in metadata['genres']: all_gs.update(ensure_genres(g))
        if 'keywords' in metadata.columns:
            for k in metadata['keywords']: all_ks.update(ensure_list(k))
            
        self.content_recommender = EnhancedContentEngine(sorted(all_gs), sorted(all_ks), metadata, embeddings_path)

    def _row_to_universal(self, row) -> UniversalMediaItem:
        iid = int(row['item_id'])
        has_emb = iid < self.model_num_items
        return UniversalMediaItem(
            tmdb_id=int(row['tmdb_id']),
            media_type=row['type'],
            title=row['title'],
            title_ru=row.get('title_ru'),
            title_uk=row.get('title_uk'),
            year=int(row['year']),
            genres=ensure_genres(row['genres']),
            keywords=ensure_list(row.get('keywords', [])),
            item_id=iid,
            has_embeddings=has_emb,
            in_graph=True,
            source='trained' if has_emb else 'catalog',
            overview=row.get('overview'),
            overview_ru=row.get('overview_ru'),
            overview_uk=row.get('overview_uk'),
            vote_average=float(row.get('vote_average', 0)),
            vote_count=int(row.get('vote_count', 0)),
            popularity=float(row.get('popularity', 0))
        )

    def _title_relevance(self, query: str, title: str, popularity: float) -> float:
        if not title:
            return 0.0
        q = query.lower().strip()
        t = title.lower().strip()
        
        if t == q:
            return 1.0
        
        if t.startswith(q):
            return 0.9
        
        title_words = set(t.replace(':', ' ').replace('-', ' ').split())
        query_words = set(q.replace(':', ' ').replace('-', ' ').split())
        
        if query_words and query_words.issubset(title_words):
            overlap_ratio = len(query_words) / max(len(title_words), 1)
            return 0.6 + 0.3 * overlap_ratio
        
        if q in t:
            return 0.5
        
        ratio = SequenceMatcher(None, q, t).ratio()
        pop_bonus = min(0.1, 0.02 * math.log1p(popularity))
        
        return max(0.0, ratio - 0.3) + pop_bonus

    def search(self, query, media_type=None, limit=10) -> SearchResult:
        items, seen = [], set()
        q = query.lower().strip()

        # Match against any of the available localized title columns. Each
        # column is independent — a uk query like "Початок" hits via
        # title_uk, "Inception" via title, "Начало" via title_ru.
        # Numeric variants (Se7en ↔ seven, M3GAN ↔ megan) are OR'd in
        # via title_normalizer.normalize_for_search.
        title_lower = self.metadata['title'].str.lower()
        ru_lower = (
            self.metadata['title_ru'].fillna('').str.lower()
            if 'title_ru' in self.metadata.columns else None
        )
        uk_lower = (
            self.metadata['title_uk'].fillna('').str.lower()
            if 'title_uk' in self.metadata.columns else None
        )

        variants = normalize_for_search(query) or [q]
        mask = pd.Series(False, index=self.metadata.index)
        for v in variants:
            sub = title_lower.str.contains(v, na=False, regex=False)
            if ru_lower is not None:
                sub |= ru_lower.str.contains(v, na=False, regex=False)
            if uk_lower is not None:
                sub |= uk_lower.str.contains(v, na=False, regex=False)
            mask |= sub
        if media_type:
            mask &= (self.metadata['type'] == media_type)

        local_matches = self.metadata[mask].copy()
        
        local_matches['_exact'] = local_matches['title'].str.lower() == q
        local_matches['_starts'] = local_matches['title'].str.lower().str.startswith(q)
        local_matches['_title_len'] = local_matches['title'].str.len()
        
        local_matches = local_matches.sort_values(
            ['_exact', '_starts', 'popularity', '_title_len'],
            ascending=[False, False, False, True]
        )
        
        for _, r in local_matches.head(limit).iterrows():
            item = self._row_to_universal(r)
            items.append(item)
            seen.add(item.tmdb_id)

        for item in self.hot_cache.search(query, media_type, limit):
            if item.tmdb_id not in seen:
                items.append(item)
                seen.add(item.tmdb_id)

        if len(items) < limit and self.tmdb_client:
            live_results = self.tmdb_client.search_multi(query, limit=limit * 2)
            
            scored_live = []
            for item in live_results:
                rel_score = self._title_relevance(q, item.title, item.popularity)
                scored_live.append((item, rel_score))
            
            scored_live.sort(key=lambda x: -x[1])
            
            for item, rel_score in scored_live:
                if rel_score < 0.3:
                    continue
                    
                if item.popularity < 10:
                    continue
                
                if item.media_type == 'tv' and item.tmdb_id < self.TV_OFFSET:
                    item.tmdb_id += self.TV_OFFSET

                if item.tmdb_id not in seen:
                    if item.tmdb_id in self.tmdb_to_item_id:
                        iid = self.tmdb_to_item_id[item.tmdb_id]
                        item.item_id = iid
                        item.has_embeddings = iid < self.model_num_items
                        item.source = 'trained' if item.has_embeddings else 'catalog'

                    items.append(item)
                    seen.add(item.tmdb_id)
            
            new_items = [i for i in items if i.source == 'tmdb_live']
            if new_items:
                self.hot_cache.upsert_batch(new_items)

        return SearchResult(query=query, results=items[:limit])

    def get_recommendations(self, liked_item_ids=None, liked_tmdb_ids=None, top_k=10, media_type=None):
        liked_all = []
        seen_initial = set()

        # --- 1. Сбор всех лайков (база для анализа) ---
        if liked_item_ids:
            for iid in liked_item_ids:
                row = self.metadata[self.metadata['item_id'] == iid]
                if not row.empty:
                    item = self._row_to_universal(row.iloc[0])
                    liked_all.append(item)
                    seen_initial.add(item.tmdb_id)

        if liked_tmdb_ids:
            for tid in liked_tmdb_ids:
                # Учитываем смещение для сериалов, если нужно
                actual_tid = tid + self.TV_OFFSET if tid < self.TV_OFFSET and media_type == 'tv' else tid
                if actual_tid in seen_initial: continue

                item = self.hot_cache.get_by_tmdb_id(actual_tid) or \
                       self.hot_cache.get_by_tmdb_id(tid) or \
                       (self.tmdb_client.get_details(tid, media_type or 'movie') if self.tmdb_client else None)

                if item:
                    if item.media_type == 'tv' and item.tmdb_id < self.TV_OFFSET:
                        item.tmdb_id += self.TV_OFFSET
                    liked_all.append(item)
                    seen_initial.add(item.tmdb_id)

        if not liked_all: return []

        # --- 2. Кластеризация интентов (Разделяем "The Boys" и "Shogun") ---
        # Эта функция разбивает список лайков на группы на основе семантики описаний
        intents = detect_user_intents(liked_all, self.content_recommender._get_embedding)

        results = []
        seen_recs = set(seen_initial)

        # Сортируем кластеры: сначала те, где больше лайков
        intents = sorted(intents, key=len, reverse=True)

        # --- 3. Цикл по каждому интересу пользователя ---
        for intent_group in intents:
            # Считаем, сколько рекомендаций выделить на этот конкретный интерес
            # Если у пользователя 10 лайков и 5 из них в этом кластере, отдаем ему 50% квоты
            cluster_ratio = len(intent_group) / len(liked_all)
            quota = max(2, math.ceil(top_k * cluster_ratio))

            intent_results = []

            # ТИР 1: LightGCN (для всех items с реальными эмбеддингами, включая TV из Trakt)
            if self.inference_engine:
                liked_trained = [i.item_id for i in intent_group if i.has_embeddings]
                if liked_trained:
                    # Берем чуть больше, чтобы потом отфильтровать дубликаты
                    gcn_recs = self.inference_engine.get_recommendations(
                        liked_trained, top_k=quota * 2,
                        popularity_debias=self.popularity_debias,
                    )
                    for r in gcn_recs:
                        res_row = self.metadata[self.metadata['item_id'] == r['item_id']]
                        if not res_row.empty:
                            item = self._row_to_universal(res_row.iloc[0])
                            if item.tmdb_id not in seen_recs:
                                intent_results.append(item)
                                seen_recs.add(item.tmdb_id)
                        if len(intent_results) >= quota // 2: break  # GCN отдает половину квоты

            # ТИР 2: Контентный движок (работает для всего, особенно для TV)
            rem = quota - len(intent_results)
            if rem > 0:
                # Фильтруем общую базу по типу (фильм/сериал)
                mask = (self.metadata['type'] == media_type) if media_type else pd.Series(True,
                                                                                          index=self.metadata.index)
                mask &= ~self.metadata['tmdb_id'].isin(seen_recs)

                # Пул кандидатов — top-3000 по семантической близости к лайкам,
                # а не по популярности: на большом каталоге популярностный head()
                # превращался в фильтр и схлопывал выдачу в блокбастеры.
                cand_rows = self.content_recommender.select_candidates_by_semantics(
                    intent_group, self.metadata[mask], top_n=3000
                )
                candidates = [self._row_to_universal(r) for _, r in cand_rows.iterrows()]

                # Вычисляем сходство (здесь работают эмбеддинги, ключевые слова и Media DNA)
                ranked = self.content_recommender.compute_similarity(intent_group, candidates)

                for item, score in ranked:
                    if item.tmdb_id not in seen_recs:
                        item.source = 'catalog'
                        intent_results.append(item)
                        seen_recs.add(item.tmdb_id)
                    if len(intent_results) >= quota: break

            results.extend(intent_results)
            if len(results) >= top_k + 5: break  # Берем с небольшим запасом

        # Перемешиваем результаты из разных кластеров для разнообразия,
        # но сохраняем лучшие рекомендации в начале
        return results[:top_k]

    def refresh_trending(self):
        if not self.tmdb_client: return 0
        m = self.tmdb_client.get_trending('movie', limit=20)
        t = self.tmdb_client.get_trending('tv', limit=20)

        for item in t:
            if item.tmdb_id < self.TV_OFFSET:
                item.tmdb_id += self.TV_OFFSET

        self.hot_cache.upsert_batch(m + t)
        return len(m) + len(t)

class TMDBLiveClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.session = requests.Session()
        self.BASE_URL = "https://api.themoviedb.org/3"
        self.GENRE_MAP = {28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Science Fiction", 10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western", 10759: "Action & Adventure", 10762: "Kids", 10765: "Sci-Fi & Fantasy", 10766: "Soap", 10767: "Talk", 10768: "War & Politics"}

    def _genres_from_ids(self, ids):
        raw = [self.GENRE_MAP[gid] for gid in ids if gid in self.GENRE_MAP]
        return normalize_genres(raw)

    def _extract_year(self, data):
        d = data.get('release_date') or data.get('first_air_date')
        return int(d[:4]) if d else 0

    def search_multi(self, query, limit=10):
        url = f"{self.BASE_URL}/search/multi"
        r = self.session.get(url, params={'api_key': self.api_key, 'query': query, 'language': 'en-US'}).json()
        items = []
        for res in r.get('results', []):
            mt = res.get('media_type')
            if mt not in ('movie', 'tv'): continue
            item = UniversalMediaItem(tmdb_id=res['id'], media_type=mt, title=res.get('title') or res.get('name'), year=self._extract_year(res), genres=self._genres_from_ids(res.get('genre_ids', [])), keywords=[], popularity=res.get('popularity', 0), vote_average=res.get('vote_average', 0), vote_count=res.get('vote_count', 0), overview=res.get('overview'), source='tmdb_live')
            items.append(item)
        return items[:limit]

    def get_trending(self, media_type, limit=20):
        url = f"{self.BASE_URL}/trending/{media_type}/week"
        r = self.session.get(url, params={'api_key': self.api_key}).json()
        return [UniversalMediaItem(tmdb_id=res['id'], media_type=media_type, title=res.get('title') or res.get('name'), year=self._extract_year(res), genres=self._genres_from_ids(res.get('genre_ids', [])), keywords=[], popularity=res.get('popularity', 0), vote_average=res.get('vote_average', 0), vote_count=res.get('vote_count', 0), source='trending') for res in r.get('results', [])[:limit]]

    def get_details(self, tid, mt='movie'):
        url = f"{self.BASE_URL}/{mt}/{tid}"
        r = self.session.get(url, params={'api_key': self.api_key}).json()
        if 'id' not in r: return None
        return UniversalMediaItem(tmdb_id=r['id'], media_type=mt, title=r.get('title') or r.get('name'), year=self._extract_year(r), genres=[g['name'] for g in r.get('genres', [])], keywords=[], overview=r.get('overview'), vote_average=r.get('vote_average', 0), vote_count=r.get('vote_count', 0), popularity=r.get('popularity', 0))

class TrendingUpdater:
    def __init__(self, engine, interval_hours=12):
        self.engine = engine
        self.interval = interval_hours * 3600
        self._running = False
    async def start(self):
        self._running = True
        while self._running:
            try:
                await asyncio.get_event_loop().run_in_executor(None, self.engine.refresh_trending)
            except Exception:
                logger.exception("trending refresh failed")
            await asyncio.sleep(self.interval)
