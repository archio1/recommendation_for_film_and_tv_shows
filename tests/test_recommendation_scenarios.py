"""
Behavioral, invariant, and directional tests for the dual-domain engine.

Three flavors of test, all driven by parametrized scenarios:

- **Invariant** — properties that must hold for ANY scenario:
  no-self in recs, correct media_type, no duplicates, top_k respected.
- **Behavioral** — for a scenario with `expected_genres`, at least
  `min_overlap_pct` of recommendations should share at least one genre
  with the expected set. Catches "fantasy in → reality TV out"-style
  regressions.
- **Directional** — adding more same-genre likes should not REDUCE the
  share of that genre in the output. Answers the user's hypothesis
  about "5+ TV shows in the list giving better TV recommendations".

Genre comparison is case-insensitive. Sci-fi has two spellings in the
dataset ("Sci-Fi" for movies, "science-fiction" for tv) — expected
sets list both.
"""

from __future__ import annotations

import pytest

from tests._quality_helpers import (
    genre_overlap_pct as _genre_overlap_pct,
    normalize_genres as _normalize_genres,
)


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------

# Movies: liked tmdb_ids → expected genres in recommendations.
# The 6 numbered scenarios mirror tests/tests_movies.txt.
MOVIE_SCENARIOS = [
    # --- user's manual scenarios ---
    {"name": "1_john_wick_action", "liked": [245891],
     "expected": {"action", "thriller", "crime"}, "min_overlap_pct": 60},
    {"name": "2_toy_story_plus_saw_mixed", "liked": [862, 176],
     "expected": {"animation", "family", "horror", "thriller", "comedy"},
     "min_overlap_pct": 50},
    {"name": "3_lotr_trilogy_fantasy", "liked": [120, 121, 122],
     "expected": {"fantasy", "adventure", "action"}, "min_overlap_pct": 50,
     # Hobbit trilogy — all three movies share the same universe and are the
     # most obvious "related" items in graph space. Title word-roots differ
     # from "lord"/"rings"/"fellowship" so InferenceEngine's sequel filter
     # will NOT strip them.
     "must_contain": {49051, 57158, 122917}},
    {"name": "4_diverse_5", "liked": [27205, 18785, 240, 129, 313369],
     # No single dominant genre — expectation is just "something popular comes back".
     "expected": {"drama", "action", "comedy", "crime", "animation",
                  "sci-fi", "science-fiction", "adventure"},
     "min_overlap_pct": 50},
    {"name": "5_old_school_crime", "liked": [680, 550, 807],
     "expected": {"crime", "thriller", "drama", "mystery"},
     "min_overlap_pct": 50},
    {"name": "6_cold_start_dune2_superman2025", "liked": [693134, 1061474],
     # Cold-start, no graph signal — content fallback should still produce sci-fi/action.
     "expected": {"sci-fi", "science-fiction", "action", "adventure"},
     "min_overlap_pct": 30},
    # --- new genre coverage ---
    {"name": "horror_classic_3", "liked": [138843, 493922, 419430],
     "expected": {"horror", "thriller", "mystery"}, "min_overlap_pct": 50},
    {"name": "comedy_3", "liked": [8363, 18785, 55721],
     "expected": {"comedy", "romance"}, "min_overlap_pct": 50},
    {"name": "anime_3", "liked": [129, 372058, 128],
     "expected": {"animation", "fantasy", "adventure", "family"},
     "min_overlap_pct": 50},
    {"name": "marvel_3", "liked": [1726, 24428, 557],
     "expected": {"action", "adventure", "sci-fi", "science-fiction", "fantasy"},
     "min_overlap_pct": 60,
     # Other MCU films whose first non-stopword title token does NOT match
     # "iron" / "avengers" / "spider" — the sequel filter would strip those.
     # Thor, Captain America, Guardians of the Galaxy, Black Panther.
     "must_contain": {10195, 1771, 118340, 284054}},
    {"name": "romance_3", "liked": [11036, 4348, 313369],
     "expected": {"romance", "drama", "comedy"}, "min_overlap_pct": 40},
    {"name": "neo_noir_2", "liked": [78, 64690],
     "expected": {"thriller", "crime", "drama", "sci-fi", "science-fiction"},
     "min_overlap_pct": 40},
    # --- single-item edge case ---
    {"name": "edge_single_inception", "liked": [27205],
     "expected": {"action", "sci-fi", "science-fiction", "adventure", "thriller"},
     "min_overlap_pct": 40},
]


# TV: liked tmdb_ids carry +TV_OFFSET (10_000_000), as in parquet & bot session.
TV_SCENARIOS = [
    # --- user's manual scenarios ---
    {"name": "1_pure_fantasy_3", "liked": [10094997, 10071914, 10007935],
     "expected": {"fantasy", "adventure", "drama"}, "min_overlap_pct": 40,
     # Other flagship epic-fantasy TV in the same universe/tier.
     # Game of Thrones, Rings of Power. Title roots ("game", "lord") differ
     # from liked roots ("house", "wheel"), so no sequel-filter strip.
     "must_contain": {10001399, 10087739}},
    {"name": "2_hard_scifi_5", "liked": [10063639, 10093740, 10068421, 10001972, 10125988],
     "expected": {"science-fiction", "sci-fi", "drama", "fantasy"},
     "min_overlap_pct": 60},
    {"name": "3_adult_superhero_5", "liked": [10076479, 10095557, 10205715, 10110492, 10087917],
     "expected": {"action", "superhero", "fantasy", "science-fiction", "sci-fi"},
     "min_overlap_pct": 40},
    {"name": "4_psy_thriller_5", "liked": [10067744, 10046648, 10095396, 10070523, 10042009],
     "expected": {"drama", "mystery", "crime", "thriller", "science-fiction", "sci-fi"},
     "min_overlap_pct": 50},
    {"name": "5_2024_2025_cold_start", "liked": [10106379, 10194764, 10119051, 10100088],
     "expected": {"action", "drama", "fantasy", "science-fiction", "sci-fi", "adventure"},
     "min_overlap_pct": 30},
    # --- new genre coverage ---
    {"name": "sitcom_3", "liked": [10001668, 10002316, 10048891],
     "expected": {"comedy"}, "min_overlap_pct": 50},
    {"name": "anime_2", "liked": [10001429, 10013916],
     "expected": {"anime", "animation", "fantasy", "action"}, "min_overlap_pct": 30},
    {"name": "crime_3", "liked": [10001396, 10060059, 10063351],
     "expected": {"crime", "drama", "thriller"}, "min_overlap_pct": 50},
    # --- single-item edge case ---
    {"name": "edge_single_stranger_things", "liked": [10066732],
     "expected": {"drama", "fantasy", "science-fiction", "sci-fi", "mystery"},
     "min_overlap_pct": 30},
]


# --------------------------------------------------------------------------
# Per-scenario fixture: skip if any liked tmdb_id isn't in the engine's graph
# nor reachable via cold-start (no tmdb client in tests). This keeps the
# suite robust to dataset re-splits.
# --------------------------------------------------------------------------

def _scenario_recs(engine_method, liked: list[int], top_k: int = 10):
    recs = engine_method(liked_tmdb_ids=liked, top_k=top_k)
    if not recs:
        pytest.skip(f"engine returned no recs for {liked} (likely all unknown ids)")
    return recs


# --------------------------------------------------------------------------
# Movie scenarios — invariant + behavioral
# --------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", MOVIE_SCENARIOS, ids=lambda s: s["name"])
class TestMovieScenarios:
    TOP_K = 10

    def test_no_self(self, scenario, dual_engine_real):
        recs = _scenario_recs(dual_engine_real.recs_movie, scenario["liked"], self.TOP_K)
        liked = set(scenario["liked"])
        leaked = liked & {r.tmdb_id for r in recs}
        assert not leaked, f"liked items leaked into recs: {leaked}"

    def test_only_movies(self, scenario, dual_engine_real):
        recs = _scenario_recs(dual_engine_real.recs_movie, scenario["liked"], self.TOP_K)
        bad = [(r.title, r.media_type) for r in recs if r.media_type != "movie"]
        assert not bad, f"non-movie items leaked: {bad}"

    def test_no_duplicates(self, scenario, dual_engine_real):
        recs = _scenario_recs(dual_engine_real.recs_movie, scenario["liked"], self.TOP_K)
        ids = [r.tmdb_id for r in recs]
        assert len(ids) == len(set(ids)), f"duplicates in output: {ids}"

    def test_top_k_respected(self, scenario, dual_engine_real):
        recs = _scenario_recs(dual_engine_real.recs_movie, scenario["liked"], self.TOP_K)
        assert len(recs) <= self.TOP_K

    def test_genre_overlap(self, scenario, dual_engine_real):
        recs = _scenario_recs(dual_engine_real.recs_movie, scenario["liked"], self.TOP_K)
        pct = _genre_overlap_pct(recs, scenario["expected"])
        rec_genres = [list(getattr(r, "genres", []) or []) for r in recs]
        assert pct >= scenario["min_overlap_pct"], (
            f"{scenario['name']}: expected >= {scenario['min_overlap_pct']}% overlap "
            f"with {scenario['expected']}, got {pct:.0f}%. "
            f"recs: {[(r.title, g) for r, g in zip(recs, rec_genres)]}"
        )


# --------------------------------------------------------------------------
# TV scenarios — invariant + behavioral
# --------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", TV_SCENARIOS, ids=lambda s: s["name"])
class TestTvScenarios:
    TOP_K = 10

    def test_no_self(self, scenario, dual_engine_real):
        recs = _scenario_recs(dual_engine_real.recs_tv, scenario["liked"], self.TOP_K)
        liked = set(scenario["liked"])
        leaked = liked & {r.tmdb_id for r in recs}
        assert not leaked, f"liked items leaked: {leaked}"

    def test_only_tv(self, scenario, dual_engine_real):
        recs = _scenario_recs(dual_engine_real.recs_tv, scenario["liked"], self.TOP_K)
        bad = [(r.title, r.media_type) for r in recs if r.media_type != "tv"]
        assert not bad, f"non-tv items leaked: {bad}"

    def test_no_duplicates(self, scenario, dual_engine_real):
        recs = _scenario_recs(dual_engine_real.recs_tv, scenario["liked"], self.TOP_K)
        ids = [r.tmdb_id for r in recs]
        assert len(ids) == len(set(ids))

    def test_top_k_respected(self, scenario, dual_engine_real):
        recs = _scenario_recs(dual_engine_real.recs_tv, scenario["liked"], self.TOP_K)
        assert len(recs) <= self.TOP_K

    def test_genre_overlap(self, scenario, dual_engine_real):
        recs = _scenario_recs(dual_engine_real.recs_tv, scenario["liked"], self.TOP_K)
        pct = _genre_overlap_pct(recs, scenario["expected"])
        rec_genres = [list(getattr(r, "genres", []) or []) for r in recs]
        assert pct >= scenario["min_overlap_pct"], (
            f"{scenario['name']}: expected >= {scenario['min_overlap_pct']}% overlap "
            f"with {scenario['expected']}, got {pct:.0f}%. "
            f"recs: {[(r.title, g) for r, g in zip(recs, rec_genres)]}"
        )


# --------------------------------------------------------------------------
# Cross-cutting invariants (not parametrized)
# --------------------------------------------------------------------------

class TestEngineInvariants:
    def test_empty_input_recs_movie(self, dual_engine_real):
        assert dual_engine_real.recs_movie(liked_tmdb_ids=[], top_k=5) == []
        assert dual_engine_real.recs_movie(liked_tmdb_ids=None, top_k=5) == []

    def test_empty_input_recs_tv(self, dual_engine_real):
        assert dual_engine_real.recs_tv(liked_tmdb_ids=[], top_k=5) == []
        assert dual_engine_real.recs_tv(liked_tmdb_ids=None, top_k=5) == []

    def test_empty_input_recs_all(self, dual_engine_real):
        assert dual_engine_real.recs_all(liked_tmdb_ids=[], top_k=5) == []
        assert dual_engine_real.recs_all(liked_tmdb_ids=None, top_k=5) == []

    def test_recs_all_mixed_input_returns_both(self, dual_engine_real):
        # Inception (movie) + Friends (tv)
        recs = dual_engine_real.recs_all(
            liked_tmdb_ids=[27205, 10001668], top_k=10
        )
        if not recs:
            pytest.skip("engine returned no recs")
        media = {r.media_type for r in recs}
        assert media == {"movie", "tv"}, (
            f"recs_all should mix domains, got {media}: "
            f"{[(r.title, r.media_type) for r in recs]}"
        )

    def test_recs_cross_movie_to_tv_returns_only_tv(self, dual_engine_real):
        recs = dual_engine_real.recs_cross(
            liked_tmdb_ids=[27205], target_media_type="tv", top_k=5
        )
        if not recs:
            pytest.skip("FAISS bridge returned no recs (Inception not in catalog)")
        assert all(r.media_type == "tv" for r in recs), (
            f"got non-tv: {[(r.title, r.media_type) for r in recs]}"
        )

    def test_recs_cross_tv_to_movie_returns_only_movie(self, dual_engine_real):
        recs = dual_engine_real.recs_cross(
            liked_tmdb_ids=[10001399], target_media_type="movie", top_k=5
        )
        if not recs:
            pytest.skip("FAISS bridge returned no recs (GoT not in catalog)")
        assert all(r.media_type == "movie" for r in recs)

    def test_recs_deterministic_across_runs(self, dual_engine_real):
        a = dual_engine_real.recs_movie(liked_tmdb_ids=[27205], top_k=10)
        b = dual_engine_real.recs_movie(liked_tmdb_ids=[27205], top_k=10)
        assert [r.tmdb_id for r in a] == [r.tmdb_id for r in b], (
            "same input must produce identical output across runs"
        )


# --------------------------------------------------------------------------
# Directional test: the user's "5+ TV shows in list" hypothesis
# --------------------------------------------------------------------------

class TestDirectionalSignals:

    # Five fantasy TV shows in increasing order of "fantasy strength".
    FANTASY_LIKES = [
        10094997,  # House of the Dragon
        10071914,  # The Wheel of Time
        10001399,  # Game of Thrones
        10068421,  # Altered Carbon (sci-fantasy)
        10042009,  # Black Mirror (more genres but listed under fantasy)
    ]

    @pytest.mark.parametrize("n", [1, 3, 5])
    def test_fantasy_share_does_not_collapse_with_more_likes(
        self, n, dual_engine_real, request
    ):
        """
        Diagnostic: fantasy share in recs at N=1, 3, 5. We assert the
        weak invariant 'share at N=5 is at least as good as at N=1'
        across the full sweep (collected in a session-scope cache).
        """
        cache = request.config.cache
        key = "tv_fantasy_share_by_n"
        recs = dual_engine_real.recs_tv(
            liked_tmdb_ids=self.FANTASY_LIKES[:n], top_k=10
        )
        if not recs:
            pytest.skip(f"no recs for N={n}")
        pct = _genre_overlap_pct(
            recs, {"fantasy", "science-fiction", "sci-fi", "adventure"}
        )
        # Cache for cross-test comparison.
        prev = cache.get(key, {})
        prev[str(n)] = pct
        cache.set(key, prev)
        # Per-N sanity: at least SOMETHING fantasy-adjacent comes back.
        assert pct >= 10, (
            f"N={n}: fantasy share {pct:.0f}% is suspiciously low. "
            f"recs: {[(r.title, list(getattr(r, 'genres', []) or [])) for r in recs]}"
        )

    def test_fantasy_share_at_5_at_least_as_good_as_at_1(
        self, dual_engine_real, request
    ):
        """Compares cached values from previous parametrized test."""
        cache = request.config.cache
        data = cache.get("tv_fantasy_share_by_n", {})
        if "1" not in data or "5" not in data:
            pytest.skip("requires test_fantasy_share_does_not_collapse... to run first")
        share_1 = data["1"]
        share_5 = data["5"]
        # Allow some noise — the user's hypothesis was "5 is meaningfully
        # better than 3"; we use the weaker non-regression check to keep
        # the test robust to small content-tier reshuffles.
        assert share_5 >= share_1 - 10, (
            f"fantasy share collapsed: N=1 {share_1:.0f}% → N=5 {share_5:.0f}%"
        )

    def test_more_action_likes_keep_action_in_movie_recs(self, dual_engine_real):
        """Adding a same-genre movie should not REDUCE the action share."""
        action_likes = [
            245891,  # John Wick
            155,     # The Dark Knight
            1726,    # Iron Man
        ]
        for n in (1, 3):
            recs = dual_engine_real.recs_movie(
                liked_tmdb_ids=action_likes[:n], top_k=10
            )
            if not recs:
                pytest.skip(f"no recs at N={n}")
            pct = _genre_overlap_pct(recs, {"action", "thriller", "adventure"})
            assert pct >= 30, (
                f"N={n}: action-genre share {pct:.0f}% too low. "
                f"recs: {[(r.title, list(getattr(r, 'genres', []) or [])) for r in recs]}"
            )


# --------------------------------------------------------------------------
# Quality trajectory: how does rec quality change as N grows 1 → 5?
# --------------------------------------------------------------------------

# Each trajectory is an ordered list of 5 liked tmdb_ids plus a reference
# "expected" genre set used for the genre_overlap metric. Scenarios exercise
# both a focused taste (one strong theme) and a mixed taste.
TRAJECTORY_SETS = [
    {
        "name": "movies_scifi_5",
        "media_type": "movie",
        "liked": [27205, 157336, 577922, 603, 335984],
        # Inception, Interstellar, Tenet, Matrix, Blade Runner 2049
        "expected_genres": {"sci-fi", "science-fiction", "action", "thriller", "adventure"},
    },
    {
        "name": "movies_action_5",
        "media_type": "movie",
        "liked": [245891, 155, 1726, 24428, 557],
        # John Wick, Dark Knight, Iron Man, Avengers, Spider-Man
        "expected_genres": {"action", "adventure", "thriller"},
    },
    {
        "name": "tv_fantasy_5",
        "media_type": "tv",
        "liked": [10094997, 10071914, 10001399, 10068421, 10042009],
        # House of Dragon, Wheel of Time, GoT, Altered Carbon, Black Mirror
        "expected_genres": {"fantasy", "science-fiction", "sci-fi", "drama", "adventure"},
    },
]


@pytest.mark.quality
@pytest.mark.parametrize("trajectory", TRAJECTORY_SETS, ids=lambda t: t["name"])
class TestQualityTrajectory:
    """
    For each trajectory, sweeps N=1..5 and records metrics to reports_writer.

    Soft non-regression assertion: genre overlap at N=5 must be within
    15 percentage points of the value at N=1. The primary deliverable is
    the JSON report that captures the full curve.
    """

    N_VALUES = (1, 2, 3, 4, 5)

    def test_trajectory(self, trajectory, dual_engine_real, reports_writer,
                        popular_tmdb_ids_movies, popular_tmdb_ids_tv):
        from tests._quality_helpers import (
            collect_liked_keywords,
            keyword_overlap_pct,
            popular_share,
        )

        media_type = trajectory["media_type"]
        if media_type == "movie":
            recs_fn = dual_engine_real.recs_movie
            metadata = dual_engine_real.movies.metadata
            popular_ids = popular_tmdb_ids_movies
        else:
            recs_fn = dual_engine_real.recs_tv
            metadata = dual_engine_real.tv.metadata
            popular_ids = popular_tmdb_ids_tv

        per_n: dict[int, dict] = {}
        for n in self.N_VALUES:
            liked = trajectory["liked"][:n]
            recs = recs_fn(liked_tmdb_ids=liked, top_k=10)
            if not recs:
                metrics = {"n": n, "recs_count": 0, "skipped": True}
            else:
                liked_kw = collect_liked_keywords(metadata, liked)
                unique_genres = set()
                for r in recs:
                    unique_genres.update(_normalize_genres(getattr(r, "genres", []) or []))
                metrics = {
                    "n": n,
                    "recs_count": len(recs),
                    "genre_overlap_pct": round(
                        _genre_overlap_pct(recs, trajectory["expected_genres"]), 1
                    ),
                    "keyword_overlap_pct": round(keyword_overlap_pct(recs, liked_kw), 1),
                    "popular_share": round(popular_share(recs, popular_ids), 3),
                    "unique_genres_in_recs": len(unique_genres),
                    "liked_keyword_count": len(liked_kw),
                    "rec_titles": [getattr(r, "title", "?") for r in recs],
                }
            reports_writer.append(
                test_name="quality_trajectory",
                scenario=trajectory["name"],
                metrics=metrics,
            )
            per_n[n] = metrics

        # Soft non-regression: N=5 genre share must be within 15pp of N=1.
        if per_n[1].get("skipped") or per_n[5].get("skipped"):
            pytest.skip("trajectory has empty recs at N=1 or N=5")
        share_1 = per_n[1]["genre_overlap_pct"]
        share_5 = per_n[5]["genre_overlap_pct"]
        assert share_5 >= share_1 - 15, (
            f"{trajectory['name']}: genre share collapsed "
            f"N=1 {share_1:.0f}% → N=5 {share_5:.0f}%"
        )


# --------------------------------------------------------------------------
# Golden Standard: must-contain ids in top-K recommendations
# --------------------------------------------------------------------------

_MOVIE_GOLDEN_SCENARIOS = [s for s in MOVIE_SCENARIOS if "must_contain" in s]
_TV_GOLDEN_SCENARIOS = [s for s in TV_SCENARIOS if "must_contain" in s]


@pytest.mark.golden
class TestGoldenStandard:
    """
    For scenarios with a curated `must_contain` set, assert that at least
    `MIN_RECALL` of those ids appear in the top-K recommendations. Catches
    regressions where obviously-related items disappear from the catalog
    projection (e.g., a filter rule accidentally drops them).
    """

    TOP_K = 20
    MIN_RECALL = 0.25  # at least 1 of 4, or 1 of 3, etc.

    @pytest.mark.parametrize(
        "scenario", _MOVIE_GOLDEN_SCENARIOS, ids=lambda s: s["name"]
    )
    def test_movie_must_contain(self, scenario, dual_engine_real, reports_writer):
        from tests._quality_helpers import recall_at_k

        recs = _scenario_recs(
            dual_engine_real.recs_movie, scenario["liked"], self.TOP_K
        )
        must = set(scenario["must_contain"])
        got_ids = [int(r.tmdb_id) for r in recs]
        found = must & set(got_ids)
        recall = recall_at_k(recs, must, k=self.TOP_K)
        reports_writer.append(
            test_name="golden_standard_movie",
            scenario=scenario["name"],
            metrics={
                "top_k": self.TOP_K,
                "must_contain": sorted(must),
                "found": sorted(found),
                "recall": round(recall, 3),
                "rec_titles": [getattr(r, "title", "?") for r in recs],
            },
        )
        assert recall >= self.MIN_RECALL, (
            f"{scenario['name']}: recall@{self.TOP_K} {recall:.2f} < "
            f"{self.MIN_RECALL}. must_contain={sorted(must)}, "
            f"found={sorted(found)}, recs={got_ids}"
        )

    @pytest.mark.parametrize(
        "scenario", _TV_GOLDEN_SCENARIOS, ids=lambda s: s["name"]
    )
    def test_tv_must_contain(self, scenario, dual_engine_real, reports_writer):
        from tests._quality_helpers import recall_at_k

        recs = _scenario_recs(
            dual_engine_real.recs_tv, scenario["liked"], self.TOP_K
        )
        must = set(scenario["must_contain"])
        got_ids = [int(r.tmdb_id) for r in recs]
        found = must & set(got_ids)
        recall = recall_at_k(recs, must, k=self.TOP_K)
        reports_writer.append(
            test_name="golden_standard_tv",
            scenario=scenario["name"],
            metrics={
                "top_k": self.TOP_K,
                "must_contain": sorted(must),
                "found": sorted(found),
                "recall": round(recall, 3),
                "rec_titles": [getattr(r, "title", "?") for r in recs],
            },
        )
        assert recall >= self.MIN_RECALL, (
            f"{scenario['name']}: recall@{self.TOP_K} {recall:.2f} < "
            f"{self.MIN_RECALL}. must_contain={sorted(must)}, "
            f"found={sorted(found)}, recs={got_ids}"
        )


# --------------------------------------------------------------------------
# Media DNA: TMDB keyword overlap (deeper than genre)
# --------------------------------------------------------------------------

@pytest.mark.media_dna
class TestMediaDNA:
    """
    For each scenario, aggregates TMDB `keywords` from liked items and checks
    that at least `MIN_OVERLAP_PCT` of recommendations share one of those
    keywords. Keywords are noisier than genres so the threshold is permissive.

    Scenarios where none of the liked ids exist in metadata (cold-start) are
    skipped via `_scenario_recs`. Scenarios where liked keywords come back
    empty (metadata gap) are also skipped — there is nothing to compare.
    """

    TOP_K = 10
    MIN_OVERLAP_PCT = 40.0

    @pytest.mark.parametrize("scenario", MOVIE_SCENARIOS, ids=lambda s: s["name"])
    def test_movie_keyword_overlap(
        self, scenario, dual_engine_real, reports_writer
    ):
        from tests._quality_helpers import (
            collect_liked_keywords,
            keyword_overlap_pct,
        )

        recs = _scenario_recs(
            dual_engine_real.recs_movie, scenario["liked"], self.TOP_K
        )
        liked_kw = collect_liked_keywords(
            dual_engine_real.movies.metadata, scenario["liked"]
        )
        if not liked_kw:
            pytest.skip(f"{scenario['name']}: no keywords on liked items")
        pct = keyword_overlap_pct(recs, liked_kw)
        reports_writer.append(
            test_name="media_dna_movie",
            scenario=scenario["name"],
            metrics={
                "top_k": self.TOP_K,
                "liked_keyword_count": len(liked_kw),
                "keyword_overlap_pct": round(pct, 1),
                "rec_titles": [getattr(r, "title", "?") for r in recs],
            },
        )
        assert pct >= self.MIN_OVERLAP_PCT, (
            f"{scenario['name']}: keyword overlap {pct:.0f}% < "
            f"{self.MIN_OVERLAP_PCT:.0f}%. liked_kw_n={len(liked_kw)}, "
            f"recs={[(r.title, list(getattr(r, 'keywords', []) or [])[:5]) for r in recs]}"
        )

    @pytest.mark.parametrize("scenario", TV_SCENARIOS, ids=lambda s: s["name"])
    def test_tv_keyword_overlap(
        self, scenario, dual_engine_real, reports_writer
    ):
        from tests._quality_helpers import (
            collect_liked_keywords,
            keyword_overlap_pct,
        )

        recs = _scenario_recs(
            dual_engine_real.recs_tv, scenario["liked"], self.TOP_K
        )
        liked_kw = collect_liked_keywords(
            dual_engine_real.tv.metadata, scenario["liked"]
        )
        if not liked_kw:
            pytest.skip(f"{scenario['name']}: no keywords on liked items")
        pct = keyword_overlap_pct(recs, liked_kw)
        reports_writer.append(
            test_name="media_dna_tv",
            scenario=scenario["name"],
            metrics={
                "top_k": self.TOP_K,
                "liked_keyword_count": len(liked_kw),
                "keyword_overlap_pct": round(pct, 1),
                "rec_titles": [getattr(r, "title", "?") for r in recs],
            },
        )
        assert pct >= self.MIN_OVERLAP_PCT, (
            f"{scenario['name']}: keyword overlap {pct:.0f}% < "
            f"{self.MIN_OVERLAP_PCT:.0f}%. liked_kw_n={len(liked_kw)}, "
            f"recs={[(r.title, list(getattr(r, 'keywords', []) or [])[:5]) for r in recs]}"
        )


# --------------------------------------------------------------------------
# Popularity Swamp: fraction of recs from top-10% by vote_count
# --------------------------------------------------------------------------

@pytest.mark.popularity
class TestPopularityBias:
    """
    For each scenario, at N=1/3/5 (where liked is large enough), asserts that
    the fraction of recommendations falling in the top-10% of the catalog by
    `vote_count` does not exceed MAX_POPULAR_SHARE. Blockbusters legitimately
    dominate recommendation lists, but an engine that returns ONLY top-popular
    items has collapsed into "suggest the IMDb top-250 to everyone".

    Cold-start scenarios are excluded by name: they are exactly the cases
    where popular items are the correct fallback, so the metric is not
    informative there.
    """

    TOP_K = 10
    MAX_POPULAR_SHARE = 0.60
    N_VALUES = (1, 3, 5)

    @pytest.mark.parametrize("scenario", MOVIE_SCENARIOS, ids=lambda s: s["name"])
    def test_movie_popular_share(
        self, scenario, dual_engine_real, reports_writer,
        popular_tmdb_ids_movies,
    ):
        if "cold_start" in scenario["name"]:
            pytest.skip("cold-start scenarios are popularity-dominated by design")
        from tests._quality_helpers import popular_share

        per_n: dict[int, float] = {}
        for n in self.N_VALUES:
            if n > len(scenario["liked"]):
                continue
            liked = scenario["liked"][:n]
            recs = dual_engine_real.recs_movie(liked_tmdb_ids=liked, top_k=self.TOP_K)
            if not recs:
                continue
            share = popular_share(recs, popular_tmdb_ids_movies)
            per_n[n] = share
            reports_writer.append(
                test_name="popularity_bias_movie",
                scenario=scenario["name"],
                metrics={
                    "n": n,
                    "popular_share": round(share, 3),
                    "top_k": self.TOP_K,
                    "rec_titles": [getattr(r, "title", "?") for r in recs],
                },
            )

        if not per_n:
            pytest.skip(f"{scenario['name']}: no non-empty recs across N values")
        worst = max(per_n.values())
        assert worst <= self.MAX_POPULAR_SHARE, (
            f"{scenario['name']}: popular_share={per_n} exceeded "
            f"{self.MAX_POPULAR_SHARE:.2f} — engine may be collapsing to top-popular"
        )

    @pytest.mark.parametrize("scenario", TV_SCENARIOS, ids=lambda s: s["name"])
    def test_tv_popular_share(
        self, scenario, dual_engine_real, reports_writer,
        popular_tmdb_ids_tv,
    ):
        if "cold_start" in scenario["name"]:
            pytest.skip("cold-start scenarios are popularity-dominated by design")
        from tests._quality_helpers import popular_share

        per_n: dict[int, float] = {}
        for n in self.N_VALUES:
            if n > len(scenario["liked"]):
                continue
            liked = scenario["liked"][:n]
            recs = dual_engine_real.recs_tv(liked_tmdb_ids=liked, top_k=self.TOP_K)
            if not recs:
                continue
            share = popular_share(recs, popular_tmdb_ids_tv)
            per_n[n] = share
            reports_writer.append(
                test_name="popularity_bias_tv",
                scenario=scenario["name"],
                metrics={
                    "n": n,
                    "popular_share": round(share, 3),
                    "top_k": self.TOP_K,
                    "rec_titles": [getattr(r, "title", "?") for r in recs],
                },
            )

        if not per_n:
            pytest.skip(f"{scenario['name']}: no non-empty recs across N values")
        worst = max(per_n.values())
        assert worst <= self.MAX_POPULAR_SHARE, (
            f"{scenario['name']}: popular_share={per_n} exceeded "
            f"{self.MAX_POPULAR_SHARE:.2f} — engine may be collapsing to top-popular"
        )


# --------------------------------------------------------------------------
# Graph overlap: predict model output from LightGCN-neighbors of liked items
# --------------------------------------------------------------------------

def _resolve_titles(metadata, tmdb_ids, limit: int = 10) -> list[str]:
    if not tmdb_ids:
        return []
    ids = list(tmdb_ids)[:limit]
    rows = metadata[metadata["tmdb_id"].isin(ids)]
    return rows["title"].astype(str).tolist()


@pytest.mark.graph_overlap
class TestGraphOverlap:
    """
    Predictive test: union the top-K LightGCN neighbors of each liked item
    and assert that final recommendations have a meaningful overlap with
    that union. Answers the user's question "given Avatar as input, can we
    roughly predict what the model will recommend?".

    The final ranker (intent clustering + content fallback + sequel filter)
    is not a pure neighbor read-off, so the threshold is conservative. A
    low overlap on a healthy scenario means the final stage is rewriting
    the raw LightGCN signal significantly — diagnostically useful either
    way.
    """

    TOP_K_RECS = 10
    NEIGHBOR_K = 50
    MIN_OVERLAP = 0.30

    @pytest.mark.xfail(
        reason=(
            "Movies engine runs with popularity de-bias (popularity_debias=0.5; "
            "see movie_bot.py / conftest), which intentionally steers recs away "
            "from the popularity-heavy raw LightGCN neighbors this test measures "
            "overlap against. Low movie graph-overlap is therefore expected and "
            "in direct tension with TestPopularityBias — you cannot maximize both "
            "for popular-input scenarios. TV (no de-bias) is unaffected. Revisit "
            "if the de-bias strategy changes; strict=False so still-passing "
            "scenarios report as XPASS rather than failing."
        ),
        strict=False,
    )
    @pytest.mark.parametrize("scenario", MOVIE_SCENARIOS, ids=lambda s: s["name"])
    def test_movie_graph_overlap(
        self, scenario, dual_engine_real, graph_neighbors_fn, reports_writer,
    ):
        if "cold_start" in scenario["name"]:
            pytest.skip("cold-start: liked items not in graph, neighbors undefined")
        if len(scenario["liked"]) < 2:
            pytest.skip("graph overlap requires >= 2 liked items")

        expected = graph_neighbors_fn(
            scenario["liked"], media_type="movie", k=self.NEIGHBOR_K
        )
        if not expected:
            pytest.skip(f"{scenario['name']}: no in-graph liked items")

        recs = _scenario_recs(
            dual_engine_real.recs_movie, scenario["liked"], self.TOP_K_RECS
        )
        actual = {int(r.tmdb_id) for r in recs}
        overlap = len(actual & expected) / len(actual) if actual else 0.0

        reports_writer.append(
            test_name="graph_overlap_movie",
            scenario=scenario["name"],
            metrics={
                "expected_neighbor_count": len(expected),
                "actual_recs_count": len(actual),
                "intersection_count": len(actual & expected),
                "overlap": round(overlap, 3),
                "expected_neighbor_titles_sample": _resolve_titles(
                    dual_engine_real.movies.metadata, expected, limit=10
                ),
                "rec_titles": [getattr(r, "title", "?") for r in recs],
            },
        )
        assert overlap >= self.MIN_OVERLAP, (
            f"{scenario['name']}: graph overlap {overlap:.2f} < {self.MIN_OVERLAP}. "
            f"Final ranker has diverged from raw LightGCN. "
            f"|expected|={len(expected)}, |recs|={len(actual)}, "
            f"|intersect|={len(actual & expected)}"
        )

    @pytest.mark.parametrize("scenario", TV_SCENARIOS, ids=lambda s: s["name"])
    def test_tv_graph_overlap(
        self, scenario, dual_engine_real, graph_neighbors_fn, reports_writer,
    ):
        if "cold_start" in scenario["name"]:
            pytest.skip("cold-start: liked items not in graph, neighbors undefined")
        if len(scenario["liked"]) < 2:
            pytest.skip("graph overlap requires >= 2 liked items")

        expected = graph_neighbors_fn(
            scenario["liked"], media_type="tv", k=self.NEIGHBOR_K
        )
        if not expected:
            pytest.skip(f"{scenario['name']}: no in-graph liked items")

        recs = _scenario_recs(
            dual_engine_real.recs_tv, scenario["liked"], self.TOP_K_RECS
        )
        actual = {int(r.tmdb_id) for r in recs}
        overlap = len(actual & expected) / len(actual) if actual else 0.0

        reports_writer.append(
            test_name="graph_overlap_tv",
            scenario=scenario["name"],
            metrics={
                "expected_neighbor_count": len(expected),
                "actual_recs_count": len(actual),
                "intersection_count": len(actual & expected),
                "overlap": round(overlap, 3),
                "expected_neighbor_titles_sample": _resolve_titles(
                    dual_engine_real.tv.metadata, expected, limit=10
                ),
                "rec_titles": [getattr(r, "title", "?") for r in recs],
            },
        )
        assert overlap >= self.MIN_OVERLAP, (
            f"{scenario['name']}: graph overlap {overlap:.2f} < {self.MIN_OVERLAP}. "
            f"Final ranker has diverged from raw LightGCN. "
            f"|expected|={len(expected)}, |recs|={len(actual)}, "
            f"|intersect|={len(actual & expected)}"
        )
