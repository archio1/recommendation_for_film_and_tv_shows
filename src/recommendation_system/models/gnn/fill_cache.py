import os

import pandas as pd
from dotenv import load_dotenv

from recommendation_system.models.gnn.bilingual_utils import ScalableMovieIntelligence
from recommendation_system.paths import ENV_FILE, PROCESSED_DIR

DATA_PATH = PROCESSED_DIR / "items_metadata_final.parquet"

load_dotenv(ENV_FILE)
TMDB_API_KEY = os.getenv("TMDB_API_KEY")


def run_prefetch():
    if not TMDB_API_KEY:
        raise RuntimeError(
            "TMDB_API_KEY is not set. Put it into .env at the project root."
        )

    print("Loading metadata...")
    metadata = pd.read_parquet(DATA_PATH)

    intel = ScalableMovieIntelligence(
        metadata=metadata,
        cache_dir=DATA_PATH.parent / "cache",
        tmdb_api_key=TMDB_API_KEY,
    )

    print(f"Starting Russian title download for {len(metadata)} items...")
    print("This may take 20-40 minutes. Press Ctrl+C to stop (progress is saved).")

    def progress(current, total):
        if current % 10 == 0:
            print(f"  Done: {current}/{total} ({(current / total) * 100:.1f}%)")

    intel.prefetch_all_translations(progress_callback=progress)
    print("All translations saved to SQLite cache.")


if __name__ == "__main__":
    run_prefetch()
