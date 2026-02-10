from pathlib import Path
import pandas as pd
from bilingual_utils import ScalableMovieIntelligence

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[4]  # или 0, если запускаешь из корня
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "items_metadata_final.parquet"

TMDB_KEY = "62814d4a01feef50a1344193e60b49be"


def run_prefetch():
    print("⏳ Загрузка метаданных...")
    metadata = pd.read_parquet(DATA_PATH)

    intel = ScalableMovieIntelligence(
        metadata=metadata,
        cache_dir=DATA_PATH.parent / "cache",
        tmdb_api_key=TMDB_KEY
    )

    print(f"🚀 Начинаю скачивание русских названий для {len(metadata)} фильмов...")
    print("Это может занять 20-40 минут. Можно прервать (Ctrl+C), прогресс сохранится.")

    def progress(current, total):
        if current % 10 == 0:
            print(f"✅ Готово: {current}/{total} ({(current / total) * 100:.1f}%)")

    intel.prefetch_all_translations(progress_callback=progress)
    print("🏁 Все переводы загружены в базу SQLite!")


if __name__ == "__main__":
    run_prefetch()