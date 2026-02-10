import pytest
import pandas as pd
from pathlib import Path
import sqlite3

# Определяем пути к реальным файлам (используя твою структуру)
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def test_metadata_file_exists():
    """Проверяем физическое наличие файла метаданных"""
    path = DATA_DIR / "items_metadata_final.parquet"
    assert path.exists(), f"Файл {path} не найден!"


def test_metadata_columns():
    """Проверяем, что в метаданных есть все необходимые колонки"""
    path = DATA_DIR / "items_metadata_final.parquet"
    df = pd.read_parquet(path)

    required_columns = ['item_id', 'title', 'year', 'genres', 'tmdb_id']
    for col in required_columns:
        assert col in df.columns, f"В метаданных нет колонки {col}"


def test_metadata_not_empty():
    """Проверяем, что датасет не пустой (защита от ошибок при генерации)"""
    df = pd.read_parquet(DATA_DIR / "items_metadata_final.parquet")
    assert len(df) > 1000, "В базе подозрительно мало фильмов!"


def test_sqlite_db_integrity():
    """Проверяем, что база переводов существует и открывается"""
    db_path = DATA_DIR / "cache" / "translations.db"
    assert db_path.exists(), "База SQLite не найдена!"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Проверяем наличие таблицы
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='translations'")
    assert cursor.fetchone() is not None, "В БД нет таблицы translations"
    conn.close()


def test_interactions_consistency():
    """Проверяем, что все айтемы из взаимодействий описаны в метаданных"""
    items_df = pd.read_parquet(DATA_DIR / "items_metadata_final.parquet")
    inter_df = pd.read_parquet(DATA_DIR / "interactions_final.parquet")

    unique_items = set(items_df['item_id'].unique())
    unique_inter_items = set(inter_df['item_id'].unique())

    # Проверяем, что нет взаимодействий для айтемов, которых нет в описании
    missing = unique_inter_items - unique_items
    assert len(missing) == 0, f"Найдено {len(missing)} айтемов без описания!"