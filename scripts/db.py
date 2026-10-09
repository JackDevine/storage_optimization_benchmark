"""Shared DuckDB access for pull scripts. We keep AEMO's raw downloaded
files (data/raw_cache/) but load compiled results straight into DuckDB
instead of writing intermediate parquet files under data/processed/.
"""

from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "benchmark.duckdb"


def write_table(table_name: str, data: pd.DataFrame) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    try:
        con.register("_incoming", data)
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _incoming")
    finally:
        con.close()
    print(f"Loaded {len(data):,} rows into {DB_PATH}::{table_name}")
