"""Pull the AEMO tables the benchmark uses from the MMSDM archive into
data/benchmark.duckdb, month by month.

Loading is incremental: `_loaded` records each (table, month) loaded, so a rerun
only downloads and loads months it doesn't have. Only the columns the benchmark
needs are kept.

The default start is 1 October 2021, the start of 5-minute settlement; earlier
prices were settled on 30-minute averages.

Usage:
    uv run scripts/pull_data.py [--start 2021-10] [--end 2026-08] [--tables dispatchprice ...]
"""

import argparse
import time
from pathlib import Path

import duckdb
import pandas as pd

from aemo_mmsdm import download, read_zip, table_files

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "benchmark.duckdb"
RAW_CACHE_DIR = PROJECT_ROOT / "data" / "raw_cache"

TS, INT, DBL, STR = "TIMESTAMP", "INTEGER", "DOUBLE", "VARCHAR"

# name -> (archive directory, AEMO table, {column: type}). The column order is
# the stored table's.
TABLES = {
    "dispatchprice": ("DATA", "DISPATCHPRICE", {
        "SETTLEMENTDATE": TS, "REGIONID": STR, "INTERVENTION": INT, "RRP": DBL,
        "APCFLAG": INT, "MARKETSUSPENDEDFLAG": INT, "PRICE_STATUS": STR,
    }),
    "p5min_regionsolution": ("P5MIN_ALL_DATA", "P5MIN_REGIONSOLUTION", {
        "RUN_DATETIME": TS, "INTERVAL_DATETIME": TS, "REGIONID": STR, "INTERVENTION": INT, "RRP": DBL,
    }),
    "predispatchprice": ("PREDISP_ALL_DATA", "PREDISPATCHPRICE", {
        "PREDISPATCHSEQNO": STR, "LASTCHANGED": TS, "DATETIME": TS, "REGIONID": STR, "INTERVENTION": INT, "RRP": DBL,
    }),
    # The market price cap (VOLL), floor and administered price threshold by
    # effective date and version. Each month's file holds the whole history.
    "market_price_thresholds": ("DATA", "MARKET_PRICE_THRESHOLDS", {
        "EFFECTIVEDATE": TS, "VERSIONNO": INT, "VOLL": DBL, "MARKETPRICEFLOOR": DBL,
        "ADMINISTERED_PRICE_THRESHOLD": DBL, "AUTHORISEDDATE": TS, "LASTCHANGED": TS,
    }),
}
DEDUP_KEYS = {"market_price_thresholds": ["EFFECTIVEDATE", "VERSIONNO"]}


def _create(con: duckdb.DuckDBPyConnection, name: str, columns: dict[str, str]) -> None:
    con.execute("CREATE TABLE IF NOT EXISTS _loaded (tbl VARCHAR, month VARCHAR, rows BIGINT, loaded_at TIMESTAMP)")
    existing = [r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position", [name],
    ).fetchall()]
    if existing and existing != list(columns):
        # A table from an older puller with a different layout: rebuild it.
        print(f"{name}: replacing the old table layout", flush=True)
        con.execute(f"DROP TABLE {name}")
        con.execute("DELETE FROM _loaded WHERE tbl = ?", [name])
        existing = []
    if not existing:
        con.execute(f"CREATE TABLE {name} ({', '.join(f'{c} {t}' for c, t in columns.items())})")


def pull(name: str, months: pd.DatetimeIndex, con: duckdb.DuckDBPyConnection) -> None:
    directory, aemo_table, columns = TABLES[name]
    _create(con, name, columns)
    done = {r[0] for r in con.execute("SELECT month FROM _loaded WHERE tbl = ?", [name]).fetchall()}
    for ts in months:
        month = f"{ts.year}-{ts.month:02d}"
        if month in done:
            continue
        files = table_files(ts.year, ts.month, directory, aemo_table)
        if files is None:
            print(f"SKIP {name} {month}: not published yet", flush=True)
            continue
        if not files:
            # AEMO's archive has gaps (e.g. no all-runs PREDISPATCHPRICE for
            # October 2022). Reported, not marked loaded, so a rerun checks again.
            print(f"MISSING {name} {month}: no {aemo_table} file in {directory}", flush=True)
            continue
        t0, rows = time.time(), 0
        con.execute("BEGIN")
        for f in files:
            rel = read_zip(con, download(ts.year, ts.month, directory, f, RAW_CACHE_DIR), columns)
            con.execute(f"INSERT INTO {name} SELECT * FROM {rel.alias}")
            rows += rel.shape[0]
            con.execute(f"DROP TABLE {rel.alias}")
        con.execute("INSERT INTO _loaded VALUES (?, ?, ?, now())", [name, month, rows])
        con.execute("COMMIT")
        print(f"OK   {name} {month}: {rows:,} rows, {time.time() - t0:.0f}s", flush=True)
    if name in DEDUP_KEYS:
        keys = ", ".join(DEDUP_KEYS[name])
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT DISTINCT ON ({keys}) * FROM {name} ORDER BY {keys}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2021-10")
    parser.add_argument("--end", default=None, help="last month (default: the latest published)")
    parser.add_argument("--tables", nargs="+", default=list(TABLES), choices=list(TABLES))
    args = parser.parse_args()

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now().normalize().replace(day=1)
    months = pd.date_range(pd.Timestamp(args.start), end, freq="MS")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    try:
        for name in args.tables:
            pull(name, months, con)
    finally:
        con.close()


if __name__ == "__main__":
    main()
