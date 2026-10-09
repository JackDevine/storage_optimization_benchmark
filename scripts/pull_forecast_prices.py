"""Pull P5MIN_REGIONSOLUTION (5-min-ahead rolling forecast), PREDISPATCHPRICE
(30-min predispatch price forecast), and PREDISPATCHREGIONSUM (30-min
predispatch demand forecast, including TOTALDEMAND/UIGF/SEMISCHEDULE_CLEAREDMW
for residual demand) from AEMO's MMSDM historical archive.

Not available via NEMOSIS (see aemo_mmsdm.py docstring), so this downloads
directly from nemweb.

Usage:
    uv run scripts/pull_forecast_prices.py [--start YYYY/MM/DD] [--end YYYY/MM/DD]
"""

import argparse
from pathlib import Path

from aemo_mmsdm import download_mmsdm_table

from db import write_table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CACHE_DIR = PROJECT_ROOT / "data" / "raw_cache"

TABLES = {
    "p5min_regionsolution": dict(
        directory="P5MIN_ALL_DATA",
        filename_stub="PUBLIC_ARCHIVE#P5MIN_REGIONSOLUTION#ALL#FILE01",
        datetime_columns=["RUN_DATETIME", "INTERVAL_DATETIME", "LASTCHANGED"],
    ),
    "predispatchprice": dict(
        directory="PREDISP_ALL_DATA",
        filename_stub="PUBLIC_ARCHIVE#PREDISPATCHPRICE#ALL#FILE01",
        datetime_columns=["DATETIME", "LASTCHANGED"],
    ),
    "predispatchregionsum": dict(
        directory="PREDISP_ALL_DATA",
        filename_stub="PUBLIC_ARCHIVE#PREDISPATCHREGIONSUM#ALL#FILE01",
        datetime_columns=["DATETIME", "LASTCHANGED"],
    ),
}


def pull_forecast_table(name: str, start: str, end: str) -> None:
    spec = TABLES[name]
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    data = download_mmsdm_table(
        start=start,
        end=end,
        directory=spec["directory"],
        filename_stub=spec["filename_stub"],
        cache_dir=RAW_CACHE_DIR,
        datetime_columns=spec["datetime_columns"],
    )

    write_table(name, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026/06/01")
    parser.add_argument("--end", default="2026/09/01")
    args = parser.parse_args()

    for table_name in TABLES:
        pull_forecast_table(table_name, args.start, args.end)
