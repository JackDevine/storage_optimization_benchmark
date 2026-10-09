"""Pull DISPATCHPRICE (5-minute regional dispatch price) from AEMO via NEMOSIS.

Usage:
    uv run scripts/pull_dispatch_price.py [--start YYYY/MM/DD] [--end YYYY/MM/DD]
"""

import argparse
from pathlib import Path

from nemosis import dynamic_data_compiler

from db import write_table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CACHE_DIR = PROJECT_ROOT / "data" / "raw_cache"


def pull_dispatch_price(start: str, end: str) -> None:
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    data = dynamic_data_compiler(
        start_time=f"{start} 00:00:00",
        end_time=f"{end} 00:00:00",
        table_name="DISPATCHPRICE",
        raw_data_location=str(RAW_CACHE_DIR),
        fformat="parquet",
    )

    write_table("dispatchprice", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026/06/01")
    parser.add_argument("--end", default="2026/09/01")
    args = parser.parse_args()

    pull_dispatch_price(args.start, args.end)
