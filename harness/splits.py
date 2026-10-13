"""Dev / holdout split for agent experiments.

The data runs from January 2023 to August 2026 (2022 is left out: the energy
crisis, administered pricing and the market suspension made it unlike any
other year). Twelve whole calendar months inside that period are the holdout,
one of each calendar month, three in each year, drawn with a fixed seed, so
dev and holdout come from the same mix of years and seasons. Everything else
is dev, less a buffer of BUFFER_DAYS either side of each holdout month: a
predispatch run looks about 40 hours ahead, and prices and spike events run
on for days, so the buffer keeps dev data from describing the holdout.

Windows are (start, end]: intervals ending after `start` up to and including
`end`, with decisions made at `interval_end - 5min`. Long periods are scored
as one window per calendar month (or the dev part of one), each starting the
battery afresh, so jobs stay short and run in parallel.

The dev database (export_dev_db) holds actual prices and forecast runs from
the dev periods only: no interval in a holdout month or its buffers, and no
forecast run published in one.

Episodes (7 days, from a trading-day boundary) are drawn once by
harness/make_episodes.py and frozen: dev episodes in harness/dev_episodes.json
(the agent's dev_eval default), holdout episodes in data/holdout_episodes.json
(outside the agent's sandbox).
"""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from bess.data import DB_PATH, INTERVAL
from bess.normalise import PRICE_MAP

REGIONS = ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]
ROOT = Path(__file__).resolve().parent.parent

DEV_START = pd.Timestamp("2023-01-01 04:05")  # first decision: needs that morning's 04:01 predispatch run
DEV_END = pd.Timestamp("2026-09-01 00:00")
HOLDOUT_START, HOLDOUT_END = DEV_START, DEV_END  # the holdout months lie inside the same period
BUFFER_DAYS = 3
HOLDOUT_SEED = 2026
YEARS = [2023, 2024, 2025, 2026]

EPISODE_DAYS = 7
TRADING_DAY_START = pd.Timedelta(hours=4)
DEV_EPISODES_PATH = Path(__file__).resolve().parent / "dev_episodes.json"
HOLDOUT_EPISODES_PATH = ROOT / "data" / "holdout_episodes.json"

Window = tuple[pd.Timestamp, pd.Timestamp]


def holdout_months(seed: int = HOLDOUT_SEED) -> list[Window]:
    """One of each calendar month, three per year, by a seeded shuffle (2026
    has only January to August)."""
    rng = np.random.default_rng(seed)
    while True:
        years = rng.permutation(np.repeat(YEARS, 3))
        if all(y < 2026 or m <= 8 for m, y in zip(range(1, 13), years)):
            break
    out = []
    for m, y in zip(range(1, 13), years):
        a = pd.Timestamp(year=int(y), month=m, day=1)
        out.append((a, a + pd.offsets.MonthBegin(1)))
    return sorted(out)


HOLDOUT_MONTHS = holdout_months()
BUFFER = pd.Timedelta(days=BUFFER_DAYS)
# Dev leaves out each holdout month and its buffers.
EXCLUDED = [(a - BUFFER, b + BUFFER) for a, b in HOLDOUT_MONTHS]

# Identifies the data behind a score (part of the config fingerprint): bump the
# number when the data or how it's built changes.
DATA_VERSION = (f"4:{DEV_START}:{DEV_END}:holdout={HOLDOUT_SEED},buffer={BUFFER_DAYS}d:"
                f"knee={PRICE_MAP.knee:g},gamma={PRICE_MAP.gamma:g}:forecast-cap=target")


def _overlaps(a: Window, b: Window) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def excluded_overlap(start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """Does (start, end] touch a holdout month or its buffers?"""
    return any(_overlaps((start, end), x) for x in EXCLUDED)


def month_windows(start: pd.Timestamp, end: pd.Timestamp, excluded: list[Window] = ()) -> list[Window]:
    """(start, end] cut at calendar-month boundaries, with `excluded` cut out."""
    edges = [start, *[t for t in pd.date_range(start.normalize(), end, freq="MS") if start < t < end], end]
    windows = list(zip(edges[:-1], edges[1:]))
    for x0, x1 in excluded:
        out = []
        for a, b in windows:
            if not _overlaps((a, b), (x0, x1)):
                out.append((a, b))
                continue
            if a < x0:
                out.append((a, x0))
            if x1 < b:
                out.append((x1, b))
        windows = out
    return [(a, b) for a, b in windows if b - a >= pd.Timedelta(days=1)]


DEV_MONTHS = month_windows(DEV_START, DEV_END, EXCLUDED)


def _load_episodes(path: Path) -> list[Window]:
    """Frozen episodes, or none if the file isn't there (the holdout file isn't
    visible inside the agent's sandbox)."""
    try:
        return [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in json.loads(path.read_text())["episodes"]]
    except (FileNotFoundError, PermissionError):
        return []


DEV_EPISODES = _load_episodes(DEV_EPISODES_PATH)
HOLDOUT_EPISODES = _load_episodes(HOLDOUT_EPISODES_PATH)


def export_dev_db(dest: Path, source: Path = DB_PATH) -> None:
    """Write the dev periods of the tables the simulator reads: actual prices
    for intervals in a dev period, forecast runs published in one (a run made
    in dev time can look into a buffer, never into a holdout month)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    first = DEV_START - pd.Timedelta(days=1)  # the first decisions' forecast runs and some price history
    zones = " AND ".join(f"NOT (t > TIMESTAMP '{a}' AND t <= TIMESTAMP '{b}')" for a, b in EXCLUDED)
    con = duckdb.connect(str(dest))
    try:
        con.execute(f"ATTACH '{source}' AS src (READ_ONLY)")
        for table, t, order in (
            ("dispatchprice", "SETTLEMENTDATE", "REGIONID, SETTLEMENTDATE"),
            ("p5min_regionsolution", f"RUN_DATETIME - INTERVAL {int(INTERVAL.total_seconds())} SECOND",
             "REGIONID, RUN_DATETIME, INTERVAL_DATETIME"),
            ("predispatchprice", "LASTCHANGED", "REGIONID, LASTCHANGED, DATETIME"),
        ):
            con.execute(f"""CREATE TABLE {table} AS SELECT * EXCLUDE (t) FROM (
                    SELECT *, {t} AS t FROM src.{table}) WHERE t > ? AND t <= ? AND {zones} ORDER BY {order}""",
                        [first, DEV_END])
        # The cap's history (it's announced well ahead, so the future entries are public).
        con.execute("CREATE TABLE market_price_thresholds AS SELECT * FROM src.market_price_thresholds")
        con.execute("CREATE TABLE price_map AS SELECT * FROM src.price_map")
    finally:
        con.close()
