"""Dev / holdout split for agent experiments.

Split by time so nothing from the holdout period leaks into development: the
agent develops on October 2021 to August 2025 and is scored on the twelve
months after, all regions. Windows are (start, end]: intervals ending after
`start` up to and including `end`, with decisions made at `interval_end - 5min`.

The data starts at 5-minute settlement (1 October 2021). The June 2022 market
suspension is excluded: prices then came from AEMO's suspension schedule, not
from bids. Long periods are scored as one window per calendar month, each
starting the battery afresh, so jobs stay short and run in parallel.

The dev database holds only data published by the last dev decision time:
actual prices for intervals ending by DEV_END, and forecast runs available by
DEV_END - 5min. (Those runs forecast into the holdout; a real participant would
have them.)
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from bess.data import DB_PATH, INTERVAL
from bess.normalise import PRICE_MAP

REGIONS = ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]

# First decision needs a predispatch run, which starts at 04:01 on 1 October 2021.
DEV_START = pd.Timestamp("2021-10-01 04:05")
DEV_END = pd.Timestamp("2025-09-01 00:00")
HOLDOUT_START = DEV_END
HOLDOUT_END = pd.Timestamp("2026-09-01 00:00")

# Periods left out of scoring and episodes:
# - trading days 12-23 June 2022: administered pricing from 12 June, then the
#   market suspension (15-24 June)
# - October 2022: AEMO's archive has no predispatch runs for it (only each
#   period's final forecast, which would leak), so there's no forecast to bid on
EXCLUDED = [
    (pd.Timestamp("2022-06-12 04:00"), pd.Timestamp("2022-06-24 04:00")),
    (pd.Timestamp("2022-10-01 00:00"), pd.Timestamp("2022-11-01 04:05")),  # first November run 04:01
]

# Submissions are scored on short episodes sampled from a window rather than
# the whole window: every (region, episode) runs in parallel, so a submission
# takes about one episode's simulated time. One episode per block: a calendar
# month in the holdout, four months in dev, so both cover every season.
EPISODE_DAYS = 3
DEV_EPISODE_BLOCK_MONTHS = 4
EPISODE_SEED = 2026
TRADING_DAY_START = pd.Timedelta(hours=4)

Window = tuple[pd.Timestamp, pd.Timestamp]

# Identifies the data behind a score (part of the config fingerprint): bump the
# number when the data or how it's built changes.
DATA_VERSION = f"3:{DEV_START}:{HOLDOUT_END}:knee={PRICE_MAP.knee:g},gamma={PRICE_MAP.gamma:g}"


def _overlaps(a: Window, b: Window) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def excluded_overlap(start: pd.Timestamp, end: pd.Timestamp) -> bool:
    return any(_overlaps((start, end), x) for x in EXCLUDED)


def month_windows(start: pd.Timestamp, end: pd.Timestamp, months: int = 1) -> list[Window]:
    """(start, end] cut at calendar-month boundaries (every `months` months
    from January), with the excluded periods cut out."""
    edges = [start, *[t for t in pd.date_range(start.normalize(), end, freq="MS") if start < t < end
                      and (t.month - 1) % months == 0], end]
    windows = list(zip(edges[:-1], edges[1:]))
    for x0, x1 in EXCLUDED:
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
    return windows


def sample_episodes(blocks: list[Window], days: int = EPISODE_DAYS, seed: int = EPISODE_SEED) -> list[Window]:
    """Seeded sample of whole-trading-day episodes (D 04:00, D+days 04:00], one
    inside each block that can hold one. Deterministic for given blocks and
    seed, so episodes follow the split automatically if the data changes.

    Each episode starts at a trading-day boundary, so the battery starts at its
    initial SOC with a fresh daily price lock.
    """
    rng = np.random.default_rng(seed)
    length = pd.Timedelta(days=days)
    episodes = []
    for start, end in blocks:
        first = (start - TRADING_DAY_START).ceil("D") + TRADING_DAY_START  # first trading-day start >= start
        candidates = [d for d in pd.date_range(first, end - length, freq="D") if not excluded_overlap(d, d + length)]
        if candidates:
            d = candidates[rng.integers(len(candidates))]
            episodes.append((d, d + length))
    return episodes


HOLDOUT_MONTHS = month_windows(HOLDOUT_START, HOLDOUT_END)
DEV_MONTHS = month_windows(DEV_START, DEV_END)
HOLDOUT_EPISODES = sample_episodes(HOLDOUT_MONTHS)
DEV_EPISODES = sample_episodes(month_windows(DEV_START, DEV_END, DEV_EPISODE_BLOCK_MONTHS))


def export_dev_db(dest: Path, source: Path = DB_PATH) -> None:
    """Write the dev-period subset of the tables the simulator reads."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    last_decision = DEV_END - INTERVAL
    con = duckdb.connect(str(dest))
    try:
        con.execute(f"ATTACH '{source}' AS src (READ_ONLY)")
        con.execute(
            """CREATE TABLE dispatchprice AS
               SELECT * FROM src.dispatchprice WHERE SETTLEMENTDATE <= ? ORDER BY REGIONID, SETTLEMENTDATE""",
            [DEV_END],
        )
        con.execute(
            """CREATE TABLE p5min_regionsolution AS
               SELECT * FROM src.p5min_regionsolution WHERE RUN_DATETIME <= ?
               ORDER BY REGIONID, RUN_DATETIME, INTERVAL_DATETIME""",
            [last_decision],
        )
        con.execute(
            """CREATE TABLE predispatchprice AS
               SELECT * FROM src.predispatchprice WHERE LASTCHANGED <= ?
               ORDER BY REGIONID, LASTCHANGED, DATETIME""",
            [last_decision],
        )
        # The cap's history (it's announced well ahead, so the future entries are public).
        con.execute("CREATE TABLE market_price_thresholds AS SELECT * FROM src.market_price_thresholds")
        con.execute("CREATE TABLE price_map AS SELECT * FROM src.price_map")
    finally:
        con.close()
