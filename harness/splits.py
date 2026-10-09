"""Dev / holdout split for agent experiments.

Split by time so nothing from the holdout period leaks into development: the
agent develops on June–July 2026 and is scored on August 2026, all regions.
Windows are (start, end]: intervals ending after `start` up to and including
`end`, with decisions made at `interval_end - 5min`.

The dev database holds only data published by the last dev decision time:
actual prices for intervals ending by DEV_END, and forecast runs available by
DEV_END - 5min. (Those runs forecast into August; a real participant would
have them.)
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from bess.data import DB_PATH, INTERVAL

REGIONS = ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]

# First decision needs a predispatch run, which starts at 04:01 on June 1.
DEV_START = pd.Timestamp("2026-06-01 04:05")
DEV_END = pd.Timestamp("2026-08-01 00:00")
HOLDOUT_START = DEV_END
HOLDOUT_END = pd.Timestamp("2026-09-01 00:00")

# Submissions are scored on short episodes sampled from a window rather than
# the whole window: every (region, episode) runs in parallel, so a submission
# takes about one episode's simulated time. Retune these once there's more data.
EPISODE_DAYS = 3
EPISODE_BLOCK_DAYS = 7  # one episode per block
EPISODE_SEED = 2026
TRADING_DAY_START = pd.Timedelta(hours=4)

Window = tuple[pd.Timestamp, pd.Timestamp]


def sample_episodes(
    start: pd.Timestamp,
    end: pd.Timestamp,
    days: int = EPISODE_DAYS,
    block_days: int = EPISODE_BLOCK_DAYS,
    seed: int = EPISODE_SEED,
) -> list[Window]:
    """Seeded sample of whole-trading-day episodes (D 04:00, D+days 04:00]
    inside (start, end]: one per consecutive block of `block_days`, placed so
    the episode stays within its block. Deterministic for a given window and
    seed, so episodes follow the split automatically if the data changes.

    Each episode starts at a trading-day boundary, so the battery starts at its
    initial SOC with a fresh daily price lock.
    """
    rng = np.random.default_rng(seed)
    first = (start - TRADING_DAY_START).ceil("D") + TRADING_DAY_START  # first trading-day start >= start
    length = pd.Timedelta(days=days)
    episodes = []
    block = first
    while block + length <= end:
        latest = min(block + pd.Timedelta(days=block_days) - length, end - length)
        candidates = pd.date_range(block, latest, freq="D")
        if len(candidates):
            d = candidates[rng.integers(len(candidates))]
            episodes.append((d, d + length))
        block += pd.Timedelta(days=block_days)
    return episodes


HOLDOUT_EPISODES = sample_episodes(HOLDOUT_START, HOLDOUT_END)
DEV_EPISODES = sample_episodes(DEV_START, DEV_END)


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
               SELECT SETTLEMENTDATE, REGIONID, INTERVENTION, RRP FROM src.dispatchprice
               WHERE SETTLEMENTDATE <= ?""",
            [DEV_END],
        )
        con.execute(
            """CREATE TABLE p5min_regionsolution AS
               SELECT RUN_DATETIME, INTERVAL_DATETIME, REGIONID, INTERVENTION, RRP
               FROM src.p5min_regionsolution WHERE RUN_DATETIME <= ?""",
            [last_decision],
        )
        con.execute(
            """CREATE TABLE predispatchprice AS
               SELECT LASTCHANGED, DATETIME, REGIONID, INTERVENTION, RRP
               FROM src.predispatchprice WHERE LASTCHANGED <= ?""",
            [last_decision],
        )
    finally:
        con.close()
