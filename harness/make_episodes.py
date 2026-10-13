"""Draw and freeze the scoring episodes (harness/splits.py).

Holdout: one 7-day episode inside each holdout month. Dev (dev_eval's
default): six 7-day episodes, one in each two-month season, from the dev
periods. Each episode starts at a trading-day boundary (04:00).

A draw is kept only if it represents its pool of days: in every region, the
mean two-hour price spread of its days (the day's top 24 five-minute prices'
mean less its bottom 24's: what a lossless 2-hour battery could earn ignoring
chronology) is within MAX_SPREAD_MISMATCH of the pool's. Draws are tried seed
by seed from EPISODE_SEED, so the result is reproducible.

    uv run python -m harness.make_episodes

Writes harness/dev_episodes.json and data/holdout_episodes.json.
"""

import json

import duckdb
import numpy as np
import pandas as pd

from bess.data import DB_PATH
from harness.splits import (DEV_EPISODES_PATH, DEV_MONTHS, EPISODE_DAYS, HOLDOUT_EPISODES_PATH, HOLDOUT_MONTHS, REGIONS,
                            TRADING_DAY_START, Window)

EPISODE_SEED = 2026
MAX_SPREAD_MISMATCH = 0.20
DEV_EPISODES = 6
LENGTH = pd.Timedelta(days=EPISODE_DAYS)
DAY = pd.Timedelta(days=1)


def daily_spreads() -> pd.DataFrame:
    """Two-hour spread per trading day (rows) and region (columns)."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    d = con.execute("""SELECT REGIONID r, SETTLEMENTDATE t, RRP p FROM dispatchprice
                       WHERE INTERVENTION = 0 AND SETTLEMENTDATE > TIMESTAMP '2022-12-31'""").df()
    con.close()
    d["day"] = (d.t - pd.Timedelta(minutes=5) - TRADING_DAY_START).dt.normalize() + TRADING_DAY_START

    def spread(p):
        v = np.sort(p.to_numpy())
        return v[-24:].mean() - v[:24].mean() if len(v) >= 48 else np.nan

    return d.groupby(["day", "r"]).p.apply(spread).unstack("r")[REGIONS]


def starts_in(window: Window) -> list[pd.Timestamp]:
    a, b = window
    first = (a - TRADING_DAY_START).ceil("D") + TRADING_DAY_START
    return list(pd.date_range(first, b - LENGTH, freq="D"))


def mismatch(episodes, pool_days, sp) -> pd.Series:
    days = [s + k * DAY for s, _ in episodes for k in range(EPISODE_DAYS)]
    return sp.loc[days].mean() / sp.loc[pool_days].mean() - 1


def draw(blocks: list[list[Window]], pool_days, sp, label):
    """One episode per block (a block is a list of candidate windows)."""
    for seed in range(EPISODE_SEED, EPISODE_SEED + 10_000):
        rng = np.random.default_rng(seed)
        eps = []
        for windows in blocks:
            starts = [s for w in windows for s in starts_in(w)]
            s = starts[rng.integers(len(starts))]
            eps.append((s, s + LENGTH))
        m = mismatch(eps, pool_days, sp)
        if m.abs().max() <= MAX_SPREAD_MISMATCH:
            print(f"{label}: seed {seed}, spread mismatch by region (%): {dict((100 * m).round(1))}")
            return sorted(eps), seed, m
    raise RuntimeError(f"no {label} draw within {MAX_SPREAD_MISMATCH:.0%}")


def days_of(windows):
    return [d for w in windows for d in pd.date_range((w[0] - TRADING_DAY_START).ceil("D") + TRADING_DAY_START,
                                                       w[1] - DAY, freq="D")]


def main() -> None:
    sp = daily_spreads()
    hold, hseed, hm = draw([[w] for w in HOLDOUT_MONTHS], days_of(HOLDOUT_MONTHS), sp, "holdout")
    # dev: one episode per two-month season, from any dev window in it long enough
    seasons = [[w for w in DEV_MONTHS if (w[0].month - 1) // 2 == k and starts_in(w)] for k in range(DEV_EPISODES)]
    dev, dseed, dm = draw(seasons, days_of(DEV_MONTHS), sp, "dev")
    for path, eps, seed, m in ((HOLDOUT_EPISODES_PATH, hold, hseed, hm), (DEV_EPISODES_PATH, dev, dseed, dm)):
        path.write_text(json.dumps({
            "episode_days": EPISODE_DAYS, "seed": seed, "max_spread_mismatch": MAX_SPREAD_MISMATCH,
            "spread_mismatch_by_region": {r: round(float(v), 4) for r, v in m.items()},
            "episodes": [[str(a), str(b)] for a, b in eps]}, indent=1))
        print(f"wrote {path} ({len(eps)} episodes)")


if __name__ == "__main__":
    main()
