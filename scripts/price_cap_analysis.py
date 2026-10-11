"""How battery arbitrage value relates to the market price cap (MPC), to choose
how older prices are mapped onto today's cap (bess/normalise.py).

Per region and trading day, from the raw prices in data/benchmark.duckdb:

- arbitrage: perfect-foresight net revenue of the benchmark battery over the
  day (bess/lp.py), starting and ending at 50% SOC
- swap: the mean RRP (what a flat swap settles against)
- cap payout: the mean of max(RRP - 300, 0) (what a $300 cap contract pays)
- spike share: the share of the day's arbitrage revenue earned in intervals
  above $300
- the MPC in force, the share of intervals within 5% of it, and administered
  pricing (APCFLAG)

Each candidate price map is then applied to the dev prices and the statistics
recomputed, so the report shows what each map changes. Only dev data
(before harness.splits.DEV_END) is used for anything that informs the choice.

    uv run python scripts/price_cap_analysis.py [--jobs 14]

Writes data/analysis/price_cap_daily.parquet (per map) and
data/analysis/price_cap_summary.json.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bess.config import TerminalCondition, load_config  # noqa: E402
from bess.lp import solve_schedule  # noqa: E402
from bess.normalise import PriceMap, mpc_at  # noqa: E402
from harness.splits import DEV_END, EXCLUDED, REGIONS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW_DB = ROOT / "data" / "benchmark.duckdb"
OUT = ROOT / "data" / "analysis"
STRIKE = 300.0
MPC_NOW = 23200.0
# A hard return to 50% by the end of each day, so a day's value isn't
# inflated by draining the battery at its end.
TERMINAL = TerminalCondition(soc_target_frac=0.5, soc_shortfall_penalty=1e5)


def maps() -> dict:
    """Candidate maps: name -> function(price, mpc_then) -> price."""
    out = {"raw": lambda p, m: p}
    out["constant scaling"] = lambda p, m: np.where(p > 0, p * MPC_NOW / m, p)
    for knee in (300.0, 1000.0):
        for gamma in (0.8, 1.0, 1.25, 1.5):
            out[f"knee {knee:g}, gamma {gamma:g}"] = PriceMap(MPC_NOW, knee, gamma)
    return out


def trading_days(df: pd.DataFrame) -> pd.Series:
    return (df.SETTLEMENTDATE - pd.Timedelta(minutes=5) - pd.Timedelta(hours=4)).dt.normalize()


def day_stats(args) -> list[dict]:
    """Statistics for each (region, day) in one chunk under one map."""
    region, days, prices_by_day, config = args
    b = config.battery
    out = []
    for day, p in zip(days, prices_by_day):
        if len(p) != 288:
            continue
        dt = np.full(288, 5 / 60)
        c, d, _ = solve_schedule(p, dt, b, b.e_initial, TERMINAL, 1e-4, p_prev=0.0)
        mw = d - c
        avg = (np.concatenate([[0.0], mw[:-1]]) + mw) / 2  # straight-line average MW per interval
        rev = avg * p * dt
        deg = b.degradation_cost * np.abs(avg) * dt
        net = float(rev.sum() - deg.sum())
        out.append({
            "region": region, "day": day, "arbitrage": net,
            "spike_revenue": float(rev[p > STRIKE].sum()),
            "swap": float(p.mean()), "cap_payout": float(np.maximum(p - STRIKE, 0).mean()),
            "max_price": float(p.max()),
        })
    return out


def compute(jobs: int) -> None:
    config = load_config()
    con = duckdb.connect(str(RAW_DB), read_only=True)
    try:
        prices = con.execute("""SELECT SETTLEMENTDATE, REGIONID, RRP, APCFLAG FROM dispatchprice
                                WHERE INTERVENTION = 0 ORDER BY REGIONID, SETTLEMENTDATE""").df()
        thresholds = con.execute("SELECT * FROM market_price_thresholds").df()
    finally:
        con.close()
    prices["day"] = trading_days(prices)
    prices = prices[prices.day >= pd.Timestamp("2021-10-01")]
    prices["mpc"] = mpc_at(thresholds, prices.SETTLEMENTDATE - pd.Timedelta(minutes=5))
    for x0, x1 in EXCLUDED:
        prices = prices[~((prices.SETTLEMENTDATE > x0) & (prices.SETTLEMENTDATE <= x1))]

    meta = prices.groupby(["REGIONID", "day"]).agg(
        mpc=("mpc", "first"), near_cap=("RRP", lambda s: 0.0), apc_share=("APCFLAG", "mean"), n=("RRP", "size"),
    ).reset_index().rename(columns={"REGIONID": "region"})
    near = (prices.RRP >= 0.95 * prices.mpc).groupby([prices.REGIONID, prices.day]).mean()
    floor = (prices.RRP <= -990).groupby([prices.REGIONID, prices.day]).mean()
    meta["near_cap"] = near.to_numpy()
    meta["near_floor"] = floor.to_numpy()

    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    raw_rows = None
    for name, fn in maps().items():
        t0 = time.time()
        mapped = prices.assign(RRP=fn(prices.RRP.to_numpy(), prices.mpc.to_numpy()))
        # Only days a map changes need solving again.
        changed = (mapped.RRP != prices.RRP).groupby([prices.REGIONID, prices.day]).any()
        tasks = []
        for region, g in mapped.groupby("REGIONID"):
            groups = [(d, x) for d, x in g.groupby("day") if raw_rows is None or changed[(region, d)]]
            for i in range(0, len(groups), 60):
                chunk = groups[i : i + 60]
                tasks.append((region, [d for d, _ in chunk], [x.RRP.to_numpy() for _, x in chunk], config))
        with ProcessPoolExecutor(jobs) as pool:
            rows = pd.DataFrame([r for chunk in pool.map(day_stats, tasks) for r in chunk])
        if raw_rows is None:
            raw_rows = rows
        else:
            keep = raw_rows.set_index(["region", "day"])
            keep = keep[~keep.index.isin(rows.set_index(["region", "day"]).index)].reset_index()
            rows = pd.concat([keep, rows], ignore_index=True)
        daily = rows.merge(meta, on=["region", "day"])
        daily["map"] = name
        results[name] = daily
        print(f"{name}: {len(daily)} days, {time.time() - t0:.0f}s", flush=True)
    pd.concat(results.values()).to_parquet(OUT / "price_cap_daily.parquet")


def fy(day: pd.Series) -> pd.Series:
    """Financial year (the cap changes each 1 July), e.g. 'FY2022' for July 2021 to June 2022."""
    return "FY" + (day.dt.year + (day.dt.month >= 7)).astype(str)


def summarise() -> dict:
    daily = pd.read_parquet(OUT / "price_cap_daily.parquet")
    daily["fy"] = fy(daily.day)
    daily["dev"] = daily.day < DEV_END - pd.Timedelta(days=1)
    raw = daily[daily.map == "raw"]
    out = {"caps": raw.groupby("fy").mpc.agg(["min", "max"]).to_dict("index")}

    def by(df, keys):
        g = df.groupby(keys)
        s = g.agg(days=("arbitrage", "size"), arbitrage=("arbitrage", "mean"), swap=("swap", "mean"),
                  cap_payout=("cap_payout", "mean"), spike_revenue=("spike_revenue", "sum"),
                  arbitrage_total=("arbitrage", "sum"), near_cap=("near_cap", "mean"), apc_share=("apc_share", "mean"),
                  near_floor=("near_floor", "mean"))
        s["spike_share"] = s.spike_revenue / s.arbitrage_total
        return s.drop(columns=["spike_revenue", "arbitrage_total"])

    out["raw_by_fy"] = by(raw, ["fy"]).reset_index().to_dict("records")
    out["raw_by_fy_region"] = by(raw, ["fy", "region"]).reset_index().to_dict("records")

    # Relationships per regime: OLS of daily arbitrage on swap and cap payout.
    def fit(df):
        X = np.c_[np.ones(len(df)), df.swap, df.cap_payout]
        coef, *_ = np.linalg.lstsq(X, df.arbitrage, rcond=None)
        return {"intercept": coef[0], "per_swap": coef[1], "per_cap_payout": coef[2]}

    out["regression_raw_by_fy"] = {k: fit(g) for k, g in raw.groupby("fy")}

    # The tail: is it prices relative to the cap, or absolute prices, that look
    # alike across regimes? Quantiles of p and of p / MPC above the strike.
    prices_tail = {}
    con = duckdb.connect(str(RAW_DB), read_only=True)
    try:
        t = con.execute(f"""SELECT SETTLEMENTDATE, RRP FROM dispatchprice
                            WHERE INTERVENTION = 0 AND RRP > {STRIKE} AND SETTLEMENTDATE <= ?""", [DEV_END]).df()
        thresholds = con.execute("SELECT * FROM market_price_thresholds").df()
    finally:
        con.close()
    t["mpc"] = mpc_at(thresholds, t.SETTLEMENTDATE - pd.Timedelta(minutes=5))
    t["fy"] = fy(t.SETTLEMENTDATE - pd.Timedelta(minutes=5))
    qs = [0.5, 0.9, 0.99, 0.999]
    for k, g in t.groupby("fy"):
        prices_tail[k] = {
            "n": len(g), "mpc": float(g.mpc.max()),
            "abs_quantiles": dict(zip(map(str, qs), np.quantile(g.RRP, qs).round(0).tolist())),
            "rel_quantiles": dict(zip(map(str, qs), np.quantile((g.RRP - STRIKE) / (g.mpc - STRIKE), qs).round(4).tolist())),
            "share_at_cap": float((g.RRP >= 0.95 * g.mpc).mean()),
        }
    out["tail_by_fy"] = prices_tail

    # Each map on dev: how much it moves prices, and the statistics it produces
    # by year (a good map makes the older years look like recent ones).
    dev = daily[daily.dev]
    base = dev[dev.map == "raw"].set_index(["region", "day"])
    out["maps"] = {}
    for name, g in dev.groupby("map"):
        gi = g.set_index(["region", "day"])
        out["maps"][name] = {
            "by_fy": by(g, ["fy"]).reset_index().to_dict("records"),
            "arbitrage_change_pct": float(100 * (gi.arbitrage.sum() / base.arbitrage.sum() - 1)),
            "swap_change_pct": float(100 * (gi.swap.sum() / base.swap.sum() - 1)),
            "regression_by_fy": {k: fit(x) for k, x in g.groupby("fy")},
        }
    (OUT / "price_cap_summary.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=14)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if not args.summary_only:
        compute(args.jobs)
    s = summarise()
    print(json.dumps({k: s[k] for k in ("caps", "tail_by_fy")}, indent=1, default=str))
