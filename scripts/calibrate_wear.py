"""Calibrate the wear (throughput) cost, battery.degradation_cost, so the LP
forecast planner (MILPForecastPolicy with integer=False) averages one
equivalent cycle a day (MWh discharged / usable MWh) over every dev month in
all five regions. Only dev data is used, and only the planner is simulated
(no perfect foresight). With --check, also reports how TWAPPolicy cycles at
the calibrated cost.

    uv run python scripts/calibrate_wear.py [--target 1.0] [--lo 5] [--hi 80] [--check-twap]
"""

import argparse
import dataclasses
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bess.config import load_config  # noqa: E402
from bess.data import load_region  # noqa: E402
from bess.policies import MILPForecastPolicy, TWAPPolicy  # noqa: E402
from bess.sim import simulate, summarise  # noqa: E402
from harness.splits import DEV_MONTHS, REGIONS  # noqa: E402

DEV_DB = Path(__file__).resolve().parent.parent / "data" / "dev.duckdb"


def job(args):
    policy, wear, deadband, region, start, end = args
    base = load_config()
    config = dataclasses.replace(base, battery=dataclasses.replace(base.battery, degradation_cost=wear))
    pol = MILPForecastPolicy(config, integer=False) if policy == "lp" else TWAPPolicy(config, deadband)
    s = summarise(simulate(pol, load_region(region, DEV_DB, start, end), config, start, end), config.battery)
    return region, start, (end - start).total_seconds() / 86400, s["equivalent_cycles"], s["valid"], s["net_revenue"]


def cycles_per_day(policy: str, wear: float, pool, deadband: float = 1.0) -> float:
    t0 = time.time()
    rows = list(pool.map(job, [(policy, wear, deadband, r, a, b) for r in REGIONS for a, b in DEV_MONTHS]))
    cyc = sum(r[3] for r in rows) / sum(r[2] for r in rows)
    by_fy, by_region = {}, {}
    for region, start, days, c, _, _ in rows:
        for key, acc in ((start.year + (start.month >= 7), by_fy), (region, by_region)):
            a = acc.setdefault(key, [0.0, 0.0])
            a[0] += c
            a[1] += days
    fmt = lambda acc: ", ".join(f"{k} {v[0] / v[1]:.2f}" for k, v in sorted(acc.items()))  # noqa: E731
    print(f"{policy} at wear ${wear:.2f}/MWh: {cyc:.3f} cycles/day, net ${sum(r[5] for r in rows):,.0f} "
          f"(invalid {sum(not r[4] for r in rows)}/{len(rows)}); by FY: {fmt(by_fy)}; by region: {fmt(by_region)} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return cyc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=1.0)
    ap.add_argument("--lo", type=float, default=5.0)
    ap.add_argument("--hi", type=float, default=80.0)
    ap.add_argument("--check-twap", action="store_true")
    args = ap.parse_args()
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 2)) as pool:
        lo, hi = args.lo, args.hi  # cycling falls as wear rises
        c_lo, c_hi = cycles_per_day("lp", lo, pool), cycles_per_day("lp", hi, pool)
        if not c_hi <= args.target <= c_lo:
            raise SystemExit(f"{args.target} cycles/day isn't between wear ${lo} ({c_lo:.3f}) and ${hi} ({c_hi:.3f})")
        side = 0  # Illinois variant of regula falsi: halve a stuck endpoint's weight
        f_lo, f_hi = c_lo - args.target, c_hi - args.target
        for _ in range(10):
            mid = round((lo * f_hi - hi * f_lo) / (f_hi - f_lo), 2)
            c = cycles_per_day("lp", mid, pool)
            f = c - args.target
            if abs(f) < 0.01:
                break
            if f > 0:
                lo, f_lo = mid, f
                if side == -1:
                    f_hi /= 2
                side = -1
            else:
                hi, f_hi = mid, f
                if side == 1:
                    f_lo /= 2
                side = 1
        print(f"calibrated wear: ${mid}/MWh")
        if args.check_twap:
            cycles_per_day("twap", mid, pool, deadband=1.0)


if __name__ == "__main__":
    main()
