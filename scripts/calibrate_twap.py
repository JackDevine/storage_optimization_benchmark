"""Calibrate TWAPPolicy's deadband scale (bess/policies.py: TWAP_DEADBAND) so
the policy averages one equivalent cycle a day (MWh discharged / usable MWh)
over every dev month in all five regions, at the config's wear cost. Only dev
data is used, and only TWAP is simulated (no perfect foresight).

    uv run python scripts/calibrate_twap.py [--target 1.0]
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bess.config import load_config  # noqa: E402
from bess.data import load_region  # noqa: E402
from bess.policies import TWAPPolicy  # noqa: E402
from bess.sim import simulate, summarise  # noqa: E402
from harness.splits import DEV_MONTHS, REGIONS  # noqa: E402

DEV_DB = Path(__file__).resolve().parent.parent / "data" / "dev.duckdb"


def job(args):
    deadband, region, start, end = args
    config = load_config()
    s = summarise(simulate(TWAPPolicy(config, deadband), load_region(region, DEV_DB, start, end), config, start, end),
                  config.battery)
    return region, start, (end - start).total_seconds() / 86400, s["equivalent_cycles"], s["valid"]


def cycles_per_day(deadband: float, pool) -> float:
    t0 = time.time()
    rows = list(pool.map(job, [(deadband, r, a, b) for r in REGIONS for a, b in DEV_MONTHS]))
    cyc = sum(r[3] for r in rows) / sum(r[2] for r in rows)
    by_fy, by_region = {}, {}
    for region, start, days, c, _ in rows:
        for key, acc in ((start.year + (start.month >= 7), by_fy), (region, by_region)):
            a = acc.setdefault(key, [0.0, 0.0])
            a[0] += c
            a[1] += days
    fmt = lambda acc: ", ".join(f"{k} {v[0] / v[1]:.2f}" for k, v in sorted(acc.items()))  # noqa: E731
    print(f"deadband {deadband:.3f}: {cyc:.3f} cycles/day (invalid {sum(not r[4] for r in rows)}/{len(rows)}); "
          f"by FY: {fmt(by_fy)}; by region: {fmt(by_region)} ({time.time() - t0:.0f}s)", flush=True)
    return cyc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=1.0)
    args = ap.parse_args()
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 2)) as pool:
        lo, hi = 0.0, 1.0  # cycling falls as the deadband widens
        c_lo, c_hi = cycles_per_day(lo, pool), cycles_per_day(hi, pool)
        if not c_hi <= args.target <= c_lo:
            raise SystemExit(f"{args.target} cycles/day isn't between deadband 0 ({c_lo:.3f}) and 1 ({c_hi:.3f})")
        for _ in range(10):
            mid = round(lo + (hi - lo) * (c_lo - args.target) / (c_lo - c_hi), 3)
            c = cycles_per_day(mid, pool)
            if abs(c - args.target) < 0.005:
                break
            if c > args.target:
                lo, c_lo = mid, c
            else:
                hi, c_hi = mid, c
        print(f"TWAP_DEADBAND = {mid}")


if __name__ == "__main__":
    main()
