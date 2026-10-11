"""Check the fast static-bid simulator (bess/static_bid.py) against the real one.

For random static bids of each shape (static, time of day, time of day + SOC),
in every region over a few dev windows: net revenue, MW and energy paths from
`simulate_static` must match bess/sim.py's `simulate` driving StaticBidPolicy,
and the real simulator must report a valid run (no rejected bids, no
non-conformance). Also tries a slow ramp rate, where stopping takes several
intervals.

    uv run python scripts/check_static_bid_parity.py
"""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bess.config import load_config  # noqa: E402
from bess.data import load_region  # noqa: E402
from bess.sim import simulate, summarise  # noqa: E402
from bess.static_bid import (  # noqa: E402
    StaticBidParams, StaticBidPolicy, interval_grid, prices_from_gaps, simulate_static,
)
from harness.splits import REGIONS  # noqa: E402

DEV_DB = Path(__file__).resolve().parent.parent / "data" / "dev.duckdb"
WINDOWS = [
    (pd.Timestamp("2022-01-03 04:00"), pd.Timestamp("2022-01-06 04:00")),
    (pd.Timestamp("2025-07-14 04:00"), pd.Timestamp("2025-07-17 04:00")),
]
SHAPES = [(1, 1), (48, 1), (48, 3)]


def random_params(rng, shape, market) -> StaticBidParams:
    load, gen = prices_from_gaps(rng.normal(0, 1, 21), market)
    # Lots of MW in the bands so SOC limits bind often.
    return StaticBidParams(load, gen, rng.uniform(0, 0.4, shape + (10,)), rng.uniform(0, 0.4, shape + (10,)))


def check(config, rng) -> float:
    worst = 0.0
    for start, end in WINDOWS:
        data = {r: load_region(r, DEV_DB, start, end) for r in REGIONS}
        ends, minutes = interval_grid(start, end)
        rrp = np.stack([data[r].actual_rrp.loc[ends].to_numpy() for r in REGIONS])
        for shape in SHAPES:
            params = [random_params(rng, shape, config.market) for _ in REGIONS]
            stack = lambda f: np.stack([getattr(p, f) for p in params])[None]  # noqa: E731
            net, mw, energy = simulate_static(
                rrp, minutes, stack("load_prices"), stack("gen_prices"), stack("load_mw"), stack("gen_mw"),
                config, detail=True,
            )
            for i, region in enumerate(REGIONS):
                out = simulate(StaticBidPolicy(config, params[i]), data[region], config, start, end)
                s = summarise(out, config.battery)
                assert s["valid"], (region, shape, s, out.bid_rejected.dropna().head(1).tolist())
                diffs = [abs(s["net_revenue"] - net[0, i]),
                         np.abs(out.mw.to_numpy() - mw[0, i]).max(),
                         np.abs(out.energy_end_mwh.to_numpy() - energy[0, i]).max()]
                assert max(diffs) < 1e-6, (region, shape, diffs)
                worst = max(worst, *diffs)
                cycles = s["equivalent_cycles"]
                print(f"  {region} {start:%m-%d} blocks={shape[0]} bins={shape[1]}: net ${s['net_revenue']:9.2f} "
                      f"cycles {cycles:5.1f}  max diff {max(diffs):.1e}")
    return worst


def main() -> None:
    config = load_config()
    rng = np.random.default_rng(0)
    print("default ramp rate")
    worst = check(config, rng)
    slow = dataclasses.replace(config, battery=dataclasses.replace(config.battery, ramp_rate_mw_per_min=0.07))
    print("slow ramp (0.07 MW/min: three intervals to stop from full power)")
    worst = max(worst, check(slow, rng))
    print(f"OK: fast simulator matches bess.sim within {worst:.1e}, every run valid")


if __name__ == "__main__":
    main()
