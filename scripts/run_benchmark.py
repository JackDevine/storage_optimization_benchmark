"""Run dispatch policies through the rolling 5-min simulator and store results
in DuckDB (`results_intervals`, `results_summary`), replacing any earlier rows
for the same (config, policy, region).

Usage:
    uv run scripts/run_benchmark.py [--config config/default.toml] [--regions SA1 NSW1 ...]
        [--policies milp_bid_curve ...] [--jobs 5] [--start 2025-08-01] [--end 2025-09-01]

The window defaults to the last dev month (harness/splits.py); perfect
foresight solves it as one MILP, so keep windows to a few months.
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import duckdb
import pandas as pd

from bess.config import DEFAULT_CONFIG_PATH, load_config
from bess.data import DB_PATH, load_region
from bess.policies import MILPBidCurvePolicy, MILPForecastPolicy, PerfectForesight
from bess.sim import simulate, summarise
from harness.splits import DEV_MONTHS, REGIONS
POLICIES = ["milp_forecast", "milp_bid_curve", "perfect_foresight"]


def write_results(
    config: str, policy: str, region: str, intervals: pd.DataFrame, summary: dict
) -> None:
    intervals = intervals.drop(columns="bid").assign(config=config, policy=policy, region=region)
    summary_df = pd.DataFrame([dict(config=config, policy=policy, region=region, **summary)])
    con = duckdb.connect(str(DB_PATH))
    try:
        for table, df in [("results_intervals", intervals), ("results_summary", summary_df)]:
            con.register("_incoming", df)
            con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM _incoming LIMIT 0")
            con.execute(
                f"DELETE FROM {table} WHERE config = ? AND policy = ? AND region = ?",
                [config, policy, region],
            )
            con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM _incoming")
            con.unregister("_incoming")
    finally:
        con.close()


def run_region(
    config_path: str, region: str, policy_names: list[str], start: pd.Timestamp, end: pd.Timestamp,
) -> list[tuple[str, pd.DataFrame, dict]]:
    """Simulate the named policies for one region. Runs in a worker process,
    so it returns results rather than writing to DuckDB."""
    config = load_config(config_path)
    data = load_region(region, DB_PATH, start, end)
    factories = {
        "milp_forecast": lambda: MILPForecastPolicy(config),
        "milp_bid_curve": lambda: MILPBidCurvePolicy(config),
        "perfect_foresight": lambda: PerfectForesight(config, data.actual_rrp, start, end),
    }
    results = []
    for name in policy_names:
        t0 = time.time()
        policy = factories[name]()
        intervals = simulate(policy, data, config, start, end)
        summary = summarise(intervals, config.battery)
        flag = "" if summary["valid"] else "  ** INVALID **"
        print(f"{region} {name}: net ${summary['net_revenue']:,.0f} ({time.time() - t0:.0f}s){flag}", flush=True)
        results.append((name, intervals, summary))
    return results


def perfect_foresight_revenue(config: str) -> pd.Series:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return con.execute(
            """SELECT region, net_revenue FROM results_summary
               WHERE config = ? AND policy = 'perfect_foresight'""",
            [config],
        ).df().set_index("region").net_revenue
    finally:
        con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--regions", nargs="+", default=REGIONS)
    parser.add_argument("--policies", nargs="+", default=POLICIES, choices=POLICIES)
    parser.add_argument("--jobs", type=int, default=1, help="regions to simulate in parallel")
    parser.add_argument("--start", default=str(DEV_MONTHS[-1][0]), help="window start (exclusive)")
    parser.add_argument("--end", default=str(DEV_MONTHS[-1][1]), help="window end (inclusive)")
    args = parser.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)

    config_name = load_config(args.config).name
    summaries = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run_region, str(args.config), region, args.policies, start, end): region for region in args.regions
        }
        for future in as_completed(futures):
            region = futures[future]
            for name, intervals, summary in future.result():
                write_results(config_name, name, region, intervals, summary)
                summaries.append(dict(policy=name, region=region, **summary))

    table = pd.DataFrame(summaries)
    # Perfect foresight may come from an earlier run with the same config.
    table["pct_of_perfect"] = 100 * table.net_revenue / table.region.map(perfect_foresight_revenue(config_name))
    cols = [
        "region", "policy", "net_revenue", "net_revenue_per_mw_day", "equivalent_cycles",
        "pct_of_perfect", "valid",
    ]
    print()
    print(table[cols].sort_values(["region", "policy"]).to_string(index=False, float_format="{:,.1f}".format))

    invalid = table[~table.valid]
    if len(invalid):
        bar = "!" * 78
        print(f"\n{bar}\n  INVALID RUNS: scores are kept but should not be compared as valid results\n{bar}")
        for row in invalid.itertuples():
            print(
                f"  {row.region} {row.policy}: {row.rejected_bids} rejected bids, "
                f"{row.nonconformance_intervals} non-conforming intervals "
                f"({row.nonconformance_mwh:.3f} MWh)"
            )
        print(bar)
