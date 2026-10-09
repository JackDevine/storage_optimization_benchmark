"""Score a policy on the development data (run this from the agent workspace).

    python -m harness.dev_eval --quick [--regions SA1 VIC1]
    python -m harness.dev_eval [--policy-dir .] [--regions SA1 VIC1] [--start 2026-07-01] [--end 2026-07-08]

Uses ./data/dev.duckdb (June-July 2026). --quick scores on sampled 3-day dev
episodes (chosen like the holdout episodes) in parallel; otherwise one
continuous window. Prints each region's net revenue and % of perfect
foresight, and the total. Per-interval results (bids rejected, MW, prices,
revenue) are written to ./dev_results/<region>_ep<k>_intervals.parquet, and
anything the policy prints goes to ./dev_results/<region>_ep<k>_stderr.log.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from bess.config import load_config
from harness.evaluator import evaluate
from harness.splits import DEV_END, DEV_EPISODES, DEV_START, REGIONS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-dir", default=".")
    parser.add_argument("--quick", action="store_true", help=f"score on {len(DEV_EPISODES)} sampled 3-day dev episodes")
    parser.add_argument("--regions", nargs="+", default=REGIONS, choices=REGIONS)
    parser.add_argument("--start", default=str(DEV_START), help="window start (exclusive), >= 2026-06-01 04:05")
    parser.add_argument("--end", default=str(DEV_END), help="window end (inclusive), <= 2026-08-01 00:00")
    parser.add_argument("--db", default="data/dev.duckdb")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--out-dir", default="dev_results")
    parser.add_argument("--timeout", type=float, default=3600, help="seconds per (region, window) job")
    args = parser.parse_args()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if start < DEV_START or end > DEV_END or start >= end:
        parser.error(f"window must lie within ({DEV_START}, {DEV_END}]")
    windows = DEV_EPISODES if args.quick else [(start, end)]

    result = evaluate(
        Path(args.policy_dir), load_config(args.config), windows,
        regions=args.regions, db_path=Path(args.db).resolve(), sandboxed=False,
        timeout_s=args.timeout, out_dir=Path(args.out_dir),
    )
    for r in result["jobs"]:
        label = f"{r['region']} episode {r['episode']} ({r['window'][0][:10]})" if args.quick else r["region"]
        if "error" in r:
            print(f"{label}: ERROR {r['error']}")
        else:
            flag = "" if r["valid"] else (
                f"  INVALID ({r['rejected_bids']} rejected bids, {r['nonconformance_intervals']} non-conforming intervals, "
                f"{r.get('bid_nonconformance_intervals', 0)} wrong-ramp-rate bids)"
            )
            print(
                f"{label}: net ${r['net_revenue']:,.0f} = {r['pct_of_perfect']:.1f}% of perfect foresight, "
                f"{r['equivalent_cycles']:.0f} cycles, {r['policy_seconds']:.0f}s{flag}"
            )
            if r["first_rejection"]:
                print(f"    first rejected bid: {r['first_rejection'][0][:500]}")
            if r.get("first_bid_nonconformance"):
                print(f"    first wrong-ramp-rate bid: {r['first_bid_nonconformance'][0][:500]}")
    if "pct_of_perfect" in result:
        print(f"TOTAL: net ${result['net_revenue']:,.0f} = {result['pct_of_perfect']:.1f}% of perfect foresight, valid={result['valid']}")
    else:
        print(f"TOTAL: failed: {result['error']}")
    print(f"({result['eval_seconds']:.0f}s)")

    if args.quick:
        for region, pct in result.get("per_region_pct", {}).items():
            print(f"  {region} across episodes: {pct:.1f}%")
    reference = Path("reference_dev.json")
    if reference.exists() and sorted(args.regions) == sorted(REGIONS):
        ref = json.loads(reference.read_text())
        if args.quick and "episodes_pct_of_perfect" in ref:
            print(f"Baseline (milp_forecast) on the same dev episodes: {ref['episodes_pct_of_perfect']:.1f}%")
        elif not args.quick and (start, end) == (DEV_START, DEV_END):
            print(f"Baseline (milp_forecast) on the full dev window: {ref['pct_of_perfect']:.1f}%")
