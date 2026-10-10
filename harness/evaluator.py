"""Score a policy directory over one or more time windows.

A policy directory holds `policy.py` (defining `class Policy`, see
bess/remote.py) and any helper files it imports. Every (region, window) pair
is a separate job: a `python -m harness.evaluator region ...` subprocess with a
wall-clock timeout, which drives the policy through RemotePolicy. Jobs run in
parallel. For holdout evaluation the policy process is additionally sandboxed
(harness/sandbox.py).

Submissions are scored on short sampled episodes (harness/splits.py) so they
take about one episode's simulated time. The score is the total net revenue
across all jobs as a percentage of perfect foresight's total over the same
windows; each window starts from the battery's initial SOC.

    uv run python -m harness.evaluator reference   # cache baseline scores
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from bess.config import Config, load_config
from bess.data import DB_PATH, load_region
from bess.policies import PerfectForesight
from bess.remote import RemotePolicy, config_from_dict, config_to_dict
from bess.sim import simulate, summarise
from harness.sandbox import PROJECT_ROOT, policy_sandbox
from harness.splits import (
    DEV_END, DEV_EPISODES, DEV_START, HOLDOUT_END, HOLDOUT_EPISODES, HOLDOUT_START, REGIONS, Window,
)

REFERENCE_POLICY_DIR = Path(__file__).resolve().parent / "reference_policy"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference_scores"
REFERENCE_PATH = PROJECT_ROOT / "data" / "reference_scores.json"  # legacy: one file for the default config
# Bump when the simulator's scoring rules change, so baselines get recomputed.
SIM_VERSION = 2  # 2: straight-line dispatch trajectory and the unit's ramp rate (1: flat MW, bid's own ramp rates; removed)


def config_fingerprint(config: Config) -> str:
    """Identifies everything that affects a score: the full config (battery,
    market rules, terminal condition, ...) plus the simulator version."""
    d = config_to_dict(config)
    d.pop("name")
    blob = json.dumps({"sim_version": SIM_VERSION, **_floats(d)}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _floats(x):
    """Numbers as floats, so `ramp_rate_mw_per_min = 1` and `= 1.0` fingerprint the same."""
    if isinstance(x, dict):
        return {k: _floats(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_floats(v) for v in x]
    return float(x) if isinstance(x, int) and not isinstance(x, bool) else x


def reference_path(config: Config) -> Path:
    return REFERENCE_DIR / f"{config_fingerprint(config)}.json"


def evaluate(
    policy_dir: Path,
    config: Config,
    windows: list[Window],
    regions: list[str] = REGIONS,
    db_path: Path = DB_PATH,
    sandboxed: bool = True,
    timeout_s: float = 900,
    out_dir: Path | None = None,
    max_workers: int | None = None,
) -> dict:
    """Evaluate every (region, window) job in parallel. `timeout_s` applies per
    job. `out_dir` (if given) receives each job's interval results
    (<region>_ep<k>_intervals.parquet) and the policy's stderr."""
    policy_dir = Path(policy_dir).resolve()
    out_dir = Path(out_dir).resolve() if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(region, k, window) for k, window in enumerate(windows) for region in regions]
    t0 = time.time()

    def run(job) -> dict:
        region, k, (start, end) = job
        tag = f"{region}_ep{k}"
        meta = {"region": region, "episode": k, "window": [str(start), str(end)]}
        cmd = [
            sys.executable, "-m", "harness.evaluator", "region",
            "--policy-dir", str(policy_dir),
            "--region", region, "--tag", tag,
            "--start", str(start), "--end", str(end),
            "--db", str(db_path),
            "--config-json", json.dumps(config_to_dict(config)),
        ]
        if not sandboxed:
            cmd.append("--no-sandbox")
        if out_dir:
            cmd += ["--out-dir", str(out_dir)]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True, cwd=PROJECT_ROOT,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            return {**meta, "error": f"timed out after {timeout_s:.0f}s"}
        if proc.returncode != 0:
            return {**meta, "error": (stderr or stdout)[-2000:]}
        return {**json.loads(stdout.strip().splitlines()[-1]), **meta}

    workers = max_workers or min(len(jobs), max(1, (os.cpu_count() or 2) - 2))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, jobs))
    return {
        **aggregate(results),
        "jobs": results,
        "windows": [[str(a), str(b)] for a, b in windows],
        "eval_seconds": time.time() - t0,
    }


def aggregate(jobs: list[dict]) -> dict:
    errors = [j for j in jobs if "error" in j]
    if errors:
        return {
            "valid": False,
            "error": "; ".join(
                f"{j['region']} episode {j['episode']}: {j['error'].strip().splitlines()[-1]}" for j in errors[:5]
            ),
        }

    def pct_by(key) -> dict:
        groups = {}
        for j in jobs:
            g = groups.setdefault(j[key], [0.0, 0.0])
            g[0] += j["net_revenue"]
            g[1] += j["perfect_foresight_net_revenue"]
        return {k: 100 * net / perfect if perfect else None for k, (net, perfect) in groups.items()}

    net = sum(j["net_revenue"] for j in jobs)
    perfect = sum(j["perfect_foresight_net_revenue"] for j in jobs)
    return {
        "valid": all(j["valid"] for j in jobs),
        "net_revenue": net,
        "perfect_foresight_net_revenue": perfect,
        "pct_of_perfect": 100 * net / perfect,
        "rejected_bids": sum(j["rejected_bids"] for j in jobs),
        "nonconformance_intervals": sum(j["nonconformance_intervals"] for j in jobs),
        "bid_nonconformance_intervals": sum(j.get("bid_nonconformance_intervals", 0) for j in jobs),
        "per_region_pct": pct_by("region"),
        "per_episode_pct": pct_by("episode"),
    }


def _run_region(args) -> dict:
    config = config_from_dict(json.loads(args.config_json))
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    data = load_region(args.region, Path(args.db))
    policy_dir = Path(args.policy_dir)
    out_dir = Path(args.out_dir) if args.out_dir else None
    tag = args.tag or args.region
    stderr = open(out_dir / f"{tag}_stderr.log", "w") if out_dir else subprocess.DEVNULL

    t0 = time.time()
    policy = RemotePolicy(
        policy_dir / "policy.py", config, args.region,
        wrapper=None if args.no_sandbox else policy_sandbox(policy_dir),
        cwd=policy_dir, stderr=stderr,
    )
    try:
        intervals = simulate(policy, data, config, start, end)
    finally:
        policy.close()
    policy_seconds = time.time() - t0
    summary = summarise(intervals, config.battery)
    perfect = summarise(
        simulate(PerfectForesight(config, data.actual_rrp, start, end), data, config, start, end),
        config.battery,
    )
    if out_dir:
        intervals.to_parquet(out_dir / f"{tag}_intervals.parquet")
    return {
        "region": args.region,
        "valid": summary["valid"],
        "net_revenue": summary["net_revenue"],
        "perfect_foresight_net_revenue": perfect["net_revenue"],
        "pct_of_perfect": 100 * summary["net_revenue"] / perfect["net_revenue"],
        "rejected_bids": summary["rejected_bids"],
        "nonconformance_intervals": summary["nonconformance_intervals"],
        "nonconformance_mwh": summary["nonconformance_mwh"],
        "bid_nonconformance_intervals": summary["bid_nonconformance_intervals"],
        "first_bid_nonconformance": intervals.bid_nonconformance.dropna().head(1).tolist(),
        "equivalent_cycles": summary["equivalent_cycles"],
        "policy_seconds": policy_seconds,
        "first_rejection": intervals.bid_rejected.dropna().head(1).tolist(),
    }


FULL_MONTH_TIMEOUT_FACTOR = 31 / 3  # a full-month job vs a 3-day episode job

REFERENCE_SETS = {
    "dev": [(DEV_START, DEV_END)],
    "dev_episodes": DEV_EPISODES,
    "holdout_episodes": HOLDOUT_EPISODES,  # what the agent's submissions are compared against
    "holdout_full": [(HOLDOUT_START, HOLDOUT_END)],  # the headline comparison
}


def compute_reference(config: Config) -> dict:
    """Baseline (milp_forecast) scores on the dev and holdout windows and
    episodes, under `config`. Cached per configuration, so a run is always
    compared with a baseline that faced the same rules (e.g. ramp rate)."""
    scores = {"config": config_to_dict(config), "sim_version": SIM_VERSION}
    for name, windows in REFERENCE_SETS.items():
        scores[name] = evaluate(REFERENCE_POLICY_DIR, config, windows, sandboxed=False, timeout_s=7200)
        print(f"{name}: {scores[name]['pct_of_perfect']:.1f}% of perfect foresight "
              f"({scores[name]['eval_seconds']:.0f}s)", flush=True)
    path = reference_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scores, indent=2, default=str))
    return scores


def load_reference(config: Config) -> dict | None:
    """Cached baseline scores for this configuration, or None if not computed yet."""
    path = reference_path(config)
    return json.loads(path.read_text()) if path.exists() else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    region = sub.add_parser("region", help="internal: evaluate one region")
    region.add_argument("--policy-dir", required=True)
    region.add_argument("--region", required=True)
    region.add_argument("--start", required=True)
    region.add_argument("--end", required=True)
    region.add_argument("--db", required=True)
    region.add_argument("--config-json", required=True)
    region.add_argument("--no-sandbox", action="store_true")
    region.add_argument("--out-dir")
    region.add_argument("--tag", help="output file prefix (default: the region)")
    ref = sub.add_parser("reference", help="compute and cache baseline scores for a configuration")
    ref.add_argument("--force", action="store_true", help="recompute even if already cached")
    ref.add_argument("--config", default=None)
    ref.add_argument("--ramp-rate", type=float, help="override battery.ramp_rate_mw_per_min")
    args = parser.parse_args()

    if args.cmd == "region":
        print(json.dumps(_run_region(args)))
    else:
        import dataclasses

        config = load_config(args.config) if args.config else load_config()
        if args.ramp_rate is not None:
            config = dataclasses.replace(config, battery=dataclasses.replace(config.battery, ramp_rate_mw_per_min=args.ramp_rate))
        print(f"config {config_fingerprint(config)}", flush=True)
        if reference_path(config).exists() and not args.force:
            print(f"Already cached: {reference_path(config)} (use --force to recompute)")
        else:
            compute_reference(config)
