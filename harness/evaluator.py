"""Score a policy directory over one or more time windows.

A policy directory holds `policy.py` (defining `class Policy`, see
bess/remote.py) and any helper files it imports. Every (region, window) pair
is a separate job: a `python -m harness.evaluator region ...` subprocess with a
wall-clock timeout, which drives the policy through RemotePolicy. Jobs run in
parallel. For holdout evaluation the policy process is additionally sandboxed
(harness/sandbox.py).

Submissions are scored on sampled 7-day episodes (harness/splits.py) so they
take about one episode's simulated time. Each window starts from the battery's
initial SOC. In every job:
- value = net revenue + the change in the stored energy's value over the
  window, V(e_end) - V(e_start) (bess/terminal.py; the same V for everyone);
- the TWAP baseline runs on the same window, and each trading day the policy
  falls behind it (value per day; the energy-value change counts on the last
  day) adds to its shortfall;
- perfect foresight plans with the same terminal value.
The score is (sum of value - SHORTFALL_LAMBDA x sum of shortfall) as a
percentage of perfect foresight's value, pooled over all jobs. Scores at
LAMBDAS and with equal weight per window are reported alongside.

    uv run python -m harness.evaluator reference   # cache baseline scores (all baselines)
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from bess.config import Config, load_config
from bess.data import DB_PATH, load_region
from bess.policies import PerfectForesight, TWAPPolicy
from bess.terminal import credit, end_value
from bess.remote import RemotePolicy, RemotePolicyError, config_from_dict, config_to_dict
from bess.sim import simulate, summarise
from harness.sandbox import PROJECT_ROOT, policy_sandbox
from harness.splits import DATA_VERSION, DEV_EPISODES, DEV_MONTHS, HOLDOUT_EPISODES, HOLDOUT_MONTHS, REGIONS, Window

REFERENCE_POLICY_DIR = Path(__file__).resolve().parent / "reference_policy"
REFERENCE_NAME = "twap"  # the headline baseline: bess.policies.TWAPPolicy
# Further baselines, scored alongside the reference policy: name -> policy
# directory. (The static-bid baselines in static_bid_policies/ are parked: their
# parameters were fitted to the old two-month split.)
OTHER_BASELINES = {
    "lp_forecast": Path(__file__).resolve().parent / "lp_policy",
}
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference_scores"
REFERENCE_PATH = PROJECT_ROOT / "data" / "reference_scores.json"  # legacy: one file for the default config
# Bump when the simulator's scoring rules change, so baselines get recomputed.
# 3: built-in policies and perfect foresight conform their bids (bess.bid.conform); TWAP headline baseline.
# 2: straight-line dispatch trajectory and the unit's ramp rate (1: flat MW, bid's own ramp rates; removed).
# 4: value includes the stored energy's change in value; score penalises daily shortfall against TWAP.
SIM_VERSION = 4
SHORTFALL_LAMBDA = 0.5  # the headline score's (and the agent's) penalty per $ of shortfall
LAMBDAS = (0.0, 0.5, 1.0)  # scores reported alongside


def config_fingerprint(config: Config) -> str:
    """Identifies everything that affects a score: the full config (battery,
    market rules, terminal condition, ...) plus the simulator and data versions."""
    d = config_to_dict(config)
    d.pop("name")
    blob = json.dumps({"sim_version": SIM_VERSION, "data_version": DATA_VERSION, **_floats(d)}, sort_keys=True)
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
        # A job that dies without a word was killed from outside (a policy crash
        # leaves a traceback), so it's retried once rather than charged to the policy.
        for attempt in range(2):
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
            if proc.returncode == 0:
                result = json.loads(stdout.strip().splitlines()[-1])
                if result.get("killed") and attempt == 0:
                    continue  # the policy process was killed from outside (e.g. out of memory): run it again
                return {**result, **meta}
            if (stderr or stdout).strip():
                return {**meta, "error": (stderr or stdout)[-2000:]}
        return {**meta, "error": f"the scoring process exited with code {proc.returncode} and no output, twice"}

    # An explicit max_workers is capped by memory too: an agent's own script
    # asking for more jobs than fit would push the machine into swap.
    workers = min(max_workers or max(1, (os.cpu_count() or 2) - 2), len(jobs), _memory_workers(job_memory_gb(policy_dir)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, jobs))
    return {
        **aggregate(results),
        "jobs": results,
        "windows": [[str(a), str(b)] for a, b in windows],
        "eval_seconds": time.time() - t0,
    }


JOB_MEMORY_GB = 0.6  # a scoring job (about 0.3 GB) and a Python policy process (about 0.2 GB)


def job_memory_gb(policy_dir: Path) -> float:
    """Memory to budget for one scoring job of this policy."""
    return JOB_MEMORY_GB


def _memory_workers(per_job_gb: float = JOB_MEMORY_GB) -> int:
    """How many scoring jobs fit in the memory available now, keeping 2 GB spare."""
    try:
        with open("/proc/meminfo") as f:
            avail_kb = next(int(line.split()[1]) for line in f if line.startswith("MemAvailable:"))
    except (OSError, StopIteration):
        return 1_000
    return max(2, int((avail_kb / 1e6 - 2) / per_job_gb))


def score(value: float, shortfall: float, perfect: float, lam: float = SHORTFALL_LAMBDA) -> float | None:
    return 100 * (value - lam * shortfall) / perfect if perfect else None


def aggregate(jobs: list[dict]) -> dict:
    errors = [j for j in jobs if "error" in j]
    if errors:
        return {
            "valid": False,
            "error": "; ".join(
                f"{j['region']} episode {j['episode']}: {(j['error'].strip().splitlines() or ['no output'])[-1]}"
                for j in errors[:5]
            ),
        }

    def sums(js):
        return (sum(j["value"] for j in js), sum(j["shortfall"] for j in js), sum(j["perfect_foresight_value"] for j in js))

    def pct_by(key) -> dict:
        groups = {}
        for j in jobs:
            groups.setdefault(j[key], []).append(j)
        return {k: score(*sums(js)) for k, js in groups.items()}

    value, shortfall, perfect = sums(jobs)
    by_window = pct_by("episode")
    return {
        "valid": all(j["valid"] for j in jobs),
        "pct_of_perfect": score(value, shortfall, perfect),  # the headline score
        "score_by_lambda": {f"{lam:g}": score(value, shortfall, perfect, lam) for lam in LAMBDAS},
        "equal_window_pct": sum(by_window.values()) / len(by_window),
        "net_revenue": sum(j["net_revenue"] for j in jobs),
        "terminal_credit": sum(j["terminal_credit"] for j in jobs),
        "value": value,
        "shortfall": shortfall,
        "twap_value": sum(j["twap_value"] for j in jobs),
        "perfect_foresight_net_revenue": sum(j["perfect_foresight_net_revenue"] for j in jobs),
        "perfect_foresight_value": perfect,
        "rejected_bids": sum(j["rejected_bids"] for j in jobs),
        "nonconformance_intervals": sum(j["nonconformance_intervals"] for j in jobs),
        "bid_nonconformance_intervals": sum(j.get("bid_nonconformance_intervals", 0) for j in jobs),
        "per_region_pct": pct_by("region"),
        "per_episode_pct": by_window,
    }


def daily_value(intervals: pd.DataFrame, terminal_credit: float) -> pd.Series:
    """Net revenue per trading day (04:00 to 04:00), the energy-value change on the last day."""
    day = (intervals.interval_end - pd.Timedelta(minutes=5) - pd.Timedelta(hours=4)).dt.floor("D")
    daily = (intervals.revenue - intervals.degradation_cost).groupby(day).sum()
    daily.iloc[-1] += terminal_credit
    return daily


def _run_region(args) -> dict:
    config = config_from_dict(json.loads(args.config_json))
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    data = load_region(args.region, Path(args.db), start, end)
    policy_dir = Path(args.policy_dir)
    out_dir = Path(args.out_dir) if args.out_dir else None
    tag = args.tag or args.region
    # The policy's stderr goes to a file even without out_dir, to tell a policy
    # that crashed (it says why) from one killed from outside (it says nothing).
    stderr = open(out_dir / f"{tag}_stderr.log", "w+") if out_dir else tempfile.TemporaryFile("w+")

    def silent() -> bool:
        stderr.flush()
        stderr.seek(0)
        return not stderr.read().strip()

    killed = {"region": args.region, "killed": True,
              "error": "the policy process was killed from outside (no error output)"}
    t0 = time.time()
    try:
        policy = RemotePolicy(
            policy_dir / "policy.py", config, args.region,
            wrapper=None if args.no_sandbox else policy_sandbox(policy_dir),
            cwd=policy_dir, stderr=stderr,
        )
    except RemotePolicyError:
        if silent():
            return killed
        raise
    try:
        intervals = simulate(policy, data, config, start, end)
    finally:
        policy.close()
    policy_seconds = time.time() - t0
    if policy.exited and silent():
        # The policy process went away without a word: killed from outside
        # (a crash leaves a traceback). Not the policy's fault, so say so.
        return killed
    b = config.battery
    terminal = end_value(data, end, b)
    summary = summarise(intervals, b)
    agent_credit = credit(terminal, float(intervals.energy_end_mwh.iloc[-1]), b.e_initial)
    pf_intervals = simulate(PerfectForesight(config, data.actual_rrp, start, end, terminal), data, config, start, end)
    perfect = summarise(pf_intervals, b)
    pf_credit = credit(terminal, float(pf_intervals.energy_end_mwh.iloc[-1]), b.e_initial)
    twap_intervals = simulate(TWAPPolicy(config), data, config, start, end)
    twap_credit = credit(terminal, float(twap_intervals.energy_end_mwh.iloc[-1]), b.e_initial)
    agent_daily, twap_daily = daily_value(intervals, agent_credit), daily_value(twap_intervals, twap_credit)
    shortfall = float((twap_daily - agent_daily.reindex(twap_daily.index, fill_value=0.0)).clip(lower=0).sum())
    if out_dir:
        intervals.to_parquet(out_dir / f"{tag}_intervals.parquet")
    value = summary["net_revenue"] + agent_credit
    pf_value = perfect["net_revenue"] + pf_credit
    return {
        "region": args.region,
        "valid": summary["valid"],
        "net_revenue": summary["net_revenue"],
        "terminal_credit": agent_credit,
        "value": value,
        "shortfall": shortfall,
        "twap_value": float(twap_daily.sum()),
        "perfect_foresight_net_revenue": perfect["net_revenue"],
        "perfect_foresight_value": pf_value,
        "pct_of_perfect": score(value, shortfall, pf_value),
        "daily_value": {str(k.date()): float(v) for k, v in agent_daily.items()},
        "twap_daily_value": {str(k.date()): float(v) for k, v in twap_daily.items()},
        "rejected_bids": summary["rejected_bids"],
        "nonconformance_intervals": summary["nonconformance_intervals"],
        "nonconformance_mwh": summary["nonconformance_mwh"],
        "bid_nonconformance_intervals": summary["bid_nonconformance_intervals"],
        "first_bid_nonconformance": intervals.bid_nonconformance.dropna().head(1).tolist(),
        "equivalent_cycles": summary["equivalent_cycles"],
        "policy_seconds": policy_seconds,
        "first_rejection": intervals.bid_rejected.dropna().head(1).tolist(),
    }


FULL_MONTH_TIMEOUT_FACTOR = 31 / 7  # a one-month job (the full holdout runs one job per month) vs a 7-day episode job

REFERENCE_SETS = {
    "dev": DEV_MONTHS,
    "dev_episodes": DEV_EPISODES,
    "holdout_episodes": HOLDOUT_EPISODES,  # what the agent's submissions are compared against
    "holdout_full": HOLDOUT_MONTHS,  # the headline comparison
}


def compute_reference(config: Config) -> dict:
    """Reference policy (TWAP) scores on the dev and holdout windows and
    episodes, under `config`, plus the other baselines (see update_reference).
    Cached per configuration, so a run is always compared with a baseline that
    faced the same rules (e.g. ramp rate)."""
    scores = {"config": config_to_dict(config), "sim_version": SIM_VERSION}
    for name, windows in REFERENCE_SETS.items():
        scores[name] = evaluate(REFERENCE_POLICY_DIR, config, windows, sandboxed=False, timeout_s=7200)
        print(f"{name}: {scores[name]['pct_of_perfect']:.1f}% of perfect foresight "
              f"({scores[name]['eval_seconds']:.0f}s)", flush=True)
    _save_reference(config, scores)
    return update_reference(config)


def _save_reference(config: Config, scores: dict) -> None:
    path = reference_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scores, indent=2, default=str))


def _params_sha(policy_dir: Path) -> str | None:
    """Identifies a baseline policy directory's files (code and any fitted parameters)."""
    files = sorted(p for p in Path(policy_dir).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    if not files:
        return None
    h = hashlib.sha1()
    for p in files:
        h.update(str(p.relative_to(policy_dir)).encode() + p.read_bytes())
    return h.hexdigest()[:12]


def update_reference(config: Config) -> dict:
    """Add the other baselines (OTHER_BASELINES) to the cached scores for
    `config`, computing any that are missing or whose fitted parameters have
    changed since. The reference policy's (REFERENCE_NAME) scores stay at the top level; the others
    go under "other_baselines"[name][set name]. Computes everything if nothing
    is cached yet."""
    scores = load_reference(config)
    if scores is None:
        return compute_reference(config)
    others = scores.setdefault("other_baselines", {})
    changed = False
    for name, policy_dir in OTHER_BASELINES.items():
        sha = _params_sha(policy_dir)
        if sha is None or others.get(name, {}).get("params_sha") == sha:
            continue
        entry = {"params_sha": sha}
        for set_name, windows in REFERENCE_SETS.items():
            entry[set_name] = evaluate(policy_dir, config, windows, sandboxed=False, timeout_s=7200)
            print(f"{name} {set_name}: {entry[set_name]['pct_of_perfect']:.1f}% of perfect foresight "
                  f"({entry[set_name]['eval_seconds']:.0f}s)", flush=True)
        others[name] = entry
        changed = True
    if changed:
        _save_reference(config, scores)
    return scores


def baseline_pcts(reference: dict, set_name: str) -> dict[str, float]:
    """% of perfect foresight on one reference set, for every baseline in
    `reference`: {"twap": ..., "lp_forecast": ...}."""
    out = {REFERENCE_NAME: reference[set_name]["pct_of_perfect"]}
    for name, entry in reference.get("other_baselines", {}).items():
        if set_name in entry:
            out[name] = entry[set_name]["pct_of_perfect"]
    return out


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
            update_reference(config)  # adds or refreshes the other baselines if needed
        else:
            compute_reference(config)
