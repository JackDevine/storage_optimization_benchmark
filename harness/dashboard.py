"""Live dashboard for agent runs.

    uv run python -m harness.dashboard [--port 8050]

Then open http://127.0.0.1:8050. It scrubs through the holdout: actual
prices up to the scrub time, with the P5MIN and predispatch forecasts that
were current at that moment ahead of it. Once the agent submits a policy,
it also shows that submission's state of charge, dispatch and bids, plus the
agent's description of the policy and its reasoning leading up to it. Run
stats (time, tokens, API-equivalent cost, tool calls) refresh as the run goes.

Serves on 127.0.0.1 only. The page reads holdout data, so never expose it to
the agent's sandbox.
"""

import argparse
import json
import threading
import time
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from bess.bid import bid_from_dict, dispatch_at_price, effective_gen_mw, effective_load_mw
from bess.config import load_config
from bess.data import DB_PATH, load_region
from bess.policies import PerfectForesight
from bess.remote import config_from_dict
from bess.sim import simulate
from harness.evaluator import REFERENCE_DIR, REFERENCE_PATH, SIM_VERSION, config_fingerprint, load_reference
from harness.run_agent import CONFIG_PATH, RUNS_DIR
from harness.splits import HOLDOUT_END, HOLDOUT_EPISODES, HOLDOUT_MONTHS, HOLDOUT_START, REGIONS

PAGE = Path(__file__).resolve().parent / "dashboard.html"
LOOKBACK = pd.Timedelta(hours=12)
LOOKAHEAD = pd.Timedelta(hours=40)
CONFIG = load_config(CONFIG_PATH)
_lock = threading.Lock()


def _iso(t) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%S")


def _clean(values) -> list:
    return [None if not np.isfinite(v) else round(float(v), 4) for v in values]


PF_CACHE_DIR = REFERENCE_DIR.parent / "dashboard_cache"


def _cached_per_region(build):
    """Memoise `build(region)`, computing each region once even when several
    requests (or the warm-up thread) ask at the same time. Per-region locks, so
    loading one region never blocks requests for another, or for run data."""
    values, locks = {}, {r: threading.Lock() for r in REGIONS}

    def get(region: str):
        with locks[region]:
            if region not in values:
                values[region] = build(region)
            return values[region]

    return get


region_data = _cached_per_region(lambda region: load_region(region, DB_PATH, HOLDOUT_START, HOLDOUT_END))


@_cached_per_region
def perfect_foresight(region: str) -> pd.DataFrame:
    """Perfect-foresight SOC over the holdout, solved one month at a time as
    it's scored (several seconds a month), so it's cached on disk per config
    fingerprint."""
    path = PF_CACHE_DIR / f"pf_{config_fingerprint(CONFIG)}_{region}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    data = region_data(region)
    pf = pd.concat([
        simulate(PerfectForesight(CONFIG, data.actual_rrp, start, end), data, CONFIG, start, end)
        .set_index("interval_end")[["energy_end_mwh"]]
        for start, end in HOLDOUT_MONTHS
    ])
    PF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pf.to_parquet(path)
    return pf


def warm_up() -> None:
    """Load every region in the background, so switching region doesn't stall."""
    for region in REGIONS:
        region_data(region)
        perfect_foresight(region)


def market(region: str, tau: pd.Timestamp) -> dict:
    data = region_data(region)
    actual = data.actual_rrp.loc[tau - LOOKBACK : tau]
    out = {"tau": _iso(tau), "actual": {"t": [_iso(t) for t in actual.index], "rrp": _clean(actual.values)}}
    p5 = data.view.p5min(tau)
    pdr = data.view.predispatch(tau)
    last_p5 = None
    if p5 is not None:
        keep = p5.interval_end > np.datetime64(tau, "ns")
        out["p5min"] = {
            "run_time": _iso(p5.run_time),
            "t": [_iso(t) for t in p5.interval_end[keep]],
            "rrp": _clean(p5.rrp[keep]),
        }
        last_p5 = p5.interval_end[keep][-1] if keep.any() else None
    if pdr is not None:
        horizon = np.datetime64(tau + LOOKAHEAD, "ns")
        keep = (pdr.interval_end > (last_p5 if last_p5 is not None else np.datetime64(tau, "ns"))) & (
            pdr.interval_end <= horizon
        )
        out["predispatch"] = {
            "run_time": _iso(pdr.run_time),
            "t": [_iso(t) for t in pdr.interval_end[keep]],
            "rrp": _clean(pdr.rrp[keep]),
        }
    pf = perfect_foresight(region).loc[tau - LOOKBACK : tau]
    out["perfect_foresight"] = {"t": [_iso(t) for t in pf.index], "soc": _clean(pf.energy_end_mwh.values)}
    return out


# ---- runs ----


class EventCache:
    """Incrementally parsed events.jsonl for one run."""

    def __init__(self, path: Path):
        self.path, self.offset, self.events = path, 0, []

    def refresh(self) -> list[dict]:
        if self.path.exists():
            with self.path.open() as f:
                f.seek(self.offset)
                for line in f:
                    if line.endswith("\n"):
                        self.events.append(json.loads(line))
                        self.offset += len(line.encode())
        return self.events


_event_caches: dict[str, EventCache] = {}


def run_events(run_id: str) -> list[dict]:
    with _lock:
        cache = _event_caches.setdefault(run_id, EventCache(RUNS_DIR / run_id / "logs" / "events.jsonl"))
        return list(cache.refresh())


def list_runs() -> list[str]:
    if not RUNS_DIR.exists():
        return []
    logs = [p / "logs" / "events.jsonl" for p in RUNS_DIR.iterdir()]
    return [f.parent.parent.name for f in sorted((f for f in logs if f.exists()), key=lambda f: f.stat().st_mtime, reverse=True)]


def run_status(run_id: str) -> dict:
    events = run_events(run_id)
    start = next((e for e in events if e["event"] == "run_start"), None)
    end = next((e for e in events if e["event"] == "run_end"), None)
    api_calls, tools, submissions, feed = {}, {}, {}, []
    final_full = None
    sdk_cost = None
    cost_so_far = 0.0
    for e in events:
        kind = e["event"]
        if kind == "api_call":  # final usage for one API call
            cost_so_far = max(cost_so_far, e.get("cumulative_cost_usd") or 0.0)
            api_calls[e["message_id"]] = e
        elif kind == "assistant_message":
            cost_so_far = max(cost_so_far, e.get("cumulative_cost_usd") or 0.0)
            if e.get("usage"):
                api_calls.setdefault(e["message_id"], e)  # placeholder until its api_call arrives
            for b in e.get("blocks", []):
                feed.append({"elapsed_s": e["elapsed_s"], "kind": b["type"], "text": b.get("text") or f"{b.get('name')} {b.get('input', '')}"})
        elif kind == "tool_end":
            s = tools.setdefault(e["tool"], {"calls": 0, "seconds": 0.0, "failures": 0})
            s["calls"] += 1
            s["seconds"] += e["duration_s"]
            s["failures"] += int(e["failed"])
        elif kind == "submission_start":
            usages = [c["usage"] for c in api_calls.values()]
            submissions[e["n"]] = {
                "n": e["n"], "summary": e.get("summary"), "rationale": e.get("rationale"), "submitted_s": e["elapsed_s"],
                "state": "evaluating",
                # where this submission sits on each x-axis of the score-progress plot
                "x": {
                    "wall_s": e["elapsed_s"],
                    "active_s": e.get("active_s", e["elapsed_s"]),
                    "cost_usd": e.get("cost_usd", cost_so_far),
                    **_token_axes(e.get("tokens") or _sum_tokens(usages)),
                },
            }
            feed.append({"elapsed_s": e["elapsed_s"], "kind": "submission", "text": f"Submitted policy {e['n']}"})
        elif kind == "submission_result":
            s = submissions.setdefault(e["n"], {"n": e["n"]})
            s.update(state="done", valid=e.get("valid"), pct_of_perfect=e.get("pct_of_perfect"), net_revenue=e.get("net_revenue"),
                     eval_seconds=e.get("eval_seconds"), error=e.get("error"), per_region=e.get("per_region"))
            score = f"{e['pct_of_perfect']:.1f}%" if e.get("pct_of_perfect") is not None else "failed"
            feed.append({"elapsed_s": e["elapsed_s"], "kind": "result", "text": f"Policy {e['n']} scored {score}, valid={e.get('valid')}"})
        elif kind == "final_evaluation_result":
            final_full = {k: e.get(k) for k in ("n", "valid", "pct_of_perfect", "net_revenue", "eval_seconds", "per_region", "error")}
            score = f"{e['pct_of_perfect']:.1f}%" if e.get("pct_of_perfect") is not None else "failed"
            feed.append({"elapsed_s": e["elapsed_s"], "kind": "result", "text": f"Full-holdout final score for policy {e['n']}: {score}"})
        elif kind == "final_evaluation_start":
            feed.append({"elapsed_s": e["elapsed_s"], "kind": "submission", "text": f"Scoring policy {e['n']} on the full holdout (every month, scored separately)"})
        elif kind in ("result",):
            sdk_cost = e.get("sdk_total_cost_usd") or sdk_cost
        elif kind in ("limit_reached", "limit_deferred", "finish", "prompt", "pause", "resume", "session_error", "run_resumed"):
            text = e.get("text") or e.get("summary") or e.get("error") or e.get("detail") or ""
            if kind == "pause":
                text = f"Paused ({e.get('reason')}) until {e.get('until_local')}"
            elif kind in ("limit_reached", "limit_deferred", "resume"):
                text = f"{e.get('reason')}. {text}".strip()
            feed.append({"elapsed_s": e["elapsed_s"], "kind": kind, "text": text})

    usages = [e["usage"] for e in api_calls.values()]
    token_keys = ["input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"]
    run_ref = RUNS_DIR / run_id / "reference.json"  # the baseline this run was judged against
    ref_path = run_ref if run_ref.exists() else REFERENCE_PATH  # legacy runs: the old single file
    reference = json.loads(ref_path.read_text()) if ref_path.exists() else None
    valid = [s for s in submissions.values() if s.get("valid")]
    state_path = RUNS_DIR / run_id / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    stale = state.get("saved_at") and time.time() - state["saved_at"] > 120  # harness not saving: process stopped
    holdout_ref = reference.get("holdout_episodes", reference.get("holdout", {})) if reference else {}
    full_ref = reference.get("holdout_full", {}) if reference else {}
    other_baselines = _other_baselines(reference)

    # Where the run's x-axes end. A live run keeps growing; once the agent has
    # stopped, every axis freezes at that moment. The full-holdout final
    # evaluation runs after the agent stops, so it mustn't stretch the axes.
    agent_end = next((e for e in events if e["event"] in ("final_evaluation_start", "run_end")), None)
    stopped = bool(end is None and stale and not state.get("paused_until"))
    if agent_end:
        wall = agent_end["elapsed_s"]
    elif stopped and events:  # harness died: freeze at its last sign of life
        wall = events[-1]["elapsed_s"]
    else:
        wall = time.time() - start["t"] if start else 0
    frozen = agent_end is not None or stopped
    calls_so_far = [c for c in api_calls.values() if not frozen or c["elapsed_s"] <= wall]
    summary_path = RUNS_DIR / run_id / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    active = state.get("active_seconds", summary.get("active_seconds"))
    now_x = {
        "wall_s": wall,
        "active_s": active if active is not None else wall,  # runs from before active time was tracked
        "cost_usd": max((c.get("cumulative_cost_usd") or 0) for c in calls_so_far) if calls_so_far else 0.0,
        **_token_axes(_sum_tokens([c["usage"] for c in calls_so_far])),
    }
    return {
        "now_x": now_x,
        "axes_frozen": frozen,
        "baseline_holdout_net_revenue": holdout_ref.get("net_revenue"),
        "perfect_foresight_holdout_net_revenue": holdout_ref.get("perfect_foresight_net_revenue"),
        "paused_until": state.get("paused_until"),
        "active_s": state.get("active_seconds"),
        "stopped": stopped,
        "run_id": run_id,
        "model": start and start.get("model"),
        "args": start and start.get("args"),
        "running": end is None,
        "ended_by": end and end.get("ended_by"),
        "elapsed_s": wall,
        "cost_usd": max((e.get("cumulative_cost_usd") or 0) for e in api_calls.values()) if api_calls else 0.0,
        "sdk_cost_usd": sdk_cost,
        "api_calls": len(api_calls),
        "tokens": {k: sum((u.get(k) or 0) for u in usages) for k in token_keys},
        "tools": tools,
        "submissions": sorted(submissions.values(), key=lambda s: s["n"]),
        "final_pct_of_perfect": valid[-1]["pct_of_perfect"] if valid else None,
        "baseline_holdout_pct": holdout_ref.get("pct_of_perfect"),
        "baseline_full_holdout_pct": full_ref.get("pct_of_perfect"),
        "other_baselines": other_baselines,
        "final_full_holdout": final_full,
        "feed": feed[-80:],
    }


def _other_baselines(reference: dict | None) -> list[dict]:
    """The baselines besides milp_forecast (static bids), from the run's own
    reference copy, or, for runs started before they existed, from the current
    cache when the run was scored under the current config."""
    others = (reference or {}).get("other_baselines")
    if others is None and reference and reference.get("sim_version") == SIM_VERSION and "config" in reference:
        if config_fingerprint(config_from_dict(reference["config"])) == config_fingerprint(CONFIG):
            others = (load_reference(CONFIG) or {}).get("other_baselines")
    return [
        {"name": name, "holdout_pct": e["holdout_episodes"].get("pct_of_perfect"),
         "holdout_net_revenue": e["holdout_episodes"].get("net_revenue"),
         "full_holdout_pct": e.get("holdout_full", {}).get("pct_of_perfect")}
        for name, e in (others or {}).items() if "holdout_episodes" in e
    ]


def compare_runs() -> dict:
    """Every run's submissions, for the cross-run comparison chart. A run is
    comparable when it was scored under the current config and simulator
    (same fingerprint); older runs used different market rules."""
    current = config_fingerprint(CONFIG)
    runs = []
    for run_id in list_runs():
        s = run_status(run_id)
        ref_path = RUNS_DIR / run_id / "reference.json"
        ref = json.loads(ref_path.read_text()) if ref_path.exists() else {}
        same_sim = ref.get("sim_version") == SIM_VERSION and "config" in ref
        fingerprint = config_fingerprint(config_from_dict(ref["config"])) if same_sim else None
        runs.append({
            "run_id": run_id, "model": s["model"], "running": s["running"], "comparable": fingerprint == current,
            "started": next((e["t"] for e in run_events(run_id)), None),
            **{k: s[k] for k in ("submissions", "now_x", "axes_frozen", "final_full_holdout",
                                 "baseline_holdout_pct", "baseline_holdout_net_revenue", "baseline_full_holdout_pct",
                                 "other_baselines")},
        })
    return {"fingerprint": current, "runs": runs}


TOKEN_KEYS = ["input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"]


def _sum_tokens(usages: list[dict]) -> dict:
    return {k: sum((u.get(k) or 0) for u in usages) for k in TOKEN_KEYS}


def _token_axes(t: dict) -> dict:
    return {"output_tokens": t.get("output_tokens", 0), "total_tokens": sum(t.get(k, 0) for k in TOKEN_KEYS)}


def narrative(run_id: str, n: int) -> list[dict]:
    """The agent's text and thinking between submission n-1 and submission n."""
    events = run_events(run_id)
    starts = {e["n"]: e["elapsed_s"] for e in events if e["event"] == "submission_start"}
    lo, hi = starts.get(n - 1, -1), starts.get(n, float("inf"))
    out = []
    for e in events:
        if e["event"] == "assistant_message" and lo < e["elapsed_s"] <= hi:
            for b in e.get("blocks", []):
                if b["type"] in ("text", "thinking") and b.get("text"):
                    out.append({"elapsed_s": e["elapsed_s"], "kind": b["type"], "text": b["text"]})
    return out


def policy_source(run_id: str, n: int) -> dict:
    snap = RUNS_DIR / run_id / "submissions" / f"{n:02d}"
    files = sorted(p for p in (snap / "policy").rglob("*.py")) if (snap / "policy").exists() else []
    description = snap / "description.json"
    return {
        "description": json.loads(description.read_text()) if description.exists() else None,
        "files": [{"path": str(p.relative_to(snap / "policy")), "source": p.read_text()[:100_000]} for p in files],
    }


@lru_cache(maxsize=64)
def _intervals(path: Path, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path).set_index("interval_end")


def _results_file(run_id: str, n: int, region: str, tau: pd.Timestamp) -> tuple[Path | None, str]:
    """The interval-results file covering decision time tau: the month of the
    full-holdout final eval containing it if that exists (complete), else the
    episode containing tau."""
    snap = RUNS_DIR / run_id / "submissions" / f"{n:02d}"
    for k, (start, end) in enumerate(HOLDOUT_MONTHS):
        path = snap / "eval_full" / f"{region}_ep{k}_intervals.parquet"
        if start <= tau < end and path.exists():
            return path, f"full holdout, {start + pd.Timedelta(days=1):%B %Y}"
    for k, (start, end) in enumerate(HOLDOUT_EPISODES):
        if start <= tau < end:
            path = snap / "eval" / f"{region}_ep{k}_intervals.parquet"
            return (path, f"episode {k + 1}") if path.exists() else (None, "episode not scored yet")
    return None, "outside the scored episodes"


def dispatch(run_id: str, n: int, region: str, tau: pd.Timestamp) -> dict:
    path, source = _results_file(run_id, n, region, tau)
    if path is None:
        return {"available": False, "reason": source}
    df = _intervals(path, path.stat().st_mtime)
    past = df.loc[tau - LOOKBACK : tau]
    out = {
        "available": True,
        "source": source,
        "t": [_iso(t) for t in past.index],
        "soc": _clean(past.energy_end_mwh.values),
        "mw": _clean(past.mw.values),
        "rejected": [bool(x) for x in past.bid_rejected.notna()],
        "nonconforming": [bool(abs(x) > 1e-6) for x in past.nonconformance_mw.values],
        "wrong_ramp": [bool(x) for x in past.bid_nonconformance.notna()] if "bid_nonconformance" in past else [False] * len(past),
        "net_revenue_to_date": float((df.loc[:tau, "revenue"] - df.loc[:tau, "degradation_cost"]).sum()),
    }
    nxt = tau + pd.Timedelta(minutes=5)
    if nxt in df.index:
        row = df.loc[nxt]
        rejected = None if pd.isna(row.bid_rejected) else row.bid_rejected
        out["next"] = {"interval_end": _iso(nxt), "rejected": rejected, "energy_start_mwh": float(row.energy_start_mwh)}
        if isinstance(row.get("bid"), str):
            bid = bid_from_dict(json.loads(row.bid))
            prices = sorted(set(bid.load.prices) | set(bid.gen.prices) | {CONFIG.market.mfp, CONFIG.market.mpc})
            out["next"]["bid"] = {
                "load": {"prices": list(bid.load.prices), "band_mw": list(bid.load.band_mw), "effective_mw": _clean(effective_load_mw(bid.load)), "max_avail": bid.load.max_avail},
                "gen": {"prices": list(bid.gen.prices), "band_mw": list(bid.gen.band_mw), "effective_mw": _clean(effective_gen_mw(bid.gen)), "max_avail": bid.gen.max_avail},
                "curve": {"price": prices, "mw": [dispatch_at_price(bid, p, CONFIG.market) for p in prices]},
            }
    return out


# ---- server ----


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                return self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
            if url.path == "/api/meta":
                body = {"regions": REGIONS, "start": _iso(HOLDOUT_START), "end": _iso(HOLDOUT_END), "runs": list_runs(),
                        "episodes": [[_iso(a), _iso(b)] for a, b in HOLDOUT_EPISODES]}
            elif url.path == "/api/market":
                body = market(q["region"], pd.Timestamp(q["tau"]))
            elif url.path == "/api/run":
                body = run_status(q["id"])
            elif url.path == "/api/compare":
                body = compare_runs()
            elif url.path == "/api/narrative":
                body = narrative(q["id"], int(q["n"]))
            elif url.path == "/api/policy":
                body = policy_source(q["id"], int(q["n"]))
            elif url.path == "/api/dispatch":
                body = dispatch(q["id"], int(q["n"]), q["region"], pd.Timestamp(q["tau"]))
            else:
                return self._send(b"not found", "text/plain", 404)
            self._send(json.dumps(body, default=str).encode(), "application/json")
        except Exception as e:  # surface errors to the page rather than dropping the connection
            self._send(json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json", 500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    threading.Thread(target=warm_up, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Dashboard: http://127.0.0.1:{args.port}")
    server.serve_forever()
