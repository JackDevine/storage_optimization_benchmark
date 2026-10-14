"""Leaderboard data for the headline-figure notebook: the v2 leaderboard runs, their
submissions, per-model summaries and the TWAP baseline, in % of perfect foresight or
$k per MW per year, on the 12-month holdout or the 12 feedback episodes.

The notebook reads `results/v2/` (committed), so the figures can be rebuilt without
the run directories. To refresh it from `runs/` (not committed):

    uv run python notebooks/headline_data.py
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
RESULTS = ROOT / "results" / "v2"

# Model names and Artificial Analysis Intelligence Index at the effort each ran at
# (artificialanalysis.ai, checked 12-13 Oct 2026; DeepSeek and GLM aren't listed at exactly our setting).
MODELS = {
    "claude-opus-5-5":              ("Opus 5.5", "max", 58),
    "openai/gpt-6.1-sol":           ("GPT-6.1 Sol", "max", 52),
    "xiaomi/mimo-v2.6-pro":         ("MiMo v2.6 Pro", "high", 46),
    "deepseek/deepseek-v4.1-flash": ("DeepSeek V4.1 Flash", "high", 39),
    "z-ai/glm-5.3":                 ("GLM 5.3", "low", 34),
    "openai/gpt-oss-120b":          ("gpt-oss-120b", "high", 12),
}
SHORT = {"Opus 5.5": "Opus", "GPT-6.1 Sol": "GPT-6.1 Sol", "MiMo v2.6 Pro": "MiMo", "DeepSeek V4.1 Flash": "DeepSeek",
         "GLM 5.3": "GLM", "gpt-oss-120b": "gpt-oss"}
SET_LABEL = {"full": "12-month holdout", "episodes": "12 feedback episodes"}


@dataclass
class Leaderboard:
    score_set: str
    units: str
    runs: pd.DataFrame        # one row per run; `score` is the chosen set in the chosen units
    subs: pd.DataFrame        # one row per submission; `episodes` in the chosen units
    board: pd.DataFrame       # one row per model, sorted by mean score
    twap: dict                # TWAP baseline per set, in the chosen units
    per_point: dict           # $k/MW/yr per percentage point, per set (1.0 in pct units)

    def fmt(self, v: float) -> str:
        return f"{v:.1f}%" if self.units == "pct" else f"${v:.0f}k"

    @property
    def unit_label(self) -> str:
        return "% of perfect foresight" if self.units == "pct" else "$k per MW per year"


def export(dest: Path = RESULTS) -> None:
    """Write runs.csv, submissions.csv and reference.json from the run directories."""
    run_ids = re.findall(r"^--- start (\S+)", (RUNS / "leaderboard-v2.log").read_text(), flags=re.M)
    runs, subs = [], []
    for run_id in run_ids:
        s = json.loads((RUNS / run_id / "summary.json").read_text())
        name, effort, index = MODELS[s["model"]]
        rep = sum(r["model"] == name for r in runs) + 1
        cost = tokens = 0.0
        at_submit = {}
        for line in (RUNS / run_id / "logs" / "events.jsonl").read_text().splitlines():
            e = json.loads(line)
            if e["event"] == "api_call":
                cost, tokens = e.get("cumulative_cost_usd", cost), tokens + e["usage"].get("output_tokens", 0)
            elif e["event"] == "submission_start":
                at_submit[e["n"]] = (cost, tokens)
        for x in s.get("submissions", []):
            c, t = at_submit.get(x["n"], (cost, tokens))
            subs.append({"model": name, "rep": rep, "n": x["n"], "active_min": x["active_s"] / 60, "cost": c,
                         "out_tokens": t, "valid": x["valid"], "episodes_pct": x.get("pct_of_perfect")})
        runs.append({"model": name, "slug": s["model"], "effort": effort, "index": index, "rep": rep, "run_id": run_id,
                     "full_pct": s.get("final_full_holdout_pct_of_perfect"), "episodes_pct": s.get("final_pct_of_perfect"),
                     "cost": s["cost_usd"], "billing": s["billing"], "active_min": s["active_seconds"] / 60,
                     "out_tokens": s["tokens"]["output_tokens"], "submissions": len(s.get("submissions", [])),
                     "ended_by": s["ended_by"]})

    reference = json.loads((RUNS / run_ids[0] / "reference.json").read_text())
    ref = {}
    for name in ("full", "episodes"):
        r = reference[f"holdout_{name}"]
        ref[name] = {
            "twap_pct": r["pct_of_perfect"],
            "lp_forecast_pct": reference["other_baselines"]["lp_forecast"][f"holdout_{name}"]["pct_of_perfect"],
            "perfect_foresight_value": r["perfect_foresight_value"],   # $, summed over every region and window
            "days": sum((pd.Timestamp(b) - pd.Timestamp(a)).total_seconds() / 86400 for a, b in r["windows"]),
            "regions": len({j["region"] for j in r["jobs"]}),
            "windows": r["windows"],
        }

    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(runs).to_csv(dest / "runs.csv", index=False, float_format="%.6g")
    pd.DataFrame(subs).to_csv(dest / "submissions.csv", index=False, float_format="%.6g")
    (dest / "reference.json").write_text(json.dumps(ref, indent=2) + "\n")


def load(score_set: str = "full", units: str = "pct", src: Path = RESULTS) -> Leaderboard:
    assert score_set in ("full", "episodes") and units in ("pct", "dollars")
    runs = pd.read_csv(src / "runs.csv")
    subs = pd.read_csv(src / "submissions.csv")
    ref = json.loads((src / "reference.json").read_text())

    def per_point(r: dict) -> float:   # $k/MW/yr for one percentage point of perfect foresight's value
        return r["perfect_foresight_value"] / 100 / r["regions"] / (r["days"] / 365) / 1000

    k = {name: (1.0 if units == "pct" else per_point(r)) for name, r in ref.items()}
    runs["full"], runs["episodes"] = runs.full_pct * k["full"], runs.episodes_pct * k["episodes"]
    subs["episodes"] = subs.episodes_pct * k["episodes"]
    runs["score"] = runs[score_set]
    twap = {name: r["twap_pct"] * k[name] for name, r in ref.items()}

    board = (runs.groupby(["model", "effort", "index"])
             .agg(runs_scored=("score", "count"), runs=("score", "size"), mean=("score", "mean"),
                  best=("score", "max"), worst=("score", "min"), cost_per_run=("cost", "mean"))
             .reset_index().sort_values("mean", ascending=False).reset_index(drop=True))
    return Leaderboard(score_set, units, runs, subs, board, twap, k)


def best_so_far(lb: Leaderboard) -> pd.DataFrame:
    """Step-line points of each run's best feedback-episode score so far against active
    time: from 0 at the start, up at each submission that beats the run's best, then
    flat to the end of the run."""
    rows = []
    for (model, rep), run in lb.runs.set_index(["model", "rep"]).iterrows():
        s = lb.subs[(lb.subs.model == model) & (lb.subs.rep == rep) & lb.subs.valid].sort_values("active_min")
        pts, best = [(0.0, 0.0)], 0.0
        for _, x in s.iterrows():
            if x.episodes > best:
                pts += [(x.active_min, best), (x.active_min, x.episodes)]
                best = x.episodes
        pts.append((max(run.active_min, pts[-1][0]), best))
        rows += [{"model": model, "rep": rep, "order": i, "active_min": a, "best": b} for i, (a, b) in enumerate(pts)]
    return pd.DataFrame(rows)


def pareto(runs: pd.DataFrame) -> pd.DataFrame:
    """Runs no other run beats on both cost (lower) and score (higher)."""
    front = []
    for _, r in runs.dropna(subset=["score"]).sort_values(["cost", "score"], ascending=[True, False]).iterrows():
        if not front or r.score > front[-1].score:
            front.append(r)
    return pd.DataFrame(front)


if __name__ == "__main__":
    export()
    print(f"wrote {RESULTS.relative_to(ROOT)}/: runs.csv, submissions.csv, reference.json")
