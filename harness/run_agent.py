"""Run a model as an agent that writes a battery bidding policy.

    uv run python -m harness.run_agent [--model claude-sonnet-5-5] [--max-hours 2] [--max-cost 20]
    uv run python -m harness.run_agent --model openai/gpt-5.1      # any OpenRouter slug

Two backends, picked by the model name:
- Claude models (names in PRICES) run in Claude Code through the Claude Agent
  SDK on your Claude subscription. Create a token with `claude setup-token`,
  then either export it as CLAUDE_CODE_OAUTH_TOKEN or save it to
  ~/.config/storage-benchmark/claude_oauth_token.
- OpenRouter slugs (anything with a `/`, e.g. `google/gemini-3-pro`) run in
  this harness's own agent loop (harness/openrouter_agent.py), billed to your
  OpenRouter credits. Put the key in OPENROUTER_API_KEY or
  ~/.config/storage-benchmark/openrouter_api_key.

The agent works in a sandboxed workspace (harness/sandbox.py) holding the task
brief (harness/task.md), the dev data and a starter policy. It can run
`harness.dev_eval` itself and calls the `submit_policy` tool to have its policy
scored on the holdout episodes.

Runs survive interruptions without losing anything:
- Hitting the subscription's usage limit (or, for OpenRouter, a rate limit or
  provider error) pauses the run, then resumes the same conversation. --max-hours measures active time,
  which excludes these pauses and the time spent scoring submissions.
- If the CLI crashes or the connection drops, the harness retries with
  backoff and resumes the conversation.
- If the harness process itself stops (reboot, Ctrl-C, a weekly limit longer
  than --max-pause-hours), restart it with `--resume <run_id>`. It rebuilds its
  state from the run's logs and state.json; a submission whose evaluation was
  cut off is re-evaluated.

Everything lands in runs/<run_id>/:
- workspace/             the agent's working directory
- submissions/NN/        snapshot of each submitted policy, plus eval/ outputs and result.json
- logs/events.jsonl      one JSON line per event: API calls (final tokens, API-equivalent cost, duration),
                         tool calls (duration, sizes), submissions, nudges, limits, result messages
- logs/transcript.jsonl  every SDK message (Claude) or raw API response (OpenRouter), in full
- logs/messages.jsonl    OpenRouter only: the conversation, for resuming
- state.json             resumable state: session id, active time, nudges, how the run ended
- summary.json           totals and the final score

For Claude runs, cost is what the tokens would cost at standard API rates; the
run itself is billed to the subscription. For OpenRouter runs it's the real
cost OpenRouter reports, and --max-cost (default $20) is a real spending cap.
"""

import argparse
import asyncio
import dataclasses
import json
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
    StreamEvent,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
    tool,
)

from bess.config import load_config
from harness.evaluator import (
    FULL_MONTH_TIMEOUT_FACTOR, REFERENCE_NAME, baseline_pcts, config_fingerprint, evaluate, load_reference, update_reference,
)
from harness.sandbox import PROJECT_ROOT, write_agent_cli_wrapper
from harness.splits import DEV_EPISODES, HOLDOUT_EPISODES, HOLDOUT_MONTHS

RUNS_DIR = PROJECT_ROOT / "runs"
TOKEN_FILE = Path.home() / ".config" / "storage-benchmark" / "claude_oauth_token"
TASK_TEMPLATE = Path(__file__).resolve().parent / "task.md"
DEV_DB = PROJECT_ROOT / "data" / "dev.duckdb"
CONFIG_PATH = PROJECT_ROOT / "config" / "default.toml"

# Standard API rates, $ per million tokens: input, output, cache read, 5-min
# cache write, 1-hour cache write.
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50, 4.00),
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00, 8.00),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125, 0.20),
    "claude-fable-5-1": (10.00, 50.00, 0.25, 12.50, 20.00),
}

STARTER_POLICY = '''"""Starter policy: never dispatches. Replace this with your own."""

from bess.bid import setpoint_bid


class Policy:
    def __init__(self, config, region):
        self.config = config

    def decide(self, obs):
        return setpoint_bid(0.0, self.config.battery, self.config.market)
'''

# The harness tools, described the same way to every backend.
SUBMIT_POLICY_DESCRIPTION = (
    "Score the policy currently in the workspace (policy.py plus helper files) on the sampled holdout "
    "episodes. Takes a few minutes. Limited submissions; your final submission is your last valid one, "
    "and it is also scored on the whole holdout period after the run. "
    "`summary`: a short description of what the policy does (how it forms bids, how it manages state "
    "of charge). `rationale`: why you expect this approach to work, and what you changed since your "
    "last submission and why."
)
STATUS_DESCRIPTION = "Show remaining time, model budget and submissions."
FINISH_DESCRIPTION = "End the run. Call this when you are done."

AGENT_TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "TodoWrite"]
SNAPSHOT_IGNORE = shutil.ignore_patterns("data", "dev_results", "__pycache__", ".*", "*.parquet")
MAX_SNAPSHOT_BYTES = 200 * 1024 * 1024


def usage_cost(model: str, usage: dict) -> float:
    p_in, p_out, p_read, p_write_5m, p_write_1h = PRICES[model]
    creation = usage.get("cache_creation") or {}
    write_1h = creation.get("ephemeral_1h_input_tokens", 0) or 0
    write_5m = (usage.get("cache_creation_input_tokens", 0) or 0) - write_1h
    return (
        (usage.get("input_tokens", 0) or 0) * p_in
        + (usage.get("output_tokens", 0) or 0) * p_out
        + (usage.get("cache_read_input_tokens", 0) or 0) * p_read
        + write_5m * p_write_5m
        + write_1h * p_write_1h
    ) / 1e6


class EventLog:
    def __init__(self, logs_dir: Path, t0: float | None = None):
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.t0 = t0 or time.time()
        self._events = open(logs_dir / "events.jsonl", "a")
        self._transcript = open(logs_dir / "transcript.jsonl", "a")

    def elapsed(self) -> float:
        return time.time() - self.t0

    def event(self, kind: str, **fields) -> None:
        record = {"t": time.time(), "elapsed_s": round(self.elapsed(), 3), "event": kind, **fields}
        self._events.write(json.dumps(record, default=str) + "\n")
        self._events.flush()

    def transcript(self, message) -> None:
        if isinstance(message, dict):  # a raw API response
            body = message
        else:
            body = dataclasses.asdict(message) if dataclasses.is_dataclass(message) else repr(message)
        record = {"t": time.time(), "type": "api_response" if isinstance(message, dict) else type(message).__name__,
                  "message": body}
        self._transcript.write(json.dumps(record, default=str) + "\n")
        self._transcript.flush()


def is_openrouter(model: str) -> bool:
    """OpenRouter slugs are `vendor/model`; Claude models on the subscription have no `/`."""
    return "/" in model


class Run:
    """Everything shared by the agent backends: the workspace, submissions and
    scoring, the time and cost budget, pausing and resuming, and the summary.
    A backend supplies `session`, `interrupt`, `call_cost` and, for resuming,
    `restore_conversation`."""

    billing = "subscription"

    def __init__(self, args):
        self.args = args
        self.model = args.model
        self.run_id = args.run_id or f"{datetime.now():%Y%m%d-%H%M%S}-{self.model.replace('/', '_')}"
        self.dir = RUNS_DIR / self.run_id
        self.workspace = self.dir / "workspace"
        self.submissions_dir = self.dir / "submissions"
        self.state_path = self.dir / "state.json"
        self.config = load_config(CONFIG_PATH)
        if getattr(args, "ramp_rate", None) is not None:  # per-run override of the unit's ramp rate
            self.config = dataclasses.replace(
                self.config, battery=dataclasses.replace(self.config.battery, ramp_rate_mw_per_min=args.ramp_rate)
            )
        # Baseline scores under this run's exact configuration. A run keeps its own
        # copy, so later changes to the cache never change how it was judged.
        self.reference_copy = self.dir / "reference.json"
        if self.reference_copy.exists():
            self.reference = json.loads(self.reference_copy.read_text())
        else:
            self.reference = load_reference(self.config)  # None until computed (see ensure_reference)
        events_path = self.dir / "logs" / "events.jsonl"
        first = json.loads(events_path.open().readline()) if events_path.exists() else None
        self.log = EventLog(self.dir / "logs", t0=first["t"] if first else None)

        self.api_calls: dict[str, dict] = {}  # message_id -> usage (final once its message_delta arrives)
        self.streaming: dict | None = None  # the API response currently streaming
        self.last_message_at = time.time()
        self.tool_starts: dict[str, float] = {}
        self.tool_stats: dict[str, dict] = {}
        self.submissions: list[dict] = []
        self.sdk_cost_usd = 0.0
        self.result_messages = 0
        self.finished_reason: str | None = None
        self.limit_reason: str | None = None
        self.agent_summary: str | None = None

        # Resumability: the Claude Code session to resume, and active (unpaused) time.
        self.session_id: str | None = None
        self.nudges = 0
        self.active_before = 0.0  # active seconds in earlier segments
        self.segment_start: float | None = None  # set while the agent is active
        self.client: ClaudeSDKClient | None = None
        self.evaluating = False
        self.submission_eval_seconds = 0.0  # wall time spent scoring submissions (excluded from active time)
        self.limit_deferred = False
        self.rate_limited = False
        self.rate_limit_resets_at: float | None = None
        self.access_error: str | None = None  # the account can't use this model at all

    # ---- setup ----

    def setup_workspace(self) -> None:
        self.workspace.mkdir(parents=True)
        (self.workspace / "data").mkdir()
        shutil.copy(DEV_DB, self.workspace / "data" / "dev.duckdb")
        # The workspace copy of the config reflects any per-run overrides.
        config_text = re.sub(
            r"(?m)^ramp_rate_mw_per_min = [^#\n]*", f"ramp_rate_mw_per_min = {self.config.battery.ramp_rate_mw_per_min:g} ",
            CONFIG_PATH.read_text(),
        )
        (self.workspace / "config.toml").write_text(config_text)
        (self.workspace / "policy.py").write_text(STARTER_POLICY)
        dev_ref = self.reference["dev"]
        (self.workspace / "reference_dev.json").write_text(json.dumps({
            "pct_of_perfect": dev_ref["pct_of_perfect"],
            "net_revenue": dev_ref["net_revenue"],
            "episodes_pct_of_perfect": self.reference["dev_episodes"]["pct_of_perfect"],
        }))
        task = TASK_TEMPLATE.read_text().format(
            max_submissions=self.args.max_submissions,
            eval_timeout_min=round(self.args.eval_timeout / 60),
            max_hours=self.args.max_hours,
            max_cost_usd=self.args.max_cost,
            baseline_dev_pct=dev_ref["pct_of_perfect"],
            baseline_dev_episodes_pct=self.reference["dev_episodes"]["pct_of_perfect"],
            n_episodes=len(HOLDOUT_EPISODES),
            n_dev_episodes=len(DEV_EPISODES),
            ramp_rate=self.config.battery.ramp_rate_mw_per_min,
            degradation_cost=self.config.battery.degradation_cost,
        )
        (self.workspace / "TASK.md").write_text(task)
        self.submissions_dir.mkdir()

    # ---- accounting ----

    def call_cost(self, usage: dict) -> float:
        """Cost of one API call from its usage record."""
        raise NotImplementedError

    def cost_usd(self) -> float:
        return sum(self.call_cost(u) for u in self.api_calls.values())

    def token_totals(self) -> dict:
        keys = ["input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"]
        return {k: sum((u.get(k) or 0) for u in self.api_calls.values()) for k in keys}

    def active_seconds(self) -> float:
        running = time.time() - self.segment_start if self.segment_start is not None else 0.0
        return self.active_before + running

    def start_clock(self) -> None:
        if self.segment_start is None:
            self.segment_start = time.time()

    def stop_clock(self) -> None:
        if self.segment_start is not None:
            self.active_before += time.time() - self.segment_start
            self.segment_start = None

    def save_state(self) -> None:
        state = {
            "run_id": self.run_id,
            "args": vars(self.args),
            "session_id": self.session_id,
            "active_seconds": self.active_seconds(),
            "nudges": self.nudges,
            "finished_reason": self.finished_reason,
            "limit_reason": self.limit_reason,
            "agent_summary": self.agent_summary,
            "paused_until": self.rate_limit_resets_at if self.segment_start is None else None,
            "saved_at": time.time(),
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, default=str))
        tmp.replace(self.state_path)

    def status_text(self) -> str:
        remaining_s = self.args.max_hours * 3600 - self.active_seconds()
        used = len(self.submissions)
        return (
            f"Elapsed {self.active_seconds() / 60:.0f} min, {remaining_s / 60:.0f} min remaining. "
            f"Model usage ${self.cost_usd():.2f} of ${self.args.max_cost:.2f}. "
            f"Submissions used {used} of {self.args.max_submissions}."
            + (f" Current final score: {self.final_score_text()}." if used else "")
        )

    def final_submission(self) -> dict | None:
        valid = [s for s in self.submissions if s["result"].get("valid")]
        return valid[-1] if valid else None

    def final_score_text(self) -> str:
        final = self.final_submission()
        if final is None:
            return "none (no valid submission yet)"
        return f"{final['result']['pct_of_perfect']:.1f}% of perfect foresight (submission {final['n']})"

    # ---- harness tools (shared by the backends) ----

    def finish(self, summary: str) -> str:
        self.finished_reason = "agent_finished"
        self.agent_summary = summary
        self.log.event("finish", summary=summary)
        return f"Run finished. Final score: {self.final_score_text()}."

    async def submit(self, summary: str, rationale: str) -> str:
        if len(self.submissions) >= self.args.max_submissions:
            return f"No submissions left ({self.args.max_submissions} used). Final score: {self.final_score_text()}."
        if not (self.workspace / "policy.py").exists():
            return "There is no policy.py in the workspace. Nothing was submitted."
        n = len(self.submissions) + 1
        snap_dir = self.submissions_dir / f"{n:02d}"
        policy_dir = snap_dir / "policy"
        if snap_dir.exists():  # left by a submission whose scoring crashed before it was recorded
            shutil.rmtree(snap_dir)
        shutil.copytree(self.workspace, policy_dir, ignore=SNAPSHOT_IGNORE)
        size = sum(f.stat().st_size for f in policy_dir.rglob("*") if f.is_file())
        if size > MAX_SNAPSHOT_BYTES:
            shutil.rmtree(snap_dir)
            return f"The workspace (excluding data/ and dev_results/) is {size / 1e6:.0f} MB; the limit is 200 MB. Nothing was submitted."

        self.log.event(
            "submission_start", n=n, summary=summary, rationale=rationale, snapshot_bytes=size,
            active_s=self.active_seconds(), cost_usd=self.cost_usd(), tokens=self.token_totals(),
        )
        (snap_dir / "description.json").write_text(json.dumps({"summary": summary, "rationale": rationale}, indent=2))
        # Scoring time is the evaluator's, not the agent's: stop the active clock (the time
        # budget) while it runs. The watchdog also won't stop the run mid-evaluation.
        self.evaluating = True
        was_running = self.segment_start is not None
        self.stop_clock()
        try:
            record = await self.evaluate_submission(n, summary, rationale)
        except Exception as e:  # a harness fault, not the policy's: don't count it
            self.log.event("submission_error", n=n, error=f"{type(e).__name__}: {e}")
            shutil.rmtree(snap_dir, ignore_errors=True)
            return (
                f"The harness failed while scoring this submission ({type(e).__name__}: {e}). "
                f"That's not a fault in your policy, and it doesn't count as a submission; you can submit again. "
                + self.status_text()
            )
        finally:
            self.evaluating = False
            if was_running:
                self.start_clock()
        return self.feedback(record)

    async def evaluate_submission(self, n: int, summary: str, rationale: str) -> dict:
        snap_dir = self.submissions_dir / f"{n:02d}"
        t0 = time.time()
        result = await asyncio.to_thread(
            evaluate, snap_dir / "policy", self.config, HOLDOUT_EPISODES,
            sandboxed=True, timeout_s=self.args.eval_timeout, out_dir=snap_dir / "eval",
        )
        eval_s = time.time() - t0
        self.submission_eval_seconds += eval_s
        record = {"n": n, "summary": summary, "rationale": rationale, "elapsed_s": self.log.elapsed(),
                  "active_s": self.active_seconds(), "eval_seconds": eval_s, "result": result}
        (snap_dir / "result.json").write_text(json.dumps(record, indent=2, default=str))
        self.submissions.append(record)
        self.log.event(
            "submission_result", n=n, eval_seconds=eval_s, valid=result.get("valid"),
            pct_of_perfect=result.get("pct_of_perfect"), net_revenue=result.get("net_revenue"),
            rejected_bids=result.get("rejected_bids"),
            bid_nonconformance_intervals=result.get("bid_nonconformance_intervals"),
            nonconformance_intervals=result.get("nonconformance_intervals"), error=result.get("error"),
            per_region=result.get("per_region_pct"), per_episode=result.get("per_episode_pct"),
        )
        self.save_state()
        return record

    def feedback(self, record: dict) -> str:
        result, baseline = record["result"], self.reference["holdout_episodes"]["pct_of_perfect"]
        lines = [f"Submission {record['n']} of {self.args.max_submissions} (holdout episodes, all regions):"]
        if "pct_of_perfect" in result:
            lines.append(
                f"net revenue ${result['net_revenue']:,.0f} = {result['pct_of_perfect']:.1f}% of perfect foresight; "
                f"valid: {result['valid']}"
            )
            if not result["valid"]:
                lines.append(
                    f"INVALID: {result['rejected_bids']} rejected bids, "
                    f"{result['nonconformance_intervals']} non-conforming intervals, "
                    f"{result.get('bid_nonconformance_intervals', 0)} bids with the wrong ramp rate. "
                    "Invalid runs earn no credit."
                )
                reasons = {j["first_rejection"][0][:300] for j in result["jobs"] if j.get("first_rejection")}
                reasons |= {j["first_bid_nonconformance"][0][:300] for j in result["jobs"] if j.get("first_bid_nonconformance")}
                lines += [f"  example rejection: {x}" for x in sorted(reasons)[:2]]
        else:
            lines.append(f"FAILED: {result['error'][:1000]}")
        lines.append(f"Baseline (TWAP) on the same holdout episodes: {baseline:.1f}% of perfect foresight.")
        others = {k: v for k, v in baseline_pcts(self.reference, "holdout_episodes").items() if k != REFERENCE_NAME}
        if others:
            lines.append("Other baselines on the same episodes: "
                         + ", ".join(f"{k} {v:.1f}%" for k, v in others.items()) + ".")
        lines.append(f"Evaluation took {record['eval_seconds'] / 60:.1f} min. {self.status_text()}")
        return "\n".join(lines)

    def check_limits(self) -> str | None:
        if self.active_seconds() > self.args.max_hours * 3600:
            return "active_time"  # the time budget excludes usage-limit pauses and submission scoring
        if self.cost_usd() > self.args.max_cost:
            return "cost"
        return None

    async def watchdog(self) -> None:
        """Saves state every 10 s and stops the agent at the time or cost limit,
        but never mid-evaluation: an in-flight submission finishes and is
        recorded first."""
        while True:
            await asyncio.sleep(10)
            self.save_state()
            reason = self.check_limits()
            if not reason or self.finished_reason or self.limit_reason:
                continue
            if self.evaluating:
                if not self.limit_deferred:
                    self.limit_deferred = True
                    self.log.event("limit_deferred", reason=reason, detail="waiting for the submission evaluation to finish")
                continue
            self.reach_limit(reason)
            await self.interrupt()

    def reach_limit(self, reason: str) -> None:
        self.limit_reason = reason
        self.log.event("limit_reached", reason=reason, cost_usd=self.cost_usd(), active_s=self.active_seconds())
        self.save_state()

    async def interrupt(self) -> None:
        """Stop the agent's current turn (a limit was reached)."""
        raise NotImplementedError

    def restore_conversation(self, state: dict) -> None:
        """Reload the backend's conversation when resuming a run."""
        raise NotImplementedError

    # ---- main loop ----

    async def run(self, resume: bool = False) -> None:
        if resume:
            prompt = await self.restore()
            if prompt is None:
                return
        else:
            await self.ensure_reference()
            self.setup_workspace()
            self.reference_copy.write_text(json.dumps(self.reference, indent=2, default=str))
            self.log.event("run_start", run_id=self.run_id, model=self.model, args=vars(self.args),
                           config_fingerprint=config_fingerprint(self.config))
            prompt = "Read TASK.md in your working directory, then complete the task it describes."
        self.save_state()
        watchdog = asyncio.create_task(self.watchdog())
        failures = 0
        try:
            while True:
                try:
                    self.start_clock()
                    outcome = await self.session(prompt)
                    failures = 0
                except Exception as e:  # CLI crash, dropped connection, ...
                    self.stop_clock()
                    failures += 1
                    self.log.event("session_error", error=f"{type(e).__name__}: {e}", consecutive=failures)
                    if failures > self.args.max_retries:
                        self.save_state()
                        raise SystemExit(f"Giving up after {failures} consecutive failures; resume with --resume {self.run_id}")
                    if self.rate_limited:
                        outcome = "rate_limited"
                    else:
                        await self.pause("error", time.time() + min(60 * 2 ** failures, 3600))
                        prompt = ("The harness lost its connection to you and has reconnected. Continue the task from "
                                  f"where you left off. {self.status_text()}")
                        continue
                self.stop_clock()
                if outcome in ("finished", "limit"):
                    break
                if outcome == "rate_limited":
                    until = self.rate_limit_resets_at or time.time() + 15 * 60
                    if until - time.time() > self.args.max_pause_hours * 3600:
                        self.log.event("stopped_for_long_limit", resets_at=until)
                        self.save_state()
                        raise SystemExit(
                            f"Usage limit resets at {datetime.fromtimestamp(until):%Y-%m-%d %H:%M}, beyond "
                            f"--max-pause-hours. Resume then with --resume {self.run_id}"
                        )
                    await self.pause("usage_limit", until + 60)
                    prompt = ("Your session was paused because the model usage limit was reached. It has now reset "
                              "and paused time didn't count against your budget. Continue the task from where you "
                              f"left off. {self.status_text()}")
        finally:
            watchdog.cancel()
            self.stop_clock()
            self.save_state()
        await self.final_evaluation()
        self.write_summary()

    async def ensure_reference(self) -> None:
        """Compute baseline scores for this configuration if they aren't cached yet
        (e.g. a new --ramp-rate), or a baseline whose fitted parameters changed.
        Takes several minutes when it has work to do; happens before the agent starts."""
        if self.reference is None:
            print(f"Computing baseline scores for config {config_fingerprint(self.config)} "
                  "(first run with this configuration; several minutes)...", flush=True)
        self.reference = await asyncio.to_thread(update_reference, self.config)

    async def pause(self, reason: str, until: float) -> None:
        self.rate_limit_resets_at = until
        self.log.event("pause", reason=reason, until=until, until_local=f"{datetime.fromtimestamp(until):%Y-%m-%d %H:%M:%S}")
        self.save_state()
        print(f"Paused ({reason}) until {datetime.fromtimestamp(until):%H:%M:%S}", flush=True)
        while time.time() < until:
            await asyncio.sleep(min(30, max(1, until - time.time())))
        self.rate_limit_resets_at = None
        self.rate_limited = False
        self.log.event("resume", reason=reason)
        self.save_state()

    async def restore(self) -> str | None:
        """Rebuild in-memory state from state.json, the event log and the
        submissions directory. Returns the prompt to resume with, or None if
        the run had already ended."""
        state = json.loads(self.state_path.read_text())
        self.restore_conversation(state)
        self.active_before = state["active_seconds"]
        self.nudges = state["nudges"]
        self.finished_reason = state["finished_reason"]
        self.limit_reason = state["limit_reason"]
        self.agent_summary = state["agent_summary"]
        sdk_cost = 0.0
        for line in (self.dir / "logs" / "events.jsonl").open():
            e = json.loads(line)
            if e["event"] == "api_call":
                self.api_calls[e["message_id"]] = e["usage"]
            elif e["event"] == "assistant_message" and e.get("usage"):
                self.api_calls.setdefault(e["message_id"], e["usage"])
            elif e["event"] == "tool_end":
                stats = self.tool_stats.setdefault(e["tool"], {"calls": 0, "seconds": 0.0, "failures": 0})
                stats["calls"] += 1
                stats["seconds"] += e["duration_s"]
                stats["failures"] += int(e["failed"])
            elif e["event"] == "result":
                sdk_cost = e.get("sdk_total_cost_usd") or sdk_cost
        self.sdk_cost_usd = sdk_cost

        notes = []
        for snap_dir in sorted(p for p in self.submissions_dir.iterdir() if p.is_dir()):
            n = int(snap_dir.name)
            if not (snap_dir / "description.json").exists():  # stopped while copying the snapshot
                shutil.rmtree(snap_dir)
                continue
            description = json.loads((snap_dir / "description.json").read_text())
            if (snap_dir / "result.json").exists():
                record = json.loads((snap_dir / "result.json").read_text())
                if "result" not in record:  # older runs saved the bare evaluation result
                    record = {"n": n, **description, "elapsed_s": None, "eval_seconds": record.get("eval_seconds"), "result": record}
                self.submissions.append(record)
                self.submission_eval_seconds += record.get("eval_seconds") or 0
            else:  # the harness stopped mid-evaluation: scoring is deterministic, so redo it
                self.log.event("submission_reevaluate", n=n)
                record = await self.evaluate_submission(n, description["summary"], description["rationale"])
                notes.append(self.feedback(record))
        self.log.event("run_resumed", session_id=self.session_id, active_s=self.active_before,
                       submissions=len(self.submissions), cost_usd=self.cost_usd())
        if self.finished_reason or self.limit_reason:
            print(f"Run {self.run_id} already ended ({self.finished_reason or self.limit_reason}).")
            await self.final_evaluation()
            self.write_summary()
            return None
        return ("The harness running this task was restarted, so your previous turn may have been cut off. "
                "Continue the task from where you left off. "
                + (" Your submission was scored while you were away:\n" + "\n".join(notes) + "\n" if notes else "")
                + self.status_text())

    async def final_evaluation(self) -> dict | None:
        """Score the final submission on the whole holdout, one job per month: the headline
        result. Runs after the agent is done, outside its budget; skipped if
        already done (e.g. on --resume of an ended run)."""
        final = self.final_submission()
        if final is None:
            return None
        snap_dir = self.submissions_dir / f"{final['n']:02d}"
        path = snap_dir / "result_full_holdout.json"
        if path.exists():
            return json.loads(path.read_text())
        self.log.event("final_evaluation_start", n=final["n"])
        print(f"Scoring submission {final['n']} on the full holdout...", flush=True)
        t0 = time.time()
        result = await asyncio.to_thread(
            evaluate, snap_dir / "policy", self.config, HOLDOUT_MONTHS,
            sandboxed=True, timeout_s=self.args.eval_timeout * FULL_MONTH_TIMEOUT_FACTOR,
            out_dir=snap_dir / "eval_full",
        )
        result["n"] = final["n"]
        path.write_text(json.dumps(result, indent=2, default=str))
        self.log.event(
            "final_evaluation_result", n=final["n"], eval_seconds=time.time() - t0, valid=result.get("valid"),
            pct_of_perfect=result.get("pct_of_perfect"), net_revenue=result.get("net_revenue"),
            per_region=result.get("per_region_pct"), error=result.get("error"),
        )
        return result

    def write_summary(self) -> None:
        final = self.final_submission()
        full_path = self.submissions_dir / f"{final['n']:02d}" / "result_full_holdout.json" if final else None
        full = json.loads(full_path.read_text()) if full_path and full_path.exists() else None
        summary = {
            "run_id": self.run_id,
            "model": self.model,
            "effort": self.args.effort,
            "billing": self.billing,  # "openrouter": the cost below is real, not API-equivalent
            "ended_by": self.finished_reason or self.limit_reason,
            "wall_seconds": self.log.elapsed(),
            "active_seconds": self.active_seconds(),  # excludes usage-limit pauses and submission scoring
            "submission_eval_seconds": self.submission_eval_seconds,
            # subscription: API-equivalent cost of the tokens used; openrouter: what OpenRouter billed
            "cost_usd": self.cost_usd(),
            "api_calls": len(self.api_calls),
            "tokens": self.token_totals(),
            "tools": self.tool_stats,
            "submissions": [
                {
                    "n": s["n"], "summary": s["summary"], "elapsed_s": s["elapsed_s"],
                    "active_s": s.get("active_s"), "eval_seconds": s["eval_seconds"],
                    "valid": s["result"].get("valid"), "pct_of_perfect": s["result"].get("pct_of_perfect"),
                    "net_revenue": s["result"].get("net_revenue"),
                }
                for s in self.submissions
            ],
            "final_submission": final["n"] if final else None,
            "final_pct_of_perfect": final["result"]["pct_of_perfect"] if final else None,
            "baseline_holdout_episodes_pct_of_perfect": self.reference["holdout_episodes"]["pct_of_perfect"],
            "final_full_holdout_pct_of_perfect": full.get("pct_of_perfect") if full else None,
            "final_full_holdout_valid": full.get("valid") if full else None,
            "baseline_full_holdout_pct_of_perfect": self.reference["holdout_full"]["pct_of_perfect"],
            "baselines": {s: baseline_pcts(self.reference, s) for s in ("holdout_episodes", "holdout_full")},
            "agent_summary": self.agent_summary,
        }
        if self.billing == "subscription":
            summary["sdk_reported_cost_usd"] = self.sdk_cost_usd  # cross-check; includes Claude Code's side calls
            summary["sdk_model_usage"] = getattr(self, "sdk_model_usage", None)
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        self.log.event("run_end", **{k: summary[k] for k in [
            "ended_by", "wall_seconds", "active_seconds", "cost_usd", "final_pct_of_perfect",
            "final_full_holdout_pct_of_perfect"]})
        print(json.dumps(summary, indent=2, default=str))


class ClaudeCodeRun(Run):
    """Claude models in Claude Code, through the Claude Agent SDK, on the subscription."""

    def setup_workspace(self) -> None:
        super().setup_workspace()
        (self.dir / "claude_config").mkdir()

    def call_cost(self, usage: dict) -> float:
        return usage_cost(self.model, usage)

    async def interrupt(self) -> None:
        if self.client is not None:
            await self.client.interrupt()

    def restore_conversation(self, state: dict) -> None:
        self.session_id = state["session_id"]

    # ---- tools ----

    def make_tools(self):
        @tool(
            "submit_policy",
            SUBMIT_POLICY_DESCRIPTION,
            {
                "type": "object",
                "properties": {"summary": {"type": "string"}, "rationale": {"type": "string"}},
                "required": ["summary", "rationale"],
            },
        )
        async def submit_policy(args):
            text = await self.submit(args.get("summary", ""), args.get("rationale", ""))
            return {"content": [{"type": "text", "text": text}]}

        @tool("status", STATUS_DESCRIPTION, {})
        async def status(args):
            return {"content": [{"type": "text", "text": self.status_text()}]}

        @tool("finish", FINISH_DESCRIPTION, {"summary": str})
        async def finish(args):
            return {"content": [{"type": "text", "text": self.finish(args.get("summary", ""))}]}

        return create_sdk_mcp_server(name="harness", tools=[submit_policy, status, finish])

    # ---- hooks ----

    async def pre_tool(self, hook_input, tool_use_id, context):
        self.tool_starts[tool_use_id] = time.time()
        self.log.event(
            "tool_start", tool=hook_input["tool_name"], tool_use_id=tool_use_id,
            input=_truncate(hook_input.get("tool_input"), 2000),
        )
        return {}

    async def post_tool(self, hook_input, tool_use_id, context):
        failed = hook_input["hook_event_name"] == "PostToolUseFailure"
        duration = time.time() - self.tool_starts.pop(tool_use_id, time.time())
        name = hook_input["tool_name"]
        response = hook_input.get("tool_response", hook_input.get("error"))
        stats = self.tool_stats.setdefault(name, {"calls": 0, "seconds": 0.0, "failures": 0})
        stats["calls"] += 1
        stats["seconds"] += duration
        stats["failures"] += int(failed)
        self.log.event(
            "tool_end", tool=name, tool_use_id=tool_use_id, duration_s=round(duration, 3), failed=failed,
            response_chars=len(json.dumps(response, default=str)), response=_truncate(response, 2000),
        )
        return {}

    # ---- message handling ----

    def handle(self, message) -> None:
        now = time.time()
        if isinstance(message, StreamEvent):
            self.handle_stream_event(message.event, now)
            return  # far too many to keep in the transcript
        self.log.transcript(message)
        session_id = getattr(message, "session_id", None) or (getattr(message, "data", None) or {}).get("session_id")
        if session_id and session_id != self.session_id:
            self.session_id = session_id
            self.log.event("session", session_id=session_id)
            self.save_state()
        if isinstance(message, AssistantMessage):
            if message.error == "rate_limit":
                self.rate_limited = True
            first_block = message.message_id not in self.api_calls
            if message.usage and first_block:
                # Placeholder: an assistant message's usage is captured at the start of the
                # response. handle_stream_event replaces it with the final usage.
                self.api_calls[message.message_id] = message.usage
            blocks = []
            for b in message.content:
                if isinstance(b, TextBlock):
                    blocks.append({"type": "text", "text": b.text})
                elif isinstance(b, ThinkingBlock):
                    blocks.append({"type": "thinking", "text": b.thinking})
                elif isinstance(b, ToolUseBlock):
                    blocks.append({"type": "tool_use", "name": b.name, "input": _truncate(b.input, 500)})
            self.log.event(
                "assistant_message", message_id=message.message_id, model=message.model,
                new_api_call=first_block, latency_s=round(now - self.last_message_at, 3),
                stop_reason=message.stop_reason, usage=message.usage,
                call_cost_usd=self.call_cost(message.usage or {}),
                cumulative_cost_usd=self.cost_usd(), blocks=blocks, error=message.error,
            )
        elif isinstance(message, ResultMessage):
            if message.api_error_status == 429:
                self.rate_limited = True
            self.result_messages += 1
            self.sdk_cost_usd = message.total_cost_usd or self.sdk_cost_usd
            self.sdk_model_usage = message.model_usage or getattr(self, "sdk_model_usage", None)
            self.log.event(
                "result", subtype=message.subtype, is_error=message.is_error, num_turns=message.num_turns,
                duration_ms=message.duration_ms, duration_api_ms=message.duration_api_ms,
                sdk_total_cost_usd=message.total_cost_usd, usage=message.usage,
                model_usage=message.model_usage, stop_reason=message.stop_reason, session_id=message.session_id,
            )
        elif isinstance(message, UserMessage):
            pass  # tool results; timing comes from the hooks
        else:
            # Progress ticks (e.g. thinking-token estimates) don't reset the latency clock.
            name = type(message).__name__
            if name == "RateLimitEvent":
                info = getattr(message, "rate_limit_info", None)
                raw = getattr(info, "raw", None) or (info if isinstance(info, dict) else {})
                windows = raw.get("unifiedWindows", {}) if isinstance(raw, dict) else {}
                status = getattr(info, "status", None)
                error_code = raw.get("errorCode") if isinstance(raw, dict) else None
                if status == "rejected" and error_code == "credits_required":
                    # Not a limit that resets: this model needs paid usage credits on this account.
                    self.access_error = f"{self.model} requires usage credits on this subscription (credits_required)"
                elif status == "rejected":
                    self.rate_limited = True
                    self.rate_limit_resets_at = getattr(info, "resets_at", None)
                self.log.event(
                    "rate_limit", status=status, error_code=error_code, limit_type=getattr(info, "rate_limit_type", None),
                    resets_at=getattr(info, "resets_at", None),
                    utilization={k: v.get("utilization") for k, v in windows.items()},
                )
            elif getattr(message, "subtype", None) != "thinking_tokens":
                self.log.event("other_message", type=name, subtype=getattr(message, "subtype", None))
            return
        self.last_message_at = now

    def handle_stream_event(self, event: dict, now: float) -> None:
        """Final per-call usage and timing from the raw API stream: message_start
        opens a call, message_delta carries its final output tokens."""
        kind = event.get("type")
        if kind == "message_start":
            msg = event.get("message", {})
            self.streaming = {"id": msg.get("id"), "model": msg.get("model"), "usage": dict(msg.get("usage") or {}),
                              "started": now, "wait_s": now - self.last_message_at}
        elif kind == "message_delta" and self.streaming:
            call = self.streaming
            usage = {**call["usage"], **{k: v for k, v in (event.get("usage") or {}).items() if v is not None}}
            self.api_calls[call["id"]] = usage
            self.streaming = None
            self.log.event(
                "api_call", message_id=call["id"], model=call["model"], usage=usage,
                stop_reason=(event.get("delta") or {}).get("stop_reason"),
                wait_s=round(call["wait_s"], 3), duration_s=round(now - call["started"], 3),
                call_cost_usd=self.call_cost(usage), cumulative_cost_usd=self.cost_usd(),
            )

    def options(self) -> ClaudeAgentOptions:
        """Agent options; resumes the previous Claude Code session if there is one."""
        cli = write_agent_cli_wrapper(self.dir / "claude_cli.sh", self.workspace, self.dir / "claude_config")
        timeout_ms = str(int(self.args.eval_timeout * 1000 + 600_000))
        return ClaudeAgentOptions(
            model=self.model,
            effort=self.args.effort,
            thinking={"type": "adaptive", "display": "summarized"},
            cwd=str(self.workspace),
            cli_path=str(cli),
            setting_sources=[],  # ignore the user's own settings, CLAUDE.md and memory
            system_prompt={"type": "preset", "preset": "claude_code"},
            tools=AGENT_TOOLS,
            mcp_servers={"harness": self.make_tools()},
            # dontAsk denies anything not pre-approved here. In particular, Claude Code's
            # sandbox proxy treats an unlisted network domain as a permission request,
            # which bypassPermissions would approve; under dontAsk it's refused, so bash
            # can't fetch the real holdout prices from the internet.
            allowed_tools=AGENT_TOOLS + ["mcp__harness__submit_policy", "mcp__harness__status", "mcp__harness__finish"],
            permission_mode="dontAsk",
            sandbox={
                "enabled": True,
                "autoAllowBashIfSandboxed": True,
                "allowUnsandboxedCommands": False,
                "failIfUnavailable": True,
                "network": {"allowedDomains": []},
            },
            env={
                "BASH_DEFAULT_TIMEOUT_MS": str(30 * 60 * 1000),
                "BASH_MAX_TIMEOUT_MS": str(90 * 60 * 1000),
                "MCP_TOOL_TIMEOUT": timeout_ms,
            },
            hooks={
                "PreToolUse": [HookMatcher(hooks=[self.pre_tool])],
                "PostToolUse": [HookMatcher(hooks=[self.post_tool])],
                "PostToolUseFailure": [HookMatcher(hooks=[self.post_tool])],
            },
            resume=self.session_id,
            include_partial_messages=True,  # stream events carry each API call's final usage
            max_budget_usd=self.args.max_cost * 1.1,  # backstop; the watchdog normally stops first
            stderr=lambda line: self.log.event("cli_stderr", line=line[:1000]),
        )

    async def session(self, prompt: str) -> str:
        """One connection to Claude Code. Returns "finished", "limit" or "rate_limited"."""
        async with ClaudeSDKClient(self.options()) as client:
            self.client = client
            try:
                while True:
                    self.rate_limited = False
                    self.log.event("prompt", text=prompt)
                    await client.query(prompt)
                    async for message in client.receive_response():
                        self.handle(message)
                    if self.access_error:
                        self.finished_reason = "no_model_access"
                        self.log.event("no_model_access", detail=self.access_error)
                        print(f"Stopping: {self.access_error}.", flush=True)
                        return "finished"
                    if self.finished_reason:
                        return "finished"
                    if self.limit_reason:
                        return "limit"
                    if self.rate_limited:
                        return "rate_limited"
                    if self.nudges >= self.args.max_nudges:
                        self.finished_reason = "max_nudges"
                        return "finished"
                    self.nudges += 1
                    prompt = (
                        "You ended your turn without calling the finish tool. "
                        f"{self.status_text()} Keep working on the task, or call finish if you are done."
                    )
            finally:
                self.client = None


def load_or_rebuild_state(run_dir: Path) -> dict:
    """state.json, or for runs that predate it, the equivalent rebuilt from the event log."""
    path = run_dir / "state.json"
    if path.exists():
        return json.loads(path.read_text())
    events = [json.loads(line) for line in (run_dir / "logs" / "events.jsonl").open()]
    start = next(e for e in events if e["event"] == "run_start")
    sessions = [e["session_id"] for e in events if e.get("session_id")]
    ended = [e for e in events if e["event"] in ("limit_reached", "finish")]
    state = {
        "run_id": run_dir.name, "args": start["args"], "session_id": sessions[-1] if sessions else None,
        "active_seconds": events[-1]["elapsed_s"], "nudges": sum(e["event"] == "prompt" for e in events) - 1,
        "finished_reason": "agent_finished" if any(e["event"] == "finish" for e in ended) else None,
        "limit_reason": next((e["reason"] for e in ended if e["event"] == "limit_reached"), None),
        "agent_summary": next((e.get("summary") for e in ended if e["event"] == "finish"), None),
    }
    path.write_text(json.dumps(state, indent=2))
    return state


def _truncate(value, limit: int):
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more chars]"


def setup_subscription_token() -> None:
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") and TOKEN_FILE.exists():
        os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = TOKEN_FILE.read_text().strip()
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        raise SystemExit(
            "No subscription token. Run `claude setup-token`, then export it as CLAUDE_CODE_OAUTH_TOKEN "
            f"or save it to {TOKEN_FILE}."
        )
    if os.environ.pop("ANTHROPIC_API_KEY", None):
        print("Note: ignoring ANTHROPIC_API_KEY so the run uses the subscription, not API billing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="claude-sonnet-5-5",
                        help=f"a Claude model on the subscription ({', '.join(sorted(PRICES))}) or an OpenRouter slug")
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"],
                        help="thinking effort (Claude) or reasoning effort (OpenRouter)")
    parser.add_argument("--max-hours", type=float, default=2.0)
    parser.add_argument("--max-cost", type=float, default=20.0,
                        help="USD: API-equivalent for Claude runs, real OpenRouter spend for OpenRouter runs")
    parser.add_argument("--max-submissions", type=int, default=5)
    parser.add_argument("--ramp-rate", type=float, help="the battery's ramp rate in MW/min (default: from the config)")
    parser.add_argument("--eval-timeout", type=float, default=900,
                        help="seconds per (region, episode) job when scoring a submission")
    parser.add_argument("--max-nudges", type=int, default=5, help="times to prompt the agent to continue")
    parser.add_argument("--max-pause-hours", type=float, default=12.0,
                        help="longest usage-limit wait to sit through; longer limits exit for a later --resume")
    parser.add_argument("--max-retries", type=int, default=5, help="consecutive API or CLI failures before giving up")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", metavar="RUN_ID", help="continue an interrupted run")
    args = parser.parse_args()
    if args.resume:
        state = load_or_rebuild_state(RUNS_DIR / args.resume)
        defaults = vars(parser.parse_args([]))  # for options added after the run started
        args = argparse.Namespace(**{**defaults, **state["args"], "run_id": args.resume, "resume": args.resume})

    if is_openrouter(args.model):
        from harness.openrouter_agent import OpenRouterRun, check_model, load_api_key

        load_api_key()
        if not args.resume:
            asyncio.run(check_model(args.model, args.max_cost))  # before a run directory exists
        run = OpenRouterRun(args)
    else:
        if args.model not in PRICES:
            raise SystemExit(f"Unknown Claude model {args.model!r}: use one of {', '.join(sorted(PRICES))}, "
                             "or an OpenRouter slug like openai/gpt-5.1")
        setup_subscription_token()
        run = ClaudeCodeRun(args)
    asyncio.run(run.run(resume=bool(args.resume)))
