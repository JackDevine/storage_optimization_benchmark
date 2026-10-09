# Storage Optimization Benchmark

Benchmark data and models for storage optimization against AEMO NEM market data.

## Setup

```
uv sync
```

## Pulling data

```
uv run scripts/pull_dispatch_price.py          # DISPATCHPRICE (actual, 5-min) via NEMOSIS
uv run scripts/pull_dispatch_regionsum.py      # DISPATCHREGIONSUM (actual demand/UIGF, 5-min) via NEMOSIS
uv run scripts/pull_forecast_prices.py         # P5MIN_REGIONSOLUTION, PREDISPATCHPRICE, PREDISPATCHREGIONSUM
```

All three load straight into `data/benchmark.duckdb` (one table per AEMO table, e.g. `dispatchprice`,
`dispatchregionsum`, `p5min_regionsolution`, `predispatchprice`, `predispatchregionsum`). AEMO's raw downloads
are kept under `data/raw_cache/`. Neither directory is committed — see `.gitignore`.

NEMOSIS doesn't support the P5MIN/PREDISPATCH forecast tables (the maintainer points to a sister package,
NEMSEER, instead — see https://github.com/UNSW-CEEM/NEMOSIS/issues/32). We're using our own small downloader
(`scripts/aemo_mmsdm.py`) for those rather than taking on NEMSEER's GPLv3 license.

## Data caching — know this before debugging a "why is this stale/missing" issue

`data/raw_cache/` is **not uniform** — the file formats present depend on which tool wrote them, and the cache
only applies to the *download* step, not to what ends up in DuckDB:

- **Our downloader** (`aemo_mmsdm.py`, used for the forecast tables) keeps the original `.zip` plus the
  extracted `.CSV`. No parquet — it re-parses the CSV with pandas on every run.
- **NEMOSIS** (used for `dispatchprice`/`dispatchregionsum`) keeps `.CSV` + `.parquet` by default
  (`keep_csv=True`); this version of NEMOSIS has no `keep_zip` option, so the zip is always deleted after
  extraction. The `.parquet` file is what NEMOSIS actually checks to decide whether to skip re-downloading a
  month — the `.CSV` sitting next to it is an unused byproduct, not load-bearing.
- Both tools skip re-hitting AEMO's servers if the relevant month's file already exists locally — this is a
  real, useful cache (confirmed empirically: re-running a pull script after the first time only attempts the
  network for months not yet downloaded).
- **The DuckDB write is not cached.** Every pull script does `CREATE OR REPLACE TABLE` — it always re-parses
  whatever's in `raw_cache/` and rebuilds the table from scratch, even if nothing changed. Extending a date
  range re-downloads only the new months but still reprocesses and rewrites the *entire* table, not just the
  new rows. There's no incremental append and no "already loaded, skip" check at the DuckDB layer.

If you add a new table/puller, don't assume raw_cache's file mix will look the same as an existing one — it's
an artifact of which library wrote it and what's been manually cleaned up since, not a deliberate convention.

## Dispatch benchmark

```
uv run scripts/run_benchmark.py [--config config/default.toml] [--regions SA1 ...] [--policies milp_bid_curve ...] [--jobs 5]
```

`--jobs` simulates regions in parallel. `pct_of_perfect` uses the stored `perfect_foresight` result for the same
config if that policy isn't part of the run.

Simulates a battery in 5-minute steps and writes `results_intervals` / `results_summary` to DuckDB. Rows are keyed
by (config, policy, region), where config is the config file's name.

- `config/default.toml`: battery spec (1 MW / 2 MWh; price-taker, energy only, no FCAS), the end-of-horizon SOC
  condition, market rules (MFP/MPC, MLFs, daily price lock), the daily bid price ladder and the MIP gap. The terminal condition is a soft
  target: `soc_shortfall_penalty` $/MWh for every MWh the final SOC falls below `soc_target_frac`. The default
  penalty is **0**, meaning no terminal condition at all, so the optimiser drains the battery at the end of each
  horizon.
- `bess/sim.py`: the simulator. At decision time `tau` a policy submits a **bidirectional unit (BDU) energy bid**
  for the interval ending `tau+5min`. New policies implement `decide(tau, state, view) -> BDUBid`. Each interval
  the simulator:
  1. Validates the bid. If `lock_daily_prices` is on, it also checks the band prices haven't changed since the
     trading day's first accepted bid. A rejected bid is dispatched at 0 MW.
  2. Clears the bid as a price-taker at the **actual** `dispatchprice.RRP`.
  3. Applies the unit's ramp rate. A bid stating any other ramp rate is still dispatched, but flagged as
     `bid_nonconformance`, which makes the run invalid.
  4. Moves the battery in a straight line from its start-of-interval MW to the end-of-interval target (AEMO's
     dispatch trajectory, `market.linear_dispatch_trajectory`). It settles the energy under that line,
     discharge and charge each with their own efficiency, and clips to power and SOC limits along the way.
     The MILP policies plan with the same trajectory and ramp limits, plus a small SOC margin
     (`bess.policies.dynamics`).
  Any clipping counts as **non-conformance**, since AEMO doesn't track SOC and managing it through MaxAvail is the
  participant's job. A run with any rejected bid or non-conformance keeps its score but is marked `valid = false`,
  and `run_benchmark.py` prints a banner listing invalid runs.
- `bess/bid.py`: the BDU bid, following AEMO's IESS format.
  - Each side (LOAD and GEN) has 10 price bands fixed for the trading day.
  - Each interval also sets MW per band, MaxAvail and ramp rates. The ramp rates must equal the unit's configured
    rate (`battery.ramp_rate_mw_per_min`, or `run_agent.py --ramp-rate`); any other value gets the bid rejected.
  - Validation follows AEMO's rules: prices strictly increase within each side, prices stay between the floor and
    cap scaled by MLF, and the effective bands pass the convexity check, so a bid can never charge and discharge
    at once.
  - `setpoint_bid(mw)` builds a bid that is dispatched to `mw` whatever the price. Discharge goes in gen band 1
    at the market floor, and charge goes in load band 10 at the cap.
  - Fixed load, PASA availability and the optional energy limit aren't modelled.
- `bess/data.py`: `ForecastView` only shows forecast runs published by `tau`. A P5MIN run counts as available
  from its `RUN_DATETIME` (one interval conservative) and a predispatch run from its `LASTCHANGED`.
- `bess/lp.py`: the arbitrage MILP (HiGHS). A binary per step makes charge and discharge mutually exclusive.
  Without it, at negative prices the solver would charge and discharge at once to burn energy through
  round-trip losses.
- `bess/policies.py`:
  - `milp_forecast` (baseline): every 5 minutes it joins the latest P5MIN run (5-min steps) to the latest
    predispatch run (30-min steps), solves the MILP, and submits the first interval's MW as a `setpoint_bid`.
  - `milp_bid_curve`: submits a real price-band bid.
    - The day's 20 band prices (10 load and 10 gen, interleaved, from `[bidding]` in the config) are the
      scenarios for the next interval's price.
    - For each scenario it re-solves the MILP with the first interval's price set to the band price, keeping
      the forecast for the rest of the horizon. The result is the first-interval MW at each price.
    - The optimal MW only rises with price, so it solves the two ends of the price range and splits the range
      in half only where the answer changes. That takes about 8 solves per interval instead of 20.
    - Gen band `i` holds the extra discharge between gen prices `i-1` and `i`. Load band `i` holds the extra
      charge between load prices `i+1` and `i`.
    - Because the response never falls as price rises, the bid always passes AEMO's convexity check. Between
      band prices it rounds towards doing less, so it never dispatches more than the MILP wanted at that price.
  - `perfect_foresight`: the same MILP solved once over actual prices. It's the ceiling that
    `pct_of_perfect` is measured against.

## Agent experiments

```
claude setup-token        # once; save the token to ~/.config/storage-benchmark/claude_oauth_token
uv run python -m harness.evaluator reference      # once; caches baseline dev/holdout scores
uv run python -m harness.run_agent --model claude-sonnet-5-5 --max-hours 2 --max-cost 20
uv run python -m harness.dashboard                 # live view at http://127.0.0.1:8050
```

A model is asked to write a bidding policy. `harness/run_agent.py` drives Claude Code through the Claude Agent SDK
on your Claude subscription. Costs are reported as what the tokens would cost at standard API rates.

- **Split** (`harness/splits.py`): the agent develops on June–July 2026 (`data/dev.duckdb`, exported from the main
  DB with only what was published by the end of July). It's scored on August 2026, all five regions.
  *Work in progress:* this split will change once much more data is downloaded. That also needs old prices adjusted
  to the current market price cap without distorting them.
- **Episodes:** submissions are scored on a seeded sample of 3-day episodes (`sample_episodes`, one per week of
  the holdout window), each starting at 04:00 with the battery at 50%. All region × episode jobs run in parallel,
  so a submission takes about one episode's simulated time, not a month's.
  - After the run ends, the final valid submission is also scored on the **whole holdout month**
    (`summary.json` → `final_full_month_pct_of_perfect`). That's the headline result.
  - `dev_eval --quick` uses the same sampling over the dev window.
- **Task** (`harness/task.md`, copied into the workspace as `TASK.md`): write `policy.py` with
  `Policy(config, region).decide(obs) -> BDUBid`. `obs` is a `bess.observation.Observation` holding the latest
  P5MIN and predispatch runs, past actual prices and the battery state.
- **Agent tools:**
  - Claude Code's Bash, Read, Write, Edit, Glob, Grep and TodoWrite.
  - `python -m harness.dev_eval` to score its policy on dev data.
  - `submit_policy`, which scores on the holdout episodes and returns only the aggregate % of perfect foresight,
    validity and the baseline's score on the same episodes. The default is 5 submissions, and the final score is the last valid one.
  - `status` and `finish`.
- **Isolation:**
  - The CLI runs in bubblewrap (`harness/sandbox.py`) with a read-only filesystem and the home directory and
    `/tmp` hidden. The full database, earlier runs and your Claude Code config aren't visible.
  - The agent's bash commands also run in Claude Code's sandbox with no network.
  - At holdout time the policy runs in its own sandboxed process with no network (`bess/remote.py`) and only
    receives observations, so it can't read actual prices from the simulator.
- **Interruptions:**
  - **Usage limit:** the run pauses until the reported reset time, then resumes the same Claude Code session with
    its full context. `--max-hours` counts active time only, which excludes these pauses and the time spent scoring submissions.
  - **CLI crash or dropped connection:** the harness retries with backoff.
  - **Harness process stopped** (reboot, Ctrl-C, or a limit longer than `--max-pause-hours`): run
    `uv run python -m harness.run_agent --resume <run_id>`. It rebuilds state from `state.json`, the event log and
    `submissions/`, and re-scores any submission whose evaluation was cut off.
  - **Time or cost cap during a submission:** the evaluation finishes and is recorded before the run stops.
- **Logs** (`runs/<run_id>/`):
  - `logs/events.jsonl`: every API call (tokens, API-equivalent cost, latency, thinking summaries), every tool
    call (duration, sizes), submissions, nudges and limits.
  - `logs/transcript.jsonl`: every message in full.
  - `submissions/NN/`: the snapshot, evaluation outputs and `result.json` for each submission.
  - `summary.json`: totals and the final score.
- **Dashboard** (`harness/dashboard.py` + `dashboard.html`): scrubs through the holdout month.
  - Price panel: actual prices up to the scrub time, plus the P5MIN and predispatch runs that were current then.
  - Once a submission has been scored, the panels beneath show its state of charge (against perfect foresight's)
    and its dispatch. A side panel shows the bid for the next interval.
  - Run stats (wall time, tokens, API-equivalent cost, tool calls) and the agent's activity refresh every 5 s.
  - Each submission shows the agent's summary and rationale, its reasoning leading up to it, and the code.
  - URL parameters: `?region=VIC1&tau=2026-08-12T08:30&run=<id>`.
  - It serves holdout data on 127.0.0.1, so keep it out of the agent's reach. The agent's bash has no network.
