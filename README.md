# Storage Optimization Benchmark

Benchmark data and models for storage optimization against AEMO NEM market data.

## Setup

```
uv sync
```

## Pulling data

```
uv run scripts/pull_data.py                 # raw AEMO tables, October 2021 to the latest published month
uv run python -m harness.build_data         # data/market.duckdb (normalised prices) and data/dev.duckdb
uv run python scripts/price_cap_analysis.py # optional: the analysis behind the price-cap mapping
```

- **`scripts/pull_data.py`** downloads AEMO's MMSDM monthly archive (`scripts/aemo_mmsdm.py`) into
  `data/benchmark.duckdb`, keeping only the columns the benchmark uses:
  - `dispatchprice`: actual 5-minute prices, with the administered-pricing (`APCFLAG`) and market-suspension flags
  - `p5min_regionsolution`: P5MIN forecast runs
  - `predispatchprice`: predispatch forecast runs
  - `market_price_thresholds`: the market price cap (`VOLL`), floor and administered price threshold by date

  It starts at 1 October 2021, the start of 5-minute settlement. Loading is incremental: the `_loaded` table records
  each (table, month), so a rerun fetches only new months. The zips are kept in `data/raw_cache/` (about 7 GB for
  five years); the CSVs are extracted to a temporary directory and deleted. AEMO renamed its archive files in
  August 2024, so files are found by listing each month's directory.
- **`harness.build_data`** writes `data/market.duckdb`, which the simulator and evaluator read, with every price
  mapped onto today's market price cap (`bess/normalise.py`), and its dev subset `data/dev.duckdb`, which is all an
  agent gets. `dispatchprice.RRP_RAW` keeps the original price.
- **The price-cap mapping.** The cap has risen every July, from $15,100/MWh in October 2021 to $23,200. Prices up to
  a knee ($1,000) are unchanged; above it, g(p) = k + (MPC_now − k)·((p − k)/(MPC_then − k))^γ, so each period's cap
  lands on today's. The data is currently built with γ = 1 (a linear stretch of the tail). Actual prices use the cap in
  force when the interval started, forecasts the cap when the forecast ran. `scripts/price_cap_analysis.py`
  compares this with other maps.

None of `data/` is committed (see `.gitignore`).

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
     dispatch trajectory). It settles the energy under that line, discharge and charge each with their own
     efficiency, and clips to power and SOC limits along the way.
     The MILP policies plan with the same trajectory and ramp limits, plus a small SOC margin
     (`bess.policies.dynamics`).
  Any clipping counts as **non-conformance**, since AEMO doesn't track SOC and managing it through MaxAvail is the
  participant's job. A run with any rejected bid or non-conformance keeps its score but is marked `valid = false`,
  and `run_benchmark.py` prints a banner listing invalid runs.
- `bess/bid.py`: the BDU bid, following AEMO's IESS format.
  - Each side (LOAD and GEN) has 10 price bands fixed for the trading day.
  - Each interval also sets MW per band, MaxAvail and ramp rates. The ramp rates must equal the unit's configured
    rate (`battery.ramp_rate_mw_per_min`, or `run_agent.py --ramp-rate`); a bid with any other value is still
    dispatched at the unit's rate but flagged as bid non-conformance, which makes the run invalid.
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
  round-trip losses. `integer=False` solves the LP relaxation instead, about 5× faster.
- `bess/bid.py: conform` caps each side's MaxAvail so whatever clears can be delivered and then stopped within the
  SOC limits. Every built-in policy (and perfect foresight) passes its bid through it, so the MILP's approximate
  energy accounting on steps that cross zero can't cause non-conformance. Policies may form bids any way they
  like and conform them last; the brief says so.
- `bess/policies.py`:
  - `twap` (**the reference baseline**, `harness/reference_policy`): every 5 minutes it takes the time-weighted
    average price V of the blended P5MIN + predispatch forecast as the value of stored energy, and bids full
    charge at prices at or below η_c·V − wear and full discharge at or above V/η_d + wear: trades that break even
    after round-trip losses and wear. Band prices are set once per trading day, in steps around that day's
    thresholds, and each interval the MW goes in the bands nearest the current thresholds. About 0.5 ms per
    decision.
  - `milp_forecast` / `lp_forecast` (the second baseline is the LP, `harness/lp_policy`): every 5 minutes it joins
    the latest P5MIN run (5-min steps) to the latest predispatch run (30-min steps), solves the MILP (or LP), and
    submits the first interval's MW as a `setpoint_bid`. If a P5MIN run is missing from AEMO's archive, the next
    hour is priced from predispatch.
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
- `bess/static_bid.py`: the static bid baselines, which use no forecasts at all. *Parked:* their parameters were
  fitted to the old two-month split, so they aren't scored as baselines until they're refitted.
  - **The bid:** 10 load bands and 10 gen bands with prices fixed for all time. Every load price is below every
    gen price, so the bid is always convex and never charges and discharges at once.
  - **Three variants**, by what the band MW may depend on:
    - `static_bid`: nothing (40 numbers per region).
    - `static_bid_tod`: the time of day, in 30-minute blocks.
    - `static_bid_tod_soc`: the time of day and the state of charge, in 3 bins.
  - **MaxAvail** is the only thing that changes otherwise. Each side's is the largest end-of-interval MW from which
    the unit can still ramp back to 0 at its own rate without leaving its SOC limits. With straight-line dispatch,
    ending an interval at x MW means x·dt/2 more next interval, so this matters. It's closed-form, and it means
    whatever clears is deliverable, so the policy never causes non-conformance.
  - **Fitting:** `scripts/fit_static_bids.py` fits one bid per region to dev prices with CMA-ES, on a fast
    vectorised simulator for these bids. `scripts/check_static_bid_parity.py` checks that simulator against
    `bess/sim.py`. The fit is a one-off, and `cma` isn't a project dependency: run it with
    `uv run --with cma==4.5.0 python scripts/fit_static_bids.py --validate`. The fitted parameters live in
    `harness/static_bid_policies/<variant>/params.json`, next to a `policy.py` in the submission format.
  - **Scoring:** `harness.evaluator reference` scores them with the MILP baseline on every reference set. A
    refit is re-scored automatically, keyed by a hash of its parameters. Submission feedback, `summary.json` and
    the dashboard report them alongside the MILP baseline. They're fitted to dev prices, so their dev scores are
    in-sample; the brief says so.
  - **Results** (% of perfect foresight; MILP baseline: dev 26.3, holdout episodes 33.3, full month 38.3):

    | Variant | June fit → July | Dev (in-sample) | Holdout episodes | Full month | Fit time |
    |---|---|---|---|---|---|
    | `static_bid` | 63.6 → 27.7 | 51.5 | 20.1 | 19.7 | 18 min |
    | `static_bid_tod` | 73.0 → 31.6 | 60.2 | 25.6 | 24.3 | 34 min |
    | `static_bid_tod_soc` | 73.1 → 30.8 | 60.3 | 25.8 | 24.4 | 40 min |

    They overfit badly, mostly to a few SA1 cap-price spikes (SA1 is negative on the holdout). The SOC bins add
    nothing; the time-of-day blocks add about 5 points out of sample.

## Agent experiments

```
claude setup-token        # once; save the token to ~/.config/storage-benchmark/claude_oauth_token
uv run python -m harness.evaluator reference      # once; caches baseline dev/holdout scores
uv run python -m harness.run_agent --model claude-sonnet-5-5 --max-hours 2 --max-cost 20
uv run python -m harness.run_agent --model openai/gpt-5.1 --max-hours 2 --max-cost 20   # any OpenRouter slug
uv run python -m harness.dashboard                 # live view at http://127.0.0.1:8050
MODELS="claude-opus-5-5 google/gemini-3-pro" scripts/run_sweep.sh   # several models, one after another
```

A model is asked to write a bidding policy. There are two backends, picked by the model name:

- **Claude models** (`claude-…`): `harness/run_agent.py` drives Claude Code through the Claude Agent SDK on your
  Claude subscription. Costs are reported as what the tokens would cost at standard API rates.
- **OpenRouter slugs** (`vendor/model`): `harness/openrouter_agent.py` is the harness's own agent loop over
  OpenRouter's chat completions API. Save your key to `~/.config/storage-benchmark/openrouter_api_key` (or export
  `OPENROUTER_API_KEY`).
  - **The cost is real.** It's what OpenRouter bills, from each response's `usage.cost`, and `--max-cost`
    (default $20) is a spending cap, checked before every API call.
  - The model must support tool calling. The harness checks this, and the key, before a run starts.
  - It gets the same brief, workspace and harness tools as the Claude runs. Its workspace tools are its own:
    `bash` (one command per call, no network), `read_file`, `write_file` and `edit_file`, with a short system
    prompt (`harness/openrouter_system.md`) instead of Claude Code's. **That's a different scaffold**, so keep it
    in mind when comparing with Claude Code runs. Running an `anthropic/claude-…` slug through the same loop is
    the control for that.
  - `--effort` becomes OpenRouter's `reasoning.effort`. Each response's reasoning (`reasoning_details`) is passed
    back unmodified on later calls.
  - When the prompt nears the model's context window, the oldest tool outputs are elided.

- **Split** (`harness/splits.py`): the agent develops on October 2021 – August 2025 (`data/dev.duckdb`, with only
  what was published by the end of August 2025). It's scored on September 2025 – August 2026, all five regions.
  Prices are mapped onto today's cap (see *Pulling data*).
  - **Excluded:** the June 2022 market suspension (trading days 12–23 June) and October 2022, which has no
    predispatch runs in AEMO's archive.
  - **Monthly windows:** long periods are scored one calendar month at a time (`month_windows`), each starting
    the battery at 50%, so every (region, month) is a short parallel job and perfect foresight stays a
    month-sized MILP. The dev reference set is every dev month; the full holdout is every holdout month.
- **Episodes:** submissions are scored on a seeded sample of 3-day episodes (`sample_episodes`), one per holdout
  month (12), each starting at 04:00 with the battery at 50%. All region × episode jobs run in parallel, so a
  submission takes about one episode's simulated time.
  - After the run ends, the final valid submission is also scored on the **full holdout**, month by month
    (`summary.json` → `final_full_holdout_pct_of_perfect`). That's the headline result.
  - `dev_eval` (no arguments) uses the same sampling over the dev period, one episode per four months (14).
    `--full` scores every dev month.
- **Task** (`harness/task.md`, copied into the workspace as `TASK.md`): write `policy.py` with
  `Policy(config, region).decide(obs) -> BDUBid`. `obs` is a `bess.observation.Observation` holding the latest
  P5MIN and predispatch runs, past actual prices and the battery state.
- **Agent tools:**
  - Claude: Claude Code's Bash, Read, Write, Edit, Glob, Grep and TodoWrite. OpenRouter: `bash`, `read_file`,
    `write_file`, `edit_file`.
  - `python -m harness.dev_eval` to score its policy on dev data.
  - `submit_policy`, which scores on the holdout episodes and returns only the aggregate % of perfect foresight,
    validity and the baseline's score on the same episodes. The default is 5 submissions, and the final score is the last valid one.
  - `status` and `finish`.
- **Isolation:**
  - The CLI runs in bubblewrap (`harness/sandbox.py`) with a read-only filesystem and the home directory and
    `/tmp` hidden. The full database, earlier runs and your Claude Code config aren't visible.
  - The agent's bash commands also run in Claude Code's sandbox with no network. For OpenRouter runs, each bash
    command runs in bubblewrap with no network, with the workspace and a per-run `/tmp` (kept between commands)
    as the only writable paths, and the file tools only reach the workspace (reads may also reach
    `bess/` and `harness/`), with symlinks resolved first.
  - The policy and OpenRouter bash sandboxes start with an empty environment, so API keys and tokens never reach
    agent code.
  - At holdout time the policy runs in its own sandboxed process with no network (`bess/remote.py`) and only
    receives observations, so it can't read actual prices from the simulator.
- **Interruptions:**
  - **Usage limit:** the run pauses until the reported reset time, then resumes the same Claude Code session with
    its full context. For OpenRouter, rate limits and provider errors are retried with backoff, with the clock
    stopped, and running out of credits ends the run (`ended_by: no_credits`). `--max-hours` counts active time only, which excludes these pauses and the time spent scoring submissions.
  - **CLI crash or dropped connection:** the harness retries with backoff.
  - **Harness process stopped** (reboot, Ctrl-C, or a limit longer than `--max-pause-hours`): run
    `uv run python -m harness.run_agent --resume <run_id>`. It rebuilds state from `state.json`, the event log and
    `submissions/`, and re-scores any submission whose evaluation was cut off.
  - **Time or cost cap during a submission:** the evaluation finishes and is recorded before the run stops.
  - **Scoring failure:** a scoring job that dies with no output (killed from outside, not a policy crash) is
    retried once. If scoring itself fails, the submission isn't counted and the agent is told it can resubmit.
- **Logs** (`runs/<run_id>/`):
  - `logs/events.jsonl`: every API call (tokens, API-equivalent cost, latency, thinking summaries), every tool
    call (duration, sizes), submissions, nudges and limits.
  - `logs/transcript.jsonl`: every message in full (OpenRouter: every raw response).
  - `logs/messages.jsonl` (OpenRouter): the conversation, which `--resume` reloads.
  - `submissions/NN/`: the snapshot, evaluation outputs and `result.json` for each submission.
  - `summary.json`: totals and the final score. `cost_usd` is the API-equivalent cost for Claude runs and the
    billed cost for OpenRouter runs (`billing` says which).
- **Dashboard** (`harness/dashboard.py` + `dashboard.html`): scrubs through the holdout.
  - Price panel: actual prices up to the scrub time, plus the P5MIN and predispatch runs that were current then.
  - Once a submission has been scored, the panels beneath show its state of charge (against perfect foresight's)
    and its dispatch. A side panel shows the bid for the next interval.
  - Run stats (wall time, tokens, API-equivalent cost, tool calls) and the agent's activity refresh every 5 s.
  - Each submission shows the agent's summary and rationale, its reasoning leading up to it, and the code.
  - URL parameters: `?region=VIC1&tau=2026-02-12T08:30&run=<id>`.
  - It serves holdout data on 127.0.0.1, so keep it out of the agent's reach. The agent's bash has no network.
