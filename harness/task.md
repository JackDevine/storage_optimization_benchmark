# Task: a bidding policy for a grid battery in the Australian NEM

Write a Python policy that operates a battery in Australia's National Electricity Market (NEM). Every 5 minutes it
receives the latest market price forecasts and the battery's state, and returns an energy bid for the next 5-minute
interval. Your goal is to **maximise net revenue on a holdout period you can't see**: the calendar month that follows
your development data, in all five NEM regions (NSW1, QLD1, SA1, TAS1, VIC1).

Submissions are scored on a fixed, undisclosed sample of **{n_episodes} three-day episodes** from the holdout month,
in every region. Each episode starts at a trading-day boundary (04:00) with the battery at 50%. After the run ends,
your final submission is also scored on the **whole holdout month**, and that is the reported result. Don't tune
for 3-day horizons specifically.

## The battery and the market

- **Battery:** 1 MW / 2 MWh. 90% efficiency charging and 90% discharging (81% round trip). State of charge must stay
  within 5–95% (0.1–1.9 MWh). It starts each region at 50%. Degradation costs $5/MWh of throughput, charged on both
  charge and discharge. The full spec is in `config.toml`.
- **Price-taker:** the battery is small enough that it never moves the price. Each interval, your bid is cleared
  against the actual dispatch price, the regional reference price (RRP), and revenue is MW × 5/60 h × RRP.
  Settlement is every 5 minutes. Prices range from the market floor of -$1,000/MWh to the cap of $23,200/MWh.
- **Dispatch path:** a dispatch target is the MW the battery must reach by the **end** of the interval. It moves in a
  straight line from where it was at the start (`obs.last_mw`) to the target, and that path is what's settled. Energy
  is the area under it, e.g. 0 → 0.5 MW delivers (0 + 0.5)/2 × 5/60 = 0.0208 MWh. If the path crosses zero, the
  discharge and charge parts each get their own efficiency. SOC limits are checked along the path, including at the
  zero crossing. See `bess.sim.settle_trajectory`.
- **Energy only.** No FCAS.

## Bids: AEMO's bidirectional unit (BDU) format

A bid (`bess.bid.BDUBid`) has a LOAD side (charging) and a GEN side (discharging). Each side (`bess.bid.SideBid`) has:

- `prices`: 10 strictly increasing prices ($/MWh) in [-1000, 23200].
- `band_mw`: MW offered at each price, each in [0, 1].
- `max_avail`: a MW cap on the side, in [0, 1].
- `ramp_up_rate`, `ramp_down_rate`: MW/min. These must equal the battery's ramp rate, **{ramp_rate:g} MW/min**
  (`config.battery.ramp_rate_mw_per_min`). You state it in the bid but can't choose it. A bid with any other value
  is still dispatched, at the real ramp rate, but it makes the run invalid. Dispatch can move at most
  ramp rate × 5 MW from one interval to the next.

**Clearing** (`bess.bid.dispatch_at_price`):
- "Effective" bands are the MW left after capping at `max_avail`. They accumulate from band 1 upward on the gen side
  and from band 10 downward on the load side.
- If the actual RRP is at or above a gen band's price, that band discharges.
- If the actual RRP is at or below a load band's price, that band charges.
- Net MW = discharge − charge.

**Validation** (`bess.bid.validate`). A bid is rejected if any of these fail:
- each side has 10 prices that strictly increase, all within the floor and cap
- band MW and `max_avail` are in [0, 1]
- **convexity:** every effective load band is priced strictly below every effective gen band

**Price bands are locked for the trading day.** The trading day runs from the interval ending 04:05 to the interval
ending 04:00 the next day. The first accepted bid of each trading day fixes both sides' 10 prices for the rest of that
day. Later bids may change only `band_mw`, `max_avail` and ramp rates; a bid with different prices is rejected.
`obs.locked_prices` tells you the current locked prices.

- A **rejected bid** dispatches 0 MW for that interval.
- If cleared dispatch would breach the battery's SOC or power limits, the simulator clips it and records
  **non-conformance**. Use `max_avail` to manage state of charge.
- A run with **any** rejected bid, non-conformance, or bid with the wrong ramp rate is **invalid** and earns no
  credit. A policy that raises an
  exception counts as a rejected bid.

Helpers in `bess.bid`:
- `setpoint_bid(mw, battery, market)` is dispatched to exactly `mw` whatever the price, and is always valid.
- `curve_bid(...)` turns a price-response curve into a valid bid.

## Your policy

Put it in `policy.py` in this directory:

```python
class Policy:
    def __init__(self, config, region):   # config: bess.config.Config, region: e.g. "SA1"
        ...
    def decide(self, obs):                # obs: bess.observation.Observation -> bess.bid.BDUBid
        ...
```

One `Policy` instance is created per region and called once per interval, in time order, so it can keep state.
`policy.py` may import other `.py` files or read data files (e.g. fitted parameters) that you put in this directory.

`obs` (`bess.observation.Observation`) holds everything known at decision time `obs.tau`. The bid is for the interval
(tau, tau + 5 min]. Timestamps are interval-ending, AEMO time (UTC+10).
- `energy_mwh`, `last_mw`, `locked_prices`, `region`
- `p5min`: the latest 5-minute pre-dispatch run, as a `bess.data.Forecast` with `run_time`, `interval_end` and `rrp`.
  It covers about 1 hour in 5-minute steps, and its first interval ending after `tau` is the one you're bidding for.
- `predispatch`: the latest 30-minute pre-dispatch run, covering 30-minute periods out to the end of the next trading
  day.
- `history_end`, `history_rrp`: actual prices for every interval that has already ended. In the holdout this includes
  the whole development period.

These forecasts are AEMO's own. They are often wrong, especially around price spikes.

## Data and tools

- `data/dev.duckdb`: development data, from 1 June 2026 04:05 to 1 August 2026 00:00. Tables:
  - `dispatchprice` (actual RRP)
  - `p5min_regionsolution` (P5MIN runs)
  - `predispatchprice` (predispatch runs)
  Load a region with `bess.data.load_region(region, Path("data/dev.duckdb"))`.
- `python -m harness.dev_eval [--regions ...] [--start ...] [--end ...]` scores `policy.py` on the dev data. It prints
  net revenue and % of perfect foresight per region, and writes per-interval results to `dev_results/`. A full run of
  5 regions × 2 months takes a few minutes for an MILP-per-interval policy. Use short windows to iterate quickly.
- The `bess` package is read-only source you can import and read. `python -c "import bess; print(bess.__path__)"`
  prints its location, and `harness/` sits beside it. It contains:
  - the simulator (`sim.py`)
  - bids (`bid.py`)
  - an arbitrage MILP (`lp.py`, solved with HiGHS via `highspy`)
  - the baseline policies (`policies.py`)

  `harness/reference_policy/policy.py` is the baseline in the submission format: it re-solves an MILP on the forecast
  every interval and bids the first interval's MW as a setpoint.
- Python libraries: numpy, pandas, duckdb, highspy, pyarrow. There is no network access, so you can't install
  anything else.
- `python -m harness.dev_eval --quick` scores on a sample of 3-day dev episodes chosen the same way as the holdout
  episodes, which is much faster than the full dev window. Its baseline is **{baseline_dev_episodes_pct:.1f}%**.
- The `submit_policy` tool scores the current workspace on the **holdout episodes** and tells you:
  - total net revenue
  - % of perfect foresight across all regions
  - whether the run was valid
  - the baseline's holdout score, for comparison

  You have **{max_submissions} submissions**. Your final score is your **last valid submission**, so don't submit
  something worse last. A submission takes a few minutes. Episodes run in parallel, and **each three-day episode
  must finish within {eval_timeout_min} minutes** or the submission is invalid. That allows about 1 second per
  interval.
- The `status` tool shows your remaining time, budget and submissions.
- The `finish` tool ends the run. Call it when you're done.

## Budget

You have **{max_hours} hours** of working time and **${max_cost_usd}** of model usage. Time spent waiting for
`submit_policy` to score a submission doesn't count against your hours. Your own `dev_eval` runs do. When either runs out, the
run stops and your last valid submission stands. Develop and test against the dev data, submit when an idea is ready,
and use the feedback to decide what to try next.

The baseline scores **{baseline_dev_pct:.1f}%** of perfect foresight on the full dev window.
