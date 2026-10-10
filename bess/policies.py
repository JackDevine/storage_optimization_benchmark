"""Dispatch policies.

- MILPForecastPolicy: the baseline. Every 5 minutes, stitch together the latest
  P5MIN run (5-min steps, ~1h) and the latest predispatch run (30-min steps,
  out to end of next trading day), solve the arbitrage MILP against that
  forecast, and bid only the first interval's power.
- MILPBidCurvePolicy: like MILPForecastPolicy, but submits a real price-band
  bid. For each of the day's band prices it re-solves the MILP with the next
  interval's price set to that band price (rest of the trajectory from the
  forecast), and turns the resulting first-interval MW at each price into a
  bid curve.
- PerfectForesight: solves the same MILP once over the whole period on actual
  prices. Not a realisable policy — it's the ceiling to score others against.

MILPForecastPolicy and PerfectForesight plan a MW schedule and submit it as a
setpoint bid (bess/bid.py), which is dispatched to that MW whatever the price.
"""

import numpy as np
import pandas as pd

from bess.bid import BDUBid, curve_bid, setpoint_bid
from bess.config import Config
from bess.data import INTERVAL
from bess.lp import solve_schedule
from bess.observation import Observation

HOUR = pd.Timedelta(hours=1)


def dynamics(config: Config, p_prev: float) -> dict:
    """MILP options so the plan matches how the simulator settles dispatch: the
    straight-line trajectory starts from the current MW, and the plan keeps a
    small SOC margin for the MILP's zero-crossing approximation (at most about a
    quarter of one interval's full-power energy; a third leaves room for solver
    tolerances)."""
    return {"p_prev": p_prev, "soc_margin_mwh": config.battery.power_mw * (5 / 60) / 3}


def blended_forecast(obs: Observation) -> tuple[np.ndarray, np.ndarray]:
    """Forecast prices and step lengths (hours) for the steps after obs.tau.

    P5MIN covers 5-min intervals up to its last interval end L. After that,
    predispatch: first a partial step (L, next half-hour] priced at the
    predispatch period containing it, then whole 30-min periods.
    """
    tau, p5, pd_run = obs.tau, obs.p5min, obs.predispatch
    if p5 is None or pd_run is None:
        raise ValueError(f"No forecast available at {tau}")

    tau64 = np.datetime64(tau, "ns")
    keep = p5.interval_end > tau64
    p5_end, p5_rrp = p5.interval_end[keep], p5.rrp[keep]
    if len(p5_end) == 0 or p5_end[0] != tau64 + np.timedelta64(INTERVAL):
        raise ValueError(f"P5MIN run at {tau} doesn't cover the next interval")

    last = p5_end[-1]
    keep = pd_run.interval_end > last
    pd_end, pd_rrp = pd_run.interval_end[keep], pd_run.rrp[keep]

    step_ends = np.concatenate([p5_end, pd_end])
    prices = np.concatenate([p5_rrp, pd_rrp])
    starts = np.concatenate([[tau64], step_ends[:-1]])
    dt_hours = (step_ends - starts) / np.timedelta64(1, "h")
    return prices, dt_hours


class MILPForecastPolicy:
    def __init__(self, config: Config):
        self.config = config
        self.name = "milp_forecast"

    def decide(self, obs: Observation) -> BDUBid:
        prices, dt_hours = blended_forecast(obs)
        cfg = self.config
        c, d, _ = solve_schedule(
            prices, dt_hours, cfg.battery, obs.energy_mwh, cfg.terminal, cfg.mip_rel_gap, **dynamics(cfg, obs.last_mw)
        )
        return setpoint_bid(float(d[0] - c[0]), cfg.battery, cfg.market)


class MILPBidCurvePolicy:
    def __init__(self, config: Config):
        self.config = config
        self.name = "milp_bid_curve"
        bidding, market = config.bidding, config.market
        # Scenario prices for the next interval are the band prices at the RRN.
        load_rrn = np.asarray(bidding.load_prices, float) / market.load_mlf
        gen_rrn = np.asarray(bidding.gen_prices, float) / market.gen_mlf
        scenario = np.concatenate([load_rrn, gen_rrn])
        self._order = np.argsort(scenario, kind="stable")
        self._scenario_prices = scenario[self._order]

    def decide(self, obs: Observation) -> BDUBid:
        prices, dt_hours = blended_forecast(obs)
        cfg = self.config

        def first_mw(price: float) -> float:
            scenario = prices.copy()
            scenario[0] = price
            c, d, _ = solve_schedule(
                scenario, dt_hours, cfg.battery, obs.energy_mwh, cfg.terminal, cfg.mip_rel_gap,
                **dynamics(cfg, obs.last_mw),
            )
            return float(d[0] - c[0])

        sorted_mw = response_curve(self._scenario_prices, first_mw)
        mw = np.empty_like(sorted_mw)
        mw[self._order] = sorted_mw
        n = len(cfg.bidding.load_prices)
        return curve_bid(cfg.bidding.load_prices, mw[:n], cfg.bidding.gen_prices, mw[n:], cfg.battery)


def response_curve(prices: np.ndarray, first_mw, tol_mw: float = 1e-6) -> np.ndarray:
    """First-interval MW at each of the sorted `prices`.

    The optimal response is non-decreasing in the next interval's price, so
    if it's the same at both ends of a range it's the same in between. Solve
    the ends, then split in half only where they differ. That typically takes
    a handful of solves instead of one per price. A running max enforces the
    monotonicity against solver noise, and near-zero MW are snapped to 0.
    """
    k = len(prices)
    mw = np.full(k, np.nan)

    def at(i: int) -> float:
        if np.isnan(mw[i]):
            mw[i] = first_mw(prices[i])
        return mw[i]

    def fill(lo: int, hi: int) -> None:
        if hi - lo <= 1:
            return
        if abs(at(hi) - at(lo)) <= tol_mw:
            mw[lo + 1 : hi] = mw[lo]
            return
        mid = (lo + hi) // 2
        at(mid)
        fill(lo, mid)
        fill(mid, hi)

    fill(0, k - 1)
    at(0), at(k - 1)
    mw = np.maximum.accumulate(mw)
    mw[np.abs(mw) < tol_mw] = 0.0
    return mw


class PerfectForesight:
    def __init__(
        self,
        config: Config,
        actual_rrp: pd.Series,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ):
        prices = actual_rrp.loc[start + INTERVAL : end]
        dt = np.full(len(prices), INTERVAL / HOUR)
        b = config.battery
        c, d, _ = solve_schedule(
            prices.to_numpy(), dt, b, b.e_initial, config.terminal, config.mip_rel_gap, **dynamics(config, 0.0)
        )
        self.schedule = pd.Series(d - c, index=prices.index)
        self.config = config
        self.name = "perfect_foresight"

    def decide(self, obs: Observation) -> BDUBid:
        mw = float(self.schedule[obs.tau + INTERVAL])
        return setpoint_bid(mw, self.config.battery, self.config.market)
