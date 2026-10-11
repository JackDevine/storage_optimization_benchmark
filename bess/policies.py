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

- TWAPPolicy: charge when the price is below the forecast's time-weighted
  average price (the value of stored energy) by more than losses and wear,
  discharge when it's above it by more.

MILPForecastPolicy and PerfectForesight plan a MW schedule and submit it as a
setpoint bid (bess/bid.py), which is dispatched to that MW whatever the price.
Every policy passes its bid through `bess.bid.conform`, which caps MaxAvail so
the unit never breaches its SOC limits: the MILP's plan is only approximate
when dispatch crosses zero within an interval, so it can't be trusted to be
deliverable on its own.
"""

import numpy as np
import pandas as pd

from bess.bid import N_BANDS, BDUBid, SideBid, conform, curve_bid, setpoint_bid
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

    AEMO's archive has the odd gap in P5MIN runs (e.g. about an hour on
    5 September 2024). If the latest run doesn't cover the next interval, the
    hour P5MIN would cover is priced from the predispatch periods containing
    each of its 5-minute intervals instead.
    """
    tau, p5, pd_run = obs.tau, obs.p5min, obs.predispatch
    if pd_run is None:
        raise ValueError(f"No predispatch forecast available at {tau}")

    tau64 = np.datetime64(tau, "ns")
    nxt = tau64 + np.timedelta64(INTERVAL)
    if p5 is not None:
        keep = p5.interval_end > tau64
        p5_end, p5_rrp = p5.interval_end[keep], p5.rrp[keep]
    if p5 is None or len(p5_end) == 0 or p5_end[0] != nxt:
        p5_end = nxt + np.arange(12) * np.timedelta64(INTERVAL)
        i = np.searchsorted(pd_run.interval_end, p5_end)  # the period containing each interval
        if i[0] == len(pd_run.interval_end):
            raise ValueError(f"No forecast covers the interval after {tau}")
        p5_end, p5_rrp = p5_end[i < len(pd_run.interval_end)], pd_run.rrp[i[i < len(pd_run.interval_end)]]

    last = p5_end[-1]
    keep = pd_run.interval_end > last
    pd_end, pd_rrp = pd_run.interval_end[keep], pd_run.rrp[keep]

    step_ends = np.concatenate([p5_end, pd_end])
    prices = np.concatenate([p5_rrp, pd_rrp])
    starts = np.concatenate([[tau64], step_ends[:-1]])
    dt_hours = (step_ends - starts) / np.timedelta64(1, "h")
    return prices, dt_hours


class MILPForecastPolicy:
    """`integer=False` plans with the LP relaxation instead (lp_forecast)."""

    def __init__(self, config: Config, integer: bool = True):
        self.config = config
        self.integer = integer
        self.name = "milp_forecast" if integer else "lp_forecast"

    def decide(self, obs: Observation) -> BDUBid:
        prices, dt_hours = blended_forecast(obs)
        cfg = self.config
        c, d, _ = solve_schedule(
            prices, dt_hours, cfg.battery, obs.energy_mwh, cfg.terminal, cfg.mip_rel_gap, **dynamics(cfg, obs.last_mw),
            integer=self.integer,
        )
        bid = setpoint_bid(float(d[0] - c[0]), cfg.battery, cfg.market)
        return conform(bid, obs.energy_mwh, obs.last_mw, cfg.battery)


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
        bid = curve_bid(cfg.bidding.load_prices, mw[:n], cfg.bidding.gen_prices, mw[n:], cfg.battery)
        return conform(bid, obs.energy_mwh, obs.last_mw, cfg.battery)


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
        bid = setpoint_bid(mw, self.config.battery, self.config.market)
        return conform(bid, obs.energy_mwh, obs.last_mw, self.config.battery)


# TWAPPolicy's deadband scale, calibrated so the policy averages one cycle a day
# over the dev months in every region (typical of the market), at the
# config's wear cost: scripts/calibrate_twap.py.
TWAP_DEADBAND = 0.413


class TWAPPolicy:
    """Threshold bidding around the forecast's time-weighted average price
    (TWAP), net of losses and wear.

    The TWAP of the blended P5MIN + predispatch forecast stands in for the
    value V of stored energy. Charging 1 MWh at price p stores eta_c MWh, so it
    pays when p <= eta_c * V - wear; discharging delivers energy that cost
    1 / eta_d stored MWh, so it pays when p >= V / eta_d + wear. Where V is so
    negative that the two thresholds cross, they're replaced by their midpoint.
    `deadband` then scales each threshold's distance from V: 1 trades only
    where a trade breaks even, 0 trades at V exactly. Full power either side,
    nothing in between.

    Band prices are locked for the trading day, so at its first decision the
    policy lays a ladder of 10 load prices around the day's charge threshold
    and 10 gen prices around its discharge threshold; each interval it puts the
    full MW in the load band nearest below the current charge threshold and the
    gen band nearest above the discharge threshold."""

    def __init__(self, config: Config, deadband: float = TWAP_DEADBAND):
        self.config = config
        self.deadband = deadband
        self.name = "twap"

    def thresholds(self, obs: Observation) -> tuple[float, float]:
        prices, dt_hours = blended_forecast(obs)
        v = float(np.sum(prices * dt_hours) / np.sum(dt_hours))
        b = self.config.battery
        charge, discharge = b.charge_eff * v - b.degradation_cost, v / b.discharge_eff + b.degradation_cost
        if charge > discharge:
            charge = discharge = (charge + discharge) / 2
        # Narrow (or widen) the gap between them around V.
        return v - self.deadband * (v - charge), v + self.deadband * (discharge - v)

    def ladder(self, charge: float, discharge: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """The day's band prices (at the regional reference node): steps of 5%
        of the thresholds' level (at least $5), the threshold at load band 6
        and gen band 5, inside the market's limits."""
        m = self.config.market
        step = max(5.0, 0.05 * max(abs(charge), abs(discharge)))
        k = np.arange(N_BANDS)

        def fit(prices):  # shift (not squash) back inside [MFP, MPC], keeping the steps
            prices = prices + max(0.0, m.mfp - prices[0]) - max(0.0, prices[-1] - m.mpc)
            return np.clip(prices, m.mfp, m.mpc)

        return tuple(fit(charge + step * (k - 5))), tuple(fit(discharge + step * (k - 4)))

    def decide(self, obs: Observation) -> BDUBid:
        cfg, b, m = self.config, self.config.battery, self.config.market
        charge, discharge = self.thresholds(obs)
        if obs.locked_prices is None:
            load_rrn, gen_rrn = self.ladder(charge, discharge)
            load_prices = tuple(float(p * m.load_mlf) for p in load_rrn)
            gen_prices = tuple(float(p * m.gen_mlf) for p in gen_rrn)
        else:
            load_prices, gen_prices = obs.locked_prices
        load_rrn = np.asarray(load_prices) / m.load_mlf
        gen_rrn = np.asarray(gen_prices) / m.gen_mlf
        load_mw, gen_mw = np.zeros(N_BANDS), np.zeros(N_BANDS)
        below = np.flatnonzero(load_rrn <= charge)
        if len(below):
            load_mw[below[-1]] = b.power_mw
        # Strictly above the charging band too, or the bid isn't convex (the
        # thresholds can meet when the TWAP is very negative).
        floor = load_rrn[below[-1]] if len(below) else -np.inf
        above = np.flatnonzero((gen_rrn >= discharge) & (gen_rrn > floor))
        if len(above):
            gen_mw[above[0]] = b.power_mw
        r = b.ramp_rate_mw_per_min
        bid = BDUBid(SideBid(load_prices, tuple(load_mw), float(load_mw.sum()), r, r),
                     SideBid(gen_prices, tuple(gen_mw), float(gen_mw.sum()), r, r))
        return conform(bid, obs.energy_mwh, obs.last_mw, b)
