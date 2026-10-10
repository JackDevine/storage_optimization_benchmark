"""Rolling 5-minute simulator shared by every dispatch policy.

At each decision time tau the policy submits a BDU energy bid (bess/bid.py)
for the interval (tau, tau+5min]. The policy only sees an Observation
(bess/observation.py): forecasts published at or before tau and actual prices
for intervals already ended. For each interval the simulator:

1. Validates the bid against AEMO's BDU rules and, if lock_daily_prices is on,
   checks the band prices match the first accepted bid of the trading day.
   A rejected bid is dispatched at 0 MW.
2. Clears the bid as a price-taker at the ACTUAL dispatch price, then limits
   the change from last interval's MW by the unit's ramp rate. A bid must
   state that ramp rate: one that doesn't is still dispatched but recorded as
   bid non-conformance.
3. Moves the unit in a straight line from its start-of-interval MW to that
   end-of-interval target (AEMO's dispatch trajectory), clipped to what the
   battery can physically do (power, SOC along the path). AEMO doesn't track
   SOC; keeping dispatch feasible is the participant's job, so any clipping is
   recorded as non-conformance.
4. Settles the energy under that line at the actual price (5-min settlement).

A run with any rejected bid, non-conformance or bid non-conformance is still
scored but marked invalid.
"""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bess.battery import Battery
from bess.bid import bid_to_dict, dispatch_at_price, ramp_rate_mismatch, validate
from bess.config import Config
from bess.data import INTERVAL, RegionData
from bess.observation import Observation, Policy

DT_H = INTERVAL / pd.Timedelta(hours=1)
DT_MIN = INTERVAL / pd.Timedelta(minutes=1)
TRADING_DAY_START = pd.Timedelta(hours=4)
TOLERANCE_MW = 1e-6


@dataclass
class State:
    energy_mwh: float
    mw: float = 0.0  # last interval's dispatched MW (+ = discharge)
    # Price bands locked for the current trading day (load, gen), or None
    # before the day's first accepted bid.
    locked_prices: tuple[tuple[float, ...], tuple[float, ...]] | None = None


def trading_day(interval_end: pd.Timestamp) -> pd.Timestamp:
    """AEMO trading day: intervals ending 04:05 through 04:00 the next day."""
    return (interval_end - INTERVAL - TRADING_DAY_START).normalize()


def apply_unit_ramp(target: float, previous: float, rate_mw_per_min: float) -> float:
    """Limit the move from `previous` MW by the unit's ramp rate, either direction."""
    step = rate_mw_per_min * DT_MIN
    return float(np.clip(target, previous - step, previous + step))


def trajectory_energies(p0: float, p1: float) -> tuple[float, float, float | None]:
    """Energy discharged and charged (MWh, both >= 0) when MW moves in a straight
    line from p0 to p1 over one interval, and the fraction of the interval at
    which it crosses zero (None if it doesn't)."""
    if p0 >= 0 and p1 >= 0:
        return (p0 + p1) / 2 * DT_H, 0.0, None
    if p0 <= 0 and p1 <= 0:
        return 0.0, -(p0 + p1) / 2 * DT_H, None
    f = p0 / (p0 - p1)
    first, second = abs(p0) / 2 * f * DT_H, abs(p1) / 2 * (1 - f) * DT_H
    return ((first, second) if p0 > 0 else (second, first)) + (f,)


def trajectory_socs(p0: float, p1: float, energy: float, b: Battery) -> list[float]:
    """Stored energy at the points where it can peak or bottom out: the zero
    crossing (if any) and the end of the interval."""
    discharged, charged, f = trajectory_energies(p0, p1)
    socs = [energy - discharged / b.discharge_eff + charged * b.charge_eff]
    if f is not None:
        first = abs(p0) / 2 * f * DT_H
        socs.insert(0, energy - first / b.discharge_eff if p0 > 0 else energy + first * b.charge_eff)
    return socs


def settle_trajectory(target: float, p0: float, energy: float, b: Battery, ramp_step: float):
    """Move from p0 towards `target` along a straight line, as far as power, ramp
    and SOC limits allow. Returns (end MW, discharged MWh, charged MWh,
    shortfall MWh). The end MW is the feasible value closest to the target.
    SOC feasibility is monotone in the end MW, so it's found by bisection. If
    even the gentlest move breaches a SOC limit (the unit is already pushing
    past it), the unit stops dead when it reaches the limit: the energy it
    couldn't deliver is the shortfall, which counts as non-conformance."""
    lo = max(-b.power_mw, p0 - ramp_step)
    hi = min(b.power_mw, p0 + ramp_step)
    p1 = min(max(target, lo), hi)

    def too_empty(x):  # discharging too far
        return min(trajectory_socs(p0, x, energy, b)) < b.e_min - 1e-12

    def too_full(x):  # charging too far
        return max(trajectory_socs(p0, x, energy, b)) > b.e_max + 1e-12

    for bad, toward in ((too_empty, lo), (too_full, hi)):
        if bad(p1):
            if bad(toward):
                p1 = toward  # no feasible end MW: best effort, then truncate below
                break
            ok, nok = toward, p1
            for _ in range(50):
                mid = (ok + nok) / 2
                ok, nok = (ok, mid) if bad(mid) else (mid, nok)
            p1 = ok

    discharged, charged, _ = trajectory_energies(p0, p1)
    shortfall = 0.0
    end = energy - discharged / b.discharge_eff + charged * b.charge_eff
    if end < b.e_min:  # stops dead at the limit
        cut = (b.e_min - end) * b.discharge_eff
        shortfall, discharged = min(cut, discharged), max(discharged - cut, 0.0)
    elif end > b.e_max:
        cut = (end - b.e_max) / b.charge_eff
        shortfall, charged = min(cut, charged), max(charged - cut, 0.0)
    return p1, discharged, charged, shortfall


def simulate(
    policy: Policy,
    data: RegionData,
    config: Config,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Simulate bids for intervals ending in (start, end]. Returns one row per
    interval."""
    battery, market = config.battery, config.market
    state = State(battery.e_initial)
    day = None
    records = []
    for interval_end in pd.date_range(start + INTERVAL, end, freq=INTERVAL):
        tau = interval_end - INTERVAL
        if trading_day(interval_end) != day:
            day = trading_day(interval_end)
            state.locked_prices = None

        history_end, history_rrp = data.price_history(tau)
        obs = Observation(
            tau=tau,
            region=data.region,
            energy_mwh=state.energy_mwh,
            last_mw=state.mw,
            locked_prices=state.locked_prices,
            p5min=data.view.p5min(tau),
            predispatch=data.view.predispatch(tau),
            history_end=history_end,
            history_rrp=history_rrp,
        )
        try:
            bid = policy.decide(obs)
            errors = validate(bid, battery, market)
        except Exception as e:  # a policy crash is a rejected bid, not a crashed run
            bid, errors = None, [f"policy error: {type(e).__name__}: {e}"]
        price = data.actual_rrp[interval_end]
        if bid is not None and market.lock_daily_prices and state.locked_prices not in (None, bid.price_bands):
            errors.append("price bands changed within the trading day")

        bid_nonconformance = None
        if errors:
            cleared = target = 0.0
        else:
            if state.locked_prices is None:
                state.locked_prices = bid.price_bands
            cleared = dispatch_at_price(bid, price, market)
            bid_nonconformance = ramp_rate_mismatch(bid, battery)
            target = apply_unit_ramp(cleared, state.mw, battery.ramp_rate_mw_per_min)

        energy_start = state.energy_mwh
        ramp_step = battery.ramp_rate_mw_per_min * DT_MIN
        mw, discharged, charged, shortfall = settle_trajectory(target, state.mw, energy_start, battery, ramp_step)
        state.energy_mwh = energy_start - discharged / battery.discharge_eff + charged * battery.charge_eff
        state.mw = mw

        records.append(
            dict(
                interval_end=interval_end,
                cleared_mw=cleared,
                target_mw=target,
                mw=mw,
                nonconformance_mw=target - mw,
                energy_shortfall_mwh=shortfall,
                discharged_mwh=discharged,
                charged_mwh=charged,
                bid_rejected="; ".join(errors) or None,
                bid_nonconformance=bid_nonconformance,
                bid=json.dumps(bid_to_dict(bid)) if bid is not None else None,
                energy_start_mwh=energy_start,
                energy_end_mwh=state.energy_mwh,
                rrp=price,
                revenue=(discharged - charged) * price,
                degradation_cost=(discharged + charged) * battery.degradation_cost,
            )
        )
    return pd.DataFrame.from_records(records)


def summarise(results: pd.DataFrame, battery: Battery) -> dict:
    discharged, charged = results.discharged_mwh.sum(), results.charged_mwh.sum()
    revenue = results.revenue.sum()
    deg = results.degradation_cost.sum()
    days = len(results) * DT_H / 24
    nonconforming = (results.nonconformance_mw.abs() > TOLERANCE_MW) | (results.energy_shortfall_mwh > 1e-9)
    rejected = int(results.bid_rejected.notna().sum())
    bid_nonconforming = int(results.bid_nonconformance.notna().sum())
    return dict(
        start=results.interval_end.min() - INTERVAL,
        end=results.interval_end.max(),
        intervals=len(results),
        revenue=revenue,
        degradation_cost=deg,
        net_revenue=revenue - deg,
        net_revenue_per_mw_day=(revenue - deg) / battery.power_mw / days,
        mwh_discharged=discharged,
        mwh_charged=charged,
        equivalent_cycles=discharged / (battery.e_max - battery.e_min),
        rejected_bids=rejected,
        nonconformance_intervals=int(nonconforming.sum()),
        nonconformance_mwh=float(results.nonconformance_mw.abs()[nonconforming].sum() * DT_H),
        bid_nonconformance_intervals=bid_nonconforming,
        valid=bool(rejected == 0 and not nonconforming.any() and bid_nonconforming == 0),
    )
