"""Static bid policies: fixed price bands, fitted once to historical prices.

A static bid is 10 load bands and 10 gen bands, each a price and a MW. The
prices never change (so the daily price lock holds trivially) and every load
price sits below every gen price (so the bid is always convex and the two sides
never clear at once). Three variants, by what the band MW may depend on:

- static: nothing (40 numbers)
- tod: the time of day, in 30-minute blocks (20 prices + 48 x 20 MW)
- tod_soc: the time of day and the state of charge, in 3 bins
  (20 prices + 48 x 3 x 20 MW)

The only thing that changes from interval to interval apart from that is each
side's MaxAvail, which keeps the unit inside its SOC limits. It's
"stopping-safe": the largest end-of-interval MW from which the unit can still
ramp back to 0 at its own rate without leaving the limits. (With straight-line
dispatch, ending an interval at x MW means the next interval delivers x*dt/2
more even if nothing clears.) Holding that at every interval means whatever
clears is physically deliverable, so the policy never causes non-conformance.

`simulate_static` is a fast simulator for these policies alone, vectorised
over many candidate parameter sets and regions at once, for fitting
(scripts/fit_static_bids.py). It must agree with bess/sim.py's `simulate`
exactly; scripts/check_static_bid_parity.py checks that.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bess.battery import Battery
from bess.bid import N_BANDS, BDUBid, SideBid
from bess.config import Config, Market
from bess.observation import Observation

BLOCK_MINUTES = 30
SOC_MARGIN_MWH = 1e-6  # keeps rounding from ever touching a SOC limit


@dataclass(frozen=True)
class StaticBidParams:
    """One region's static bid. `load_mw` and `gen_mw` have shape
    (blocks, soc_bins, 10): blocks = 1 (no time of day) or 48, soc_bins = 1 or
    more. Prices are connection-point $/MWh, strictly increasing, all load
    prices below all gen prices; MW are in [0, power_mw]."""

    load_prices: np.ndarray  # (10,)
    gen_prices: np.ndarray  # (10,)
    load_mw: np.ndarray  # (blocks, soc_bins, 10)
    gen_mw: np.ndarray  # (blocks, soc_bins, 10)

    @property
    def blocks(self) -> int:
        return self.load_mw.shape[0]

    @property
    def soc_bins(self) -> int:
        return self.load_mw.shape[1]

    def to_dict(self) -> dict:
        return {
            "load_prices": self.load_prices.tolist(), "gen_prices": self.gen_prices.tolist(),
            "load_mw": self.load_mw.tolist(), "gen_mw": self.gen_mw.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StaticBidParams":
        return cls(*(np.asarray(d[k], float) for k in ("load_prices", "gen_prices", "load_mw", "gen_mw")))


def load_params(path: Path) -> dict[str, StaticBidParams]:
    """Fitted parameters by region, from a file written by fit_static_bids.py."""
    raw = json.loads(Path(path).read_text())
    return {region: StaticBidParams.from_dict(d) for region, d in raw["regions"].items()}


# ---- shared rules (scalars or arrays) ----

def block_of(minute_of_day, blocks: int):
    """Time-of-day block for a decision at `minute_of_day` (the interval's start)."""
    return (minute_of_day // BLOCK_MINUTES) % blocks if blocks > 1 else minute_of_day * 0


def soc_bin(energy_mwh, battery: Battery, bins: int):
    """Equal-width bins of usable energy, e_min..e_max."""
    if bins == 1:
        return np.zeros_like(energy_mwh, dtype=int)
    frac = (energy_mwh - battery.e_min) / (battery.e_max - battery.e_min)
    return np.clip(np.floor(frac * bins).astype(int), 0, bins - 1)


def stopping_safe_avail(budget_mwh, p_same, battery: Battery, ramp_step: float, dt_h: float):
    """Largest end MW x on one side such that moving there from the current MW
    and then ramping back to 0 at `ramp_step` MW per interval delivers at most
    `budget_mwh` (at the meter) on that side. `p_same` is the current MW on the
    same side (>= 0; 0 if the unit is on the other side, which only helps).

    Ending at x then ramping down takes n = ceil(x / ramp_step) more intervals;
    the energy is dt/2 * (p_same + 2 n x - ramp_step n (n-1)), linear in x for
    each n, so solve each segment and keep the one that fits."""
    n_max = int(np.ceil(battery.power_mw / ramp_step - 1e-9))
    target = 2 * np.asarray(budget_mwh, float) / dt_h - p_same
    x = np.zeros_like(target)
    for n in range(1, n_max + 1):
        lo, hi = (n - 1) * ramp_step, n * ramp_step
        xn = (target + ramp_step * n * (n - 1)) / (2 * n)
        x = np.where(xn >= lo, np.minimum(xn, hi), x)
    return np.clip(x, 0.0, battery.power_mw)


def max_avails(energy_mwh, last_mw, battery: Battery, dt_h: float):
    """(load MaxAvail, gen MaxAvail) that keep the unit stopping-safe."""
    ramp_step = battery.ramp_rate_mw_per_min * dt_h * 60
    gen_budget = np.maximum(energy_mwh - battery.e_min - SOC_MARGIN_MWH, 0.0) * battery.discharge_eff
    load_budget = np.maximum(battery.e_max - SOC_MARGIN_MWH - energy_mwh, 0.0) / battery.charge_eff
    gen = stopping_safe_avail(gen_budget, np.maximum(last_mw, 0.0), battery, ramp_step, dt_h)
    load = stopping_safe_avail(load_budget, np.maximum(-last_mw, 0.0), battery, ramp_step, dt_h)
    return load, gen


# ---- the policy ----

class StaticBidPolicy:
    def __init__(self, config: Config, params: StaticBidParams, name: str = "static_bid"):
        self.config, self.params, self.name = config, params, name
        self._load_prices = tuple(float(x) for x in params.load_prices)
        self._gen_prices = tuple(float(x) for x in params.gen_prices)

    def decide(self, obs: Observation) -> BDUBid:
        b, p = self.config.battery, self.params
        dt_h = 5 / 60
        block = block_of(obs.tau.hour * 60 + obs.tau.minute, p.blocks)
        bin_ = int(soc_bin(np.float64(obs.energy_mwh), b, p.soc_bins))
        load_avail, gen_avail = max_avails(obs.energy_mwh, obs.last_mw, b, dt_h)
        ramp = b.ramp_rate_mw_per_min
        return BDUBid(
            load=SideBid(self._load_prices, tuple(float(x) for x in p.load_mw[block, bin_]), float(load_avail), ramp, ramp),
            gen=SideBid(self._gen_prices, tuple(float(x) for x in p.gen_mw[block, bin_]), float(gen_avail), ramp, ramp),
        )


# ---- parametrisation for fitting ----

PRICE_SCALE = 50.0  # $/MWh; prices are spaced evenly in asinh(price / scale)


def price_bounds(market: Market) -> tuple[float, float]:
    return np.arcsinh(market.mfp / PRICE_SCALE), np.arcsinh(market.mpc / PRICE_SCALE)


def prices_from_gaps(z: np.ndarray, market: Market) -> tuple[np.ndarray, np.ndarray]:
    """21 unconstrained numbers per candidate (..., 21) -> 20 RRN prices, strictly
    increasing between MFP and MPC: the softmax of z gives the gaps in asinh space
    from MFP to load band 1, ..., gen band 10 to MPC. The first 10 are the load
    prices and the last 10 the gen prices, so every load price is below every gen price."""
    u_min, u_max = price_bounds(market)
    w = np.exp(z - z.max(axis=-1, keepdims=True))
    u = u_min + (u_max - u_min) * np.cumsum(w, axis=-1)[..., :-1] / w.sum(axis=-1, keepdims=True)
    prices = PRICE_SCALE * np.sinh(u)
    return prices[..., :N_BANDS], prices[..., N_BANDS:]


def gaps_from_prices(load_prices, gen_prices, market: Market) -> np.ndarray:
    """Inverse of prices_from_gaps (for seeding a fit from a known ladder)."""
    u_min, u_max = price_bounds(market)
    u = np.arcsinh(np.concatenate([load_prices, gen_prices]) / PRICE_SCALE)
    gaps = np.diff(np.concatenate([[u_min], u, [u_max]]))
    if np.any(gaps <= 0):
        raise ValueError("seed prices must strictly increase, load below gen, inside (MFP, MPC)")
    return np.log(gaps)


# ---- fast simulator for fitting ----

def simulate_static(
    rrp: np.ndarray,
    minute_of_day: np.ndarray,
    load_prices: np.ndarray,
    gen_prices: np.ndarray,
    load_mw: np.ndarray,
    gen_mw: np.ndarray,
    config: Config,
    detail: bool = False,
    valid: np.ndarray | None = None,
):
    """Net revenue of K candidate static bids in R lanes (regions, or region
    and window pairs) over T intervals.

    rrp: (R, T) actual prices; minute_of_day: (T,) decision time of each interval;
    load_prices, gen_prices: (K, R, 10) connection-point prices; load_mw,
    gen_mw: (K, R, blocks, soc_bins, 10). Every lane starts at the initial SOC
    and 0 MW. `valid` (R, T) marks the intervals each lane simulates; elsewhere
    (padding, so windows of different lengths share one time axis) its state
    stands still. Returns net revenue (K, R), or with `detail` also the MW and
    energy paths (K, R, T).

    Follows bess/sim.py step for step for these bids: the stopping-safe
    MaxAvail caps each side, the bid clears as a price-taker, the result is
    limited by the unit's ramp rate and settled on a straight-line trajectory.
    Stopping-safety means the SOC never needs clipping, so there's no
    bisection to do.
    """
    b, m = config.battery, config.market
    K, R = load_prices.shape[:2]
    T = rrp.shape[1]
    blocks, bins = load_mw.shape[2], load_mw.shape[3]
    dt_h = 5 / 60
    ramp_step = b.ramp_rate_mw_per_min * dt_h * 60

    # How many gen bands clear (a prefix from band 1) and how many load bands
    # (a suffix down from band 10) at each interval's price: (T, K, R).
    gen_rrn, load_rrn = gen_prices / m.gen_mlf, load_prices / m.load_mlf
    n_gen = np.zeros((T, K, R), np.int8)
    n_load = np.zeros((T, K, R), np.int8)
    price = rrp.T[:, None, :]  # (T, 1, R)
    for i in range(N_BANDS):
        n_gen += gen_rrn[None, :, :, i] <= price
        n_load += load_rrn[None, :, :, i] >= price
    # Cumulative MW offered: gen bands 1..j and load bands 10..(11-j), j = 0..10,
    # laid out (blocks, K, R, bins * 11) so each step gathers from one block.
    zeros = np.zeros(load_mw.shape[:-1] + (1,))
    gen_cum = np.concatenate([zeros, np.cumsum(gen_mw, axis=-1)], axis=-1)
    load_cum = np.concatenate([zeros, np.cumsum(load_mw[..., ::-1], axis=-1)], axis=-1)
    gen_cum = np.moveaxis(gen_cum, 2, 0).reshape(blocks, K, R, bins * (N_BANDS + 1))
    load_cum = np.moveaxis(load_cum, 2, 0).reshape(blocks, K, R, bins * (N_BANDS + 1))
    block = block_of(minute_of_day.astype(np.int64), blocks)

    energy = np.full((K, R), b.e_initial)
    mw = np.zeros((K, R))
    net = np.zeros((K, R))
    paths = (np.empty((K, R, T)), np.empty((K, R, T))) if detail else None
    for t in range(T):
        energy_before = energy
        load_avail, gen_avail = max_avails(energy, mw, b, dt_h)
        offset = soc_bin(energy, b, bins) * (N_BANDS + 1)
        g = np.take_along_axis(gen_cum[block[t]], (offset + n_gen[t])[..., None], -1)[..., 0]
        l = np.take_along_axis(load_cum[block[t]], (offset + n_load[t])[..., None], -1)[..., 0]
        cleared = np.minimum(g, gen_avail) - np.minimum(l, load_avail)
        p0 = mw
        p1 = np.clip(cleared, np.maximum(p0 - ramp_step, -b.power_mw), np.minimum(p0 + ramp_step, b.power_mw))
        # trajectory_energies, vectorised
        cross = p0 * p1 < 0
        f = np.where(cross, p0 / np.where(cross, p0 - p1, 1.0), 0.0)
        pos = np.where(cross, np.where(p0 > 0, p0 * f, p1 * (1 - f)), np.maximum(p0 + p1, 0.0))
        neg = np.where(cross, np.where(p0 > 0, -p1 * (1 - f), -p0 * f), np.maximum(-(p0 + p1), 0.0))
        discharged, charged = pos / 2 * dt_h, neg / 2 * dt_h
        energy = energy - discharged / b.discharge_eff + charged * b.charge_eff
        step = (discharged - charged) * rrp[:, t] - (discharged + charged) * b.degradation_cost
        if valid is not None and not valid[:, t].all():
            v = valid[:, t]
            energy = np.where(v, energy, energy_before)
            p1 = np.where(v, p1, p0)
            step = np.where(v, step, 0.0)
        net += step
        mw = p1
        if detail:
            paths[0][..., t], paths[1][..., t] = p1, energy
    return (net, *paths) if detail else net


def interval_grid(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Interval ends in (start, end] and each interval's decision minute of day."""
    ends = pd.date_range(start + pd.Timedelta(minutes=5), end, freq="5min")
    starts = ends - pd.Timedelta(minutes=5)
    return ends, (starts.hour * 60 + starts.minute).to_numpy()
