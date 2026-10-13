"""The value of energy left in the battery at the end of a scoring window.

Scoring credits each window with the change in the value of stored energy,
V(e_end) - V(e_start), so a policy isn't rewarded for emptying the battery
just before an end it can't see, nor penalised for holding energy that is
worth something tomorrow. V is the same for every policy (and for perfect
foresight and the baselines): a fixed value-of-stored-energy dynamic program
over the forecast known at the window's end.

The DP: the latest P5MIN and predispatch runs at the end time, blended as in
bess.policies.blended_forecast and cut to TERMINAL_HORIZON_H hours. Backward
over the steps, V_k(e) = max over MW of the step's revenue less wear plus
V_{k+1}(next energy), with energy left at the horizon worth the forecast's
time-weighted average price x discharge efficiency (as TWAP values it). The
grid's value is then made concave, so it can enter perfect foresight's MILP.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from bess.battery import Battery
from bess.data import RegionData
from bess.policies import blended_forecast

TERMINAL_HORIZON_H = 24.0
GRID_POINTS, ACTIONS = 41, 21


def concave(grid: np.ndarray, value: np.ndarray) -> np.ndarray:
    """The least concave majorant of (grid, value), on the same grid."""
    hull = [0]
    for i in range(1, len(grid)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            # drop b if it lies on or below the chord from a to i
            if (value[b] - value[a]) * (grid[i] - grid[a]) <= (value[i] - value[a]) * (grid[b] - grid[a]):
                hull.pop()
            else:
                break
        hull.append(i)
    return np.interp(grid, grid[hull], value[hull])


def end_value(data: RegionData, tau: pd.Timestamp, battery: Battery) -> tuple[np.ndarray, np.ndarray]:
    """(energy grid, V) for energy held at `tau`, from the forecast known then."""
    b = battery
    obs = SimpleNamespace(tau=tau, p5min=data.view.p5min(tau), predispatch=data.view.predispatch(tau))
    prices, dt = blended_forecast(obs)
    keep = np.cumsum(dt) <= TERMINAL_HORIZON_H + 1e-9
    prices, dt = prices[keep], dt[keep]
    grid = np.linspace(b.e_min, b.e_max, GRID_POINTS)
    de = grid[1] - grid[0]
    acts = np.linspace(-b.power_mw, b.power_mw, ACTIONS)  # + = discharge
    twap = float(np.sum(prices * dt) / np.sum(dt))
    value = grid * twap * b.discharge_eff
    for p, h in zip(prices[::-1], dt[::-1]):
        delta = np.where(acts > 0, -acts / b.discharge_eff, -acts * b.charge_eff) * h
        nxt = grid[:, None] + delta[None, :]
        pos = (nxt - b.e_min) / de
        i0 = np.clip(np.floor(pos).astype(int), 0, GRID_POINTS - 2)
        w = np.clip(pos - i0, 0.0, 1.0)
        future = value[i0] + w * (value[i0 + 1] - value[i0])
        gain = (acts * p - np.abs(acts) * b.degradation_cost) * h
        ok = (nxt >= b.e_min - 1e-9) & (nxt <= b.e_max + 1e-9)
        value = np.where(ok, future + gain[None, :], -np.inf).max(axis=1)
    return grid, concave(grid, value)


def credit(terminal: tuple[np.ndarray, np.ndarray], e_end: float, e_start: float) -> float:
    """V(e_end) - V(e_start): the change in the stored energy's value."""
    grid, value = terminal
    return float(np.interp(e_end, grid, value) - np.interp(e_start, grid, value))
