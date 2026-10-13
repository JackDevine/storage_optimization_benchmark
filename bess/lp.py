"""Price-taking battery arbitrage MILP, solved with HiGHS.

Variables per step k: charge c_k, discharge d_k (MW), end-of-step energy e_k
(MWh), and binary u_k (1 = charging allowed, 0 = discharging allowed).

    max  sum_k  dt_k * [ price_k * (d~_k - c~_k) - deg * (c~_k + d~_k) ]  -  penalty * s
    s.t. e_k = e_{k-1} + dt_k * (eta_c * c~_k - d~_k / eta_d)
         c_k <= P * u_k,   d_k <= P * (1 - u_k)
         0 <= c_k, d_k,   E_min <= e_k <= E_max
         |(d_k - c_k) - (d_{k-1} - c_{k-1})| <= ramp * step minutes
         e_N + s >= E_target,   s >= 0          (only when penalty > 0)
    plus V(e_N) for a concave terminal value of stored energy (optional, as
    piecewise-linear segments: e_N = grid_0 + sum_j z_j, 0 <= z_j <= width_j,
    each z_j worth its segment's slope)

c~ and d~ are the step's average flows. A 5-minute step follows AEMO's
straight-line path from the previous step's MW to this step's end-of-interval
target, so c~_k = (c_{k-1} + c_k) / 2 (the previous MW is `p_prev` for the
first step). That matches the simulator exactly unless the step crosses zero,
where it slightly overstates both flows; and the simulator also checks SOC at
the zero crossing, which the MILP doesn't see. `soc_margin_mwh` keeps planned
SOC that far inside its limits to absorb both (relaxed to the current SOC if
the battery is already inside the margin). Longer (predispatch) steps are
coarse planning steps, so their MW is held flat. Consecutive MW targets differ
by at most the battery's ramp rate × step length.

The binary stops the solver charging and discharging in the same step. Without
it, an LP burns energy through round-trip losses at negative prices.

Steps can have different lengths (5-min P5MIN steps then 30-min predispatch
steps).
"""

import highspy
import numpy as np

from bess.battery import Battery
from bess.config import TerminalCondition

FIVE_MIN_H = 5 / 60


def solve_schedule(
    prices: np.ndarray,
    dt_hours: np.ndarray,
    battery: Battery,
    e0: float,
    terminal: TerminalCondition,
    mip_rel_gap: float = 1e-4,
    p_prev: float = 0.0,
    soc_margin_mwh: float = 0.0,
    time_limit: float | None = None,
    integer: bool = True,
    terminal_value: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (charge_mw, discharge_mw, energy_mwh) per step. With `time_limit`
    (seconds), the best solution found by then is returned. `integer=False`
    relaxes the binaries to an LP (about 5x faster; it may then charge and
    discharge in the same step). `p_prev` is the
    unit's MW now (+ = discharge), where the first step's straight line starts.
    `terminal_value` is (energy grid, value) of the energy left at the end,
    concave in energy (bess.terminal.concave makes it so)."""
    n = len(prices)
    p, dt, b = np.asarray(prices, float), np.asarray(dt_hours, float), battery
    inf = highspy.kHighsInf
    soft_terminal = terminal.soc_shortfall_penalty > 0
    k = np.arange(n)
    C, D, E, U = 0, n, 2 * n, 3 * n  # column offsets
    num_col = 4 * n + int(soft_terminal)
    prev = float(p_prev)
    c_prev, d_prev = max(-prev, 0.0), max(prev, 0.0)

    # Weight on the previous step's MW in each step's average flow: half on
    # 5-minute (straight-line) steps, none on longer flat steps.
    w = np.where(dt <= FIVE_MIN_H + 1e-9, 0.5, 0.0)
    own = dt * (1 - w)  # each step's MW, in its own step
    nxt = np.append(dt[1:] * w[1:], 0.0)  # each step's MW, in the following step

    # HiGHS minimises.
    cost = np.concatenate([
        (p + b.degradation_cost) * own + (np.append(p[1:], 0.0) + b.degradation_cost) * nxt,
        (-p + b.degradation_cost) * own + (-np.append(p[1:], 0.0) + b.degradation_cost) * nxt,
        np.zeros(2 * n),
    ])
    e_lo = min(b.e_min + soc_margin_mwh, max(e0, b.e_min))
    e_hi = max(b.e_max - soc_margin_mwh, min(e0, b.e_max))
    col_lower = np.concatenate([np.zeros(2 * n), np.full(n, e_lo), np.zeros(n)])
    col_upper = np.concatenate([np.full(2 * n, b.power_mw), np.full(n, e_hi), np.ones(n)])
    integrality = np.concatenate([np.zeros(3 * n), np.full(n, 1 if integer else 0)])

    # Rows 0..n-1, energy balance:
    #   e_k - e_{k-1} + own_k*(d_k/eta_d - eta_c*c_k) + nxt_{k-1}*(d_{k-1}/eta_d - eta_c*c_{k-1}) = 0
    #   (for k = 0 the previous MW is the constant p_prev: moved to the right-hand side)
    # Rows n..2n-1:   c_k - P*u_k <= 0
    # Rows 2n..3n-1:  d_k + P*u_k <= P
    rows = [k, k, k, k[1:], n + k, n + k, 2 * n + k, 2 * n + k, k[1:], k[1:]]
    cols = [C + k, D + k, E + k, E + k[:-1], C + k, U + k, D + k, U + k, C + k[:-1], D + k[:-1]]
    vals = [
        -b.charge_eff * own,
        own / b.discharge_eff,
        np.ones(n),
        -np.ones(n - 1),
        np.ones(n),
        np.full(n, -b.power_mw),
        np.ones(n),
        np.full(n, b.power_mw),
        -b.charge_eff * nxt[:-1],
        nxt[:-1] / b.discharge_eff,
    ]
    row_lower = np.concatenate([np.zeros(n), np.full(2 * n, -inf)])
    row_upper = np.concatenate([np.zeros(n), np.zeros(n), np.full(n, b.power_mw)])
    first = e0 - dt[0] * w[0] * (d_prev / b.discharge_eff - b.charge_eff * c_prev)
    row_lower[0] = row_upper[0] = first

    # Rows 3n..4n-1, ramp: (d_k - c_k) - (d_{k-1} - c_{k-1}) within ±ramp × step minutes.
    step = b.ramp_rate_mw_per_min * dt * 60
    r = 3 * n + k
    rows += [r, r, r[1:], r[1:]]
    cols += [D + k, C + k, D + k[:-1], C + k[:-1]]
    vals += [np.ones(n), -np.ones(n), -np.ones(n - 1), np.ones(n - 1)]
    lo, hi = -step, step.copy()
    lo[0] += prev
    hi[0] += prev
    row_lower = np.concatenate([row_lower, lo])
    row_upper = np.concatenate([row_upper, hi])

    if soft_terminal:
        # Last row: e_{N-1} + s >= E_target, with shortfall s penalised.
        s, row = 4 * n, len(row_lower)
        target = min(max(terminal.soc_target_frac * b.energy_mwh, b.e_min), b.e_max)
        rows += [np.array([row, row])]
        cols += [np.array([E + n - 1, s])]
        vals += [np.ones(2)]
        row_lower = np.append(row_lower, target)
        row_upper = np.append(row_upper, inf)
        cost = np.append(cost, terminal.soc_shortfall_penalty)
        col_lower = np.append(col_lower, 0.0)
        col_upper = np.append(col_upper, inf)
        integrality = np.append(integrality, 0)

    if terminal_value is not None:
        # e_{N-1} - sum_j z_j = grid_0, each z_j in [0, width_j] earning its slope
        grid, value = (np.asarray(x, float) for x in terminal_value)
        width, slope = np.diff(grid), np.diff(value) / np.diff(grid)
        z0, row, m = num_col, len(row_lower), len(width)
        rows += [np.full(m + 1, row)]
        cols += [np.concatenate([[E + n - 1], z0 + np.arange(m)])]
        vals += [np.concatenate([[1.0], -np.ones(m)])]
        row_lower = np.append(row_lower, grid[0])
        row_upper = np.append(row_upper, grid[0])
        cost = np.append(cost, -slope)
        col_lower = np.append(col_lower, np.zeros(m))
        col_upper = np.append(col_upper, width)
        integrality = np.append(integrality, np.zeros(m))
        num_col += m

    rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    keep = vals != 0  # flat steps add explicit zeros for the previous-step terms
    rows, cols, vals = rows[keep], cols[keep], vals[keep]
    order = np.lexsort((rows, cols))  # CSC: sort by column, then row
    rows, cols, vals = rows[order], cols[order], vals[order]
    starts = np.searchsorted(cols, np.arange(num_col))

    lp = highspy.HighsLp()
    lp.num_col_ = num_col
    lp.num_row_ = len(row_lower)
    lp.col_cost_ = cost
    lp.col_lower_ = col_lower
    lp.col_upper_ = col_upper
    lp.row_lower_ = row_lower
    lp.row_upper_ = row_upper
    lp.integrality_ = [
        highspy.HighsVarType.kInteger if i else highspy.HighsVarType.kContinuous for i in integrality
    ]
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = np.append(starts, len(vals)).astype(np.int32)
    lp.a_matrix_.index_ = rows.astype(np.int32)
    lp.a_matrix_.value_ = vals

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("mip_rel_gap", mip_rel_gap)
    if time_limit is not None:
        h.setOptionValue("time_limit", float(time_limit))
    h.passModel(lp)
    h.run()
    status = h.getModelStatus()
    timed_out = status == highspy.HighsModelStatus.kTimeLimit and h.getInfo().primal_solution_status == 2
    if status != highspy.HighsModelStatus.kOptimal and not timed_out:
        raise RuntimeError(f"MILP not optimal: {h.modelStatusToString(status)}")

    x = np.asarray(h.getSolution().col_value)
    return x[C : C + n], x[D : D + n], x[E : E + n]
