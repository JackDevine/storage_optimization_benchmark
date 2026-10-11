"""Map historical prices onto today's market price cap (MPC).

The MPC has risen every July (from $15,100/MWh in October 2021). Older prices
are mapped so their cap lands on today's, changing ordinary prices as little as
possible: the identity up to a knee k, and above it

    g(p) = k + (MPC_now - k) * ((p - k) / (MPC_then - k)) ** gamma

which is monotone, continuous at k, and sends MPC_then to MPC_now exactly.
gamma = 1 stretches the tail linearly. Prices at or below the knee, including
negative prices and the market floor (unchanged at -$1,000 since 2021), are
left alone, as are prices in a period whose cap is already today's.

Forecast prices are mapped with the cap in force when the forecast was run, and
actual prices with the cap in force when the interval ended.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PriceMap:
    mpc_now: float
    knee: float = 300.0
    gamma: float = 1.0

    def __call__(self, price: np.ndarray, mpc_then: np.ndarray) -> np.ndarray:
        p = np.asarray(price, float)
        m = np.broadcast_to(np.asarray(mpc_then, float), p.shape)
        k = self.knee
        tail = (p > k) & (m != self.mpc_now)
        out = p.copy()
        x = np.clip((p[tail] - k) / (m[tail] - k), 0.0, None)
        out[tail] = k + (self.mpc_now - k) * x**self.gamma
        return out


def mpc_at(thresholds: pd.DataFrame, t) -> np.ndarray:
    """The MPC (VOLL) in force at each time in `t`: the latest version of the
    latest entry effective at or before it. `thresholds` is the
    market_price_thresholds table."""
    latest = thresholds.sort_values(["EFFECTIVEDATE", "VERSIONNO"]).drop_duplicates("EFFECTIVEDATE", keep="last")
    eff = latest.EFFECTIVEDATE.to_numpy("datetime64[ns]")
    i = np.searchsorted(eff, np.asarray(pd.to_datetime(t), "datetime64[ns]"), side="right") - 1
    if (i < 0).any():
        raise ValueError("a time before the first price threshold entry")
    return latest.VOLL.to_numpy()[i]


# The mapping the benchmark's data is built with (harness/build_data.py).
PRICE_MAP = PriceMap(mpc_now=23200.0, knee=300.0, gamma=1.0)
