"""What a dispatch policy sees at each decision time.

Policies implement `decide(obs: Observation) -> BDUBid`. Everything in an
Observation was known at `obs.tau`: the battery's state, the latest P5MIN and
predispatch forecast runs published by then, and actual dispatch prices for
intervals that have already ended.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from bess.bid import BDUBid
from bess.data import Forecast


@dataclass
class Observation:
    tau: pd.Timestamp  # decision time; the bid is for the interval (tau, tau + 5min]
    region: str
    energy_mwh: float  # stored energy now
    last_mw: float  # MW dispatched in the interval that just ended (+ = discharge)
    # Price bands (load, gen) locked for the current trading day, or None
    # before the day's first accepted bid.
    locked_prices: tuple[tuple[float, ...], tuple[float, ...]] | None
    p5min: Forecast | None  # latest P5MIN run (5-min intervals, ~1h ahead)
    predispatch: Forecast | None  # latest predispatch run (30-min periods, to end of next trading day)
    history_end: np.ndarray  # interval-ending times (datetime64[ns]) of past actual prices
    history_rrp: np.ndarray  # actual dispatch prices for those intervals


class Policy(Protocol):
    name: str

    def decide(self, obs: Observation) -> BDUBid: ...
