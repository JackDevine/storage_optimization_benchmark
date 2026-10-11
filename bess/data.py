"""Load one region's actual and forecast prices from DuckDB, and expose the
forecasts through a ForecastView that only shows what was published at a
given decision time.

Timestamp conventions (AEMO labels intervals by their END time):
- dispatchprice.SETTLEMENTDATE = T means the 5-min interval (T-5, T].
- p5min RUN_DATETIME = R is the run executed at ~R-5min; its first interval
  ends at R. We treat it as available from R onward (one interval
  conservative), so at decision time tau the latest usable run is
  RUN_DATETIME <= tau, and we use its intervals ending after tau.
- predispatch has no clean run time, so we treat a run as available from its
  LASTCHANGED. DATETIME = P means the 30-min period (P-30, P].
"""

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Built by harness/build_data.py from the raw AEMO tables in data/benchmark.duckdb.
DB_PATH = PROJECT_ROOT / "data" / "market.duckdb"

INTERVAL = pd.Timedelta(minutes=5)
# How far before a window to load forecast runs, so the latest run before its
# first decision is there.
FORECAST_LOOKBACK = pd.Timedelta(days=1)


@dataclass
class Forecast:
    """A forecast run: when it became available, and its interval-ending
    times and prices ($/MWh at the regional reference node)."""

    run_time: np.datetime64
    interval_end: np.ndarray  # datetime64[ns]
    rrp: np.ndarray


class ForecastView:
    """Forecast access restricted to runs published at or before `tau`."""

    def __init__(self, p5min: pd.DataFrame, predispatch: pd.DataFrame):
        self._p5_runs, self._p5 = _group_runs(p5min, "RUN_DATETIME", "INTERVAL_DATETIME")
        self._pd_runs, self._pd = _group_runs(predispatch, "LASTCHANGED", "DATETIME")

    def p5min(self, tau: pd.Timestamp) -> Forecast | None:
        return _latest(self._p5_runs, self._p5, tau)

    def predispatch(self, tau: pd.Timestamp) -> Forecast | None:
        return _latest(self._pd_runs, self._pd, tau)


def _group_runs(df: pd.DataFrame, run_col: str, interval_col: str):
    df = df.sort_values([run_col, interval_col])
    runs = df[run_col].unique()
    forecasts = [
        Forecast(np.datetime64(run, "ns"), g[interval_col].to_numpy(), g["RRP"].to_numpy())
        for run, g in df.groupby(run_col, sort=True)
    ]
    return np.asarray(runs, dtype="datetime64[ns]"), forecasts


def _latest(runs: np.ndarray, forecasts: list[Forecast], tau: pd.Timestamp) -> Forecast | None:
    i = np.searchsorted(runs, np.datetime64(tau, "ns"), side="right") - 1
    return forecasts[i] if i >= 0 else None


@dataclass
class RegionData:
    region: str
    actual_rrp: pd.Series  # indexed by interval-ending time
    view: ForecastView

    def __post_init__(self):
        self._actual_end = self.actual_rrp.index.to_numpy(dtype="datetime64[ns]")
        self._actual_rrp = self.actual_rrp.to_numpy()

    def price_history(self, tau: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        """Actual prices for intervals ending at or before tau (array views)."""
        n = np.searchsorted(self._actual_end, np.datetime64(tau, "ns"), side="right")
        return self._actual_end[:n], self._actual_rrp[:n]


def load_region(
    region: str,
    db_path: Path = DB_PATH,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> RegionData:
    """One region's actual prices and forecast runs. With `start`/`end`, only
    the forecast runs a simulation of (start, end] can use are loaded (actual
    prices are always loaded in full, as they make up the price history)."""
    lo = pd.Timestamp.min if start is None else start - FORECAST_LOOKBACK
    hi = pd.Timestamp.max if end is None else end
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        actual = con.execute(
            """SELECT SETTLEMENTDATE, RRP FROM dispatchprice
               WHERE REGIONID = ? AND INTERVENTION = 0 ORDER BY SETTLEMENTDATE""",
            [region],
        ).df()
        p5min = con.execute(
            """SELECT RUN_DATETIME, INTERVAL_DATETIME, RRP FROM p5min_regionsolution
               WHERE REGIONID = ? AND INTERVENTION = 0 AND RUN_DATETIME > ? AND RUN_DATETIME <= ?""",
            [region, lo, hi],
        ).df()
        # A predispatch run's LASTCHANGED is the same for every period in it.
        predispatch = con.execute(
            """SELECT LASTCHANGED, DATETIME, RRP FROM predispatchprice
               WHERE REGIONID = ? AND INTERVENTION = 0 AND LASTCHANGED > ? AND LASTCHANGED <= ?""",
            [region, lo, hi],
        ).df()
    finally:
        con.close()

    return RegionData(
        region=region,
        actual_rrp=actual.set_index("SETTLEMENTDATE")["RRP"],
        view=ForecastView(p5min, predispatch),
    )
