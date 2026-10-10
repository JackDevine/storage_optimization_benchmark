"""Benchmark configuration, loaded from a TOML file (see config/default.toml)."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from bess.battery import Battery

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.toml"


@dataclass(frozen=True)
class TerminalCondition:
    """Soft end-of-horizon SOC target. A zero penalty means no terminal
    condition: the optimiser is free to drain the battery at the horizon end."""

    soc_target_frac: float = 0.5
    soc_shortfall_penalty: float = 0.0  # $/MWh below target


@dataclass(frozen=True)
class Market:
    """Market rules applied to bids. Prices are $/MWh at the regional
    reference node; MLFs convert to and from connection-point bid prices."""

    mfp: float = -1000.0  # market floor price
    mpc: float = 23200.0  # market price cap (FY2026-27, matches the dispatch data)
    gen_mlf: float = 1.0
    load_mlf: float = 1.0
    # AEMO fixes band prices for the trading day: rebids may only move MW
    # between bands. When true, a bid that changes prices mid-day is rejected.
    lock_daily_prices: bool = True


# Market options that were removed when their behaviour became the only one: the
# straight-line dispatch trajectory, and ramping at the unit's own rate. Older
# config files and run records still carry them, so `true` is accepted and
# ignored; `false` asks for behaviour that no longer exists.
REMOVED_MARKET_OPTIONS = ("enforce_bid_ramp_rate", "linear_dispatch_trajectory")


def market_from_dict(d: dict) -> Market:
    d = dict(d)
    for key in REMOVED_MARKET_OPTIONS:
        if key in d and not d.pop(key):
            raise ValueError(f"market.{key} = false is no longer supported (the option was removed; it's always on)")
    return Market(**d)


@dataclass(frozen=True)
class Bidding:
    """Daily price bands ($/MWh, connection point) used by price-band bidding
    policies. AEMO fixes these for the trading day, so a fixed ladder spanning
    MFP to MPC stands in for re-pricing each interval. Load and gen prices
    interleave so the bid curve has 20 distinct price points."""

    load_prices: tuple[float, ...] = (-1000, -100, -20, 20, 60, 100, 170, 300, 1000, 10000)
    gen_prices: tuple[float, ...] = (-500, -50, 0, 40, 80, 130, 220, 500, 3000, 23200)


@dataclass(frozen=True)
class Config:
    name: str = "default"
    battery: Battery = field(default_factory=Battery)
    terminal: TerminalCondition = field(default_factory=TerminalCondition)
    market: Market = field(default_factory=Market)
    bidding: Bidding = field(default_factory=Bidding)
    mip_rel_gap: float = 1e-4


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)
    return Config(
        name=path.stem,
        battery=Battery(**raw.get("battery", {})),
        terminal=TerminalCondition(**raw.get("terminal", {})),
        market=market_from_dict(raw.get("market", {})),
        bidding=Bidding(**{k: tuple(v) for k, v in raw.get("bidding", {}).items()}),
        mip_rel_gap=raw.get("solver", {}).get("mip_rel_gap", 1e-4),
    )
