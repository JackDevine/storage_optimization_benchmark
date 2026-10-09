"""Bidirectional unit (BDU) energy bids, modelled on AEMO's IESS bid format.

Since IESS (June 2024) a grid battery is one DUID with one energy bid that has
a LOAD side and a GEN side. Each side has 10 price bands (fixed for the trading
day), and per 5-min interval: MW per band, MaxAvail, and ramp rates. AEMO's
documents disagree on whether load-side quantities are negative; here every
quantity is positive MW and the side says which direction it is.

AEMO validation rules implemented in `validate`:
- prices strictly increase from band 1 to 10 within each side
- connection-point prices lie in [MFP * MLF, MPC * MLF] for that side's MLF
- "effective" bands are the non-zero MW left after capping by MaxAvail,
  accumulated from band 10 down to 1 on the load side and from band 1 up to 10
  on the gen side
- MLF-adjusted (RRN) prices of effective bands strictly increase from load
  band 1 through load band 10 then gen band 1 through gen band 10 ("bid
  convexity"). This is what stops a BDU being dispatched to charge and
  discharge at once.
Not modelled: fixed load, PASA availability, the optional energy limit.

Sources: AEMO "Bidirectional unit price band bid validation" fact sheet
(May 2024) and "IESS BDU DUID bid format" (Dec 2023).
"""

import dataclasses
from dataclasses import dataclass

import numpy as np

from bess.battery import Battery
from bess.config import Market

N_BANDS = 10


@dataclass(frozen=True)
class SideBid:
    prices: tuple[float, ...]  # $/MWh at the connection point, band 1..10
    band_mw: tuple[float, ...]  # MW offered per band, band 1..10
    max_avail: float  # MW
    ramp_up_rate: float  # MW/min
    ramp_down_rate: float  # MW/min


@dataclass(frozen=True)
class BDUBid:
    load: SideBid
    gen: SideBid

    @property
    def price_bands(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return self.load.prices, self.gen.prices


def bid_to_dict(bid: BDUBid) -> dict:
    return dataclasses.asdict(bid)


def bid_from_dict(d: dict) -> BDUBid:
    def side(s: dict) -> SideBid:
        return SideBid(
            prices=tuple(float(x) for x in s["prices"]),
            band_mw=tuple(float(x) for x in s["band_mw"]),
            max_avail=float(s["max_avail"]),
            ramp_up_rate=float(s["ramp_up_rate"]),
            ramp_down_rate=float(s["ramp_down_rate"]),
        )

    return BDUBid(load=side(d["load"]), gen=side(d["gen"]))


def effective_load_mw(side: SideBid) -> np.ndarray:
    """MW per load band after capping by MaxAvail, accumulated from band 10 down."""
    return _cap(np.asarray(side.band_mw, float)[::-1], side.max_avail)[::-1]


def effective_gen_mw(side: SideBid) -> np.ndarray:
    """MW per gen band after capping by MaxAvail, accumulated from band 1 up."""
    return _cap(np.asarray(side.band_mw, float), side.max_avail)


def _cap(mw: np.ndarray, max_avail: float) -> np.ndarray:
    before = np.concatenate([[0.0], np.cumsum(mw)[:-1]])
    return np.clip(np.minimum(mw, max_avail - before), 0.0, None)


def validate(bid: BDUBid, battery: Battery, market: Market) -> list[str]:
    """Return the reasons AEMO would reject this bid (empty if valid)."""
    errors = []
    for name, side, mlf in [("load", bid.load, market.load_mlf), ("gen", bid.gen, market.gen_mlf)]:
        prices, mw = np.asarray(side.prices, float), np.asarray(side.band_mw, float)
        if len(prices) != N_BANDS or len(mw) != N_BANDS:
            errors.append(f"{name}: need {N_BANDS} price bands and {N_BANDS} band MW")
            continue
        if np.any(np.diff(prices) <= 0):
            errors.append(f"{name}: prices must strictly increase from band 1 to 10")
        if prices.min() < market.mfp * mlf or prices.max() > market.mpc * mlf:
            errors.append(f"{name}: prices must lie in [MFP*MLF, MPC*MLF]")
        if np.any(mw < 0) or np.any(mw > battery.power_mw + 1e-9):
            errors.append(f"{name}: band MW must be in [0, registered capacity]")
        if not 0 <= side.max_avail <= battery.power_mw + 1e-9:
            errors.append(f"{name}: MaxAvail must be in [0, registered capacity]")
        if side.ramp_up_rate <= 0 or side.ramp_down_rate <= 0:
            errors.append(f"{name}: ramp rates must be positive")

    if errors:
        return errors

    load_rrn = np.asarray(bid.load.prices) / market.load_mlf
    gen_rrn = np.asarray(bid.gen.prices) / market.gen_mlf
    load_eff = load_rrn[effective_load_mw(bid.load) > 0]
    gen_eff = gen_rrn[effective_gen_mw(bid.gen) > 0]
    if len(load_eff) and len(gen_eff) and load_eff.max() >= gen_eff.min():
        errors.append(
            "convexity: effective load band prices must all be below effective gen band prices (at RRN)"
        )
    return errors


def ramp_rate_mismatch(bid: BDUBid, battery: Battery) -> str | None:
    """A bid must state the unit's ramp rate; it can't choose its own. Returns
    why this bid doesn't, or None. The simulator still dispatches such a bid
    (at the unit's real ramp rate) but flags the run as non-conforming."""
    rate = battery.ramp_rate_mw_per_min
    wrong = [
        f"{name} {direction} {value:g}"
        for name, side in [("load", bid.load), ("gen", bid.gen)]
        for direction, value in [("up", side.ramp_up_rate), ("down", side.ramp_down_rate)]
        if abs(value - rate) > 1e-9
    ]
    if not wrong:
        return None
    return f"bid ramp rates ({', '.join(wrong)} MW/min) don't match the unit's ramp rate ({rate:g} MW/min)"


def dispatch_at_price(bid: BDUBid, rrp: float, market: Market) -> float:
    """Net MW (+ = discharge) a price-taking BDU is dispatched to at regional
    price `rrp`. Gen bands clear when their RRN price <= rrp, load bands when
    their RRN price >= rrp. A band priced exactly at rrp is fully dispatched,
    since a price-taker never sets the price."""
    gen_rrn = np.asarray(bid.gen.prices) / market.gen_mlf
    load_rrn = np.asarray(bid.load.prices) / market.load_mlf
    gen = effective_gen_mw(bid.gen)[gen_rrn <= rrp].sum()
    load = effective_load_mw(bid.load)[load_rrn >= rrp].sum()
    return float(gen - load)


def price_ladder(market: Market, mlf: float) -> tuple[float, ...]:
    """A fixed, valid price ladder from MFP to MPC (connection-point prices)."""
    return tuple(np.linspace(market.mfp * mlf, market.mpc * mlf, N_BANDS))


def setpoint_bid(mw: float, battery: Battery, market: Market) -> BDUBid:
    """A bid that is dispatched to `mw` whatever the price: discharge offered
    in gen band 1 at the market floor, charge offered in load band 10 at the
    market cap, and the other side's MaxAvail set to 0. Uses the same price
    ladder every interval, so it never breaks the daily price lock."""
    gen_mw, load_mw = max(mw, 0.0), max(-mw, 0.0)
    ramp = battery.ramp_rate_mw_per_min
    zeros = (0.0,) * N_BANDS
    return BDUBid(
        load=SideBid(
            prices=price_ladder(market, market.load_mlf),
            band_mw=zeros[:-1] + (load_mw,),
            max_avail=load_mw,
            ramp_up_rate=ramp,
            ramp_down_rate=ramp,
        ),
        gen=SideBid(
            prices=price_ladder(market, market.gen_mlf),
            band_mw=(gen_mw,) + zeros[1:],
            max_avail=gen_mw,
            ramp_up_rate=ramp,
            ramp_down_rate=ramp,
        ),
    )


def curve_bid(
    load_prices: tuple[float, ...],
    load_mw: np.ndarray,
    gen_prices: tuple[float, ...],
    gen_mw: np.ndarray,
    battery: Battery,
) -> BDUBid:
    """Build a bid from a response curve: the net MW (+ = discharge) the unit
    wants at each band's price, `load_mw[i]` at load_prices[i] and `gen_mw[i]`
    at gen_prices[i]. The curve must be non-decreasing in price across both
    sides together.

    Gen band i holds the extra discharge between gen price i-1 and i; load band
    i holds the extra charge between load price i+1 and i. Under price-taking
    dispatch the bid then clears to the curve's value at each band price.
    Because the curve is non-decreasing, MW sits in load bands only where the
    curve is negative and in gen bands only where it's positive, so the
    effective load bands are always priced below the effective gen bands.
    """
    discharge = np.maximum(np.asarray(gen_mw, float), 0.0)
    charge = np.maximum(-np.asarray(load_mw, float), 0.0)
    gen_bands = np.clip(np.diff(discharge, prepend=0.0), 0.0, None)
    load_bands = np.clip(charge - np.append(charge[1:], 0.0), 0.0, None)
    ramp = battery.ramp_rate_mw_per_min
    return BDUBid(
        load=SideBid(tuple(load_prices), tuple(load_bands), float(load_bands.sum()), ramp, ramp),
        gen=SideBid(tuple(gen_prices), tuple(gen_bands), float(gen_bands.sum()), ramp, ramp),
    )
