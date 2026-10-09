from dataclasses import dataclass


@dataclass(frozen=True)
class Battery:
    """Battery spec. Deliberately small by default (1 MW / 2 MWh) so it's a
    price-taker: we don't model the battery's own effect on price."""

    power_mw: float = 1.0
    energy_mwh: float = 2.0
    charge_eff: float = 0.9  # one-way; round trip = charge_eff * discharge_eff
    discharge_eff: float = 0.9
    soc_min_frac: float = 0.05
    soc_max_frac: float = 0.95
    initial_soc_frac: float = 0.5
    # Throughput cost charged on both charge and discharge MWh, to stop the
    # optimiser cycling for negligible spreads.
    degradation_cost: float = 5.0
    # Used for the ramp rates in bids built by setpoint_bid.
    ramp_rate_mw_per_min: float = 1.0

    @property
    def e_min(self) -> float:
        return self.soc_min_frac * self.energy_mwh

    @property
    def e_max(self) -> float:
        return self.soc_max_frac * self.energy_mwh

    @property
    def e_initial(self) -> float:
        return self.initial_soc_frac * self.energy_mwh
