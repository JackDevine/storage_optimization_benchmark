"""Static bid baseline (tod_soc: band MW by time of day and state of charge (3 bins)) in the submission format.

The band prices are fixed for all time and the band MW follow the fitted
profile; only each side's MaxAvail changes, to keep the battery inside its SOC
limits. Parameters are fitted to dev prices by scripts/fit_static_bids.py
(params.json). See bess/static_bid.py.
"""

from pathlib import Path

from bess.static_bid import StaticBidPolicy, load_params

PARAMS = load_params(Path(__file__).with_name("params.json"))


class Policy:
    def __init__(self, config, region):
        self._inner = StaticBidPolicy(config, PARAMS[region], name="static_bid_tod_soc")

    def decide(self, obs):
        return self._inner.decide(obs)
