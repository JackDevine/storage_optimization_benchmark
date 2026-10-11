"""Second reference policy: the forecast planner in the submission format.

Every 5 minutes, solve the arbitrage LP (the MILP with its binaries relaxed) on
the latest P5MIN + predispatch forecast and bid the first interval's MW as a
setpoint, passed through bess.bid.conform.
"""

from bess.policies import MILPForecastPolicy


class Policy:
    def __init__(self, config, region):
        self._inner = MILPForecastPolicy(config, integer=False)

    def decide(self, obs):
        return self._inner.decide(obs)
