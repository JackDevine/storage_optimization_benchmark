"""Reference policy: the milp_forecast baseline in the submission format.

Every 5 minutes, solve the arbitrage MILP on the latest P5MIN + predispatch
forecast and bid the first interval's MW as a setpoint (dispatched whatever
the price).
"""

from bess.policies import MILPForecastPolicy


class Policy:
    def __init__(self, config, region):
        self._inner = MILPForecastPolicy(config)

    def decide(self, obs):
        return self._inner.decide(obs)
