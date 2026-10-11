"""Reference policy: the TWAP baseline in the submission format.

Every 5 minutes, take the time-weighted average price (TWAP) of the latest
P5MIN + predispatch forecast as the value of stored energy, and bid to charge
at full power below it and discharge at full power above it, by just enough to
cover round-trip losses and wear. Band prices are set once per trading day
around that day's thresholds. See bess.policies.TWAPPolicy.
"""

from bess.policies import TWAPPolicy


class Policy:
    def __init__(self, config, region):
        self._inner = TWAPPolicy(config)

    def decide(self, obs):
        return self._inner.decide(obs)
