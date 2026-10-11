"""Run a policy in a separate process, talking JSON lines over stdin/stdout.

An agent-written policy must not share a Python process with the simulator,
or it could read actual prices out of the simulator's memory. RemotePolicy is
the simulator-side end: it sends each Observation (only what's new since the
last one) to a child process running `python -m bess.remote <policy.py>`,
which rebuilds the Observation, calls the policy, and sends back the bid. The
child can be wrapped in a sandbox command (see harness/sandbox.py).

The policy file must define `class Policy` with `__init__(self, config, region)`
and `decide(self, obs) -> BDUBid`.
"""

import dataclasses
import importlib.util
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from bess.battery import Battery
from bess.bid import BDUBid, bid_from_dict, bid_to_dict
from bess.config import Bidding, Config, TerminalCondition, market_from_dict
from bess.data import Forecast
from bess.observation import Observation


class RemotePolicyError(RuntimeError):
    pass


def config_to_dict(config: Config) -> dict:
    return dataclasses.asdict(config)


def config_from_dict(d: dict) -> Config:
    return Config(
        name=d["name"],
        battery=Battery(**d["battery"]),
        terminal=TerminalCondition(**d["terminal"]),
        market=market_from_dict(d["market"]),
        bidding=Bidding(**{k: tuple(v) for k, v in d["bidding"].items()}),
        mip_rel_gap=d["mip_rel_gap"],
    )


def _ns(t) -> int:
    return int(np.datetime64(t, "ns").astype(np.int64))


def _forecast_to_dict(f: Forecast) -> dict:
    return {
        "run_time": _ns(f.run_time),
        "interval_end": f.interval_end.astype("datetime64[ns]").astype(np.int64).tolist(),
        "rrp": f.rrp.tolist(),
    }


def _forecast_from_dict(d: dict) -> Forecast:
    return Forecast(
        run_time=np.datetime64(d["run_time"], "ns"),
        interval_end=np.asarray(d["interval_end"], dtype=np.int64).astype("datetime64[ns]"),
        rrp=np.asarray(d["rrp"], dtype=float),
    )


class RemotePolicy:
    """Simulator-side proxy for a policy running in a child process."""

    def __init__(
        self,
        policy_path: Path,
        config: Config,
        region: str,
        name: str = "remote",
        wrapper: list[str] | None = None,
        python: str = sys.executable,
        cwd: Path | None = None,
        stderr=None,
    ):
        self.name = name
        cmd = (wrapper or []) + [python, "-m", "bess.remote", str(policy_path)]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
            cwd=cwd,
        )
        self.exited = False  # the policy process went away mid-protocol
        self._p5_run = self._pd_run = None
        self._history_sent = 0
        self._call({"init": {"region": region, "config": config_to_dict(config)}})

    def _call(self, msg: dict) -> dict:
        try:
            self._proc.stdin.write(json.dumps(msg) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
        except BrokenPipeError:
            line = ""
        if not line:
            self.exited = True
            raise RemotePolicyError(f"policy process exited (code {self._proc.poll()})")
        reply = json.loads(line)
        if "error" in reply:
            raise RemotePolicyError(reply["error"])
        return reply

    def decide(self, obs: Observation) -> BDUBid:
        msg = {
            "tau": _ns(obs.tau),
            "energy_mwh": obs.energy_mwh,
            "last_mw": obs.last_mw,
            "locked_prices": obs.locked_prices,
        }
        # Forecast runs only change every 5 or 30 minutes; send them when they do.
        if obs.p5min is not None and obs.p5min.run_time != self._p5_run:
            msg["p5min"] = _forecast_to_dict(obs.p5min)
            self._p5_run = obs.p5min.run_time
        if obs.predispatch is not None and obs.predispatch.run_time != self._pd_run:
            msg["predispatch"] = _forecast_to_dict(obs.predispatch)
            self._pd_run = obs.predispatch.run_time
        n = len(obs.history_end)
        if n > self._history_sent:
            msg["history"] = {
                "interval_end": obs.history_end[self._history_sent :].astype(np.int64).tolist(),
                "rrp": obs.history_rrp[self._history_sent :].tolist(),
            }
            self._history_sent = n
        return bid_from_dict(self._call(msg)["bid"])

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.stdin.close()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()


class _GrowingArray:
    """Append-only numpy buffer so history slices stay cheap."""

    def __init__(self, dtype):
        self._buf = np.empty(1024, dtype=dtype)
        self._n = 0

    def extend(self, values) -> None:
        values = np.asarray(values, dtype=self._buf.dtype)
        while self._n + len(values) > len(self._buf):
            self._buf = np.resize(self._buf, 2 * len(self._buf))
        self._buf[self._n : self._n + len(values)] = values
        self._n += len(values)

    def view(self) -> np.ndarray:
        return self._buf[: self._n]


def _load_policy_class(path: Path):
    sys.path.insert(0, str(path.parent))  # let the policy import its own helper modules
    spec = importlib.util.spec_from_file_location("submitted_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Policy


def _serve(policy_path: Path) -> None:
    out = sys.stdout
    sys.stdout = sys.stderr  # keep the policy's prints off the protocol channel

    def send(msg: dict) -> None:
        out.write(json.dumps(msg) + "\n")
        out.flush()

    policy = region = None
    p5 = pd_run = None
    hist_end, hist_rrp = _GrowingArray("datetime64[ns]"), _GrowingArray(float)
    for line in sys.stdin:
        msg = json.loads(line)
        try:
            if "init" in msg:
                region = msg["init"]["region"]
                config = config_from_dict(msg["init"]["config"])
                policy = _load_policy_class(policy_path)(config, region)
                send({"ok": True})
                continue
            if "p5min" in msg:
                p5 = _forecast_from_dict(msg["p5min"])
            if "predispatch" in msg:
                pd_run = _forecast_from_dict(msg["predispatch"])
            if "history" in msg:
                hist_end.extend(np.asarray(msg["history"]["interval_end"], dtype=np.int64).astype("datetime64[ns]"))
                hist_rrp.extend(msg["history"]["rrp"])
            locked = msg["locked_prices"]
            obs = Observation(
                tau=pd.Timestamp(msg["tau"]),
                region=region,
                energy_mwh=msg["energy_mwh"],
                last_mw=msg["last_mw"],
                locked_prices=None if locked is None else (tuple(locked[0]), tuple(locked[1])),
                p5min=p5,
                predispatch=pd_run,
                history_end=hist_end.view(),
                history_rrp=hist_rrp.view(),
            )
            send({"bid": bid_to_dict(policy.decide(obs))})
        except Exception:
            send({"error": traceback.format_exc(limit=5)})


if __name__ == "__main__":
    _serve(Path(sys.argv[1]).resolve())
