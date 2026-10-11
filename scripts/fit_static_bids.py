"""Fit the static bid baselines (bess/static_bid.py) to dev prices with CMA-ES.

Each region gets its own bid; the five regions' CMA-ES runs step in lockstep so
every generation is one call to the vectorised simulator. Only dev data
(data/dev.duckdb) is used. Like the reference scores, the dev period is
simulated one calendar month at a time (each starting at 50%), every
(region, month) a lane of the vectorised simulator.

- static: 21 price parameters (gaps between the 20 band prices) + 20 band MW.
  Several starts, one seeded from a hand-made ladder; the best is kept.
- tod: the same 21 price parameters + 48 x 20 MW, started from the static fit
  with the same MW in every block (so it can only improve on it), using
  separable CMA-ES (diagonal covariance) for the dimension count.
- tod_soc: + 3 SOC bins, started from the tod fit the same way.

With `--validate`, each variant is also fitted on the dev months before
VALIDATION_START and scored on the dev year after it, to show how much it
overfits. Results go to
harness/static_bid_policies/<variant>/params.json, with a fit report.

`cma` isn't a project dependency (the agents' sandboxes share the project venv,
and the fit is a one-off), so run with:

    uv run --with cma==4.5.0 python scripts/fit_static_bids.py [--variant all] [--validate]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cma
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bess.config import Config, load_config  # noqa: E402
from bess.data import load_region  # noqa: E402
from bess.policies import PerfectForesight  # noqa: E402
from bess.sim import simulate, summarise  # noqa: E402
from bess.static_bid import (  # noqa: E402
    StaticBidParams, gaps_from_prices, interval_grid, prices_from_gaps, simulate_static,
)
from harness.evaluator import config_fingerprint  # noqa: E402
from harness.splits import DEV_END, DEV_MONTHS, REGIONS, Window  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEV_DB = ROOT / "data" / "dev.duckdb"
OUT_DIR = ROOT / "harness" / "static_bid_policies"
VALIDATION_START = pd.Timestamp("2024-09-01")
N_PRICE = 21
SHAPES = {"static": (1, 1), "tod": (48, 1), "tod_soc": (48, 3)}

# A hand-made starting ladder: charge below about $75, discharge above $100.
SEED_LOAD = [-900, -300, -100, -30, 0, 15, 30, 45, 60, 75]
SEED_GEN = [100, 130, 170, 220, 300, 500, 1000, 3000, 10000, 23000]


class Prices:
    """Actual prices for every region on a set of windows, one simulator lane
    per (region, window), region-major. Each window is padded at its front to
    the midnight before it, so all lanes share one time-of-day axis, and at its
    end to the longest window."""

    def __init__(self, data: dict, windows: list[Window]):
        self.windows = windows
        days = [w[0].normalize() for w in windows]
        offsets = [(w[0] - d) // pd.Timedelta(minutes=5) for w, d in zip(windows, days)]
        grids = [interval_grid(*w)[0] for w in windows]
        T = max(o + len(g) for o, g in zip(offsets, grids))
        self.minutes = (np.arange(T) * 5) % 1440
        self.rrp = np.full((len(REGIONS) * len(windows), T), np.nan)
        self.valid = np.zeros(self.rrp.shape, bool)
        for i, r in enumerate(REGIONS):
            for j, (o, g) in enumerate(zip(offsets, grids)):
                self.rrp[i * len(windows) + j, o : o + len(g)] = data[r].actual_rrp.loc[g].to_numpy()
                self.valid[i * len(windows) + j, o : o + len(g)] = True
        self.lane_region = np.repeat(np.arange(len(REGIONS)), len(windows))

    def simulate(self, params, config) -> np.ndarray:
        """Net revenue (K, R) of per-region candidates over all the windows."""
        lanes = [a[:, self.lane_region] for a in params]
        net = simulate_static(self.rrp, self.minutes, *lanes, config, valid=self.valid)
        return net.reshape(net.shape[0], len(REGIONS), len(self.windows)).sum(-1)


def decode(x: np.ndarray, shape: tuple[int, int], config: Config):
    """Candidate vectors (K, R, dim) -> simulator arrays."""
    K, R = x.shape[:2]
    load_p, gen_p = prices_from_gaps(x[..., :N_PRICE], config.market)
    mw = np.clip(x[..., N_PRICE:], 0, 1).reshape(K, R, *shape, 2, 10) * config.battery.power_mw
    mlf_l, mlf_g = config.market.load_mlf, config.market.gen_mlf
    return load_p * mlf_l, gen_p * mlf_g, mw[..., 0, :], mw[..., 1, :]


def to_params(x: np.ndarray, shape, config: Config) -> list[StaticBidParams]:
    lp, gp, lm, gm = decode(x[None], shape, config)
    return [StaticBidParams(lp[0, r], gp[0, r], lm[0, r], gm[0, r]) for r in range(len(REGIONS))]


def score(x: np.ndarray, shape, config: Config, prices: Prices) -> np.ndarray:
    """Net revenue per region (R,) of one candidate per region (R, dim)."""
    return prices.simulate(decode(x[None], shape, config), config)[0]


def cma_fit(x0: np.ndarray, shape, config: Config, prices: Prices, generations: int, popsize: int,
            separable: bool, seed: int, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Lockstep CMA-ES, one per region, maximising net revenue. Returns the
    best vector per region (R, dim) and its net revenue (R,)."""
    R, dim = x0.shape
    stds = np.r_[np.full(N_PRICE, 0.5), np.full(dim - N_PRICE, 0.15)]
    opts = {
        "bounds": [[-8] * N_PRICE + [0] * (dim - N_PRICE), [8] * N_PRICE + [1] * (dim - N_PRICE)],
        "CMA_stds": stds.tolist(), "popsize": popsize, "seed": seed, "verbose": -9,
        "CMA_diagonal": separable, "tolfun": 1e-3, "tolx": 1e-6,
    }
    strategies = [cma.CMAEvolutionStrategy(x0[r], 1.0, {**opts, "seed": seed + r}) for r in range(R)]
    best_x, best_f = x0.copy(), score(x0, shape, config, prices)
    t0 = time.time()
    for g in range(generations):
        asks = [np.asarray(es.ask()) for es in strategies]  # R x (popsize, dim)
        X = np.stack(asks, axis=1)  # (popsize, R, dim)
        net = prices.simulate(decode(X, shape, config), config)  # (popsize, R)
        for r, es in enumerate(strategies):
            if not es.stop():
                es.tell(list(asks[r]), (-net[:, r]).tolist())
            k = int(np.argmax(net[:, r]))
            if net[k, r] > best_f[r]:
                best_f[r], best_x[r] = net[k, r], X[k, r]
        if g % 25 == 0 or g == generations - 1:
            print(f"  {label} gen {g:4d}  {time.time() - t0:6.0f}s  best net "
                  + " ".join(f"{r}:{f:8.0f}" for r, f in zip(REGIONS, best_f)), flush=True)
        if all(es.stop() for es in strategies):
            break
    return best_x, best_f


def expand(x: np.ndarray, old_shape, new_shape) -> np.ndarray:
    """Copy each region's MW into every block and bin of a finer shape."""
    R = x.shape[0]
    mw = x[:, N_PRICE:].reshape(R, *old_shape, 2, 10)
    mw = np.broadcast_to(mw[:, :1, :1], (R, *new_shape, 2, 10)) if old_shape == (1, 1) else \
        np.broadcast_to(mw[:, :, :1], (R, *new_shape, 2, 10))
    return np.concatenate([x[:, :N_PRICE], mw.reshape(R, -1)], axis=1)


def fit_variant(variant: str, start_x, config, prices, args, seed: int) -> tuple[np.ndarray, np.ndarray]:
    shape = SHAPES[variant]
    if variant == "static":
        seeds = [np.r_[gaps_from_prices(SEED_LOAD, SEED_GEN, config.market), np.full(20, 0.1)]]
        rng = np.random.default_rng(seed)
        seeds += [np.r_[rng.normal(0, 1, N_PRICE), rng.uniform(0, 0.3, 20)] for _ in range(args.starts - 1)]
        best_x, best_f = None, None
        for i, s in enumerate(seeds):
            x, f = cma_fit(np.tile(s, (len(REGIONS), 1)), shape, config, prices, args.generations,
                           args.popsize, False, seed + 100 * i, f"static start {i}")
            if best_x is None:
                best_x, best_f = x, f
            else:
                better = f > best_f
                best_x[better], best_f[better] = x[better], f[better]
        return best_x, best_f
    old = "static" if variant == "tod" else "tod"
    x0 = expand(start_x, SHAPES[old], shape)
    return cma_fit(x0, shape, config, prices, args.generations_profile, args.popsize_profile, True, seed, variant)


def _perfect_one(args) -> float:
    config, region, start, end = args
    data = load_region(region, DEV_DB, start, end)
    return summarise(simulate(PerfectForesight(config, data.actual_rrp, start, end), data, config, start, end),
                     config.battery)["net_revenue"]


def perfect(config, windows: list[Window]) -> np.ndarray:
    """Perfect-foresight net revenue per region, summed over the windows."""
    from concurrent.futures import ProcessPoolExecutor

    tasks = [(config, r, a, b) for r in REGIONS for a, b in windows]
    with ProcessPoolExecutor(max(1, os.cpu_count() - 2)) as pool:
        nets = list(pool.map(_perfect_one, tasks))
    return np.asarray(nets).reshape(len(REGIONS), len(windows)).sum(-1)


def exact_check(params, config, data, windows: list[Window], shape) -> list[dict]:
    """Score the fitted policy with the real simulator (bess.sim) on a few
    windows, and the fast simulator on the same ones, which must agree."""
    from bess.static_bid import StaticBidPolicy

    out = []
    fast = Prices(data, windows)
    fast_net = fast.simulate([np.stack([getattr(p, k) for p in params])[None] for k in
                              ("load_prices", "gen_prices", "load_mw", "gen_mw")], config)[0]
    for i, (r, p) in enumerate(zip(REGIONS, params)):
        nets, ok = 0.0, True
        for start, end in windows:
            window_data = load_region(r, DEV_DB, start, end)
            s = summarise(simulate(StaticBidPolicy(config, p), window_data, config, start, end), config.battery)
            nets += s["net_revenue"]
            ok &= bool(s["valid"]) and s["nonconformance_intervals"] == 0 and s["rejected_bids"] == 0
        out.append({"net_revenue": nets, "fast_net_revenue": float(fast_net[i]), "valid": ok})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all", choices=["all", *SHAPES])
    ap.add_argument("--validate", action="store_true",
                    help=f"also fit on the dev months before {VALIDATION_START:%Y-%m} and score on the rest")
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--popsize", type=int, default=32)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--generations-profile", type=int, default=1500)
    ap.add_argument("--popsize-profile", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    config = load_config()
    # Actual prices only: the fast simulator needs nothing else (the exact
    # check loads its own windows).
    data = {r: load_region(r, DEV_DB, DEV_END, DEV_END) for r in REGIONS}
    variants = list(SHAPES) if args.variant == "all" else [args.variant]
    early = [w for w in DEV_MONTHS if w[1] <= VALIDATION_START]
    late = [w for w in DEV_MONTHS if w[0] >= VALIDATION_START]
    sets = {"dev": DEV_MONTHS, **({"early": early, "late": late} if args.validate else {})}
    prices = {name: Prices(data, w) for name, w in sets.items()}
    t0 = time.time()
    pf = {name: perfect(config, w) for name, w in sets.items()}
    print(f"perfect foresight: {time.time() - t0:.0f}s", flush=True)
    pct = lambda net, name: 100 * net.sum() / pf[name].sum()  # noqa: E731
    check_windows = [DEV_MONTHS[0], DEV_MONTHS[-1]]

    previous = {}
    for variant in variants:
        shape = SHAPES[variant]
        start = previous.get("static" if variant == "tod" else "tod")
        if start is None and variant != "static":
            start = _load_x(OUT_DIR / ("static" if variant == "tod" else "tod") / "params.json", config)
        report = {"variant": variant, "config_fingerprint": config_fingerprint(config), "shape": shape}

        vx = None
        if args.validate:
            print(f"== {variant}: fit before {VALIDATION_START:%Y-%m}, score after", flush=True)
            t0 = time.time()
            vx, vf = fit_variant(variant, start.get("early") if start else None, config, prices["early"], args,
                                 args.seed)
            report["validation"] = {
                "fit_seconds": time.time() - t0,
                "fit_until": str(VALIDATION_START),
                "fit_pct": pct(vf, "early"),
                "unseen_pct": pct(score(vx, shape, config, prices["late"]), "late"),
            }
            print(f"   fit {report['validation']['fit_pct']:.1f}%  unseen {report['validation']['unseen_pct']:.1f}%",
                  flush=True)

        print(f"== {variant}: fit on all dev", flush=True)
        t0 = time.time()
        x, f = fit_variant(variant, start["dev"] if start else None, config, prices["dev"], args, args.seed)
        params = to_params(x, shape, config)
        fit_seconds = time.time() - t0
        exact = exact_check(params, config, data, check_windows, shape)
        report.update({
            "fit_seconds": fit_seconds,
            "dev_pct": pct(f, "dev"),
            "dev_pct_by_region": {r: 100 * f[i] / pf["dev"][i] for i, r in enumerate(REGIONS)},
            "exact_check": {"windows": [[str(a), str(b)] for a, b in check_windows], **dict(zip(REGIONS, exact))},
        })
        assert all(e["valid"] for e in exact), exact
        assert np.allclose([e["net_revenue"] for e in exact], [e["fast_net_revenue"] for e in exact], atol=1e-6), exact
        print(f"   dev {report['dev_pct']:.1f}% of perfect foresight, fit {fit_seconds:.0f}s; "
              f"exact simulator agrees, valid", flush=True)

        out = OUT_DIR / variant
        out.mkdir(parents=True, exist_ok=True)
        (out / "params.json").write_text(json.dumps({
            "report": report,
            "x": {"dev": x.tolist(), **({"early": vx.tolist()} if vx is not None else {})},
            "regions": {r: p.to_dict() for r, p in zip(REGIONS, params)},
        }))
        previous[variant] = {"dev": x, "early": vx}


def _load_x(path: Path, config) -> dict:
    raw = json.loads(path.read_text())["x"]
    return {k: np.asarray(v) for k, v in raw.items()} | ({"early": None} if "early" not in raw else {})


if __name__ == "__main__":
    main()
