"""Do the new forecast-feature tables (regional demand and availability from
P5MIN and predispatch, ST PASA reserve, P5MIN price sensitivities) improve
price forecasts beyond the price forecasts themselves?

Stage 1, `build`: a sample of decision times (hourly, every region, dev data
only). At each, for a set of lead times, everything known then: the latest
P5MIN or predispatch price for the target, recent actual prices, the latest
regional summary for the target, the latest ST PASA run, and (from August
2024) the latest P5MIN price sensitivities. The target is the actual price
(5-minute for P5MIN leads, the 30-minute mean for predispatch leads).
Prices are the benchmark's cap-normalised ones (data/market.duckdb); the
feature tables come from data/benchmark.duckdb.

    uv run --no-project --with duckdb --with pandas --with pyarrow python scripts/forecast_feature_test.py build

Writes data/forecast_features/{p5,pd}.parquet.

Stage 2, `fit`: LightGBM models of the actual price, per forecast family
(P5MIN, predispatch), on nested feature sets:
- aemo: the AEMO forecast itself, no model
- A: price information only (the forecast, lead, time of day/week/year,
  region, recent actual prices, the predispatch path's next 6 hours)
- B: A plus the regional summaries (demand, availability, margin,
  interchange, wind and solar) and ST PASA (reserve, demand spread, LOR)
- S: B plus the P5MIN price sensitivities (P5MIN only, from August 2024)
Scored on two splits of dev: chronological (train to Aug 2024, test Sep 2024 -
Aug 2025) and interleaved months (5 folds by month).

    uv run --no-project --with lightgbm --with scikit-learn --with pandas --with pyarrow python scripts/forecast_feature_test.py fit

Writes data/forecast_features/fit_results.json.

Stage 3, `predict` then `dispatch`: does a better forecast earn more? Models A
and B are trained on the chronological training years and predict every
forecast step at every 5-minute decision in the dispatch-test episodes
(`build-dispatch` builds those rows). The same value-of-stored-energy policy
then runs on each forecast (the AEMO forecast, A, B, and the actual prices as
a perfect forecast): a backward DP over stored energy to 24 hours, bidding to
charge below and discharge above the marginal value of energy, net of losses
and $30/MWh wear. The battery holds a flat MW each interval (the same
simplified physics for every forecast); perfect foresight is the LP on the
actual prices. `predict` runs in the LightGBM environment, `dispatch` in the
project's.

    uv run --no-project --with duckdb --with pandas --with pyarrow python scripts/forecast_feature_test.py build-dispatch
    uv run --no-project --with lightgbm --with pandas --with pyarrow python scripts/forecast_feature_test.py predict
    uv run python scripts/forecast_feature_test.py dispatch

Writes data/forecast_features/dispatch_results.json.
"""

import sys
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "forecast_features"
DEV_START, DEV_END = "2021-10-02 04:00", "2025-09-01 00:00"
EXCLUDED = [("2022-06-12 04:00", "2022-06-24 04:00"), ("2022-10-01 00:00", "2022-11-01 04:05")]
P5_LEADS_MIN = [5, 15, 30, 60]
PD_LEADS_MIN = [60, 180, 360, 720, 1440]
# dispatch test: 3-day episodes in the chronological test year, two per month and region
DISPATCH_YEAR = ("2024-09-01", "2025-09-01")
EPISODES_PER_MONTH, EPISODE_DAYS, SEED = 2, 3, 7
HORIZON_MIN = 24 * 60
REGIONSUM_COLS = ["TOTALDEMAND", "AVAILABLEGENERATION", "NETINTERCHANGE", "SS_SOLAR_UIGF", "SS_WIND_UIGF"]


def episodes() -> list[tuple[str, str]]:
    """(start, end) of the dispatch-test episodes: trading-day starts drawn per
    month, the same for every region."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(SEED)
    out = []
    for m in pd.date_range(*DISPATCH_YEAR, freq="MS", inclusive="left"):
        days = pd.date_range(m, m + pd.offsets.MonthEnd(0) - pd.Timedelta(days=EPISODE_DAYS), freq="D")
        # one start in each equal part of the month, at least EPISODE_DAYS apart, so episodes never overlap
        parts = np.array_split(np.arange(len(days)), EPISODES_PER_MONTH)
        for part in parts:
            start = days[rng.choice(part[: max(1, len(part) - EPISODE_DAYS + 1)])] + pd.Timedelta(hours=4)
            out.append((str(start), str(start + pd.Timedelta(days=EPISODE_DAYS))))
    return out


def build(kind: str = "train") -> None:
    """kind "train": hourly decision times over dev, sampled leads. kind
    "dispatch": every 5-minute decision in the dispatch-test episodes, every
    forecast step to HORIZON_MIN."""
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"ATTACH '{ROOT / 'data' / 'market.duckdb'}' AS m (READ_ONLY)")
    con.execute(f"ATTACH '{ROOT / 'data' / 'benchmark.duckdb'}' AS b (READ_ONLY)")
    con.execute("SET preserve_insertion_order = false")
    excl = " AND ".join(f"NOT (tau > TIMESTAMP '{a}' AND tau <= TIMESTAMP '{b}' + INTERVAL 2 DAY)" for a, b in EXCLUDED)
    t0 = time.time()

    def step(msg):
        print(f"{time.time() - t0:5.0f}s {msg}", flush=True)

    regions = "(SELECT unnest(['NSW1','QLD1','SA1','TAS1','VIC1']) r)"
    if kind == "train":  # decision times: hourly, every region
        con.execute(f"""CREATE TEMP TABLE taus AS
            SELECT r, tau FROM {regions},
                 (SELECT unnest(range(TIMESTAMP '{DEV_START}', TIMESTAMP '{DEV_END}' - INTERVAL 1 DAY, INTERVAL 1 HOUR)) tau)
            WHERE {excl}""")
        p5_steps = f"p.INTERVAL_DATETIME = x.tau + to_minutes(lead_min)"
        p5_x = f"(SELECT * FROM taus, (SELECT unnest([{', '.join(map(str, P5_LEADS_MIN))}]) lead_min))"
        pd_steps = ("p.DATETIME = time_bucket(INTERVAL 30 MINUTE, x.tau + to_minutes(lead_min) - INTERVAL 1 SECOND)"
                    " + INTERVAL 30 MINUTE")
        pd_x = f"(SELECT * FROM taus, (SELECT unnest([{', '.join(map(str, PD_LEADS_MIN))}]) lead_min))"
        p5_lead = pd_lead = "lead_min"
    else:  # every 5 minutes in each episode, every step of the latest runs
        eps = " UNION ALL ".join(
            f"SELECT {i} episode, unnest(range(TIMESTAMP '{a}', TIMESTAMP '{b}', INTERVAL 5 MINUTE)) tau"
            for i, (a, b) in enumerate(episodes()))
        con.execute(f"CREATE TEMP TABLE taus AS SELECT r, episode, tau FROM {regions}, ({eps})")
        p5_steps = "p.INTERVAL_DATETIME > x.tau AND p.INTERVAL_DATETIME <= x.tau + INTERVAL 60 MINUTE"
        pd_steps = f"p.DATETIME > x.tau + INTERVAL 60 MINUTE AND p.DATETIME <= x.tau + INTERVAL {HORIZON_MIN} MINUTE"
        p5_x = pd_x = "taus"
        p5_lead = "CAST(epoch(p.INTERVAL_DATETIME - x.tau) / 60 AS INTEGER) lead_min"
        pd_lead = "CAST(epoch(p.DATETIME - x.tau) / 60 AS INTEGER) lead_min"
    prefix = "" if kind == "train" else "dispatch_"
    con.execute("""CREATE TEMP TABLE act AS SELECT REGIONID r, SETTLEMENTDATE t, RRP a FROM m.dispatchprice WHERE INTERVENTION = 0""")
    # recent actual prices at tau
    con.execute("""CREATE TEMP TABLE recent AS
        SELECT x.r, x.tau,
          (SELECT a FROM act WHERE act.r = x.r AND act.t = x.tau) last_rrp,
          (SELECT avg(a) FROM act WHERE act.r = x.r AND act.t > x.tau - INTERVAL 1 HOUR AND act.t <= x.tau) mean_rrp_1h,
          (SELECT avg(least(a, 1000)) FROM act WHERE act.r = x.r AND act.t > x.tau - INTERVAL 1 DAY AND act.t <= x.tau) mean_rrp_24h,
          (SELECT max(a) FROM act WHERE act.r = x.r AND act.t > x.tau - INTERVAL 1 DAY AND act.t <= x.tau) max_rrp_24h
        FROM taus x""")
    step("recent prices")
    # current actual regional balance at tau
    con.execute("""CREATE TEMP TABLE now_rs AS
        SELECT x.r, x.tau, d.TOTALDEMAND now_demand, d.AVAILABLEGENERATION - d.TOTALDEMAND now_margin
        FROM taus x LEFT JOIN b.dispatchregionsum d ON d.REGIONID = x.r AND d.SETTLEMENTDATE = x.tau AND d.INTERVENTION = 0""")

    # latest runs known at tau (P5MIN is usable 5 minutes before its run time)
    con.execute("""CREATE TEMP TABLE pd_runs AS SELECT DISTINCT REGIONID r, LASTCHANGED run FROM m.predispatchprice""")
    con.execute("""CREATE TEMP TABLE pdrs_runs AS SELECT DISTINCT REGIONID r, LASTCHANGED run FROM b.predispatchregionsum""")
    con.execute("""CREATE TEMP TABLE p5_runs AS SELECT DISTINCT REGIONID r, RUN_DATETIME run FROM m.p5min_regionsolution""")
    con.execute("""CREATE TEMP TABLE st_runs AS SELECT DISTINCT REGIONID r, LASTCHANGED pub, RUN_DATETIME run FROM b.stpasa_regionsolution""")
    con.execute("""CREATE TEMP TABLE se_runs AS SELECT DISTINCT REGIONID r, RUN_DATETIME run FROM b.p5min_pricesensitivities""")
    con.execute("""CREATE TEMP TABLE latest AS
        SELECT x.r, x.tau, pd.run pd_run, pdrs.run pdrs_run, p5.run p5_run, st.run st_run, se.run se_run
        FROM taus x
        ASOF LEFT JOIN pd_runs pd ON pd.r = x.r AND x.tau >= pd.run
        ASOF LEFT JOIN pdrs_runs pdrs ON pdrs.r = x.r AND x.tau >= pdrs.run
        ASOF LEFT JOIN p5_runs p5 ON p5.r = x.r AND x.tau + INTERVAL 5 MINUTE >= p5.run
        ASOF LEFT JOIN st_runs st ON st.r = x.r AND x.tau >= st.pub
        ASOF LEFT JOIN se_runs se ON se.r = x.r AND x.tau + INTERVAL 5 MINUTE >= se.run""")
    step("latest runs")
    # the predispatch price path's next-6-hour max, as context for every row
    con.execute("""CREATE TEMP TABLE pd_ctx AS
        SELECT l.r, l.tau, max(p.RRP) pd_max6h, avg(least(p.RRP, 1000)) pd_mean6h
        FROM latest l JOIN m.predispatchprice p ON p.REGIONID = l.r AND p.LASTCHANGED = l.pd_run AND p.INTERVENTION = 0
          AND p.DATETIME > l.tau AND p.DATETIME <= l.tau + INTERVAL 6 HOUR
        GROUP BY ALL""")
    # ST PASA: tightest surplus reserve over the run's next 24 hours from tau
    con.execute("""CREATE TEMP TABLE st_ctx AS
        SELECT l.r, l.tau, min(s.SURPLUSRESERVE) st_min_reserve_24h, max(s.LORCONDITION) st_max_lor_24h
        FROM latest l JOIN b.stpasa_regionsolution s ON s.REGIONID = l.r AND s.RUN_DATETIME = l.st_run
          AND s.INTERVAL_DATETIME > l.tau AND s.INTERVAL_DATETIME <= l.tau + INTERVAL 24 HOUR
        GROUP BY ALL""")
    step("context")

    rs = ", ".join(f"rs.{c} f_{c.lower()}" for c in REGIONSUM_COLS)
    st = ("st.SURPLUSRESERVE st_reserve, st.DEMAND50 st_demand50, st.DEMAND90 - st.DEMAND10 st_demand_spread, "
          "st.LORCONDITION st_lor, st.AGGREGATECAPACITYAVAILABLE - st.DEMAND50 st_margin")
    common = """x.r, x.tau, rec.last_rrp, rec.mean_rrp_1h, rec.mean_rrp_24h, rec.max_rrp_24h,
        now.now_demand, now.now_margin, ctx.pd_max6h, ctx.pd_mean6h, sc.st_min_reserve_24h, sc.st_max_lor_24h"""
    joins = """JOIN latest l USING (r, tau) LEFT JOIN recent rec USING (r, tau) LEFT JOIN now_rs now USING (r, tau)
        LEFT JOIN pd_ctx ctx USING (r, tau) LEFT JOIN st_ctx sc USING (r, tau)"""

    # ST PASA starts about a day ahead, so the latest run covering a target is
    # usually yesterday's: join each row to the latest one published by tau
    con.execute("""CREATE TEMP TABLE st_periods AS
        SELECT REGIONID r, INTERVAL_DATETIME period, LASTCHANGED pub, SURPLUSRESERVE, DEMAND50, DEMAND10, DEMAND90,
               LORCONDITION, AGGREGATECAPACITYAVAILABLE FROM b.stpasa_regionsolution""")

    def with_st(select):
        return f"""SELECT q.*, {st} FROM ({select}) q
            ASOF LEFT JOIN st_periods st ON st.r = q.r
              AND st.period = time_bucket(INTERVAL 30 MINUTE, q.target - INTERVAL 5 MINUTE) + INTERVAL 30 MINUTE
              AND q.tau >= st.pub"""

    # P5MIN rows
    sens = ", ".join(f"se.RRP{i}" for i in range(1, 44))
    con.execute(f"""COPY ({with_st(f'''
        SELECT {common}, {p5_lead}, p.INTERVAL_DATETIME AS target, p.RRP fc,
          {rs}, rs.AVAILABLEGENERATION - rs.TOTALDEMAND f_margin,
          greatest({sens}) se_max, least({sens}) se_min,
          act.a actual
        FROM {p5_x} x {joins}
        JOIN m.p5min_regionsolution p ON p.REGIONID = x.r AND p.RUN_DATETIME = l.p5_run AND p.INTERVENTION = 0
          AND {p5_steps}
        LEFT JOIN b.p5min_regionsum rs ON rs.REGIONID = x.r AND rs.RUN_DATETIME = l.p5_run AND rs.INTERVENTION = 0
          AND rs.INTERVAL_DATETIME = p.INTERVAL_DATETIME
        LEFT JOIN b.p5min_pricesensitivities se ON se.REGIONID = x.r AND se.RUN_DATETIME = l.se_run AND se.INTERVENTION = 0
          AND se.INTERVAL_DATETIME = p.INTERVAL_DATETIME
        JOIN act ON act.r = x.r AND act.t = p.INTERVAL_DATETIME''')}
    ) TO '{OUT / f"{prefix}p5.parquet"}' (FORMAT parquet)""")
    step("p5 rows")

    # predispatch rows: the 30-minute periods
    con.execute(f"""COPY ({with_st(f'''
        SELECT {common}, {pd_lead}, p.DATETIME AS target, p.RRP fc,
          {rs}, rs.AVAILABLEGENERATION - rs.TOTALDEMAND f_margin,
          (SELECT avg(a) FROM act WHERE act.r = x.r AND act.t > p.DATETIME - INTERVAL 30 MINUTE AND act.t <= p.DATETIME) actual
        FROM {pd_x} x {joins}
        JOIN m.predispatchprice p ON p.REGIONID = x.r AND p.LASTCHANGED = l.pd_run AND p.INTERVENTION = 0
          AND {pd_steps}
        LEFT JOIN b.predispatchregionsum rs ON rs.REGIONID = x.r AND rs.LASTCHANGED = l.pdrs_run AND rs.INTERVENTION = 0
          AND rs.DATETIME = p.DATETIME''')}
    ) TO '{OUT / f"{prefix}pd.parquet"}' (FORMAT parquet)""")
    step("pd rows")
    for f in ("p5", "pd"):
        n, cov = con.execute(f"SELECT count(*), avg((f_totaldemand IS NOT NULL)::INT) FROM '{OUT / (prefix + f)}.parquet'").fetchone()
        print(f"{f}: {n:,} rows, regional summary present in {cov:.0%}")


BASE = ["fc", "lead", "hour", "dow", "doy_sin", "doy_cos", "region", "last_rrp", "mean_rrp_1h", "mean_rrp_24h",
        "max_rrp_24h", "pd_max6h", "pd_mean6h"]
NEW = ["f_totaldemand", "f_availablegeneration", "f_netinterchange", "f_ss_solar_uigf", "f_ss_wind_uigf", "f_margin",
       "now_demand", "now_margin", "st_reserve", "st_demand50", "st_demand_spread", "st_lor", "st_margin",
       "st_min_reserve_24h", "st_max_lor_24h"]
SENS_F = ["se_max", "se_min", "se_spread"]
SPIKE = 300.0


def prepare(family: str):
    import numpy as np
    import pandas as pd
    d = pd.read_parquet(OUT / f"{family}.parquet")
    d["lead"] = d.pop("lead_min")
    t = pd.to_datetime(d.target)
    d["hour"] = t.dt.hour + t.dt.minute / 60
    d["dow"] = t.dt.dayofweek
    doy = 2 * np.pi * t.dt.dayofyear / 365.25
    d["doy_sin"], d["doy_cos"] = np.sin(doy), np.cos(doy)
    d["region"] = d.r.astype("category")
    if "se_max" in d:
        d["se_spread"] = d.se_max - d.se_min
    d["month"] = t.dt.year * 12 + t.dt.month
    return d.dropna(subset=["actual", "fc"]).reset_index(drop=True)


def fit() -> None:
    import json

    import lightgbm as lgb
    import numpy as np
    from sklearn.metrics import roc_auc_score

    def metrics(y, p):
        spike = y > SPIKE
        return {"mae": float(np.mean(np.abs(y - p))), "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
                "mae_clip1000": float(np.mean(np.abs(np.clip(y, -1000, 1000) - np.clip(p, -1000, 1000)))),
                "spike_auc": float(roc_auc_score(spike, p)) if 0 < spike.sum() < len(y) else None,
                "spike_share": float(spike.mean()), "n": int(len(y))}

    params = PARAMS
    results = {}
    for family in ("p5", "pd"):
        d = prepare(family)
        sets = {"A": BASE, "B": BASE + NEW}
        chrono_test = d.target >= "2024-09-01"
        folds = {"chronological": [(~chrono_test, chrono_test)],
                 "interleaved": [((d.month % 5) != k, (d.month % 5) == k) for k in range(5)]}
        if family == "p5":  # sensitivities exist from August 2024: interleaved months within that year only
            has = d.se_max.notna()
            sets["B (sens. year)"] = BASE + NEW
            sets["S (sens. year)"] = BASE + NEW + SENS_F
        for split, fl in folds.items():
            for name, cols in sets.items():
                if "sens. year" in name and split == "chronological":
                    continue
                y_all, p_all, f_all = [], [], []
                for tr, te in fl:
                    if "sens. year" in name:
                        tr, te = tr & has, te & has
                    m = lgb.train(params, lgb.Dataset(d.loc[tr, cols], d.loc[tr, "actual"]), num_boost_round=600)
                    y_all.append(d.loc[te, "actual"].to_numpy()); p_all.append(m.predict(d.loc[te, cols]))
                    f_all.append(d.loc[te, "fc"].to_numpy())
                y, p, f = map(np.concatenate, (y_all, p_all, f_all))
                key = f"{family} | {split}"
                results.setdefault(key, {})
                if "sens. year" not in name or "aemo (sens. year)" not in results[key]:
                    results[key]["aemo" + (" (sens. year)" if "sens. year" in name else "")] = metrics(y, f)
                results[key][name] = metrics(y, p)
                if name == "B" and split == "chronological":
                    imp = sorted(zip(cols, m.feature_importance("gain")), key=lambda x: -x[1])
                    tot = sum(v for _, v in imp)
                    results[key]["importance_B"] = {c: round(v / tot, 3) for c, v in imp[:15]}
                print(f"{key:28s} {name:16s}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics(y, p).items()}, flush=True)
    (OUT / "fit_results.json").write_text(json.dumps(results, indent=1))


PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=16)


def predict() -> None:
    import lightgbm as lgb
    import pandas as pd
    out = []
    for family in ("p5", "pd"):
        train = prepare(family)
        train = train[train.target < DISPATCH_YEAR[0]]
        d = prepare(f"dispatch_{family}")
        for name, cols in {"A": BASE, "B": BASE + NEW}.items():
            m = lgb.train(PARAMS, lgb.Dataset(train[cols], train["actual"]), num_boost_round=600)
            d[name] = m.predict(d[cols])
        d["dt_family"] = family
        out.append(d[["r", "tau", "target", "lead", "dt_family", "fc", "A", "B", "actual"]])
        print(family, len(d), flush=True)
    pd.concat(out).to_parquet(OUT / "dispatch_preds.parquet")


def dispatch() -> None:
    import json

    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(ROOT))
    from bess.config import TerminalCondition, load_config
    from bess.lp import solve_schedule

    b = load_config().battery
    G, A = 41, 21
    grid = np.linspace(b.e_min, b.e_max, G)
    de = grid[1] - grid[0]
    acts = np.linspace(-b.power_mw, b.power_mw, A)  # + = discharge
    H5 = 5 / 60

    def thresholds(v):
        c, d = b.charge_eff * v - b.degradation_cost, v / b.discharge_eff + b.degradation_cost
        mid = (c + d) / 2
        return np.where(c > d, mid, c), np.where(c > d, mid, d)

    def values(prices, dt):
        """Value of stored energy after the first step, for a batch of
        forecasts (B, H) with step lengths dt (B, H) in hours, 0 = padding."""
        Bn, H = prices.shape
        twap = np.nansum(np.nan_to_num(prices) * dt, 1) / np.maximum(dt.sum(1), 1e-9)
        value = grid[None, :] * twap[:, None] * b.discharge_eff
        for j in range(H - 1, 0, -1):
            p, h = np.nan_to_num(prices[:, j]), dt[:, j]
            delta = np.where(acts > 0, -acts / b.discharge_eff, -acts * b.charge_eff)[None, None, :] * h[:, None, None]
            nxt = grid[None, :, None] + delta
            pos = (nxt - b.e_min) / de
            i0 = np.clip(np.floor(pos).astype(int), 0, G - 2)
            w = np.clip(pos - i0, 0, 1)
            v0 = np.take_along_axis(value[:, :, None].repeat(A, 2), i0, 1)
            v1 = np.take_along_axis(value[:, :, None].repeat(A, 2), i0 + 1, 1)
            gain = (acts[None, :] * p[:, None] - np.abs(acts)[None, :] * b.degradation_cost) * h[:, None]
            ok = (nxt >= b.e_min - 1e-9) & (nxt <= b.e_max + 1e-9)
            total = np.where(ok, v0 + w * (v1 - v0) + gain[:, None, :], -np.inf).max(2)
            value = np.where((h > 0)[:, None], total, value)
        return value

    def run_episode(df, arm):
        """df: one region-episode's forecast rows. Returns net revenue."""
        df = df.sort_values(["tau", "target"])
        taus = np.sort(df.tau.unique())
        g = df.groupby("tau", sort=True)
        H = int(g.size().max())
        prices, dt = np.full((len(taus), H), np.nan), np.zeros((len(taus), H))
        for k, (tau, x) in enumerate(g):
            ends = x.target.to_numpy()
            starts = np.r_[np.datetime64(tau, "ns"), ends[:-1]]
            prices[k, :len(x)] = x[arm].to_numpy()
            dt[k, :len(x)] = (ends - starts) / np.timedelta64(1, "h")
        value = values(prices, dt)
        actual = g.first().actual.to_numpy()  # the first step is the interval being bid for
        e, net = b.e_initial, 0.0
        for k in range(len(taus)):
            up, down = min(e + b.power_mw * H5 * b.charge_eff, b.e_max), max(e - b.power_mw * H5 / b.discharge_eff, b.e_min)
            v = np.interp([down, e, up], grid, value[k])
            vc = (v[2] - v[1]) / (up - e) if up > e + 1e-9 else np.inf
            vd = (v[1] - v[0]) / (e - down) if e > down + 1e-9 else -np.inf
            charge = thresholds(vc)[0] if np.isfinite(vc) else -np.inf
            discharge = thresholds(vd)[1] if np.isfinite(vd) else np.inf
            if charge > discharge:
                charge = discharge = (charge + discharge) / 2
            p = actual[k]
            mw = b.power_mw if p >= discharge else (-b.power_mw if p <= charge else 0.0)
            mw = min(max(mw, -max(0.0, (b.e_max - e) / b.charge_eff / H5)), max(0.0, (e - b.e_min) * b.discharge_eff / H5))
            out, into = max(mw, 0) * H5, max(-mw, 0) * H5
            e += -out / b.discharge_eff + into * b.charge_eff
            net += (out - into) * p - (out + into) * b.degradation_cost
        return net, actual

    d = pd.read_parquet(OUT / "dispatch_preds.parquet")
    d["perfect"] = d.actual
    eps = episodes()
    rows = []
    for (r, k), x in d.assign(ep=np.searchsorted([pd.Timestamp(a) for a, _ in eps], pd.to_datetime(d.tau), side="right") - 1).groupby(["r", "ep"]):
        res = {"region": r, "episode": int(k)}
        for arm in ("fc", "A", "B", "perfect"):
            res[arm], actual = run_episode(x, arm)
        ch, dis, _ = solve_schedule(actual, np.full(len(actual), H5 + 1e-9), b, b.e_initial, TerminalCondition(), 1e-4,
                                    time_limit=60)
        mw = dis - ch
        res["pf"] = float(np.sum(mw * actual * H5 - np.abs(mw) * H5 * b.degradation_cost))
        rows.append(res)
        print(res, flush=True)
    df = pd.DataFrame(rows)
    summary = {arm: round(100 * df[arm].sum() / df.pf.sum(), 1) for arm in ("fc", "A", "B", "perfect")}
    by_region = {r: {arm: round(100 * x[arm].sum() / x.pf.sum(), 1) for arm in ("fc", "A", "B", "perfect")}
                 for r, x in df.groupby("region")}
    print("% of perfect foresight:", summary)
    print(json.dumps(by_region, indent=1))
    (OUT / "dispatch_results.json").write_text(json.dumps({"total": summary, "by_region": by_region, "episodes": rows}, indent=1))


if __name__ == "__main__":
    {"build": build, "build-dispatch": lambda: build("dispatch"), "fit": fit, "predict": predict,
     "dispatch": dispatch}[sys.argv[1]]()
