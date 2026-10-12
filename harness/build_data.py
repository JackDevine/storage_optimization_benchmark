"""Build the databases the simulator reads from the raw AEMO tables.

    uv run python -m harness.build_data

Reads data/benchmark.duckdb (written by scripts/pull_data.py) and writes:

- data/market.duckdb: every period, with prices mapped onto today's market
  price cap (bess/normalise.py). The evaluator scores holdout windows from it.
- data/dev.duckdb: its dev-period subset (harness/splits.py), which is all the
  agent gets.

Both hold `dispatchprice` (with AEMO's administered-price flag, APCFLAG, and
the price before mapping, RRP_RAW),
`p5min_regionsolution` and `predispatchprice`, plus `price_map`, recording the
mapping used.
"""

import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np

from bess.data import DB_PATH
from bess.normalise import PRICE_MAP, mpc_at
from harness.sandbox import PROJECT_ROOT
from harness.splits import DATA_VERSION, export_dev_db

RAW_DB = PROJECT_ROOT / "data" / "benchmark.duckdb"
DEV_DB = PROJECT_ROOT / "data" / "dev.duckdb"
INTERVAL_SQL = "INTERVAL 5 MINUTE"

# table -> (columns kept, the time that picks the cap, sort order)
TABLES = {
    "dispatchprice": (
        "SETTLEMENTDATE, REGIONID, INTERVENTION, RRP, APCFLAG, MARKETSUSPENDEDFLAG",
        f"SETTLEMENTDATE - {INTERVAL_SQL}",  # the interval's start
        "REGIONID, SETTLEMENTDATE",
    ),
    "p5min_regionsolution": (
        "RUN_DATETIME, INTERVAL_DATETIME, REGIONID, INTERVENTION, RRP",
        f"RUN_DATETIME - {INTERVAL_SQL}",  # when the run was made
        "REGIONID, RUN_DATETIME, INTERVAL_DATETIME",
    ),
    "predispatchprice": (
        "LASTCHANGED, DATETIME, REGIONID, INTERVENTION, RRP",
        "LASTCHANGED",
        "REGIONID, LASTCHANGED, DATETIME",
    ),
}


def build(dest: Path = DB_PATH, raw: Path = RAW_DB) -> None:
    tmp = dest.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    try:
        con.execute(f"ATTACH '{raw}' AS raw (READ_ONLY)")
        thresholds = con.execute("SELECT * FROM raw.market_price_thresholds").df()
        caps = thresholds.sort_values(["EFFECTIVEDATE", "VERSIONNO"]).drop_duplicates("EFFECTIVEDATE", keep="last")
        con.register("caps", caps[["EFFECTIVEDATE", "VOLL"]])
        if caps.VOLL.iloc[-1] != PRICE_MAP.mpc_now:
            raise ValueError(f"the latest cap is {caps.VOLL.iloc[-1]}, but PRICE_MAP maps to {PRICE_MAP.mpc_now}")
        k, m_now, gamma = PRICE_MAP.knee, PRICE_MAP.mpc_now, PRICE_MAP.gamma
        for name, (columns, when, order) in TABLES.items():
            t0 = time.time()
            # The cap in force: the last effective date at or before `when`
            # (an ASOF join), then g(p) from bess/normalise.py in SQL.
            # TODO(cap fix, next rebuild): forecasts are mapped with the cap at their run
            # time, but their prices are set under the cap of the period they forecast.
            # Runs on 30 June 2024 forecast July periods under the new $17,500 cap and were
            # stretched by the old $16,600 one: 119 predispatch rows map above $23,200 (up to
            # $24,042). Map forecasts with the cap in force at the forecast period instead.
            mapped = (
                f"CASE WHEN t.RRP > {k} AND c.VOLL <> {m_now} THEN "
                f"{k} + ({m_now} - {k}) * pow(greatest((t.RRP - {k}) / (c.VOLL - {k}), 0), {gamma}) "
                f"ELSE t.RRP END"
            )
            cols = ", ".join(f"t.{c.strip()}" if c.strip() != "RRP" else f"{mapped} AS RRP" for c in columns.split(","))
            con.execute(f"""
                CREATE TABLE {name} AS
                SELECT {cols}{", t.RRP AS RRP_RAW" if name == "dispatchprice" else ""}
                FROM (SELECT *, {when} AS _when FROM raw.{name}) t
                ASOF JOIN caps c ON t._when >= c.EFFECTIVEDATE
                ORDER BY {order}
            """)
            n = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            print(f"{name}: {n:,} rows ({time.time() - t0:.0f}s)", flush=True)
        con.execute("CREATE TABLE price_map AS SELECT ? AS data_version, ? AS params",
                    [DATA_VERSION, json.dumps(PRICE_MAP.__dict__)])
        con.execute("CREATE TABLE market_price_thresholds AS SELECT * FROM raw.market_price_thresholds")
    finally:
        con.close()
    tmp.replace(dest)
    _check(dest, thresholds)


def _check(dest: Path, thresholds) -> None:
    """The SQL mapping agrees with PRICE_MAP on a sample."""
    con = duckdb.connect(str(dest), read_only=True)
    try:
        s = con.execute("""SELECT SETTLEMENTDATE, RRP, RRP_RAW FROM dispatchprice
                           USING SAMPLE 20000 ROWS""").df()
        extra = con.execute("""SELECT SETTLEMENTDATE, RRP, RRP_RAW FROM dispatchprice
                               WHERE RRP_RAW > 1000 LIMIT 2000""").df()
    finally:
        con.close()
    for d in (s, extra):
        expect = PRICE_MAP(d.RRP_RAW.to_numpy(), mpc_at(thresholds, d.SETTLEMENTDATE - np.timedelta64(5, "m")))
        assert np.allclose(expect, d.RRP.to_numpy(), rtol=1e-9, atol=1e-6), "SQL and PRICE_MAP disagree"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-market", action="store_true", help="only re-export data/dev.duckdb")
    args = parser.parse_args()
    if not args.skip_market:
        build()
    export_dev_db(DEV_DB)
    print(f"wrote {DB_PATH} and {DEV_DB} ({DATA_VERSION})")
