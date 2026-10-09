# Storage Optimization Benchmark

Benchmark data and models for storage optimization against AEMO NEM market data.

## Setup

```
uv sync
```

## Pulling data

```
uv run scripts/pull_dispatch_price.py          # DISPATCHPRICE (actual, 5-min) via NEMOSIS
uv run scripts/pull_dispatch_regionsum.py      # DISPATCHREGIONSUM (actual demand/UIGF, 5-min) via NEMOSIS
uv run scripts/pull_forecast_prices.py         # P5MIN_REGIONSOLUTION, PREDISPATCHPRICE, PREDISPATCHREGIONSUM
```

All three load straight into `data/benchmark.duckdb` (one table per AEMO table, e.g. `dispatchprice`,
`dispatchregionsum`, `p5min_regionsolution`, `predispatchprice`, `predispatchregionsum`). AEMO's raw downloads
are kept under `data/raw_cache/`. Neither directory is committed — see `.gitignore`.

NEMOSIS doesn't support the P5MIN/PREDISPATCH forecast tables (the maintainer points to a sister package,
NEMSEER, instead — see https://github.com/UNSW-CEEM/NEMOSIS/issues/32). We're using our own small downloader
(`scripts/aemo_mmsdm.py`) for those rather than taking on NEMSEER's GPLv3 license.

## Data caching — know this before debugging a "why is this stale/missing" issue

`data/raw_cache/` is **not uniform** — the file formats present depend on which tool wrote them, and the cache
only applies to the *download* step, not to what ends up in DuckDB:

- **Our downloader** (`aemo_mmsdm.py`, used for the forecast tables) keeps the original `.zip` plus the
  extracted `.CSV`. No parquet — it re-parses the CSV with pandas on every run.
- **NEMOSIS** (used for `dispatchprice`/`dispatchregionsum`) keeps `.CSV` + `.parquet` by default
  (`keep_csv=True`); this version of NEMOSIS has no `keep_zip` option, so the zip is always deleted after
  extraction. The `.parquet` file is what NEMOSIS actually checks to decide whether to skip re-downloading a
  month — the `.CSV` sitting next to it is an unused byproduct, not load-bearing.
- Both tools skip re-hitting AEMO's servers if the relevant month's file already exists locally — this is a
  real, useful cache (confirmed empirically: re-running a pull script after the first time only attempts the
  network for months not yet downloaded).
- **The DuckDB write is not cached.** Every pull script does `CREATE OR REPLACE TABLE` — it always re-parses
  whatever's in `raw_cache/` and rebuilds the table from scratch, even if nothing changed. Extending a date
  range re-downloads only the new months but still reprocesses and rewrites the *entire* table, not just the
  new rows. There's no incremental append and no "already loaded, skip" check at the DuckDB layer.

If you add a new table/puller, don't assume raw_cache's file mix will look the same as an existing one — it's
an artifact of which library wrote it and what's been manually cleaned up since, not a deliberate convention.
