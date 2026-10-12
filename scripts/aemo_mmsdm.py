"""Downloader for AEMO's MMSDM historical data archive.

Each month's archive is a directory of zipped CSVs, one or more files per
table. AEMO renamed them in August 2024:

- before: `PUBLIC_DVD_P5MIN_REGIONSOLUTION_ALL_202110010000.zip`
- after:  `PUBLIC_ARCHIVE#P5MIN_REGIONSOLUTION#ALL#FILE01#202608010000.zip`

so files are found by listing the month's directory rather than by building
their names. AEMO's CDN 403s requests without a browser-like User-Agent, and
the `#` in file names must be sent percent-escaped.

The zips are kept under `data/raw_cache/`; the CSVs are extracted to a temporary
directory, read with DuckDB, and deleted.
"""

import re
import tempfile
import zipfile
from functools import cache
from pathlib import Path

import duckdb
import requests

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)

BASE_URL = "https://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM"

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def _month_url(year: int, month: int, directory: str) -> str:
    return f"{BASE_URL}/{year}/MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader/{directory}/"


@cache
def list_month(year: int, month: int, directory: str) -> tuple[str, ...] | None:
    """File names in one month's archive directory (unescaped), or None if the
    month isn't published yet."""
    response = session.get(_month_url(year, month, directory))
    if response.status_code == 404:
        return None
    response.raise_for_status()
    names = re.findall(r'HREF="[^"]*/([^"/]+\.zip)"', response.text, flags=re.IGNORECASE)
    return tuple(sorted({n.replace("%23", "#") for n in names}))


def table_files(year: int, month: int, directory: str, table: str) -> list[str] | None:
    """The month's file names for one AEMO table (e.g. `P5MIN_REGIONSOLUTION`),
    in either naming scheme. None if the month isn't published."""
    names = list_month(year, month, directory)
    if names is None:
        return None
    stamp = f"{year}{month:02d}010000"
    old = re.compile(rf"PUBLIC_DVD_{table}(_ALL)?_(FILE\d+_)?{stamp}\.zip", re.IGNORECASE)
    new = re.compile(rf"PUBLIC_ARCHIVE#{table}(#ALL)?#FILE\d+#{stamp}\.zip", re.IGNORECASE)
    return [n for n in names if old.fullmatch(n) or new.fullmatch(n)]


def download(year: int, month: int, directory: str, name: str, cache_dir: Path) -> Path:
    """Download one archive zip into `cache_dir` (skipped if already there)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / name.replace("#", "%23")
    if path.exists():
        return path
    tmp = path.with_suffix(".part")
    response = session.get(_month_url(year, month, directory) + name.replace("#", "%23"), stream=True)
    response.raise_for_status()
    with open(tmp, "wb") as f:
        for chunk in response.iter_content(chunk_size=2**20):
            f.write(chunk)
    tmp.rename(path)
    return path


def read_zip(con: duckdb.DuckDBPyConnection, zip_path: Path, columns: dict[str, str],
             filter: str | None = None) -> duckdb.DuckDBPyRelation:
    """Read the data rows of an MMSDM CSV zip into a DuckDB relation with the
    given {column: SQL type} (timestamps parse AEMO's `YYYY/MM/DD HH:MM:SS`).
    Columns a file doesn't have (older table versions) come back NULL. `filter`
    is an SQL condition on the raw text columns, applied before casting.

    An MMSDM CSV is a `C` metadata row, an `I` header row, `D` data rows and a
    `C,END OF REPORT` footer. The relation is materialised before the CSV is
    deleted."""
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(zip_path) as z:
        (member,) = [m for m in z.namelist() if m.upper().endswith(".CSV")]
        csv = Path(tmp) / "table.csv"  # DuckDB mis-sniffs paths containing '#'
        # Drop the C row, and the CRs: newer files mix \n and \r\n line ends,
        # which DuckDB's dialect sniffer rejects.
        with z.open(member) as src, open(csv, "wb") as dst:
            src.readline()
            while chunk := src.read(2**22):
                dst.write(chunk.replace(b"\r", b""))
        raw = con.read_csv(str(csv), header=True, all_varchar=True, null_padding=True, sep=",", quotechar='"')
        con.register("_mmsdm_raw", raw)
        first = raw.columns[0]
        present = set(raw.columns)
        select = ", ".join(
            f"NULL::{t} AS {c}" if c not in present
            else f"try_strptime(\"{c}\", '%Y/%m/%d %H:%M:%S') AS {c}" if t == "TIMESTAMP"
            else f"TRY_CAST(\"{c}\" AS {t}) AS {c}"
            for c, t in columns.items()
        )
        name = f"_mmsdm_{abs(hash(str(zip_path)))}"
        con.execute(f"CREATE OR REPLACE TEMP TABLE {name} AS SELECT {select} FROM _mmsdm_raw WHERE \"{first}\" = 'D'"
                    + (f" AND ({filter})" if filter else ""))
        con.unregister("_mmsdm_raw")
    return con.table(name)
