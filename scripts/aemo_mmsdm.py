"""Shared downloader for AEMO's MMSDM historical data archive.

NEMOSIS only wires up the small set of tables under the MMSDM archive's
`DATA/` directory (DISPATCHPRICE, TRADINGPRICE, etc). P5MIN_REGIONSOLUTION
and PREDISPATCHPRICE live under their own `P5MIN_ALL_DATA/` and
`PREDISP_ALL_DATA/` directories and aren't supported by NEMOSIS, so this
module replicates just enough of its download logic for those two tables.

Current (post-2024) archive filenames use a literal `#` on disk, e.g.
`PUBLIC_ARCHIVE#P5MIN_REGIONSOLUTION#ALL#FILE01#202606010000.zip`. AEMO's
CDN also 403s requests without a browser-like User-Agent.
"""

import zipfile
from pathlib import Path

import pandas as pd
import requests

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)

BASE_URL = "https://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM"

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def _archive_url(year: int, month: int, directory: str, filename_stub: str) -> str:
    month_str = f"{month:02d}"
    filename = f"{filename_stub}#{year}{month_str}010000"
    # Literal `#` in the on-disk filename must be sent as a single
    # percent-escape for nemweb to resolve it (not left raw, which `#`
    # would otherwise truncate as a URL fragment).
    filename = filename.replace("#", "%23")
    return (
        f"{BASE_URL}/{year}/MMSDM_{year}_{month_str}/"
        f"MMSDM_Historical_Data_SQLLoader/{directory}/{filename}.zip"
    )


def download_month(
    year: int,
    month: int,
    directory: str,
    filename_stub: str,
    cache_dir: Path,
) -> Path | None:
    """Download and extract one month's archive zip, returning the CSV path.

    Returns None if AEMO hasn't published that month yet (404).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = _archive_url(year, month, directory, filename_stub)
    zip_name = url.rsplit("/", 1)[-1]
    zip_path = cache_dir / zip_name
    csv_path = cache_dir / zip_name.replace(".zip", ".CSV").replace("%23", "#")

    if csv_path.exists():
        return csv_path

    if not zip_path.exists():
        response = session.get(url, stream=True)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=2**16):
                f.write(chunk)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(cache_dir)

    return csv_path


def load_mmsdm_csv(csv_path: Path) -> pd.DataFrame:
    """Parse an MMSDM CSV: skip the leading `C` metadata row, use the `I`
    row as the header, and drop the trailing `C,END OF REPORT` footer row.
    """
    data = pd.read_csv(csv_path, skiprows=1, low_memory=False)
    return data.iloc[:-1, :]


def download_mmsdm_table(
    start: str,
    end: str,
    directory: str,
    filename_stub: str,
    cache_dir: Path,
    datetime_columns: list[str],
) -> pd.DataFrame:
    """Download every month in [start, end) and concatenate into one frame.

    `start`/`end` are 'YYYY/MM/DD' strings, `end` exclusive.
    """
    months = pd.date_range(start=start, end=end, freq="MS", inclusive="left")
    frames = []
    for ts in months:
        csv_path = download_month(ts.year, ts.month, directory, filename_stub, cache_dir)
        if csv_path is None:
            print(f"SKIP {filename_stub} {ts.year}-{ts.month:02d}: not yet published")
            continue
        print(f"OK   {filename_stub} {ts.year}-{ts.month:02d}")
        frames.append(load_mmsdm_csv(csv_path))

    data = pd.concat(frames, axis=0, ignore_index=True)
    for col in datetime_columns:
        data[col] = pd.to_datetime(data[col])
    return data
