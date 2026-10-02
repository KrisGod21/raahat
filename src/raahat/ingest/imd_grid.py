"""IMD 0.25 degree gridded daily rainfall -- the ground truth (spec 4.1).

135 x 129 grid, 6.5N-38.5N, 66.5E-100.0E, mm/day, missing value -999.0.

imdpune.gov.in is INTERMITTENT. It served three requests then began timing out
inside the same session. Everything here therefore reads the local cache first
and only touches the network when a year is genuinely missing, with retries and
a loud, honest failure (spec 0 rule 2: never fabricate, fail and log).

The IMD daily grid is already on the 0830 IST rainfall-day convention, so the
TRUTH side needs no time alignment. It is the FORECAST side that must be
aggregated 03Z-03Z -- see ingest/time_align.py.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from raahat import contract as C

CACHE_DIR = C.DATA_DIR / "raw" / "imd_grid"
MANIFEST = C.DATA_DIR / "_manifest"
MISSING = -999.0


#: A full year of 135x129 float32 days is ~25 MB. A truncated or empty file is
#: worse than a missing one: it looks cached, loads as a short record, and
#: silently corrupts every climatology built on it.
MIN_PLAUSIBLE_BYTES = 1_000_000


def cached_years(var: str = "rain") -> list[int]:
    """Years on disk that are actually usable, readable with no network.

    A zero-byte file is what imdpune.gov.in leaves behind when a year is not
    yet published -- 2026 did exactly this while its season was still running.
    Such files are reported as NOT cached so that a retry can replace them.
    """
    d = CACHE_DIR / var
    if not d.exists():
        return []
    out = []
    for p in d.glob("*.grd"):
        try:
            year = int(p.stem)
        except ValueError:
            continue
        if p.stat().st_size < MIN_PLAUSIBLE_BYTES:
            continue
        out.append(year)
    return sorted(out)


def prune_truncated(var: str = "rain") -> list[int]:
    """Delete implausibly small cache files. Returns the years removed."""
    d = CACHE_DIR / var
    removed = []
    if not d.exists():
        return removed
    for p in d.glob("*.grd"):
        if p.stat().st_size < MIN_PLAUSIBLE_BYTES:
            removed.append(p.stem)
            p.unlink()
    return removed


def _log_error(msg: str) -> None:
    MANIFEST.mkdir(parents=True, exist_ok=True)
    with (MANIFEST / "errors.log").open("a", encoding="utf-8") as fh:
        fh.write(f"{pd.Timestamp.now(tz='UTC').isoformat()}  {msg}\n")


def fetch_year(year: int, var: str = "rain", *, attempts: int = 4, pause: float = 5.0) -> bool:
    """Download one year into the cache. Returns True if it is now available.

    Never raises on a network failure -- it logs and returns False, so a
    partial archive can still be used and the gap is visible rather than
    silently filled.
    """
    import imdlib

    if year in cached_years(var):
        return True
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(1, attempts + 1):
        try:
            imdlib.get_data(var, year, year, fn_format="yearwise", file_dir=str(CACHE_DIR))
            if year in cached_years(var):
                return True
            _log_error(f"imd_grid {var} {year}: download reported success but no file appeared")
        except Exception as exc:  # network, TLS, upstream 5xx -- all transient here
            _log_error(f"imd_grid {var} {year}: attempt {i}/{attempts} {type(exc).__name__}: {exc}")
            if i < attempts:
                time.sleep(pause * i)  # linear backoff; the host is slow, not rate-limiting
    return False


def load(years: list[int], var: str = "rain") -> xr.DataArray:
    """Load cached years as one masked DataArray (time, lat, lon), mm/day.

    Raises if a requested year is not cached -- a silently short record would
    corrupt every climatology downstream.
    """
    import imdlib

    have = cached_years(var)
    missing = [y for y in years if y not in have]
    if missing:
        raise FileNotFoundError(
            f"IMD {var} not cached for {missing} (have {have}). "
            f"Run ingest.imd_grid.fetch_year() while imdpune.gov.in is reachable."
        )
    parts = []
    for y in years:
        ds = imdlib.open_data(var, y, y, fn_format="yearwise", file_dir=str(CACHE_DIR)).get_xarray()
        parts.append(ds[var])
    da = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
    return da.where(da != MISSING)


def district_means(
    da: xr.DataArray, master: pd.DataFrame, *, method: str = "nearest"
) -> pd.DataFrame:
    """Grid -> district daily mean rainfall.

    `method='nearest'` samples the cell containing each district centroid.
    That is the cheap version; spec 5.1 wants area-weighted aggregation over
    the polygon, which district_means_areal() does. Nearest is kept because it
    is a useful cross-check: where the two disagree badly, the district is
    small relative to 0.25 deg (~28 km) and its truth is representativeness-
    limited no matter what we do.
    """
    lats = xr.DataArray(master["lat_centroid"].to_numpy(), dims="district")
    lons = xr.DataArray(master["lon_centroid"].to_numpy(), dims="district")
    sub = da.sel(lat=lats, lon=lons, method=method)
    df = sub.to_pandas()
    df.columns = master["district_id"].to_numpy()
    out = df.stack(future_stack=True).rename("obs_rain_mm").reset_index()
    out.columns = ["valid_date", "district_id", "obs_rain_mm"]
    out["valid_date"] = pd.to_datetime(out["valid_date"]).dt.date
    out["district_id"] = out["district_id"].astype(np.int32)
    out["obs_rain_mm"] = out["obs_rain_mm"].astype(np.float32)
    return out


def district_means_areal(da: xr.DataArray, geo, *, min_cells: int = 1) -> pd.DataFrame:
    """Area-weighted grid -> district aggregation (spec 5.1, the real one).

    Every 0.25 deg cell whose centre falls inside the district polygon is
    averaged, weighted by the cosine of latitude so that cells are weighted by
    true area rather than by degree extent. Districts smaller than a cell fall
    back to the containing cell, and are flagged via n_cells so that
    representativeness-limited districts can be identified later.
    """
    import shapely

    lat = da["lat"].values
    lon = da["lon"].values
    lon2d, lat2d = np.meshgrid(lon, lat)
    pts = shapely.points(lon2d.ravel(), lat2d.ravel())
    weights = np.cos(np.deg2rad(lat2d.ravel()))

    times = pd.to_datetime(da["time"].values)
    values = da.values.reshape(len(times), -1)  # (time, cell)

    tree = shapely.STRtree(pts)
    rows = []
    for _, row in geo.iterrows():
        geom = row.geometry
        idx = tree.query(geom, predicate="intersects")
        if len(idx) < min_cells:
            # district smaller than a grid cell -- use the nearest cell centre
            idx = np.array([tree.nearest(geom.centroid)])
        w = weights[idx]
        v = values[:, idx]
        good = ~np.isnan(v)
        wsum = (good * w).sum(axis=1)
        tot = np.where(wsum > 0, np.nansum(v * w, axis=1) / np.where(wsum > 0, wsum, 1), np.nan)
        rows.append(
            pd.DataFrame(
                {
                    "valid_date": times.date,
                    "district_id": np.int32(row.district_id),
                    "obs_rain_mm": tot.astype(np.float32),
                    "n_cells": np.int16(len(idx)),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)
