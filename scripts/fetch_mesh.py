"""Fetch the synoptic meshes (spec 4.3).

Two meshes, deliberately different, because they serve different masters:

  ANALYSIS mesh (ERA5 archive, 2 deg, 378 points)
      MSLP + precipitation only -- Open-Meteo serves no upper air at all.
      Supervises the regime LABELS. Resolution matters here because the LPS
      detector needs a pressure field fine enough to resolve a closed low.

  FORECAST mesh (previous-runs, 3 deg, 168 points, SURFACE ONLY)
      What the classifier actually reads at lead time.

WHY SURFACE ONLY -- this is a real constraint, not a shortcut. The Previous
Runs API exposes `<var>_previous_dayN` for surface variables only; every
pressure-level variable returns HTTP 400 ("Cannot initialize
SurfacePressureAndHeightVariable"). The Single Runs API does serve pressure
levels but requires a `run` parameter, i.e. one call per init time per point --
122 inits x 378 points is ~46,000 calls against a 10,000/day tier.

So the design splits: rich reanalysis defines what a regime IS, and the
classifier must recognise it from what a forecast actually publishes. That is
arguably more honest than the spec's plan, and it is exactly the constraint an
operational system would face.

Resolution asymmetry is a latency decision: a forecast-mesh point takes ~10s
(five leads x six variables, server-side), an analysis point ~1.2s.

Usage:
    python scripts/fetch_mesh.py --seasons 2024 2025
    python scripts/fetch_mesh.py --seasons 2024 --which analysis
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raahat import contract as C  # noqa: E402
from raahat.ingest import time_align as TA  # noqa: E402
from raahat.ingest.http_cache import FetchError, cache_stats, get_json  # noqa: E402

INTERIM = C.DATA_DIR / "interim"

ANALYSIS_STEP = 2.0
FORECAST_STEP = 4.0   # 110 points, down from 168 at 3 deg
LAT_RANGE = (5.0, 40.0)
LON_RANGE = (60.0, 100.0)

# LEAN CONFIG (2026-09-22). The previous version requested seven pressure-level
# variables that Open-Meteo returns as ALL-NULL (docs/DATA_NOTES.md item 16) and
# six forecast variables where four carry the signal. Open-Meteo weights calls
# by variables x days, so those nulls cost real quota and bought nothing -- the
# fetch then died at 25% on HTTP 429 (item 25). Only fields that actually
# return data are requested now.
ANALYSIS_VARS = ["pressure_msl", "precipitation"]
FORECAST_VARS = [
    "precipitation", "pressure_msl", "wind_speed_10m", "wind_direction_10m",
]
LEADS = (1, 2, 3, 4, 5)


def mesh_points(step: float) -> list[tuple[float, float]]:
    lats = np.arange(LAT_RANGE[0], LAT_RANGE[1] + 1e-9, step)
    lons = np.arange(LON_RANGE[0], LON_RANGE[1] + 1e-9, step)
    return [(round(float(a), 2), round(float(o), 2)) for a in lats for o in lons]


def season_window(season: int) -> tuple[str, str]:
    return f"{season}-05-25", f"{season}-10-02"


def _uv(speed, direction):
    """Meteorological speed/bearing -> u,v. Must happen BEFORE any averaging:
    a mean of bearings is meaningless once they wrap through 360 deg."""
    rad = np.deg2rad(np.asarray(direction, dtype="float64"))
    s = np.asarray(speed, dtype="float64")
    return -s * np.sin(rad), -s * np.cos(rad)


def fetch_analysis(points, season, *, offline=False) -> pd.DataFrame:
    start, end = season_window(season)
    rows, fails = [], 0
    t0 = time.time()
    for i, (lat, lon) in enumerate(points, 1):
        url = ("https://archive-api.open-meteo.com/v1/archive"
               f"?latitude={lat}&longitude={lon}&hourly={','.join(ANALYSIS_VARS)}"
               f"&start_date={start}&end_date={end}&timezone=GMT")
        try:
            h = get_json(url, namespace="mesh_analysis", offline=offline)["hourly"]
        except (FetchError, KeyError):
            fails += 1
            continue
        df = pd.DataFrame({k: v for k, v in h.items() if k != "time"})
        df["valid_date"] = TA.to_rainfall_day(pd.to_datetime(h["time"]))
        for lvl in ("850hPa", "200hPa"):
            sp, dr = f"wind_speed_{lvl}", f"wind_direction_{lvl}"
            if sp in df and dr in df:
                u, v = _uv(df[sp], df[dr])
                df[f"u{lvl[:3]}"], df[f"v{lvl[:3]}"] = u, v
                df = df.drop(columns=[sp, dr])
        n = df.groupby("valid_date").size()
        daily = df.groupby("valid_date").mean(numeric_only=True)
        daily["precip_sum"] = df.groupby("valid_date")["precipitation"].sum()
        daily = daily[n >= 24]
        daily["lat"], daily["lon"] = lat, lon
        rows.append(daily.reset_index())
        if i % 50 == 0:
            rate = i / max(time.time() - t0, 1e-9)
            print(f"    analysis {i}/{len(points)}  {rate:.1f}/s  "
                  f"eta {(len(points)-i)/max(rate,1e-9)/60:.1f} min  fails {fails}")
    if not rows:
        raise SystemExit("analysis mesh: nothing fetched")
    return pd.concat(rows, ignore_index=True)


def fetch_forecast(points, season, *, offline=False, pause=6.0) -> pd.DataFrame:
    start, end = season_window(season)
    hourly = ",".join(f"{v}_previous_day{d}" for v in FORECAST_VARS for d in LEADS)
    rows, fails = [], 0
    t0 = time.time()
    for i, (lat, lon) in enumerate(points, 1):
        url = ("https://previous-runs-api.open-meteo.com/v1/forecast"
               f"?latitude={lat}&longitude={lon}&hourly={hourly}"
               f"&start_date={start}&end_date={end}&timezone=GMT&models=ecmwf_ifs025")
        try:
            h = get_json(url, namespace="mesh_forecast", offline=offline)["hourly"]
        except (FetchError, KeyError):
            fails += 1
            continue
        dates = TA.to_rainfall_day(pd.to_datetime(h["time"]))
        for lead in LEADS:
            cols = {}
            for v in FORECAST_VARS:
                key = f"{v}_previous_day{lead}"
                if key in h:
                    cols[v] = h[key]
            if not cols:
                continue
            df = pd.DataFrame(cols)
            df["valid_date"] = dates
            if "wind_speed_10m" in df and "wind_direction_10m" in df:
                u, v10 = _uv(df["wind_speed_10m"], df["wind_direction_10m"])
                df["u10"], df["v10"] = u, v10
                df = df.drop(columns=["wind_direction_10m"])
            n = df.groupby("valid_date").size()
            daily = df.groupby("valid_date").mean(numeric_only=True)
            daily["precip_sum"] = df.groupby("valid_date")["precipitation"].sum()
            daily = daily[n >= 24]
            daily["lat"], daily["lon"] = lat, lon
            daily["lead_day"] = np.int8(lead)
            rows.append(daily.reset_index())
        if i % 10 == 0:
            rate = i / max(time.time() - t0, 1e-9)
            print(f"    forecast {i}/{len(points)}  {rate:.2f}/s  "
                  f"eta {(len(points)-i)/max(rate,1e-9)/60:.1f} min  fails {fails}", flush=True)
        # Open-Meteo rate-limits these (30 series x 131 days is an expensive
        # call). Pacing beats hammering: a 429 costs far more than a pause.
        if not offline:
            time.sleep(pause)
    if not rows:
        raise SystemExit("forecast mesh: nothing fetched")
    return pd.concat(rows, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025])
    ap.add_argument("--which", choices=["analysis", "forecast", "both"], default="both")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args(argv)

    INTERIM.mkdir(parents=True, exist_ok=True)
    an_pts, fc_pts = mesh_points(ANALYSIS_STEP), mesh_points(FORECAST_STEP)
    print(f"analysis mesh {ANALYSIS_STEP} deg -> {len(an_pts)} points")
    print(f"forecast mesh {FORECAST_STEP} deg -> {len(fc_pts)} points")
    print(f"cache before: {cache_stats('mesh_analysis')['entries']} analysis, "
          f"{cache_stats('mesh_forecast')['entries']} forecast\n")

    for season in args.seasons:
        if args.which in ("analysis", "both"):
            print(f"  analysis mesh {season} ...")
            df = fetch_analysis(an_pts, season, offline=args.offline)
            df.to_parquet(INTERIM / f"mesh_analysis_{season}.parquet", index=False)
            print(f"    -> mesh_analysis_{season}.parquet  {len(df):,} rows, "
                  f"{df.groupby(['lat','lon']).ngroups} points")
        if args.which in ("forecast", "both"):
            print(f"  forecast mesh {season} ...")
            df = fetch_forecast(fc_pts, season, offline=args.offline)
            df.to_parquet(INTERIM / f"mesh_forecast_{season}.parquet", index=False)
            print(f"    -> mesh_forecast_{season}.parquet  {len(df):,} rows, "
                  f"{df.groupby(['lat','lon']).ngroups} points, "
                  f"leads {sorted(int(x) for x in df.lead_day.unique())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
