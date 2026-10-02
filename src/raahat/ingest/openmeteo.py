"""Open-Meteo clients (spec 4.2). Free, keyless, CC BY 4.0.

Attribution is REQUIRED and must appear in the dashboard footer and on the data
slide: "Weather data by Open-Meteo.com (CC BY 4.0)".

Everything here requests `hourly=` with `timezone=GMT` and aggregates to IMD
rainfall days via ingest.time_align. The provider's `daily=` sums use LOCAL
MIDNIGHT and must never be used -- see spec 4.2 and the measurement recorded in
time_align's docstring.

VERIFIED against the live API on 2026-09-21 (spec 0 rule 3):
  * `precipitation_previous_day1..5` exist as HOURLY variables and return data
  * ERA5 archive serves pressure_msl, wind_speed_850hPa, wind_direction_850hPa,
    geopotential_height_500hPa, relative_humidity_850hPa, temperature_850hPa
  * see docs/DATA_NOTES.md for the per-model archive-depth findings
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from raahat.ingest import time_align as TA
from raahat.ingest.http_cache import get_json

PREVIOUS_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"

ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0)"

#: Models whose previous-runs archive was verified usable back to Jul 2024.
#: GFS is excluded by default -- see docs/DATA_NOTES.md; it collapses on the
#: heavy-rain windows that are the entire point of this project.
DEFAULT_MODELS = ("ecmwf_ifs025", "icon_seamless")


def _build(url: str, params: dict) -> str:
    parts = []
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            v = ",".join(str(x) for x in v)
        parts.append(f"{k}={v}")
    return f"{url}?{'&'.join(parts)}"


def fetch_point_forecasts(
    lat: float,
    lon: float,
    start: str,
    end: str,
    *,
    model: str = "ecmwf_ifs025",
    leads: tuple[int, ...] = (1, 2, 3, 4, 5),
    offline: bool = False,
) -> pd.DataFrame:
    """Forecast rainfall at one point, aggregated to IMD rainfall days.

    Returns long-form [valid_date, lead_day, pr_mm, n_hours], one row per
    (rainfall day x lead). The lead-0 analysis is returned as lead_day 0 so
    that callers can sanity-check the model's own best estimate against truth.
    """
    variables = ["precipitation"] + [f"precipitation_previous_day{d}" for d in leads]
    url = _build(
        PREVIOUS_RUNS,
        {
            "latitude": round(float(lat), 4),
            "longitude": round(float(lon), 4),
            "hourly": variables,
            "start_date": start,
            "end_date": end,
            "timezone": "GMT",
            "models": model,
        },
    )
    h = get_json(url, offline=offline).get("hourly", {})
    if "time" not in h:
        raise KeyError(f"no hourly block returned for {lat},{lon} {model}")

    times = h["time"]
    out = []
    for lead in (0, *leads):
        key = "precipitation" if lead == 0 else f"precipitation_previous_day{lead}"
        if key not in h:
            continue
        agg = TA.aggregate_hourly(times, h[key])
        agg["lead_day"] = np.int8(lead)
        agg = agg.rename(columns={"precip_mm": "pr_mm"})
        out.append(agg)
    if not out:
        raise KeyError(f"none of {variables} returned for {lat},{lon} {model}")
    return pd.concat(out, ignore_index=True)[["valid_date", "lead_day", "pr_mm", "n_hours"]]


def fetch_point_analysis(
    lat: float, lon: float, start: str, end: str, *, offline: bool = False
) -> pd.DataFrame:
    """ERA5 reanalysis precipitation at one point, on IMD rainfall days.

    Used for the time-alignment check and as a circulation source. NOT used as
    truth -- truth is the IMD gauge-based grid (spec 4.1).
    """
    url = _build(
        ARCHIVE,
        {
            "latitude": round(float(lat), 4),
            "longitude": round(float(lon), 4),
            "hourly": "precipitation",
            "start_date": start,
            "end_date": end,
            "timezone": "GMT",
        },
    )
    h = get_json(url, offline=offline)["hourly"]
    return TA.aggregate_hourly(h["time"], h["precipitation"]).rename(
        columns={"precip_mm": "era5_mm"}
    )


#: Synoptic fields for the regime mesh (spec 4.3). Names verified live.
MESH_VARIABLES = [
    "pressure_msl",
    "wind_speed_850hPa",
    "wind_direction_850hPa",
    "wind_speed_200hPa",
    "wind_direction_200hPa",
    "geopotential_height_500hPa",
    "relative_humidity_850hPa",
    "temperature_850hPa",
    "precipitation",
]


def fetch_mesh_point(
    lat: float,
    lon: float,
    start: str,
    end: str,
    *,
    source: str = "archive",
    model: str | None = None,
    offline: bool = False,
) -> pd.DataFrame:
    """One synoptic mesh point, daily means on IMD rainfall days.

    source='archive' gives the ERA5 ANALYSIS mesh, which supervises the regime
    labels. source='previous_runs' gives the FORECAST mesh at lead time, which
    is what the classifier actually reads (spec 4.3).
    """
    base = ARCHIVE if source == "archive" else PREVIOUS_RUNS
    params = {
        "latitude": round(float(lat), 4),
        "longitude": round(float(lon), 4),
        "hourly": MESH_VARIABLES,
        "start_date": start,
        "end_date": end,
        "timezone": "GMT",
    }
    if model:
        params["models"] = model
    h = get_json(_build(base, params), offline=offline).get("hourly", {})
    if "time" not in h:
        raise KeyError(f"no hourly block for mesh point {lat},{lon}")

    df = pd.DataFrame({k: v for k, v in h.items() if k != "time"})
    df["valid_date"] = TA.to_rainfall_day(pd.to_datetime(h["time"]))

    # wind speed/direction -> u, v on ingest (spec 4.3), BEFORE averaging:
    # averaging a direction in degrees across a day is meaningless once it
    # wraps through 360, so the components must be formed first.
    for level in ("850hPa", "200hPa"):
        spd, dr = f"wind_speed_{level}", f"wind_direction_{level}"
        if spd in df and dr in df:
            rad = np.deg2rad(df[dr].astype("float64"))
            df[f"u{level[:3]}"] = -df[spd] * np.sin(rad)
            df[f"v{level[:3]}"] = -df[spd] * np.cos(rad)
            df = df.drop(columns=[dr])

    daily = df.groupby("valid_date").mean(numeric_only=True)
    daily["precip_sum"] = df.groupby("valid_date")["precipitation"].sum()
    n = df.groupby("valid_date").size()
    daily = daily[n >= 24]  # drop partial days rather than half-count them
    daily["lat"] = float(lat)
    daily["lon"] = float(lon)
    return daily.reset_index()
