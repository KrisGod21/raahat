"""Active / break monsoon index (spec 4.5), after Rajeevan, Gadgil & Bhate (2010).

Published definition: active (break) spells are periods when the STANDARDISED
RAINFALL ANOMALY averaged over the monsoon core zone exceeds +1 (falls below
-1) for at least THREE CONSECUTIVE DAYS. Core monsoon zone ~18-28N, 65-88E.

This is the one label the loss of upper-air data does not touch, because the
published definition is built on rainfall, not circulation -- and we hold the
IMD 0.25 deg grid locally. So ACTIVE and BREAK, the two highest-frequency
classes, are labelled exactly as the literature intends.

Two versions are produced, and keeping them apart is a leakage question, not a
style question (spec 5.7):

  OBSERVED  -- from the IMD grid. A LABEL. Never a feature.
  FORECAST  -- from forecast-mesh precipitation at each lead. A FEATURE.
               This is what the classifier is allowed to see.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from raahat import contract as C

_cfg = C.load_config("regimes")
CMZ = _cfg["core_monsoon_zone"]
_ab = _cfg["active_break"]

THRESHOLD = float(_ab["threshold"])
MIN_PERSISTENCE = int(_ab["min_persistence_days"])
SMOOTH_WINDOW = int(_ab["smoothing_window_days"])


def cmz_series(da, *, box: dict | None = None) -> pd.Series:
    """Area-average rainfall over the core monsoon zone, cosine-lat weighted."""
    box = box or CMZ
    sub = da.sel(
        lat=slice(box["lat_min"], box["lat_max"]),
        lon=slice(box["lon_min"], box["lon_max"]),
    )
    w = np.cos(np.deg2rad(sub["lat"]))
    s = sub.weighted(w).mean(dim=("lat", "lon")).to_pandas()
    s.index = pd.to_datetime(s.index)
    return s.rename("cmz_rain_mm")


def doy_climatology(series: pd.Series, *, window: int = SMOOTH_WINDOW) -> pd.DataFrame:
    """Day-of-year mean and SD, smoothed with a circular rolling window.

    The window wraps through 31 Dec so that late-December and early-January
    are not each computed from a truncated window. JJAS never touches the
    wrap, but a silently asymmetric climatology is the kind of thing that
    surfaces two months later as an unexplained seasonal drift.
    """
    df = pd.DataFrame({"rain": series})
    df["doy"] = df.index.dayofyear
    g = df.groupby("doy")["rain"]
    clim = pd.DataFrame({"mean": g.mean(), "sd": g.std()}).reindex(range(1, 367))
    clim = clim.interpolate(limit_direction="both")

    tripled = pd.concat([clim, clim, clim])
    sm = tripled.rolling(window, center=True, min_periods=1).mean()
    out = sm.iloc[len(clim) : 2 * len(clim)].copy()
    out.index = clim.index
    out["sd"] = out["sd"].replace(0.0, np.nan).fillna(out["sd"].mean())
    return out


def standardised_anomaly(series: pd.Series, clim: pd.DataFrame) -> pd.Series:
    doy = pd.Series(series.index.dayofyear, index=series.index)
    mu = doy.map(clim["mean"])
    sd = doy.map(clim["sd"])
    return ((series - mu) / sd).rename("cmz_anom")


def _persistent_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """True only where `mask` is part of a run of at least `min_len`."""
    out = np.zeros_like(mask, dtype=bool)
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out[start:i] = True
            start = None
    if start is not None and len(mask) - start >= min_len:
        out[start:] = True
    return out


def classify(
    anom: pd.Series, *, threshold: float = THRESHOLD, min_days: int = MIN_PERSISTENCE
) -> pd.DataFrame:
    """Label each day ACTIVE / BREAK / NEUTRAL with the persistence rule."""
    a = anom.to_numpy()
    active = _persistent_runs(a > threshold, min_days)
    brk = _persistent_runs(a < -threshold, min_days)
    label = np.where(active, "ACTIVE", np.where(brk, "BREAK", "NEUTRAL"))
    return pd.DataFrame(
        {"valid_date": [d.date() for d in anom.index], "cmz_anom": a, "ab_label": label}
    )


def observed_index(years: list[int], *, clim_years: list[int] | None = None) -> pd.DataFrame:
    """The LABEL-side active/break index, from the IMD grid.

    `clim_years` must exclude the validation and test periods (spec 7.2 trap 4).
    """
    from raahat.ingest import imd_grid

    clim_years = clim_years or [y for y in imd_grid.cached_years() if y < min(years)]
    if len(clim_years) < 3:
        raise FileNotFoundError(
            f"need >=3 climatology years before {min(years)}; have {clim_years}"
        )
    clim = doy_climatology(cmz_series(imd_grid.load(sorted(clim_years))))
    series = cmz_series(imd_grid.load(sorted(years)))
    out = classify(standardised_anomaly(series, clim))
    out.attrs["clim_years"] = sorted(clim_years)
    return out


def forecast_index(mesh: pd.DataFrame, clim: pd.DataFrame, *, box: dict | None = None) -> pd.DataFrame:
    """The FEATURE-side index, from forecast-mesh precipitation at each lead.

    Same arithmetic, different source. This one the classifier may see.
    """
    box = box or CMZ
    m = mesh[
        mesh["lat"].between(box["lat_min"], box["lat_max"])
        & mesh["lon"].between(box["lon_min"], box["lon_max"])
    ].copy()
    if m.empty:
        raise ValueError("no mesh points inside the core monsoon zone")
    w = np.cos(np.deg2rad(m["lat"]))
    m["_w"] = w
    m["_wx"] = m["precip_sum"] * w

    keys = ["valid_date", "lead_day"] if "lead_day" in m.columns else ["valid_date"]
    g = m.groupby(keys)
    s = (g["_wx"].sum() / g["_w"].sum()).rename("cmz_rain_fcst").reset_index()
    doy = pd.to_datetime(s["valid_date"]).dt.dayofyear
    s["cmz_rain_anom_fcst"] = (s["cmz_rain_fcst"] - doy.map(clim["mean"])) / doy.map(clim["sd"])
    return s
