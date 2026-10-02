"""National regime features from FORECAST RAINFALL alone (no mesh required).

Why this exists: Open-Meteo's daily quota ran out with the synoptic forecast
mesh only 25% fetched, and no free upper-air field is reachable anyway (see
docs/DATA_NOTES.md items 16 and 25). Rather than leave the central USP
untested, this builds a reduced Stage-A feature set from the district forecast
table that is ALREADY CACHED -- zero new API calls.

It is a genuine reduction, not a substitute, and the reduction is stated
plainly wherever the results appear:

  supported    ACTIVE / BREAK. Spec 4.5's published index is defined on core-
               monsoon-zone RAINFALL, so these labels are exactly as intended.
  weakened     LPS. Detectable here only through the spatial SIGNATURE of the
               rain (concentrated, coherent centroid in the LPS corridor)
               rather than through a closed pressure low.
  unsupported  WD, EASTERLY_COASTAL. Both need circulation. Not attempted.

The spatial features are the interesting part. A monsoon depression produces
concentrated rain with a definable centroid; an active spell produces
widespread rain; a break shifts the rain belt north towards the foothills. So
the CENTROID LATITUDE of forecast rainfall is a usable proxy for the monsoon
trough position, and the CONCENTRATION distinguishes a depression from a
broad active spell. Both come free from data we already hold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from raahat import contract as C

_cfg = C.load_config("regimes")
CMZ = _cfg["core_monsoon_zone"]

#: Feature names this module produces. Stage A uses whichever of these exist.
RAINFALL_REGIME_FEATURES = [
    "cmz_rain_fcst", "cmz_rain_anom_fcst", "cmz_rain_anom_fcst_3day",
    "nat_rain_mean", "nat_rain_max", "nat_frac_wet", "nat_frac_heavy",
    "rain_mean_W", "rain_mean_C", "rain_mean_N",
    "rain_max_W", "rain_max_C", "rain_max_N",
    "contrast_W_C", "contrast_N_C",
    "rain_centroid_lat", "rain_centroid_lon", "rain_concentration",
    "rain_spread_km", "cmz_rain_frac_of_national",
    "doy_sin", "doy_cos", "days_since_onset_national",
]


def _weighted_centroid(lat, lon, w):
    tot = np.nansum(w)
    if tot <= 0:
        return np.nan, np.nan, np.nan
    clat = float(np.nansum(lat * w) / tot)
    clon = float(np.nansum(lon * w) / tot)
    # rms distance of the rain from its own centroid, in km
    dx = (lon - clon) * 111.32 * np.cos(np.deg2rad(clat))
    dy = (lat - clat) * 110.57
    spread = float(np.sqrt(np.nansum((dx**2 + dy**2) * w) / tot))
    return clat, clon, spread


def build(
    fcst_wide: pd.DataFrame,
    master: pd.DataFrame,
    *,
    rain_col: str = "mm_mean",
    clim: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """District forecast table -> one national row per (valid_date, lead_day).

    `fcst_wide` needs district_id, valid_date, lead_day and `rain_col`.
    """
    m = master.set_index("district_id")
    df = fcst_wide[["district_id", "valid_date", "lead_day", rain_col]].copy()
    df["lat"] = df["district_id"].map(m["lat_centroid"])
    df["lon"] = df["district_id"].map(m["lon_centroid"])
    df["zone"] = df["district_id"].map(m["zone_code"])
    df["in_cmz"] = (
        df["lat"].between(CMZ["lat_min"], CMZ["lat_max"])
        & df["lon"].between(CMZ["lon_min"], CMZ["lon_max"])
    )
    # cos-latitude weighting so a northern district is not over-counted
    df["w"] = np.cos(np.deg2rad(df["lat"]))

    rows = []
    for (date, lead), g in df.groupby(["valid_date", "lead_day"], sort=True):
        r = g[rain_col].to_numpy(dtype="float64")
        w = g["w"].to_numpy()
        cm = g["in_cmz"].to_numpy()
        rec = {"valid_date": date, "lead_day": np.int8(lead)}

        rec["cmz_rain_fcst"] = (
            float(np.nansum(r[cm] * w[cm]) / np.nansum(w[cm])) if cm.any() else np.nan
        )
        rec["nat_rain_mean"] = float(np.nansum(r * w) / np.nansum(w))
        rec["nat_rain_max"] = float(np.nanmax(r)) if np.isfinite(r).any() else np.nan
        rec["nat_frac_wet"] = float(np.nanmean(r > 1.0))
        rec["nat_frac_heavy"] = float(np.nanmean(r > C.THRESHOLDS["heavy"]))
        tot = np.nansum(r * w)
        rec["cmz_rain_frac_of_national"] = (
            float(np.nansum(r[cm] * w[cm]) / tot) if tot > 0 and cm.any() else np.nan
        )

        for z in ("W", "C", "N"):
            sel = (g["zone"] == z).to_numpy()
            rec[f"rain_mean_{z}"] = float(np.nanmean(r[sel])) if sel.any() else np.nan
            rec[f"rain_max_{z}"] = float(np.nanmax(r[sel])) if sel.any() else np.nan
        rec["contrast_W_C"] = rec["rain_mean_W"] - rec["rain_mean_C"]
        rec["contrast_N_C"] = rec["rain_mean_N"] - rec["rain_mean_C"]

        # Where is the rain, and how concentrated? An LPS is compact with a
        # coherent centroid; an active spell is broad; a break pushes the belt
        # north towards the foothills.
        clat, clon, spread = _weighted_centroid(
            g["lat"].to_numpy(), g["lon"].to_numpy(), np.nan_to_num(r * w)
        )
        rec["rain_centroid_lat"] = clat
        rec["rain_centroid_lon"] = clon
        rec["rain_spread_km"] = spread
        # share of total rain falling in the wettest decile of districts
        k = max(int(0.1 * len(r)), 1)
        top = np.sort(np.nan_to_num(r))[-k:].sum()
        rec["rain_concentration"] = float(top / tot) if tot > 0 else np.nan
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values(["lead_day", "valid_date"])

    doy = pd.to_datetime(out["valid_date"]).dt.dayofyear
    if clim is not None:
        out["cmz_rain_anom_fcst"] = (
            out["cmz_rain_fcst"] - doy.map(clim["mean"])
        ) / doy.map(clim["sd"])
    else:
        mu, sd = out["cmz_rain_fcst"].mean(), out["cmz_rain_fcst"].std()
        out["cmz_rain_anom_fcst"] = (out["cmz_rain_fcst"] - mu) / (sd or 1.0)
    out["cmz_rain_anom_fcst_3day"] = out.groupby("lead_day")[
        "cmz_rain_anom_fcst"
    ].transform(lambda s: s.rolling(3, min_periods=1).mean())

    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["days_since_onset_national"] = (doy - 152).clip(lower=0)
    out["season"] = pd.to_datetime(out["valid_date"]).dt.year.astype(np.int16)
    return out.reset_index(drop=True)
