"""Build the national Stage-A table from a synoptic mesh (spec 4.3, 5.4).

One row per (valid_date, lead_day). Every feature is derived from SURFACE
fields, because no free upper-air field is reachable -- see docs/DATA_NOTES.md
item 16. The substitutions, and why each is defensible:

  850 hPa vorticity  ->  curl of the 10 m wind, plus closed-MSLP-low detection.
                         A monsoon depression is a closed surface low; that is
                         the classical definition and close to IMD's own.
  500 hPa geopotential -> nothing adequate. WD detection is weak and is
                         reported as such rather than dressed up.
  IVT                ->  2 m relative humidity x 10 m wind. A crude proxy for
                         moisture transport, honest about being one.

Anomalies are taken against a day-of-year climatology fitted on the TRAINING
period only (spec 7.2 trap 4). With one training season that cycle is noisy, so
it is smoothed hard -- its job is only to remove the seasonal march, not to
define a normal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from raahat import contract as C

BOXES = C.STAGE_A_BOXES
_KEYS = ["valid_date", "lead_day"]


def _keys(mesh: pd.DataFrame) -> list[str]:
    return _KEYS if "lead_day" in mesh.columns else ["valid_date"]


def _in_box(mesh: pd.DataFrame, name: str) -> pd.DataFrame:
    b = BOXES[name]
    return mesh[
        mesh["lat"].between(b["lat"][0], b["lat"][1])
        & mesh["lon"].between(b["lon"][0], b["lon"][1])
    ]


def _agg(mesh: pd.DataFrame, box: str, col: str, how: str, out: str) -> pd.DataFrame:
    sub = _in_box(mesh, box)
    if sub.empty or col not in sub:
        return pd.DataFrame(columns=_keys(mesh) + [out])
    g = sub.groupby(_keys(mesh))[col]
    s = {"mean": g.mean, "max": g.max, "min": g.min, "sum": g.sum}[how]()
    return s.rename(out).reset_index()


def _doy_clim(mesh: pd.DataFrame, col: str, *, window: int = 15) -> pd.Series:
    """Smoothed day-of-year climatology of a national-mean field."""
    keys = _keys(mesh)
    nat = mesh.groupby(keys)[col].mean().reset_index()
    if "lead_day" in nat:
        nat = nat.groupby("valid_date")[col].mean().reset_index()
    nat["doy"] = pd.to_datetime(nat["valid_date"]).dt.dayofyear
    clim = nat.groupby("doy")[col].mean().reindex(range(1, 367)).interpolate(
        limit_direction="both"
    )
    tri = pd.concat([clim, clim, clim])
    sm = tri.rolling(window, center=True, min_periods=1).mean()
    out = sm.iloc[len(clim) : 2 * len(clim)]
    out.index = clim.index
    return out


def _vorticity_max(mesh: pd.DataFrame, box: str) -> pd.DataFrame:
    """Max relative vorticity of the 10 m wind inside a box, per day/lead.

    Centred finite differences on the mesh (spec 4.3). Units 1e-5 /s.
    """
    keys = _keys(mesh)
    sub = _in_box(mesh, box)
    if sub.empty or "u10" not in sub or "v10" not in sub:
        return pd.DataFrame(columns=keys + [f"vort10_max_{box}"])
    rows = []
    for key, day in sub.groupby(keys):
        u = day.pivot_table(index="lat", columns="lon", values="u10", aggfunc="mean")
        v = day.pivot_table(index="lat", columns="lon", values="v10", aggfunc="mean")
        if u.shape[0] < 3 or u.shape[1] < 3:
            continue
        lat = u.index.to_numpy()
        lon = u.columns.to_numpy()
        # metres per degree, latitude-corrected for the zonal direction
        dy = np.gradient(lat) * 110_570.0
        dx = np.gradient(lon) * 111_320.0 * np.cos(np.deg2rad(lat.mean()))
        dvdx = np.gradient(v.to_numpy(), axis=1) / dx[None, :]
        dudy = np.gradient(u.to_numpy(), axis=0) / dy[:, None]
        zeta = (dvdx - dudy) * 1e5
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        rec[f"vort10_max_{box}"] = float(np.nanmax(zeta)) if np.isfinite(zeta).any() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def build_stage_a(
    mesh: pd.DataFrame,
    *,
    cmz_clim: pd.DataFrame,
    onset_doy: int = 152,
    mslp_clim: pd.Series | None = None,
    t2m_clim: pd.Series | None = None,
) -> pd.DataFrame:
    """Mesh -> the national Stage-A feature table."""
    from raahat.labels import active_break as AB, lps_detector as LPS

    keys = _keys(mesh)
    out = mesh[keys].drop_duplicates().reset_index(drop=True)

    def join(df: pd.DataFrame) -> None:
        nonlocal out
        if not df.empty:
            out = out.merge(df, on=keys, how="left")

    # --- active / break, the published index --------------------------------
    ab = AB.forecast_index(mesh, cmz_clim)
    join(ab)
    out = out.sort_values(keys)
    grp = out.groupby("lead_day") if "lead_day" in out else out.groupby(lambda _: 0)
    out["cmz_rain_anom_fcst_3day"] = grp["cmz_rain_anom_fcst"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )

    # --- surface pressure structure -----------------------------------------
    mslp_clim = mslp_clim if mslp_clim is not None else _doy_clim(mesh, "pressure_msl")
    doy = pd.to_datetime(out["valid_date"]).dt.dayofyear
    ref = doy.map(mslp_clim)
    for box, name in (("cmz", "cmz_mslp_anom"), ("nw", "mslp_anom_nw")):
        d = _agg(mesh, box, "pressure_msl", "mean", name)
        join(d)
        if name in out:
            out[name] = out[name] - ref
    for box, name in (("bay", "mslp_min_anom_bay"), ("land_c", "mslp_min_anom_land_c")):
        d = _agg(mesh, box, "pressure_msl", "min", name)
        join(d)
        if name in out:
            out[name] = out[name] - ref

    span = mesh.groupby(keys)["pressure_msl"].agg(lambda s: s.max() - s.min())
    join(span.rename("mslp_grad_max").reset_index())

    # --- closed lows and the trough -----------------------------------------
    if "lead_day" in mesh.columns:
        lps = pd.concat(
            [LPS.detect(g).assign(lead_day=k) for k, g in mesh.groupby("lead_day")],
            ignore_index=True,
        )
        trough = pd.concat(
            [LPS.monsoon_trough_lat(g).assign(lead_day=k) for k, g in mesh.groupby("lead_day")],
            ignore_index=True,
        )
    else:
        lps, trough = LPS.detect(mesh), LPS.monsoon_trough_lat(mesh)
    join(lps)
    join(trough)

    # --- low-level flow ------------------------------------------------------
    join(_agg(mesh, "cmz", "u10", "mean", "u10_cmz_mean"))
    join(_agg(mesh, "cmz", "v10", "mean", "v10_cmz_mean"))
    join(_agg(mesh, "eastcoast", "u10", "mean", "u10_eastcoast_mean"))
    join(_agg(mesh, "westcoast", "u10", "mean", "u10_westcoast_mean"))
    if {"u10_cmz_mean", "v10_cmz_mean"} <= set(out.columns):
        out["wspd10_cmz_mean"] = np.hypot(out["u10_cmz_mean"], out["v10_cmz_mean"])
    for box in ("bay", "land_c"):
        join(_vorticity_max(mesh, box))

    # --- moisture and thermal ------------------------------------------------
    rh = "relative_humidity_2m"
    if rh in mesh:
        join(mesh.groupby(keys)[rh].mean().rename("rh2m_national_mean").reset_index())
        join(_agg(mesh, "cmz", rh, "mean", "rh2m_cmz_mean"))
    if "temperature_2m" in mesh:
        t2m_clim = t2m_clim if t2m_clim is not None else _doy_clim(mesh, "temperature_2m")
        tref = doy.map(t2m_clim)
        for box, name in (("nw", "t2m_anom_nw"), ("cmz", "t2m_anom_cmz")):
            join(_agg(mesh, box, "temperature_2m", "mean", name))
            if name in out:
                out[name] = out[name] - tref

    # --- regional rainfall pattern -------------------------------------------
    rain = "precip_sum" if "precip_sum" in mesh else "precipitation"
    join(_agg(mesh, "bay", rain, "max", "rain_max_bay"))
    join(_agg(mesh, "westcoast", rain, "max", "rain_max_westcoast"))
    join(_agg(mesh, "eastcoast", rain, "max", "rain_max_eastcoast"))
    join(_agg(mesh, "nw", rain, "mean", "rain_mean_nw"))

    # --- calendar -------------------------------------------------------------
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["days_since_onset_national"] = (doy - onset_doy).clip(lower=0)

    for f in C.STAGE_A_FEATURES:
        if f not in out:
            out[f] = np.nan
    out["season"] = pd.to_datetime(out["valid_date"]).dt.year.astype(np.int16)
    out["init_datetime_utc"] = (
        pd.to_datetime(out["valid_date"])
        - pd.to_timedelta(out.get("lead_day", 0), unit="D")
    ).dt.tz_localize("UTC")
    return out
