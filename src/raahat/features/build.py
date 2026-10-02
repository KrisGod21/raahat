"""Assemble features_{season}.parquet from the interim tables (spec 5, 13.2).

One row per (district_id, valid_date, lead_day). Every column name comes from
contract.py, which reads config/features.yaml -- nothing is hard-coded here.

The features that matter most, and why:

  pr_*_nbr_max   Spec 5.1 calls this the single most important predictor of
                 heavy rain, because at district scale the model usually gets
                 the AMOUNT roughly right and the LOCATION wrong. A district
                 whose neighbours are forecast 120 mm is at risk even if its
                 own cell says 40.

  upslope_flux   Spec 5.5. Low-level wind dotted with the terrain gradient,
                 scaled by humidity. Positive and large = air being forced up a
                 slope through wet air, which is orographic rain happening now.
                 This is what replaces a separate "orographic regime".

  clim_*         Fitted on TRAINING YEARS ONLY (spec 7.2 trap 4). Fitting
                 normals on the full record leaks the test period into every
                 anomaly and is one of the easiest ways to fool yourself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from raahat import contract as C

INTERIM = C.DATA_DIR / "interim"
STATIC = C.DATA_DIR / "static"
PROCESSED = C.DATA_DIR / "processed"

#: Open-Meteo model id -> the short name used in contract column names.
MODEL_ALIAS = {"ecmwf_ifs025": "ifs", "icon_seamless": "icon", "gfs_seamless": "gfs"}

#: Years used to fit day-of-year normals. Must exclude validation and test.
CLIMATOLOGY_YEARS = list(range(2010, 2024))


def _pivot_models(fcst: pd.DataFrame) -> pd.DataFrame:
    """Long (one row per model) -> wide (pr_ifs, pr_icon, ...)."""
    f = fcst[fcst["lead_day"] > 0].copy()          # lead 0 is the analysis
    f["short"] = f["model"].map(MODEL_ALIAS).fillna(f["model"])
    wide = f.pivot_table(
        index=["district_id", "valid_date", "lead_day"],
        columns="short", values="pr_mm", aggfunc="first",
    )
    wide.columns = [f"pr_{c}" for c in wide.columns]
    return wide.reset_index()


def _neighbourhood(wide: pd.DataFrame, nbrs: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """nbr_mean / nbr_max / nbr_p90 / grad over each district AND its neighbours.

    Self is included in the neighbourhood: a district is its own first
    neighbour, and excluding it would make nbr_mean diverge from pr on isolated
    units.
    """
    self_edges = pd.DataFrame(
        {"district_id": master["district_id"], "neighbour_id": master["district_id"]}
    )
    edges = pd.concat([nbrs, self_edges], ignore_index=True).drop_duplicates()

    pr_cols = [c for c in wide.columns if c.startswith("pr_")]
    joined = edges.merge(
        wide.rename(columns={"district_id": "neighbour_id"}),
        on="neighbour_id", how="inner",
    )
    g = joined.groupby(["district_id", "valid_date", "lead_day"])
    agg = g[pr_cols].agg(["mean", "max", "min", lambda s: s.quantile(0.9)])
    agg.columns = [
        f"{c}_nbr_{ {'mean':'mean','max':'max','min':'min','<lambda_0>':'p90'}[s] }"
        for c, s in agg.columns
    ]
    return agg.reset_index()


def _climatology(master: pd.DataFrame) -> pd.DataFrame:
    """Day-of-year normals and percentiles per district, training years only."""
    from raahat.ingest import imd_grid

    have = [y for y in CLIMATOLOGY_YEARS if y in imd_grid.cached_years()]
    if len(have) < 3:
        raise FileNotFoundError(
            f"climatology needs several years; only {have} are cached. "
            "Run the IMD climatology fetch first."
        )
    da = imd_grid.load(have)
    lats = __import__("xarray").DataArray(master["lat_centroid"].to_numpy(), dims="district")
    lons = __import__("xarray").DataArray(master["lon_centroid"].to_numpy(), dims="district")
    sub = da.sel(lat=lats, lon=lons, method="nearest")

    df = sub.to_pandas()
    df.columns = master["district_id"].to_numpy()
    long = df.stack(future_stack=True).rename("rain").reset_index()
    long.columns = ["time", "district_id", "rain"]
    long["doy"] = pd.to_datetime(long["time"]).dt.dayofyear

    g = long.groupby(["district_id", "doy"])["rain"]
    clim = pd.DataFrame(
        {
            "clim_mean_district_doy": g.mean(),
            "clim_p90_district_doy": g.quantile(0.90),
            "clim_p99_district_doy": g.quantile(0.99),
        }
    ).reset_index()

    # 5-day smoothing (spec 4.5): a single day-of-year over ~14 years is noisy
    clim = clim.sort_values(["district_id", "doy"])
    for col in ("clim_mean_district_doy", "clim_p90_district_doy", "clim_p99_district_doy"):
        clim[col] = (
            clim.groupby("district_id")[col]
            .transform(lambda s: s.rolling(5, center=True, min_periods=1).mean())
        )
    clim.attrs["years"] = have
    return clim


def build_season(season: int, *, regime_probs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build one season's contract-conforming feature table."""
    master = pd.read_csv(STATIC / "districts_master.csv")
    nbrs = pd.read_parquet(STATIC / "district_neighbours.parquet")
    terrain = pd.read_parquet(STATIC / "district_terrain.parquet")
    obs = pd.read_parquet(INTERIM / f"district_obs_{season}.parquet")
    fcst = pd.read_parquet(INTERIM / f"district_fcst_{season}.parquet")

    wide = _pivot_models(fcst)
    nbr = _neighbourhood(wide, nbrs, master)
    df = wide.merge(nbr, on=["district_id", "valid_date", "lead_day"], how="left")

    models = [c[3:] for c in wide.columns if c.startswith("pr_")]
    for m in models:
        base, mx, mn = f"pr_{m}", f"pr_{m}_nbr_max", f"pr_{m}_nbr_min"
        # a sharp rain edge across the neighbourhood = high displacement risk
        df[f"pr_{m}_grad"] = (df[mx] - df[mn]).astype("float32")
        df = df.drop(columns=[mn])

    df = df.sort_values(["district_id", "lead_day", "valid_date"])
    for m in models:
        g = df.groupby(["district_id", "lead_day"])[f"pr_{m}"]
        df[f"pr_{m}_prev_day"] = g.shift(1)
        df[f"pr_{m}_3day_sum"] = g.transform(
            lambda s: s.rolling(3, min_periods=1).sum()
        )

    stack = df[[f"pr_{m}" for m in models]].to_numpy(dtype="float64")
    df["mm_mean"] = np.nanmean(stack, axis=1)
    df["mm_median"] = np.nanmedian(stack, axis=1)
    df["mm_spread"] = np.nanstd(stack, axis=1)
    df["mm_range"] = np.nanmax(stack, axis=1) - np.nanmin(stack, axis=1)
    df["mm_pop"] = np.nanmean(stack > 0.2, axis=1)
    df["mm_agree_heavy"] = np.nanmean(stack > C.THRESHOLDS["heavy"], axis=1)
    for c in C.STAGE_B_ENSEMBLE:
        df[c] = np.nan                       # ECMWF ENS is Level 2

    df = df.merge(obs[["district_id", "valid_date", "obs_rain_mm"]],
                  on=["district_id", "valid_date"], how="left")
    df = df.merge(
        master[["district_id", "zone_code", "subdivision", "lat_centroid", "lon_centroid"]],
        on="district_id", how="left",
    )
    df = df.merge(terrain, on="district_id", how="left")

    d = pd.to_datetime(df["valid_date"])
    doy = d.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["season"] = np.int16(season)
    df["init_datetime_utc"] = (
        pd.to_datetime(df["valid_date"]) - pd.to_timedelta(df["lead_day"], unit="D")
    ).dt.tz_localize("UTC")

    clim = _climatology(master)
    df["doy"] = doy
    df = df.merge(clim, on=["district_id", "doy"], how="left")

    onset = (
        df[df["obs_rain_mm"].notna()]
        .assign(wet=lambda x: x["obs_rain_mm"] > 5)
        .groupby("district_id")
        .apply(lambda g: g.loc[g["wet"], "doy"].min() if g["wet"].any() else 152,
               include_groups=False)
        .rename("onset_doy")
    )
    df = df.merge(onset, on="district_id", how="left")
    df["days_since_local_onset"] = (df["doy"] - df["onset_doy"]).clip(lower=0)

    if regime_probs is not None:
        df = df.merge(regime_probs, on=["valid_date", "lead_day"], how="left")
    for c in C.REGIME_PROB_COLS + ["regime_entropy", "p_LPS_trend"]:
        if c not in df:
            df[c] = np.nan
    if "regime_argmax" not in df:
        df["regime_argmax"] = pd.NA
    for c in ("label_regime",):
        if c not in df:
            df[c] = pd.NA
    if "label_regime_soft" not in df:
        df["label_regime_soft"] = [np.full(C.N_REGIMES, np.nan, dtype=np.float32)] * len(df)

    for c in C.STAGE_B_SYNOPTIC:
        if c not in df:
            df[c] = np.nan                   # filled from the mesh by a later pass

    cols = [f.name for f in C.features_schema()]
    for c in cols:
        if c not in df:
            df[c] = np.nan
    out = df[cols].copy()
    f32 = [f.name for f in C.features_schema() if f.type == "float"]
    out[f32] = out[f32].astype(np.float32)
    out["district_id"] = out["district_id"].astype(np.int32)
    out["lead_day"] = out["lead_day"].astype(np.int8)
    out["season"] = out["season"].astype(np.int16)
    return out


def terrain_interaction(df: pd.DataFrame, mesh: pd.DataFrame | None = None) -> pd.DataFrame:
    """upslope_flux and coast_onshore_flux (spec 5.5).

    Needs low-level wind, which comes from the mesh. Separated from
    build_season so the feature table can exist before the mesh lands.
    """
    if mesh is None or "u850" not in df or df["u850"].isna().all():
        return df
    # wind . terrain gradient, scaled by humidity: air forced up a wet slope
    df["upslope_flux"] = (
        (df["u850"] * df["grad_east"] + df["v850"] * df["grad_north"])
        * (df["rh850"].fillna(70.0) / 100.0)
    ).astype("float32")
    bearing = np.deg2rad(df["coast_normal_bearing"].fillna(270.0))
    onshore = df["u850"] * np.sin(bearing) + df["v850"] * np.cos(bearing)
    df["coast_onshore_flux"] = (
        onshore * df["ivt_mag"].fillna(300.0) / 1000.0
        * np.exp(-df["dist_to_coast_km"].fillna(999.0) / 200.0)
    ).astype("float32")
    return df
