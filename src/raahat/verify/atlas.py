"""The India Regime Error Atlas -- USP 1 (spec 3, 11.4).

For every (synoptic regime x homogeneous zone x lead day), how wrong is the raw
NWP rainfall forecast, and how much of that does RAAHAT remove?

Spec 3 ranks this first among the USPs, and the reasoning is worth restating:
the method is elementary -- stratified verification, nothing more. What does
not exist publicly is the ARTEFACT for India at district resolution. "Not a new
algorithm. A new measurement." That claim is small, true and unattackable,
and unlike the gating claim it does not depend on our model working.

It is also useful to a forecaster who never runs the model, which is what turns
the project from a tool into a finding.

Spec 11.4 is non-negotiable here: every cell carries its event count, and any
cell with fewer than 30 events is marked insufficient rather than reported. A
CSI computed on four events is noise wearing a decimal point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from raahat import contract as C
from raahat.verify import metrics as M


def build(
    df: pd.DataFrame,
    *,
    raw_col: str = "pr_ifs",
    corrected_col: str | None = "corrected_median",
    prob_col: str | None = None,
    threshold: float | None = None,
    alpha: float = 0.15,
    regime_col: str = "regime_argmax",
    zone_col: str = "zone_code",
) -> pd.DataFrame:
    """One row per (regime, zone, lead). Bias, error and skill, raw vs corrected.

    `prob_col` + `alpha` score the corrected side by exceedance probability
    where available (spec 6.4); otherwise the corrected value is thresholded
    directly. Which was used is recorded per row.
    """
    threshold = threshold if threshold is not None else C.THRESHOLDS["heavy"]
    need = {raw_col, C.TARGET_COL, regime_col, zone_col, "lead_day"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"atlas needs {sorted(missing)}")

    rows = []
    grouped = df.groupby([regime_col, zone_col, "lead_day"], dropna=False)
    for (regime, zone, lead), g in grouped:
        obs = g[C.TARGET_COL].to_numpy(dtype="float64")
        raw = g[raw_col].to_numpy(dtype="float64")
        ok = np.isfinite(obs) & np.isfinite(raw)
        if ok.sum() == 0:
            continue

        raw_cont = M.continuous_scores(obs, raw)
        raw_cat = M.categorical_scores(obs, raw, threshold)
        rec = {
            "regime": regime, "zone": zone, "lead_day": int(lead),
            "n_district_days": int(ok.sum()),
            "n_events": raw_cat["n_events"],
            "sufficient": raw_cat["sufficient"],
            "raw_bias": raw_cont["bias"], "raw_rmse": raw_cont["rmse"],
            "raw_corr": raw_cont["corr"],
            "raw_pod": raw_cat["POD"], "raw_far": raw_cat["FAR"],
            "raw_csi": raw_cat["CSI"], "raw_ets": raw_cat["ETS"],
            "obs_mean": float(np.nanmean(obs[ok])),
        }

        if corrected_col and corrected_col in g:
            cor = g[corrected_col].to_numpy(dtype="float64")
            cor_cont = M.continuous_scores(obs, cor)
            if prob_col and prob_col in g:
                p = g[prob_col].to_numpy(dtype="float64")
                cor_cat = M.categorical_scores(
                    obs, np.where(p > alpha, threshold + 1e-6, 0.0), threshold
                )
                rec["scored_from"] = f"P>{alpha:g}"
            else:
                cor_cat = M.categorical_scores(obs, cor, threshold)
                rec["scored_from"] = "value"
            rec.update(
                cor_bias=cor_cont["bias"], cor_rmse=cor_cont["rmse"],
                cor_corr=cor_cont["corr"], cor_pod=cor_cat["POD"],
                cor_far=cor_cat["FAR"], cor_csi=cor_cat["CSI"],
                cor_ets=cor_cat["ETS"],
                d_bias=abs(cor_cont["bias"]) - abs(raw_cont["bias"]),
                d_rmse=cor_cont["rmse"] - raw_cont["rmse"],
                d_csi=cor_cat["CSI"] - raw_cat["CSI"],
            )
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.attrs["threshold"] = threshold
    return out.sort_values(["lead_day", "regime", "zone"]).reset_index(drop=True)


def heatmap(atlas: pd.DataFrame, *, value: str = "raw_bias", lead: int = 3,
            annotate: str | None = "d_csi") -> pd.DataFrame:
    """Regimes on rows, zones on columns -- the object for the USP-1 slide.

    Insufficient cells come back as NaN so they render greyed out rather than
    quietly contributing a number nobody should trust.
    """
    sub = atlas[atlas["lead_day"] == lead].copy()
    sub.loc[~sub["sufficient"], [value] + ([annotate] if annotate else [])] = np.nan
    grid = sub.pivot(index="regime", columns="zone", values=value)
    order = [r for r in C.REGIMES if r in grid.index]
    grid = grid.reindex(order)
    if annotate and annotate in sub:
        grid.attrs["annotation"] = sub.pivot(
            index="regime", columns="zone", values=annotate
        ).reindex(order)
    grid.attrs["n_events"] = sub.pivot(
        index="regime", columns="zone", values="n_events"
    ).reindex(order)
    return grid


def format_heatmap(grid: pd.DataFrame, *, title: str = "", unit: str = "mm") -> str:
    """Terminal rendering, with n in every cell as spec 11.4 requires."""
    ann = grid.attrs.get("annotation")
    nev = grid.attrs.get("n_events")
    zones = list(grid.columns)
    lines = []
    if title:
        lines += [title, "=" * max(len(title), 60)]
    lines.append(f"  {'regime':<18}" + "".join(f"{z:>20}" for z in zones))
    for regime in grid.index:
        cells = []
        for z in zones:
            v = grid.loc[regime, z]
            n = int(nev.loc[regime, z]) if nev is not None and pd.notna(nev.loc[regime, z]) else 0
            if pd.isna(v):
                cells.append(f"{'insuff. (n=' + str(n) + ')':>20}")
            else:
                a = ann.loc[regime, z] if ann is not None else np.nan
                txt = f"{v:+.1f}{unit}"
                if pd.notna(a):
                    txt += f" {a:+.3f}"
                cells.append(f"{txt + ' n=' + str(n):>20}")
        lines.append(f"  {regime:<18}" + "".join(cells))
    lines.append(f"\n  cell = raw bias ({unit}); second number = change in CSI from "
                 f"correction; n = heavy-rain events")
    lines.append(f"  cells with n < {M.MIN_EVENTS} are marked insufficient (spec 11.4)")
    return "\n".join(lines)
