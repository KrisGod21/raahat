"""Verification metrics (spec 11.3). Every formula is from the spec verbatim.

Spec 11.2 is explicit about what NOT to report and why, and the reasoning is
worth keeping next to the code so nobody "helpfully" adds them later:

  overall accuracy  meaningless when ~90% of district-days are dry
  R^2               dominated by the zero mass
  MAPE              undefined at zero rainfall
  ROC-AUC           insensitive to calibration, which is the entire point

Sample size travels with every number. Spec 11.4 requires any cell with fewer
than 30 events to be greyed out and labelled insufficient -- a CSI computed on
four events is noise wearing a decimal point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_EVENTS = 30


def contingency(obs: np.ndarray, fcst: np.ndarray, threshold: float) -> dict[str, int]:
    """a=hits b=false alarms c=misses d=correct negatives (spec 11.3)."""
    o = np.asarray(obs, dtype="float64") > threshold
    f = np.asarray(fcst, dtype="float64") > threshold
    ok = np.isfinite(obs) & np.isfinite(fcst)
    o, f = o[ok], f[ok]
    return {
        "a": int((o & f).sum()), "b": int((~o & f).sum()),
        "c": int((o & ~f).sum()), "d": int((~o & ~f).sum()),
    }


def categorical_scores(obs, fcst, threshold: float) -> dict[str, float]:
    t = contingency(obs, fcst, threshold)
    a, b, c, d = t["a"], t["b"], t["c"], t["d"]
    n = a + b + c + d
    a_ref = (a + b) * (a + c) / n if n else np.nan
    denom_ets = a + b + c - a_ref
    return {
        **t,
        "n": n,
        "n_events": a + c,
        "POD": a / (a + c) if (a + c) else np.nan,
        "FAR": b / (a + b) if (a + b) else np.nan,
        "CSI": a / (a + b + c) if (a + b + c) else np.nan,
        "bias_score": (a + b) / (a + c) if (a + c) else np.nan,
        "ETS": (a - a_ref) / denom_ets if denom_ets else np.nan,
        "sufficient": (a + c) >= MIN_EVENTS,
    }


def continuous_scores(obs, fcst) -> dict[str, float]:
    o = np.asarray(obs, dtype="float64")
    f = np.asarray(fcst, dtype="float64")
    ok = np.isfinite(o) & np.isfinite(f)
    o, f = o[ok], f[ok]
    if not len(o):
        return {"n": 0, "bias": np.nan, "mae": np.nan, "rmse": np.nan, "corr": np.nan}
    e = f - o
    return {
        "n": int(len(o)),
        "bias": float(e.mean()),
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e**2).mean())),
        "corr": float(np.corrcoef(o, f)[0, 1]) if len(o) > 2 and o.std() > 0 else np.nan,
    }


def brier(prob, obs, threshold: float) -> dict[str, float]:
    """Brier score and skill against climatology (spec 11.3)."""
    p = np.asarray(prob, dtype="float64")
    o = (np.asarray(obs, dtype="float64") > threshold).astype("float64")
    ok = np.isfinite(p) & np.isfinite(obs)
    p, o = p[ok], o[ok]
    if not len(p):
        return {"brier": np.nan, "bss": np.nan, "n": 0, "base_rate": np.nan}
    bs = float(((p - o) ** 2).mean())
    base = float(o.mean())
    bs_clim = float(((base - o) ** 2).mean())
    return {
        "brier": bs,
        "bss": 1 - bs / bs_clim if bs_clim > 0 else np.nan,
        "n": int(len(p)),
        "base_rate": base,
    }


def reliability(prob, obs, threshold: float, *, bins: int = 10) -> pd.DataFrame:
    """Reliability diagram data: is 70% really 70%? (spec 11.1)"""
    p = np.asarray(prob, dtype="float64")
    o = (np.asarray(obs, dtype="float64") > threshold).astype("float64")
    ok = np.isfinite(p) & np.isfinite(obs)
    p, o = p[ok], o[ok]
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append(
            {
                "bin_lower": edges[b], "bin_upper": edges[b + 1],
                "forecast_prob": float(p[m].mean()) if m.any() else np.nan,
                "observed_freq": float(o[m].mean()) if m.any() else np.nan,
                "n": int(m.sum()),
            }
        )
    return pd.DataFrame(rows)


def crps_from_quantiles(obs, q10, q50, q90) -> float:
    """CRPS approximated from three quantiles via the pinball loss.

    The Level-1 corrector emits quantiles, not a distribution, so a proper
    CRPS is not available. The mean pinball loss over the quantile levels is a
    consistent lower-bound-flavoured stand-in and is directionally comparable
    ACROSS MODELS SCORED THE SAME WAY -- it is not comparable to the true CRPS
    the Level-2 CSGD will produce. Reported as `pinball`, never as `CRPS`, so
    the two never get mixed up in a table.
    """
    o = np.asarray(obs, dtype="float64")
    out = []
    for alpha, q in ((0.1, q10), (0.5, q50), (0.9, q90)):
        qq = np.asarray(q, dtype="float64")
        ok = np.isfinite(o) & np.isfinite(qq)
        if not ok.any():
            continue
        d = o[ok] - qq[ok]
        out.append(np.maximum(alpha * d, (alpha - 1) * d).mean())
    return float(np.mean(out)) if out else np.nan


def scorecard(
    df: pd.DataFrame, obs_col: str, fcst_col: str, thresholds: list[float]
) -> pd.DataFrame:
    """One row per threshold: continuous scores plus categorical scores."""
    cont = continuous_scores(df[obs_col], df[fcst_col])
    rows = []
    for t in thresholds:
        s = categorical_scores(df[obs_col], df[fcst_col], t)
        rows.append({"threshold": t, **cont, **{k: v for k, v in s.items() if k != "n"}})
    return pd.DataFrame(rows)
