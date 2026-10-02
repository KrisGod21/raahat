"""Why did the model say that? (spec 9.4, 13.2, 14)

Uses LightGBM's native `pred_contrib=True`, which computes exact TreeSHAP
inside the booster. Spec 8 lists the `shap` package, but pulling it in would
add a dependency for something the model already does exactly and faster --
and an approximation would be worse, not merely different.

Contributions are in the units of the prediction, so a driver reading
"+18.4 mm" means that feature moved the corrected median up by 18.4 mm. That
is directly sayable to a forecaster, which is the whole point: spec 9.4 wants
"top-3 drivers", not a feature-importance bar chart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from raahat import contract as C

#: Forecaster-readable names. Anything absent falls back to the raw column,
#: which is ugly but honest -- a missing label should be visible, not silent.
LABELS: dict[str, str] = {
    "mm_mean": "multi-model mean rainfall",
    "mm_spread": "disagreement between models",
    "mm_agree_heavy": "models agreeing on heavy rain",
    "pr_ifs": "ECMWF forecast rainfall",
    "pr_icon": "ICON forecast rainfall",
    "pr_ifs_nbr_max": "heaviest rain forecast nearby",
    "pr_icon_nbr_max": "heaviest rain forecast nearby (ICON)",
    "pr_ifs_nbr_mean": "rainfall across the surrounding districts",
    "pr_ifs_grad": "sharpness of the rain edge nearby",
    "pr_ifs_3day_sum": "three-day forecast total",
    "pr_ifs_prev_day": "rain forecast the previous day",
    "p_LPS": "chance of a low-pressure system",
    "p_ACTIVE": "chance of an active monsoon spell",
    "p_BREAK": "chance of a break in the monsoon",
    "p_WEAK": "chance of unsettled/weak conditions",
    "regime_entropy": "uncertainty about the weather pattern",
    "p_LPS_trend": "low-pressure system approaching or departing",
    "mean_elev": "district elevation",
    "elev_range": "terrain relief within the district",
    "orog_exposure": "exposure of slopes to the monsoon flow",
    "upslope_flux": "wind forcing air up the slopes",
    "coast_onshore_flux": "onshore moisture flow",
    "dist_to_coast_km": "distance from the coast",
    "clim_mean_district_doy": "what is normal here at this time of year",
    "clim_p90_district_doy": "a wet day here at this time of year",
    "clim_p99_district_doy": "an extreme day here at this time of year",
    "days_since_local_onset": "days since the monsoon arrived locally",
    "doy_sin": "time of season", "doy_cos": "time of season",
}


#: Per-model variants, generated rather than listed, so adding an NWP model to
#: config/features.yaml cannot leave raw column names like `pr_icon_nbr_mean`
#: showing in a forecaster's explanation panel.
_MODEL_NAME = {"ifs": "ECMWF", "icon": "ICON", "gfs": "GFS"}
_SUFFIX_TEXT = {
    "": "{m} forecast rainfall",
    "nbr_max": "heaviest rain forecast nearby ({m})",
    "nbr_mean": "rainfall across the surrounding districts ({m})",
    "nbr_p90": "rain in the wettest nearby districts ({m})",
    "grad": "sharpness of the rain edge nearby ({m})",
    "prev_day": "rain forecast the previous day ({m})",
    "3day_sum": "three-day forecast total ({m})",
}
for _short, _name in _MODEL_NAME.items():
    for _sfx, _text in _SUFFIX_TEXT.items():
        _key = f"pr_{_short}" + (f"_{_sfx}" if _sfx else "")
        LABELS.setdefault(_key, _text.format(m=_name))


def label(feature: str) -> str:
    """Forecaster-readable name, falling back to a tidied column name.

    A raw identifier appearing in the explanation panel is a visible bug, so
    the fallback at least reads as words rather than snake_case.
    """
    if feature in LABELS:
        return LABELS[feature]
    return feature.replace("_", " ")


def contributions(model, X: pd.DataFrame) -> pd.DataFrame:
    """Per-row, per-feature SHAP contributions in prediction units.

    The last column returned by LightGBM is the base value (the expected
    prediction), which is not a feature and is dropped here -- including it as
    a "driver" would put "the average district-day" at the top of every
    explanation.
    """
    booster = model.booster_ if hasattr(model, "booster_") else model.booster
    raw = booster.predict(X, pred_contrib=True)
    raw = np.asarray(raw)
    return pd.DataFrame(raw[:, :-1], columns=list(X.columns), index=X.index)


def top_drivers(
    model, X: pd.DataFrame, *, k: int = 3, signed: bool = True
) -> pd.DataFrame:
    """The k features that moved each prediction most, with direction.

    Ranked by absolute contribution: a driver that pulled the forecast DOWN is
    as informative as one that pushed it up, and hiding the negative ones would
    make the explanation a sales pitch rather than an account.
    """
    contrib = contributions(model, X)
    vals = contrib.to_numpy()
    order = np.argsort(-np.abs(vals), axis=1)[:, :k]
    cols = np.array(contrib.columns)

    out = pd.DataFrame(index=X.index)
    for i in range(k):
        idx = order[:, i]
        feat = cols[idx]
        val = vals[np.arange(len(vals)), idx]
        out[f"shap_top{i + 1}_feature"] = feat
        out[f"shap_top{i + 1}_value"] = val.astype(np.float32)
        if signed:
            out[f"shap_top{i + 1}_label"] = [label(f) for f in feat]
            out[f"shap_top{i + 1}_direction"] = np.where(val >= 0, "raised", "lowered")
    return out


def narrative(row: pd.Series, *, k: int = 3, unit: str = "mm") -> str:
    """One forecaster-readable sentence. Templated, never model-authored.

    Spec 18 rules out an LLM deciding content. The wording is fixed here and
    only the numbers vary, so the sentence can never say something the model
    did not compute.
    """
    parts = []
    for i in range(1, k + 1):
        f = row.get(f"shap_top{i}_feature")
        v = row.get(f"shap_top{i}_value")
        if f is None or v is None or (isinstance(v, float) and not np.isfinite(v)):
            continue
        verb = "raised" if v >= 0 else "lowered"
        parts.append(f"{label(str(f))} {verb} it by {abs(float(v)):.1f} {unit}")
    if not parts:
        return "No single driver dominated this forecast."
    if len(parts) == 1:
        return f"Mainly, {parts[0]}."
    return f"Mainly, {parts[0]}; then {'; then '.join(parts[1:])}."


def attach(model, df: pd.DataFrame, features: list[str], *, k: int = 3) -> pd.DataFrame:
    """Convenience: drivers plus narrative, indexed like `df`."""
    use = [f for f in features if f in df.columns]
    drivers = top_drivers(model, df[use], k=k)
    drivers["explanation"] = [narrative(r, k=k) for _, r in drivers.iterrows()]
    return drivers
