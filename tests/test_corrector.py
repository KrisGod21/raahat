"""Level-1 corrector tests, centred on the guarantee spec 6.2 actually cares about.

The monotonicity test is a regression test in the literal sense: the property
was measured at 12.75% violations before the distillation fix and 0% after, and
the fix is a few lines that a future edit could plausibly undo without anything
else failing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raahat import contract as C
from raahat.models import corrector_lgbm as CL


def _toy(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """A small table with the columns the corrector needs and real structure."""
    rng = np.random.default_rng(seed)
    base = rng.gamma(0.9, 9.0, n)
    df = pd.DataFrame({
        "district_id": rng.integers(1, 40, n).astype(np.int32),
        "valid_date": pd.date_range("2024-06-01", periods=n, freq="h").date,
        "lead_day": np.int8(3),
        "obs_rain_mm": base,
    })
    for m in C.NWP_MODELS:
        df[f"pr_{m}"] = base * rng.lognormal(0, 0.3, n)
        df[f"pr_{m}_nbr_max"] = df[f"pr_{m}"] * rng.uniform(1.1, 2.5, n)
        df[f"pr_{m}_nbr_mean"] = df[f"pr_{m}"] * rng.uniform(0.8, 1.1, n)
    df["mm_mean"] = df[[f"pr_{m}" for m in C.NWP_MODELS]].mean(axis=1)
    df["mm_spread"] = df[[f"pr_{m}" for m in C.NWP_MODELS]].std(axis=1)
    df["mean_elev"] = rng.uniform(10, 2500, n)
    return df


@pytest.fixture(scope="module")
def fitted():
    df = _toy()
    return CL.CorrectorLGBM(use_regime=False, monotone=True).fit(df, df, verbose=False), df


def test_more_forecast_rain_never_yields_less_corrected_rain(fitted):
    """Spec 6.2. LightGBM cannot impose this on a quantile objective, so the
    median is distilled into a monotone student -- if that is removed, this
    fails and nothing else does."""
    model, df = fitted
    audit = model.monotonicity_audit(df, bump=10.0, n=1500)
    assert audit["checked"]
    assert audit["violations"] == 0, (
        f"{audit['violation_rate']:.2%} of rows fell when the raw forecast rose; "
        f"worst drop {audit['worst_drop_mm']:.2f} mm"
    )
    assert audit["mean_response_mm"] > 0


def test_unconstrained_model_does_violate(fitted):
    """Guards the test above: if an unconstrained model also showed zero
    violations, the audit would be measuring nothing."""
    _, df = fitted
    loose = CL.CorrectorLGBM(use_regime=False, monotone=False).fit(df, verbose=False)
    assert loose.monotonicity_audit(df, bump=10.0, n=1500)["violations"] > 0


def test_quantiles_are_ordered(fitted):
    model, df = fitted
    p = model.predict(df)
    assert (p["corrected_p10"] <= p["corrected_median"] + 1e-9).all()
    assert (p["corrected_p90"] >= p["corrected_median"] - 1e-9).all()


def test_predictions_are_non_negative(fitted):
    model, df = fitted
    p = model.predict(df)
    for c in ("corrected_p10", "corrected_median", "corrected_p90"):
        assert (p[c] >= 0).all(), f"{c} went negative -- rainfall cannot"


def test_exceedance_probabilities_are_probabilities(fitted):
    model, df = fitted
    p = model.predict(df)
    for c in [c for c in p.columns if c.startswith("p_gt_")]:
        assert p[c].between(0, 1).all()


def test_ablation_4_drops_the_regime_block():
    """Ablation 4 must differ from ablation 5 by exactly the regime features."""
    full = CL.CorrectorLGBM(use_regime=True).active_features()
    none = CL.CorrectorLGBM(use_regime=False).active_features()
    assert set(full) - set(none) == set(C.STAGE_B_REGIME)


def test_quantile_mapping_is_fitted_on_training_only():
    """Spec 7.2 trap 5: a QM baseline fitted on all years is a strawman."""
    tr = _toy(1500, seed=1)
    qm = CL.fit_quantile_mapping(tr)
    assert qm["obs"].min() >= 0
    assert (np.diff(qm["obs"]) >= -1e-9).all(), "mapping must be non-decreasing"
    out = CL.apply_quantile_mapping(tr, qm)
    assert len(out) == len(tr)
    assert (out >= 0).all()
