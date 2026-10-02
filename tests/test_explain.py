"""Explanation-layer tests (spec 3 USP 4, 9.4).

These outputs are read by a human and quoted back, so the tests check the
things a forecaster would notice being wrong -- a driver pointing the wrong
way, an analog that is really the day itself, a sentence that claims something
the model did not compute.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raahat import contract as C
from raahat.explain import shap_drivers as SH
from raahat.models import analogs as AN


@pytest.fixture(scope="module")
def fitted():
    import lightgbm as lgb
    rng = np.random.default_rng(0)
    n = 1500
    X = pd.DataFrame({
        "mm_mean": rng.gamma(1.0, 8.0, n),
        "pr_ifs_nbr_max": rng.gamma(1.2, 10.0, n),
        "mean_elev": rng.uniform(10, 2000, n),
    })
    # target depends strongly and positively on mm_mean
    y = X["mm_mean"] * 1.7 + rng.normal(0, 2, n)
    m = lgb.LGBMRegressor(n_estimators=120, num_leaves=15, verbose=-1).fit(X, y)
    return m, X


# --------------------------------------------------------------- drivers ---


def test_contributions_exclude_the_base_value(fitted):
    """LightGBM appends the expected prediction as a final column. Treating it
    as a feature would put 'the average district-day' atop every explanation."""
    m, X = fitted
    c = SH.contributions(m, X)
    assert list(c.columns) == list(X.columns)
    assert len(c) == len(X)


def test_top_driver_is_the_feature_that_actually_drives_it(fitted):
    m, X = fitted
    top = SH.top_drivers(m, X, k=3)
    # mm_mean generated the target, so it should dominate most rows
    share = (top["shap_top1_feature"] == "mm_mean").mean()
    assert share > 0.7, f"mm_mean led only {share:.0%} of explanations"


def test_direction_matches_the_sign(fitted):
    m, X = fitted
    top = SH.top_drivers(m, X, k=2)
    pos = top["shap_top1_value"] >= 0
    assert (top.loc[pos, "shap_top1_direction"] == "raised").all()
    assert (top.loc[~pos, "shap_top1_direction"] == "lowered").all()


def test_negative_drivers_are_not_hidden(fitted):
    """A driver that pulled the forecast DOWN is as informative as one that
    pushed it up; ranking by absolute value must surface both."""
    m, X = fitted
    top = SH.top_drivers(m, X, k=3)
    assert (top["shap_top1_value"] < 0).any()


def test_labels_are_forecaster_readable():
    assert SH.label("upslope_flux") == "wind forcing air up the slopes"
    assert SH.label("p_LPS") == "chance of a low-pressure system"


def test_per_model_labels_are_generated_for_every_nwp_model():
    """A raw column name in the explanation panel is a visible bug. `pr_icon_*`
    reached the UI once because only the ECMWF variants had been listed by
    hand, so the variants are now generated from the model list."""
    for model in C.NWP_MODELS:
        for col in (f"pr_{model}", f"pr_{model}_nbr_max", f"pr_{model}_nbr_mean"):
            assert "_" not in SH.label(col), f"{col} still reads as a column name"


def test_unknown_feature_falls_back_to_words_not_snake_case():
    assert SH.label("some_new_feature") == "some new feature"


def test_narrative_is_templated_and_mentions_the_numbers(fitted):
    m, X = fitted
    top = SH.top_drivers(m, X, k=3)
    s = SH.narrative(top.iloc[0])
    assert s.startswith("Mainly,")
    assert "mm" in s
    assert any(w in s for w in ("raised", "lowered"))


def test_narrative_degrades_gracefully_with_no_drivers():
    assert "No single driver" in SH.narrative(pd.Series(dtype=object))


# --------------------------------------------------------------- analogs ---


def _history(n=400, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "district_id": np.repeat([1, 2], n // 2),
        "valid_date": list(pd.date_range("2024-06-01", periods=n // 2).date) * 2,
        "mm_mean": rng.gamma(1, 9, n),
        "mm_spread": rng.gamma(1, 2, n),
        "p_LPS": rng.random(n),
        "doy_sin": rng.random(n),
        C.TARGET_COL: rng.gamma(1, 11, n),
    })


def test_analogs_retrieve_within_district_by_default():
    """'A similar day somewhere in India' is a far weaker statement than
    'a similar day here', and the outcome is only interpretable if the
    terrain matches."""
    h = _history()
    mem = AN.AnalogMemory(features=["mm_mean", "mm_spread", "p_LPS"]).fit(h)
    assert set(mem._store) == {1, 2}
    got = mem.query(h[h.district_id == 1].head(5), k=2)
    dates_d1 = set(h[h.district_id == 1].valid_date)
    for d in got["analog_date_1"].dropna():
        assert d in dates_d1


def test_analog_never_returns_the_query_day_itself():
    h = _history()
    mem = AN.AnalogMemory(features=["mm_mean", "mm_spread", "p_LPS"]).fit(h)
    sub = h.head(20)
    got = mem.query(sub, k=2)
    assert not (got["analog_date_1"].to_numpy() == sub["valid_date"].to_numpy()).any()


def test_analogs_only_index_days_with_a_known_outcome():
    """An analog whose outcome we cannot state is a coincidence, not a memory."""
    h = _history()
    h.loc[h.index[:100], C.TARGET_COL] = np.nan
    mem = AN.AnalogMemory(features=["mm_mean", "mm_spread", "p_LPS"]).fit(h)
    total = sum(len(v) for v in mem._store.values())
    assert total == len(h) - 100


def test_analog_narrative_reads_like_the_spec_example():
    row = pd.Series({"analog_date_1": pd.Timestamp("2024-07-30").date(),
                     "analog_obs_1": 320.0,
                     "analog_date_2": pd.Timestamp("2024-08-16").date(),
                     "analog_obs_2": 180.0})
    s = AN.narrative(row, "Wayanad")
    assert "Wayanad" in s and "320 mm" in s and "180 mm" in s
    assert "most closely resembles" in s


def test_analog_narrative_degrades_when_nothing_matches():
    assert "No closely matching day" in AN.narrative(pd.Series(dtype=object))
