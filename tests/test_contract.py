"""The data contract is frozen in Phase 1 and everyone codes against it.

These tests exist so that a change to config/features.yaml that breaks a
downstream workstream fails here rather than in someone else's module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raahat import contract as C


def test_regime_order_is_stable():
    """Regime order is load-bearing: probability vectors follow it."""
    assert C.REGIMES == [
        "ACTIVE", "BREAK", "LPS", "WD", "EASTERLY_COASTAL", "WEAK_TRANSITION",
    ]
    assert C.N_REGIMES == 6
    assert C.REGIME_PROB_COLS[-1] == "p_WEAK"  # spec 13.2 uses the short form
    assert len(C.REGIME_PROB_COLS) == C.N_REGIMES


def test_imd_thresholds_exact():
    """These are policy numbers, not tunables."""
    assert C.THRESHOLD_VALUES == [64.5, 115.6, 204.5]


def test_schemas_have_no_duplicate_columns():
    for schema in (C.features_schema(), C.regime_input_schema(), C.predictions_schema()):
        names = [f.name for f in schema]
        assert len(names) == len(set(names)), "duplicate column in schema"


def test_ablation_feature_set_drops_exactly_the_regime_block():
    """Ablation 4 of spec 6.3 is 'the same model, regime information removed'."""
    dropped = set(C.STAGE_B_FEATURES) - set(C.STAGE_B_FEATURES_NO_REGIME)
    assert dropped == set(C.STAGE_B_REGIME)
    assert len(C.STAGE_B_FEATURES_NO_REGIME) < len(C.STAGE_B_FEATURES)


def test_monotone_features_exist():
    """A monotonicity constraint on a non-existent column silently does nothing."""
    for col in C.MONOTONE_INCREASING:
        assert col in C.STAGE_B_FEATURES, f"{col} is constrained but not a feature"


def _minimal_frame(n: int = 4) -> pd.DataFrame:
    schema = C.features_schema()
    df = pd.DataFrame(index=range(n))
    for f in schema:
        if f.name == "district_id":
            df[f.name] = np.arange(1, n + 1, dtype=np.int32)
        elif f.name == "valid_date":
            df[f.name] = pd.date_range("2024-07-01", periods=n).date
        elif f.name == "lead_day":
            df[f.name] = np.int8(3)
        elif f.name == "init_datetime_utc":
            df[f.name] = pd.Timestamp("2024-06-28", tz="UTC")
        elif f.name == "season":
            df[f.name] = np.int16(2024)
        elif f.name in ("zone_code", "subdivision", "regime_argmax", "label_regime"):
            df[f.name] = "W"
        elif f.name == "label_regime_soft":
            df[f.name] = [np.zeros(C.N_REGIMES, dtype=np.float32)] * n
        elif f.name in C.REGIME_PROB_COLS:
            df[f.name] = np.float32(1.0 / C.N_REGIMES)
        else:
            df[f.name] = np.float32(0.0)
    return df


def test_validate_accepts_a_conforming_frame():
    C.validate(_minimal_frame())


def test_validate_rejects_missing_column():
    df = _minimal_frame().drop(columns=["mm_mean"])
    with pytest.raises(C.ContractError, match="missing"):
        C.validate(df)


def test_validate_rejects_extra_column():
    df = _minimal_frame()
    df["a_column_nobody_agreed_to"] = 1.0
    with pytest.raises(C.ContractError, match="not in contract"):
        C.validate(df)


def test_validate_rejects_probabilities_that_do_not_sum_to_one():
    df = _minimal_frame()
    df["p_ACTIVE"] = np.float32(0.9)  # now sums to >1
    with pytest.raises(C.ContractError, match="sum to 1"):
        C.validate(df)


def test_validate_rejects_out_of_range_lead():
    df = _minimal_frame()
    df["lead_day"] = np.int8(7)
    with pytest.raises(C.ContractError, match="lead_day"):
        C.validate(df)


def test_assert_no_labels_is_the_inference_guard():
    """Spec 13.2: label columns present at inference is a leakage bug."""
    df = _minimal_frame()
    with pytest.raises(C.ContractError, match="leakage"):
        C.assert_no_labels(df)
    C.assert_no_labels(df.drop(columns=C.LABEL_COLS))
