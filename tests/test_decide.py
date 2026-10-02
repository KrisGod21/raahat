"""Decision-layer tests (spec 6.4).

The safety guard and the hysteresis rule are both here because of how the
system fails in front of people, not how it scores. Tests accordingly check
behaviour a forecaster would notice, not just arithmetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raahat import contract as C
from raahat.decide import colour as CO, cost_loss as CL


# ------------------------------------------------------------ cost-loss ---


def test_alpha_direction_lower_means_more_warnings():
    """Easy to get backwards: LOWER alpha = act on weaker evidence = more
    warnings = more miss-averse."""
    p = np.linspace(0, 1, 101)
    assert CL.act(p, 0.05).sum() > CL.act(p, 0.50).sum()


def test_alpha_is_clamped_to_the_ui_range():
    assert CL.clamp_alpha(0.0) == CL.ALPHA_MIN
    assert CL.clamp_alpha(9.9) == CL.ALPHA_MAX
    assert CL.clamp_alpha(0.15) == pytest.approx(0.15)


def test_default_alpha_is_the_measured_one_not_the_spec_suggestion():
    """Spec 6.4 suggests 0.30; 0.15 was measured better (DATA_NOTES item 19)."""
    assert CL.ALPHA_DEFAULT == pytest.approx(0.15)


def test_expected_cost_is_indifferent_at_the_breakeven_probability():
    """The rule is optimal at P = C/L, so acting and not acting cost the same
    there. If this breaks, the rule is not a cost-loss rule."""
    a = 0.3
    assert CL.expected_cost([a], a)[0] == pytest.approx(a, abs=1e-9)


def test_sweep_trades_pod_against_far_monotonically():
    rng = np.random.default_rng(0)
    obs = rng.random(4000) < 0.05
    p = np.clip(obs * 0.5 + rng.random(4000) * 0.4, 0, 1)
    s = CL.sweep(p, obs)
    assert s["POD"].is_monotonic_decreasing
    assert s["warn_rate"].is_monotonic_decreasing


# -------------------------------------------------------------- colours ---


def test_colour_ladder_matches_imd_thresholds():
    v = [0.0, 70.0, 120.0, 250.0]
    assert list(CO.colour_from_value(v)) == [CO.GREEN, CO.YELLOW, CO.ORANGE, CO.RED]


def test_highest_qualifying_probability_wins():
    c = CO.colour_from_probabilities([0.9], [0.9], [0.9], alpha=0.15)
    assert c[0] == CO.RED
    c = CO.colour_from_probabilities([0.9], [0.01], [0.01], alpha=0.15)
    assert c[0] == CO.YELLOW


def test_never_hard_codes_a_half_probability():
    """Spec 6.4: 'Never a hard-coded 0.5.' A 0.2 probability must warn at
    alpha=0.15 and not at alpha=0.5."""
    assert CO.colour_from_probabilities([0.2], [0], [0], alpha=0.15)[0] == CO.YELLOW
    assert CO.colour_from_probabilities([0.2], [0], [0], alpha=0.50)[0] == CO.GREEN


# --------------------------------------------------------- safety guard ---


def test_guard_blocks_a_silent_downgrade():
    """Raw says 120 mm (ORANGE) but our probabilities say GREEN, and we are
    not confident it is dry. The guard must hold the warning up."""
    guarded, applied = CO.apply_safety_guard([CO.GREEN], [120.0], [0.10], alpha=0.15)
    assert guarded[0] == CO.ORANGE
    assert applied[0]


def test_guard_releases_when_we_are_genuinely_confident_it_is_dry():
    """Otherwise the system could never cancel a false alarm, which is half
    its value (spec 10.1 event 4)."""
    guarded, applied = CO.apply_safety_guard([CO.GREEN], [120.0], [0.001], alpha=0.15)
    assert guarded[0] == CO.GREEN
    assert not applied[0]


def test_guard_never_lowers_an_upgrade():
    """It is a floor, not a clamp: we may warn ABOVE the raw model freely."""
    guarded, applied = CO.apply_safety_guard([CO.RED], [10.0], [0.9], alpha=0.15)
    assert guarded[0] == CO.RED
    assert not applied[0]


# ---------------------------------------------------------- hysteresis ---


def test_hysteresis_removes_an_isolated_flicker():
    s = pd.Series([CO.ORANGE, CO.YELLOW, CO.ORANGE], dtype=np.int8)
    g = pd.Series(["d1"] * 3)
    assert list(CO.apply_hysteresis(s, g)) == [CO.ORANGE] * 3


def test_hysteresis_lets_a_sustained_escalation_through():
    """Damping a real escalation would be dangerous, not merely annoying."""
    s = pd.Series([CO.GREEN, CO.ORANGE, CO.ORANGE, CO.ORANGE], dtype=np.int8)
    g = pd.Series(["d1"] * 4)
    assert list(CO.apply_hysteresis(s, g)) == [CO.GREEN, CO.ORANGE, CO.ORANGE, CO.ORANGE]


def test_hysteresis_does_not_leak_across_districts():
    s = pd.Series([CO.RED, CO.GREEN, CO.RED], dtype=np.int8)
    g = pd.Series(["a", "b", "a"])
    assert list(CO.apply_hysteresis(s, g)) == [CO.RED, CO.GREEN, CO.RED]


# -------------------------------------------------------------- end to end ---


def test_assign_produces_contract_shaped_output():
    df = pd.DataFrame({
        "district_id": [1, 1, 2], "valid_date": pd.date_range("2025-08-01", periods=3).date,
        "lead_day": np.int8(3), "mm_mean": [5.0, 130.0, 0.0],
        "p_gt_64_5": [0.02, 0.60, 0.01], "p_gt_115_6": [0.0, 0.40, 0.0],
        "p_gt_204_5": [0.0, 0.05, 0.0],
    })
    out = CO.assign(df, alpha=0.15)
    assert set(out.columns) >= {"colour_code", "colour_name", "colour_alpha_used",
                                "guard_applied"}
    assert out["colour_code"].between(0, 3).all()
    assert out.loc[1, "colour_code"] == CO.ORANGE     # p_115 = 0.40 > 0.15
    assert out.loc[2, "colour_code"] == CO.GREEN


def test_missing_extreme_classifier_cannot_invent_a_red():
    """The 204.5 mm classifier is often unfittable for lack of events. Its
    absence must fail safe, not loudly."""
    df = pd.DataFrame({"mm_mean": [10.0], "p_gt_64_5": [0.9], "p_gt_115_6": [0.9]})
    out = CO.assign(df, alpha=0.15, hysteresis=False)
    assert out["colour_code"].iloc[0] == CO.ORANGE
