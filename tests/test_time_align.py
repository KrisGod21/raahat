"""Spec 21 item 3 -- the 0830 IST rainfall day, unit-tested on day one.

Spec 4.2 calls a mis-aligned rainfall day a silent 15-20% skill killer. The
tests below pin the window boundaries exactly, because the failure mode is not
a crash, it is every downstream number being quietly and plausibly wrong.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from raahat.ingest import time_align as TA


def test_boundary_is_0300_utc():
    assert TA.RAINFALL_DAY_START_UTC_HOUR == 3


def test_default_convention_is_the_measured_one():
    """label_by_end beat label_by_start on 8/8 districts, every one decisive.

    This is pinned because it is the single highest-cost thing in the pipeline
    to get wrong, and a plausible-looking edit could silently flip it.
    """
    assert TA.DEFAULT_CONVENTION == "label_by_end"


def test_0300_utc_starts_a_new_rainfall_day_label_by_start():
    """Mechanics of the boundary, convention held explicit."""
    c = "label_by_start"
    assert TA.to_rainfall_day(pd.DatetimeIndex(["2024-07-30 02:59"]), c)[0] == date(2024, 7, 29)
    assert TA.to_rainfall_day(pd.DatetimeIndex(["2024-07-30 03:00"]), c)[0] == date(2024, 7, 30)


def test_0300_utc_starts_a_new_rainfall_day_default():
    """Under the measured convention the same boundary shifts by one day: rain
    at 02:59Z on 30 Jul is credited to 30 Jul, 03:00Z begins 31 Jul."""
    assert TA.to_rainfall_day(pd.DatetimeIndex(["2024-07-30 02:59"]))[0] == date(2024, 7, 30)
    assert TA.to_rainfall_day(pd.DatetimeIndex(["2024-07-30 03:00"]))[0] == date(2024, 7, 31)


def test_whole_window_maps_to_one_day():
    """Every hour from 03Z D-1 to 02Z D is rainfall day D (label_by_end)."""
    hours = pd.date_range("2024-07-29 03:00", "2024-07-30 02:00", freq="h")
    days = TA.to_rainfall_day(hours)
    assert len(hours) == 24
    assert set(days) == {date(2024, 7, 30)}


def test_local_midnight_would_have_been_wrong():
    """The bug this module exists to prevent: 00Z-23Z is NOT a rainfall day.

    Under the IMD convention, 04:00Z-23:00Z on 30 Jul belongs to rainfall day
    31 Jul. A naive calendar-day sum files it under 30 Jul -- a clean one-day
    shift that halves the correlation against observations.
    """
    late = pd.DatetimeIndex(["2024-07-30 04:00", "2024-07-30 12:00", "2024-07-30 23:00"])
    days = TA.to_rainfall_day(late)
    assert set(days) == {date(2024, 7, 31)}
    assert set(late.date) == {date(2024, 7, 30)}
    assert list(days) != list(late.date)


def test_conventions_differ_by_exactly_one_day():
    hours = pd.date_range("2024-07-01 03:00", periods=48, freq="h")
    a = TA.to_rainfall_day(hours, "label_by_start")
    b = TA.to_rainfall_day(hours, "label_by_end")
    deltas = {(y - x).days for x, y in zip(a, b)}
    assert deltas == {1}


def test_unknown_convention_raises():
    with pytest.raises(ValueError, match="unknown convention"):
        TA.to_rainfall_day(pd.DatetimeIndex(["2024-07-30 03:00"]), "label_by_vibes")


def test_tz_aware_input_is_handled():
    aware = pd.DatetimeIndex(["2024-07-30T03:00:00Z"])
    naive = pd.DatetimeIndex(["2024-07-30 03:00"])
    assert list(TA.to_rainfall_day(aware)) == list(TA.to_rainfall_day(naive))


# ------------------------------------------------------------ aggregation ---


def test_aggregate_sums_the_right_24_hours():
    """1 mm in every hour of rainfall day 30 Jul (03Z 29th -> 03Z 30th), 99 outside."""
    hours = pd.date_range("2024-07-28 03:00", "2024-07-30 02:00", freq="h")
    inside = (hours >= pd.Timestamp("2024-07-29 03:00")) & (hours < pd.Timestamp("2024-07-30 03:00"))
    vals = np.where(inside, 1.0, 99.0)
    out = TA.aggregate_hourly(hours, vals).set_index("valid_date")["precip_mm"]
    assert out.loc[date(2024, 7, 30)] == pytest.approx(24.0)
    assert out.loc[date(2024, 7, 29)] == pytest.approx(24 * 99.0)


def test_partial_days_are_null_not_small():
    """An incomplete day that looks dry is worse than an absent one: it teaches
    the model to expect zero."""
    hours = pd.date_range("2024-07-29 03:00", periods=10, freq="h")
    out = TA.aggregate_hourly(hours, np.ones(10)).set_index("valid_date")["precip_mm"]
    assert np.isnan(out.loc[date(2024, 7, 30)])


def test_nan_hours_do_not_silently_shorten_a_day():
    hours = pd.date_range("2024-07-29 03:00", periods=24, freq="h")
    vals = np.ones(24)
    vals[5] = np.nan
    out = TA.aggregate_hourly(hours, vals).set_index("valid_date")["precip_mm"]
    assert np.isnan(out.loc[date(2024, 7, 30)])  # 23 valid hours < min_hours


def test_window_description_is_0830_ist_to_0830_ist():
    """Rainfall day 30 Jul ENDS at 0830 IST on 30 Jul under the measured convention."""
    s, e = TA.ist_window(date(2024, 7, 30))
    assert (s, e) == (pd.Timestamp("2024-07-29 03:00"), pd.Timestamp("2024-07-30 03:00"))
    assert TA.ist_window(date(2024, 7, 30), "label_by_start") == (
        pd.Timestamp("2024-07-30 03:00"), pd.Timestamp("2024-07-31 03:00"))
    assert "08:30 IST" in TA.describe_window(date(2024, 7, 30))
