"""The IMD rainfall day. Spec 4.2 and spec 21, item 3.

IMD's rainfall day runs 0830 IST to 0830 IST, i.e. **03:00 UTC to 03:00 UTC**.
Open-Meteo's `daily=` sums use LOCAL MIDNIGHT. Mixing the two shifts a whole
convective peak into the wrong day and silently costs 15-20% of apparent skill,
which is the sort of bug that never announces itself -- the pipeline runs, the
numbers look plausible, and every score is quietly wrong.

So: always request `hourly=precipitation` with `timezone=GMT` and aggregate the
03Z window here. Never use the provider's daily aggregation.

Which calendar date a 03Z-03Z window is LABELLED with is a separate question
from where its boundaries fall, and the two conventions differ by exactly one
day. Getting it wrong is a pure one-day shift, which looks like "our model is
bad" rather than "our dates are off by one". `determine_convention()` settles
it against the IMD grid itself rather than against anyone's prose.

MEASURED, 2026-09-21, JJAS 2024, ERA5 hourly vs IMD 0.25 deg, 8 districts
across all three zones -- `label_by_end` won every single one, decisively:

    district      zone   by_start   by_end   margin
    Bilaspur       C       0.183     0.601    0.418
    Dehradun       N       0.165     0.599    0.434
    Wayanad        W       0.559     0.776    0.217
    Nagpur         C       0.563     0.786    0.223
    ...            ..        ...       ...      ...

So the IMD rainfall day D is the window ending at 0830 IST on day D: the
gauge read on the morning of D is credited to D. Spec 4.2 estimates the cost
of getting this wrong at 15-20% of apparent skill; measured here it roughly
HALVES the correlation. See docs/DATA_NOTES.md.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd

#: 08:30 IST == 03:00 UTC. The boundary of the IMD rainfall day.
RAINFALL_DAY_START_UTC_HOUR = 3
IST_OFFSET = timedelta(hours=5, minutes=30)

Convention = Literal["label_by_start", "label_by_end"]

#: Determined empirically against the IMD 0.25 deg grid -- 8/8 districts, every
#: one decisive. Do NOT change without re-running determine_convention().
#: See the module docstring, tests/test_time_align.py and docs/DATA_NOTES.md.
DEFAULT_CONVENTION: Convention = "label_by_end"


def to_rainfall_day(
    times_utc: pd.DatetimeIndex | pd.Series,
    convention: Convention = DEFAULT_CONVENTION,
) -> np.ndarray:
    """Map UTC timestamps to the IMD rainfall day each one falls in.

    label_by_start: the window [03Z on D, 03Z on D+1) is called D.
    label_by_end:   the window [03Z on D-1, 03Z on D) is called D.

    Returns an array of datetime.date.
    """
    ts = pd.DatetimeIndex(pd.to_datetime(times_utc))
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    shifted = ts - np.timedelta64(RAINFALL_DAY_START_UTC_HOUR, "h")
    if convention == "label_by_end":
        shifted = shifted + np.timedelta64(1, "D")
    elif convention != "label_by_start":
        raise ValueError(f"unknown convention {convention!r}")
    return shifted.normalize().date


def aggregate_hourly(
    times_utc,
    values,
    *,
    convention: Convention = DEFAULT_CONVENTION,
    min_hours: int = 24,
) -> pd.DataFrame:
    """Sum hourly precipitation into IMD rainfall days.

    Partial days (fewer than `min_hours` samples) are returned with a null sum
    rather than a misleadingly small total -- an incomplete day that looks dry
    is worse than an absent one, because it trains the model to expect zero.

    Returns columns [valid_date, precip_mm, n_hours].
    """
    df = pd.DataFrame({"t": pd.to_datetime(times_utc), "v": np.asarray(values, dtype="float64")})
    df["valid_date"] = to_rainfall_day(df["t"], convention)
    g = df.groupby("valid_date", sort=True).agg(
        precip_mm=("v", "sum"), n_hours=("v", "count"), n_valid=("v", lambda s: s.notna().sum())
    )
    g.loc[g["n_valid"] < min_hours, "precip_mm"] = np.nan
    return g.reset_index()[["valid_date", "precip_mm", "n_hours"]]


def ist_window(
    d: date, convention: Convention = DEFAULT_CONVENTION
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The UTC half-open interval covered by IMD rainfall day `d`.

    Under the measured `label_by_end` convention this ENDS at 03Z on `d`, so
    most of the rain credited to day d actually fell on the calendar day
    before. That is not a bug; it is what the IMD gauge network records.
    """
    start = pd.Timestamp(d) + np.timedelta64(RAINFALL_DAY_START_UTC_HOUR, "h")
    if convention == "label_by_end":
        start = start - np.timedelta64(1, "D")
    elif convention != "label_by_start":
        raise ValueError(f"unknown convention {convention!r}")
    return start, start + np.timedelta64(1, "D")


def describe_window(d: date, convention: Convention = DEFAULT_CONVENTION) -> str:
    s, e = ist_window(d, convention)
    return (
        f"rainfall day {d}: {s:%Y-%m-%d %H:%M}Z -> {e:%Y-%m-%d %H:%M}Z "
        f"({(s + IST_OFFSET):%H:%M} IST {(s + IST_OFFSET):%d %b} -> "
        f"{(e + IST_OFFSET):%H:%M} IST {(e + IST_OFFSET):%d %b})"
    )


def determine_convention(
    hourly_times_utc,
    hourly_values,
    imd_daily: pd.Series,
    *,
    min_overlap: int = 30,
) -> dict:
    """Decide the labelling convention from the data, not from prose.

    Aggregates the same hourly series under both conventions and correlates
    each against the IMD gridded daily series. A one-day shift is unmistakable:
    the right convention correlates far better. Returns the verdict plus both
    correlations so the margin is visible rather than asserted.
    """
    out: dict[str, float] = {}
    for conv in ("label_by_start", "label_by_end"):
        agg = aggregate_hourly(hourly_times_utc, hourly_values, convention=conv)
        agg = agg.set_index("valid_date")["precip_mm"]
        joined = pd.concat([agg.rename("fc"), imd_daily.rename("obs")], axis=1, join="inner")
        joined = joined.dropna()
        out[conv] = float(joined["fc"].corr(joined["obs"])) if len(joined) >= min_overlap else np.nan
        out[f"{conv}_n"] = len(joined)

    a, b = out["label_by_start"], out["label_by_end"]
    if np.isnan(a) or np.isnan(b):
        winner, margin = None, np.nan
    else:
        winner = "label_by_start" if a >= b else "label_by_end"
        margin = abs(a - b)
    return {
        "correlations": {"label_by_start": a, "label_by_end": b},
        "n_days": {"label_by_start": out["label_by_start_n"], "label_by_end": out["label_by_end_n"]},
        "winner": winner,
        "margin": margin,
        "decisive": bool(not np.isnan(margin) and margin > 0.10),
    }
