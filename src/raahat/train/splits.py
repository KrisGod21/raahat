"""Blocked splits, with the test block under lock and key (spec 7.1).

Spec 7.1 assumes three whole seasons. IMD had published only two as of
2026-09-21, so the split is defined as contiguous DATE BLOCKS rather than whole
seasons -- see the rationale in config/model.yaml and docs/DATA_NOTES.md.

What does NOT change: splits are contiguous in time and never random. Spec 7.2
trap 1 -- consecutive days inside one LPS are near-duplicates, so a random
split inflates skill dramatically. Trap 2 -- neighbouring districts on the same
day are not independent either, so districts are never split across.

The test block is opened exactly once, by scripts/final_eval.py.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from raahat import contract as C

_cfg = C.load_config("model")["splits"]
LEAD_DAYS: list[int] = list(_cfg["lead_days"])

SPLIT_NAMES = ("train", "validation", "test")


def _blocks(name: str) -> list[dict]:
    out = []
    for b in _cfg[name]:
        out.append(
            {
                "season": int(b["season"]),
                "start": pd.Timestamp(b["start"]).date(),
                "end": pd.Timestamp(b["end"]).date(),
            }
        )
    return out


BLOCKS: dict[str, list[dict]] = {n: _blocks(n) for n in SPLIT_NAMES}

#: Seasons each split touches. A season can appear in more than one.
SEASONS: dict[str, list[int]] = {
    n: sorted({b["season"] for b in BLOCKS[n]}) for n in SPLIT_NAMES
}
TRAIN_SEASONS = SEASONS["train"]
VALIDATION_SEASONS = SEASONS["validation"]
TEST_SEASONS = SEASONS["test"]

#: Flipped only by unlock_test_block(). Never set this from library code.
_TEST_UNLOCKED = False


class TestSeasonLocked(RuntimeError):
    """Raised when anything but final_eval.py reaches for the test block."""


def split_of(d: date) -> str | None:
    """Which split a valid_date belongs to, or None if it is in no block."""
    d = pd.Timestamp(d).date() if not isinstance(d, date) else d
    for name in SPLIT_NAMES:
        for b in BLOCKS[name]:
            if b["start"] <= d <= b["end"]:
                return name
    return None


def get_split(season: int) -> str:
    """Splits a season touches, as a string. Kept for readability in logs.

    With a season cut across two splits this returns e.g. 'validation+test',
    which is deliberately awkward -- a caller that wants one answer per season
    is asking the wrong question now and should use split_of(date).
    """
    hit = [n for n in SPLIT_NAMES if int(season) in SEASONS[n]]
    if not hit:
        raise KeyError(f"season {season} is in no split; known: {sorted(set().union(*SEASONS.values()))}")
    return "+".join(hit)


def unlock_test_block(reason: str) -> None:
    """Open the test block. scripts/final_eval.py only. Prints a loud banner."""
    global _TEST_UNLOCKED
    caller = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "<interactive>"
    allowed = caller in ("final_eval.py", "pytest", "py.test", "_jb_pytest_runner.py")
    if not allowed and os.environ.get("RAAHAT_ALLOW_TEST") != "1":
        raise TestSeasonLocked(
            f"test block unlock attempted from '{caller}'. Only scripts/final_eval.py "
            "may open it. Set RAAHAT_ALLOW_TEST=1 to override, and write down why in "
            "docs/DATA_NOTES.md."
        )
    bar = "!" * 78
    blocks = ", ".join(f"{b['start']}..{b['end']}" for b in BLOCKS["test"])
    print(f"\n{bar}\n!!  TEST BLOCK OPENED: {blocks}\n!!  reason: {reason}\n"
          f"!!  caller: {caller}\n{bar}\n", flush=True)
    _TEST_UNLOCKED = True


# Back-compat alias; the old name read as though whole seasons were the unit.
unlock_test_season = unlock_test_block


def assert_allowed(split: str) -> None:
    """Guard. Raises unless the test block has been explicitly unlocked."""
    if split == "test" and not _TEST_UNLOCKED:
        raise TestSeasonLocked(
            "the test block is held out and locked. If this is scripts/final_eval.py, "
            "call unlock_test_block() first."
        )


def filter_to_split(df: pd.DataFrame, split: str, *, date_col: str = "valid_date") -> pd.DataFrame:
    """Rows falling inside a named split's date blocks."""
    if split not in SPLIT_NAMES:
        raise KeyError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")
    assert_allowed(split)
    d = pd.to_datetime(df[date_col]).dt.date
    mask = pd.Series(False, index=df.index)
    for b in BLOCKS[split]:
        mask |= (d >= b["start"]) & (d <= b["end"])
    return df.loc[mask].reset_index(drop=True)


# ------------------------------------------------------------------ io ---


def features_path(season: int, root: Path | None = None) -> Path:
    """Where features_{season}.parquet lives. Falls back to synthetic."""
    root = root or C.DATA_DIR
    processed = root / "processed" / f"features_{season}.parquet"
    if processed.exists():
        return processed
    synthetic = root / "synthetic" / f"features_{season}.parquet"
    if synthetic.exists():
        return synthetic
    raise FileNotFoundError(
        f"no features for season {season}: looked in {processed} and {synthetic}."
    )


def load_split(split: str, *, columns: list[str] | None = None) -> pd.DataFrame:
    """Load exactly the rows of one split, test guarded."""
    assert_allowed(split)
    parts = [pd.read_parquet(features_path(s), columns=columns) for s in SEASONS[split]]
    return filter_to_split(pd.concat(parts, ignore_index=True), split)


def load_regime_split(split: str) -> pd.DataFrame:
    """Load the national Stage-A table for one split, test guarded."""
    assert_allowed(split)
    for base in ("processed", "synthetic"):
        path = C.DATA_DIR / base / "regime_inputs.parquet"
        if path.exists():
            return filter_to_split(pd.read_parquet(path), split)
    raise FileNotFoundError("no regime_inputs.parquet found.")


def describe() -> str:
    lines = []
    for n in SPLIT_NAMES:
        blocks = ", ".join(f"{b['start']}..{b['end']}" for b in BLOCKS[n])
        days = sum((b["end"] - b["start"]).days + 1 for b in BLOCKS[n])
        lock = " [LOCKED]" if n == "test" and not _TEST_UNLOCKED else ""
        lines.append(f"  {n:<11} {blocks}  ({days} days){lock}")
    return "splits (spec 7.1, adapted -- see config/model.yaml):\n" + "\n".join(lines)
