"""Combine the detectors into soft regime labels (spec 4.4, 6.1).

Spec 6.1: "Labels are soft where they overlap: a day inside an LPS envelope
during an active spell gets {LPS: 0.6, ACTIVE: 0.4}." That example is
implemented literally below.

WHICH CLASSES ARE ACTUALLY LABELLABLE. Spec 2.2 wants six; spec 2.1's Level 1
wants four -- ACTIVE, BREAK, LPS, OTHER. Four is what the available data
supports, so Level 1 is met exactly:

  ACTIVE / BREAK   from the published core-monsoon-zone rainfall index
                   (spec 4.5). Fully supported -- the definition is
                   rainfall-based and we hold the IMD grid.
  LPS              from closed MSLP lows with a rainfall test
                   (labels/lps_detector). Calibrated on JJAS 2024 to 29.5% of
                   days with centres at 22.6N 81.9E, the published Bay-head
                   corridor.
  WEAK_TRANSITION  the residual.

  WD               NOT labellable. Needs 500 hPa geopotential; Open-Meteo
                   serves no upper air (docs/DATA_NOTES.md item 16).
  EASTERLY_COASTAL NOT labellable. Needs low-level wind on the ANALYSIS mesh,
                   which the lean config does not carry.

Both absent classes stay in the contract vector at probability zero. They are
reported with their support count of zero rather than quietly dropped, because
a six-wide vector with two dead slots is honest and a silently four-wide one
is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from raahat import contract as C

#: Spec 6.1's own worked example: an LPS embedded in an active spell.
LPS_WEIGHT_IN_ACTIVE = 0.6

#: Below this LPS strength the day is not counted as an LPS at all.
LPS_MIN_STRENGTH = 0.0


def assemble(
    ab_index: pd.DataFrame,
    lps: pd.DataFrame,
    *,
    lps_weight: float = LPS_WEIGHT_IN_ACTIVE,
) -> pd.DataFrame:
    """Merge the active/break index and LPS detections into soft labels.

    `ab_index` needs [valid_date, ab_label]; `lps` needs [valid_date, n_lows,
    lps_strength]. Returns one row per valid_date with `label_regime` (the
    argmax, for grouping and display only) and `label_regime_soft`.
    """
    df = ab_index[["valid_date", "ab_label"]].merge(
        lps[["valid_date", "n_lows", "lps_strength", "lps_deficit_hpa"]],
        on="valid_date", how="left",
    )
    df["n_lows"] = df["n_lows"].fillna(0)
    df["lps_strength"] = df["lps_strength"].fillna(0.0)

    i_act = C.REGIMES.index("ACTIVE")
    i_brk = C.REGIMES.index("BREAK")
    i_lps = C.REGIMES.index("LPS")
    i_weak = C.REGIMES.index("WEAK_TRANSITION")

    soft = np.zeros((len(df), C.N_REGIMES), dtype=np.float32)
    for k, row in enumerate(df.itertuples(index=False)):
        has_lps = row.n_lows > 0 and row.lps_strength >= LPS_MIN_STRENGTH
        ab = row.ab_label

        if has_lps and ab == "ACTIVE":
            # spec 6.1's example. Real days are blends; this is the case the
            # whole soft-gating argument of spec 1.5 exists for.
            soft[k, i_lps] = lps_weight
            soft[k, i_act] = 1.0 - lps_weight
        elif has_lps and ab == "BREAK":
            # genuinely unusual -- a closed low while the core zone is dry.
            # Split rather than forcing a choice; the classifier can learn it.
            soft[k, i_lps] = 0.5
            soft[k, i_brk] = 0.5
        elif has_lps:
            soft[k, i_lps] = 1.0
        elif ab == "ACTIVE":
            soft[k, i_act] = 1.0
        elif ab == "BREAK":
            soft[k, i_brk] = 1.0
        else:
            soft[k, i_weak] = 1.0

    out = df[["valid_date"]].copy()
    out["label_regime_soft"] = list(soft)
    out["label_regime"] = [C.REGIMES[i] for i in soft.argmax(axis=1)]
    out["label_is_blend"] = (soft > 0).sum(axis=1) > 1
    return out


def summarise(labels: pd.DataFrame) -> str:
    """Support counts, including the classes that are structurally absent."""
    soft = np.vstack(labels["label_regime_soft"].to_numpy())
    lines = [f"  {'regime':<20}{'argmax':>8}{'soft mass':>11}{'share':>8}"]
    total = len(labels)
    for i, name in enumerate(C.REGIMES):
        n = int((soft.argmax(axis=1) == i).sum())
        mass = float(soft[:, i].sum())
        note = ""
        if mass == 0:
            note = "   not labellable from available data"
        lines.append(f"  {name:<20}{n:>8}{mass:>11.1f}{n/total:>8.1%}{note}")
    blends = int(labels["label_is_blend"].sum())
    lines.append(f"\n  blended days: {blends}/{total} ({blends/total:.1%})  "
                 f"-- spec 6.1 wants these soft, not forced to a single class")
    return "\n".join(lines)
