"""Probabilities -> IMD warning colours (spec 6.4).

GREEN no warning / YELLOW be updated / ORANGE be prepared / RED take action.
These four are policy, not design choices, and they are the only saturated
colours permitted anywhere in the interface (spec 9.2).

Two rules sit on top of the plain threshold logic, and both exist because of
how this fails in the room rather than how it scores:

  SAFETY GUARD. The assigned colour may never fall BELOW the colour the raw
  multi-model mean would have produced, unless the exceedance probability is
  very low (alpha * safety_guard_release). A post-processor that quietly
  downgrades a real warning is operationally unacceptable, and one live example
  of that on stage ends the presentation. Spec 6.4 is explicit.

  HYSTERESIS. Colours must not flicker between consecutive runs. A district
  oscillating orange-yellow-orange every six hours destroys trust faster than
  being wrong once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from raahat import contract as C
from raahat.decide.cost_loss import ALPHA_DEFAULT, clamp_alpha

_CFG = C.load_config("model")["decision"]

GREEN, YELLOW, ORANGE, RED = 0, 1, 2, 3
NAMES = {GREEN: "GREEN", YELLOW: "YELLOW", ORANGE: "ORANGE", RED: "RED"}

SAFETY_GUARD: bool = bool(_CFG["safety_guard"])
GUARD_RELEASE: float = float(_CFG["safety_guard_release"])
HYSTERESIS_STEPS: int = int(_CFG["hysteresis_steps"])

#: Which threshold promotes to which colour (spec 6.4).
_LADDER = [
    (C.THRESHOLDS["extremely_heavy"], RED),
    (C.THRESHOLDS["very_heavy"], ORANGE),
    (C.THRESHOLDS["heavy"], YELLOW),
]


def colour_from_value(mm) -> np.ndarray:
    """The colour a deterministic rainfall value implies. Used for the guard."""
    v = np.asarray(mm, dtype="float64")
    out = np.full(v.shape, GREEN, dtype=np.int8)
    for thresh, code in reversed(_LADDER):
        out = np.where(v > thresh, code, out)
    return out


def colour_from_probabilities(
    p_heavy, p_very_heavy, p_extremely_heavy, *, alpha: float = ALPHA_DEFAULT
) -> np.ndarray:
    """Highest colour whose exceedance probability clears the cost-loss bar."""
    a = clamp_alpha(alpha)
    ph = np.asarray(p_heavy, dtype="float64")
    pv = np.asarray(p_very_heavy, dtype="float64")
    pe = np.asarray(p_extremely_heavy, dtype="float64")
    out = np.full(ph.shape, GREEN, dtype=np.int8)
    out = np.where(np.nan_to_num(ph) > a, YELLOW, out)
    out = np.where(np.nan_to_num(pv) > a, ORANGE, out)
    out = np.where(np.nan_to_num(pe) > a, RED, out)
    return out


def apply_safety_guard(
    proposed, raw_mm, p_heavy, *, alpha: float = ALPHA_DEFAULT
) -> tuple[np.ndarray, np.ndarray]:
    """Never silently downgrade the raw model's warning (spec 6.4).

    Returns (guarded_colour, guard_applied). The escape hatch is a genuinely
    low exceedance probability -- otherwise a post-processor could never
    correct a false alarm, which is half its job (spec 10.1 event 4).
    """
    proposed = np.asarray(proposed, dtype=np.int8)
    raw_colour = colour_from_value(raw_mm)
    ph = np.nan_to_num(np.asarray(p_heavy, dtype="float64"))
    release = clamp_alpha(alpha) * GUARD_RELEASE

    would_downgrade = proposed < raw_colour
    confident_downgrade = ph < release
    apply = would_downgrade & ~confident_downgrade
    return np.where(apply, raw_colour, proposed).astype(np.int8), apply


def apply_hysteresis(
    colours: pd.Series, groups: pd.Series, *, steps: int = HYSTERESIS_STEPS
) -> np.ndarray:
    """Damp single-run flickers within each district's time series.

    A colour that differs from BOTH its neighbours by the same amount is an
    isolated spike; it is pulled back to the surrounding level. A sustained
    change -- two or more consecutive runs -- passes through untouched, which
    is the behaviour you want: real escalation must not be damped.
    """
    if steps <= 0:
        return colours.to_numpy(dtype=np.int8)
    out = colours.to_numpy(dtype=np.int8).copy()
    df = pd.DataFrame({"g": groups.to_numpy(), "c": out})
    for _, idx in df.groupby("g").groups.items():
        pos = df.index.get_indexer(idx)
        seq = out[pos]
        if len(seq) < 3:
            continue
        fixed = seq.copy()
        for i in range(1, len(seq) - 1):
            if seq[i - 1] == seq[i + 1] and seq[i] != seq[i - 1]:
                fixed[i] = seq[i - 1]      # isolated one-run spike or dip
        out[pos] = fixed
    return out


def assign(
    df: pd.DataFrame,
    *,
    alpha: float = ALPHA_DEFAULT,
    raw_col: str = "mm_mean",
    guard: bool = SAFETY_GUARD,
    hysteresis: bool = True,
) -> pd.DataFrame:
    """Full decision layer. Returns colour_code, alpha_used, guard_applied.

    Expects p_gt_64_5 / p_gt_115_6 / p_gt_204_5 and the raw value column.
    Missing exceedance columns are treated as zero probability, which is the
    safe direction: an absent 204.5 mm classifier cannot invent a RED.
    """
    def col(name):
        return df[name] if name in df.columns else pd.Series(0.0, index=df.index)

    proposed = colour_from_probabilities(
        col("p_gt_64_5"), col("p_gt_115_6"), col("p_gt_204_5"), alpha=alpha
    )
    applied = np.zeros(len(df), dtype=bool)
    if guard and raw_col in df.columns:
        proposed, applied = apply_safety_guard(
            proposed, df[raw_col], col("p_gt_64_5"), alpha=alpha
        )

    out = pd.DataFrame(index=df.index)
    out["colour_code"] = proposed
    if hysteresis and {"district_id", "valid_date"} <= set(df.columns):
        order = df.sort_values(["district_id", "lead_day", "valid_date"]).index
        s = pd.Series(proposed, index=df.index).loc[order]
        g = df.loc[order].apply(lambda r: (r["district_id"], r["lead_day"]), axis=1)
        damped = apply_hysteresis(s, g)
        out.loc[order, "colour_code"] = damped
    out["colour_name"] = [NAMES[int(c)] for c in out["colour_code"]]
    out["colour_alpha_used"] = clamp_alpha(alpha)
    out["guard_applied"] = applied
    return out


def summarise(colours: pd.DataFrame) -> str:
    lines = [f"  {'colour':<8}{'n':>9}{'share':>9}"]
    n = len(colours)
    for code in (RED, ORANGE, YELLOW, GREEN):
        k = int((colours["colour_code"] == code).sum())
        lines.append(f"  {NAMES[code]:<8}{k:>9,}{k / n:>9.2%}")
    if "guard_applied" in colours:
        g = int(colours["guard_applied"].sum())
        lines.append(f"\n  safety guard raised {g:,} of {n:,} ({g / n:.2%}) "
                     f"-- downgrades the raw model would not have made")
    return "\n".join(lines)
