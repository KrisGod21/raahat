"""Cost-loss decision rule (spec 6.4).

Act when P(event) > C/L, where C is the cost of preparing and L the loss if
caught unprepared. That ratio, alpha, is the ONLY knob, and it belongs to the
user rather than to us -- a district collector's miss-aversion is not a
modelling parameter.

Direction matters and is easy to get backwards: a LOWER alpha means acting on
weaker evidence, so lower = more warnings = more miss-averse.

Spec 6.4 suggests 0.30. We measured 0.15 (docs/DATA_NOTES.md item 19): at 0.30
the system scored WORSE than raw ECMWF on heavy-rain CSI, because 0.30 is far
too high a bar for a ~1.5% base rate. The config default is 0.15; the slider
range stays 0.05-0.60 as the spec specifies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from raahat import contract as C

_CFG = C.load_config("model")["decision"]

ALPHA_DEFAULT: float = float(_CFG["alpha_default"])
ALPHA_MIN, ALPHA_MAX = (float(x) for x in _CFG["alpha_ui_range"])


def clamp_alpha(alpha: float) -> float:
    """Keep alpha inside the range the UI exposes."""
    return float(np.clip(alpha, ALPHA_MIN, ALPHA_MAX))


def act(prob, alpha: float = ALPHA_DEFAULT):
    """The rule itself: act when P(event) exceeds the cost-loss ratio."""
    return np.asarray(prob, dtype="float64") > clamp_alpha(alpha)


def expected_cost(prob, alpha: float, *, loss: float = 1.0) -> np.ndarray:
    """Expected cost per district-day under the rule, in units of L.

    Acting costs C = alpha*L whether or not the event occurs. Not acting costs
    L when it does. This is what makes the rule optimal at P = C/L, and having
    it in code lets the UI show WHY a threshold is where it is rather than
    asserting it.
    """
    p = np.asarray(prob, dtype="float64")
    a = clamp_alpha(alpha)
    return np.where(act(p, a), a * loss, p * loss)


def sweep(prob, obs_event, alphas=None) -> pd.DataFrame:
    """Score the rule across alphas. Feeds the UI slider and the §19 slide.

    `obs_event` is a boolean array of whether the event actually occurred.
    """
    alphas = alphas if alphas is not None else np.round(np.arange(0.05, 0.61, 0.05), 2)
    p = np.asarray(prob, dtype="float64")
    o = np.asarray(obs_event, dtype=bool)
    ok = np.isfinite(p)
    p, o = p[ok], o[ok]

    rows = []
    for a in alphas:
        f = act(p, a)
        hits = int((o & f).sum()); fa = int((~o & f).sum()); miss = int((o & ~f).sum())
        rows.append({
            "alpha": float(a),
            "warned": int(f.sum()),
            "warn_rate": float(f.mean()),
            "hits": hits, "false_alarms": fa, "misses": miss,
            "POD": hits / (hits + miss) if (hits + miss) else np.nan,
            "FAR": fa / (hits + fa) if (hits + fa) else np.nan,
            "CSI": hits / (hits + fa + miss) if (hits + fa + miss) else np.nan,
            "expected_cost": float(expected_cost(p, a).mean()),
        })
    return pd.DataFrame(rows)
