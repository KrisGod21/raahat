"""Probability calibration (spec 6.1, 7.3).

Spec 7.1 fits the calibrator on the validation block. With three whole seasons
that is fine. With one season cut in two it is not, and we measured the cost:
calibrating on Jun-Jul 2025 and applying to Aug-Sep 2025 left the model
UNDER-CONFIDENT by roughly a factor of two in the band where it signals real
risk -- it said 29%, reality was 63% (docs/DATA_NOTES.md item 34).

TWO WRONG EXPLANATIONS, RECORDED SO THEY ARE NOT RE-ADOPTED:

  "June-July is the quiet half, so the mapping understates the stormy half."
  FALSE. June-July is the WETTER half at district level in both seasons
  (2024: 2.30% vs 1.30% base rate; 2025: 1.56% vs 1.40%). Calibrating on a
  wetter period and applying to a drier one predicts OVER-confidence. We
  observed the opposite, so this cannot be the mechanism.

  "Fit the calibrator on out-of-fold predictions across the whole training
  season." TRIED AND WORSE THAN DOING NOTHING -- nested ECE 0.0210 against
  0.0170 uncalibrated. Each fold's model trains on less data and emits
  differently-scaled probabilities, so the mapping does not transfer to the
  full model.

WHAT IS ACTUALLY WRONG. Only 141 validation days sit above raw probability
0.10. Isotonic regression is a step function, and on data that thin it
flattened everything from 0.25 upward onto one value, 0.278 -- a day the model
rated 0.9 came out identical to one it rated 0.3.

There is a second, deeper effect the fix does NOT address: reliability is
regime-dependent. In validation, days the raw model rated 0.8-1.0 verified at
33%; in the test block the same-confidence days verified at 63%. August-
September is the LPS half, the regime signal is stronger, and confident
forecasts genuinely deserve more trust there. A single global mapping cannot
represent that. Regime-conditional calibration is the principled answer and
needs more data than two seasons of 197 districts provide.

THE FIX IMPLEMENTED HERE: pool the calibration data across all five leads (5x
the points) and use a smooth two-parameter mapping that cannot collapse a
range. Measured held-out by lead: mean ECE 0.0170 -> 0.0037. See
fit_pooled_calibrators and PlattCalibrator below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from raahat import contract as C


def time_blocked_folds(dates: pd.Series, k: int = 4) -> list[np.ndarray]:
    """k contiguous blocks in date order. Returns boolean masks."""
    d = pd.to_datetime(pd.Series(dates).to_numpy())
    order = np.argsort(d.to_numpy())
    uniq = np.array(sorted(pd.unique(d)))
    edges = np.array_split(uniq, k)
    masks = []
    for block in edges:
        if len(block) == 0:
            continue
        masks.append(np.isin(d.to_numpy(), block))
    del order
    return masks


def out_of_fold_probabilities(
    train: pd.DataFrame,
    features: list[str],
    threshold: float,
    *,
    k: int = 4,
    params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold P(y > threshold) across the whole training period.

    Returns (probabilities, observed 0/1), both aligned to `train`'s rows.
    Rows in folds where the positive class was too rare to fit are returned as
    NaN rather than zero -- a fold with no heavy rain cannot inform the
    mapping, and filling it with zeros would drag the calibration down.
    """
    import lightgbm as lgb

    cfg = params or C.load_config("model")["corrector_lgbm"]
    y = (train[C.TARGET_COL].to_numpy() > threshold).astype(int)
    oof = np.full(len(train), np.nan)

    for mask in time_blocked_folds(train["valid_date"], k):
        fit_idx = ~mask
        if y[fit_idx].sum() < 10 or mask.sum() == 0:
            continue
        pos_w = float((y[fit_idx] == 0).sum() / max(y[fit_idx].sum(), 1))
        m = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=cfg.get("n_estimators", 800),
            learning_rate=cfg.get("learning_rate", 0.05),
            num_leaves=cfg.get("num_leaves", 63),
            min_child_samples=cfg.get("min_child_samples", 50),
            subsample=cfg.get("subsample", 0.8),
            subsample_freq=1,
            colsample_bytree=cfg.get("colsample_bytree", 0.8),
            scale_pos_weight=pos_w,
            random_state=cfg.get("random_state", 42),
            n_jobs=-1,
            verbose=-1,
        )
        m.fit(train.loc[fit_idx, features], y[fit_idx])
        oof[mask] = m.predict_proba(train.loc[mask, features])[:, 1]
    return oof, y


def fit_cv_calibrator(
    train: pd.DataFrame,
    features: list[str],
    threshold: float,
    *,
    k: int = 4,
) -> IsotonicRegression | None:
    """Isotonic calibrator fitted on out-of-fold predictions (the fix)."""
    p, y = out_of_fold_probabilities(train, features, threshold, k=k)
    ok = np.isfinite(p)
    if ok.sum() < 100 or y[ok].sum() < 5:
        return None
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p[ok], y[ok])
    return iso


def reliability_gap(prob, obs_event, *, bins: int = 8, min_n: int = 20) -> dict:
    """How far from the diagonal, weighted by how many days each bin holds.

    The headline number is the EXPECTED CALIBRATION ERROR: the average gap
    between stated and observed probability, weighted by bin population. A
    single badly-calibrated bin holding 43 of 12,017 days barely moves it,
    which is correct -- and is also why ECE alone is not enough, so the worst
    populated bin is reported alongside it.
    """
    p = np.asarray(prob, dtype="float64")
    o = np.asarray(obs_event, dtype="float64")
    ok = np.isfinite(p) & np.isfinite(o)
    p, o = p[ok], o[ok]
    if not len(p):
        return {"ece": np.nan, "worst_gap": np.nan, "worst_bin": None, "n": 0}

    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece, worst, worst_bin = 0.0, 0.0, None
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        gap = abs(p[m].mean() - o[m].mean())
        ece += gap * m.sum() / len(p)
        if m.sum() >= min_n and gap > worst:
            worst, worst_bin = gap, (float(edges[b]), float(edges[b + 1]),
                                     float(p[m].mean()), float(o[m].mean()), int(m.sum()))
    return {"ece": float(ece), "worst_gap": float(worst),
            "worst_bin": worst_bin, "n": int(len(p))}


# ---------------------------------------------------------- pooled Platt ---
#
# MEASURED FIX (docs/DATA_NOTES.md item 37). Per-lead isotonic calibration
# collapsed: only 141 validation days sat above raw probability 0.10, and
# isotonic -- a step function -- flattened everything from 0.25 upward onto a
# single value of 0.278. A day the model rated 0.9 came out identical to one it
# rated 0.3.
#
# Two changes, both supported by a held-out-by-lead comparison:
#   POOL ACROSS LEADS. 5x the calibration data. Mean ECE 0.0170 -> 0.0037.
#   USE A SMOOTH MAPPING. Logistic-on-log-odds has two parameters and cannot
#   collapse a range the way isotonic can on thin data. Same ECE as pooled
#   isotonic, but it keeps the upper range spread (raw 0.9 -> 0.335 rather
#   than 0.312) which is what the display number needs.
#
# Cross-validated calibration within the training season was tried first and
# was WORSE than doing nothing (ECE 0.0210 vs 0.0170). Each fold's model is
# trained on less data and produces differently-scaled probabilities, so the
# mapping learned from them does not transfer. Recorded so it is not retried.


class PlattCalibrator:
    """Logistic regression on the log-odds. Monotone, smooth, 2 parameters."""

    __slots__ = ("model", "n_fit", "n_events")

    def __init__(self):
        self.model = None
        self.n_fit = 0
        self.n_events = 0

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, prob, obs_event) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        p = np.asarray(prob, dtype="float64")
        y = np.asarray(obs_event).astype(int)
        ok = np.isfinite(p)
        p, y = p[ok], y[ok]
        if len(p) < 50 or y.sum() < 5 or y.sum() == len(y):
            return self                      # too thin to fit; predict() passes through
        self.model = LogisticRegression(C=1.0, max_iter=1000).fit(self._logit(p), y)
        self.n_fit, self.n_events = int(len(p)), int(y.sum())
        return self

    def predict(self, prob) -> np.ndarray:
        p = np.asarray(prob, dtype="float64")
        if self.model is None:
            return p                          # uncalibrated beats mis-calibrated
        return self.model.predict_proba(self._logit(p))[:, 1]

    @property
    def fitted(self) -> bool:
        return self.model is not None


def fit_pooled_calibrators(
    frames: dict, threshold: float
) -> PlattCalibrator:
    """One calibrator from every lead's validation predictions pooled.

    `frames` maps lead -> (raw_probabilities, observed_event).
    """
    p = np.concatenate([np.asarray(v[0], dtype="float64") for v in frames.values()])
    y = np.concatenate([np.asarray(v[1]).astype(int) for v in frames.values()])
    return PlattCalibrator().fit(p, y)
