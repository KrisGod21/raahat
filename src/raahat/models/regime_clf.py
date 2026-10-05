"""Stage A -- the regime classifier (spec 6.1).

Reads the FORECAST mesh at lead L and emits six calibrated probabilities. This
is the piece that makes the whole project work at lead time: a detector run on
analysis fields tells you what regime it *was*, which is useless three days
before the warning has to go out.

Design notes that are not obvious from the spec:

* One model per lead day (spec 6.1). Five small models beat one model with
  lead as a feature, and each trains in seconds.

* The classifier is NATIONAL. It trains on one row per (valid_date, lead_day),
  not one row per district-day. Training on district-replicated rows would
  multiply the apparent sample size by ~750 while adding no information, and
  would make every cross-validation estimate meaningless.

* Soft labels are handled by row replication with sample weights, as spec 6.1
  directs: a day that is {LPS: 0.6, ACTIVE: 0.4} becomes two weighted rows.

* class_weight='balanced' is applied over the WEIGHTED class support rather
  than raw row counts, because after soft-label expansion the row count is no
  longer the support. Same intent as the spec, correct arithmetic.

* subsample requires subsample_freq >= 1 in LightGBM or it is silently
  ignored. That is set here; without it the spec's 0.8 does nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import lightgbm as lgb

from raahat import contract as C

_MODEL_CFG = C.load_config("model")
_CFG = _MODEL_CFG["regime_clf"]

SOFT_LABEL_COL = "label_regime_soft"
HARD_LABEL_COL = "label_regime"


# ------------------------------------------------------------- helpers ---


def soft_label_matrix(df: pd.DataFrame) -> np.ndarray:
    """(n, 6) soft-label matrix from either the soft or the hard label column."""
    if SOFT_LABEL_COL in df.columns and df[SOFT_LABEL_COL].notna().any():
        return np.vstack([np.asarray(v, dtype=np.float64) for v in df[SOFT_LABEL_COL]])
    if HARD_LABEL_COL not in df.columns:
        raise KeyError(f"need '{SOFT_LABEL_COL}' or '{HARD_LABEL_COL}' to train")
    idx = df[HARD_LABEL_COL].map({r: i for i, r in enumerate(C.REGIMES)})
    if idx.isna().any():
        bad = sorted(set(df.loc[idx.isna(), HARD_LABEL_COL]))
        raise ValueError(f"unknown regime label(s): {bad}")
    out = np.zeros((len(df), C.N_REGIMES))
    out[np.arange(len(df)), idx.astype(int)] = 1.0
    return out


def expand_soft_labels(
    X: pd.DataFrame, soft: np.ndarray, *, min_weight: float = 1e-6
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """One weighted row per non-zero regime component (spec 6.1).

    Returns (X_expanded, y_expanded, weights).
    """
    rows, classes = np.nonzero(soft > min_weight)
    return X.iloc[rows].reset_index(drop=True), classes, soft[rows, classes]


def _balanced_class_weights(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-row multiplier that equalises WEIGHTED support across classes."""
    support = np.array([w[y == k].sum() for k in range(C.N_REGIMES)], dtype=float)
    present = support > 0
    factor = np.ones(C.N_REGIMES)
    factor[present] = support[present].sum() / (present.sum() * support[present])
    return factor[y]


def entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row. High = transitional day (spec 5.3)."""
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


# --------------------------------------------------------------- model ---


class _BoosterProba:
    """A restored LightGBM Booster wearing an sklearn-shaped face.

    Reconstructing an LGBMClassifier from a saved booster by assigning its
    private attributes does not restore the objective, and prediction then
    fails with 'Number of classes must be 1 for non-multiclass training'.
    Wrapping the booster is the supported route.
    """

    __slots__ = ("booster", "classes_")

    def __init__(self, booster: lgb.Booster, classes: list[int]):
        self.booster = booster
        self.classes_ = np.asarray(classes, dtype=int)

    def predict_proba(self, X) -> np.ndarray:
        p = np.asarray(self.booster.predict(X))
        return p.reshape(len(X), -1)


@dataclass
class RegimeClassifier:
    """Per-lead LightGBM multiclass classifier with per-class isotonic calibration."""

    features: list[str] = field(default_factory=lambda: list(C.STAGE_A_FEATURES))
    params: dict = field(default_factory=lambda: dict(_CFG))
    models: dict[int, lgb.LGBMClassifier] = field(default_factory=dict)
    calibrators: dict[int, list[IsotonicRegression | None]] = field(default_factory=dict)
    #: regimes actually present in each lead's training data. A class absent
    #: here can never be predicted -- see _proba_full.
    seen_classes: dict[int, list[int]] = field(default_factory=dict)
    train_report: dict = field(default_factory=dict)

    @staticmethod
    def _proba_full(model: lgb.LGBMClassifier, X) -> np.ndarray:
        """predict_proba scattered into the full 6-slot contract vector.

        LightGBM emits one column per class it SAW IN TRAINING. WD is rare in
        JJAS (spec 6.1 puts it at 2-5% of days) and can be absent from a
        training season entirely, so the raw output is not contract-shaped.
        Absent classes get probability zero, which is the honest answer: a
        model that never saw a Western Disturbance cannot predict one.
        """
        raw = model.predict_proba(X)
        classes = np.asarray(model.classes_, dtype=int)
        if raw.shape[1] == C.N_REGIMES and np.array_equal(classes, np.arange(C.N_REGIMES)):
            return raw
        out = np.zeros((raw.shape[0], C.N_REGIMES))
        out[:, classes] = raw
        return out

    # -- fit ---------------------------------------------------------------

    def _lgb_params(self) -> dict:
        p = self.params
        return dict(
            objective=p.get("objective", "multiclass"),
            num_class=p.get("num_class", C.N_REGIMES),
            n_estimators=p.get("n_estimators", 600),
            learning_rate=p.get("learning_rate", 0.05),
            num_leaves=p.get("num_leaves", 8),
            min_child_samples=p.get("min_child_samples", 5),
            min_split_gain=p.get("min_split_gain", 0.0),
            subsample=p.get("subsample", 0.8),
            subsample_freq=1,  # without this LightGBM ignores subsample
            colsample_bytree=p.get("colsample_bytree", 0.8),
            random_state=p.get("random_state", 42),
            n_jobs=-1,
            verbose=-1,
        )

    def fit(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame | None = None,
        *,
        leads: list[int] | None = None,
        verbose: bool = True,
    ) -> RegimeClassifier:
        """Train one model per lead; calibrate on `valid` only (spec 6.1)."""
        leads = leads or sorted(train["lead_day"].unique().tolist())
        missing = [f for f in self.features if f not in train.columns]
        if missing:
            raise KeyError(f"{len(missing)} Stage-A feature(s) missing: {missing[:8]}")

        for lead in leads:
            tr = train[train["lead_day"] == lead]
            if tr.empty:
                continue
            Xe, ye, we = expand_soft_labels(tr[self.features], soft_label_matrix(tr))
            if self.params.get("class_weight") == "balanced":
                we = we * _balanced_class_weights(ye, we)

            present = sorted(set(ye.tolist()))
            params = self._lgb_params()
            params["num_class"] = len(present)  # LightGBM counts only seen classes
            model = lgb.LGBMClassifier(**params)
            model.fit(Xe, ye, sample_weight=we)
            self.models[int(lead)] = model
            self.seen_classes[int(lead)] = present

            cal: list[IsotonicRegression | None] = [None] * C.N_REGIMES
            if valid is not None:
                va = valid[valid["lead_day"] == lead]
                if not va.empty:
                    raw = self._proba_full(model, va[self.features])
                    target = soft_label_matrix(va)
                    for k in range(C.N_REGIMES):
                        if target[:, k].sum() <= 0:
                            continue  # class absent in validation -- leave uncalibrated
                        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                        iso.fit(raw[:, k], target[:, k])
                        cal[k] = iso
            self.calibrators[int(lead)] = cal

            if verbose:
                n_cal = sum(c is not None for c in cal)
                absent = [C.REGIMES[k] for k in range(C.N_REGIMES) if k not in present]
                note = f"  UNSEEN: {absent}" if absent else ""
                print(
                    f"  lead {lead}: {len(tr):>5,} days -> {len(Xe):>5,} weighted rows, "
                    f"{n_cal}/{C.N_REGIMES} classes calibrated{note}"
                )

        never_seen = sorted(
            set(range(C.N_REGIMES)) - {k for ks in self.seen_classes.values() for k in ks}
        )
        if never_seen and verbose:
            names = [C.REGIMES[k] for k in never_seen]
            print(
                f"\n  WARNING: {names} absent from the training season entirely.\n"
                f"  These regimes can never be predicted. Spec 6.1 anticipates this for\n"
                f"  WD in JJAS -- merge or softly weight, do not oversample into fake\n"
                f"  confidence. Record the decision in docs/DATA_NOTES.md."
            )

        self.train_report = {
            "leads": sorted(self.models),
            "n_features": len(self.features),
            "calibrated": valid is not None,
            "never_seen_regimes": [C.REGIMES[k] for k in never_seen],
        }
        return self

    # -- predict -----------------------------------------------------------

    def predict_proba(self, df: pd.DataFrame, *, calibrated: bool = True) -> np.ndarray:
        """(n, 6) probabilities in contract regime order. Rows sum to 1."""
        if not self.models:
            raise RuntimeError("classifier is not fitted")
        out = np.zeros((len(df), C.N_REGIMES))
        for lead, model in self.models.items():
            mask = (df["lead_day"] == lead).to_numpy()
            if not mask.any():
                continue
            raw = self._proba_full(model, df.loc[mask, self.features])
            if calibrated:
                cal = self.calibrators.get(lead) or [None] * C.N_REGIMES
                adj = np.column_stack(
                    [cal[k].predict(raw[:, k]) if cal[k] is not None else raw[:, k]
                     for k in range(C.N_REGIMES)]
                )
                total = adj.sum(axis=1, keepdims=True)
                # a fully-clipped row would divide by zero; fall back to raw
                flat = (total.ravel() <= 1e-9)
                adj[flat] = raw[flat]
                total[flat] = adj[flat].sum(axis=1, keepdims=True)
                raw = adj / total
            out[mask] = raw

        unseen = sorted(set(df["lead_day"].unique()) - set(self.models))
        if unseen:
            raise KeyError(f"no model trained for lead day(s) {unseen}")
        return out

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Probabilities plus the derived columns of spec 5.3."""
        probs = self.predict_proba(df)
        out = pd.DataFrame(probs, columns=C.REGIME_PROB_COLS, index=df.index)
        out["regime_entropy"] = entropy(probs)
        out["regime_argmax"] = [C.REGIMES[i] for i in probs.argmax(axis=1)]
        return out

    # -- evaluation --------------------------------------------------------

    def report(self, df: pd.DataFrame) -> dict:
        """Per-class precision/recall/F1 with support, plus a confusion matrix.

        Scored against the ARGMAX of the soft label, which is the honest
        hard-label view of a soft problem. Support counts are printed because
        spec 6.1 requires them -- WD is rare in JJAS and a bare F1 hides that.
        """
        truth = soft_label_matrix(df).argmax(axis=1)
        pred = self.predict_proba(df).argmax(axis=1)
        labels = list(range(C.N_REGIMES))
        p, r, f1, sup = precision_recall_fscore_support(
            truth, pred, labels=labels, zero_division=0
        )
        return {
            "accuracy": float((truth == pred).mean()),
            "per_class": [
                {
                    "regime": C.REGIMES[k],
                    "precision": float(p[k]),
                    "recall": float(r[k]),
                    "f1": float(f1[k]),
                    "support": int(sup[k]),
                }
                for k in labels
            ],
            "confusion": confusion_matrix(truth, pred, labels=labels).tolist(),
            "n": int(len(df)),
        }

    def accuracy_by_lead(self, df: pd.DataFrame) -> dict[int, float]:
        acc: dict[int, float] = {}
        for lead in sorted(df["lead_day"].unique()):
            sub = df[df["lead_day"] == lead]
            truth = soft_label_matrix(sub).argmax(axis=1)
            pred = self.predict_proba(sub).argmax(axis=1)
            acc[int(lead)] = float((truth == pred).mean())
        return acc

    def leakage_canary(self, df: pd.DataFrame) -> dict:
        """Spec 5.7: accuracy MUST degrade with lead time.

        If day-5 accuracy is close to day-1 accuracy then something observed
        after initialisation has reached the feature set.

        One trap the spec does not mention: a DEGENERATE model -- one that has
        collapsed to predicting the majority class -- also has flat accuracy
        across leads, and so trips this canary for entirely the wrong reason.
        Chasing a phantom leak when the real fault is an under-fitted model
        costs days, so degeneracy is checked first and reported separately.
        """
        acc = self.accuracy_by_lead(df)
        limit = float(self.params.get("leakage_canary_max_ratio", 0.95))
        lo, hi = min(acc), max(acc)
        ratio = acc[hi] / acc[lo] if acc[lo] > 0 else float("inf")

        truth = soft_label_matrix(df).argmax(axis=1)
        pred = self.predict_proba(df).argmax(axis=1)
        n_predicted = int(len(set(pred.tolist())))
        majority = float(np.bincount(truth, minlength=C.N_REGIMES).max() / len(truth))
        overall = float((truth == pred).mean())
        degenerate = n_predicted < 2 or overall <= majority + 1e-9

        if degenerate:
            msg = (
                f"DEGENERATE MODEL -- predicts {n_predicted} distinct class(es); "
                f"accuracy {overall:.3f} vs majority base rate {majority:.3f}. "
                "The canary is uninformative until the model learns something. "
                "Check sample size against min_child_samples / num_leaves."
            )
        else:
            msg = (
                f"acc(lead {hi})/acc(lead {lo}) = {ratio:.3f} "
                f"{'<=' if ratio <= limit else '>'} {limit} -- "
                f"{'ok' if ratio <= limit else 'LEAKAGE SUSPECTED'}"
            )

        return {
            "accuracy_by_lead": acc,
            "first_lead": lo,
            "last_lead": hi,
            "ratio": float(ratio),
            "max_ratio": limit,
            "degenerate": degenerate,
            "n_classes_predicted": n_predicted,
            "majority_base_rate": majority,
            "overall_accuracy": overall,
            "passed": bool(not degenerate and ratio <= limit),
            "message": msg,
        }

    # -- persistence -------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """Freeze to models/frozen/{run_id}/ in the layout of spec 13.3."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for lead, model in self.models.items():
            booster = model.booster_ if hasattr(model, "booster_") else model.booster
            booster.save_model(str(directory / f"regime_clf_lead{lead}.txt"))
        import pickle

        with (directory / "regime_calibrators.pkl").open("wb") as fh:
            pickle.dump(self.calibrators, fh)
        (directory / "feature_list.json").write_text(
            json.dumps(
                {
                    "stage_a_features": self.features,
                    "regimes": C.REGIMES,
                    "seen_classes": {str(k): v for k, v in self.seen_classes.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> RegimeClassifier:
        import pickle

        directory = Path(directory)
        meta = json.loads((directory / "feature_list.json").read_text(encoding="utf-8"))
        if meta["regimes"] != C.REGIMES:
            raise ValueError(
                "frozen model was trained on a different regime taxonomy:\n"
                f"  frozen : {meta['regimes']}\n  current: {C.REGIMES}"
            )
        obj = cls(features=meta["stage_a_features"])
        seen = {int(k): v for k, v in meta.get("seen_classes", {}).items()}
        for path in sorted(directory.glob("regime_clf_lead*.txt")):
            lead = int(path.stem.replace("regime_clf_lead", ""))
            booster = lgb.Booster(model_file=str(path))
            obj.models[lead] = _BoosterProba(
                booster, seen.get(lead, list(range(C.N_REGIMES)))
            )
            obj.seen_classes[lead] = seen.get(lead, list(range(C.N_REGIMES)))
        with (directory / "regime_calibrators.pkl").open("rb") as fh:
            obj.calibrators = pickle.load(fh)
        return obj


def format_report(rep: dict) -> str:
    """Human-readable version of report(), with support counts (spec 6.1)."""
    lines = [
        f"  accuracy {rep['accuracy']:.3f}  (n={rep['n']:,})",
        f"  {'regime':<18}{'prec':>7}{'recall':>8}{'f1':>7}{'support':>9}",
    ]
    for row in rep["per_class"]:
        flag = "  <- thin" if row["support"] < 30 else ""
        lines.append(
            f"  {row['regime']:<18}{row['precision']:>7.3f}{row['recall']:>8.3f}"
            f"{row['f1']:>7.3f}{row['support']:>9,}{flag}"
        )
    return "\n".join(lines)
