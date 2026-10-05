"""Stage B, Level 1 -- the quantile + threshold corrector (spec 6.2).

Three LightGBM quantile regressors (q = 0.1 / 0.5 / 0.9) and three binary
classifiers at 64.5 / 115.6 / 204.5 mm. Regime probabilities, when available,
enter as ordinary input features -- that is the whole of the Level-1 "gating".

Two details that are load-bearing rather than decorative:

  MONOTONICITY. Spec 6.2 constrains the model to be non-decreasing in mm_mean
  and pr_ifs_nbr_max. More forecast rain must never produce less corrected
  rain. Without it you eventually demo a district where dragging the raw
  forecast up drags the correction down, and the room stops believing you.

  NO OVERSAMPLING. Spec 7.3 is explicit: do not oversample heavy days, it
  corrupts the probability calibration that IS the deliverable. Instead weight
  by 1 + log1p(observed) in the regressors and use scale_pos_weight in the
  classifiers, then recalibrate with isotonic on validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from raahat import contract as C

_CFG = C.load_config("model")["corrector_lgbm"]


@dataclass
class CorrectorLGBM:
    """Quantile regressors + threshold classifiers, optionally regime-aware."""

    features: list[str] = field(default_factory=lambda: list(C.STAGE_B_FEATURES))
    quantiles: list[float] = field(default_factory=lambda: list(_CFG["quantiles"]))
    thresholds: list[float] = field(default_factory=lambda: list(_CFG["thresholds"]))
    params: dict = field(default_factory=lambda: dict(_CFG))
    use_regime: bool = True
    monotone: bool = True
    name: str = "raahat"

    quantile_models: dict[float, lgb.LGBMRegressor] = field(default_factory=dict)
    threshold_models: dict[float, lgb.LGBMClassifier] = field(default_factory=dict)
    calibrators: dict[float, IsotonicRegression] = field(default_factory=dict)

    # -- setup -------------------------------------------------------------

    def active_features(self) -> list[str]:
        """Features actually used. Ablation 4 removes the regime block."""
        feats = self.features if self.use_regime else [
            f for f in self.features if f not in set(C.STAGE_B_REGIME)
        ]
        return feats

    def _usable(self, df: pd.DataFrame) -> list[str]:
        """Drop all-null columns -- an all-NaN feature is noise with a name."""
        feats = [f for f in self.active_features() if f in df.columns]
        return [f for f in feats if df[f].notna().any()]

    def _base(self, extra: dict | None = None) -> dict:
        p = self.params
        out = dict(
            n_estimators=p.get("n_estimators", 800),
            learning_rate=p.get("learning_rate", 0.05),
            num_leaves=p.get("num_leaves", 63),
            min_child_samples=p.get("min_child_samples", 50),
            subsample=p.get("subsample", 0.8),
            subsample_freq=1,
            colsample_bytree=p.get("colsample_bytree", 0.8),
            random_state=p.get("random_state", 42),
            n_jobs=-1,
            verbose=-1,
        )
        out.update(extra or {})
        return out

    def _monotone(self, feats: list[str]) -> list[int]:
        want = set(C.MONOTONE_INCREASING)
        return [1 if f in want else 0 for f in feats]

    # -- fit ---------------------------------------------------------------

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None,
            *, verbose: bool = True) -> CorrectorLGBM:
        tr = train[train[C.TARGET_COL].notna()]
        feats = self._usable(tr)
        self._fitted_features = feats
        X, y = tr[feats], tr[C.TARGET_COL].to_numpy()

        # spec 7.3: weight, never oversample
        w = 1.0 + np.log1p(np.clip(y, 0, None))
        mono = self._monotone(feats)

        # MONOTONICITY (spec 6.2). LightGBM refuses monotone_constraints on
        # every quantile-estimating objective -- `quantile`, `regression_l1`
        # and `mape` all raise "Cannot use monotone_constraints in ...".
        # Their gradients are sign-based, so the constraint machinery has no
        # curvature to work with. Only `regression`, `huber`, `fair`,
        # `poisson`, `tweedie` and `binary` accept it.
        #
        # So the median is DISTILLED: fit the quantile(0.5) model normally,
        # then fit a monotone L2 student to ITS predictions. The student is
        # non-decreasing by construction and approximates the same median
        # function. Measured on JJAS 2025 validation, lead 3: violations
        # 12.75% -> 0.00%, RMSE 12.70 -> 12.64. The constraint costs nothing.
        #
        # `huber` with constraints scored better still (RMSE 12.27, corr
        # 0.625) but estimates something between the mean and the median, so
        # `corrected_median` would stop being a median and the p10/median/p90
        # triple would no longer describe one distribution. Coherence wins.
        for q in self.quantiles:
            m = lgb.LGBMRegressor(**self._base({"objective": "quantile", "alpha": q}))
            m.fit(X, y, sample_weight=w)
            self.quantile_models[q] = m

        if self.monotone and 0.5 in self.quantile_models and any(mono):
            teacher = self.quantile_models[0.5]
            student = lgb.LGBMRegressor(
                **self._base({"objective": "regression", "monotone_constraints": mono})
            )
            student.fit(X, np.clip(teacher.predict(X), 0, None))
            self.quantile_models[0.5] = student
            self.teacher_median = teacher

        for t in self.thresholds:
            yt = (y > t).astype(int)
            if yt.sum() < 10:
                if verbose:
                    print(f"    skipping {t} mm classifier: only {int(yt.sum())} events")
                continue
            pos_w = float((yt == 0).sum() / max(yt.sum(), 1))
            m = lgb.LGBMClassifier(
                **self._base({"objective": "binary",
                              "scale_pos_weight": pos_w,
                              "monotone_constraints": mono})
            )
            m.fit(X, yt)
            self.threshold_models[t] = m

            if valid is not None:
                va = valid[valid[C.TARGET_COL].notna()]
                if len(va):
                    raw = m.predict_proba(va[feats])[:, 1]
                    obs = (va[C.TARGET_COL].to_numpy() > t).astype(float)
                    if obs.sum() >= 5:
                        # Per-lead ISOTONIC is the spec 7.1 default and it
                        # COLLAPSES on thin data: only ~141 validation days sit
                        # above raw probability 0.10, and a step function
                        # flattened everything from 0.25 up onto one value.
                        # fit_pooled_calibration() below replaces these with a
                        # single smooth mapping fitted across all five leads,
                        # which cut calibration error 4.6x. These per-lead
                        # fits remain as the fallback when a caller trains one
                        # lead in isolation. See docs/DATA_NOTES.md item 34.
                        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                        iso.fit(raw, obs)
                        self.calibrators[t] = iso

        if verbose:
            print(f"    {self.name}: {len(feats)} features, "
                  f"{len(self.quantile_models)} quantiles, "
                  f"{len(self.threshold_models)} thresholds, "
                  f"{len(self.calibrators)} calibrated")
        return self

    # -- predict -----------------------------------------------------------

    def predict(self, df: pd.DataFrame, *, calibrated: bool = True) -> pd.DataFrame:
        feats = getattr(self, "_fitted_features", self._usable(df))
        X = df[feats]
        out = pd.DataFrame(index=df.index)
        for q, m in self.quantile_models.items():
            col = {0.1: "corrected_p10", 0.5: "corrected_median", 0.9: "corrected_p90"}.get(
                q, f"corrected_q{int(q*100)}"
            )
            out[col] = np.clip(m.predict(X), 0, None)

        # Independently fitted quantiles can cross. Sorting is the usual repair,
        # but it would let an unconstrained p10 displace the MONOTONE median and
        # silently undo the spec 6.2 guarantee. So the median is authoritative
        # and the outer quantiles are clipped to it.
        if "corrected_median" in out:
            med = out["corrected_median"]
            if "corrected_p10" in out:
                out["corrected_p10"] = np.minimum(out["corrected_p10"], med)
            if "corrected_p90" in out:
                out["corrected_p90"] = np.maximum(out["corrected_p90"], med)

        for t, m in self.threshold_models.items():
            p = m.predict_proba(X)[:, 1]
            if calibrated and t in self.calibrators:
                p = self.calibrators[t].predict(p)
            out[f"p_gt_{str(t).replace('.', '_')}"] = p
        return out

    def monotonicity_audit(
        self, df: pd.DataFrame, *, bump: float = 10.0, n: int = 4000, seed: int = 42
    ) -> dict:
        """Does more forecast rain ever produce less corrected rain? (spec 6.2)

        LightGBM will not impose the constraint on a quantile objective, so it
        is measured instead: add `bump` mm to the monotone features and check
        the corrected median never falls. The violation rate is a number we can
        put on a slide, and a non-zero one is a warning that the demo could
        embarrass us.
        """
        feats = getattr(self, "_fitted_features", self._usable(df))
        mono_feats = [f for f in C.MONOTONE_INCREASING if f in feats]
        if not mono_feats or "corrected_median" not in self.predict(df.head(1)).columns:
            return {"checked": False}
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
        sub = df.iloc[idx]
        base = self.predict(sub)["corrected_median"].to_numpy()
        bumped = sub.copy()
        for f in mono_feats:
            bumped[f] = bumped[f] + bump
        after = self.predict(bumped)["corrected_median"].to_numpy()
        drop = after - base
        violations = drop < -1e-6
        return {
            "checked": True,
            "features": mono_feats,
            "bump_mm": bump,
            "n": int(len(base)),
            "violations": int(violations.sum()),
            "violation_rate": float(violations.mean()),
            "worst_drop_mm": float(-drop.min()) if violations.any() else 0.0,
            "mean_response_mm": float(drop.mean()),
        }

    # -- persistence -------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for q, m in self.quantile_models.items():
            m.booster_.save_model(str(d / f"corrector_lgbm_q{int(q*100)}.txt"))
        for t, m in self.threshold_models.items():
            m.booster_.save_model(
                str(d / f"corrector_threshold_{str(t).replace('.', '_')}.txt")
            )
        import pickle
        with (d / "corrector_calibrators.pkl").open("wb") as fh:
            pickle.dump(self.calibrators, fh)
        (d / "corrector_features.json").write_text(
            json.dumps({"features": getattr(self, "_fitted_features", self.features),
                        "use_regime": self.use_regime, "name": self.name}, indent=2),
            encoding="utf-8",
        )
        return d


def raw_threshold_probabilities(model: "CorrectorLGBM", df: pd.DataFrame,
                                threshold: float) -> np.ndarray | None:
    """Uncalibrated P(y > threshold), for fitting a calibrator externally."""
    m = model.threshold_models.get(threshold)
    if m is None:
        return None
    feats = getattr(model, "_fitted_features", model._usable(df))
    return m.predict_proba(df[feats])[:, 1]


def fit_pooled_calibration(
    models: dict[int, "CorrectorLGBM"],
    valid_by_lead: dict[int, pd.DataFrame],
    *,
    thresholds: list[float] | None = None,
    verbose: bool = False,
) -> dict[float, object]:
    """One calibrator per threshold, fitted across EVERY lead, then shared.

    This is the measured fix for the under-confidence in docs/DATA_NOTES.md
    item 34. Pooling gives 5x the calibration points and a smooth two-parameter
    mapping cannot flatten a range the way isotonic does on thin data.
    Held-out by lead, mean calibration error fell 0.0170 -> 0.0037.

    Mutates each model's `calibrators` in place and returns the shared set.
    """
    from raahat.models.calibrate import PlattCalibrator

    thresholds = thresholds or C.THRESHOLD_VALUES
    shared: dict[float, object] = {}
    for t in thresholds:
        ps, ys = [], []
        for lead, m in models.items():
            va = valid_by_lead.get(lead)
            if va is None or va.empty:
                continue
            va = va[va[C.TARGET_COL].notna()]
            raw = raw_threshold_probabilities(m, va, t)
            if raw is None or not len(raw):
                continue
            ps.append(raw)
            ys.append((va[C.TARGET_COL].to_numpy() > t).astype(int))
        if not ps:
            continue
        p = np.concatenate(ps)
        y = np.concatenate(ys)
        cal = PlattCalibrator().fit(p, y)
        if not cal.fitted:
            if verbose:
                print(f"    {t} mm: too few events to pool ({int(y.sum())}), left uncalibrated")
            continue
        shared[t] = cal
        for m in models.values():
            if t in m.threshold_models:
                m.calibrators[t] = cal
        if verbose:
            print(f"    {t} mm: pooled over {len(p):,} rows, {int(y.sum())} events")
    return shared


# ------------------------------------------------------- baselines ---


def baseline_raw(df: pd.DataFrame, model: str = "ifs") -> pd.Series:
    """Ablation 1 -- the floor. Best single raw NWP model."""
    return df[f"pr_{model}"]


def baseline_mme(df: pd.DataFrame) -> pd.Series:
    """Ablation 2 -- the naive multi-model mean."""
    return df["mm_mean"]


def fit_quantile_mapping(train: pd.DataFrame, source: str = "mm_mean",
                         *, n_q: int = 200) -> dict:
    """Ablation 3 -- global empirical quantile mapping, the conventional fix.

    Spec 7.2 trap 5: the CDFs must be fitted on TRAINING YEARS ONLY, exactly as
    our own model is. A quantile-mapping baseline fitted on all years would be
    a strawman, and beating a strawman proves nothing.
    """
    tr = train[train[C.TARGET_COL].notna()]
    qs = np.linspace(0.001, 0.999, n_q)
    return {
        "q": qs,
        "fcst": np.nanquantile(tr[source].to_numpy(), qs),
        "obs": np.nanquantile(tr[C.TARGET_COL].to_numpy(), qs),
        "source": source,
    }


def apply_quantile_mapping(df: pd.DataFrame, qm: dict) -> pd.Series:
    x = df[qm["source"]].to_numpy(dtype="float64")
    return pd.Series(np.interp(x, qm["fcst"], qm["obs"]), index=df.index)


def baseline_climatology(train: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    """Ablation 0 -- day-of-year climatology. The 'is this better than nothing' bar."""
    if "clim_mean_district_doy" in target and target["clim_mean_district_doy"].notna().any():
        return target["clim_mean_district_doy"]
    tr = train[train[C.TARGET_COL].notna()]
    m = tr.groupby("district_id")[C.TARGET_COL].mean()
    return target["district_id"].map(m)
