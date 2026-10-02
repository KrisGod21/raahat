"""The five ablations of spec 6.3. This table IS the evidence, not a garnish.

  1  raw NWP (best single model)        the floor
  2  multi-model mean                   the naive ensemble
  3  global quantile mapping            the conventional post-processor
  4  our model, regime block removed    isolates the value of regime info
  5  our model, full                    the claim

Spec 6.3: "If (5) does not beat (4) on per-regime heavy-rain CSI, the USP is
not real and we must say so." That check is implemented here and printed
whether it passes or fails.

Everything is scored on VALIDATION. The test block stays locked until
scripts/final_eval.py (spec 7.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from raahat import contract as C
from raahat.models import corrector_lgbm as CL
from raahat.verify import metrics as M

#: Smallest CSI difference worth calling a difference. Roughly the
#: sampling noise on ~200 events; below this the sign is not stable.
MEANINGFUL_CSI_DELTA = 0.01

HEAVY = C.THRESHOLDS["heavy"]
VERY_HEAVY = C.THRESHOLDS["very_heavy"]


def run(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    thresholds: list[float] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Fit every configuration, score them all on `valid`."""
    thresholds = thresholds or C.THRESHOLD_VALUES
    preds: dict[str, pd.Series] = {}
    models: dict[str, object] = {}

    if verbose:
        print(f"  train {len(train):,} rows   validation {len(valid):,} rows")

    preds["1 raw ECMWF"] = CL.baseline_raw(valid, "ifs")
    preds["2 multi-model mean"] = CL.baseline_mme(valid)
    qm = CL.fit_quantile_mapping(train)
    preds["3 quantile mapping"] = CL.apply_quantile_mapping(valid, qm)
    preds["0 climatology"] = CL.baseline_climatology(train, valid)

    has_regime = any(
        c in train.columns and train[c].notna().any() for c in C.STAGE_B_REGIME
    )

    if verbose:
        print("  fitting ablation 4 (regime block removed) ...")
    m4 = CL.CorrectorLGBM(use_regime=False, name="4 RAAHAT no-regime")
    m4.fit(train, valid, verbose=verbose)
    p4 = m4.predict(valid)
    preds["4 RAAHAT no-regime"] = p4["corrected_median"]
    models["4 RAAHAT no-regime"] = (m4, p4)

    if has_regime:
        if verbose:
            print("  fitting ablation 5 (full, regime-gated) ...")
        m5 = CL.CorrectorLGBM(use_regime=True, name="5 RAAHAT full")
        m5.fit(train, valid, verbose=verbose)
        p5 = m5.predict(valid)
        preds["5 RAAHAT full"] = p5["corrected_median"]
        models["5 RAAHAT full"] = (m5, p5)
    elif verbose:
        print("  ablation 5 SKIPPED: regime features are not populated yet.")

    # --- scoring ---------------------------------------------------------
    # Continuous metrics use the point estimate. CATEGORICAL metrics use the
    # EXCEEDANCE PROBABILITY where one exists, thresholded at the cost-loss
    # alpha of spec 6.4 -- not the median.
    #
    # This distinction is the whole argument of spec 1.5. A median is a central
    # estimate; thresholding it for a rare event guarantees under-detection,
    # because the median of a right-skewed rainfall distribution sits well
    # below its tail. Scoring our probabilistic model by its median while the
    # baselines are scored by their only output would understate it AND would
    # not be how the system is actually used.
    alpha = float(C.load_config("model")["decision"]["alpha_default"])
    obs = valid[C.TARGET_COL].to_numpy()
    rows = []
    for name, series in preds.items():
        fc = np.asarray(series, dtype="float64")
        rec = {"config": name, **M.continuous_scores(obs, fc)}
        probs = models.get(name, (None, None))[1]
        for t in thresholds:
            tag = f"{t:g}"
            pcol = f"p_gt_{str(t).replace('.', '_')}"
            if probs is not None and pcol in probs:
                p = probs[pcol].to_numpy()
                s = M.categorical_scores(obs, np.where(p > alpha, t + 1e-6, 0.0), t)
                bs = M.brier(p, obs, t)
                rec[f"brier@{tag}"] = bs["brier"]
                rec[f"BSS@{tag}"] = bs["bss"]
                rec[f"src@{tag}"] = f"P>{alpha:g}"
            else:
                s = M.categorical_scores(obs, fc, t)
                rec[f"brier@{tag}"] = np.nan
                rec[f"BSS@{tag}"] = np.nan
                rec[f"src@{tag}"] = "value"
            rec[f"POD@{tag}"] = s["POD"]
            rec[f"FAR@{tag}"] = s["FAR"]
            rec[f"CSI@{tag}"] = s["CSI"]
            rec[f"n_ev@{tag}"] = s["n_events"]
        rows.append(rec)
    table = pd.DataFrame(rows).sort_values("config").reset_index(drop=True)

    verdict = _decision_gate(table, models, valid, thresholds)
    return table, {"models": models, "qm": qm, "verdict": verdict,
                   "has_regime": has_regime}


def _decision_gate(table, models, valid, thresholds) -> dict:
    """Spec 6.3 / spec 16 phase 4: does the gate earn its place?"""
    if "5 RAAHAT full" not in table["config"].values:
        return {"decided": False,
                "reason": "regime features not available; ablation 5 not run"}
    # Pick the most operationally decisive threshold that actually has enough
    # events. Spec 11.4 wants 115.6 mm, but comparing two zeros computed on 19
    # events is not a decision, it is noise -- and it silently reports "the
    # gate does not earn its place" when nothing was measured at all.
    col = None
    for t in sorted(thresholds, reverse=True):
        n = table[f"n_ev@{t:g}"].max()
        if n >= M.MIN_EVENTS:
            col = f"CSI@{t:g}"
            chosen_n, chosen_t = int(n), t
            break
    if col is None:
        return {"decided": False,
                "reason": f"no threshold has >= {M.MIN_EVENTS} events; "
                          f"max is {int(table[[c for c in table if c.startswith('n_ev@')]].max().max())}"}

    g4 = float(table.loc[table.config == "4 RAAHAT no-regime", col].iloc[0])
    g5 = float(table.loc[table.config == "5 RAAHAT full", col].iloc[0])
    delta = g5 - g4
    return {
        "decided": True, "metric": col, "threshold": chosen_t, "n_events": chosen_n,
        "no_regime": g4, "full": g5, "delta": delta,
        # A CSI delta of +0.0008 on 188 events is not a result. Measured
        # across leads the sign FLIPS (+0.0044 / +0.0008 / -0.0044), which is
        # what noise looks like. Require a margin big enough to mean something
        # rather than declaring victory on the fourth decimal.
        "gate_earns_its_place": bool(delta > MEANINGFUL_CSI_DELTA),
        "tied": bool(abs(delta) <= MEANINGFUL_CSI_DELTA),
        "margin_required": MEANINGFUL_CSI_DELTA,
        "note": (f"spec 11.4 asks for {VERY_HEAVY:g} mm; it had too few events, "
                 f"so {chosen_t:g} mm was used") if chosen_t != VERY_HEAVY else "",
    }


def format_table(table: pd.DataFrame, thresholds: list[float] | None = None) -> str:
    thresholds = thresholds or C.THRESHOLD_VALUES
    head = f"  {'config':<22}{'bias':>8}{'RMSE':>8}{'MAE':>8}{'corr':>7}"
    for t in thresholds:
        head += f"{'POD@'+f'{t:g}':>11}{'FAR':>7}{'CSI':>7}{'n':>7}{'from':>8}"
    lines = [head, "  " + "-" * (len(head) - 2)]
    for _, r in table.iterrows():
        line = (f"  {r['config']:<22}{r['bias']:>8.2f}{r['rmse']:>8.2f}"
                f"{r['mae']:>8.2f}{r['corr']:>7.3f}")
        for t in thresholds:
            tag = f"{t:g}"
            n = r[f"n_ev@{tag}"]
            mark = "*" if n < M.MIN_EVENTS else " "
            src = str(r.get(f"src@{tag}", ""))[:6]
            line += (f"{r[f'POD@{tag}']:>11.3f}{r[f'FAR@{tag}']:>7.3f}"
                     f"{r[f'CSI@{tag}']:>7.3f}{int(n):>6}{mark}{src:>8}")
        lines.append(line)
    lines.append(f"\n  * fewer than {M.MIN_EVENTS} events -- insufficient sample (spec 11.4)")
    return "\n".join(lines)


def per_regime_csi(
    valid: pd.DataFrame, preds: dict[str, pd.Series], threshold: float, *,
    regime_col: str = "regime_argmax",
) -> pd.DataFrame:
    """Spec 11.4 -- the single table that decides whether the project worked."""
    if regime_col not in valid or valid[regime_col].isna().all():
        return pd.DataFrame()
    rows = []
    for regime, idx in valid.groupby(regime_col).groups.items():
        sub = valid.loc[idx]
        rec = {"regime": regime, "n_days": len(sub)}
        for name, series in preds.items():
            s = M.categorical_scores(
                sub[C.TARGET_COL], np.asarray(series)[valid.index.get_indexer(idx)], threshold
            )
            rec[name] = s["CSI"] if s["sufficient"] else np.nan
            rec[f"{name}__n"] = s["n_events"]
        rows.append(rec)
    return pd.DataFrame(rows)
