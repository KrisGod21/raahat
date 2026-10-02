"""THE ONLY FILE ALLOWED TO OPEN THE TEST BLOCK (spec 7.1, 16 phase 9).

Everything is fitted on train, calibrated on validation, and evaluated ONCE on
the held-out block. Models are frozen and a MANIFEST.json is written BEFORE the
test block is touched, so the numbers cannot be quietly re-tuned afterwards --
that ordering is the whole point and it is enforced below, not just intended.

Spec 13.3: nothing goes in the presentation that is not traceable to a
MANIFEST.json.

Usage:
    python scripts/final_eval.py
    python scripts/final_eval.py --dry-run    # everything except opening the test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raahat import contract as C  # noqa: E402
from raahat.models import corrector_lgbm as CL  # noqa: E402
from raahat.train import ablations, splits  # noqa: E402
from raahat.verify import atlas as ATL, metrics as M  # noqa: E402

FINAL = C.DATA_DIR.parent / "results" / "final"
FROZEN = C.DATA_DIR.parent / "models" / "frozen"


def config_fingerprint() -> dict:
    """Hash every config file. Stands in for a git SHA -- this is not a repo."""
    out = {}
    for p in sorted(C.CONFIG_DIR.glob("*.yaml")):
        out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def data_fingerprint() -> dict:
    out = {}
    for p in sorted((C.DATA_DIR / "processed").glob("*.parquet")):
        out[p.name] = {"bytes": p.stat().st_size,
                       "sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}
    return out


def per_regime_table(df: pd.DataFrame, preds: dict, threshold: float,
                     alpha: float) -> pd.DataFrame:
    """Spec 11.4 -- the single table that decides whether the project worked."""
    rows = []
    for regime, g in df.groupby("regime_argmax", dropna=False):
        idx = g.index
        rec = {"regime": regime, "n_district_days": len(g)}
        base = M.categorical_scores(g[C.TARGET_COL], g["pr_ifs"], threshold)
        rec["n_events"] = base["n_events"]
        rec["sufficient"] = base["sufficient"]
        rec["raw_csi"] = base["CSI"] if base["sufficient"] else np.nan
        for name, (vals, prob) in preds.items():
            v = np.asarray(vals)[df.index.get_indexer(idx)]
            if prob is not None:
                p = np.asarray(prob)[df.index.get_indexer(idx)]
                s = M.categorical_scores(
                    g[C.TARGET_COL], np.where(p > alpha, threshold + 1e-6, 0.0), threshold)
            else:
                s = M.categorical_scores(g[C.TARGET_COL], v, threshold)
            rec[name] = s["CSI"] if s["sufficient"] else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lead", type=int, default=3, help="headline lead (spec 11.4)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    FINAL.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    alpha = float(C.load_config("model")["decision"]["alpha_default"])
    t0 = time.time()

    print(f"run_id {run_id}   alpha {alpha}   headline lead {args.lead}")
    print(splits.describe(), "\n")

    # ---- fit on train, calibrate on validation. Test is still LOCKED. ----
    train = splits.load_split("train")
    valid = splits.load_split("validation")
    print(f"train {len(train):,} rows   validation {len(valid):,} rows")

    models = {}
    for lead in splits.LEAD_DAYS:
        tr, va = train[train.lead_day == lead], valid[valid.lead_day == lead]
        if tr.empty:
            continue
        m4 = CL.CorrectorLGBM(use_regime=False, monotone=True,
                              name=f"no-regime L{lead}").fit(tr, va, verbose=False)
        m5 = CL.CorrectorLGBM(use_regime=True, monotone=True,
                              name=f"full L{lead}").fit(tr, va, verbose=False)
        qm = CL.fit_quantile_mapping(tr)
        models[lead] = (m4, m5, qm)
        print(f"  lead {lead}: fitted ablation 4 and 5")

    # ---- freeze BEFORE opening the test block ----
    frozen_dir = FROZEN / run_id
    for lead, (m4, m5, _) in models.items():
        m5.save(frozen_dir / f"lead{lead}")
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "note": "not a git repository; configs are fingerprinted instead of a SHA",
        "config_sha256": config_fingerprint(),
        "data_sha256": data_fingerprint(),
        "splits": {k: [{"season": b["season"], "start": str(b["start"]),
                        "end": str(b["end"])} for b in v]
                   for k, v in splits.BLOCKS.items()},
        "alpha": alpha,
        "hyperparameters": C.load_config("model"),
        "regimes": C.REGIMES,
        "n_stage_b_features": len(C.STAGE_B_FEATURES),
        "frozen_before_test_opened": True,
    }
    frozen_dir.mkdir(parents=True, exist_ok=True)
    (frozen_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str),
                                              encoding="utf-8")
    print(f"\n  frozen to {frozen_dir} BEFORE the test block is opened")

    if args.dry_run:
        print("\n--dry-run: stopping before the test block is opened.")
        return 0

    # ---- open the test block. Once. ----
    splits.unlock_test_block(
        f"scripts/final_eval.py run {run_id}: single held-out evaluation (spec 7.1)"
    )
    test = splits.load_split("test")
    print(f"test {len(test):,} rows  "
          f"{test.valid_date.min()} .. {test.valid_date.max()}")

    results, tables = {}, {}
    for lead, (m4, m5, qm) in models.items():
        te = test[test.lead_day == lead]
        if te.empty:
            continue
        obs = te[C.TARGET_COL].to_numpy()
        p4, p5 = m4.predict(te), m5.predict(te)
        preds = {
            "0 climatology": (CL.baseline_climatology(train, te), None),
            "1 raw ECMWF": (CL.baseline_raw(te, "ifs"), None),
            "2 multi-model mean": (CL.baseline_mme(te), None),
            "3 quantile mapping": (CL.apply_quantile_mapping(te, qm), None),
            "4 RAAHAT no-regime": (p4["corrected_median"], p4.get("p_gt_64_5")),
            "5 RAAHAT full": (p5["corrected_median"], p5.get("p_gt_64_5")),
        }
        rows = []
        for name, (vals, prob) in preds.items():
            v = np.asarray(vals, dtype="float64")
            rec = {"config": name, "lead": lead, **M.continuous_scores(obs, v)}
            for t in C.THRESHOLD_VALUES:
                tag = f"{t:g}"
                pc = f"p_gt_{str(t).replace('.', '_')}"
                src = p5 if name.startswith("5") else (p4 if name.startswith("4") else None)
                if src is not None and pc in src:
                    pr = src[pc].to_numpy()
                    s = M.categorical_scores(obs, np.where(pr > alpha, t + 1e-6, 0.0), t)
                    b = M.brier(pr, obs, t)
                    rec[f"brier@{tag}"], rec[f"BSS@{tag}"] = b["brier"], b["bss"]
                else:
                    s = M.categorical_scores(obs, v, t)
                    rec[f"brier@{tag}"] = rec[f"BSS@{tag}"] = np.nan
                rec[f"POD@{tag}"], rec[f"FAR@{tag}"] = s["POD"], s["FAR"]
                rec[f"CSI@{tag}"], rec[f"n_ev@{tag}"] = s["CSI"], s["n_events"]
            rows.append(rec)
        tables[lead] = pd.DataFrame(rows)
        results[lead] = (te, preds, p4, p5)

    full = pd.concat(tables.values(), ignore_index=True)
    full.to_csv(FINAL / "scorecard_test.csv", index=False)

    lead = args.lead if args.lead in tables else min(tables)
    print("\n" + "=" * 100)
    print(f"FINAL SCORECARD -- HELD-OUT TEST BLOCK, lead {lead}, alpha {alpha}")
    print("=" * 100)
    t = tables[lead]
    print(f"  {'config':<22}{'bias':>8}{'RMSE':>8}{'MAE':>8}{'corr':>7}"
          f"{'POD':>8}{'FAR':>7}{'CSI':>7}{'BSS':>8}{'n_ev':>7}")
    for _, r in t.iterrows():
        bss = r["BSS@64.5"]
        print(f"  {r['config']:<22}{r['bias']:>8.2f}{r['rmse']:>8.2f}{r['mae']:>8.2f}"
              f"{r['corr']:>7.3f}{r['POD@64.5']:>8.3f}{r['FAR@64.5']:>7.3f}"
              f"{r['CSI@64.5']:>7.3f}"
              f"{(f'{bss:.3f}' if pd.notna(bss) else '-'):>8}{int(r['n_ev@64.5']):>7}")

    # ---- spec 11.4: per-regime CSI ----
    te, preds, p4, p5 = results[lead]
    pr_tab = per_regime_table(te, preds, C.THRESHOLDS["heavy"], alpha)
    pr_tab.to_csv(FINAL / "per_regime_csi_test.csv", index=False)
    print(f"\n  PER-REGIME CSI at {C.THRESHOLDS['heavy']} mm, lead {lead} (spec 11.4)")
    print(f"  {'regime':<20}{'n_dd':>8}{'n_ev':>7}{'raw':>8}{'no-regime':>11}{'full':>8}")
    for _, r in pr_tab.iterrows():
        if not r["sufficient"]:
            print(f"  {str(r['regime']):<20}{r['n_district_days']:>8}{int(r['n_events']):>7}"
                  f"{'  insufficient (spec 11.4)':>34}")
        else:
            print(f"  {str(r['regime']):<20}{r['n_district_days']:>8}{int(r['n_events']):>7}"
                  f"{r['raw_csi']:>8.3f}{r['4 RAAHAT no-regime']:>11.3f}"
                  f"{r['5 RAAHAT full']:>8.3f}")

    # ---- reliability, OUT OF SAMPLE this time ----
    rel = M.reliability(p5["p_gt_64_5"], te[C.TARGET_COL], C.THRESHOLDS["heavy"], bins=8)
    rel.to_csv(FINAL / "reliability_test.csv", index=False)
    print(f"\n  RELIABILITY (out of sample -- calibrator fitted on validation)")
    print(f"  {'bin':>12}{'forecast':>10}{'observed':>10}{'n':>8}")
    for _, r in rel[rel.n > 0].iterrows():
        print(f"  {r.bin_lower:.2f}-{r.bin_upper:.2f}{r.forecast_prob:>10.3f}"
              f"{r.observed_freq:>10.3f}{int(r.n):>8}")

    # ---- the Atlas, on the test block ----
    tea = te.copy()
    tea["corrected_median"] = p5["corrected_median"].to_numpy()
    tea["p_gt_64_5"] = p5["p_gt_64_5"].to_numpy()
    atl = ATL.build(tea, raw_col="pr_ifs", corrected_col="corrected_median",
                    prob_col="p_gt_64_5", threshold=C.THRESHOLDS["heavy"], alpha=alpha)
    atl.to_csv(FINAL / "regime_error_atlas_test.csv", index=False)
    grid = ATL.heatmap(atl, value="raw_bias", lead=lead, annotate="d_csi")
    print("\n" + ATL.format_heatmap(
        grid, title=f"REGIME ERROR ATLAS -- held-out test, lead {lead}"))

    manifest["test_opened_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["headline"] = {
        str(k): v[v.config.isin(["1 raw ECMWF", "4 RAAHAT no-regime", "5 RAAHAT full"])]
        .set_index("config")[["rmse", "corr", "CSI@64.5", "n_ev@64.5"]].to_dict("index")
        for k, v in tables.items()
    }
    manifest["elapsed_s"] = round(time.time() - t0, 1)
    (frozen_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (FINAL / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(f"\n  results -> {FINAL}")
    print(f"  manifest -> {frozen_dir / 'MANIFEST.json'}")
    print(f"  ({manifest['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
