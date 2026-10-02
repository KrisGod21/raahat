"""Build data/processed/predictions_{season}.parquet (spec 13.2).

Everything the API serves, precomputed. Spec 8 is emphatic: no model inference
happens inside a request handler. The only thing computed live is the
cost-loss recomputation, which is arithmetic on stored probabilities.

Per district-day-lead this writes: corrected quantiles, exceedance
probabilities, regime vector, warning colour with the safety guard, the top
three SHAP drivers, and the two nearest historical analogs.

NOTE on the test block. The held-out evaluation is already complete and frozen
(results/final/MANIFEST.json). Predictions are generated over every date here
because the DEMO needs them, and because generating a forecast for display is
what the system does operationally. No metric is recomputed; nothing here can
change the reported numbers.

Usage:
    python scripts/build_predictions.py --seasons 2024 2025
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raahat import contract as C  # noqa: E402
from raahat.decide import colour as CO  # noqa: E402
from raahat.explain import shap_drivers as SH  # noqa: E402
from raahat.models import analogs as AN, corrector_lgbm as CL  # noqa: E402
from raahat.train import splits  # noqa: E402

PROC = C.DATA_DIR / "processed"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025])
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--analog-k", type=int, default=2)
    args = ap.parse_args(argv)

    alpha = args.alpha if args.alpha is not None else float(
        C.load_config("model")["decision"]["alpha_default"])
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    t0 = time.time()
    print(f"run_id {run_id}   alpha {alpha}")

    # Fit on train, calibrate on validation -- exactly as final_eval did, so
    # what the UI shows is the same model the scorecard reports.
    train = splits.load_split("train")
    valid = splits.load_split("validation")
    print(f"train {len(train):,}   validation {len(valid):,}")

    models, valid_by_lead = {}, {}
    for lead in splits.LEAD_DAYS:
        tr, va = train[train.lead_day == lead], valid[valid.lead_day == lead]
        if tr.empty:
            continue
        models[lead] = CL.CorrectorLGBM(use_regime=True, monotone=True,
                                        name=f"L{lead}").fit(tr, va, verbose=False)
        valid_by_lead[lead] = va
        print(f"  lead {lead} fitted")

    # Replace the per-lead isotonic calibrators with ONE smooth mapping fitted
    # across every lead. Per-lead isotonic collapsed the whole upper range onto
    # a single value, so a district the model rated 90% reported the same
    # probability as one it rated 30% (docs/DATA_NOTES.md item 34).
    print("\n  pooling calibration across leads ...")
    CL.fit_pooled_calibration(models, valid_by_lead, verbose=True)

    print("\nbuilding analog memory from observed history ...")
    memory = AN.AnalogMemory().fit(train)
    print(f"  indexed {len(memory._store)} districts")

    master = pd.read_csv(C.DATA_DIR / "static" / "districts_master.csv")
    names = master.set_index("district_id")["district_name"]

    for season in args.seasons:
        path = PROC / f"features_{season}.parquet"
        if not path.exists():
            print(f"  skip {season}: no feature table")
            continue
        df = pd.read_parquet(path)
        parts = []
        for lead, model in models.items():
            sub = df[df.lead_day == lead]
            if sub.empty:
                continue
            p = model.predict(sub)
            out = pd.DataFrame(index=sub.index)
            out["district_id"] = sub["district_id"].to_numpy()
            out["valid_date"] = sub["valid_date"].to_numpy()
            out["lead_day"] = sub["lead_day"].to_numpy()
            out["init_datetime_utc"] = sub["init_datetime_utc"].to_numpy()
            out["raw_mm"] = sub["mm_mean"].to_numpy()
            for c in ("corrected_p10", "corrected_median", "corrected_p90"):
                out[c] = p[c].to_numpy() if c in p else np.nan
            for t in C.THRESHOLD_VALUES:
                c = f"p_gt_{str(t).replace('.', '_')}"
                out[c] = p[c].to_numpy() if c in p else np.nan
            for c in ("csgd_mu", "csgd_sigma", "csgd_delta"):
                out[c] = np.nan                       # Level 2, not built
            for c in C.REGIME_PROB_COLS + ["regime_argmax"]:
                out[c] = sub[c].to_numpy() if c in sub else np.nan

            dec = CO.assign(pd.concat([sub[["district_id", "valid_date", "lead_day",
                                            "mm_mean"]], p], axis=1), alpha=alpha)
            out["colour_code"] = dec["colour_code"].to_numpy()
            out["colour_alpha_used"] = dec["colour_alpha_used"].to_numpy()

            drivers = SH.attach(model.quantile_models[0.5], sub,
                                getattr(model, "_fitted_features", []))
            for c in drivers.columns:
                if c.startswith("shap_top") and not c.endswith(("_label", "_direction")):
                    out[c] = drivers[c].to_numpy()

            an = memory.query(sub, k=args.analog_k)
            for c in an.columns:
                if c in ("analog_date_1", "analog_date_2", "analog_obs_1", "analog_obs_2"):
                    out[c] = an[c].to_numpy()

            out["model_version"] = "raahat-level1-4class"
            out["run_id"] = run_id
            parts.append(out)
            print(f"  {season} lead {lead}: {len(out):,} rows")

        full = pd.concat(parts, ignore_index=True)
        cols = [f.name for f in C.predictions_schema()]
        for c in cols:
            if c not in full:
                full[c] = np.nan
        full = full[cols]
        full["district_id"] = full["district_id"].astype(np.int32)
        full["lead_day"] = full["lead_day"].astype(np.int8)
        full["colour_code"] = full["colour_code"].fillna(0).astype(np.int8)
        # Write to a temp file and rename. The API reads these parquet files
        # live, and a reader that opens one mid-write gets a truncated file and
        # returns a 500. os.replace is atomic on the same filesystem, so a
        # reader sees either the old complete file or the new one, never a
        # half-written one.
        out_path = PROC / f"predictions_{season}.parquet"
        tmp = out_path.with_suffix(".parquet.tmp")
        full.to_parquet(tmp, index=False)
        os.replace(tmp, out_path)

        print(f"\n  predictions_{season}.parquet  {len(full):,} rows")
        print(CO.summarise(full.assign(guard_applied=False)))
        ex = full[full.colour_code >= CO.ORANGE].head(1)
        if len(ex):
            r = ex.iloc[0]
            nm = names.get(int(r.district_id), "this district")
            print(f"\n  example: {nm}, {r.valid_date}, lead {int(r.lead_day)}")
            print(f"    raw {r.raw_mm:.0f} mm -> corrected {r.corrected_median:.0f} mm  "
                  f"P(>64.5)={r.p_gt_64_5:.2f}  {CO.NAMES[int(r.colour_code)]}")
            print(f"    {SH.narrative(r)}")
            print(f"    {AN.narrative(r, nm)}")

    print(f"\n  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
