"""Labels -> Stage A -> regime probabilities in the feature tables.

Phase 3 of spec 16, end to end:

  1. observed ACTIVE/BREAK index from the IMD grid   (spec 4.5, published)
  2. LPS detection from the ANALYSIS mesh             (spec 4.4, MSLP-based)
  3. soft 4-class labels                              (spec 6.1)
  4. Stage-A features from the FORECAST mesh at each lead
  5. train one classifier per lead, calibrate on validation
  6. broadcast the six probabilities to every district and write them into
     data/processed/features_{season}.parquet

The classifier only ever sees FORECAST fields. The labels come from analysis
and observations. Keeping those apart is the whole reason the system works at
lead time rather than only in hindsight (spec 3, USP 2).

Usage:
    python scripts/build_regimes.py
    python scripts/build_regimes.py --seasons 2024 2025 --out models/frozen/dev
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raahat import contract as C  # noqa: E402
from raahat.features import synoptic as SYN  # noqa: E402
from raahat.ingest import imd_grid  # noqa: E402
from raahat.labels import active_break as AB, assemble as ASM, lps_detector as LPS  # noqa: E402
from raahat.models.regime_clf import RegimeClassifier, format_report  # noqa: E402
from raahat.train import splits  # noqa: E402

INTERIM = C.DATA_DIR / "interim"
PROCESSED = C.DATA_DIR / "processed"


def jjas(df: pd.DataFrame, season: int) -> pd.DataFrame:
    d = pd.to_datetime(df["valid_date"])
    return df[(d >= f"{season}-06-01") & (d <= f"{season}-09-30")]


def build_labels(seasons: list[int], clim_years: list[int]) -> pd.DataFrame:
    """Observed soft labels, one row per valid_date."""
    ab = AB.observed_index(seasons, clim_years=clim_years)
    parts = []
    for s in seasons:
        path = INTERIM / f"mesh_analysis_{s}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run scripts/fetch_mesh.py --which analysis")
        mesh = jjas(pd.read_parquet(path), s)
        lps = LPS.detect(mesh)
        parts.append(ASM.assemble(jjas(ab, s), lps))
    return pd.concat(parts, ignore_index=True)


def build_stage_a(seasons: list[int], clim: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Stage-A feature table from the FORECAST mesh, joined to labels."""
    parts = []
    for s in seasons:
        path = INTERIM / f"mesh_forecast_{s}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run scripts/fetch_mesh.py --which forecast")
        mesh = jjas(pd.read_parquet(path), s)
        parts.append(SYN.build_stage_a(mesh, cmz_clim=clim))
    sa = pd.concat(parts, ignore_index=True)
    return sa.merge(labels, on="valid_date", how="inner")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    clim_years = [y for y in imd_grid.cached_years() if 2010 <= y <= 2023]
    if len(clim_years) < 3:
        raise SystemExit(f"need >=3 climatology years, have {clim_years}")
    print(f"climatology: {clim_years[0]}-{clim_years[-1]} ({len(clim_years)} years)")

    print("\n[1/5] observed labels ...")
    labels = build_labels(args.seasons, clim_years)
    print(ASM.summarise(labels))

    print("\n[2/5] Stage-A features from the forecast mesh ...")
    clim = AB.doy_climatology(AB.cmz_series(imd_grid.load(clim_years)))
    sa = build_stage_a(args.seasons, clim, labels)
    feats = [f for f in C.STAGE_A_FEATURES if f in sa.columns and sa[f].notna().any()]
    print(f"  {len(sa):,} rows, {len(feats)}/{len(C.STAGE_A_FEATURES)} features usable")
    missing = [f for f in C.STAGE_A_FEATURES if f not in feats]
    if missing:
        print(f"  unavailable: {missing}")
    sa.to_parquet(PROCESSED / "regime_inputs.parquet", index=False)

    print("\n[3/5] training Stage A ...")
    tr = splits.filter_to_split(sa, "train")
    va = splits.filter_to_split(sa, "validation")
    print(f"  train {len(tr)} rows   validation {len(va)} rows")
    clf = RegimeClassifier(features=feats).fit(tr, va, verbose=True)

    print("\n[4/5] validation ...")
    rep = clf.report(va)
    print(format_report(rep))
    canary = clf.leakage_canary(va)
    print("\n  leakage canary (spec 5.7):")
    for lead, acc in canary["accuracy_by_lead"].items():
        print(f"    lead {lead}: {acc:.3f}  {'#' * int(acc * 40)}")
    print(f"    {canary['message']}")

    print("\n[5/5] broadcasting probabilities to districts ...")
    for s in args.seasons:
        path = PROCESSED / f"features_{s}.parquet"
        if not path.exists():
            print(f"    skip {s}: no feature table")
            continue
        df = pd.read_parquet(path)
        part = sa[sa["season"] == s]
        if part.empty:
            print(f"    skip {s}: no Stage-A rows")
            continue
        pf = clf.predict_frame(part)
        pf["valid_date"] = part["valid_date"].to_numpy()
        pf["lead_day"] = part["lead_day"].to_numpy()
        drop = [c for c in C.REGIME_PROB_COLS + ["regime_entropy", "regime_argmax"]
                if c in df.columns]
        df = df.drop(columns=drop).merge(pf, on=["valid_date", "lead_day"], how="left")
        if "p_LPS_trend" not in df or df["p_LPS_trend"].isna().all():
            df = df.sort_values(["district_id", "valid_date", "lead_day"])
            df["p_LPS_trend"] = df.groupby(["district_id", "valid_date"])["p_LPS"].diff(-1)
            df["p_LPS_trend"] = df["p_LPS_trend"].fillna(0.0)
        cols = [f.name for f in C.features_schema()]
        for c in cols:
            if c not in df:
                df[c] = np.nan
        df = df[cols]
        C.validate(df)
        df.to_parquet(path, index=False)
        cov = df[C.REGIME_PROB_COLS].notna().all(axis=1).mean()
        print(f"    features_{s}: {len(df):,} rows, regime coverage {cov:.1%}")

    if args.out:
        out = Path(args.out)
        clf.save(out)
        print(f"\n  frozen to {out}")

    print("\nPHASE 3: " + ("passed" if canary["passed"] else "canary FAILED -- see above"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
