"""Train Stage A and run its acceptance tests (spec 16, phase 3).

Phase 3 is green when:
  * a confusion matrix with support counts exists
  * accuracy degrades with lead time (day5/day1 ratio < 0.95)

Usage:
    python -m raahat.train.train_regime
    python -m raahat.train.train_regime --out models/frozen/dev
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from raahat import contract as C
from raahat.models.regime_clf import RegimeClassifier, format_report
from raahat.train import splits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="directory to freeze the model into")
    ap.add_argument("--no-calibration", action="store_true")
    args = ap.parse_args(argv)

    print(f"splits: {splits.describe()}\n")

    train = splits.load_regime_split("train")
    valid = splits.load_regime_split("validation")
    print(f"train {len(train):,} rows  |  validation {len(valid):,} rows")
    print(f"features: {len(C.STAGE_A_FEATURES)}\n")

    clf = RegimeClassifier()
    clf.fit(train, None if args.no_calibration else valid)

    print("\n-- validation season ------------------------------------------")
    rep = clf.report(valid)
    print(format_report(rep))

    print("\n  confusion (rows = truth, cols = predicted)")
    width = max(len(r) for r in C.REGIMES)
    header = " " * (width + 4) + "".join(f"{r[:6]:>8}" for r in C.REGIMES)
    print("  " + header)
    for i, row in enumerate(rep["confusion"]):
        print(f"  {C.REGIMES[i]:<{width + 2}}" + "".join(f"{v:>8,}" for v in row))

    print("\n-- leakage canary (spec 5.7) ----------------------------------")
    canary = clf.leakage_canary(valid)
    for lead, acc in canary["accuracy_by_lead"].items():
        bar = "#" * int(acc * 40)
        print(f"  lead {lead}:  {acc:.3f}  {bar}")
    print(f"\n  {canary['message']}")

    if args.out:
        out = Path(args.out)
        clf.save(out)
        (out / "regime_report.json").write_text(
            json.dumps({"validation": rep, "canary": canary}, indent=2), encoding="utf-8"
        )
        print(f"\n  frozen to {out}")

    if not canary["passed"]:
        print("\nPHASE 3 ACCEPTANCE: FAILED (leakage canary)")
        return 1
    print("\nPHASE 3 ACCEPTANCE: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
