"""The data contract (spec 13.2), as code.

This module is the single source of truth for column names, dtypes and feature
groupings. Nothing anywhere else in the codebase may hard-code a feature name.
Spec 17: five of six workstreams are blocked until this exists, so it is
deliberately dependency-light -- yaml and pyarrow only.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

# --------------------------------------------------------------- paths ---

ROOT = Path(os.environ.get("RAAHAT_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"


@lru_cache(maxsize=None)
def load_config(name: str) -> dict:
    """Load config/<name>.yaml. Cached -- configs are immutable at runtime."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"missing config: {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# -------------------------------------------------------------- regimes ---

_regimes_cfg = load_config("regimes")

#: Canonical regime order. This order is load-bearing: probability vectors,
#: model output columns and the soft-label list all follow it. Never reorder.
REGIMES: list[str] = [r["name"] for r in _regimes_cfg["regimes"]]
N_REGIMES = len(REGIMES)

#: Column names carrying the regime probabilities. WEAK_TRANSITION is stored
#: as `p_WEAK` -- the spec's 13.2 column list uses the short form.
_PROB_SUFFIX = {"WEAK_TRANSITION": "WEAK"}
REGIME_PROB_COLS: list[str] = [f"p_{_PROB_SUFFIX.get(r, r)}" for r in REGIMES]

THRESHOLDS: dict[str, float] = _regimes_cfg["thresholds"]
THRESHOLD_VALUES: list[float] = [
    THRESHOLDS["heavy"],
    THRESHOLDS["very_heavy"],
    THRESHOLDS["extremely_heavy"],
]
COLOURS: dict[int, dict] = _regimes_cfg["colours"]

# ------------------------------------------------------------- features ---

_features_cfg = load_config("features")

STAGE_A_BOXES: dict[str, dict] = _features_cfg["stage_a"]["boxes"]
STAGE_A_FEATURES: list[str] = list(_features_cfg["stage_a"]["features"])

_sb = _features_cfg["stage_b"]
NWP_MODELS: list[str] = list(_sb["nwp_models"])


def _per_model_cols() -> list[str]:
    cols: list[str] = []
    for m in NWP_MODELS:
        cols.append(f"pr_{m}")
        cols.extend(f"pr_{m}_{sfx}" for sfx in _sb["per_model_suffixes"])
    return cols


STAGE_B_NWP: list[str] = _per_model_cols()
STAGE_B_MULTIMODEL: list[str] = list(_sb["multi_model"])
STAGE_B_ENSEMBLE: list[str] = list(_sb["ensemble"])  # nullable, Level 2 only
STAGE_B_REGIME: list[str] = list(_sb["regime"])
STAGE_B_SYNOPTIC: list[str] = list(_sb["synoptic"])
STAGE_B_TERRAIN: list[str] = list(_sb["terrain"])
STAGE_B_CALENDAR: list[str] = list(_sb["calendar"])
STAGE_B_CATEGORICAL: list[str] = list(_sb["categorical"])

#: Every numeric Stage-B feature, in contract order.
STAGE_B_FEATURES: list[str] = (
    STAGE_B_NWP
    + STAGE_B_MULTIMODEL
    + STAGE_B_ENSEMBLE
    + STAGE_B_REGIME
    + STAGE_B_SYNOPTIC
    + STAGE_B_TERRAIN
    + STAGE_B_CALENDAR
)

#: Stage B with the regime block removed -- ablation 4 of spec 6.3.
STAGE_B_FEATURES_NO_REGIME: list[str] = [
    c for c in STAGE_B_FEATURES if c not in set(STAGE_B_REGIME)
]

MONOTONE_INCREASING: list[str] = list(_features_cfg["monotone_increasing"])

# ---------------------------------------------------------------- keys ---

KEY_COLS = ["district_id", "valid_date", "lead_day"]
META_COLS = ["init_datetime_utc", "season", "zone_code", "subdivision"]
TARGET_COL = "obs_rain_mm"

#: TRAINING ONLY. assert_no_labels() enforces their absence at inference.
LABEL_COLS = ["label_regime", "label_regime_soft"]

# -------------------------------------------------------------- schemas ---

class _LazyPyarrow:
    """pyarrow, imported on first attribute access.

    Every schema below is a BUILD-TIME artefact: the ingest and training steps
    validate against them, and the API never calls one. A plain top-level
    import still cost 84 MB of pyarrow in any bundle that imports this module,
    which was on its own most of the difference between fitting inside a
    serverless function size limit and not fitting. Attribute access is
    identical, so `pa.field(...)` below is unchanged.
    """

    _mod = None

    def __getattr__(self, name: str):
        if _LazyPyarrow._mod is None:
            import pyarrow
            _LazyPyarrow._mod = pyarrow
        return getattr(_LazyPyarrow._mod, name)


pa = _LazyPyarrow()


def _dedupe(fields: list[pa.Field]) -> pa.Schema:
    """Preserve first-seen order, drop repeats.

    regime_entropy is declared both in the regime feature block and in the
    spec's own 13.2 column list, so it would otherwise appear twice.
    """
    seen: set[str] = set()
    out: list[pa.Field] = []
    for f in fields:
        if f.name in seen:
            continue
        seen.add(f.name)
        out.append(f)
    return pa.schema(out)


def features_schema() -> pa.Schema:
    """Schema of data/processed/features_{season}.parquet (spec 13.2)."""
    fields = [
        pa.field("district_id", pa.int32(), nullable=False),
        pa.field("valid_date", pa.date32(), nullable=False),
        pa.field("lead_day", pa.int8(), nullable=False),
        pa.field("init_datetime_utc", pa.timestamp("s", tz="UTC"), nullable=False),
        pa.field("season", pa.int16(), nullable=False),
        pa.field("zone_code", pa.string(), nullable=False),
        pa.field("subdivision", pa.string(), nullable=True),
        pa.field(TARGET_COL, pa.float32(), nullable=True),
    ]
    fields += [pa.field(c, pa.float32(), nullable=True) for c in STAGE_B_FEATURES]
    fields += [
        pa.field("regime_entropy", pa.float32(), nullable=True),
        pa.field("regime_argmax", pa.string(), nullable=True),
        pa.field("label_regime", pa.string(), nullable=True),
        pa.field("label_regime_soft", pa.list_(pa.float32(), N_REGIMES), nullable=True),
    ]
    return _dedupe(fields)


def regime_input_schema() -> pa.Schema:
    """Schema of the Stage-A table: one row per (valid_date, lead_day).

    Clarifies spec 5.3/13.1 -- see the note at the top of config/features.yaml.
    The regime is a national synoptic state, so the classifier trains on one
    row per day per lead, not one row per district-day.
    """
    fields = [
        pa.field("valid_date", pa.date32(), nullable=False),
        pa.field("lead_day", pa.int8(), nullable=False),
        pa.field("init_datetime_utc", pa.timestamp("s", tz="UTC"), nullable=False),
        pa.field("season", pa.int16(), nullable=False),
    ]
    fields += [pa.field(c, pa.float32(), nullable=True) for c in STAGE_A_FEATURES]
    fields += [
        pa.field("label_regime", pa.string(), nullable=True),
        pa.field("label_regime_soft", pa.list_(pa.float32(), N_REGIMES), nullable=True),
    ]
    return _dedupe(fields)


def predictions_schema() -> pa.Schema:
    """Schema of data/processed/predictions_{season}.parquet (spec 13.2)."""
    fields = [
        pa.field("district_id", pa.int32(), nullable=False),
        pa.field("valid_date", pa.date32(), nullable=False),
        pa.field("lead_day", pa.int8(), nullable=False),
        pa.field("init_datetime_utc", pa.timestamp("s", tz="UTC"), nullable=False),
        pa.field("raw_mm", pa.float32(), nullable=True),
        pa.field("corrected_p10", pa.float32(), nullable=True),
        pa.field("corrected_median", pa.float32(), nullable=True),
        pa.field("corrected_p90", pa.float32(), nullable=True),
        pa.field("p_gt_64_5", pa.float32(), nullable=True),
        pa.field("p_gt_115_6", pa.float32(), nullable=True),
        pa.field("p_gt_204_5", pa.float32(), nullable=True),
        pa.field("csgd_mu", pa.float32(), nullable=True),
        pa.field("csgd_sigma", pa.float32(), nullable=True),
        pa.field("csgd_delta", pa.float32(), nullable=True),
    ]
    fields += [pa.field(c, pa.float32(), nullable=True) for c in REGIME_PROB_COLS]
    fields += [
        pa.field("regime_argmax", pa.string(), nullable=True),
        pa.field("colour_code", pa.int8(), nullable=True),
        pa.field("colour_alpha_used", pa.float32(), nullable=True),
    ]
    for i in (1, 2, 3):
        fields.append(pa.field(f"shap_top{i}_feature", pa.string(), nullable=True))
        fields.append(pa.field(f"shap_top{i}_value", pa.float32(), nullable=True))
    for i in (1, 2):
        fields.append(pa.field(f"analog_date_{i}", pa.date32(), nullable=True))
        fields.append(pa.field(f"analog_obs_{i}", pa.float32(), nullable=True))
    fields += [
        pa.field("model_version", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
    ]
    return _dedupe(fields)


# ----------------------------------------------------------- validation ---


class ContractError(ValueError):
    """Raised when a table violates the frozen data contract."""


def validate(df, schema: pa.Schema | None = None, *, allow_extra: bool = False) -> None:
    """Check a pandas DataFrame against the contract. Raises ContractError."""
    schema = schema if schema is not None else features_schema()
    expected = [f.name for f in schema]
    cols = list(df.columns)

    missing = [c for c in expected if c not in cols]
    if missing:
        raise ContractError(f"{len(missing)} column(s) missing: {missing[:10]}")

    if not allow_extra:
        extra = [c for c in cols if c not in set(expected)]
        if extra:
            raise ContractError(f"{len(extra)} column(s) not in contract: {extra[:10]}")

    for key in KEY_COLS:
        if key in cols and df[key].isna().any():
            raise ContractError(f"key column '{key}' contains nulls")

    if "lead_day" in cols and len(df):
        bad = sorted(set(df["lead_day"].dropna().unique()) - set(range(1, 6)))
        if bad:
            raise ContractError(f"lead_day outside 1..5: {bad}")

    probs = [c for c in REGIME_PROB_COLS if c in cols]
    if len(probs) == N_REGIMES and len(df):
        # Rows where every probability is null are "Stage A has not run yet",
        # which is a legitimate state for a freshly built feature table. Only
        # rows carrying actual values are checked -- the point is to catch a
        # vector that is populated WRONGLY, not one that is absent.
        populated = df[probs].notna().any(axis=1)
        sums = df.loc[populated, probs].sum(axis=1)
        if len(sums) and not ((sums - 1.0).abs() < 1e-3).all():
            worst = float((sums - 1.0).abs().max())
            raise ContractError(
                f"regime probabilities do not sum to 1 (max error {worst:.4g}); "
                f"{int((~((sums - 1.0).abs() < 1e-3)).sum())} of {len(sums)} rows bad"
            )


def assert_no_labels(df) -> None:
    """Spec 13.2: label columns must be absent at inference time."""
    present = [c for c in LABEL_COLS if c in df.columns]
    if present:
        raise ContractError(
            f"label column(s) {present} present at inference -- this is a leakage bug"
        )


def summary() -> str:
    return (
        "RAAHAT contract\n"
        f"  regimes         : {N_REGIMES}  {REGIMES}\n"
        f"  prob columns    : {REGIME_PROB_COLS}\n"
        f"  stage A feats   : {len(STAGE_A_FEATURES)}\n"
        f"  stage B feats   : {len(STAGE_B_FEATURES)} "
        f"(+{len(STAGE_B_CATEGORICAL)} categorical)\n"
        f"  no-regime feats : {len(STAGE_B_FEATURES_NO_REGIME)}  (ablation 4)\n"
        f"  features schema : {len(features_schema())} columns\n"
        f"  stage A schema  : {len(regime_input_schema())} columns\n"
        f"  predictions     : {len(predictions_schema())} columns\n"
        f"  root            : {ROOT}"
    )


if __name__ == "__main__":
    print(summary())
