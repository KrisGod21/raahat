"""DuckDB over Parquet (spec 8).

Spec 8 justifies this choice and it is worth keeping the reasoning next to the
code: the whole dataset is a few hundred MB of immutable, append-only
analytical tables. DuckDB queries the Parquet directly with no server, no
setup, sub-100 ms responses, and -- the operative property -- it CANNOT FAIL TO
START during a demo. PostGIS is the right answer for a production deployment
with concurrent writers; saying so when asked is a better answer than having
chosen the impressive tool.

Everything served here is precomputed. No model inference happens in a request
handler; the only live computation is the cost-loss recolouring, which is
arithmetic on stored probabilities.
"""

from __future__ import annotations

import functools
import os
import threading
from pathlib import Path

import duckdb
import pandas as pd

from raahat import contract as C

PROC = C.DATA_DIR / "processed"
STATIC = C.DATA_DIR / "static"


@functools.lru_cache(maxsize=1)
def _root() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute("SET enable_progress_bar=false")
    # A serverless filesystem is read-only apart from /tmp. DuckDB only spills
    # to disk if a query exceeds memory, which this workload should never do --
    # but if it ever did, the failure would be an opaque permission error at
    # request time rather than anything diagnosable.
    if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        con.execute("SET temp_directory='/tmp'")
    return con


#: A DuckDB connection is NOT safe to execute on from several threads at once.
#: Uvicorn runs sync handlers in a threadpool, so the browser firing three
#: requests in parallel had them stepping on each other's results -- one
#: thread's execute() clobbering another's pending fetch. The symptom was
#: maddening: every endpoint worked perfectly when called one at a time with
#: curl, and returned sporadic 500s ("DataFrame has no attribute ...",
#: "NoneType has no attribute ...") the moment the page loaded them together.
#:
#: The documented fix is one cursor per thread. They share the same underlying
#: database, so nothing is duplicated or re-read.
_local = threading.local()


def connect() -> duckdb.DuckDBPyConnection:
    """A cursor private to the calling thread, over the shared database."""
    cur = getattr(_local, "cur", None)
    if cur is None:
        cur = _root().cursor()
        _local.cur = cur
    return cur


def predictions_glob() -> str:
    return str(PROC / "predictions_*.parquet").replace("\\", "/")


def features_glob() -> str:
    return str(PROC / "features_*.parquet").replace("\\", "/")


def available() -> dict:
    """What is actually on disk. Drives /health and fails loudly if empty."""
    preds = sorted(PROC.glob("predictions_*.parquet"))
    feats = sorted(PROC.glob("features_*.parquet"))
    return {
        "predictions": [p.name for p in preds],
        "features": [p.name for p in feats],
        # EITHER file is enough to serve the map. districts_geojson() prefers
        # the simplified one and a deployment ships only that, so checking the
        # 28 MB original alone made /health report the map as unavailable on
        # exactly the builds where it was working fine.
        "districts_geojson": any(
            (STATIC / n).exists()
            for n in ("districts_simplified.geojson", "districts.geojson")
        ),
        "ready": bool(preds),
    }


class DataUnavailable(RuntimeError):
    """A read failed because the underlying file is being rewritten."""


def q(sql: str, params: list | None = None) -> pd.DataFrame:
    """Query, translating a mid-rebuild read into a clear, retryable error.

    A batch job rewriting predictions_*.parquet used to surface as an opaque
    500. The writer now renames atomically, so this should not happen -- but a
    reader that loses the race deserves a message saying to retry rather than a
    stack trace.
    """
    try:
        return connect().execute(sql, params or []).fetch_df()
    except Exception as exc:  # duckdb raises several unrelated types here
        msg = str(exc).lower()
        if any(k in msg for k in ("no such file", "not found", "invalid", "corrupt",
                                  "unexpected end", "io error")):
            raise DataUnavailable(
                "Prediction data is being rebuilt right now. Retry in a moment."
            ) from exc
        raise


#: valid_date is stored as a timestamp, so a bare cast gives
#: "2024-06-01 00:00:00" while the UI speaks plain "2024-06-01". Every date
#: comparison and every date we emit goes through these, or the two sides
#: silently never match and the map renders empty.
_DATE_SQL = "CAST(valid_date AS DATE)"


def _as_date(value) -> str:
    return str(pd.Timestamp(value).date())


def data_range() -> tuple[str | None, str | None]:
    if not available()["ready"]:
        return None, None
    df = q(f"SELECT min(valid_date) lo, max(valid_date) hi "
           f"FROM read_parquet('{predictions_glob()}')")
    return _as_date(df.lo.iloc[0]), _as_date(df.hi.iloc[0])


def forecast_layer(date: str, lead: int, layer: str, alpha: float | None = None) -> pd.DataFrame:
    """One lightweight array for choropleth rendering (spec 14, <200 KB).

    `alpha` recolours live from the stored probabilities -- this is the one
    thing the API computes rather than looks up, and it is arithmetic.
    """
    col = {
        "raw": "raw_mm", "corrected": "corrected_median",
        "p64": "p_gt_64_5", "p115": "p_gt_115_6", "p204": "p_gt_204_5",
        "delta": "corrected_median - raw_mm", "colour": "colour_code",
    }.get(layer)
    if col is None:
        raise KeyError(f"unknown layer {layer!r}")

    df = q(
        f"""SELECT district_id, {col} AS value, colour_code, regime_argmax,
                   raw_mm, corrected_median, p_gt_64_5, p_gt_115_6, p_gt_204_5
            FROM read_parquet('{predictions_glob()}')
            WHERE {_DATE_SQL} = CAST(? AS DATE) AND lead_day = ?""",
        [_as_date(date), lead],
    )
    if isinstance(alpha, (int, float)) and len(df):
        from raahat.decide import colour as CO
        df["colour_code"] = CO.colour_from_probabilities(
            df.p_gt_64_5, df.p_gt_115_6, df.p_gt_204_5, alpha=alpha)
        guarded, _ = CO.apply_safety_guard(
            df.colour_code, df.raw_mm, df.p_gt_64_5, alpha=alpha)
        df["colour_code"] = guarded
        if layer == "colour":
            df["value"] = df["colour_code"]
    return df


def district_detail(district_id: int, date: str, lead: int) -> dict:
    df = q(
        f"""SELECT * FROM read_parquet('{predictions_glob()}')
            WHERE district_id = ? AND {_DATE_SQL} = CAST(? AS DATE)
              AND lead_day = ?""",
        [district_id, _as_date(date), lead],
    )
    return {} if df.empty else df.iloc[0].to_dict()


def regime_timeline(date: str, lead_max: int = 5) -> pd.DataFrame:
    """National regime vector by lead -- the stacked-area chart of spec 3."""
    cols = ", ".join(f"avg({c}) AS {c}" for c in C.REGIME_PROB_COLS)
    return q(
        f"""SELECT lead_day, {cols}, avg(regime_entropy) AS regime_entropy
            FROM read_parquet('{features_glob()}')
            WHERE {_DATE_SQL} = CAST(? AS DATE) AND lead_day <= ?
            GROUP BY lead_day ORDER BY lead_day""",
        [_as_date(date), lead_max],
    )


def districts_geojson() -> dict:
    """Prefer the simplified geometry (spec 4.6: under 1.5 MB for the web).

    The full-resolution file is 28 MB. Served raw it exceeds the frontend's
    3-second timeout and the map never draws -- which is the exact failure
    spec 4.6 exists to prevent. Full resolution stays on disk for the
    area-weighted district aggregation, which needs it.
    """
    import json
    for name in ("districts_simplified.geojson", "districts.geojson"):
        path = STATIC / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(STATIC / "districts_simplified.geojson")


def district_names() -> pd.DataFrame:
    return pd.read_csv(STATIC / "districts_master.csv")[
        ["district_id", "district_name", "state", "zone_code",
         "lat_centroid", "lon_centroid"]
    ]


def distinct_dates(lead: int = 3) -> list[str]:
    if not available()["ready"]:
        return []
    df = q(f"""SELECT DISTINCT {_DATE_SQL} AS d
               FROM read_parquet('{predictions_glob()}')
               WHERE lead_day = ? ORDER BY d""", [lead])
    return [_as_date(d) for d in df.d]
