"""FastAPI service (spec 14).

Base /api/v1. Every endpoint reads precomputed Parquet through DuckDB; the
only live computation is the cost-loss recolouring (spec 8).

Errors are `{"error": {"code", "message"}}` as spec 14 requires, and they say
what to DO rather than only what went wrong -- a 503 that reads "run
scripts/build_predictions.py" is worth more than one that reads "not found".

Run:
    uvicorn raahat.api.main:app --reload --port 8000
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from raahat import contract as C
from raahat.api import db
from raahat.api.db import DataUnavailable
from raahat.decide import colour as CO, cost_loss as CLoss
from raahat.explain import shap_drivers as SH
from raahat.models import analogs as AN

app = FastAPI(
    title="RAAHAT",
    description="Regime-aware post-processing of district rainfall forecasts",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

API = "/api/v1"

ATTRIBUTION = [
    "Weather data by Open-Meteo.com (CC BY 4.0)",
    "Observed rainfall: India Meteorological Department, Pune",
    "District boundaries: Datameet (ODbL), Census 2011",
]


def fail(code: str, message: str, status: int = 400):
    raise HTTPException(status_code=status,
                        detail={"error": {"code": code, "message": message}})


@app.exception_handler(DataUnavailable)
async def _rebuilding(_, exc: DataUnavailable):
    return JSONResponse(status_code=503, content={
        "error": {"code": "rebuilding", "message": str(exc)}})


@app.exception_handler(HTTPException)
async def _handler(_, exc: HTTPException):
    detail = exc.detail
    if not (isinstance(detail, dict) and "error" in detail):
        detail = {"error": {"code": "http_error", "message": str(detail)}}
    return JSONResponse(status_code=exc.status_code, content=detail)


def _clean(obj: Any) -> Any:
    """NaN is not valid JSON. Nulls are, and mean the same thing here."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)) or hasattr(obj, "isoformat"):
        return str(obj)
    if obj is pd.NaT or (obj is not None and obj is pd.NA):
        return None
    return obj


def _alpha(value) -> float:
    """Resolve an alpha that may arrive as a number, None, or -- when one
    handler calls another directly -- FastAPI's unresolved Query default."""
    return float(value) if isinstance(value, (int, float)) else CLoss.ALPHA_DEFAULT


def _require_data():
    if not db.available()["ready"]:
        fail("no_data",
             "No predictions on disk. Run: python scripts/build_predictions.py",
             503)


# ----------------------------------------------------------------- meta ---


@app.get(f"{API}/health")
def health():
    avail = db.available()
    lo, hi = db.data_range() if avail["ready"] else (None, None)
    return _clean({
        "status": "ok" if avail["ready"] else "no_data",
        "model_version": "raahat-level1-4class",
        "data_through": hi,
        "data_from": lo,
        "mode": "replay",
        "available": avail,
        "attribution": ATTRIBUTION,
    })


@app.get(f"{API}/districts")
def districts():
    try:
        return db.districts_geojson()
    except FileNotFoundError:
        fail("no_geometry", "Run: python scripts/build_districts.py", 503)


@app.get(f"{API}/dates")
def dates(lead: int = Query(3, ge=1, le=5)):
    _require_data()
    return {"lead": lead, "dates": db.distinct_dates(lead)}


@app.get(f"{API}/events")
def events():
    """Curated replay events (spec 10.1). Only those inside our archive."""
    lo, hi = db.data_range()
    catalogue = [
        {"event_id": "gujarat_2024_08", "title": "Gujarat deep depression",
         "date": "2024-08-29", "regime": "LPS", "zone": "C", "default_lead": 3,
         "description": "Closed low at 23N 70E, 994.5 hPa. Detected by the "
                        "MSLP detector with 17.5 mm of rain at the centre."},
        {"event_id": "bay_head_2024_09", "title": "Bay-head depression",
         "date": "2024-09-15", "regime": "LPS", "zone": "C", "default_lead": 3,
         "description": "Strongest system of JJAS 2024: 10.3 hPa deficit at "
                        "23N 88E."},
        {"event_id": "odisha_2024_09", "title": "Odisha coast genesis",
         "date": "2024-09-09", "regime": "LPS", "zone": "C", "default_lead": 3,
         "description": "Textbook Bay-head genesis at 19N 86E."},
    ]
    live = [e for e in catalogue if lo and lo <= e["date"] <= hi]
    return {"events": live, "note": "events outside the cached archive are hidden"}


# ------------------------------------------------------------- forecast ---


@app.get(f"{API}/forecast")
def forecast(
    date: str,
    lead: int = Query(3, ge=1, le=5),
    layer: str = Query("corrected"),
    alpha: float | None = Query(None, ge=0.01, le=0.99),
):
    _require_data()
    try:
        df = db.forecast_layer(date, lead, layer, _alpha(alpha))
    except KeyError as e:
        fail("bad_layer", str(e))
    if df.empty:
        fail("not_found", f"No forecast for {date} at lead {lead}", 404)
    return _clean({
        "date": date, "lead": lead, "layer": layer,
        "alpha": _alpha(alpha),
        "model_version": "raahat-level1-4class", "mode": "replay",
        "values": df[["district_id", "value", "colour_code", "regime_argmax"]]
        .to_dict("records"),
    })


@app.get(f"{API}/district/{{district_id}}")
def district(
    district_id: int,
    date: str,
    lead: int = Query(3, ge=1, le=5),
    alpha: float | None = Query(None, ge=0.01, le=0.99),
):
    _require_data()
    row = db.district_detail(district_id, date, lead)
    if not row:
        fail("not_found", f"No forecast for district {district_id} on {date}", 404)

    a = _alpha(alpha)
    names = db.district_names().set_index("district_id")
    meta = names.loc[district_id].to_dict() if district_id in names.index else {}
    s = pd.Series(row)

    proposed = CO.colour_from_probabilities(
        [row.get("p_gt_64_5") or 0], [row.get("p_gt_115_6") or 0],
        [row.get("p_gt_204_5") or 0], alpha=a)
    guarded, applied = CO.apply_safety_guard(
        proposed, [row.get("raw_mm") or 0], [row.get("p_gt_64_5") or 0], alpha=a)

    drivers = [
        {"feature": row.get(f"shap_top{i}_feature"),
         "label": SH.label(str(row.get(f"shap_top{i}_feature"))),
         "contribution": row.get(f"shap_top{i}_value"),
         "direction": "raised" if (row.get(f"shap_top{i}_value") or 0) >= 0 else "lowered"}
        for i in (1, 2, 3) if row.get(f"shap_top{i}_feature") is not None
    ]
    analogs = [
        {"date": row.get(f"analog_date_{i}"), "observed_mm": row.get(f"analog_obs_{i}")}
        for i in (1, 2) if row.get(f"analog_date_{i}") is not None
    ]

    return _clean({
        "district": {"district_id": district_id, **meta},
        "date": date, "lead": lead,
        "raw": {"multi_model_mean": row.get("raw_mm")},
        "corrected": {"p10": row.get("corrected_p10"),
                      "median": row.get("corrected_median"),
                      "p90": row.get("corrected_p90")},
        "exceedance": {"p64_5": row.get("p_gt_64_5"),
                       "p115_6": row.get("p_gt_115_6"),
                       "p204_5": row.get("p_gt_204_5")},
        "regime": {"probs": {c: row.get(c) for c in C.REGIME_PROB_COLS},
                   "argmax": row.get("regime_argmax")},
        "explanation": {"drivers": drivers, "narrative": SH.narrative(s)},
        "analogs": analogs,
        "analog_narrative": AN.narrative(s, meta.get("district_name", "this district")),
        "warning": {
            "suggested_colour": CO.NAMES[int(guarded[0])],
            "colour_code": int(guarded[0]),
            "alpha_used": a,
            "raw_colour": CO.NAMES[int(CO.colour_from_value([row.get("raw_mm") or 0])[0])],
            "guard_applied": bool(applied[0]),
        },
    })


@app.get(f"{API}/regime")
def regime(date: str, lead_max: int = Query(5, ge=1, le=5)):
    _require_data()
    df = db.regime_timeline(date, lead_max)
    if df.empty:
        fail("not_found", f"No regime data for {date}", 404)
    return _clean({
        "date": date,
        "timeline": [
            {"lead": int(r.lead_day),
             "probs": {c: getattr(r, c) for c in C.REGIME_PROB_COLS},
             "entropy": r.regime_entropy}
            for r in df.itertuples()
        ],
        "note": "WD and EASTERLY_COASTAL are not labellable from the available "
                "data and are always zero; see docs/DATA_NOTES.md item 16",
    })


# ------------------------------------------------------------- decision ---


class DecisionRequest(BaseModel):
    date: str
    lead: int = 3
    alpha: float = CLoss.ALPHA_DEFAULT


@app.post(f"{API}/decision")
def decision(req: DecisionRequest):
    """Recolour every district at a new alpha. Pure arithmetic, <100 ms."""
    _require_data()
    df = db.forecast_layer(req.date, req.lead, "colour", req.alpha)
    if df.empty:
        fail("not_found", f"No forecast for {req.date} at lead {req.lead}", 404)
    counts = df.colour_code.value_counts().to_dict()
    return _clean({
        "date": req.date, "lead": req.lead,
        "alpha": CLoss.clamp_alpha(req.alpha),
        "counts": {CO.NAMES[int(k)]: int(v) for k, v in counts.items()},
        "values": df[["district_id", "colour_code"]].to_dict("records"),
    })


@app.get(f"{API}/cost-loss")
def cost_loss_curve(date: str, lead: int = Query(3, ge=1, le=5)):
    """The alpha slider's own evidence: how the trade-off moves."""
    _require_data()
    df = db.forecast_layer(date, lead, "corrected")
    if df.empty:
        fail("not_found", f"No forecast for {date}", 404)
    rows = []
    for a in np.round(np.arange(0.05, 0.61, 0.05), 2):
        c = CO.colour_from_probabilities(df.p_gt_64_5, df.p_gt_115_6,
                                         df.p_gt_204_5, alpha=float(a))
        rows.append({"alpha": float(a), "warned": int((c > 0).sum()),
                     "warn_rate": float((c > 0).mean())})
    return _clean({"date": date, "lead": lead, "curve": rows,
                   "default_alpha": CLoss.ALPHA_DEFAULT,
                   "note": "measured optimum 0.10-0.15; spec 6.4 suggested 0.30 "
                           "which scored worse than raw"})


@app.get(f"{API}/bulletin/{{district_id}}")
def bulletin(district_id: int, date: str, lead: int = Query(3, ge=1, le=5)):
    """Templated text. Spec 18: an LLM may rephrase, never decide content."""
    d = district(district_id, date, lead, CLoss.ALPHA_DEFAULT)
    name = d["district"].get("district_name", f"District {district_id}")
    w = d["warning"]["suggested_colour"]
    med = d["corrected"]["median"]
    p64 = d["exceedance"]["p64_5"]
    action = {v["name"]: v["action"] for v in C.COLOURS.values()}.get(w, "")
    text = (f"{name}: forecast rainfall for {date} is about {med:.0f} mm "
            f"(day {lead}). Probability of heavy rain above 64.5 mm is "
            f"{p64:.0%}. Suggested warning level: {w} — {action}.")
    return _clean({"district_id": district_id, "date": date, "lead": lead,
                   "colour": w, "text_en": text,
                   "caveat": "District areal mean, not 'isolated places within'. "
                             "Complementary to IMD warnings, not a replacement."})


@app.get(f"{API}/verification")
def verification():
    """The evidence screen (spec 9.4 screen 4).

    Serves the frozen held-out results verbatim from results/final/. Nothing is
    recomputed here -- these are the numbers in the MANIFEST, and the UI must
    not be able to drift from them.
    """
    import json
    from pathlib import Path

    final = C.DATA_DIR.parent / "results" / "final"
    if not (final / "scorecard_test.csv").exists():
        fail("no_results", "Run scripts/final_eval.py first.", 503)

    def csv(name):
        f = final / name
        return pd.read_csv(f).to_dict("records") if f.exists() else []

    manifest = {}
    meta = json.loads((final / "reference_meta.json").read_text(encoding="utf-8"))
    mf = final / "MANIFEST.json"
    if mf.exists():
        m = json.loads(mf.read_text(encoding="utf-8"))
        manifest = {
            "run_id": m.get("run_id"),
            "created_utc": m.get("created_utc"),
            "splits": m.get("splits"),
            "alpha": m.get("alpha"),
            "frozen_before_test_opened": m.get("frozen_before_test_opened"),
        }
    return _clean({
        "scorecard": csv("scorecard_test.csv"),
        "per_regime": csv("per_regime_csi_test.csv"),
        "reliability": csv("reliability_test.csv"),
        "manifest": manifest,
        "caveats": meta["caveats"],
    })


@app.get(f"{API}/atlas")
def atlas():
    """Regime Error Atlas (spec 3 USP 1, screen 3).

    How wrong the raw forecast is per weather situation and region, and how
    much of that the correction removes.
    """
    final = C.DATA_DIR.parent / "results" / "final"
    f = final / "regime_error_atlas_test.csv"
    if not f.exists():
        fail("no_results", "Run scripts/final_eval.py first.", 503)
    df = pd.read_csv(f)
    import json
    meta = json.loads((final / "reference_meta.json").read_text(encoding="utf-8"))
    return _clean({
        "cells": df.to_dict("records"),
        "min_events": meta["atlas_min_events"],
        "note": meta["atlas_note"],
    })


@app.get(f"{API}/about")
def about():
    """Data provenance (spec 9.4 screen 5). Pre-empts half the judge questions."""
    import json

    source = C.DATA_DIR / "static" / "about.json"
    return _clean(json.loads(source.read_text(encoding="utf-8")))


@app.get(f"{API}/imd-benchmark")
def imd_benchmark():
    fail("not_available",
         "IMD API registration was never completed, so no warning archive "
         "exists to benchmark against (spec 4.8).", 501)
