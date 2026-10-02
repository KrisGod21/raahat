"""Fetch district-centroid forecasts and build the observed truth table.

Phase 1 of spec 16. Writes:
    data/interim/district_fcst_{season}.parquet   forecast rainfall per lead
    data/interim/district_obs_{season}.parquet    IMD truth, area-weighted

Spec 4.2 sizes this at ~750 calls, well inside Open-Meteo's 10,000/day free
tier. Every response is cached by URL hash, so re-running costs nothing and
works offline.

Usage:
    python scripts/fetch_forecasts.py --seasons 2024 2025
    python scripts/fetch_forecasts.py --seasons 2024 --offline   # cache only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raahat import contract as C  # noqa: E402
from raahat.ingest import imd_grid, openmeteo  # noqa: E402
from raahat.ingest.http_cache import FetchError, cache_stats  # noqa: E402

INTERIM = C.DATA_DIR / "interim"
STATIC = C.DATA_DIR / "static"


def season_window(season: int) -> tuple[str, str]:
    """JJAS plus a few days of run-up, so lead-5 and 3-day rolling features
    have their history and the first of June is not a partial row."""
    return f"{season}-05-25", f"{season}-10-02"


def build_observations(seasons: list[int]) -> pd.DataFrame:
    """IMD grid -> area-weighted district daily rainfall (spec 5.1)."""
    print("building observed truth from the IMD 0.25 deg grid ...")
    da = imd_grid.load(seasons)
    geo = gpd.read_file(STATIC / "districts.geojson")
    obs = imd_grid.district_means_areal(da, geo)
    obs["valid_date"] = pd.to_datetime(obs["valid_date"]).dt.date
    return obs


def fetch_all(
    master: pd.DataFrame,
    seasons: list[int],
    models: tuple[str, ...],
    *,
    offline: bool = False,
    pause: float = 0.12,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    failures: list[dict] = []
    total = len(master) * len(seasons) * len(models)
    done = 0
    t0 = time.time()

    for season in seasons:
        start, end = season_window(season)
        for model in models:
            for _, row in master.iterrows():
                done += 1
                try:
                    df = openmeteo.fetch_point_forecasts(
                        row.lat_centroid, row.lon_centroid, start, end,
                        model=model, offline=offline,
                    )
                except (FetchError, KeyError) as e:
                    failures.append(
                        {"district_id": int(row.district_id), "season": season,
                         "model": model, "error": str(e)[:160]}
                    )
                    continue
                df["district_id"] = np.int32(row.district_id)
                df["season"] = np.int16(season)
                df["model"] = model
                frames.append(df)
                if done % 100 == 0 or done == total:
                    rate = done / max(time.time() - t0, 1e-9)
                    eta = (total - done) / max(rate, 1e-9)
                    print(f"  {done:>5}/{total}  {rate:5.1f}/s  eta {eta/60:5.1f} min  "
                          f"failures {len(failures)}")
                if not offline:
                    time.sleep(pause)

    if failures:
        INTERIM.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(INTERIM / "fetch_failures.csv", index=False)
        print(f"\n  {len(failures)} fetch failure(s) -> data/interim/fetch_failures.csv")
    if not frames:
        raise SystemExit("no forecasts fetched at all -- check connectivity and the log")
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025])
    ap.add_argument("--models", nargs="+", default=list(openmeteo.DEFAULT_MODELS))
    ap.add_argument("--offline", action="store_true", help="use the cache only")
    ap.add_argument("--limit", type=int, default=None, help="first N districts (smoke test)")
    ap.add_argument("--pause", type=float, default=0.12,
                    help="seconds between calls. Open-Meteo rate-limits hard on "
                         "long runs; 1.5-2.0 survives a 2500-call sweep where the "
                         "default 0.12 dies at ~12%% with HTTP 429.")
    args = ap.parse_args(argv)

    INTERIM.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(STATIC / "districts_master.csv")
    if args.limit:
        master = master.head(args.limit)

    print(f"districts {len(master)}  seasons {args.seasons}  models {args.models}")
    print(f"cache before: {cache_stats()}")

    obs = build_observations(args.seasons)
    for season in args.seasons:
        lo, hi = pd.Timestamp(f"{season}-06-01").date(), pd.Timestamp(f"{season}-09-30").date()
        part = obs[(obs.valid_date >= lo) & (obs.valid_date <= hi)]
        part.to_parquet(INTERIM / f"district_obs_{season}.parquet", index=False)
        miss = part["obs_rain_mm"].isna().mean()
        wet = (part["obs_rain_mm"] > 64.5).mean()
        print(f"  obs {season}: {len(part):>7,} rows, missing {miss:.2%}, >64.5mm {wet:.2%}, "
              f"max {part['obs_rain_mm'].max():.1f} mm")

    print(f"\nfetching forecasts ({len(master) * len(args.seasons) * len(args.models)} calls) ...")
    fc = fetch_all(master, args.seasons, tuple(args.models), offline=args.offline,
                   pause=args.pause)

    for season in args.seasons:
        lo, hi = pd.Timestamp(f"{season}-06-01").date(), pd.Timestamp(f"{season}-09-30").date()
        part = fc[(fc.season == season) & (fc.valid_date >= lo) & (fc.valid_date <= hi)]
        part.to_parquet(INTERIM / f"district_fcst_{season}.parquet", index=False)
        print(f"  fcst {season}: {len(part):>8,} rows, "
              f"{part.district_id.nunique()} districts, leads {sorted(part.lead_day.unique())}")

    print(f"\ncache after: {cache_stats()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
