"""Freeze the district master table. Spec 21, item 2 -- do this before anything.

Writes:
    data/static/districts_master.csv       FROZEN. Never regenerate casually.
    data/static/district_neighbours.parquet first-order adjacency (spec 5.1)
    data/static/districts.geojson           the subset actually used
    data/static/unmatched_names.csv         every zones.yaml name that missed

Spec 4.6: India's district list changes constantly, so the stable key is OURS
(`district_id`), assigned once here and never reassigned. Name harmonisation is
explicit -- nothing is fuzzy-matched silently.

Usage:
    python scripts/build_districts.py            # Level-1 zones only (~190)
    python scripts/build_districts.py --all-india
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raahat import contract as C  # noqa: E402

BOUNDARIES = C.DATA_DIR / "raw" / "boundaries" / "dists11.geojson"
STATIC = C.DATA_DIR / "static"

#: Equal-area projection for India, so area_km2 and centroids are not nonsense.
#: EPSG:7755 is the Asia South Albers Equal Area Conic used for Indian work.
EQUAL_AREA = "EPSG:7755"


def assign_zones(gdf: gpd.GeoDataFrame, zones_cfg: dict) -> tuple[pd.Series, list[dict]]:
    """Map each district to a zone code. Returns (zone_series, unmatched)."""
    zone = pd.Series(pd.NA, index=gdf.index, dtype="object")
    unmatched: list[dict] = []

    # FIRST ASSIGNMENT WINS. zones.yaml lists the spec 2.3 regime-contrast
    # zones (W, C, N) first, and their membership is load-bearing -- the whole
    # Level-1 result is stratified by them. A later broad rule like "Karnataka
    # is peninsular interior" must NOT steal coastal Karnataka out of W, and a
    # "whole of Andhra Pradesh" rule must not steal the Telangana districts out
    # of C. Without this, extending to all-India silently reshaped the zones
    # every published number was computed against.
    for code, spec in zones_cfg["zones"].items():
        for state in spec.get("whole_states", []) or []:
            hit = (gdf["ST_NM"] == state) & zone.isna()
            if not (gdf["ST_NM"] == state).any():
                unmatched.append({"zone": code, "state": state, "district": "<whole state>"})
            zone[hit] = code

        for state, names in (spec.get("districts") or {}).items():
            in_state = gdf["ST_NM"] == state
            if not in_state.any():
                unmatched.append({"zone": code, "state": state, "district": "<state missing>"})
                continue
            available = set(gdf.loc[in_state, "DISTRICT"])
            for name in names:
                if name in available:
                    hit = in_state & (gdf["DISTRICT"] == name) & zone.isna()
                    zone[hit] = code
                else:
                    unmatched.append({"zone": code, "state": state, "district": name})
    return zone, unmatched


def first_order_neighbours(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Districts sharing a boundary. Drives the nbr_* features of spec 5.1.

    `pr_nbr_max` is called out there as the single most important predictor of
    heavy rain, because displacement error dominates at district scale -- so
    this adjacency table is load-bearing, not decoration.
    """
    # `touches` requires boundaries to meet while interiors do NOT overlap, so a
    # single sliver of topological noise makes an interior district look
    # isolated -- Kandhamal and Koraput both did. `intersects` is robust to
    # that, and self-matches are dropped explicitly.
    sindex = gdf.sindex
    rows: list[dict] = []
    geoms = gdf.geometry.to_numpy()
    ids = gdf["district_id"].to_numpy()
    for i, geom in enumerate(geoms):
        for j in sindex.query(geom, predicate="intersects"):
            if i == j:
                continue
            rows.append({"district_id": ids[i], "neighbour_id": ids[j]})
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all-india", action="store_true", help="keep all 641 districts")
    ap.add_argument("--force", action="store_true", help="overwrite a frozen master")
    ap.add_argument("--extend", action="store_true",
                    help="APPEND new districts, keeping every existing district_id")
    args = ap.parse_args(argv)

    out_csv = STATIC / "districts_master.csv"
    existing = pd.read_csv(out_csv) if out_csv.exists() else None

    if existing is not None and not (args.force or args.extend):
        print(f"REFUSING: {out_csv} already exists and is FROZEN (spec 4.6).")
        print("Everything downstream keys off its district_id.")
        print("  --extend  append new districts, keep existing ids (safe)")
        print("  --force   reassign every id, invalidating all cached features,")
        print("            frozen models and the completed evaluation")
        return 1

    if not BOUNDARIES.exists():
        raise SystemExit(f"missing {BOUNDARIES}")

    STATIC.mkdir(parents=True, exist_ok=True)
    zones_cfg = C.load_config("zones")

    print(f"reading {BOUNDARIES.name} ...")
    gdf = gpd.read_file(BOUNDARIES)
    gdf["DISTRICT"] = gdf["DISTRICT"].str.strip()
    gdf["ST_NM"] = gdf["ST_NM"].str.strip()
    print(f"  {len(gdf)} districts, crs {gdf.crs}")

    zone, unmatched = assign_zones(gdf, zones_cfg)
    gdf["zone_code"] = zone

    if unmatched:
        pd.DataFrame(unmatched).to_csv(STATIC / "unmatched_names.csv", index=False)
        print(f"\n  {len(unmatched)} name(s) in zones.yaml matched nothing "
              f"-> data/static/unmatched_names.csv")
        for u in unmatched[:15]:
            print(f"    {u['zone']}  {u['state']:<18} {u['district']}")

    if not args.all_india:
        gdf = gdf[gdf["zone_code"].notna()].copy()
    else:
        gdf["zone_code"] = gdf["zone_code"].fillna("X")
    gdf = gdf.sort_values(["ST_NM", "DISTRICT"]).reset_index(drop=True)

    # Stable key. Spec 4.6 requires it never be reassigned, so --extend matches
    # on (district_name, state) and keeps every id that already exists, giving
    # new ids only to genuinely new rows. Without this, going all-India would
    # silently renumber all 197 existing districts and invalidate every cached
    # feature, every frozen model and the completed held-out evaluation.
    if args.extend and existing is not None:
        key = existing.set_index(["district_name", "state"])["district_id"]
        pairs = list(zip(gdf["DISTRICT"], gdf["ST_NM"]))
        ids, next_id = [], int(existing["district_id"].max()) + 1
        kept = 0
        for p in pairs:
            if p in key.index:
                ids.append(int(key.loc[p])); kept += 1
            else:
                ids.append(next_id); next_id += 1
        gdf["district_id"] = np.array(ids, dtype=np.int32)
        dropped = set(key.index) - set(pairs)
        print(f"\n  --extend: kept {kept} existing ids, added {len(gdf) - kept} new")
        if dropped:
            print(f"  WARNING: {len(dropped)} previously-frozen district(s) are not in "
                  f"this selection and would be LOST: {sorted(dropped)[:5]}")
            print("  refusing -- a frozen id must never disappear")
            return 1
        if gdf["district_id"].duplicated().any():
            print("  refusing -- duplicate district_id after extend")
            return 1
    else:
        gdf["district_id"] = np.arange(1, len(gdf) + 1, dtype=np.int32)

    proj = gdf.to_crs(EQUAL_AREA)
    cent = proj.geometry.centroid.to_crs("EPSG:4326")
    gdf["lon_centroid"] = cent.x.astype(np.float32)
    gdf["lat_centroid"] = cent.y.astype(np.float32)
    gdf["area_km2"] = (proj.geometry.area / 1e6).astype(np.float32)

    master = pd.DataFrame(
        {
            "district_id": gdf["district_id"],
            "district_name": gdf["DISTRICT"],
            "state": gdf["ST_NM"],
            "imd_obj_id": pd.NA,  # filled if/when the IMD API is granted (spec 4.8)
            "lat_centroid": gdf["lat_centroid"],
            "lon_centroid": gdf["lon_centroid"],
            "area_km2": gdf["area_km2"],
            "zone_code": gdf["zone_code"],
            "subdivision": gdf["ST_NM"],  # placeholder until IMD subdivisions land
            "censuscode": gdf["censuscode"],
        }
    )
    master.to_csv(out_csv, index=False)

    gdf[["district_id", "DISTRICT", "ST_NM", "zone_code", "geometry"]].to_file(
        STATIC / "districts.geojson", driver="GeoJSON"
    )

    print("\ncomputing first-order adjacency ...")
    nbrs = first_order_neighbours(gdf)
    nbrs.to_parquet(STATIC / "district_neighbours.parquet", index=False)

    print(f"\nFROZEN: {out_csv}")
    print(f"  districts     {len(master)}")
    print(f"  adjacency     {len(nbrs)} edges, "
          f"{nbrs.groupby('district_id').size().mean():.1f} neighbours on average")
    print(f"  isolated      {len(set(master.district_id) - set(nbrs.district_id))}")
    print("\nby zone:")
    for code, n in master["zone_code"].value_counts().sort_index().items():
        label = zones_cfg["zones"].get(code, {}).get("label", "unassigned")
        print(f"  {code}  {n:>4}  {label}")
    print("\nlat range: "
          f"{master.lat_centroid.min():.2f} .. {master.lat_centroid.max():.2f}   "
          f"lon range: {master.lon_centroid.min():.2f} .. {master.lon_centroid.max():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
