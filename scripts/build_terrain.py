"""Per-district terrain features (spec 4.7, 5.5).

Spec 5.5 calls `upslope_flux` the feature that lets one national model handle
Mahabaleshwar and Cherrapunji without a separate "orographic regime". It is the
dot product of the low-level wind with the district's mean terrain GRADIENT
VECTOR, so a scalar elevation is not enough -- we need the direction the land
rises in, which means sampling inside each polygon and fitting a plane.

Elevation comes from Open-Meteo's keyless elevation API (batched 100 points per
call) rather than ETOPO. Same idea as spec 4.7, no download, and the sampling
is what we need anyway.

COAST DETECTION. `dist_to_coast_km` must mean distance to SEA, not distance to
the national boundary -- otherwise Himachal and Uttarakhand look coastal
because Pakistan and China are nearby, and `coast_onshore_flux` fires in the
Himalaya. So: dissolve all 641 districts, walk the exterior boundary, step a
little way outward, and keep only the points where the ground is at or below
sea level. Those are the real coast.

Usage:
    python scripts/build_terrain.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raahat import contract as C  # noqa: E402
from raahat.ingest.http_cache import get_json  # noqa: E402

STATIC = C.DATA_DIR / "static"
ELEV_URL = "https://api.open-meteo.com/v1/elevation"
BATCH = 100
EQUAL_AREA = "EPSG:7755"

#: Monsoon comes from the south-west, so slopes facing 200-290 deg are windward.
WINDWARD_MIN, WINDWARD_MAX = 200.0, 290.0


def elevations(points: list[tuple[float, float]], *, pause: float = 0.15) -> np.ndarray:
    """Elevation (m) for many lat/lon pairs, batched and cached."""
    out = np.full(len(points), np.nan)
    for i in range(0, len(points), BATCH):
        chunk = points[i : i + BATCH]
        lats = ",".join(f"{a:.4f}" for a, _ in chunk)
        lons = ",".join(f"{o:.4f}" for _, o in chunk)
        url = f"{ELEV_URL}?latitude={lats}&longitude={lons}"
        try:
            d = get_json(url, namespace="elevation")
            vals = d.get("elevation") or []
            out[i : i + len(vals)] = vals
        except Exception as e:  # noqa: BLE001 -- logged by http_cache, keep going
            print(f"    elevation batch {i} failed: {type(e).__name__}")
        time.sleep(pause)
    return out


def sample_polygon(geom, n_target: int = 36) -> list[tuple[float, float]]:
    """A regular grid of points inside the polygon; centroid if it is tiny."""
    minx, miny, maxx, maxy = geom.bounds
    side = max(int(np.sqrt(n_target)), 3)
    xs = np.linspace(minx, maxx, side + 2)[1:-1]
    ys = np.linspace(miny, maxy, side + 2)[1:-1]
    grid = [(float(y), float(x)) for y in ys for x in xs
            if geom.contains(shapely.Point(x, y))]
    if not grid:
        c = geom.centroid
        grid = [(float(c.y), float(c.x))]
    return grid


def plane_gradient(lats, lons, elev) -> tuple[float, float]:
    """Least-squares terrain gradient (m per km east, m per km north).

    Positive gx means the land rises towards the east. Wind dotted with this
    vector is upslope flow, which is the physical trigger for orographic rain.
    """
    ok = np.isfinite(elev)
    if ok.sum() < 4:
        return 0.0, 0.0
    lat0 = float(np.mean(lats[ok]))
    # local equirectangular: degrees -> km
    x = (lons[ok] - np.mean(lons[ok])) * 111.32 * np.cos(np.deg2rad(lat0))
    y = (lats[ok] - lat0) * 110.57
    A = np.column_stack([x, y, np.ones(ok.sum())])
    try:
        coef, *_ = np.linalg.lstsq(A, elev[ok], rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0
    return float(coef[0]), float(coef[1])


def find_coastline(all_districts: gpd.GeoDataFrame, *, step_km: float = 25.0) -> np.ndarray:
    """Points on the national outline where the sea begins. Returns (n,2) lat/lon."""
    print("  dissolving India outline ...")
    land = shapely.union_all(all_districts.geometry.to_numpy())
    boundary = land.boundary
    length = boundary.length  # degrees, adequate for spacing
    n = max(int(length / (step_km / 111.0)), 200)
    pts = [boundary.interpolate(t, normalized=True) for t in np.linspace(0, 1, n)]
    print(f"  probing {len(pts)} boundary points for sea ...")

    # step ~15 km outward along the outward normal, then ask if it is underwater
    cent = land.centroid
    probes = []
    for p in pts:
        dx, dy = p.x - cent.x, p.y - cent.y
        norm = np.hypot(dx, dy) or 1.0
        probes.append((float(p.y + 0.14 * dy / norm), float(p.x + 0.14 * dx / norm)))
    elev = elevations(probes)
    sea = np.array([(p.y, p.x) for p in pts])[np.nan_to_num(elev, nan=999.0) <= 0.0]
    print(f"  {len(sea)} coastal points found (of {len(pts)} boundary points)")
    return sea


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=36)
    args = ap.parse_args(argv)

    geo = gpd.read_file(STATIC / "districts.geojson")
    allg = gpd.read_file(C.DATA_DIR / "raw" / "boundaries" / "dists11.geojson")
    print(f"{len(geo)} districts, {args.samples} sample points each")

    coast = find_coastline(allg)
    coast_xy = gpd.GeoSeries(
        [shapely.Point(lon, lat) for lat, lon in coast], crs="EPSG:4326"
    ).to_crs(EQUAL_AREA)
    coast_tree = shapely.STRtree(coast_xy.to_numpy())

    print("\n  sampling district interiors ...")
    all_pts, owner = [], []
    for _, row in geo.iterrows():
        pts = sample_polygon(row.geometry, args.samples)
        all_pts.extend(pts)
        owner.extend([int(row.district_id)] * len(pts))
    print(f"  {len(all_pts)} points -> {int(np.ceil(len(all_pts)/BATCH))} elevation calls")
    elev = elevations(all_pts)
    print(f"  elevation retrieved, {np.isfinite(elev).mean():.1%} valid")

    df = pd.DataFrame(
        {"district_id": owner,
         "lat": [p[0] for p in all_pts], "lon": [p[1] for p in all_pts], "elev": elev}
    )

    cent = geo.to_crs(EQUAL_AREA).geometry.centroid
    rows = []
    for (did, g), cpt in zip(df.groupby("district_id"), cent.to_numpy()):
        e = g["elev"].to_numpy()
        valid = e[np.isfinite(e)]
        gx, gy = plane_gradient(g["lat"].to_numpy(), g["lon"].to_numpy(), e)
        slope = float(np.hypot(gx, gy))                       # m per km
        aspect = float((np.degrees(np.arctan2(-gx, -gy))) % 360)  # downhill bearing
        windward = WINDWARD_MIN <= aspect <= WINDWARD_MAX

        idx = coast_tree.nearest(cpt)
        dist_km = float(cpt.distance(coast_xy.to_numpy()[idx]) / 1000.0)
        cpoint = coast_xy.to_crs("EPSG:4326").to_numpy()[idx]
        bearing = float(
            np.degrees(np.arctan2(cpoint.x - g["lon"].mean(), cpoint.y - g["lat"].mean())) % 360
        )
        rows.append(
            {
                "district_id": int(did),
                "mean_elev": float(np.mean(valid)) if valid.size else 0.0,
                "max_elev": float(np.max(valid)) if valid.size else 0.0,
                "elev_range": float(np.ptp(valid)) if valid.size > 1 else 0.0,
                "mean_slope": slope,
                "grad_east": gx,
                "grad_north": gy,
                "dominant_aspect": aspect,
                "orog_exposure": slope * (1.0 if windward else 0.25) / 10.0,
                "dist_to_coast_km": dist_km,
                "coast_normal_bearing": bearing,
                "n_samples": int(np.isfinite(e).sum()),
            }
        )

    out = pd.DataFrame(rows)
    out.to_parquet(STATIC / "district_terrain.parquet", index=False)

    master = pd.read_csv(STATIC / "districts_master.csv")
    j = master.merge(out, on="district_id")
    print(f"\nwrote district_terrain.parquet  ({len(out)} districts)")
    print(f"\n{'zone':>5}{'mean_elev':>11}{'elev_range':>12}{'slope m/km':>12}"
          f"{'dist_coast':>12}{'orog_exp':>10}")
    for z, g in j.groupby("zone_code"):
        print(f"{z:>5}{g.mean_elev.mean():>11.0f}{g.elev_range.mean():>12.0f}"
              f"{g.mean_slope.mean():>12.1f}{g.dist_to_coast_km.mean():>12.0f}"
              f"{g.orog_exposure.mean():>10.2f}")
    print("\nsteepest districts (the orographic story):")
    for _, r in j.nlargest(6, "mean_slope")[
        ["district_name", "state", "zone_code", "mean_elev", "mean_slope", "dist_to_coast_km"]
    ].iterrows():
        print(f"   {r.district_name:<16}{r.state:<18}{r.zone_code}  "
              f"elev {r.mean_elev:>6.0f} m  slope {r.mean_slope:>6.1f} m/km  "
              f"coast {r.dist_to_coast_km:>5.0f} km")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
