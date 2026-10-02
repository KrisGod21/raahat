"""Simplify district polygons for the web (spec 4.6).

Spec 4.6: "Simplify polygons for the web to < 1.5 MB total for the whole
country. The map must not be the slow part of the demo."

The raw Census 2011 boundaries are 24.5 MB of GeoJSON. Served as-is they blow
straight through the frontend's 3-second timeout and the map never draws --
which is exactly the failure this instruction exists to prevent.

Douglas-Peucker simplification, searching for the largest tolerance that still
keeps every district a valid, non-empty polygon. Topology is not preserved
between neighbours (that would need mapshaper), so tiny slivers can appear
between districts at high zoom. At the zoom levels a national choropleth is
viewed at, that is invisible; the alternative is a map that does not load.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raahat import contract as C  # noqa: E402

STATIC = C.DATA_DIR / "static"
SRC = STATIC / "districts.geojson"
OUT = STATIC / "districts_simplified.geojson"
TARGET_MB = 1.5


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}; run scripts/build_districts.py")
    gdf = gpd.read_file(SRC)
    before = SRC.stat().st_size / 1e6
    print(f"source: {len(gdf)} districts, {before:.1f} MB")

    keep = [c for c in ("district_id", "DISTRICT", "ST_NM", "zone_code") if c in gdf.columns]
    gdf = gdf[keep + ["geometry"]]

    best = None
    for tol in (0.002, 0.005, 0.01, 0.02, 0.03, 0.05):
        s = gdf.copy()
        s["geometry"] = s.geometry.simplify(tol, preserve_topology=True)
        # a tolerance that empties or invalidates a district is too coarse --
        # a missing district is far worse than a slightly jagged one
        if s.geometry.is_empty.any() or not s.geometry.is_valid.all():
            print(f"  tol {tol}: rejected (empty or invalid geometry)")
            continue
        tmp = OUT.with_suffix(".tmp.geojson")
        s.to_file(tmp, driver="GeoJSON")
        mb = tmp.stat().st_size / 1e6
        print(f"  tol {tol}: {mb:.2f} MB")
        best = (tol, mb, tmp)
        if mb <= TARGET_MB:
            break
        tmp.unlink(missing_ok=True)
        best = None

    if best is None:
        raise SystemExit("no tolerance reached the size target without breaking geometry")

    tol, mb, tmp = best
    tmp.replace(OUT)
    print(f"\nwrote {OUT.name}: {mb:.2f} MB at tolerance {tol} "
          f"({before / mb:.0f}x smaller, target was {TARGET_MB} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
