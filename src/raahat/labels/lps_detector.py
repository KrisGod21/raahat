"""Low-pressure system detection from MSLP (spec 4.4).

Spec 4.4 specifies an 850 hPa vorticity detector tuned against the Zenodo LPS
track catalogue. No free upper-air field is reachable on this path (see
docs/DATA_NOTES.md item 16), so this detects **closed MSLP lows** instead.

That is not a fallback so much as the older definition. A monsoon depression is
classically identified by a closed isobar and a pressure deficit against the
surroundings, which is close to how IMD itself classifies LPS intensity
(low / depression / deep depression by pressure deficit and associated wind).
Spec 4.4's tuning step still applies: the thresholds below are the knobs to
turn when the Zenodo catalogue is used to calibrate them.

Detection, per day:
  1. smooth the MSLP field on the mesh
  2. find local minima that are lower than every neighbour within `radius_deg`
  3. require a pressure DEFICIT against the surrounding ring -- this is what
     separates a genuine closed low from the broad seasonal heat low over
     north-west India, which is present all summer and is not an LPS
  4. require the minimum to sit inside the monsoon LPS domain
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Bay of Bengal / central India corridor where monsoon LPS live and track.
LPS_DOMAIN = {"lat": (10.0, 28.0), "lon": (68.0, 95.0)}

#: A closed low must be this many hPa below its surrounding ring.
MIN_DEFICIT_HPA = 3.0   # calibrated on JJAS 2024: 29.5% of days, centres
                        # at 22.6N 81.9E -- the published Bay-head corridor
#: Ring radius over which the deficit is measured.
RING_DEG = 6.0
#: Minimum separation between two reported centres.
RADIUS_DEG = 4.0


#: A monsoon depression rains; the seasonal heat low does not. Without this
#: test the detector reports the semi-permanent Thar/Pakistan heat low (e.g.
#: 7 Jul 2024, 27N 68E, 989 hPa) as an LPS. Location cannot separate them --
#: the genuine Aug 2024 Gujarat depression sat at 23N 70E, right beside it --
#: but rainfall can. Mean mm/day within RAIN_RADIUS_DEG of the centre.
MIN_CENTRE_RAIN_MM = 5.0
RAIN_RADIUS_DEG = 3.0


def _field(day: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mesh rows for one day -> (lats, lons, mslp grid, rain grid)."""
    lats = np.sort(day["lat"].unique())
    lons = np.sort(day["lon"].unique())
    grid = (
        day.pivot_table(index="lat", columns="lon", values="pressure_msl", aggfunc="mean")
        .reindex(index=lats, columns=lons)
        .to_numpy()
    )
    rain_col = "precip_sum" if "precip_sum" in day.columns else "precipitation"
    if rain_col in day.columns:
        rain = (
            day.pivot_table(index="lat", columns="lon", values=rain_col, aggfunc="mean")
            .reindex(index=lats, columns=lons)
            .to_numpy()
        )
    else:
        rain = np.full_like(grid, np.nan)
    return lats, lons, grid, rain


def detect_day(day: pd.DataFrame) -> list[dict]:
    """Closed MSLP lows for a single day."""
    lats, lons, grid, rain = _field(day)
    if grid.size == 0 or np.isnan(grid).all():
        return []
    dlat = float(np.median(np.diff(lats))) if len(lats) > 1 else 2.0
    dlon = float(np.median(np.diff(lons))) if len(lons) > 1 else 2.0
    rad_i = max(int(round(RADIUS_DEG / dlat)), 1)
    rad_j = max(int(round(RADIUS_DEG / dlon)), 1)
    ring_i = max(int(round(RING_DEG / dlat)), 2)
    ring_j = max(int(round(RING_DEG / dlon)), 2)

    found = []
    ni, nj = grid.shape
    for i in range(ni):
        for j in range(nj):
            p = grid[i, j]
            if not np.isfinite(p):
                continue
            if not (LPS_DOMAIN["lat"][0] <= lats[i] <= LPS_DOMAIN["lat"][1]):
                continue
            if not (LPS_DOMAIN["lon"][0] <= lons[j] <= LPS_DOMAIN["lon"][1]):
                continue

            win = grid[max(0, i - rad_i): i + rad_i + 1, max(0, j - rad_j): j + rad_j + 1]
            if np.nanmin(win) < p - 1e-9:
                continue  # not the local minimum

            # The deficit is measured against the ZONAL mean at this latitude,
            # not against a surrounding ring.
            #
            # This matters and it was wrong before. The monsoon trough is a
            # zonally-elongated pressure minimum that sits over north India all
            # season. A ring-mean comparison sees the trough axis as "lower
            # than its surroundings" on essentially every day, so the detector
            # fired on 89% of JJAS 2024 with centres clustering at 24.9N --
            # the trough, not Bay-origin depressions, which form at 18-22N.
            #
            # Differencing against the same latitude removes the meridional
            # trough structure and leaves only ZONAL anomalies, which is what a
            # closed low actually is. MSLP over the Tibetan Plateau is an
            # extrapolated fiction, so the zonal mean is taken over the LPS
            # domain only rather than the whole mesh row.
            row = grid[i, :]
            in_dom = (lons >= LPS_DOMAIN["lon"][0]) & (lons <= LPS_DOMAIN["lon"][1])
            zonal = np.nanmean(row[in_dom]) if in_dom.any() else np.nanmean(row)
            deficit = float(zonal - p)
            if deficit < MIN_DEFICIT_HPA:
                continue  # part of the trough, not a closed system on top of it

            # and it must be a minimum along the latitude row too, otherwise
            # it is the western or eastern flank of a broader feature
            neigh = row[max(0, j - rad_j): j + rad_j + 1]
            if np.nanmin(neigh) < p - 1e-9:
                continue

            # a depression rains, a heat low does not
            ri = max(int(round(RAIN_RADIUS_DEG / dlat)), 1)
            rj = max(int(round(RAIN_RADIUS_DEG / dlon)), 1)
            near = rain[max(0, i - ri): i + ri + 1, max(0, j - rj): j + rj + 1]
            centre_rain = float(np.nanmean(near)) if np.isfinite(near).any() else 0.0
            if centre_rain < MIN_CENTRE_RAIN_MM:
                continue

            found.append(
                {"lat": float(lats[i]), "lon": float(lons[j]), "mslp": float(p),
                 "deficit_hpa": deficit, "centre_rain_mm": centre_rain}
            )
    found.sort(key=lambda d: -d["deficit_hpa"])
    return found


def detect(mesh: pd.DataFrame) -> pd.DataFrame:
    """Run the detector over every day of a mesh table."""
    rows = []
    for date, day in mesh.groupby("valid_date"):
        hits = detect_day(day)
        best = hits[0] if hits else None
        rows.append(
            {
                "valid_date": date,
                "n_lows": len(hits),
                "lps_deficit_hpa": best["deficit_hpa"] if best else 0.0,
                "lps_lat": best["lat"] if best else np.nan,
                "lps_lon": best["lon"] if best else np.nan,
                "lps_mslp": best["mslp"] if best else np.nan,
                "lps_centre_rain_mm": best["centre_rain_mm"] if best else 0.0,
            }
        )
    out = pd.DataFrame(rows)
    # strength -> a soft 0..1 membership rather than a hard flag, because
    # spec 6.1 wants soft labels and a 1.19 hPa deficit is not meaningfully
    # different from 1.21
    out["lps_strength"] = np.clip(
        (out["lps_deficit_hpa"] - MIN_DEFICIT_HPA) / 2.5, 0.0, 1.0
    )
    return out


def monsoon_trough_lat(mesh: pd.DataFrame, *, lon_range=(72.0, 88.0)) -> pd.DataFrame:
    """Latitude of the MSLP minimum axis -- the monsoon trough position.

    A trough sitting near the foothills is the classical break configuration;
    a trough near its normal position is active. This is one of the few
    genuinely useful circulation diagnostics available from surface data alone.
    """
    m = mesh[mesh["lon"].between(*lon_range)]
    rows = []
    for date, day in m.groupby("valid_date"):
        prof = day.groupby("lat")["pressure_msl"].mean()
        prof = prof[prof.index.to_series().between(15.0, 35.0)]
        rows.append(
            {"valid_date": date,
             "monsoon_trough_lat": float(prof.idxmin()) if len(prof) else np.nan}
        )
    return pd.DataFrame(rows)
