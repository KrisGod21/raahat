"""Synthetic, contract-conforming data so model work can start before ingest.

Spec 0, rule 2: this is the ONE file permitted to invent numbers, and it writes
to data/synthetic/ only. It must never write to data/processed/.

The synthetic world is built to have the structure the project claims exists,
so that the model plumbing can be verified end to end:

  * a latent regime that PERSISTS for several days (so season-blocked splits
    matter and consecutive days are genuinely near-duplicates, spec 7.2)
  * regimes that BLEND -- some days are {LPS: 0.6, ACTIVE: 0.4} (spec 6.1)
  * forecast-mesh features whose noise GROWS with lead time, so the leakage
    canary of spec 5.7 is actually exercised rather than trivially passed
  * raw NWP rainfall whose bias is CONDITIONAL on regime and zone -- the
    central premise. Break days over-forecast, LPS days under-forecast,
    windward-zone active days under-forecast hardest.

If a model cannot recover that structure from this data, the bug is in the
model, not the data. That is the entire point of this file.

Usage:
    python scripts/make_synthetic.py --districts 190 --seasons 2024 2025 2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raahat import contract as C  # noqa: E402

OUT_DIR = C.DATA_DIR / "synthetic"

ZONES = ["W", "C", "N"]

# Per-regime multiplicative bias of the raw NWP rainfall, by zone. This is the
# thing the corrector has to learn. Values are deliberately strong so that a
# broken pipeline is obvious rather than marginal.
RAW_BIAS = {
    #              W (orographic)  C (LPS corridor)  N (WD belt)
    "ACTIVE":           {"W": 0.45, "C": 0.85, "N": 0.90},
    "BREAK":            {"W": 1.70, "C": 2.10, "N": 1.80},
    "LPS":              {"W": 0.80, "C": 0.50, "N": 0.95},
    "WD":               {"W": 1.10, "C": 1.05, "N": 0.60},
    "EASTERLY_COASTAL": {"W": 0.95, "C": 0.70, "N": 1.20},
    "WEAK_TRANSITION":  {"W": 1.25, "C": 1.30, "N": 1.20},
}

# Rainfall intensity per regime: (gamma shape, gamma scale) for the wet part,
# and P(wet). Tuned so ~1-3% of district-days clear 64.5 mm (spec 4.11).
RAIN_PARAMS = {
    "ACTIVE":           {"p_wet": 0.68, "shape": 0.90, "scale": 12.5},
    "BREAK":            {"p_wet": 0.22, "shape": 0.70, "scale": 4.5},
    "LPS":              {"p_wet": 0.76, "shape": 1.15, "scale": 18.5},
    "WD":               {"p_wet": 0.42, "shape": 0.80, "scale": 8.0},
    "EASTERLY_COASTAL": {"p_wet": 0.55, "shape": 0.85, "scale": 9.5},
    "WEAK_TRANSITION":  {"p_wet": 0.35, "shape": 0.75, "scale": 7.0},
}

# Physically-legible Stage-A signatures. Features not listed get a fixed
# random offset per regime, so every feature carries some signal but these
# few are inspectable by eye.
SIGNATURES: dict[str, dict[str, float]] = {
    # Names track config/features.yaml, which was rewritten to surface-only
    # fields on 2026-09-21 (docs/DATA_NOTES.md item 16). Unknown names are
    # skipped with a warning rather than raising, so a feature-list edit
    # degrades the synthetic signal instead of breaking the generator.
    "ACTIVE": {"cmz_rain_anom_fcst": 1.6, "u10_cmz_mean": 8.0,
               "rh2m_national_mean": 80.0, "rain_max_westcoast": 55.0},
    "BREAK": {"cmz_rain_anom_fcst": -1.5, "u10_cmz_mean": 2.0,
              "rh2m_national_mean": 52.0, "monsoon_trough_lat": 30.0},
    "LPS": {"lps_deficit_hpa": 4.0, "lps_strength": 0.9,
            "vort10_max_bay": 8.0, "mslp_min_anom_land_c": -6.0},
    "WD": {"t2m_anom_nw": -3.5, "mslp_anom_nw": 3.0, "rain_mean_nw": 12.0},
    "EASTERLY_COASTAL": {"u10_eastcoast_mean": -9.0, "rain_max_eastcoast": 40.0},
    "WEAK_TRANSITION": {"cmz_rain_anom_fcst": -0.2, "rh2m_national_mean": 62.0},
}


# Regimes that plausibly co-occur -> soft labels (spec 6.1).
BLEND_PAIRS = [("LPS", "ACTIVE", 0.6), ("WD", "ACTIVE", 0.55), ("EASTERLY_COASTAL", "ACTIVE", 0.6)]


def jjas_dates(season: int) -> pd.DatetimeIndex:
    """1 Jun - 30 Sep of the given year. 122 days."""
    return pd.date_range(f"{season}-06-01", f"{season}-09-30", freq="D")


def synth_districts(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """A synthetic district master. NOT data/static/districts_master.csv."""
    zone = rng.choice(ZONES, size=n, p=[0.32, 0.40, 0.28])
    mean_elev = np.where(
        zone == "N", rng.uniform(400, 3200, n),
        np.where(zone == "W", rng.uniform(10, 1300, n), rng.uniform(150, 700, n)),
    )
    elev_range = mean_elev * rng.uniform(0.25, 1.1, n) + rng.uniform(20, 200, n)
    dist_coast = np.where(
        zone == "W", rng.uniform(2, 140, n),
        np.where(zone == "C", rng.uniform(60, 700, n), rng.uniform(300, 1100, n)),
    )
    # orographic exposure: steep + windward. Highest in W, high in N.
    orog = (elev_range / 1500.0) * np.where(zone == "W", 1.0, np.where(zone == "N", 0.8, 0.25))
    return pd.DataFrame(
        {
            "district_id": np.arange(1, n + 1, dtype=np.int32),
            "zone_code": zone,
            "subdivision": [f"SUBDIV_{z}_{i % 7}" for i, z in enumerate(zone)],
            "mean_elev": mean_elev.astype(np.float32),
            "elev_range": elev_range.astype(np.float32),
            "dist_to_coast_km": dist_coast.astype(np.float32),
            "orog_exposure": np.clip(orog, 0, 3).astype(np.float32),
        }
    )


def synth_regime_sequence(dates: pd.DatetimeIndex, rng: np.random.Generator):
    """Persistent latent regime per day, plus soft labels.

    Persistence is the point: consecutive days inside one LPS are near
    duplicates, which is exactly the autocorrelation trap of spec 7.2.
    """
    n_days = len(dates)
    # seasonal base rates; WD is rare in JJAS (spec 6.1 says ~2-5%)
    base = np.array([0.34, 0.24, 0.20, 0.03, 0.09, 0.10])
    hard = np.empty(n_days, dtype=object)
    soft = np.zeros((n_days, C.N_REGIMES), dtype=np.float32)

    idx = rng.choice(C.N_REGIMES, p=base)
    remaining = 0
    for d in range(n_days):
        if remaining == 0:
            idx = rng.choice(C.N_REGIMES, p=base)
            # spell length: LPS 3-7 days, active/break 4-12, others 2-5
            name = C.REGIMES[idx]
            if name == "LPS":
                remaining = int(rng.integers(3, 8))
            elif name in ("ACTIVE", "BREAK"):
                remaining = int(rng.integers(4, 13))
            else:
                remaining = int(rng.integers(2, 6))
        remaining -= 1

        name = C.REGIMES[idx]
        hard[d] = name
        soft[d, idx] = 1.0

        # blend where regimes genuinely co-occur
        for a, b, w in BLEND_PAIRS:
            if name == a and rng.random() < 0.45:
                soft[d] = 0.0
                soft[d, C.REGIMES.index(a)] = w
                soft[d, C.REGIMES.index(b)] = 1.0 - w
                break
    return hard, soft


def synth_stage_a(
    seasons: list[int], leads: list[int], rng: np.random.Generator
) -> tuple[pd.DataFrame, dict]:
    """National Stage-A table: one row per (valid_date, lead_day)."""
    feats = C.STAGE_A_FEATURES
    # fixed random mean vector per regime, then overwrite the legible ones
    means = rng.normal(0, 1.0, size=(C.N_REGIMES, len(feats))).astype(np.float64)
    missing = set()
    for ri, rname in enumerate(C.REGIMES):
        for fname, val in SIGNATURES.get(rname, {}).items():
            if fname not in feats:
                missing.add(fname)
                continue
            means[ri, feats.index(fname)] = val
    if missing:
        print(f"  note: {len(missing)} signature feature(s) no longer in the "
              f"contract, skipped: {sorted(missing)}")
    # Noise is scaled to each feature's OWN spread across regimes. Without
    # this the signature features (vorticity ~9, z500 ~-55, IVT ~420) dwarf a
    # fixed noise term and become noiseless discriminators, so day-5 stays as
    # easy as day-1 and the leakage canary of spec 5.7 is never exercised.
    spread = means.std(axis=0)
    spread[spread < 1e-6] = 1.0
    scale = spread * np.abs(rng.normal(1.0, 0.15, size=len(feats)))

    rows = []
    truth: dict[tuple, tuple] = {}
    for season in seasons:
        dates = jjas_dates(season)
        hard, soft = synth_regime_sequence(dates, rng)
        onset = 5  # days into the window
        for d, date in enumerate(dates):
            truth[(season, date.date())] = (hard[d], soft[d])
            doy = date.dayofyear
            for lead in leads:
                # noise grows with lead -> classifier accuracy MUST decay
                # real regime prediction degrades sharply by day 5
                sigma = scale * (0.80 + 0.55 * (lead - 1))
                x = soft[d] @ means + rng.normal(0, sigma)
                rec = {
                    "valid_date": date.date(),
                    "lead_day": np.int8(lead),
                    "init_datetime_utc": pd.Timestamp(date - pd.Timedelta(days=int(lead)), tz="UTC"),
                    "season": np.int16(season),
                    "label_regime": hard[d],
                    "label_regime_soft": soft[d].astype(np.float32),
                }
                rec.update({f: np.float32(v) for f, v in zip(feats, x)})
                # calendar features are exact, not noisy
                rec["doy_sin"] = np.float32(np.sin(2 * np.pi * doy / 365.25))
                rec["doy_cos"] = np.float32(np.cos(2 * np.pi * doy / 365.25))
                rec["days_since_onset_national"] = np.float32(max(0, d - onset))
                rows.append(rec)

    df = pd.DataFrame(rows)
    cols = [f.name for f in C.regime_input_schema()]
    return df[cols], truth


def synth_stage_b(
    districts: pd.DataFrame,
    seasons: list[int],
    leads: list[int],
    truth: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Per-district feature table conforming to spec 13.2."""
    frames = []
    n_d = len(districts)
    did = districts["district_id"].to_numpy()
    zone = districts["zone_code"].to_numpy()

    for season in seasons:
        dates = jjas_dates(season)
        for d, date in enumerate(dates):
            hard, soft = truth[(season, date.date())]
            rp = RAIN_PARAMS[hard]
            doy = date.dayofyear

            # --- observed truth (identical across leads; it is the same day)
            wet = rng.random(n_d) < rp["p_wet"]
            amount = rng.gamma(rp["shape"], rp["scale"], n_d)
            # orography amplifies, strongly in W and N during wet regimes
            amp = 1.0 + districts["orog_exposure"].to_numpy() * np.where(
                np.isin(zone, ["W", "N"]), 1.05, 0.22
            )
            obs = np.where(wet, amount * amp, 0.0)

            for lead in leads:
                bias = np.array([RAW_BIAS[hard][z] for z in zone])
                # forecast error grows with lead
                err = rng.lognormal(0.0, 0.22 + 0.10 * (lead - 1), n_d)
                base_fc = obs * bias * err
                # displacement: neighbourhood max exceeds the district value
                nbr_max = base_fc * rng.uniform(1.15, 2.8, n_d)
                nbr_mean = base_fc * rng.uniform(0.75, 1.15, n_d)

                rec = {
                    "district_id": did.astype(np.int32),
                    "valid_date": np.repeat(date.date(), n_d),
                    "lead_day": np.full(n_d, lead, dtype=np.int8),
                    "init_datetime_utc": np.repeat(
                        pd.Timestamp(date - pd.Timedelta(days=int(lead)), tz="UTC"), n_d
                    ),
                    "season": np.full(n_d, season, dtype=np.int16),
                    "zone_code": zone,
                    "subdivision": districts["subdivision"].to_numpy(),
                    "obs_rain_mm": obs,
                }

                for mi, m in enumerate(C.NWP_MODELS):
                    jitter = rng.lognormal(0.0, 0.14, n_d)
                    pm = base_fc * jitter
                    rec[f"pr_{m}"] = pm
                    rec[f"pr_{m}_nbr_mean"] = nbr_mean * jitter
                    rec[f"pr_{m}_nbr_max"] = nbr_max * jitter
                    rec[f"pr_{m}_nbr_p90"] = (nbr_mean + 0.6 * (nbr_max - nbr_mean)) * jitter
                    rec[f"pr_{m}_grad"] = np.abs(nbr_max - nbr_mean) * rng.uniform(0.3, 1.0, n_d)
                    rec[f"pr_{m}_prev_day"] = pm * rng.uniform(0.4, 1.6, n_d)
                    rec[f"pr_{m}_3day_sum"] = pm * rng.uniform(1.6, 3.4, n_d)

                stack = np.vstack([rec[f"pr_{m}"] for m in C.NWP_MODELS])
                rec["mm_mean"] = stack.mean(axis=0)
                rec["mm_median"] = np.median(stack, axis=0)
                rec["mm_spread"] = stack.std(axis=0)
                rec["mm_range"] = stack.max(axis=0) - stack.min(axis=0)
                rec["mm_pop"] = (stack > 0.2).mean(axis=0)
                rec["mm_agree_heavy"] = (stack > C.THRESHOLDS["heavy"]).mean(axis=0)
                for c in C.STAGE_B_ENSEMBLE:
                    rec[c] = np.full(n_d, np.nan)

                for i, c in enumerate(C.REGIME_PROB_COLS):
                    rec[c] = np.full(n_d, soft[i], dtype=np.float64)
                p = np.clip(soft, 1e-9, 1.0)
                rec["regime_entropy"] = np.full(n_d, float(-(p * np.log(p)).sum()))
                rec["p_LPS_trend"] = rng.normal(0, 0.1, n_d)
                rec["regime_argmax"] = np.repeat(C.REGIMES[int(np.argmax(soft))], n_d)

                lps = soft[C.REGIMES.index("LPS")]
                rec["vort850_max_300km"] = rng.normal(2.0 + 7.0 * lps, 1.5, n_d)
                rec["vort850_dist_km"] = rng.uniform(40, 900, n_d)
                rec["vort850_bearing"] = rng.uniform(0, 360, n_d)
                rec["mslp_anom"] = rng.normal(-4.0 * lps, 1.8, n_d)
                u850 = rng.normal(8.0 * soft[C.REGIMES.index("ACTIVE")] + 2.0, 3.0, n_d)
                v850 = rng.normal(1.0, 3.0, n_d)
                rec["u850"] = u850
                rec["v850"] = v850
                rec["wspd850"] = np.hypot(u850, v850)
                rec["shear_u"] = rng.normal(20.0, 6.0, n_d)
                rh = np.clip(rng.normal(62 + 18 * soft[C.REGIMES.index("ACTIVE")], 8, n_d), 5, 100)
                rec["rh850"] = rh
                rec["ivt_mag"] = np.clip(rng.normal(320 + 200 * lps, 90, n_d), 10, None)
                rec["ivt_dir"] = rng.uniform(0, 360, n_d)
                rec["z500_anom"] = rng.normal(-40 * soft[C.REGIMES.index("WD")], 18, n_d)
                rec["cmz_rain_anom_fcst"] = np.full(
                    n_d, 1.6 * soft[C.REGIMES.index("ACTIVE")] - 1.5 * soft[C.REGIMES.index("BREAK")]
                )

                # terrain interaction: the feature that replaces "orographic regime"
                grad = districts["elev_range"].to_numpy() / 1000.0
                rec["orog_exposure"] = districts["orog_exposure"].to_numpy()
                rec["upslope_flux"] = u850 * grad * (rh / 100.0)
                rec["coast_onshore_flux"] = (
                    u850 * rec["ivt_mag"] / 1000.0
                    * np.exp(-districts["dist_to_coast_km"].to_numpy() / 200.0)
                )
                rec["dist_to_coast_km"] = districts["dist_to_coast_km"].to_numpy()
                rec["mean_elev"] = districts["mean_elev"].to_numpy()
                rec["elev_range"] = districts["elev_range"].to_numpy()

                rec["doy_sin"] = np.full(n_d, np.sin(2 * np.pi * doy / 365.25))
                rec["doy_cos"] = np.full(n_d, np.cos(2 * np.pi * doy / 365.25))
                rec["clim_mean_district_doy"] = amp * 6.0
                rec["clim_p90_district_doy"] = amp * 26.0
                rec["clim_p99_district_doy"] = amp * 85.0
                rec["days_since_local_onset"] = np.full(n_d, float(max(0, d - 5)))

                rec["label_regime"] = np.repeat(hard, n_d)
                rec["label_regime_soft"] = [soft.astype(np.float32)] * n_d
                frames.append(pd.DataFrame(rec))

    df = pd.concat(frames, ignore_index=True)
    cols = [f.name for f in C.features_schema()]
    df = df[cols]
    f32 = [f.name for f in C.features_schema() if f.type == "float"]
    df[f32] = df[f32].astype(np.float32)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--districts", type=int, default=190)
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025, 2026])
    ap.add_argument("--leads", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if "processed" in str(OUT_DIR):
        raise SystemExit("refusing to write synthetic data outside data/synthetic/")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    districts = synth_districts(args.districts, rng)
    districts.to_csv(OUT_DIR / "districts_master_SYNTHETIC.csv", index=False)

    stage_a, truth = synth_stage_a(args.seasons, args.leads, rng)
    C.validate(stage_a, C.regime_input_schema())
    stage_a.to_parquet(OUT_DIR / "regime_inputs.parquet", index=False)

    for season in args.seasons:
        part = synth_stage_b(districts, [season], args.leads, truth, rng)
        C.validate(part)
        part.to_parquet(OUT_DIR / f"features_{season}.parquet", index=False)
        heavy = (part["obs_rain_mm"] > C.THRESHOLDS["heavy"]).mean()
        vheavy = (part["obs_rain_mm"] > C.THRESHOLDS["very_heavy"]).mean()
        print(
            f"  features_{season}.parquet  rows={len(part):>7,}  "
            f">64.5mm {heavy:6.2%}   >115.6mm {vheavy:6.2%}"
        )

    print(f"  regime_inputs.parquet     rows={len(stage_a):>7,}")
    print(f"  districts                 {len(districts)}")
    print(f"\nwrote to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
