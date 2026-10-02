# RAAHAT

**R**egime-**A**ware **A**djustment of **H**eavy-rainfall **A**lerts & **T**hresholds

> The monsoon has moods. A single bias correction cannot follow them.
> RAAHAT predicts which weather regime India will be in — three days ahead, from the forecast itself — and applies the correction that regime deserves.

Smart India Hackathon submission. Problem statement: *AI/ML-based rainfall post-processing system that identifies the prevailing weather regime and applies suitable correction to raw NWP rainfall forecasts, to improve district/grid-level rainfall forecasts, especially for heavy and very heavy rainfall events.*

---

## 0. HOW TO USE THIS DOCUMENT

This is a **build specification**, not documentation of something that exists. It is written to be handed to an AI coding agent and to human teammates simultaneously.

**Rules for the implementing agent:**

1. **Build in the phase order given in §16.** Do not start Phase 3 before Phase 1's acceptance test passes.
2. **Never fabricate data.** If a download fails, fail loudly and write the error to `data/_manifest/errors.log`. Do not generate plausible-looking numbers and continue. The one exception is `scripts/make_synthetic.py`, which exists only to unblock frontend work and must write to `data/synthetic/` and never to `data/processed/`.
3. **Every external URL and API parameter in §4 must be verified against the live docs before use.** API parameter names drift. If a documented parameter name in this spec does not exist, check the provider's docs page (linked), use the correct one, and record the correction in `docs/DATA_NOTES.md`.
4. **CPU only.** No GPU anywhere. If a model needs a GPU, it is the wrong model for this project.
5. **Determinism.** Set `random_state=42` / `torch.manual_seed(42)` everywhere. Every result in the presentation must be reproducible by `make all`.
6. **The demo must run with the network cable unplugged.** Anything the demo touches lives in `data/demo_cache/`.
7. **Do not add features not in this spec.** If you think something is missing, write it to `docs/PROPOSALS.md` and keep building.

---

## 1. THE EXACT PROBLEM WE ARE SOLVING

### 1.1 The problem in one paragraph

Numerical weather prediction (NWP) models have systematic rainfall errors over India, and **those errors are conditional on the synoptic situation, not constant**. The same model that over-forecasts drizzle during a break monsoon will under-forecast the 250 mm core of a Bay of Bengal depression and miss orographic enhancement on the windward Western Ghats. Operational post-processing at IMD uses a multi-model ensemble with weights derived from season-long correlations — i.e. averaged across all regimes. Published post-processing studies over India confirm the failure signature: analog and logistic-regression post-processing of GEFS improved forecasts overall but **under-performed specifically during the monsoon season**, and the analog method under-performed specifically over the Western Ghats. A regime-blind corrector applied to a regime-switching atmosphere fixes the average and breaks the extremes.

### 1.2 What we are actually building

A **post-processing layer** that sits between raw NWP output and the district warning desk. It:

1. Predicts, from the **forecast fields themselves** (so it works at lead time, not just in hindsight), a **soft probability distribution over six synoptic regimes** for each valid day.
2. Uses those probabilities as **mixture weights over regime-specialist correction heads** (a Mixture of Regime Experts), producing a full **predictive rainfall distribution** per district per lead day.
3. Converts that distribution into **calibrated probabilities of exceeding IMD's operational thresholds** (64.5 / 115.6 / 204.5 mm per 24 h) and into IMD's four-colour warning codes via an adjustable cost-loss decision rule.
4. Reports **verification stratified by regime** — which is simultaneously the validation of the system and a scientific artefact (the Regime Error Atlas) that has value even if nobody runs the model.

### 1.3 What we are NOT building

We are not building a weather model. We do not run WRF. We do not train a global AI forecast model. We do not replace IMD, the Bharat Forecast System, or NCUM. **We consume their output and calibrate it.** State this on slide 1 — it pre-empts the single most dangerous judge question.

### 1.4 Users

| Role | Who | What they get |
|---|---|---|
| Primary user | IMD RMC/MC duty forecaster | A district table with corrected rainfall, exceedance probabilities, suggested colour code, and the *reason* |
| Secondary user | District Collector / SDMA-DDMA officer | A map and a plain-language district bulletin with a confidence number |
| Beneficiary | Population of affected districts; rain-fed farmers | Fewer missed warnings, fewer false red alerts |
| Who would fund it | MoES (IMD / NCMRWF / IITM); secondarily CWC Flood Met Offices, state SDMAs, PMFBY crop insurance |

**Do not pitch this as a consumer weather app.** It is operational decision support for the people who issue warnings.

### 1.5 Three deliberate disagreements with the problem statement

Stating these makes us look like engineers who thought, not transcribers. Put them on a slide titled *"Where we differ from the brief — and why."*

| PS says | We do | Why |
|---|---|---|
| "First identifies the regime, **then** applies suitable correction" | **Soft gating**: regime probabilities are mixture weights, never an `argmax` switch | Real days are blends (an LPS embedded in an active spell while a WD interacts aloft — Himachal, Aug 2023). Hard switching creates discontinuities at regime boundaries, fragments the training set, and fails catastrophically on a single misclassification instead of degrading gracefully. |
| Lists "orographic" and "coastal" as regimes alongside "active/break/depression" | Treats them as **continuous terrain modifiers** on a second, orthogonal axis | Orographic and coastal rainfall are *geographic response modes*, not synoptic states. Mumbai in an active spell and Mumbai in a break are both "coastal". Modelling them as peers of "break monsoon" is a category error, creates unlabelable classes, and worsens class imbalance. We still deliver every named regime in the UI — as `synoptic regime × terrain exposure`. |
| Implies a deterministic corrected rainfall number | Outputs a **predictive distribution**; the number is its median | "Heavy rainfall probability" is an explicit required outcome. A distribution gives it natively and honestly; a corrected point value plus a bolted-on probability head does not. |

---

## 2. THE EXACT MVP

Two levels. **Level 1 is not the demo — it is the thing that must work by week 3 so that Level 2 is possible.** Level 2 is what we present.

### 2.1 LEVEL 1 — Working core (target: end of week 3)

| | Exact choice |
|---|---|
| **Spatial scope** | 3 contrasting zones, ~190 districts (see §2.3) |
| **Temporal scope** | JJAS (1 Jun – 30 Sep) only. Seasons 2024, 2025, 2026 |
| **Lead times** | Day 1, 2, 3 |
| **NWP inputs** | GFS + ICON via Open-Meteo Previous Runs API |
| **Truth** | IMD 0.25° gridded daily rainfall → area-weighted district totals |
| **Regimes** | 4 classes: `ACTIVE`, `BREAK`, `LPS`, `OTHER` + orographic exposure as a feature |
| **Regime model** | LightGBM multiclass, ~30 features, soft probability output |
| **Corrector** | LightGBM: 3 quantile regressors (q=0.1/0.5/0.9) + 3 binary classifiers at 64.5/115.6/204.5 mm, with regime probabilities as input features |
| **Baselines** | (1) raw GFS, (2) multi-model mean, (3) global empirical quantile mapping, (4) climatology |
| **Output** | Parquet table: `district × date × lead × {raw, corrected_median, p64, p115, p204, regime_probs, colour}` |
| **UI** | District choropleth + date/lead slider + district detail panel |
| **Backend** | FastAPI reading DuckDB over Parquet |

**Explicitly NOT in Level 1:** CSGD, the MoE gate, Western Disturbances, ensemble spread, ECMWF, live data, the analog engine, the atlas, authentication, mobile, alerts, bulletin text generation.

### 2.2 LEVEL 2 — The presented prototype (target: week 8)

Everything in Level 1, plus:

- **All-India** (~750 districts), leads 1–5
- **CSGD corrector trained on CRPS** (censored-shifted-gamma predictive distribution) replacing the quantile hack
- **Soft Mixture-of-Regime-Experts gating**, 6 regimes including Western Disturbance and Easterly/Coastal
- **ECMWF IFS 0.25 + AIFS** added via ECMWF open data; **ensemble spread** as a feature
- **The Regime Error Atlas** — regime × zone × lead-time error heatmap (this is USP #1)
- **Reliability diagrams, Brier skill scores, per-regime stratified scorecard**
- **Analog Memory panel** — nearest historical analog event with its observed outcome
- **Cost-loss threshold slider** — user-adjustable miss-aversion
- **IMD warning benchmark** (if API registration granted; see §4.8)
- **Replay mode** for three real events, fully offline
- Auto-generated district bulletin text (templated, not LLM-decided)

### 2.3 Zones for Level 1 (chosen for regime contrast, not convenience)

| Zone | States/subdivisions | Dominant regimes it exercises |
|---|---|---|
| **W — West coast orographic** | Konkan & Goa, Madhya Maharashtra, Coastal Karnataka, Kerala | Active monsoon + strong orographic modifier; offshore vortices |
| **C — LPS corridor** | Vidarbha, Chhattisgarh, Odisha, Telangana | LPS/depression tracks, active/break contrast, the core monsoon zone |
| **N — WD interaction belt** | Himachal Pradesh, Uttarakhand, Punjab, Haryana, West UP | Western Disturbance × monsoon-trough interaction, extreme orography |

These three zones are where the three highest-impact demo events live, which is not a coincidence.

---

## 3. THE EXACT USPs

Know the difference cold, because judges probe it:

- **Feature** = something we built. ("A heavy-rain probability map.") Not a USP.
- **Innovation** = a technically meaningful improvement. ("Regime predicted from forecast fields, not diagnosed from observations.")
- **USP** = something that makes us meaningfully different from what exists. Must survive the question *"hasn't someone already done this?"*

### USP 1 — The India Regime Error Atlas ★ LEAD WITH THIS

**What it is:** A quantified national atlas of *conditional* NWP rainfall error. For every (synoptic regime × homogeneous zone × lead day), we publish the raw model's bias, spread, POD, FAR and CSI at operational thresholds — and how much of it RAAHAT removes.

**Why nobody has it:** Published Indian bias-correction work is basin- or city-scale (Hirakud catchment; Chennai and Mumbai; five river basins). Published Indian regime work is diagnostic climatology, not forecast-error attribution. The two literatures have never been joined at district resolution.

**Why it matters:** It is useful to a forecaster *even if they never run our model*. That converts the project from "a tool" into **"a finding"**. Judges remember findings.

**How we implement it:** It falls out of the verification module for free. Zero additional risk. `src/verify/atlas.py`.

**How we demonstrate it:** One heatmap. Regimes on rows, zones on columns, cell colour = mean bias, cell annotation = ΔCSI. Let it sit on screen in silence for five seconds.

**Is it actually novel?** The *method* is elementary — stratified verification. The *artefact for India at this granularity* does not exist publicly. Say exactly that: *"Not a new algorithm. A new measurement."* Honesty here is worth more than a fake novelty claim, and it is unattackable.

### USP 2 — Forecast-time soft regime gating (Mixture of Regime Experts)

**What it is:** A classifier that reads the **forecast** 850 hPa vorticity, MSLP anomaly, integrated vapour transport, 500 hPa geopotential and core-zone rainfall anomaly at each lead time, and emits a probability vector over six regimes. Those probabilities become mixture weights over regime-expert correction heads.

**Why existing systems don't do this:** IMD's operational MME uses per-grid-point weights derived from correlation between forecasts and observations across a whole season — regime-averaged by construction. Regime-dependent post-processing exists in the literature (Allen et al., regime-dependent EMOS for North Atlantic wind; Du & DiMego at NCEP) but for **mid-latitude wind and temperature, not Indian monsoon rainfall at district scale**.

> **Be honest about this in the PPT.** Do not claim "regime-aware post-processing is our novel idea." Claim: *"The method class is established in Europe and the US for wind and temperature. It has never been instantiated for the Indian monsoon regime taxonomy, at district scale, with an operational heavy-rain decision layer."* That is both true and strong. If you overclaim, a knowledgeable judge ends your presentation with one citation.

**How we demonstrate it:** A stacked-area chart of regime probability against lead time as an LPS approaches, with the correction magnitude tracking underneath. This is the most cinematic 15 seconds of the demo.

**Proof it is real, not decoration:** the verification table must show that the gated model beats the *same architecture with the gate removed*. Run that ablation. Report it even if the gain is small.

### USP 3 — Calibrated probabilities at IMD's own thresholds, benchmarked against issued warnings

**What it is:** Output is `P(rain > 64.5 mm)`, `P(> 115.6 mm)`, `P(> 204.5 mm)` — IMD's exact operational heavy / very heavy / extremely heavy boundaries, which map to the yellow / orange / red colour codes. Calibration is verified with reliability diagrams. The colour threshold is set by a **cost-loss rule** the user can move, not a fixed 50%.

**Why it matters:** Recent work on operational AI forecasting for India found that models used for national-scale dissemination were **overconfident**, with predicted probabilities exceeding observed frequencies, and explicitly noted that probability calibration schemes are the remedy. We are that missing layer.

**The benchmark:** IMD publishes a REST API with district-wise warnings carrying Day-1 to Day-5 codes including `2 = Heavy Rain`, `16 = Very Heavy Rain`, `17 = Extremely Heavy Rain` plus colour codes. We archive these daily and compute POD/FAR/CSI for both IMD's issued warnings and ours, on the same districts and days.

> **Framing discipline — this can sink us if handled badly.** IMD's warning means "heavy rain at *isolated places within* the district." Our forecast is a district **areal mean**. These are different quantities. We present this as *complementary decision support and a consistency check*, **never** as "we beat IMD." Put that caveat on the slide itself. A judge who spots it before you say it will dismiss the whole project; a judge who sees you state it first will trust everything else you say.

### USP 4 — Analog Memory (cheap, and the best explainability device available)

For any forecast, retrieve the *k* most similar historical (regime vector + circulation + forecast-rainfall-pattern) days from the archive and show what actually happened on those days. *"Today's pattern most closely resembles 26 July 2005 and 16 Aug 2024; on those days this district recorded 320 mm and 180 mm."*

Costs two days of work. Produces the line judges quote back to you. Implement with `faiss-cpu` or plain `sklearn.neighbors` over a 32-dim feature vector — no training needed.

### USP ranking for the pitch

1. **Regime Error Atlas** — unattackable, because it is a measurement of Indian data, not a claimed invention.
2. **Soft regime gating** — our engineering.
3. **Calibrated operational thresholds + warning benchmark** — our credibility.
4. **Analog Memory** — our memorability.

---

## 4. THE EXACT DATASETS TO DOWNLOAD

> Every source below is **free**. Four of the six require **no account at all**. Nothing on the critical path requires a government login. This is a competitive claim — put it on a slide.

### 4.1 Ground truth — IMD 0.25° gridded daily rainfall ★ CRITICAL PATH

- **What:** IMD high-resolution daily gridded rainfall, 135 × 129 grid, 6.5°N–38.5°N, 66.5°E–100.0°E, 0.25° spacing, mm/day, 1901–2024 (updated annually).
- **Where:** `https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html` (NetCDF) and `.../Rainfall_25_Bin.html` (binary).
- **Access:** free, no login.
- **How:** `pip install imdlib`, then `imdlib.get_data('rain', start_yr, end_yr, fn_format='yearwise')` → xarray `(time, lat=129, lon=135)`. Missing value is `-999.0` — mask it. Also fetch the **1° product** (`Rainfall_1_NetCDF.html`) as a homogeneity cross-check.
- **Years to download:** 1981–2024 for climatology and percentiles; 2024–2026 for training/validation truth.
- **Note:** for the 2025 and 2026 seasons, `imdlib` also exposes near-real-time 0.25° data. Verify availability; if the current season is missing, cap the test season at 2025 and say so.
- **Known caveat to state in the PPT:** the gridded extremes record is affected by an evolving station network, with evidence of a step change in gridded extremes around the mid-1970s traced to station availability. Our window (2024–2026) is unaffected, but the climatology baseline should be computed from 1981 onward, not 1901.

### 4.2 Raw NWP forecasts — Open-Meteo ★ CRITICAL PATH

Free, **no API key**, CC BY 4.0, 10,000 calls/day on the non-commercial tier. Attribution required — put "Weather data by Open-Meteo.com (CC BY 4.0)" in the dashboard footer and on the data slide.

| Endpoint | Host | Use |
|---|---|---|
| **Previous Runs API** | `https://previous-runs-api.open-meteo.com/v1/forecast` | **Primary.** Values at fixed lead offsets of 1–7 days. Most models archived from **January 2024**. |
| **Historical Forecast API** | `https://historical-forecast-api.open-meteo.com/v1/forecast` | Seamless past forecast series from **2021** — effectively short-lead. Extends the lead-1 record to 6 monsoons. |
| **Single Runs API** | `https://single-runs-api.open-meteo.com/v1/forecast` | Full forecast horizon for one exact init time. ECMWF IFS HRES from **March 2024**. Use for clean init-time-based demo replay. |
| **Archive (ERA5)** | `https://archive-api.open-meteo.com/v1/archive` | ERA5 reanalysis for circulation features, **no CDS registration needed**. |
| **Ensemble API** | `https://ensemble-api.open-meteo.com/v1/ensemble` | Optional: member spread from GFS ENS / ICON EPS. Check historical depth before depending on it. |

**Verify parameter names against `https://open-meteo.com/en/docs/previous-runs-api` before coding.** The pattern is `<variable>_previous_dayN`. Expect `precipitation_sum_previous_day1 … _day7` under `daily=`, and hourly equivalents under `hourly=`.

**★ THE ONE DETAIL THAT WILL SILENTLY DESTROY OUR SKILL SCORES IF WE GET IT WRONG:**

IMD's rainfall day runs **0830 IST to 0830 IST** (i.e. 03:00 UTC to 03:00 UTC). Open-Meteo's `daily=` sums use local midnight. **Therefore: always request `hourly=precipitation` with `timezone=GMT`, and aggregate 03:00 UTC (day D) → 03:00 UTC (day D+1) ourselves.** A 3-hour misalignment shifts a whole convective peak into the wrong day and will quietly cost us 15–20% of apparent skill. Implement this once, in `src/ingest/time_align.py`, and unit-test it in Phase 1.

**Models to request:** `gfs_seamless` (or `gfs_global`), `icon_seamless`, `ecmwf_ifs025`, and `ecmwf_aifs025_single` for Level 2.

**Query volume:** ~750 district centroids × 1 call each covering a multi-year span. Well inside the free tier. Cache every response to `data/raw/openmeteo/` keyed by a hash of the query so re-runs never re-hit the network.

### 4.3 Synoptic fields for regime detection — Open-Meteo mesh (keyless) ★ CRITICAL PATH

We need gridded fields to compute vorticity and moisture transport, but we want to avoid CDS registration for the MVP. Solution: **sample a coarse regular mesh through the same point APIs.**

- **Mesh:** 2.0° spacing, 5°N–40°N × 60°E–100°E → **18 × 21 = 378 points**.
- **Variables per point** (verify names in the Open-Meteo pressure-level docs):
  `pressure_msl`, `wind_speed_850hPa`, `wind_direction_850hPa`, `wind_speed_200hPa`, `wind_direction_200hPa`, `geopotential_height_500hPa`, `relative_humidity_850hPa`, `temperature_850hPa`, `precipitation`, `total_column_integrated_water_vapour` (if available; otherwise derive a proxy from 850/700 hPa q).
- **Two versions of the mesh:**
  - **Analysis mesh** from the ERA5 Archive API → used to build regime *labels*.
  - **Forecast mesh** from the Previous Runs API at each lead → used as classifier *features*.
- Convert wind speed/direction to u, v on ingest. Compute relative vorticity by centred finite differences on the mesh in `src/features/synoptic.py`.

**Upgrade path (Level 3, optional):** ERA5 from Copernicus CDS via `cdsapi` (free registration, queue delays), or IMDAA 12 km from NCMRWF RDS (free registration). Shortcut to IMDAA without the portal: **BharatBench** on Kaggle (`maslab/bharatbench`, IMDAA-derived, ML-ready NetCDF) or **IndiaWeatherBench** on Hugging Face (`tungnd/IndiaWeatherBench`, IMDAA 12 km, Zarr + HDF5, CC BY-NC-SA 4.0). Do not put these on the critical path.

### 4.4 Regime label catalogues — Zenodo (no account)

These are how our rule-based detectors get *calibrated against published science* instead of being invented.

| Catalogue | Source | Coverage | Role |
|---|---|---|---|
| **Monsoon LPS tracks** (Vishnu, Boos, Ullrich, O'Brien) | Zenodo DOI `10.5281/zenodo.3890646` | 1979–2019, 5 reanalyses, TempestExtremes-based, tuned to match subjectively analysed LPS | Tune the LPS detector's vorticity/closed-contour thresholds |
| **Western Disturbance tracks v6** (ERA5) | Zenodo record `22101482` | 1950–2025, 16,298 trajectories, 460,411 three-hourly points | Tune the WD detector |

**★ The important architectural consequence:** neither catalogue covers 2026. So the pipeline is three layers, and this is a strength, not a weakness:

```
Published catalogue (1979–2019 / 1950–2025)
        ↓ used to TUNE thresholds of
Rule-based detector on ERA5 analysis fields  ──► REGIME LABELS for 2024–2026
        ↓ labels supervise
LightGBM classifier on FORECAST fields  ──────► REGIME PREDICTION at lead 1–5
```

Say this out loud in the demo: *"Our regime labels are not invented. They come from a detector tuned to reproduce a peer-reviewed, published track catalogue."*

### 4.5 Active / break monsoon index — computed, not downloaded

Standard published definition: active (break) spells are periods when the **standardised rainfall anomaly averaged over the monsoon core zone** exceeds +1 (falls below −1) for **at least three consecutive days**. The core monsoon zone is approximately **18°N–28°N, 65°E–88°E**.

Implement in `src/labels/active_break.py`:
1. Area-average IMD 0.25° rainfall over the CMZ box → daily series.
2. Compute day-of-year climatological mean and SD from 1981–2020 (smoothed with a 5-day window).
3. Standardised anomaly; apply ±1.0 threshold with 3-day persistence.
4. Also produce a **forecast-side** version from the forecast mesh precipitation, which becomes a classifier feature.
5. Provide `±0.5` as a configurable sensitivity option (some studies use the looser threshold to capture more spells) and report which was used.

### 4.6 District boundaries

- **Primary:** an open India district polygon set (e.g. the Datameet `maps` repository, ODbL). **Fallback:** GADM level-2 (non-commercial use only — fine for SIH, note the licence).
- **★ Freeze a district master table in Phase 1 and never change it.** India's district list changes constantly (new districts in Chhattisgarh, MP, Telangana, Ladakh…). Create `data/static/districts_master.csv` with: `district_id` (our stable key), `district_name`, `state`, `imd_obj_id` (nullable, filled when the IMD API is available), `lat_centroid`, `lon_centroid`, `area_km2`, `zone_code`, `subdivision`.
- Name harmonisation is a real half-day of work. Write `src/static/harmonise_names.py` with an explicit alias table in `data/static/district_aliases.csv`. Do not fuzzy-match silently — log every non-exact match for human review.
- Simplify polygons for the web with `mapshaper`/`topojson` to **< 1.5 MB total** for the whole country. The map must not be the slow part of the demo.

### 4.7 Terrain

- **Primary:** ETOPO 2022 60-arcsec global relief NetCDF from NOAA NCEI — single file, no auth, ~2 km resolution. Sufficient for a district-scale orographic exposure index.
- **Upgrade:** SRTM 90 m (CGIAR-CSI) or Copernicus DEM GLO-90 via OpenTopography (free key) if orographic features prove weak.
- Derived static fields per district: `mean_elev`, `max_elev`, `elev_range`, `mean_slope`, `dominant_aspect`, `dist_to_coast_km`, `coast_normal_bearing`.

### 4.8 IMD public API — the strategic wildcard ★ DO THIS ON DAY 1

- **Portal:** `https://api.imd.gov.in/public/index.php` — has a **Register** flow. **Reference:** `https://api.imd.gov.in/public/api_reference.html`.
- **Verified status:** calling `https://api.imd.gov.in/api/v1/districtrainfall?id=164` without credentials returns **HTTP 401**. Registration is required.
- **Endpoints that matter to us:**
  - `/api/v1/districtwarning` — district Day-1..Day-5 warning codes + colours. Codes: `2` Heavy Rain, `16` Very Heavy Rain, `17` Extremely Heavy Rain. **This is the operational benchmark for USP 3.**
  - `/api/v1/districtrainfall` — daily actual / normal / departure per district. Independent check on our grid-derived district truth.
  - `/api/v1/state_district_rainfall_forecast` — official 5-day district rainfall distribution forecast.
  - `/api/v1/basinqpf` — river-basin QPF, Day-1..Day-5.
  - `/api/v1/aws_data` — AWS/ARG observations, useful as a sanity layer.
- **Critical property: the API serves *current* state, not history.**

> **ACTION, TODAY, BEFORE ANY CODE:** two teammates register at `api.imd.gov.in`. Simultaneously, deploy `scripts/harvest_imd.py` as a daily cron (GitHub Actions on a schedule works and is free). Even if credentials are pending, harvest anything reachable and log failures. By the grand finale we will hold a **months-long archive of operational IMD district warnings that no other team has.** This single background job is the highest return-on-effort action in the entire project.
>
> **If registration is denied:** fall back to manually extracting warnings from the public IMD daily bulletin PDFs (`mausam.imd.gov.in`) for our ~15 case-study dates. Sufficient for the demo, insufficient for national statistics. Nothing else in the system depends on it.

### 4.9 Live forecast source for the demo — ECMWF Open Data

ECMWF opened its entire real-time catalogue on 1 October 2025 under **CC-BY-4.0**, with a free 25 km subset and an AWS S3 mirror (`s3://ecmwf-forecasts`, `eu-central-1`). Includes IFS HRES, ENS (51 members) and **AIFS**, the AI model.

- Use `pip install ecmwf-opendata` for the client.
- **Required attribution wording** (put it in the dashboard footer verbatim): *"This service is based on data and products of the European Centre for Medium-Range Weather Forecasts (ECMWF). Source: www.ecmwf.int. This ECMWF data is published under a Creative Commons Attribution 4.0 International (CC BY 4.0)."*
- Role: one genuinely live run in the demo, clearly labelled. **The demo must not depend on it.**

### 4.10 Optional deep-record source — GEFSv12 Reforecast on AWS

GEFSv12 reforecasts spanning **2000–2019**, free and anonymous at `s3://noaa-gefs-retrospective` (`https://noaa-gefs-retrospective.s3.amazonaws.com/index.html`), GRIB2 on a 0.25° grid, 3-hourly for the first 10 days.

Use **only** if regime-specific sample counts prove too thin after Level 2 verification. Subset with `.idx` byte-range requests (`herbie` or manual) — never download whole global files. Treat as a **separate experiment**, not merged with the operational data, because the model version and ensemble configuration differ.

### 4.11 Data sufficiency — the honest number

Put this on the data slide. Judges respect a team that knows its own sample size.

| | Value |
|---|---|
| Districts (all-India) | ~750 |
| JJAS days per season | 122 |
| Seasons with lead-stratified pairs (2024–2026) | 3 |
| **District-days per lead time** | **~275,000** |
| Fraction exceeding 64.5 mm (heavy) | ~1–3% |
| **Heavy-rain events available per lead time** | **~3,000–8,000** |
| Effective independent events (≈10-day synoptic decorrelation) | order 10³ |

**Consequence, and it drives every modelling choice in §6:** this is ample for gradient-boosted trees and small parametric neural networks with district embeddings. It is **not** enough to train a U-Net, transformer or diffusion model from scratch. Anyone proposing those is proposing to overfit.

---

## 5. THE EXACT FEATURES

All features are computed per `(district_id, valid_date, lead_day)`. Target is IMD district-mean rainfall for that rainfall-day.

### 5.1 Raw forecast features (per NWP model m ∈ {gfs, icon, ifs})

| Feature | Definition |
|---|---|
| `pr_{m}` | Forecast rainfall for the district, 03 UTC–03 UTC, area-weighted over grid cells in the polygon |
| `pr_{m}_nbr_mean` | Mean over the district and its first-order neighbouring districts |
| `pr_{m}_nbr_max` | Max over the same neighbourhood — **this is the single most important predictor of heavy rain**, because displacement error dominates at district scale |
| `pr_{m}_nbr_p90` | 90th percentile over the neighbourhood grid cells |
| `pr_{m}_grad` | Magnitude of the spatial gradient of forecast rainfall at the district — high values flag a sharp rain edge, i.e. high displacement risk |
| `pr_{m}_prev_day` | Same district, previous valid day (antecedent wetness proxy) |
| `pr_{m}_3day_sum` | Rolling 3-day forecast total |

### 5.2 Multi-model / ensemble features

| Feature | Definition |
|---|---|
| `mm_mean`, `mm_median` | Across the available models |
| `mm_spread` | SD across models — our cheap uncertainty proxy |
| `mm_range` | max − min |
| `mm_pop` | Fraction of models (or ensemble members) with > 0.2 mm |
| `mm_agree_heavy` | Fraction of models exceeding 64.5 mm |
| `ens_sd`, `ens_p90`, `ens_p99` | Level 2 only, from ECMWF ENS 51 members |

### 5.3 Regime probability features ★ the USP

`p_ACTIVE`, `p_BREAK`, `p_LPS`, `p_WD`, `p_EASTERLY_COASTAL`, `p_WEAK` — six floats summing to 1, produced by the regime classifier from the **forecast** mesh at that lead time.

Plus derived:
- `regime_entropy` — Shannon entropy of the vector; high entropy = transitional day = we should widen the predictive distribution
- `regime_argmax` — categorical, for grouping and display only, **never** used to switch logic
- `p_LPS_trend` — change in `p_LPS` from lead+1 to lead (approaching vs departing system)

### 5.4 Synoptic features at the district (interpolated from the forecast mesh)

| Feature | Why |
|---|---|
| `vort850_max_300km` | Max 850 hPa relative vorticity within 300 km — LPS proximity |
| `vort850_dist_km`, `vort850_bearing` | Distance and bearing to the nearest vorticity maximum: **rainfall in a monsoon depression is strongly asymmetric, concentrated in the SW quadrant** — this pair lets the model learn that asymmetry, and it is a genuinely physics-informed feature worth calling out in the PPT |
| `mslp_anom` | MSLP minus day-of-year climatology |
| `u850`, `v850`, `wspd850` | Low-level flow |
| `shear_u` | u200 − u850, monsoon strength proxy |
| `rh850` | Moisture |
| `ivt_mag`, `ivt_dir` | Integrated vapour transport magnitude and direction |
| `z500_anom` | 500 hPa geopotential anomaly — WD signature |
| `cmz_rain_anom_fcst` | Forecast standardised rainfall anomaly over the core monsoon zone — the forecast-side active/break index |

### 5.5 Terrain-interaction features ★ replaces "orographic regime"

| Feature | Definition |
|---|---|
| `orog_exposure` | Static: mean slope × windward-facing fraction |
| `upslope_flux` | **Dynamic:** dot product of the 850 hPa wind vector with the district's mean terrain gradient vector, × `rh850`. Positive and large = active orographic forcing **right now**. |
| `coast_onshore_flux` | Dot product of the 850 hPa wind with the coast-normal bearing, × `ivt_mag`, gated by `dist_to_coast_km` |
| `dist_to_coast_km`, `mean_elev`, `elev_range` | Static |

`upslope_flux` is the feature that lets one national model handle Mahabaleshwar and Cherrapunji without a separate "orographic regime". It is cheap, physical, and explains well in SHAP.

### 5.6 Climatological / calendar features

`doy_sin`, `doy_cos`, `clim_mean_district_doy`, `clim_p90_district_doy`, `clim_p99_district_doy`, `days_since_local_onset`, `lead_day`, `district_id` (categorical → embedding or LightGBM native categorical).

### 5.7 Feature hygiene — non-negotiable

- **Nothing observed after the initialisation time may ever be a feature.** Build `tests/test_no_leakage.py` that asserts, for every feature column, that its source timestamp ≤ init time. Run it in CI.
- The observation-derived active/break index is a **label**, never a feature. Only the forecast-derived version enters the model.
- **★ Leakage canary:** the regime classifier's accuracy **must degrade with lead time**. If day-1 and day-5 accuracy are similar, we have leaked. Plot accuracy vs lead in every training run and fail the build if the day-5/day-1 ratio exceeds 0.95.

---

## 6. THE EXACT MODEL

Two stages, gated. Build Stage A first — it is independently demonstrable and de-risks everything.

### 6.1 Stage A — Regime classifier

```
Model:      LightGBM, objective="multiclass", num_class=6
Input:      ~35 synoptic + calendar features from the FORECAST mesh at lead L
Output:     6 calibrated probabilities
Params:     n_estimators=600, learning_rate=0.05, num_leaves=63,
            min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42
Calibration: isotonic regression per class, fitted on the validation season only
Training:   one model per lead day (5 small models) — simpler and more accurate
            than one model with lead as a feature; each trains in under a minute
```

**Classes:** `ACTIVE`, `BREAK`, `LPS`, `WD`, `EASTERLY_COASTAL`, `WEAK_TRANSITION`.

**Labels** come from `src/labels/` detectors (§4.4–4.5), applied to the ERA5 analysis mesh. Labels are **soft where they overlap**: a day inside an LPS envelope during an active spell gets `{LPS: 0.6, ACTIVE: 0.4}`. Train with soft targets (LightGBM: replicate rows with sample weights; or switch this head to a small PyTorch MLP with soft cross-entropy — the MLP is cleaner and is the Level-2 choice).

**Validation requirement:** report per-class precision/recall/F1 **with support counts**, a confusion matrix, and accuracy vs lead time. Expect WD to be rare in JJAS (~2–5% of days) — merge or softly weight, do not oversample into fake confidence.

### 6.2 Stage B — Regime-gated rainfall corrector

**Level 1 implementation (must exist by week 3):**

```
Three LightGBM quantile regressors:  objective="quantile", alpha ∈ {0.1, 0.5, 0.9}
Three LightGBM binary classifiers:   targets  y > 64.5,  y > 115.6,  y > 204.5
Regime probabilities are ordinary input features.
Monotonicity constraint: +1 on mm_mean and pr_gfs_nbr_max
   (more forecast rain must never produce less corrected rain — this single
    constraint prevents the embarrassing demo failure where the corrected
    value drops as the raw forecast rises)
```

**Level 2 implementation — CSGD Mixture of Regime Experts (the real thing):**

Predictive distribution: **censored, shifted gamma (CSGD)** — a gamma distribution allowed probability mass at negative values and left-censored at zero, so it represents the point mass at zero rainfall plus a skewed positive tail with three parameters. This is the established parametric choice for ensemble precipitation post-processing (Scheuerer & Hamill 2015).

```
Architecture (PyTorch, CPU):

  x  ──► shared trunk: Linear(F→128) → GELU → LayerNorm → Linear(128→64) → GELU
         + district embedding (750 → 16)  concatenated into the trunk input
                    │
       ┌────────────┴────────────┐
       ▼                         ▼
  gate g(x) = softmax(...)   K=6 expert heads
  (initialised FROM the      each: Linear(64→3) → (μ_k, σ_k, δ_k)
   regime classifier's       with softplus on μ,σ and a bounded shift on δ
   probabilities, then
   fine-tuned end-to-end)
                    │
                    ▼
     Mixture: parameters combined as a probability mixture of CSGDs
              (mix the CDFs, not the parameters — mixing parameters is
               mathematically wrong and will bite you on bimodal days)

  Loss: CRPS, computed numerically as
        CRPS = ∫₀^500 (F(x) − 1{x ≥ y})² dx
        on a fixed 501-point grid (1 mm spacing), fully differentiable.
        Use the numerical form, not a hand-derived closed form — it is
        exact enough, impossible to get subtly wrong, and takes 20 minutes
        to implement instead of two days.

  Tail emphasis: train with plain CRPS first. Then fine-tune with
        threshold-weighted CRPS (weight w(x) rising above 60 mm) and
        KEEP IT ONLY IF the heavy-rain CSI improves without the overall
        CRPS degrading by more than 3%. Report both.

  Size: ~60k parameters. Trains in minutes on a laptop.
```

**Gate initialisation from the classifier is the key trick.** It means the gate starts out meteorologically meaningful and stays interpretable after fine-tuning, instead of collapsing into six anonymous clusters. If the fine-tuned gate drifts far from the classifier (KL divergence above a threshold), freeze it — an uninterpretable gate destroys USP 2.

### 6.3 Required ablations (these ARE the evidence for the USP)

Run all five. Put the table on a slide.

| # | Configuration | What it proves |
|---|---|---|
| 1 | Raw NWP (best single model) | The floor |
| 2 | Multi-model mean | The naive ensemble |
| 3 | Global quantile mapping | The conventional post-processor |
| 4 | **Our model, gate removed** (regime features zeroed) | Isolates the value of regime information |
| 5 | **Our model, full** | The claim |

**If (5) does not beat (4) on per-regime heavy-rain CSI, the USP is not real and we must say so.** Find that out in week 5, not in the finale. If the gain is genuinely small, pivot the headline USP to the Atlas (USP 1), which does not depend on the gate working — this is exactly why the Atlas is ranked first.

### 6.4 Decision layer — probability to colour

```python
# IMD operational thresholds (mm / 24 h)
HEAVY, VERY_HEAVY, EXTREMELY_HEAVY = 64.5, 115.6, 204.5
# Colours: GREEN (no warning) / YELLOW (be updated) /
#          ORANGE (be prepared) / RED (take action)

# Cost-loss decision rule: act when P(event) > C/L
# C = cost of preparing, L = loss if unprepared.
# Default alpha = C/L = 0.30 (miss-averse, disaster-management posture).
# UI slider exposes 0.05 (very miss-averse) to 0.60 (false-alarm-averse).
# Never a hard-coded 0.5.

# Safety guard: the assigned colour may never be LOWER than the colour the
# raw multi-model mean would have produced, unless P(exceed) < alpha/2.
# Rationale: a post-processor that quietly downgrades a real warning is
# operationally unacceptable, and one live example of that on stage ends us.
```

Add 1-step hysteresis across consecutive runs so colours do not flicker.

---

## 7. THE EXACT TRAINING STRATEGY

### 7.1 Splits — by whole season, never random

| Split | Seasons |
|---|---|
| Train | JJAS 2024 (+ 2021–2023 lead-1 pairs from the Historical Forecast API for extra volume at lead 1) |
| Validation | JJAS 2025 — hyperparameters, calibration, early stopping |
| **Test** | **JJAS 2026 — touched exactly once, at the very end** |

Enforce it in code: `src/train/splits.py` exposes `get_split(season)` and raises if any training routine requests the test season outside `scripts/final_eval.py`. Print a loud banner when the test set is opened.

For spatial-generalisation claims, additionally run **leave-one-zone-out** CV and report it separately.

### 7.2 Leakage traps — enumerate these on a slide, it signals maturity

1. **Temporal autocorrelation.** Consecutive days inside one LPS are near-duplicates; random splits inflate skill dramatically. Season-blocked splits only.
2. **Spatial autocorrelation.** Neighbouring districts on the same day are not independent. Never random-split across districts.
3. **Observation-derived regime index used as a feature.** Forecast-side only. Canary test in §5.7.
4. **Climatology fitted on the full record including the test season.** Fit normals on training years only.
5. **Quantile-mapping CDFs fitted on all years.** Same fix — and the QM baseline must get the same treatment, or we are beating a strawman.
6. **Hyperparameter tuning on the test season.** Validation only.
7. **Reforecast/operational mixing.** GEFSv12 reforecast (5 members, 00 UTC, frozen model version) is a different animal from operational data. Separate experiment, separately reported.

### 7.3 Class and event imbalance

- Do **not** oversample heavy-rain days. It corrupts probability calibration, which is the deliverable.
- Instead: sample weights proportional to `1 + log1p(observed_rain)` in the regression heads, and `scale_pos_weight` in the threshold classifiers, then **recalibrate with isotonic regression** on the validation season.
- Report reliability diagrams before and after calibration — the improvement is a good slide.

### 7.4 Recipe

```
1. Build features → data/processed/features_{season}.parquet
2. Fit regime classifier per lead on train; calibrate on validation
3. Generate regime probabilities for all splits (no leakage: the classifier
   never saw validation/test)
4. Fit Level-1 LightGBM corrector; record all metrics
5. Fit Level-2 CSGD-MoE; gate initialised from step 2
6. Run all five ablations (§6.3)
7. Fit isotonic calibration of exceedance probabilities on validation
8. ONE evaluation on the test season → results/final/
9. Build the Regime Error Atlas from the test-season results
10. Freeze models to models/frozen/ with a MANIFEST.json recording
    git SHA, data versions, hyperparameters, and every metric
```

`make all` must execute steps 1–10 end to end and reproduce every number in the presentation.

---

## 8. THE EXACT BACKEND ARCHITECTURE

```
Python 3.11
FastAPI + uvicorn                 HTTP API
Pydantic v2                       request/response schemas
DuckDB                            query engine over Parquet
Parquet (pyarrow)                 all tabular storage
GeoJSON / TopoJSON                district geometry
xarray + netCDF4 + rioxarray      gridded data
geopandas + shapely               spatial ops
LightGBM, scikit-learn            Stage A + Level-1 Stage B
PyTorch (CPU)                     Level-2 CSGD-MoE
SHAP                              explainability
faiss-cpu OR sklearn.neighbors    analog retrieval
httpx + tenacity                  resilient cached API clients
typer                             CLI
Docker + docker-compose           packaging
pytest                            tests
```

**Why DuckDB + Parquet and not PostgreSQL/PostGIS:** the entire dataset is a few hundred MB of immutable, append-only analytical tables. DuckDB queries Parquet directly with zero server, zero setup, sub-100 ms responses, and **cannot fail to start during a demo**. PostGIS is the right answer for a Level-3 production deployment with concurrent writers and live spatial queries — say exactly that when asked, and have `docker-compose.prod.yml` with a PostGIS service ready to show. Choosing the simple correct tool and being able to justify it is a better answer than choosing the impressive one.

**Everything is precomputed.** A nightly (or on-demand) batch job writes `predictions.parquet`. The API only reads and filters. No model inference happens inside a request handler except for the interactive cost-loss recomputation, which is pure arithmetic on stored probabilities.

**Caching:** every external HTTP call goes through `src/ingest/http_cache.py`, which keys on a SHA-256 of the full request URL and writes to `data/raw/_cache/`. Re-running the pipeline offline must work.

---

## 9. THE EXACT FRONTEND

### 9.1 Stack

`Vite + React 18 + TypeScript`, `MapLibre GL JS` (vector tiles, smooth on a projector), `Recharts` for charts, `Tailwind CSS`, `TanStack Query` for data fetching. No global state library — `useState` and query cache are enough.

### 9.2 Design direction

The subject matter is an **operational warning desk**, not a SaaS analytics product. The design must look like something that belongs on a forecaster's second monitor at 2 a.m.

**The palette is dictated by policy, and that is the design anchor.** IMD's four warning colours are mandated and unambiguous: green (no warning), yellow (be updated), orange (be prepared), red (take action). Therefore: **those four are the only saturated colours permitted anywhere in the interface.** Everything else — chrome, charts, typography, the map basemap — is a restrained cool-neutral scale. When a district turns red, it must be the only red thing on screen. This is the single governing principle; everything else follows from it.

```
Tokens
  ink        #14181D   text, map outlines
  slate      #3A4450   secondary text
  mist       #8A96A4   tertiary, axis labels
  paper      #F2F4F6   app background
  card       #FFFFFF   surfaces
  rule       #DDE3E9   1px dividers
  --- policy colours, used ONLY for warning state ---
  green      #1B8A3F
  yellow     #F2C200
  orange     #F07C00
  red        #D31F26
Type
  IBM Plex Sans            everything
  IBM Plex Sans Devanagari district names in Devanagari where available
  Tabular figures ON for every number in a table or readout —
  rainfall values must align vertically when scanned down a column.
  Scale: 12 / 14 / 16 / 20 / 28 / 40. Sentence case. No all-caps labels.
Motion
  One orchestrated moment only: when the lead-time slider moves, the
  choropleth cross-fades over 220ms and the regime stack-chart animates
  its transition. Nothing else animates. No hover lifts, no card shadows,
  no entrance fades.
```

Avoid: rounded cards with identical radii and soft grey shadows; gradient washes; an all-caps eyebrow above every heading; a "→" glued to link text; monospace as decoration. Tabular figures are used because numbers must align, not because monospace looks technical.

### 9.3 Layout

```
┌──────────────────────────────────────────────────────────────────────────┐
│ RAAHAT   Regime-aware rainfall post-processing                           │
│ [ REPLAY: 29 Jul 2024, 00 UTC ]   Lead ● 1 ─ 2 ─ 3 ─ 4 ─ 5   [Live run] │  ← replay banner
├───────────────────────────────────────┬──────────────────────────────────┤  is always
│                                       │  REGIME OUTLOOK                  │  visible and
│                                       │  ████████░░░░  Active      62%   │  unmistakable
│          DISTRICT MAP                 │  ██████░░░░░░  LPS         28%   │
│          (MapLibre choropleth)        │  ██░░░░░░░░░░  Break        7%   │
│                                       │  ░░░░░░░░░░░░  Other        3%   │
│   toggle: [Raw] [Corrected] [Δ]       │  ┌────────────────────────────┐  │
│           [P>64.5] [P>115.6]          │  │ regime probability × lead  │  │
│                                       │  │ (stacked area chart)       │  │
│                                       │  └────────────────────────────┘  │
│                                       ├──────────────────────────────────┤
│                                       │  WAYANAD, Kerala                 │
│                                       │  Raw GFS            96 mm        │
│                                       │  RAAHAT median     180 mm        │
│                                       │  P(>64.5)           0.94         │
│                                       │  P(>115.6)          0.71         │
│                                       │  P(>204.5)          0.41         │
│                                       │  Suggested warning  ● RED        │
│                                       │  ┌── why ────────────────────┐   │
│                                       │  │ predictive distribution    │   │
│                                       │  │ + top-3 drivers (SHAP)     │   │
│                                       │  │ + nearest analog days      │   │
│                                       │  └────────────────────────────┘   │
├───────────────────────────────────────┴──────────────────────────────────┤
│ Tabs: District table │ Regime Error Atlas │ Verification │ About the data │
└──────────────────────────────────────────────────────────────────────────┘
Footer: Weather data by Open-Meteo.com (CC BY 4.0) · Observations: IMD Pune ·
        [full ECMWF attribution string]
```

### 9.4 Required screens

1. **Forecast** — the layout above. Map, regime panel, district detail.
2. **District table** — sortable, filterable by state/colour/probability, CSV export. This is what a forecaster actually wants; the map is what a judge wants. Build both.
3. **Regime Error Atlas** — the heatmap. Regime rows × zone columns, cell = raw bias, annotation = ΔCSI. Clicking a cell filters the verification tab. **This screen must be perfect; it is USP 1.**
4. **Verification** — the ablation scorecard, reliability diagrams, POD/FAR/CSI vs threshold, skill vs lead time. Include at least one cell where we did not improve, visibly.
5. **About the data** — every source, licence, access method, and the sample-size table from §4.11. Pre-empts half the judge questions.

### 9.5 Robustness

- The frontend must render fully against `data/synthetic/mock_api/*.json` with the backend switched off. Build it that way from day 1 (`VITE_USE_MOCK=true`).
- Every request has a 3-second timeout and a visible, non-apologetic error state that explains what to do ("Backend not reachable. Showing cached run from 29 Jul 2024.").
- Empty states are invitations: "Select a district on the map to see its forecast breakdown."
- Test at 1280×720 — that is the projector resolution in most SIH venues, not your 1440p laptop.

---

## 10. THE EXACT DEMO FLOW

### 10.1 Events (all inside the archive window, all regime-contrasting)

| # | Event | Date | Regime showcased | The point |
|---|---|---|---|---|
| 1 | **Wayanad, Kerala** | 30 Jul 2024 | Active + strong orographic + offshore vortex | Extreme under-forecast; terrain modifier fires |
| 2 | **Central India LPS** | pick a 2025/2026 Vidarbha–Odisha depression | LPS | Gate switches; SW-quadrant asymmetry visible |
| 3 | **WD × monsoon, Himachal/Uttarakhand** | pick from Aug 2024/2025 | WD + Active simultaneously | **Proves soft gating**; hard switching would be wrong here |
| 4 | **Break-monsoon false-alarm case** | pick any | Break | We correctly *downgrade* a warning |

Event 1 involved loss of life. Present it factually and without dramatisation — no imagery, no death tolls on slides. State the meteorology and the forecast error. Judges notice restraint.

Event 4 is the one nobody else will show and everybody will remember: *"Fewer unnecessary red alerts is also value."*

### 10.2 Six-minute script

**0:00–0:45 — The hook. One number, no title slide.**
> "On 30 July 2024, the operational model forecast 96 mm for Wayanad district. It recorded over 200. That error is not random — it is the same error the model makes *every time* this pattern occurs. Our system knows that."

Show one scatter plot: raw forecast vs observed, points coloured by regime, with the regime clusters visibly separated on different lines. This single plot is the entire thesis.

**0:45–1:30 — The gap.**
IMD already runs a district multi-model ensemble. Published skill: day-1 correlation 0.58 with observed district rainfall, RMSE 12.7 mm/day. Published post-processing over India **under-performed specifically during the monsoon**. *"Correction is regime-blind. The monsoon is not."*

**1:30–2:15 — The insight. The Atlas.**
Show the heatmap. Say nothing for five seconds. Then: *"This has not been measured for India at district level before. This is our starting point, not our conclusion."*

**2:15–3:30 — The system, live, in replay mode.**
Banner visible throughout. Scrub the lead slider from 5 → 1: the regime stack shifts, the map recolours, raw orange becomes corrected red. Click Wayanad. Open the explanation panel.

**3:30–4:15 — Truth.**
Reveal observed. Three panels: raw / corrected / observed. Then the Analog Memory line.

**4:15–5:00 — Evidence, not anecdote.**
The ablation scorecard on the held-out test season. Per-regime CSI at 115.6 mm, day-3 lead. The reliability diagram. **Point at the one cell where we did not improve and say so out loud.** This buys more credibility than any other single act in the presentation.

**5:00–5:45 — The false alarm we prevented.**

**5:45–6:00 — The ask.**
> "Every input is free and open. It trains in eleven minutes on a laptop with no GPU. It sits upstream of the Bharat Forecast System, not against it. And we have already started archiving IMD's own district warnings so it can be tested against what actually went out."

### 10.3 Demo robustness

- **Airplane mode is the default.** Rehearse with wifi off. Everything the demo touches is in `data/demo_cache/`.
- Never train live. Never fit anything live. Only load-and-predict.
- Two laptops, both fully provisioned, both tested on an external display.
- A screen-recorded MP4 of the full demo on the desktop, and a PDF of every figure, as tier-3 fallback.
- If showing a genuine live ECMWF run, label it **"Live run — fetched just now"** and have a keyboard shortcut to fall back to replay instantly.
- **Never fake live data.** The replay banner exists so that no judge can later accuse us of implying real-time capability we do not have. Integrity here is also a competitive advantage — mention it.

---

## 11. THE EXACT EVALUATION METRICS

### 11.1 What we report

| Metric | Applied to | Why it belongs |
|---|---|---|
| Mean bias | rainfall, **stratified by regime** | The thing we claim to fix. Conditional, not overall. |
| RMSE, MAE | rainfall | Directly comparable to published IMD MME numbers |
| **CRPS / CRPSS** | predictive distribution | The proper score for a probabilistic forecast; our training loss |
| **Brier score + BSS** | P(>64.5), P(>115.6), P(>204.5) | Direct measure of the heavy-rain deliverable |
| **Reliability diagram + sharpness** | exceedance probabilities | Answers "is 70% really 70%?" — the documented weakness of current Indian operational AI forecasts |
| **POD, FAR, CSI, ETS, Frequency Bias** | thresholded warnings | Required by the PS; the operational currency |
| **FSS** | gridded field, scales 1/3/5/9 cells | Rewards getting the rain roughly in the right place — the honest way to score extremes |
| Lead time where skill > climatology | — | The "so what" number decision-makers understand |
| Latency, memory, training time | system | "<40 ms per district-day, CPU only, no GPU" is a competitive claim |

### 11.2 What we deliberately do NOT report — and why (put this on a slide)

- **Overall accuracy** — meaningless when 90% of district-days are dry.
- **R²** — dominated by the zero mass.
- **MAPE** — undefined at zero rainfall.
- **ROC-AUC as the headline** — AUC is insensitive to calibration, and calibration is our entire point. We report it, we do not lead with it.

Showing that we know which metrics are inappropriate is itself evidence of competence.

### 11.3 Formulas (implement in `src/verify/metrics.py`, unit-tested)

```
Contingency table at threshold t:  a=hits  b=false alarms  c=misses  d=correct negatives

POD  = a / (a + c)
FAR  = b / (a + b)
CSI  = a / (a + b + c)
Bias = (a + b) / (a + c)
a_ref = (a + b)(a + c) / (a + b + c + d)
ETS  = (a − a_ref) / (a + b + c − a_ref)

FSS(n) = 1 − MSE(n) / (MSE_ref(n))
         over n×n neighbourhood fractional fields of forecast and observation

Brier = mean( (p − o)² ),  o ∈ {0,1}
BSS   = 1 − Brier / Brier_climatology

CRPS  = ∫₀^∞ (F(x) − 1{x ≥ y})² dx
        numerically on a 0–500 mm grid at 1 mm spacing
CRPSS = 1 − CRPS / CRPS_climatology
```

### 11.4 The single table that decides whether the project worked

> **Per-regime CSI at 115.6 mm, day-3 lead, held-out test season, against all four baselines, with sample counts printed in every cell.**

Operationally decisive threshold (orange alert), at the lead time where decisions are actually made, on data the model has never seen, stratified along the exact axis the USP claims to exploit. If the regime conditioning is real, this table proves it. If it is not, this table reveals it — and we need to know that in week 5, not in December.

Every cell with n < 30 events must be greyed out and labelled "insufficient sample". Do not report a CSI computed on four events.

---

## 12. THE EXACT ARCHITECTURE DIAGRAM

Reproduce this in the PPT as a clean vector graphic. Four columns, left to right: **Data → Learn → Decide → Deliver**. Do not draw ten boxes; draw these.

```
┌─ DATA (all free, 4 of 6 need no account) ────────────────────────────────┐
│                                                                          │
│  Open-Meteo            IMD Pune              Zenodo            ETOPO /   │
│  Previous Runs         0.25° gridded         LPS tracks        district  │
│  GFS · ICON · IFS      daily rainfall        WD tracks v6      polygons  │
│  leads 1–5             1901–2024             (label tuning)    (static)  │
│        │                     │                     │               │     │
└────────┼─────────────────────┼─────────────────────┼───────────────┼─────┘
         │                     │                     │               │
         ▼                     ▼                     ▼               ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │  INGEST + ALIGN    03 UTC→03 UTC rainfall day (IMD convention)       │
   │                    area-weighted grid → district aggregation         │
   └───────────────┬──────────────────────────────────┬───────────────────┘
                   │                                  │
                   ▼                                  ▼
   ┌───────────────────────────────┐      ┌──────────────────────────────┐
   │  RULE-BASED REGIME DETECTORS  │      │  FEATURE ENGINE              │
   │  tuned to reproduce published │      │  forecast · synoptic ·       │
   │  Zenodo track catalogues      │      │  terrain-interaction ·       │
   │           ↓ LABELS            │      │  climatology                 │
   └───────────────┬───────────────┘      └──────────────┬───────────────┘
                   │                                     │
                   └──────────────┬──────────────────────┘
                                  ▼
                   ┌──────────────────────────────┐
                   │  ① REGIME CLASSIFIER         │   ← predicts regime from
                   │     LightGBM, soft p(6)      │     FORECAST fields, so it
                   │     one model per lead       │     works at lead time
                   └──────────────┬───────────────┘
                                  │  p(regime | lead)
                                  ▼
                   ┌──────────────────────────────┐
                   │  ② MIXTURE OF REGIME EXPERTS │   ← soft gate, never argmax
                   │     6 × CSGD heads           │
                   │     trained on CRPS          │
                   │     → full rainfall PDF      │
                   └──────────────┬───────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            ▼                     ▼                     ▼
   ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────┐
   │ ③ DECISION      │  │ ④ EXPLANATION    │  │ ⑤ VERIFICATION     │
   │  P(>64.5/115.6/ │  │  regime attrib.  │  │  per-regime CSI    │
   │   204.5 mm)     │  │  SHAP drivers    │  │  ETS POD FAR FSS   │
   │  cost–loss →    │  │  analog memory   │  │  Brier reliability │
   │  IMD colour     │  │                  │  │  → REGIME ERROR    │
   │  + safety guard │  │                  │  │     ATLAS          │
   └────────┬────────┘  └────────┬─────────┘  └─────────┬──────────┘
            └────────────────────┼──────────────────────┘
                                 ▼
              ┌────────────────────────────────────────┐
              │  FastAPI  ·  DuckDB/Parquet            │
              │  React + MapLibre forecast desk        │
              │  district map · table · atlas · scores │
              └────────────────────────────────────────┘
```

**Annotate the diagram with these three callouts in the PPT** — they are what the judge should read even if they read nothing else:

- On ① → *"Regime is **forecast**, not diagnosed. That is what makes it usable at day 3."*
- On ② → *"**Soft** gating. Real days are blends. Hard switching breaks at regime boundaries."*
- On ⑤ → *"Verification stratified by regime **is** the product, not just the proof."*

---

## 13. THE EXACT STORAGE STRUCTURE

### 13.1 Layout

```
data/
├── static/
│   ├── districts.geojson              full-resolution polygons
│   ├── districts_simplified.topojson  < 1.5 MB, for the web
│   ├── districts_master.csv           FROZEN in Phase 1
│   ├── district_aliases.csv           name harmonisation table
│   ├── district_terrain.parquet       static terrain features
│   ├── district_neighbours.parquet    first-order adjacency
│   └── synoptic_mesh.csv              378 mesh points
├── raw/
│   ├── imd_grid/rain_{YYYY}.nc
│   ├── openmeteo/{endpoint}/{hash}.json
│   ├── ecmwf/{YYYYMMDD}_{HH}/*.grib2
│   ├── zenodo/lps_tracks.csv, wd_v6_summary.parquet
│   └── imd_api/{YYYY-MM-DD}/districtwarning.json     ← daily harvest
├── interim/
│   ├── district_obs_{season}.parquet
│   ├── district_fcst_{season}.parquet
│   ├── mesh_analysis_{season}.parquet
│   └── mesh_forecast_{season}.parquet
├── processed/
│   ├── regime_labels_{season}.parquet
│   ├── features_{season}.parquet      ← THE DATA CONTRACT
│   └── predictions_{season}.parquet   ← what the API serves
├── demo_cache/                        everything the demo needs, offline
├── synthetic/                         frontend unblocking only
└── _manifest/
    ├── downloads.json                 source, URL, sha256, timestamp, rows
    └── errors.log
```

### 13.2 The data contract — freeze in Phase 1, everyone codes against it

`data/processed/features_{season}.parquet` — one row per `(district_id, valid_date, lead_day)`:

```
district_id            int32    stable key from districts_master.csv
valid_date             date32   the IMD rainfall day (03 UTC → 03 UTC)
lead_day               int8     1..5
init_datetime_utc      ts       forecast initialisation (for leakage audit)
season                 int16    2024 / 2025 / 2026
zone_code              string   'W' / 'C' / 'N' / ...
subdivision            string   IMD meteorological subdivision

-- target
obs_rain_mm            float32  IMD district-mean, NULL if unavailable

-- raw forecasts (§5.1), per model
pr_gfs, pr_gfs_nbr_mean, pr_gfs_nbr_max, pr_gfs_nbr_p90,
pr_gfs_grad, pr_gfs_prev_day, pr_gfs_3day_sum          float32
pr_icon_*, pr_ifs_*                                     float32

-- multi-model (§5.2)
mm_mean, mm_median, mm_spread, mm_range, mm_pop, mm_agree_heavy   float32
ens_sd, ens_p90, ens_p99                                float32  (nullable)

-- regime probabilities (§5.3)
p_ACTIVE, p_BREAK, p_LPS, p_WD, p_EASTERLY_COASTAL, p_WEAK        float32
regime_entropy, p_LPS_trend                             float32
regime_argmax                                           string

-- synoptic (§5.4)
vort850_max_300km, vort850_dist_km, vort850_bearing,
mslp_anom, u850, v850, wspd850, shear_u, rh850,
ivt_mag, ivt_dir, z500_anom, cmz_rain_anom_fcst         float32

-- terrain interaction (§5.5)
orog_exposure, upslope_flux, coast_onshore_flux,
dist_to_coast_km, mean_elev, elev_range                 float32

-- climatology / calendar (§5.6)
doy_sin, doy_cos, clim_mean_district_doy,
clim_p90_district_doy, clim_p99_district_doy,
days_since_local_onset                                  float32

-- labels (TRAINING ONLY — assert absent at inference)
label_regime                                            string
label_regime_soft                                       list<float32>[6]
```

`data/processed/predictions_{season}.parquet`:

```
district_id, valid_date, lead_day, init_datetime_utc,
raw_mm, corrected_p10, corrected_median, corrected_p90,
p_gt_64_5, p_gt_115_6, p_gt_204_5,
csgd_mu, csgd_sigma, csgd_delta,
p_ACTIVE..p_WEAK, regime_argmax,
colour_code (0..3), colour_alpha_used,
shap_top1_feature, shap_top1_value,
shap_top2_feature, shap_top2_value,
shap_top3_feature, shap_top3_value,
analog_date_1, analog_obs_1, analog_date_2, analog_obs_2,
model_version, run_id
```

Partition both by `season` and `lead_day`. DuckDB will use the partitions; queries stay under 100 ms.

### 13.3 Model registry

```
models/frozen/{run_id}/
  regime_clf_lead{1..5}.txt        LightGBM
  regime_calibrators.pkl
  corrector_lgbm_q{10,50,90}.txt
  corrector_threshold_{64,115,204}.txt
  csgd_moe.pt
  feature_list.json
  MANIFEST.json   git SHA, data manifest hashes, hyperparameters,
                  train/val/test seasons, ALL metrics, timestamp
```

Nothing goes in the presentation that is not traceable to a `MANIFEST.json`.

---

## 14. THE EXACT APIs

Base `/api/v1`. All responses JSON. All errors `{"error": {"code": str, "message": str}}`.

```
GET  /health
     → {status, model_version, data_through, mode: "replay"|"live"}

GET  /districts
     → TopoJSON FeatureCollection (simplified). Cache 24 h.

GET  /events
     → [{event_id, title, date, regime, zone, description, default_lead}]
       The curated replay events for the demo.

GET  /forecast?date=YYYY-MM-DD&lead=3&layer=corrected&alpha=0.30
     layer ∈ raw | corrected | delta | p64 | p115 | p204 | colour
     → {date, lead, layer, model_version, mode,
        values: [{district_id, value, colour_code, regime_argmax}]}
       One lightweight array for choropleth rendering. Keep under 200 KB.

GET  /district/{district_id}?date=&lead=&alpha=
     → {district: {...}, raw: {...per model}, corrected: {p10, median, p90},
        exceedance: {p64, p115, p204}, distribution: {x: [...], cdf: [...]},
        regime: {probs, argmax, entropy},
        explanation: {drivers: [{feature, label, contribution, direction}],
                      narrative: "..."},
        analogs: [{date, similarity, observed_mm, outcome_note}],
        warning: {suggested_colour, alpha_used, raw_colour, guard_applied}}

GET  /regime?date=&lead_max=5
     → {timeline: [{lead, probs:{...}}],
        national_field: [{mesh_id, lat, lon, argmax, probs}]}

GET  /atlas?metric=bias&threshold=115.6&lead=3
     metric ∈ bias | rmse | csi | pod | far | ets | crpss
     → {cells: [{regime, zone, lead, raw_value, corrected_value,
                 delta, n_events, n_sufficient: bool}]}

GET  /verification?season=2026&lead=3&threshold=115.6&stratify=regime
     → {models: [{name, rmse, mae, bias, crps, crpss,
                  pod, far, csi, ets, bias_score, brier, bss}],
        reliability: [{bin_lower, bin_upper, forecast_prob,
                       observed_freq, n}],
        by_regime: [...], notes: [...]}

POST /decision
     body {date, lead, alpha}
     → recomputed colour codes for all districts (pure arithmetic on
       stored probabilities; must return in < 100 ms)

GET  /bulletin/{district_id}?date=&lead=
     → {text_en, text_hi, colour, issued_at}
       Templated. The template is in config/, not in code. An LLM may
       rephrase but must never decide content or colour.

GET  /imd-benchmark?season=&lead=&threshold=
     → comparison of RAAHAT vs archived IMD district warnings, with the
       "areal mean vs isolated places" caveat returned in the payload
       so the UI cannot display the numbers without it.
```

**Every endpoint must respond in under 300 ms from warm cache.** If one does not, precompute more aggressively.

---

## 15. THE EXACT REPOSITORY STRUCTURE

```
raahat/
├── README.md                       this file
├── Makefile                        make data | features | train | eval | atlas | demo | all
├── pyproject.toml
├── docker-compose.yml
├── docker-compose.prod.yml         PostGIS variant, for the "how would you scale" question
├── .github/workflows/
│   ├── ci.yml                      lint + tests + leakage canary
│   └── harvest_imd.yml             ★ daily IMD warning archiver — SET UP ON DAY 1
├── config/
│   ├── datasets.yaml               every URL, variable list, date range
│   ├── regimes.yaml                class names, detector thresholds, colours
│   ├── features.yaml               the canonical feature list
│   ├── model.yaml                  hyperparameters
│   ├── zones.yaml                  zone → subdivision mapping
│   └── bulletin_templates/         en.j2, hi.j2
├── src/raahat/
│   ├── ingest/
│   │   ├── http_cache.py           SHA-keyed on-disk cache, tenacity retries
│   │   ├── imd_grid.py             imdlib wrapper + masking
│   │   ├── openmeteo.py            point + mesh clients
│   │   ├── ecmwf_open.py
│   │   ├── zenodo.py
│   │   ├── imd_api.py              harvester (auth-aware, degrades gracefully)
│   │   └── time_align.py           ★ 03 UTC → 03 UTC. Unit-tested. Critical.
│   ├── static/
│   │   ├── districts.py            master table, polygons, neighbours
│   │   ├── terrain.py              ETOPO → per-district terrain features
│   │   └── harmonise_names.py
│   ├── features/
│   │   ├── district_agg.py         area-weighted grid → district
│   │   ├── synoptic.py             vorticity, IVT, shear on the mesh
│   │   ├── terrain_interaction.py  upslope_flux, coast_onshore_flux
│   │   ├── climatology.py          train-years-only normals and percentiles
│   │   └── build.py                assembles features_{season}.parquet
│   ├── labels/
│   │   ├── active_break.py         Rajeevan-style CMZ index
│   │   ├── lps_detector.py         tuned against the Zenodo LPS catalogue
│   │   ├── wd_detector.py          tuned against WD v6
│   │   ├── easterly_coastal.py
│   │   └── assemble.py             soft multi-label resolution
│   ├── models/
│   │   ├── regime_clf.py
│   │   ├── corrector_lgbm.py       Level 1
│   │   ├── csgd.py                 CDF, numerical CRPS, sampling
│   │   ├── moe.py                  Level 2 gated mixture
│   │   ├── calibrate.py            isotonic
│   │   └── analogs.py
│   ├── decide/
│   │   ├── cost_loss.py
│   │   └── colour.py               thresholds, safety guard, hysteresis
│   ├── explain/
│   │   ├── shap_drivers.py
│   │   └── narrative.py            feature → forecaster-readable sentence
│   ├── verify/
│   │   ├── metrics.py              POD/FAR/CSI/ETS/FSS/Brier/CRPS
│   │   ├── reliability.py
│   │   ├── atlas.py                ★ USP 1
│   │   └── report.py
│   ├── train/
│   │   ├── splits.py               ★ guards the test season
│   │   ├── train_regime.py
│   │   ├── train_corrector.py
│   │   └── ablations.py
│   └── api/
│       ├── main.py
│       ├── routes/
│       ├── schemas.py
│       └── db.py                   DuckDB connection + queries
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── pages/{Forecast,Table,Atlas,Verification,DataNotes}.tsx
│   │   ├── components/{DistrictMap,RegimeStack,DistrictPanel,
│   │   │              DistributionChart,ExplanationCard,AnalogList,
│   │   │              ReliabilityDiagram,AtlasHeatmap,ReplayBanner,
│   │   │              LeadSlider,CostLossSlider}.tsx
│   │   ├── api/client.ts           honours VITE_USE_MOCK
│   │   └── styles/tokens.css
│   └── public/mock/*.json
├── scripts/
│   ├── download_all.py
│   ├── build_demo_cache.py         ★ makes the demo offline-proof
│   ├── harvest_imd.py
│   ├── make_synthetic.py
│   └── final_eval.py               the ONLY file allowed to open the test season
├── notebooks/                      exploration only; nothing in the pipeline imports these
├── tests/
│   ├── test_time_align.py          ★
│   ├── test_no_leakage.py          ★
│   ├── test_district_agg.py
│   ├── test_metrics.py             against hand-computed contingency tables
│   ├── test_csgd.py                CDF monotone in [0,1]; CRPS ≥ 0
│   └── test_api.py
├── results/
│   ├── figures/
│   ├── tables/
│   └── final/
└── docs/
    ├── DATA_NOTES.md               every API-name correction, every caveat
    ├── PROPOSALS.md                ideas deferred, not built
    ├── JUDGE_QA.md                 rehearsed answers
    └── PPT_OUTLINE.md
```

---

## 16. BUILD PHASES WITH ACCEPTANCE TESTS

No phase starts until the previous phase's acceptance test is green.

| Phase | Deliverable | Acceptance test |
|---|---|---|
| **0 — Day 1** | IMD API registration submitted; `harvest_imd.yml` cron live; district master frozen; data contract published | A warning JSON (or a logged 401) appears in `data/raw/imd_api/` for today. `districts_master.csv` exists and never changes again. |
| **1 — Week 1** | Ingest + alignment | `pytest tests/test_time_align.py` passes. `district_obs_2024.parquet` exists for 190 districts × 122 days with < 2% missing. Manual spot-check: three known heavy-rain dates match published IMD figures within 15%. |
| **2 — Week 2** | Features + baselines | `features_2024.parquet` matches the contract exactly. Raw-GFS RMSE and bias reproduce a plausible, documented number. `test_no_leakage.py` passes. |
| **3 — Week 3** | Regime labels + classifier | Confusion matrix with support counts. **Accuracy degrades with lead time** (day-5/day-1 ratio < 0.95). LPS detector reproduces ≥ 70% of Zenodo-catalogued LPS days in an overlap year. |
| **4 — Week 4** | Level-1 corrector + all ablations | Ablation table on the **validation** season. Monotonicity constraints verified. ★ **Decision gate: does config 5 beat config 4 on heavy-rain CSI?** If not, escalate now and re-rank the USPs. |
| **5 — Week 5** | API + frontend skeleton on synthetic data | Frontend renders fully with `VITE_USE_MOCK=true` and the backend stopped. |
| **6 — Week 6** | All-India expansion; CSGD-MoE | CRPSS > 0 vs climatology on validation. Reliability diagram produced. |
| **7 — Week 7** | Atlas, verification screens, explanations, analogs | Atlas heatmap renders from real numbers with `n_events` in every cell; insufficient cells greyed out. |
| **8 — Week 8** | Demo cache, replay mode, polish, rehearsal | ★ **Full six-minute demo runs end to end with the network disabled.** Rehearsed three times on the projector resolution. |
| **9 — Week 9** | Final evaluation on the test season (once), PPT, video fallback | `results/final/` regenerated by `make all` from a clean clone. Every number in the PPT traceable to a `MANIFEST.json`. |

---

## 17. TEAM SPLIT (6 people)

| Role | Owns | Phase-1 unblocking task |
|---|---|---|
| **Data engineer** | `ingest/`, `static/`, caching, the harvester | **Freeze `districts_master.csv` and publish the data contract by day 3.** Everyone else is blocked until this exists. |
| **ML — regimes** | `labels/`, `models/regime_clf.py` | Detectors + Zenodo tuning |
| **ML — correction** | `models/corrector*`, `csgd.py`, `moe.py`, `train/` | Level-1 LightGBM first, CSGD second |
| **Backend** | `api/`, DuckDB, Docker, demo cache | Ship the API against synthetic data in week 1 so the frontend is never blocked |
| **Frontend** | `frontend/` | Build against `public/mock/` from day 1 |
| **Verification + narrative** | `verify/`, `atlas.py`, PPT, `JUDGE_QA.md` | ★ Owns the Atlas — the lead USP. This is not a "documentation" role; it is the role that produces the headline artefact. |

Nobody waits for anybody: the data contract (§13.2) and the mock JSON are the two interfaces that decouple all six workstreams.

---

## 18. WHAT WE DELIBERATELY DO NOT BUILD

Write this list on a wall. Every item below has killed a hackathon team.

| Not building | Why |
|---|---|
| Our own NWP model / WRF runs | Months of compute. We are a post-processor by design. |
| GAN / diffusion / latent-diffusion rainfall generation | Weeks of GPU, unstable, unverifiable, unexplainable to a forecaster, catastrophic if it mode-collapses on stage. Classic novelty theatre. |
| U-Net / transformer trained from scratch on gridded fields | ~3 monsoons of data. Guaranteed overfit. See §4.11. |
| LSTM predicting rainfall from rainfall history | Ignores the NWP entirely. Loses to the raw model by day 3. Not post-processing. |
| Blockchain, IoT rain gauges, drones | No connection to the problem. |
| A consumer mobile app with push alerts | Wrong user. Dilutes the pitch. Huge surface area. |
| An LLM chatbot over the forecast | Adds risk, subtracts credibility. A templated bulletin is better and safer. |
| Radar / DWR assimilation, nowcasting | Different problem, different data, different lead times. Future work slide. |
| User accounts, roles, admin panels | Zero demo value. |
| Real-time streaming (Kafka, websockets) | Rainfall forecasts update four times a day. A nightly batch is correct engineering. |
| Trying to "beat IMD" | Wrong framing, factually shaky (areal mean vs isolated places), and reads as arrogant. We are a calibration layer for IMD. |
| More than three demo events | Every extra event is another thing that can break. |

---

## 19. PPT BLUEPRINT (13 slides)

Judges read the deck before they meet you. Slide 3 must land in ten seconds.

| # | Title | The one message | The visual | What NOT to put on it |
|---|---|---|---|---|
| 1 | RAAHAT | Regime-aware rainfall post-processing for district heavy-rain warnings | One full-bleed regime-coloured scatter: raw forecast vs observed, clusters visibly separated | Team photos, college logo grid, a mission statement |
| 2 | The error is not random | Same model, same district, opposite bias in two regimes | Two side-by-side scatter panels: break vs LPS days | Any text over three lines |
| 3 | **Why existing correction fails** | Post-processing over India improved forecasts overall but **under-performed in the monsoon**; IMD's MME weights are season-averaged | Quote box with the published finding + the IMD MME day-1 numbers (CC 0.58, RMSE 12.7 mm/day) | A generic "problem in India" paragraph |
| 4 | **What we measured: the Regime Error Atlas** | Conditional NWP error for India, by regime × zone × lead — not published before | The heatmap. Full slide. Minimal chrome. | Anything else. This slide holds one object. |
| 5 | Our solution in one picture | Regime **forecast** → soft mixture of experts → calibrated probability → colour | The §12 diagram with its three callouts | Ten boxes. Four stages only. |
| 6 | Where we differ from the brief | Soft gating, terrain-as-modifier, distribution-not-number | The three-row table from §1.5 | Apology or hedging. State it as engineering judgment. |
| 7 | The model | LightGBM gate + CSGD experts, CRPS loss, 60k parameters, **no GPU** | Compact architecture schematic + "trains in 11 min on a laptop" | Formulas. Nobody reads them. Keep one in backup. |
| 8 | Data — all free, mostly keyless | 6 sources, 4 need no account, every licence named | The source table + the sample-size table from §4.11 | Hiding the sample size. Showing it is a credibility move. |
| 9 | Prototype | Screenshots of the live desk | Forecast screen + district panel with the explanation card | A feature checklist |
| 10 | **Evidence** | Held-out season, five configurations, per-regime CSI at 115.6 mm | The ablation table + the reliability diagram, **with one non-improving cell visibly marked** | Cherry-picking. The marked failure is why the rest is believed. |
| 11 | The case we got right, and the alarm we prevented | Wayanad + the break-monsoon downgrade | Raw / corrected / observed triptych, then the false-alarm case | Casualty figures, disaster imagery, dramatisation |
| 12 | Honest limits | Areal mean ≠ isolated places; 3 seasons of data; regimes are a convention, not ground truth | Three short bullets | Pretending there are none |
| 13 | Deployment path | Sits upstream of BFS; free inputs; runs on commodity hardware; IMD warnings already being archived for benchmarking | Simple integration diagram with IMD/NCMRWF | "Revenue model", "10 crore users", fake partnership logos |

**Design rules for the deck:** one idea per slide; the IMD warning colours are the only saturated colours; every chart is regenerated by `make figures` so nothing is a screenshot of a screenshot; every number carries its `n`; no slide has more than 25 words of body text.

---

## 20. TWENTY JUDGE QUESTIONS — REHEARSE THESE

Full answers live in `docs/JUDGE_QA.md`. The five that decide the outcome:

**Q1. "IMD already has the Bharat Forecast System at 6 km. Why do we need this?"**
Higher resolution reduces representativeness error. It does not remove conditional model bias — those are different errors with different fixes. Every major forecasting centre runs both dynamical improvement and statistical post-processing, because they are complementary. BFS output is our best future *input*, not our competitor. Our system would improve BFS the same way it improves GFS.

**Q2. "Regime-dependent post-processing already exists. What's new?"**
Correct, and we cite it: regime-dependent EMOS for North Atlantic wind, and regime-dependent bias correction proposed at NCEP. The method class is established — for mid-latitude wind and temperature. It has never been instantiated for the Indian monsoon regime taxonomy, at district scale, with an operational heavy-rain decision layer. And our primary contribution is not the algorithm — it is the Regime Error Atlas, a measurement of Indian forecast error that does not exist publicly.

**Q3. "You only have three monsoon seasons. Isn't that too little?"**
It is too little for a deep gridded model — which is exactly why we did not build one. It is roughly 275,000 district-days and 3,000–8,000 heavy-rain events per lead time, which is ample for gradient-boosted trees and a 60k-parameter parametric network with district embeddings. We report effective sample size accounting for synoptic autocorrelation, we grey out every verification cell with fewer than 30 events, and we have a documented path to the 2000–2019 GEFSv12 reforecast if regime-specific counts prove thin.

**Q4. "Bias correction usually destroys extremes. How do you know yours doesn't?"**
That is the central risk and we tested it first, not last. We train on CRPS with a censored-shifted-gamma tail rather than on MSE, so the tail is fitted rather than averaged away. Our headline metric is CSI at 115.6 mm, not RMSE. And the decision layer carries a hard guard: the assigned colour can never fall below the colour the raw multi-model mean would have produced unless the exceedance probability is very low.

**Q5. "How do we know the regime part is doing anything, and not just the extra features?"**
Ablation 4 versus ablation 5: identical architecture, regime information zeroed. The difference in per-regime heavy-rain CSI is the answer, it is in the deck, and if it had been zero we would have said so.

Also rehearse: false-alarm cost · what happens when the classifier is wrong · why not deep learning · who maintains it · how it integrates with SACHET/CAP · why district and not grid · data licensing · what breaks at scale · inference cost · how a forecaster overrides it · how you handle a new district being created · why DuckDB · what you would do with six more months · what you would remove if you had to ship tomorrow.

---

## 21. FIRST THREE THINGS TO DO TODAY

1. **Register two accounts at `api.imd.gov.in` and deploy `harvest_imd.yml` as a daily cron.** Highest return on effort in the project. Every day of delay is a day of operational warning archive lost forever, because the API serves only current state.
2. **Freeze `data/static/districts_master.csv` and publish the data contract (§13.2).** Five of six workstreams are blocked until this exists.
3. **Write and unit-test `src/ingest/time_align.py`.** The 0830 IST rainfall-day convention is a silent 15–20% skill killer and the cheapest possible thing to get right on day one.

---

## 22. ATTRIBUTION (required, must appear in the dashboard footer and on the data slide)

- Weather forecast data by **Open-Meteo.com**, licensed CC BY 4.0.
- Observed rainfall: **India Meteorological Department**, Pune — high-resolution gridded daily rainfall dataset.
- *"This service is based on data and products of the European Centre for Medium-Range Weather Forecasts (ECMWF). Source: www.ecmwf.int. This ECMWF data is published under a Creative Commons Attribution 4.0 International (CC BY 4.0)."*
- Monsoon LPS track dataset: Vishnu, Boos, Ullrich & O'Brien, Zenodo.
- Western Disturbance track catalogue v6 (ERA5, 1950–2025), Zenodo.
- Active/break spell definition after Rajeevan, Gadgil & Bhate (2010), *J. Earth Syst. Sci.*
- CSGD post-processing after Scheuerer & Hamill (2015), *Mon. Wea. Rev.*

Check the licence of the district polygon source before the finale and cite it here.
