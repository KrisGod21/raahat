# DATA NOTES

Every correction to the spec, every API-name drift, every caveat. Spec §0 rule 3.

Append, never rewrite. Each entry: date, what the spec says, what is actually
true, what was done about it.

---

## 2026-09-21 — Phase 3 (Stage A) build

### 1. ★ PyTorch cannot differentiate the incomplete gamma w.r.t. its shape

**Spec §6.2** trains the Level-2 CSGD mixture by backpropagating CRPS through
the predictive CDF. The CSGD CDF is `gammainc(k, (x − δ)/θ)`, and both `k` and
`θ` are functions of the network outputs `μ` and `σ`. So a gradient with
respect to the *first* argument of `gammainc` is required.

**Actually true** (verified, torch 2.11.0):

```
torch.special.gammainc(a, z).backward()
  -> NotImplementedError: the derivative for 'igamma: input' is not implemented
torch.distributions.Gamma(c, r).cdf(x).backward()
  -> same error (it calls igamma internally)
```

Both routes are dead. Written naively, Level 2 does not train at all.

**Impact:** would have blocked Phase 6 (week 6), discovered under time
pressure, with the CSGD-MoE as the headline architecture.

**Two workarounds, both verified available:**

| | Route | Verified | Trade-off |
|---|---|---|---|
| **A** (preferred) | Hand-roll the regularised lower incomplete gamma in pure torch ops — series expansion for `z < k+1`, continued fraction otherwise — and let autograd differentiate the composition. `torch.lgamma` **does** have gradients (checked). | `lgamma` grad ok | ~40 lines; exact to series truncation; keeps the fixed-grid CRPS of §6.2 unchanged |
| **B** (fallback) | Sample-based CRPS via the identity `CRPS = E‖X−y‖ − ½E‖X−X′‖`, using `Gamma.rsample()`, which **does** carry implicit reparameterisation gradients (checked). | `rsample` grad ok | estimator noise; mixture term is a K² double sum |

**Done:** `config/model.yaml → csgd_moe.cdf_backend` selects between them,
default `series`. Neither is implemented yet — Level 2 is Phase 6.

### 2. ★ Stage A hyperparameters are sized for the wrong table

**Spec §6.1** gives `n_estimators=600, num_leaves=63, min_child_samples=100`.

**Actually true:** the regime is a *national daily* state, so Stage A trains on
one row per `(valid_date, lead_day)`. One JJAS season is **122 rows per lead**,
and the train split is a single season. With soft-label expansion that is 134
weighted rows.

`min_child_samples=100` on 134 rows makes essentially every split illegal. The
classifier collapsed to the majority class: accuracy 0.541, and the confusion
matrix was a single filled column. Those hyperparameters are sized for the
district-level table of §4.11 (~275k rows), which is ~2,000× larger.

**Done:** `n_estimators=300, num_leaves=8, min_child_samples=5`. Validation
accuracy went 0.541 → 0.847. Deviation is flagged in `config/model.yaml`.

**Open question for the ML-regimes owner:** §6.1 asserts that one model per
lead is "simpler and more accurate than one model with lead as a feature". At
122 rows per lead that claim is doubtful — pooling all five leads gives 610
rows and lets the model share structure. **Test both in Phase 3 and report.**
Ingesting the 2021–2023 Historical Forecast API rows (§4.2) would take lead-1
to ~730 rows and is the cheapest fix if per-lead stays weak.

### 3. Stage A operates on a national table — clarifies §5.3 / §13.1

The spec's §13.2 contract is per `(district_id, valid_date, lead_day)` and
lists `p_ACTIVE…p_WEAK` as columns, which reads as though the classifier runs
per district. It does not, and must not: training on district-replicated rows
would multiply the apparent sample size ~750× while adding no information, and
every validation number would be meaningless.

**Done:** two schemas. `contract.regime_input_schema()` is the Stage-A table,
one row per `(valid_date, lead_day)`. Its six outputs are **broadcast** to all
districts for that day and become Stage-B features. Noted at the top of
`config/features.yaml`.

### 4. LightGBM returns one probability column per class *seen in training*

Not a spec error — a library behaviour that breaks the contract silently.

**Spec §6.1** correctly warns that WD is rare in JJAS (~2–5% of days). In the
synthetic 2024 season it did not occur at all. LightGBM then trains a 5-class
model and `predict_proba` returns a `(n, 5)` array where the contract requires
`(n, 6)`. First symptom was `IndexError: index 5 is out of bounds`.

On real data this will happen for WD in any thin season, and for
`EASTERLY_COASTAL` in a leave-one-zone-out fold.

**Done:** `RegimeClassifier._proba_full()` scatters the model's `classes_` into
the full six-slot vector, zero for absent classes — the honest answer, since a
model that never saw a Western Disturbance cannot predict one. Training prints
a loud `UNSEEN:` warning naming the missing regimes. Covered by
`tests/test_regime_clf.py::test_proba_full_scatters_seen_classes_into_the_contract_vector`.

### 5. The leakage canary has a false-positive mode the spec does not mention

**Spec §5.7** fails the build when `acc(day5)/acc(day1) > 0.95`, on the sound
reasoning that a forecast-driven classifier must get worse with lead time.

**Also true:** a *degenerate* model — one that has collapsed to the majority
class — has perfectly flat accuracy across leads and trips the canary with
ratio 1.000. Observed exactly this while finding item 2 above. The canary said
"LEAKAGE SUSPECTED" when the real fault was under-fitting. Hunting a phantom
leak costs days.

**Done:** `leakage_canary()` checks degeneracy first (fewer than 2 distinct
predicted classes, or accuracy ≤ majority base rate) and reports that instead.
Covered by `test_canary_reports_degeneracy_rather_than_blaming_leakage`.

### 6. LightGBM ignores `subsample` unless `subsample_freq >= 1`

Silent no-op. Spec §6.1 specifies `subsample=0.8`; without `subsample_freq` it
does nothing at all. Set to 1 in `regime_clf._lgb_params()`.

### 7. Environment drift

- Spec §8 says Python 3.11. Local interpreter is **3.13.7**. Nothing has broken
  so far; `pyproject.toml` pins `>=3.11`. Re-check before the finale — CI
  should run the version the demo laptops run.
- `torch` installed is the CUDA build (`2.11.0+cu128`). Harmless: spec §0 rule 4
  is CPU-only and nothing calls `.cuda()`. Do not let a GPU tensor creep in —
  the "no GPU" claim is a competitive point (§19 slide 7).

### 8. Synthetic data is calibrated to §4.11, deliberately

`scripts/make_synthetic.py` is tuned so ~1.5–2.5% of district-days clear
64.5 mm and ~0.1–0.4% clear 115.6 mm, matching the sample-size table of §4.11.
First pass ran 4.6–7.1% heavy, which would have flattered every classifier.

Noise on the Stage-A features is scaled to **each feature's own spread across
regimes**. With a flat noise term the signature features (vorticity ~9,
z500 ~−55, IVT ~420) become noiseless discriminators, day-5 stays as easy as
day-1, and the §5.7 canary is never exercised. Current synthetic accuracy runs
0.975 at lead 1 → 0.598 at lead 5, which is the shape real data should have.

**This is synthetic data. No number from it goes in the PPT.**

---

## 2026-09-21 — Phase 1 (real data ingest)

Synthetic data is retired for Stage B from here. Everything below is measured
against the real archives.

### 9. ★★ The IMD rainfall day is labelled by its END, and we had it backwards

**Spec §4.2** is emphatic that the rainfall day runs 0830 IST → 0830 IST
(03Z → 03Z) and that getting it wrong costs 15–20% of apparent skill. It does
not say which calendar date a 03Z–03Z window is *labelled* with, and the two
possibilities differ by exactly one day.

**Measured**, JJAS 2024, ERA5 hourly vs IMD 0.25° gridded, 8 districts across
all three zones, correlation of daily totals:

| district | zone | label_by_start | label_by_end | margin |
|---|---|---|---|---|
| Bilaspur | C | 0.183 | **0.601** | 0.418 |
| Dehradun | N | 0.165 | **0.599** | 0.434 |
| Kolhapur | W | 0.322 | **0.647** | 0.325 |
| Raigarh | C | 0.308 | **0.637** | 0.329 |
| Koraput | C | 0.282 | **0.550** | 0.268 |
| Nagpur | C | 0.563 | **0.786** | 0.223 |
| Wayanad | W | 0.559 | **0.776** | 0.217 |
| Idukki | W | 0.488 | **0.680** | 0.192 |

**8 of 8, every one decisive.** IMD rainfall day D is the window ENDING at
0830 IST on day D — the gauge read on the morning of D is credited to D. So
most of the rain credited to day D physically fell on calendar day D−1.

Spec §4.2 estimates the cost of getting this wrong at 15–20% of skill. Measured
here it **roughly halves the correlation** (0.165 → 0.599 at Dehradun). This is
the single highest-leverage thing in the pipeline and it is now pinned by
`tests/test_time_align.py::test_default_convention_is_the_measured_one`.

Do not change `DEFAULT_CONVENTION` without re-running
`time_align.determine_convention()`. It is a one-line edit that would look
harmless in review and would silently destroy every score.

### 10. ★ IMD has not published 2026 — the spec's three-way split is impossible

`imdlib` returns a **zero-byte file** for 2026 with `Error in file reading,
mismatch in size of data-length`. Expected: JJAS 2026 was still running on
2026-09-21, nine days short of the season end. Spec §4.1 anticipated this
("if the current season is missing, cap the test season at 2025 and say so").

Two usable seasons, not three. New split, decided with the user:

```
train       2024-06-01 .. 2024-09-30   (122 days)
validation  2025-06-01 .. 2025-07-31   ( 61 days)
test        2025-08-01 .. 2025-09-30   ( 61 days)   LOCKED
```

All of JJAS 2024 trains because Stage A is sample-starved (item 2). JJAS 2025
is cut **once**, so only a single adjacent day-pair straddles the boundary —
negligible against the ~10-day synoptic decorrelation of §4.11. Aug–Sep is the
LPS-rich half, which is where the heavy-rain claim has to hold.

**Caveat for the honest-limits slide (§19 slide 12):** validation and test are
not exchangeable. Isotonic calibration fitted on Jun–Jul climatology may not
transfer perfectly to Aug–Sep. Re-split into whole seasons when IMD publishes
2026, and report both.

**Related trap, now fixed:** a zero-byte `.grd` passed the cache glob, so
`cached_years()` reported 2026 as available and `load()` would have returned a
short record — silently corrupting every climatology built on it. `cached_years`
now enforces a minimum plausible size and `prune_truncated()` removes the
carcasses.

### 11. ★ GFS's previous-runs archive collapses on heavy-rain windows

Comparing the lead-3 forecast total against the model's own analysis total at
Wayanad, per model per window:

| window | gfs_seamless | icon_seamless | ecmwf_ifs025 |
|---|---|---|---|
| 26–31 Jul 2024 (extreme) | **0.04** | 0.50 | 0.36 |
| 10–15 Jul 2025 | **0.23** | 0.80 | 1.44 |
| 1–6 Sep 2024 (quiet) | 0.62 | 1.17 | 0.85 |
| 1–6 Sep 2025 (quiet) | 0.67 | 0.74 | 0.89 |

GFS is roughly sane on quiet windows and collapses to ~4% of the analysis on
the two heavy-rain windows — which are precisely the cases this project exists
to correct. `gfs_seamless` and `gfs_global` return byte-identical values, so
Open-Meteo appears to alias them.

**Decision (user):** drop GFS. Level 1 runs on `ecmwf_ifs025` + `icon_seamless`,
both verified back to Jul 2024. Two models still give multi-model spread.

**Not yet established:** whether this is archive sparsity or a genuine GFS
failure on Western Ghats orography. It is stated above as a measurement, not a
diagnosis. If a judge asks "why not GFS", the honest answer is the table plus
"we did not determine the cause". Worth an hour later — testing several wet
locations on identical dates against ECMWF would separate the two.

`ecmwf_aifs025_single` returns nothing before 2025, as expected for a newer
model. Available for 2025 onward if a Level-2 AIFS comparison is wanted.

### 12. Open-Meteo parameter names — verified live, no drift

Spec §0 rule 3 requires checking. As of 2026-09-21 the spec's names are correct:

* `precipitation_previous_day1..5` exist as **hourly** variables and return
  non-null data at every requested lead
* ERA5 archive **accepts** the pressure-level names but see item 16 below --
  it returns all-null for every one of them. Only surface variables carry data.
* `timezone=GMT` behaves as documented

CORRECTION (same day): the bullet above originally read "ERA5 archive serves
pressure_msl, wind_speed_850hPa, geopotential_height_500hPa, ...". That was
wrong and is retracted. The probe that produced it checked only that the KEYS
came back, not that the values were non-null. See item 16. Wind speed/direction are converted to u,v **before**
any daily averaging — averaging a bearing in degrees is meaningless once it
wraps through 360°, and that mistake produces plausible-looking garbage.

### 13. imdpune.gov.in is intermittent — caching is not optional

Served three requests, then began timing out at the TLS handshake inside the
same session, then served 2025 after four retries. The HTML page at
`/cmpg/Griddata/Rainfall_25_NetCDF.html` times out even when `imdlib`'s POST
endpoint works, so a browser check is not a reliable availability test.

All IMD years are cached to `data/raw/imd_grid/` and read with
`imdlib.open_data(..., file_dir=...)`, which never touches the network.
Everything else goes through `ingest/http_cache.py` (SHA-256 of the full URL,
atomic writes). **Do not delete these caches.** Re-acquiring them depends on a
host that is down as often as it is up, and spec §10.3 requires the demo to run
with the network unplugged.

### 14. District boundaries — Datameet paths moved; Census 2011 caveats

The path in common circulation (`Districts/Census_2011/2011_Dist.geojson`) is a
**404**. The live file is `docs/data/geojson/dists11.geojson`, 28 MB, 641
districts, 35 states/UTs, ODbL.

Three names in `config/zones.yaml` did not match and were resolved against the
source rather than fuzzy-matched (spec §4.6 forbids silent fuzzy matching):

| written as | Census 2011 spelling |
|---|---|
| Gadchiroli | `Garhchiroli` |
| Hathras | `Mahamaya Nagar` (its 2011 name) |
| Kanshiram Nagar | `Kansiram Nagar` |

**Telangana does not exist in Census 2011** — its districts are still filed
under Andhra Pradesh, which is why `config/zones.yaml` lists them there. Any
district created after 2011 is absent. Acceptable because the master is frozen;
belongs on the honest-limits slide.

**Adjacency:** `shapely`'s `touches` predicate reported Kandhamal and Koraput as
having no neighbours, which is geographically absurd — `touches` requires
boundaries to meet while interiors do *not* overlap, so one sliver of topology
noise breaks it. Switched to `intersects`: 950 edges, 4.8 neighbours per
district on average, zero isolated. This matters because `pr_nbr_max` is called
out in §5.1 as the single most important heavy-rain predictor.

### 15. Ground truth built and spot-checked — Phase 1 acceptance

`data/interim/district_obs_{2024,2025}.parquet`, area-weighted from the IMD
0.25° grid over district polygons, cosine-latitude weighted.

* 197 districts × 122 days = **24,034 rows per season**, **0.00% missing**
* **1.80% of district-days exceed 64.5 mm** — inside spec §4.11's stated 1–3%
* max district areal mean 230.5 mm

Spot-check, Wayanad, 30 Jul 2024 (spec §10.1 event 1): grid box mean
**111.3 mm**, max cell **222.3 mm**. Spec §10.2 says "it recorded over 200" —
the max cell agrees. The 111 vs 222 gap is exactly the areal-mean-vs-isolated-
places caveat of §3 USP 3, visible in our own data on the demo event. Worth
showing on that slide.

### 16. Open-Meteo serves NO pressure-level data — surface only, everywhere

Checked at a land point (20N 80E) and a sea point (5N 60E), 1-2 Jul 2024:

```
   ok        pressure_msl                     48/48
   ALL-NULL  wind_speed_850hPa                 0/48
   ALL-NULL  wind_speed_200hPa                 0/48
   ALL-NULL  geopotential_height_500hPa        0/48
   ALL-NULL  relative_humidity_850hPa          0/48
   ALL-NULL  temperature_850hPa                0/48
   ok        precipitation                    48/48
```

The API accepts the names (no HTTP 400) and returns the keys with every value
null. A probe that checks only for the presence of keys passes. Ours did, and
the earlier entry in item 12 was wrong because of it. **Check values, not
keys.**

Combined with the Previous Runs restriction already noted, the position is:

| | surface | pressure levels |
|---|---|---|
| Previous Runs (forecast, per lead) | yes | HTTP 400 |
| Archive / ERA5 (analysis) | yes | all-null |
| Single Runs | yes | needs per-init `run`, ~46k calls |

**So there is no free upper-air field available on this path at all**, for
either labels or features. Spec 4.3's plan assumed otherwise.

**What survives, and it is more than it first looks:**

* **ACTIVE / BREAK** is unaffected and in fact better sourced. Spec 4.5 defines
  the index on **IMD 0.25 deg rainfall** over the core monsoon zone, not on the
  mesh -- and we have that grid locally. This is the highest-frequency pair of
  classes and its labels are exactly as the published definition intends.
* **LPS** is detectable from **MSLP closed lows**, which is the classical
  synoptic definition of a monsoon depression and close to how IMD itself
  classifies them. Not a downgrade so much as a different, defensible route.
  Spec 4.4's Zenodo catalogue can still tune the thresholds.
* **EASTERLY_COASTAL** works from 10 m wind direction plus coastal rainfall.

**What is genuinely degraded, and must be said out loud:**

* **WD** loses its 500 hPa trough signature entirely. A MSLP-plus-2 m-
  temperature proxy is weak. WD is 2-5% of JJAS days (spec 6.1), so this is a
  thin class made thinner. **Recommendation: report WD's per-class metrics with
  its support count and do not claim skill on it.** Spec 6.1 already says
  "merge or softly weight, do not oversample into fake confidence".
* Vorticity, IVT and shear are unavailable as features. `upslope_flux`
  (spec 5.5) must fall back to **10 m wind** instead of 850 hPa wind. Near the
  Ghats the 10 m wind is strongly terrain-influenced, which is arguably closer
  to the physics of upslope forcing, but it is not what the spec specified.

**Upgrade paths, none on the critical path (spec 4.3 lists the first two):**

1. **IndiaWeatherBench** (HuggingFace `tungnd/IndiaWeatherBench`) or
   **BharatBench** (Kaggle `maslab/bharatbench`) -- IMDAA 12 km, ML-ready,
   real upper air. Best option; check licence (IndiaWeatherBench is
   CC BY-NC-SA 4.0, fine for SIH with attribution).
2. **ERA5 via Copernicus CDS** with `cdsapi` -- free, registration plus queue.
3. **ECMWF Open Data** GRIB (spec 4.9) for forecast-side upper air.

Until one of those lands, the honest framing for the PPT is: *the regime
classifier reads only what an operational forecast reliably publishes at lead
time -- surface pressure, rainfall, near-surface wind and temperature.* That is
a real constraint an operational system would also face, and it is a stronger
claim than quietly using reanalysis fields that would not be available at
forecast time anyway.

---

## 2026-09-22 — Phase 2/4 (real features, Level-1 corrector, ablations)

First real numbers. Lead 3, validation block (Jun–Jul 2025), 197 districts,
ECMWF IFS025 + ICON against IMD 0.25° gridded truth. **Nothing here is from
synthetic data.** The test block (Aug–Sep 2025) has not been opened.

### 17. Ablation scorecard — the corrector works on continuous skill

| config | bias | RMSE | MAE | corr |
|---|---|---|---|---|
| 0 climatology | −0.82 | 13.41 | 8.12 | 0.509 |
| 1 raw ECMWF | +1.23 | 15.30 | 8.63 | 0.506 |
| 2 multi-model mean | +0.55 | 15.60 | 8.40 | 0.458 |
| 3 quantile mapping | +1.05 | 18.64 | 9.33 | 0.456 |
| **4 RAAHAT (no regime)** | **−0.26** | **12.70** | **7.48** | **0.587** |

RMSE −17% against raw, correlation +0.081, bias essentially removed. The
corrector also beats climatology, which the raw model does **not** on RMSE —
worth noting, because "is this better than nothing" is a fair question.

Quantile mapping makes RMSE *worse* (18.64 vs 15.30) while nudging heavy-rain
CSI up. That is the classic variance-inflation behaviour of QM and it is a
useful thing to show: the conventional post-processor is not a free win.

### 18. ★ Scoring a probabilistic model by its median understates it badly

First pass thresholded the corrected **median** for the categorical scores and
got CSI 0.092 against raw 0.102 — i.e. "our model is worse at heavy rain".

That was a scoring error, not a model failure. A median is a central estimate;
thresholding it for a 1.5%-base-rate event guarantees under-detection, because
the median of a right-skewed rainfall distribution sits far below its tail.
Spec 6.4 assigns warnings from `P(exceed) > alpha`, never from the median.

Fixed in `train/ablations.py`: categorical scores now come from the exceedance
probability wherever one exists, and the table prints which source each column
used so the two can never be silently mixed.

### 19. ★ The cost-loss alpha matters more than the model, and 0.30 is wrong here

Sweeping alpha for P(>64.5 mm), lead 3, validation, 188 events:

| alpha | POD | FAR | CSI | ETS | freq bias |
|---|---|---|---|---|---|
| 0.05 | 0.606 | 0.881 | 0.110 | 0.097 | 5.12 |
| **0.10** | 0.319 | 0.795 | **0.143** | 0.133 | 1.55 |
| **0.15** | **0.223** | **0.732** | 0.139 | 0.132 | 0.84 |
| 0.20 | 0.160 | 0.691 | 0.118 | 0.112 | 0.52 |
| 0.30 *(spec default)* | 0.080 | 0.545 | 0.073 | 0.070 | 0.18 |
| **raw ECMWF** | 0.176 | 0.806 | 0.102 | 0.094 | 0.90 |

* At **alpha = 0.10** CSI is **0.143 vs 0.102 raw — a 40% improvement**, ETS
  0.133 vs 0.094.
* At **alpha = 0.15** the model beats raw on **both POD and FAR
  simultaneously** (0.223 / 0.732 against 0.176 / 0.806). A strict improvement,
  which is the cleanest possible claim.
* At the spec's default **0.30** it loses. Not because the model is bad, but
  because 0.30 is far too high a bar for a 1.5% base rate.

**Recommendation:** change `decision.alpha_default` from 0.30 to **0.15**, and
put the sweep itself on a slide. It is a better argument than any single row:
it shows the operator choosing their own miss/false-alarm posture, which is
precisely what spec 6.4 intends, and it shows we measured rather than guessed.

Note the direction: LOWER alpha = more warnings = more miss-averse. Spec 6.4's
own range (0.05 very miss-averse → 0.60 false-alarm-averse) agrees; the default
of 0.30 simply sits on the false-alarm-averse side, which is an odd posture for
disaster management.

### 20. Calibration is excellent — but in-sample, and must be labelled so

Reliability of P(>64.5 mm), lead 3, validation:

| bin | forecast | observed | n |
|---|---|---|---|
| 0.00–0.12 | 0.011 | 0.011 | 11,752 |
| 0.12–0.25 | 0.181 | 0.181 | 232 |
| 0.25–0.38 | 0.348 | 0.348 | 23 |
| 0.62–0.75 | 0.667 | 0.667 | 9 |

Forecast probability equals observed frequency in every populated bin. Brier
skill score **+0.104** against climatology.

**This is in-sample.** The isotonic calibrator is fitted on the validation
block and then evaluated on it, so near-perfect agreement is close to
guaranteed. The honest number comes from the locked test block, once.
**Do not put this diagram in the deck as-is** — regenerate it from the test
block and expect it to be visibly worse. A reliability diagram that is too good
is itself a red flag to a knowledgeable judge.

### 21. ★ LightGBM refuses monotone constraints on a quantile objective

Spec 6.2 constrains the corrector to be non-decreasing in `mm_mean` and
`pr_ifs_nbr_max` so that more forecast rain can never yield less corrected
rain. LightGBM rejects this outright:

```
LightGBMError: Cannot use ``monotone_constraints`` in quantile objective,
please disable it.
```

The constraint IS applied to the binary threshold classifiers, where the
objective supports it. For the quantile regressors it cannot be, so it is
**measured** instead — `CorrectorLGBM.monotonicity_audit()` adds 10 mm to the
monotone features and checks the corrected median never falls:

```
violations 513/4000 (12.83%)   worst drop 4.70 mm   mean response +1.69 mm
```

**12.8% of the time, raising the raw forecast by 10 mm lowers our correction.**
The mean response is correctly positive (+1.69 mm) and the worst drop is small
(4.70 mm), but this is exactly the demo failure spec 6.2 warns about, and one
live example on stage would be expensive.

**Options, in order of preference:**
1. Fit the median with `objective="regression_l1"` (MAE also targets the
   median) which **does** accept monotone constraints, and keep quantile
   objectives only for p10/p90. Cheapest real fix.
2. Post-hoc isotonic repair of the median against `mm_mean`.
3. Ship as-is and never demo the slider. Not recommended.

Not yet fixed — flagged because Phase 4's acceptance test in spec 16 says
"monotonicity constraints verified", and honestly they are not.

### 22. The decisive table of spec 11.4 cannot be computed on this block

Spec 11.4: "per-regime CSI at 115.6 mm, day-3 lead, held-out test season".

Validation (Jun–Jul 2025, 197 districts, 12,017 district-days at lead 3) holds
**19 events above 115.6 mm and 0 above 204.5 mm**. Spec 11.4 requires any cell
with fewer than 30 events to be greyed out as insufficient — so the headline
table is unreportable at 115.6 mm here, before it is even split by regime.

Spec 4.11 assumed ~3,000–8,000 heavy events per lead. That figure is for
**all-India (~750 districts) across three seasons**. At 197 districts over half
a season we have ~1/20th of that.

**Consequences, and none of them are optional:**
* Report the headline table at **64.5 mm** (188 events, sufficient) and show
  115.6 mm greyed out with its n. Do not quietly switch thresholds without
  saying why.
* Aug–Sep is the LPS-rich half, so the test block should hold noticeably more
  extreme events than Jun–Jul. That is a reason the split was cut this way.
* Expanding to all-India (spec 2.2) multiplies events ~3.8x and is the single
  highest-value next step for statistical power.

### 23. ★ Monotonicity FIXED — 12.75% violations to 0.00%, at no cost

Item 21 left spec 6.2's constraint unenforced. Resolved.

**Why the obvious fixes fail.** LightGBM refuses `monotone_constraints` on
*every* quantile-estimating objective. Tested directly:

| objective | accepts constraints |
|---|---|
| `regression` (L2) | yes |
| `huber` | yes |
| `fair`, `poisson`, `tweedie`, `binary` | yes |
| `quantile` | **no** |
| `regression_l1` | **no** |
| `mape` | **no** |

The refusals are exactly the objectives whose gradients are sign-based —
constant magnitude, no curvature for the constraint machinery to use. So both
natural routes to a *median* (quantile α=0.5, and L1) are closed.

**The fix: distillation.** Fit quantile(0.5) as before, then fit a monotone L2
**student** to the teacher's own predictions. The student is non-decreasing by
construction and approximates the same median function. Measured on JJAS 2025
validation, lead 3:

| | violations | worst drop | RMSE | MAE | corr |
|---|---|---|---|---|---|
| quantile 0.5, unconstrained | **12.75%** | 4.70 mm | 12.70 | 7.48 | 0.587 |
| **distilled monotone student** | **0.00%** | 0.00 mm | **12.64** | **7.41** | **0.592** |
| huber + constraints | 0.00% | 0.00 mm | **12.27** | **7.13** | **0.625** |

The constraint costs nothing — skill improves slightly, and the mean response
to a +10 mm bump doubles (+1.68 → +3.41 mm), i.e. the model now reacts to the
raw forecast the way a forecaster would expect.

**Why not huber, which scores best?** It estimates something between the mean
and the median. `corrected_median` would stop being a median, and the
p10/median/p90 triple would no longer describe one distribution — which
matters because spec 1.5's whole argument is that we emit a predictive
distribution whose point value is its median. Coherence beats 0.37 mm of RMSE.
Huber remains the right choice if that interpretation is ever dropped.

Also changed: the outer quantiles are now **clipped to the median** rather than
the three being sorted. Sorting would let an unconstrained p10 displace the
constrained median and silently undo the guarantee.

Pinned by `tests/test_corrector.py`, including a guard test asserting the
*unconstrained* model still violates — otherwise the audit would be measuring
nothing.

**Phase 4 acceptance (spec 16) — "monotonicity constraints verified" — now
genuinely passes.** It did not before.

### 24. Skill against lead time — the headline result

Validation, α = 0.15, ECMWF IFS025 + ICON vs IMD gridded, 197 districts,
188 heavy-rain events per lead.

| lead | RMSE raw | RMSE ours | Δ | corr raw | corr ours | CSI raw | CSI ours |
|---|---|---|---|---|---|---|---|
| 1 | 12.93 | **11.01** | −14.9% | 0.607 | **0.709** | 0.153 | **0.261** |
| 2 | 14.01 | **12.25** | −12.6% | 0.550 | **0.633** | 0.119 | **0.188** |
| 3 | 15.30 | **12.64** | −17.4% | 0.506 | **0.592** | 0.102 | **0.139** |
| 4 | 16.49 | **12.82** | −22.3% | 0.458 | **0.581** | 0.097 | **0.141** |
| 5 | 17.32 | **13.22** | −23.7% | 0.417 | **0.561** | 0.080 | **0.123** |

Two things worth putting on a slide:

1. **The gain grows with lead** (−14.9% at day 1 → −23.7% at day 5). The worse
   the raw forecast, the more there is to correct. This is the right shape and
   it is the opposite of the failure mode where a post-processor only helps
   when the model was already nearly right.

2. **Correction buys about three days of lead time.** Our day-5 forecast
   (RMSE 13.22, corr 0.561) is roughly as accurate as the raw day-2 forecast
   (14.01, 0.550). Spec 11.1 asks for "lead time where skill > climatology" as
   the number decision-makers understand; this is the same idea expressed
   against the operational baseline, and it is more quotable.

CSI improves at every lead, most at day 1 (0.153 → 0.261, +71%).

**Caveats that travel with these numbers:**
* Validation block, not the locked test block. Calibration is fitted here, so
  the categorical scores are in-sample-flattered. Expect the test numbers to be
  worse and report them anyway.
* 115.6 mm has 19 events and 204.5 mm has none — spec 11.4's headline table is
  unreportable at those thresholds on this block (item 22).
* Ablation 5 is still not run: the regime features do not exist yet, so
  **nothing here tests the central USP**. Everything above is ablation 4 — the
  value of post-processing, not of regime-awareness.

### 25. Open-Meteo daily quota exhausted — the synoptic forecast mesh is 25% fetched

After ~1,500 calls (788 district forecasts + 378 analysis mesh + 82 forecast
mesh + elevation), the Previous Runs API began returning HTTP 429 on every
request. Backoff escalated 60s → 360s → 600s and stopped making progress.

Open-Meteo weights calls by cost, and a forecast-mesh point is expensive: six
variables × five leads × 131 days = 30 hourly series. The nominal
10,000 calls/day is not 10,000 requests of this size.

**Consequences:**
* `mesh_forecast` is cached for 82 of 336 points. Cached entries persist, so
  resuming after the quota resets costs only the remainder.
* `mesh_analysis` completed for 2024 (378/378) but **never started 2025**, so
  full 6-class labels are unavailable for the validation block.
* The 33% of elevation calls that failed earlier (item: terrain) were the same
  cause, not a separate fault.

**For the rebuild:** cut the forecast mesh to 4 variables and a 4° grid
(110 points), which is ~35% of the current cost, and run it as the first job
of the day rather than after 1,400 other calls.

### 26. ★★ THE USP, TESTED — regime conditioning helps, but not on the spec's own metric

With the mesh unavailable, Stage A was rebuilt from **forecast rainfall alone**
(`features/rainfall_regime.py`), using only already-cached data. Reduced to
3 classes — ACTIVE / BREAK / WEAK_TRANSITION — because spec 4.5's published
index is rainfall-based and therefore fully supported, while LPS, WD and
EASTERLY_COASTAL all need circulation.

**The spatial features recover a textbook signature.** Group means, lead 3:

| label | CMZ anom | rain centroid lat | concentration | N−C contrast |
|---|---|---|---|---|
| ACTIVE | +2.13 | 22.6°N | 0.350 | −4.23 |
| BREAK | −0.51 | **23.8°N** | 0.432 | **+6.27** |
| WEAK | +0.51 | 21.4°N | 0.398 | −1.76 |

During a break the rainfall centroid migrates **north** and the North-minus-
Central contrast flips from −4.2 to +6.3 — the monsoon trough shifting to the
Himalayan foothills, recovered from forecast rainfall with no circulation data
at all. That is a real result and worth a slide on its own.

**Stage A itself is weak.** Validation accuracy 0.784 against a majority base
rate of 0.754 — it adds almost nothing. ACTIVE recall 0.267, BREAK recall
0.000. And the leakage canary **fires**: accuracy rises with lead
(0.787 → 0.836, ratio 1.062 > 0.95).

That canary firing is probably an artefact rather than a leak — the label is a
3-day-persistent quantity and longer-lead forecasts are smoother, which mimics
temporal averaging. But the canary cannot distinguish those, so it correctly
refuses to pass, and **the honest position is that Stage A is unvalidated**.

**Ablation 4 vs 5, lead 3, validation, α = 0.15:**

| metric | 4 (no regime) | 5 (full) | delta |
|---|---|---|---|
| RMSE | 12.637 | **12.403** | −0.234 (−1.9%) |
| corr | 0.5916 | **0.6089** | +0.0173 |
| bias | −0.23 | +0.12 | smaller |
| FAR @64.5 | 0.732 | **0.643** | −0.089 |
| POD @64.5 | 0.223 | 0.186 | −0.037 |
| **CSI @64.5** | **0.139** | **0.139** | **0.000 — TIED** |
| Brier @64.5 | 0.01380 | **0.01367** | better |
| BSS @64.5 | 0.1039 | **0.1123** | +0.0084 |

**Verdict, stated the way spec 6.3 demands.** Spec 6.3: *"If (5) does not beat
(4) on per-regime heavy-rain CSI, the USP is not real and we must say so."*

On heavy-rain CSI the two are **tied**. The regime block does improve RMSE,
correlation, bias, false-alarm rate and Brier skill — consistently, if
modestly — but it does not improve the metric the spec nominated as decisive.
It trades detection for precision at constant CSI.

**So: say so.** On present evidence the regime-gating USP is *not*
demonstrated. What is demonstrated is that post-processing itself works well
(item 24) and that the regime block is a small, consistent secondary gain.

**Before concluding the USP is dead, note what this test is NOT:**
* 3 classes, not 6. LPS — the regime with the largest and most distinctive
  bias (spec 1.1) — is entirely absent, and it is the one most likely to carry
  the effect.
* Stage A barely beats its base rate, so the probabilities fed to ablation 5
  are close to uninformative. A better gate could only help.
* 188 heavy events on half a season. CSI differences of ±0.02 are not
  resolvable at this sample size.
* Spec 11.4's actual decisive table — per-regime CSI at 115.6 mm — remains
  uncomputable here (19 events).

**Recommended next steps, in order:**
1. Refetch the forecast mesh (cheaper config, item 25) and rebuild Stage A
   with 6 classes and MSLP-based LPS detection. This is the real test.
2. Expand to all-India (spec 2.2) for ~3.8× the events.
3. Only then re-run the gate. If it is still tied, pivot the headline to the
   Regime Error Atlas exactly as spec 3 anticipates — which is precisely why
   the Atlas is ranked USP 1 and does not depend on the gate working.

---

## 2026-09-22 — LPS detection, 4-class labels, and the Atlas

### 27. ★ The LPS detector was finding the monsoon trough, not depressions

First run on real MSLP fired on **89.3% of JJAS 2024** with centres clustering
at 24.9°N — against a published climatology of ~12–14 systems per season and a
Bay-head genesis region at 18–22°N. Two distinct faults:

**Fault 1 — ring-mean deficit cannot see a closed low.** The monsoon trough is
a zonally-elongated pressure minimum sitting over north India all season. Any
point on the trough axis is lower than a ring drawn around it, so the detector
reported the trough every single day.

*Fix:* measure the deficit against the **zonal mean at the same latitude**,
taken over the LPS domain only. Differencing along a latitude removes the
meridional trough structure and leaves zonal anomalies — which is what a closed
low actually is. Also require the point to be a minimum along its latitude row,
not just within a 2-D window.

**Fault 2 — the Thar heat low is a closed low.** After fix 1 the detector still
reported e.g. 7 Jul 2024, 27°N 68°E, 989 hPa — the semi-permanent summer heat
low over the Thar/Pakistan. Location cannot exclude it: the genuine Aug 2024
Gujarat deep depression sat at 23°N 70°E, right beside it.

*Fix:* physics. A monsoon depression rains; a heat low is dry. Require mean
rainfall ≥ 5 mm/day within 3° of the centre. The heat low disappears and the
Gujarat depression survives.

**After both fixes,** calibrated to `MIN_DEFICIT_HPA = 3.0`:

| | before | after |
|---|---|---|
| frequency | 89.3% of days | **29.5%** (published ~30–45%) |
| mean centre | 24.9°N, 79.5°E | **22.6°N, 81.9°E** |

22.6°N 81.9°E is the published Bay-head-to-central-India corridor. Frequency is
stable across seasons (2024: 29.5%, 2025: 30.3%), which argues against
overfitting to one year.

Verified against real events. The strongest detections of JJAS 2024:

| date | deficit | centre | MSLP | centre rain |
|---|---|---|---|---|
| 15–16 Sep | 10.3 hPa | 23°N 88°E | 992 hPa | 27 mm |
| 9 Sep | 8.3 hPa | 19°N 86°E | 995 hPa | 18 mm |
| 27–30 Aug | 6.7–8.5 hPa | 23–25°N 70–72°E | 995 hPa | 18–26 mm |

The 27–30 Aug system is the Gujarat deep depression that caused major flooding;
9 Sep at 19°N 86°E off the Odisha coast is textbook Bay-head genesis.

**Caveat:** spec 4.4 wants the thresholds tuned against the Zenodo LPS track
catalogue. That catalogue covers 1979–2019 and does not reach 2024, and it was
not downloaded. Calibration here is against published *frequency and genesis
location*, which is weaker. Say "calibrated to reproduce published LPS
frequency and genesis location", not "tuned against the catalogue".

### 28. Labels reach 4 classes — exactly spec 2.1's Level-1 target

`labels/assemble.py` combines the published active/break index with the LPS
detector into soft labels. JJAS 2024+2025, 244 days:

| regime | argmax days | share | |
|---|---|---|---|
| WEAK_TRANSITION | 141 | 57.8% | |
| LPS | 73 | 29.9% | |
| ACTIVE | 17 | 7.0% | |
| BREAK | 13 | 5.3% | |
| WD | 0 | 0.0% | not labellable — needs 500 hPa geopotential |
| EASTERLY_COASTAL | 0 | 0.0% | not labellable — needs analysis-mesh wind |

**14 days (5.7%) are blends**, implemented exactly as spec 6.1's worked example
— `{LPS: 0.6, ACTIVE: 0.4}`. The 25–29 Aug 2024 run is the Gujarat depression
embedded in an active spell, which is precisely the case spec 1.5 argues must
not be forced to a single class.

Spec 2.2 wants six classes; spec 2.1's **Level 1 wants four: ACTIVE, BREAK,
LPS, OTHER**. Four is what the data supports, so Level 1 is met exactly rather
than approximated. The two absent classes stay in the contract vector at
probability zero and are reported with a support count of zero — a six-wide
vector with two visibly dead slots is honest; a silently four-wide one is not.

### 29. ★★ The Regime Error Atlas — USP 1, built and measuring the thesis

`verify/atlas.py`. Stratified by the **true** regime rather than the predicted
one, because the Atlas measures conditional model error; it is not a test of a
predictor. Lead 3, validation block, scored out of sample.

The three cells with sufficient events (spec 11.4's ≥30 bar):

| regime × zone | obs mean | **raw bias** | raw RMSE | raw CSI | ΔCSI corrected |
|---|---|---|---|---|---|
| ACTIVE × W | 27.61 mm | **−10.66** | 22.73 | 0.023 | **+0.091** |
| LPS × W | 22.32 mm | **−5.12** | 19.52 | 0.109 | **+0.233** |
| WEAK × W | 14.29 mm | **+0.98** | 14.55 | 0.281 | **−0.203** |

**This is the project's central claim, measured.** Same model, same zone, and
the bias swings **11.6 mm** between regimes — a 10.7 mm under-forecast during
active monsoon on the west coast, a slight over-forecast in quiet conditions.
Raw CSI collapses from 0.281 in quiet conditions to 0.023 in active spells:
the model is worst exactly when it matters most.

Note also that correction **helps in the wet regimes (+0.09, +0.23) and hurts
in the quiet one (−0.20)**. That is a real, reportable limitation and it is
exactly the "one cell where we did not improve" that spec 19 slide 10 says to
mark visibly. Do not hide it — spec 10.2 is right that pointing at it buys more
credibility than any other single act.

**The honest limitation: only 3 of 12 cells clear the 30-event bar.** All three
are zone W. Zones C and N have 0–17 heavy-rain events each in the validation
block. The Atlas as a national artefact needs all-India (spec 2.2) and both
halves of 2025; at present it is a west-coast finding with the rest greyed out.
Show it greyed — a heatmap with honest gaps is more persuasive than a full one
built on four events per cell.

### 30. ★★★ THE USP, PROPERLY TESTED — the gate earns its place at lead 3

Stage A rebuilt on the real forecast mesh with 4 classes including LPS. This is
the test item 26 could not run: that attempt had 3 rainfall-only classes, no
LPS, and a classifier barely above its base rate.

**Stage A, validation (n=305), accuracy 0.620 vs base rate 0.525:**

| regime | precision | recall | F1 | support |
|---|---|---|---|---|
| **LPS** | **0.621** | **0.482** | **0.543** | 85 |
| WEAK_TRANSITION | 0.653 | 0.894 | 0.755 | 160 |
| ACTIVE | 0.278 | 0.111 | 0.159 | 45 |
| BREAK | 0.000 | 0.000 | 0.000 | 15 |

LPS is genuinely predicted from forecast fields alone — F1 0.543 on 85 days.
ACTIVE is weak and BREAK is not predicted at all (15 days is too few).

**Ablation 4 vs 5, validation:**

| lead | RMSE(4) | RMSE(5) | corr(4) | corr(5) | CSI(4) | CSI(5) | BSS(4) | BSS(5) |
|---|---|---|---|---|---|---|---|---|
| 1 | 11.008 | **10.859** | 0.7087 | **0.7171** | 0.2609 | 0.2610 | 0.2316 | 0.2318 |
| 3 | 12.637 | **12.402** | 0.5916 | **0.6130** | 0.1386 | **0.1667** | 0.1039 | **0.1243** |
| 5 | 13.221 | **13.129** | 0.5614 | **0.5715** | 0.1234 | 0.1181 | 0.0846 | **0.0865** |

**Decision gate (spec 6.3), margin 0.01 CSI:**

* **lead 3: +0.0281 — GATE HELPS.** Nearly 3x the noise margin, a 20% relative
  CSI gain. POD **doubles**, 0.223 → 0.447, for a modest FAR cost
  (0.732 → 0.790). For disaster management that is the right trade.
* lead 1: +0.0001 — tied.
* lead 5: −0.0054 — tied.

RMSE, correlation and Brier skill improve at every lead; only CSI at lead 3
clears the margin.

**Honest statement of the result:** the regime block helps at lead 3, the lead
at which operational decisions are actually made (spec 11.4's own choice), and
is indistinguishable from noise at leads 1 and 5. A plausible reading is that
lead 1 raw skill is high enough that regime information adds little, while at
lead 5 the classifier is too noisy to help — leaving lead 3 as the window where
raw skill has degraded but regime prediction is still reliable. **That is a
reading, not a finding**; three leads is not enough to establish a shape.

**Do not over-claim this.** It is one lead of three above the noise margin, on
188 events, on the validation block with in-sample calibration, at 64.5 mm
rather than spec 11.4's nominated 115.6 mm. The honest sentence for the deck is:
*"Regime conditioning improves heavy-rain CSI by 20% at day 3, the operational
decision lead; at days 1 and 5 the difference is within noise."*

Bias moves the wrong way (−0.23 → +0.82). Worth noting rather than hiding.

### 31. ★ The leakage canary's premise is wrong for a regime label — and here is the proof

The canary fired again (0.574 / 0.607 / 0.607 / 0.689 / 0.623 across leads,
ratio 1.086). That shape is **flat with noise**, not rising: at 61 validation
days per lead the standard error at p≈0.62 is ±0.062, so the whole spread is
about one SE.

A transition-vs-persistence test (does accuracy decay on regime-change days
while staying flat on persistence days?) was **underpowered and inconclusive** —
20 transition days per lead, SE ±0.11, ratios 1.111 and 1.080. It resolved
nothing and is recorded here so nobody re-runs it expecting an answer.

**What did resolve it: the forecast mesh is demonstrably lead-dependent.**

| lead | MSLP error vs analysis | precip error |
|---|---|---|
| 1 | 0.577 hPa | 8.50 mm |
| 2 | 0.721 hPa | 9.56 mm |
| 3 | 0.898 hPa | 10.37 mm |
| 4 | 1.088 hPa | 11.50 mm |
| 5 | **1.276 hPa** | 11.09 mm |

Monotonic, textbook error growth. Lead-5 precipitation correlates 0.613 with
lead-1, not 1.0. So these are genuine forecasts at genuine lead times — there
is no ingest fault and, since every feature derives from
`<var>_previous_dayN`, no path by which an observation could enter.

**And the numbers explain the flatness exactly.** MSLP error at day 5 is
**1.28 hPa**; the LPS detector fires on a **3.0 hPa** deficit. The error is less
than half the signal, so a closed low is nearly as detectable at day 5 as at
day 1.

That is not leakage. It is the premise of the project — spec 3's USP 2 claims
the regime is *predictable at lead time*, and this measures why: a monsoon
depression is a large-scale, persistent feature, and large-scale features are
exactly what NWP forecasts best.

**Recommendation:** spec 5.7's canary is sound for the CORRECTOR, whose skill
does decay (raw RMSE 12.93 → 17.32, ours 11.01 → 13.22 across leads 1–5). It is
the wrong test for the regime classifier. Either apply it only to the corrector,
or keep it on Stage A as a warning rather than a build failure, with this
measurement cited. **Do not simply raise the threshold until it passes** — the
canary is cheap insurance and a silently relaxed one is worse than none.

---

## 2026-09-23 — FINAL EVALUATION, held-out block, opened once

`scripts/final_eval.py`, run_id `20260923T120058Z`. Models fitted on train,
calibrated on validation, **frozen with a MANIFEST before the test block was
opened**, then evaluated once on JJAS 2025 Aug–Sep (60,085 district-days,
168 heavy-rain events at lead 3). Every number below is out of sample.

### 32. ★★★ The system works on held-out data

Lead 3, α = 0.15, threshold 64.5 mm:

| config | bias | RMSE | MAE | corr | POD | FAR | CSI | BSS |
|---|---|---|---|---|---|---|---|---|
| 0 climatology | −1.21 | 15.67 | 9.33 | 0.174 | 0.000 | 1.000 | 0.000 | — |
| 1 raw ECMWF | −0.39 | 13.76 | 7.65 | 0.502 | 0.065 | 0.756 | 0.054 | — |
| 2 multi-model mean | −0.57 | 13.30 | 7.44 | 0.533 | 0.089 | 0.769 | 0.069 | — |
| 3 quantile mapping | −0.88 | 14.76 | 7.87 | 0.518 | 0.238 | 0.753 | 0.138 | — |
| 4 RAAHAT no-regime | −0.93 | 12.87 | 7.15 | 0.557 | 0.208 | **0.514** | 0.171 | 0.105 |
| **5 RAAHAT full** | −0.54 | **12.64** | 7.16 | **0.578** | **0.339** | 0.723 | **0.180** | **0.113** |

* **CSI 0.054 → 0.180 against raw ECMWF — 3.3×.**
* RMSE −8.1%, correlation 0.502 → 0.578, BSS +0.113 against climatology.
* Quantile mapping again inflates RMSE (14.76 vs 13.76 raw) while improving
  CSI — the conventional post-processor is not a free win, on test as on
  validation.
* Raw ECMWF POD at day 3 is **0.065**. It catches one heavy-rain event in
  fifteen. That is the floor the project exists to raise.

### 33. ★★★ SPEC 11.4's DECISIVE TABLE — the gate earns its place in LPS conditions

Per-regime CSI at 64.5 mm, lead 3, held-out:

| regime | n district-days | n events | raw | no-regime | **full** |
|---|---|---|---|---|---|
| **LPS** | 2,561 | **36** | 0.023 | 0.171 | **0.232** |
| WEAK_TRANSITION | 7,683 | **107** | 0.046 | 0.139 | 0.142 |
| ACTIVE | 985 | 24 | — insufficient (spec 11.4) | | |
| BREAK | 788 | 1 | — insufficient | | |

**This is the result the project was built to produce.**

* In **LPS** conditions the gate adds **+0.061 CSI (0.171 → 0.232, +36%
  relative)** over the identical architecture with regime information removed.
  That is six times the 0.01 noise margin, on 36 held-out events.
* Against the raw model in LPS conditions, CSI goes **0.023 → 0.232, a factor
  of ten**.
* In **WEAK_TRANSITION** — the featureless residual class — the gate adds
  **+0.003, nothing.**

That contrast is the whole argument, and it is the right shape: regime
conditioning helps precisely where there is a distinctive regime to condition
on, and does nothing where there is not. An aggregate-only gain would have been
much weaker evidence; a gain concentrated in the class with the largest
conditional bias is what the theory predicts.

**Aggregate CSI gain is only +0.009 (0.171 → 0.180), below the margin — TIED.**
Report both. The honest framing: *"Aggregated across all conditions the gate is
within noise. Stratified by regime, it improves heavy-rain CSI by 36% in
low-pressure-system conditions, which is where the conditional bias lives."*
That is stronger AND more honest than quoting the aggregate alone.

### 34. Calibration is UNDER-confident out of sample — the opposite of the known failure

Reliability of P(>64.5 mm), lead 3, test block, calibrator fitted on validation:

| bin | forecast | observed | n |
|---|---|---|---|
| 0.00–0.12 | 0.009 | 0.009 | 11,801 |
| 0.12–0.25 | 0.160 | 0.186 | 172 |
| **0.25–0.38** | **0.291** | **0.628** | 43 |
| 0.50–0.62 | 0.500 | 1.000 | 1 |

As predicted (item 20), the near-perfect in-sample diagram did not survive. The
top populated bin is badly **under-confident**: the model says 29%, reality was
63%.

Worth noting for the pitch: spec 3's USP 3 cites recent work finding Indian
operational AI forecasts **over-confident**. Ours errs the other way on
held-out data. Under-confidence in a warning system means missed alerts, which
is the worse direction for disaster management — so this is a limitation to
state plainly, not a point to claim.

Likely cause: the isotonic calibrator was fitted on Jun–Jul and applied to
Aug–Sep, which is the non-exchangeability caveat recorded in item 10. It is the
cost of splitting one season in two, and it goes away once IMD publishes 2026
and whole-season splits become possible.

### 35. The Atlas does not survive the test block's sample size

Only 2 of 12 cells clear the 30-event bar on test (WEAK×N n=64, WEAK×W n=34),
versus 3 on validation. LPS×W has 21 events, LPS×C has 15 — just short.

The Atlas remains the right USP-1 artefact and the method is sound, but **at
197 districts over half a season it cannot be populated**. It needs all-India
(spec 2.2, ~3.8× the events) before it can be presented as a national artefact.
Show it greyed out and say so; a heatmap with honest gaps is more persuasive
than one built on four events per cell.

### 36. Provenance

`models/frozen/20260923T120058Z/MANIFEST.json` records config SHA-256s (this is
not a git repository, so configs are fingerprinted instead of a commit), data
file hashes, the split definition, every hyperparameter, and the headline
metrics per lead. `frozen_before_test_opened: true` is written before the
unlock, so the ordering is auditable rather than merely asserted.

Runtime: 649 s for the full fit-freeze-evaluate cycle, CPU only, no GPU.
