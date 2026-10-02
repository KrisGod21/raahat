# RAAHAT -- spec 15. `make all` must reproduce every number in the presentation.
#
# Ordering is the contract here, not a convenience. districts must be frozen
# before features exist; features before regimes; regimes before ablations; and
# final-eval LAST, because it is the only target that opens the held-out block
# (spec 7.1) and it freezes the models before doing so.
#
# Every network step is cached by URL hash, so re-running is free and works
# offline (spec 10.3). Nothing below re-fetches what is already on disk.

PY      := PYTHONPATH=src python
DATA    := data
STATIC  := $(DATA)/static
INTERIM := $(DATA)/interim
PROC    := $(DATA)/processed

SEASONS := 2024 2025

.PHONY: all data features regimes train eval atlas demo test clean-derived help

help:
	@echo "make districts  freeze the district master (spec 21 item 2) -- run once"
	@echo "make terrain    per-district elevation, slope, coast distance"
	@echo "make data       IMD truth + district forecasts + synoptic meshes"
	@echo "make features   contract-conforming feature tables (spec 13.2)"
	@echo "make regimes    labels -> Stage A -> regime probabilities"
	@echo "make eval       ablations on VALIDATION (test block stays locked)"
	@echo "make final      THE HELD-OUT EVALUATION. Opens the test block. Once."
	@echo "make test       pytest"
	@echo "make all        everything except final"

# ----------------------------------------------------------------- static ---

$(STATIC)/districts_master.csv:
	$(PY) scripts/build_districts.py

districts: $(STATIC)/districts_master.csv

$(STATIC)/district_terrain.parquet: $(STATIC)/districts_master.csv
	$(PY) scripts/build_terrain.py

terrain: $(STATIC)/district_terrain.parquet

# ------------------------------------------------------------------- data ---

$(INTERIM)/district_obs_2024.parquet: $(STATIC)/districts_master.csv
	$(PY) scripts/fetch_forecasts.py --seasons $(SEASONS)

$(INTERIM)/mesh_forecast_2024.parquet: $(STATIC)/districts_master.csv
	$(PY) scripts/fetch_mesh.py --seasons $(SEASONS)

data: $(INTERIM)/district_obs_2024.parquet $(INTERIM)/mesh_forecast_2024.parquet

# --------------------------------------------------------------- features ---

$(PROC)/features_2024.parquet: $(INTERIM)/district_obs_2024.parquet \
                               $(STATIC)/district_terrain.parquet
	$(PY) -c "from raahat.features import build; from raahat import contract as C; \
	  [build.build_season(s).to_parquet(C.DATA_DIR/'processed'/f'features_{s}.parquet', \
	   index=False) for s in ($(shell echo $(SEASONS) | tr ' ' ','))]"

features: $(PROC)/features_2024.parquet

# ---------------------------------------------------------------- regimes ---

$(PROC)/regime_inputs.parquet: $(PROC)/features_2024.parquet \
                               $(INTERIM)/mesh_forecast_2024.parquet
	$(PY) scripts/build_regimes.py --seasons $(SEASONS) --out models/frozen/dev

regimes: $(PROC)/regime_inputs.parquet

# ------------------------------------------------------------ evaluation ---

eval: regimes
	$(PY) -c "import warnings; warnings.filterwarnings('ignore'); \
	  from raahat.train import splits, ablations; \
	  tr=splits.load_split('train'); va=splits.load_split('validation'); \
	  t,i=ablations.run(tr[tr.lead_day==3], va[va.lead_day==3], verbose=False); \
	  print(ablations.format_table(t,[64.5])); print(); print(i['verdict']); \
	  t.to_csv('results/tables/ablations_lead3_validation.csv', index=False)"

# THE test block. Opens it once, after freezing the models (spec 7.1, 13.3).
final: regimes
	$(PY) scripts/final_eval.py

synthetic:
	python scripts/make_synthetic.py

test:
	python -m pytest -q

all: districts terrain data features regimes eval test
	@echo
	@echo "all targets green. 'make final' opens the held-out block -- once."

# Removes DERIVED artefacts only. It deliberately does NOT touch data/raw:
# imdpune.gov.in is intermittent and Open-Meteo is quota-limited, so those
# caches are expensive to rebuild and the demo depends on them (spec 10.3).
clean-derived:
	rm -f $(PROC)/*.parquet $(INTERIM)/*.parquet
	@echo "derived data removed; data/raw/ caches kept deliberately"
