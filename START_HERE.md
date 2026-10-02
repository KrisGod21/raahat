# RAAHAT — deployment package

Regime-Aware Adjustment of Heavy-rainfall Alerts & Thresholds. An ML
post-processing layer that corrects numerical weather forecasts of rainfall for
Indian districts, conditioned on the prevailing synoptic weather regime.

This package contains everything needed to **run and deploy** the system. Full
deployment instructions are in **[docs/DEPLOY_VERCEL.md](docs/DEPLOY_VERCEL.md)**.

---

## Deploy to Vercel

From this folder:

```bash
npm i -g vercel
```

```bash
vercel login && vercel --prod
```

`vercel.json` supplies the build and function configuration; accept the
detected settings. There is no git repository here, so the CLI (which uploads
files directly) is the shortest path. `docs/DEPLOY_VERCEL.md` covers the GitHub
route and the constraints worth knowing before you start.

---

## Run it locally first

Two processes. Backend:

```bash
pip install -r requirements.txt && PYTHONPATH=src uvicorn raahat.api.main:app --port 8000
```

Frontend, in a second terminal:

```bash
cd frontend && npm ci && npm run build && npm run preview
```

Then open **http://localhost:4173**. The frontend proxies `/api` to port 8000,
so start the backend first. On Windows PowerShell, set `PYTHONPATH` with
`$env:PYTHONPATH = "src"` on its own line rather than inline.

To check the backend on its own: <http://localhost:8000/api/v1/health> should
report `"ready": true`.

---

## What is in here

| | |
|---|---|
| `src/raahat/` | the system: ingest, features, regime classifier, corrector, decision layer, API |
| `frontend/` | React + Vite + MapLibre interface |
| `api/index.py` | Vercel serverless entrypoint; re-exports the FastAPI app |
| `data/processed/` | precomputed predictions and features (Parquet) — what the API serves |
| `data/static/` | district boundaries, master list, terrain, neighbours |
| `results/final/` | frozen held-out evaluation; the Evidence and Atlas screens read these |
| `config/` | zones, regimes, features, model settings (YAML) |
| `tests/` | 77 tests |
| `docs/` | deployment guide and DATA_NOTES.md, the running record of findings |
| `README_SIH.md` | the full build specification |

Verify the package on arrival:

```bash
PYTHONPATH=src python -m pytest -q
```

Expect **76 passed, 1 xfailed**.

---

## What is deliberately NOT in here

- `data/raw/` and `data/interim/` — ingest inputs and intermediates, several
  hundred MB, never read at request time
- `models/` — trained model artefacts (225 MB). Predictions are already
  computed into `data/processed/`, so the API does not need them. You need them
  only to re-run the batch prediction step.
- the source IMD NetCDF files (76 MB)
- `node_modules/` and `frontend/dist/` — recreated by `npm ci` and `npm run build`

If you need the full working tree (to retrain or to rebuild predictions), ask
for those directories separately rather than trying to regenerate them — the
forecast archive takes hours to refetch from Open-Meteo.

---

## Two things to know before you demo

**It is a replay, not a live system.** The interface says so in the header and
must keep saying so. Everything served is precomputed over JJAS 2024 and 2025;
there is no live feed.

**The map draws all 641 districts but only 197 have predictions.** The rest are
drawn pale and labelled "no data" in the legend, and clicking one says the
archive does not cover it. That is honest and intentional — the all-India
forecast backfill was not finished. Districts with data are concentrated in
central India, the Western Disturbance belt and the west coast.

---

Verified before packaging: 76 tests pass, the frontend builds clean from this
folder, and all 12 API endpoints serve correctly using only the files included
here.
