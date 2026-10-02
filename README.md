# RAAHAT

RAAHAT is a district rainfall forecast desk. It presents archived, regime-aware rainfall corrections, exceedance probabilities, and suggested warning colours alongside an interactive India map, evidence views, and a sortable district table.

## Run locally

The frontend uses Vite and the API uses FastAPI. Install the API dependencies into a Python environment, then start both processes from the repository root:

```bash
pip install -r requirements.txt uvicorn
PYTHONPATH=src uvicorn raahat.api.main:app --port 8000
```

In a second terminal:

```bash
npm --prefix frontend ci
npm --prefix frontend run dev
```

Open `http://localhost:5173`. On PowerShell, set `$env:PYTHONPATH="src"` before starting Uvicorn.

## Deploy on Vercel

Import this GitHub repository into Vercel with the root directory set to `/`. The root `vercel.json` builds the Vite frontend, serves `api/index.py` as a Python function, and routes `/api/v1/*` to the API. The frontend build copies the simplified district GeoJSON into its public assets, so the map loads from the CDN.

The API reads checked-in, precomputed Parquet files. Rebuild those files locally when model data changes; the deployment does not run training or ingest jobs. See [deployment details](docs/DEPLOY_VERCEL.md) for the function size and runtime constraints.

The UI labels the data as an archive replay. Suggested colours support review and are complementary to official IMD warnings.
