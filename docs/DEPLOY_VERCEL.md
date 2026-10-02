# Deploying RAAHAT on Vercel

What ships: the Vite frontend as static files on Vercel's CDN, and the FastAPI
app as a single Python serverless function behind `/api/v1/*`.

What does **not** ship: everything offline — fetching forecasts, building
features, training, rebuilding predictions. Those write files and take minutes.
They run on your machine (or in CI); the Parquet they produce is deployed as
part of the function bundle.

The files that make this work are already in the repo: `vercel.json`,
`api/index.py`, `requirements.txt`, `.vercelignore`.

---

## Why this needed a code change first

A serverless function on Vercel gets **250 MB unzipped**. The development
environment is nowhere near that:

| package | size | used at request time? |
|---|---|---|
| torch | ~4.4 GB | no — CDF experiments only |
| scipy (via scikit-learn) | 120 MB | no |
| pyarrow | 84 MB | no — schema definitions |
| scikit-learn | 42 MB | no — analog *fitting* only |
| pandas | 65 MB | **yes** |
| duckdb | 37 MB | **yes** |
| numpy | 32 MB | **yes** |

Two of those were imported at module scope by code the API loads, which meant
importing the app pulled in 246 MB of libraries that never execute a line in a
handler:

- `raahat/contract.py` imported pyarrow for the Parquet schemas. The schemas
  are build-time artefacts; the API never calls one. It now goes through a
  lazy proxy, so `pa.field(...)` is unchanged but the import happens on first
  attribute access.
- `raahat/models/analogs.py` imported scikit-learn at module scope. The API
  only calls `narrative()`, which is pandas and a format string — `fit()` is
  the only thing that needs sklearn, so the import moved inside it.

After that the app imports cleanly with pyarrow, scikit-learn, scipy, lightgbm
and torch **all absent** (verified by blocking them at import time; the check
is written out in `requirements.txt`). Measured bundle:

```
dependencies   ~149 MB   (pandas 65, duckdb 37, numpy 32, FastAPI stack 11)
data            ~25 MB   (predictions + features Parquet, simplified boundaries)
src + config     ~0.2 MB
               --------
                ~174 MB   against a 250 MB limit
```

**If you ever add a top-level import of one of those five to a module the API
loads, local development stays green and the deployed function starts failing
at cold start.** Re-run the slim-import check in `requirements.txt` before
deploying.

---

## Before you deploy

The function serves precomputed files, so they must exist and be current:

```bash
ls data/processed/predictions_*.parquet data/static/districts_simplified.geojson
```

If those are missing or stale, rebuild them locally first — Vercel cannot.

---

## Deploy

The repository is initialized and can be imported from GitHub or deployed with
the Vercel CLI.

### Option A — Vercel CLI (no git needed)

```bash
npm i -g vercel
```

```bash
vercel login
```

From the project root:

```bash
vercel --prod
```

Accept the detected settings; `vercel.json` supplies the build and function
config. The first run asks which scope and project to use.

### Option B — GitHub

```bash
git init && git add -A && git commit -m "RAAHAT"
```

Push to a new GitHub repo, then **Add New → Project** in the Vercel dashboard
and import it. Leave the Root Directory as `/` — `vercel.json` handles the
rest. Do not set it to `frontend`, or `api/` and `data/` will not be uploaded.

The Parquet files total ~25 MB, comfortably under GitHub's 100 MB per-file
limit, so no Git LFS is needed.

---

## What the config does

```jsonc
{
  "installCommand": "npm --prefix frontend ci",
  "buildCommand":   "npm --prefix frontend run build",
  "outputDirectory": "frontend/dist",

  "functions": {
    "api/index.py": {
      "memory": 1024,
      "maxDuration": 30,
      "includeFiles": "{config,data,results,src}/**"
    }
  },

  "rewrites": [
    { "source": "/api/v1/(.*)", "destination": "/api/index" }
  ]
}
```

- `includeFiles` is required. Vercel traces Python imports statically, and
  `api/index.py` reaches `src/` by manipulating `sys.path` at runtime, so
  tracing alone would not bundle it. `config/` and `data/` are read by path and
  would never be traced either.
- The rewrite sends only `/api/v1/*` to the function. There is deliberately
  **no** catch-all rewrite to `index.html`: the app has no client-side router
  (the tabs are component state), and a catch-all placed above the API rule is
  the classic way to make every endpoint return HTML.
- `.vercelignore` keeps `data/raw/` (2,000+ cached Open-Meteo responses), the
  28 MB unsimplified boundary file, and the model artefacts out of the upload.

---

## Known constraints, honestly

**Python version.** Vercel pins its own Python (3.12 at time of writing); this
project develops on 3.13. The pinned versions in `requirements.txt` all have
3.12 wheels, but if a build fails resolving one, relax that pin rather than
forcing a Python version.

**Cold starts.** Importing pandas + numpy + duckdb and opening the Parquet is
roughly 2–4 s on a cold instance. Warm requests are the usual sub-100 ms.
For a judged demo, load the page once a minute before you present.

**`{config,data,src}/**` brace expansion.** Supported by the glob
implementation Vercel uses, but if the build logs show the data missing,
replace it with `"**"` — `.vercelignore` already keeps the upload small.

**Nothing may write to disk.** The filesystem is read-only apart from `/tmp`.
`db.py` points DuckDB's spill directory at `/tmp` when it detects Vercel. The
batch prediction rebuild cannot run there at all.

**`maxDuration: 30`** is within the Hobby plan's ceiling. Every endpoint
measures well under 100 ms warm, so this is headroom for cold starts, not a
requirement.

---

## Static district boundaries

The frontend build copies `data/static/districts_simplified.geojson` to
`frontend/public/districts.geojson`. The browser requests the CDN asset first
and falls back to `/api/v1/districts` during local development or if the static
asset is unavailable.

The same prebuild step generates `atlas.json` and `verification.json` from the
frozen CSVs in `results/final/`, and copies `data/static/about.json`. These three
reference screens load the static assets first and fall back to the equivalent
API endpoints. The build fails if any required source file is missing.

---

## The alternative worth considering

A read-only DuckDB-over-Parquet API is a reasonable serverless fit, but it is
not a natural one: the whole design assumes a process that stays up with the
Parquet already open. A small always-on container — Render, Fly.io, Railway —
gives warm queries, no bundle limit, no import-weight discipline to maintain,
and lets the batch rebuild run beside the API.

If you go that way, deploy only the frontend to Vercel and point it at the
container by replacing the rewrite:

```jsonc
"rewrites": [
  { "source": "/api/v1/(.*)", "destination": "https://<your-api-host>/api/v1/$1" }
]
```

`main.py` already has CORS middleware, so a direct cross-origin call works too.
