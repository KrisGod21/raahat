"""Vercel serverless entrypoint.

Vercel's Python runtime looks for files under /api at the project root and,
when the module exposes an ASGI callable named `app`, serves it directly --
so this file only has to make `src/` importable and re-export the real app.

Everything RAAHAT serves is precomputed (spec 8), so a read-only, stateless
function is a genuine fit: no model inference happens in a request handler.
What does NOT run here is the whole offline side -- fetching forecasts,
building features, training, rebuilding predictions. Those write files, take
minutes, and must run on your machine or in CI; the Parquet they produce is
deployed as part of the bundle.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raahat.api.main import app  # noqa: E402

__all__ = ["app"]
