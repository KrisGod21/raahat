"""SHA-keyed on-disk HTTP cache (spec 8).

Every external call goes through here. Two reasons, both non-negotiable:

  * Spec 10.3 -- the demo runs with the network unplugged. Anything the demo
    touches must already be on disk.
  * imdpune.gov.in and, occasionally, Open-Meteo are intermittent. Re-running
    the pipeline must never depend on a third party being up.

Cache key is a SHA-256 of the full request URL, so a changed parameter is a
different entry and a stale response can never masquerade as a fresh one.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from raahat import contract as C

CACHE_ROOT = C.DATA_DIR / "raw" / "_cache"
MANIFEST = C.DATA_DIR / "_manifest"
USER_AGENT = "raahat-sih/0.1 (+research prototype)"


class FetchError(RuntimeError):
    """Raised after all retries are exhausted. Never swallowed silently."""


def cache_path(url: str, namespace: str = "openmeteo") -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return CACHE_ROOT / namespace / digest[:2] / f"{digest}.json"


def log_error(msg: str) -> None:
    MANIFEST.mkdir(parents=True, exist_ok=True)
    with (MANIFEST / "errors.log").open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}  {msg}\n")


def get_json(
    url: str,
    *,
    namespace: str = "openmeteo",
    attempts: int = 4,
    pause: float = 2.0,
    timeout: float = 90.0,
    offline: bool = False,
) -> dict:
    """Fetch JSON, cached. Raises FetchError after `attempts` failures.

    Spec 0 rule 2: a failure is loud and logged. It never returns plausible
    filler, because filler that looks like data is the worst possible outcome.
    """
    path = cache_path(url, namespace)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            path.unlink()  # truncated write from an interrupted run

    if offline:
        raise FetchError(f"offline and not cached: {url}")

    last = ""
    delay = pause
    for i in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode("utf-8")
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("error"):
                raise FetchError(f"API error: {data.get('reason', data)}")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(raw, encoding="utf-8")
            tmp.replace(path)  # atomic: a crash never leaves a half-written entry
            return data
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code} {e.reason}"
            if e.code in (400, 404):
                break  # a bad request will not become good by repeating it
            if e.code == 429:
                # Rate limiting needs minute-scale backoff, not seconds. A few
                # seconds of retry against a 429 just burns the remaining quota
                # and guarantees the next call fails too. Honour Retry-After
                # when the server sends it.
                retry_after = e.headers.get("Retry-After") if e.headers else None
                try:
                    delay = float(retry_after) if retry_after else max(delay * 3, 60.0)
                except (TypeError, ValueError):
                    delay = max(delay * 3, 60.0)
                delay = min(delay, 600.0)
                log_error(f"http_cache: 429, backing off {delay:.0f}s")
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        if i < attempts:
            time.sleep(delay)
            delay = min(delay * 2, 600.0)

    log_error(f"http_cache: {last}  url={url}")
    raise FetchError(f"{last} after {attempts} attempts: {url}")


def cache_stats(namespace: str = "openmeteo") -> dict:
    root = CACHE_ROOT / namespace
    if not root.exists():
        return {"entries": 0, "bytes": 0}
    files = list(root.rglob("*.json"))
    return {"entries": len(files), "bytes": sum(f.stat().st_size for f in files)}
