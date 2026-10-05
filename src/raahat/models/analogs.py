"""Analog Memory -- USP 4 (spec 3).

"Today's pattern most closely resembles 26 July 2005 and 16 Aug 2024; on those
days this district recorded 320 mm and 180 mm."

Spec 3 is right that this is the cheapest explainability device available and
the one judges quote back. It needs no training: nearest neighbours over a
small standardised feature vector, using sklearn rather than faiss because at
this scale faiss buys nothing and adds a dependency.

Two design points that matter:

  It retrieves WITHIN DISTRICT by default. "A similar day somewhere in India"
  is a much weaker statement than "a similar day here", and the observed
  outcome is only interpretable if the terrain is the same.

  It is DISPLAY ONLY. Analogs never feed the corrector. If they did, the
  system would be part nearest-neighbour model and the ablations would no
  longer isolate what they claim to isolate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from raahat import contract as C

#: Small, interpretable, and all available at forecast time (spec 3: a 32-dim
#: vector). Anything observed would make the analog a cheat rather than a memory.
DEFAULT_FEATURES = [
    "mm_mean", "mm_spread", "mm_agree_heavy",
    "pr_ifs", "pr_ifs_nbr_max", "pr_ifs_3day_sum",
    "p_ACTIVE", "p_BREAK", "p_LPS", "p_WEAK", "regime_entropy",
    "upslope_flux", "coast_onshore_flux",
    "clim_mean_district_doy", "doy_sin", "doy_cos",
]


class AnalogMemory:
    """Nearest historical days, per district, with what actually happened."""

    def __init__(self, features: list[str] | None = None, *, per_district: bool = True):
        self.features = features or list(DEFAULT_FEATURES)
        self.per_district = per_district
        self._index: dict = {}
        self._store: dict = {}

    def fit(self, history: pd.DataFrame) -> AnalogMemory:
        """Build the index from days whose outcome is KNOWN.

        Rows without an observation are dropped -- an analog whose outcome we
        cannot state is not an analog, it is a coincidence.
        """
        # sklearn (and its scipy dependency) are imported HERE, not at module
        # scope, because fit() is the only thing that needs them. The API
        # imports this module solely for narrative(), which is pandas and a
        # format string -- and sklearn + scipy are 162 MB of a serverless
        # bundle that would never execute a line.
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler

        use = [f for f in self.features if f in history.columns]
        if not use:
            raise KeyError("none of the analog features are present")
        self._used = use
        hist = history[history[C.TARGET_COL].notna()].copy()

        keys = hist["district_id"] if self.per_district else pd.Series(0, index=hist.index)
        for key, g in hist.groupby(keys):
            X = g[use].to_numpy(dtype="float64")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            if len(g) < 2:
                continue
            scaler = StandardScaler().fit(X)
            nn = NearestNeighbors(n_neighbors=min(10, len(g))).fit(scaler.transform(X))
            self._index[key] = (scaler, nn)
            self._store[key] = g[["valid_date", C.TARGET_COL]].reset_index(drop=True)
        return self

    def query(self, df: pd.DataFrame, *, k: int = 2, exclude_same_date: bool = True):
        """k nearest past days per row, with their observed rainfall.

        `exclude_same_date` stops a day retrieving itself when the query rows
        are part of the fitted history, which would otherwise produce a
        perfect, meaningless analog.
        """
        use = self._used
        n = len(df)
        dates: list[list] = [[] for _ in range(n)]
        obs: list[list] = [[] for _ in range(n)]
        sims = np.full(n, np.nan)

        # Batched per district. Row-by-row kneighbors took ~30 minutes over
        # 240k district-days; one query per district over its whole block is
        # the same arithmetic in a fraction of the time.
        groups = (df.groupby("district_id").indices if self.per_district
                  else {0: np.arange(n)})
        for key, pos in groups.items():
            if key not in self._index:
                continue
            scaler, nn = self._index[key]
            store = self._store[key]
            X = np.nan_to_num(
                df.iloc[pos][use].to_numpy(dtype="float64"),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            n_ask = min(k + (2 if exclude_same_date else 0), len(store))
            dist, idx = nn.kneighbors(scaler.transform(X), n_neighbors=n_ask)
            qdates = df.iloc[pos]["valid_date"].to_numpy() if "valid_date" in df else None
            sd = store["valid_date"].to_numpy()
            so = store[C.TARGET_COL].to_numpy()
            for r, p in enumerate(pos):
                for d, i in zip(dist[r], idx[r]):
                    if exclude_same_date and qdates is not None and sd[i] == qdates[r]:
                        continue
                    dates[p].append(sd[i])
                    obs[p].append(float(so[i]))
                    if np.isnan(sims[p]):
                        sims[p] = float(1.0 / (1.0 + d))  # display only
                    if len(dates[p]) == k:
                        break

        out = {}
        for i in range(1, k + 1):
            out[f"analog_date_{i}"] = [d[i - 1] if len(d) >= i else None for d in dates]
            out[f"analog_obs_{i}"] = [o[i - 1] if len(o) >= i else np.nan for o in obs]
        out["analog_similarity_1"] = sims
        return pd.DataFrame(out, index=df.index)


def narrative(row: pd.Series, district_name: str = "this district") -> str:
    """The sentence from spec 3. Templated; only the numbers vary."""
    pairs = []
    for i in (1, 2):
        d, o = row.get(f"analog_date_{i}"), row.get(f"analog_obs_{i}")
        if d is not None and o is not None and np.isfinite(o):
            pairs.append((d, o))
    if not pairs:
        return "No closely matching day was found in the archive."
    # "%-d" is a POSIX strftime extension and raises on Windows; strip the
    # leading zero by hand so the sentence reads "9 Sep" rather than "09 Sep"
    # on every platform.
    def _fmt(d):
        try:
            return pd.Timestamp(d).strftime("%d %b %Y").lstrip("0")
        except (ValueError, TypeError):
            return str(d)

    dates = " and ".join(_fmt(d) for d, _ in pairs)
    obs = " and ".join(f"{o:.0f} mm" for _, o in pairs)
    return (f"Today's pattern most closely resembles {dates}; "
            f"on those days {district_name} recorded {obs}.")
