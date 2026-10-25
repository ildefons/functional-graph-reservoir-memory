#!/usr/bin/env python3
"""Reproduce manuscript Table 1 (finite-size row-participation diagnostic).

q_i(tau) = sum_{j != i} p_ij(tau)^2 for the row-softmax allocation of i.i.d. standard Gaussian scores.

* N = 60 and 500: the same 100 substrates per size as dominance/measure_dominance.py
  (seed rule 3320923 + N*100003 + seed, standard_normal((N, N)), diagonal excluded),
  so these rows use exactly the matrices behind Fig. 2 / Table 4.
* N = 50,000: 2,000 independent receiving rows, each with N-1 = 49,999 i.i.d. incoming scores,
  drawn from numpy default_rng(SEED_LARGE); no full recurrent matrix is constructed.

Usage: python measure_participation.py [output_dir]   (default: directory of this script)
Writes participation_per_seed.csv and participation_summary.csv.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
TAUS = [0.08, 0.15]
SEED_LARGE, N_LARGE, ROWS_LARGE, CHUNK = 20261020, 50_000, 2_000, 100

rows = []
for n in (60, 500):
    for seed in range(100):
        rng = np.random.default_rng(3320923 + n * 100003 + seed)
        a = rng.standard_normal((n, n)); np.fill_diagonal(a, -np.inf)
        for tau in TAUS:
            q = (softmax(a / tau, axis=1) ** 2).sum(axis=1)
            rows.append(dict(N=n, sampling="100 graphs", unit=seed, tau=tau, q_mean=q.mean()))

rng = np.random.default_rng(SEED_LARGE)
q_large = {tau: [] for tau in TAUS}
for start in range(0, ROWS_LARGE, CHUNK):
    a = rng.standard_normal((CHUNK, N_LARGE - 1))       # incoming scores of CHUNK independent rows
    for tau in TAUS:
        q_large[tau].append((softmax(a / tau, axis=1) ** 2).sum(axis=1))
for tau in TAUS:
    q = np.concatenate(q_large[tau])
    rows += [dict(N=N_LARGE, sampling=f"{ROWS_LARGE:,} rows", unit=i, tau=tau, q_mean=v) for i, v in enumerate(q)]

d = pd.DataFrame(rows)
d.to_csv(OUT / "participation_per_seed.csv", index=False)
s = d.groupby(["N", "tau", "sampling"], as_index=False).agg(q_bar=("q_mean", "mean"), q_se=("q_mean", "sem"))
s["alpha_N"] = s["tau"] * np.sqrt(2 * np.log(s["N"] - 1))
s["one_minus_alpha"] = 1 - s["alpha_N"]
s = s.sort_values(["tau", "N"])[["N", "tau", "sampling", "alpha_N", "q_bar", "q_se", "one_minus_alpha"]]
s.to_csv(OUT / "participation_summary.csv", index=False)
print(s.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
