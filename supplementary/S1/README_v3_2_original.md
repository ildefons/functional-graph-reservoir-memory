# Online Resource 1: Supplementary Software S1

**Article:** From Functional Graphs to Dense Reservoirs: Connectivity Concentration and Accessible Memory  
**Journal:** Natural Computing (submission draft)  
**Author and corresponding author:** Ildefons Magrans de Abril  
**Affiliation:** Universitat Politècnica de Catalunya - BarcelonaTech (UPC), Barcelona, Spain  
**Contact:** ildefons.magrans@gmail.com


This material accompanies **From Functional Graphs to Dense Reservoirs: Connectivity Concentration and Accessible Memory**, v3.1 (17 October 2026). It has not yet been published as a separate GitHub release.

## Contents
- `graph/`: Exact polynomial graph scaling for 700 graphs, code, verification and tables, unchanged from v2.
- `dominance/`: Full-row finite-temperature softmax dominance experiment; 100 seeds per size, unchanged from v2.
- `memory/`: New paired 20-seed experiment with both RMS scales `s0=0.6` and `s0=0.8`, lag zero included, linear/tanh systems, primary SVD cutoff `1e-9` and alternative `1e-11`. Raw per-seed, per-delay and summary CSV. Graph generation and data streams match v2. The recurrent matrices are global-RMS scaled; spectral radii are measured, **not** forcibly fixed.
- `matched/`: Previously independently calibrated 80-reference/13-architecture arms evaluated on fresh seeds 400-419, plus **new** direct-input 30-lag NARMA10 baseline under the same frozen generator and splits, validation-only ridge selection.
- `v3_checks.json`: Cross-version reproduction and stability audit.
- `SHA256SUMS.txt`: Cryptographic hashes of all supplementary content, excluding the hash file itself.

## Reproduction (Python >=3.11, NumPy, SciPy, pandas, Numba, matplotlib)

```sh
pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
python graph/graph_exact_scaling.py --seeds 100 --out graph
python dominance/measure_dominance.py
python memory/run_memory_scale_sensitivity.py --out memory
python matched/run_linear_narma_fresh.py
# To rerun the complete previously frozen baseline evaluation:
python matched/run_fresh_matched.py --stage evaluate --out matched
python matched/audit_fresh.py
```

`memory/run_memory_scale_sensitivity.py` reruns both scales from scratch; on a CPU it may take minutes. Re-running commands overwrites some results; verify checksums **before** rerunning, or use a copy. The original v2 fixed-scale memory CSVs are preserved in the separate v2 package, and are verified numerically against the new repeated s0=0.6 experiment.

## Inferential boundaries
1. Exact polynomial degree and sqrt(N) expected rank are theorems for the **linear zero-temperature** reservoir with generic scalar input weights, not for tanh dynamics.
2. The measured SVD-based total memory includes **lag zero**, and depends on the finite readout threshold, data length, and gain. The original v2 table excluded lag zero.
3. Finite-temperature dominance is a descriptive 90% single-row share statistic, not a finite-temperature capacity theorem. Dominance and memory were measured in separate substrate ensembles.
4. The fresh-seed 400-419 baseline evaluation was performed after inspecting earlier pilot results; it is not prospectively preregistered. The direct linear-lag model has seven validation-only ridge candidates, not the ESN architecture-tuning budget.
5. Retaining the archived `neurocomputing-v1.0` release intact is intentional. The present S1 supplement must accompany any new submission and be released separately if the author chooses.

## v3.1 preparation note
Only manuscript wording and memory-plot axis labels changed from v3; experimental CSV files, seed allocations, and numerical conclusions are unchanged. Memory figures are reproduced using `python memory/regenerate_memory_figures.py` (default output: `regenerated_figures/`; override with `TCR_FIGURE_OUTPUT_DIR=/path/to/manuscript`).


## v3.2 theory/presentation additions
No new benchmark experiments were run. The new plotted `dominance_scaled.pdf` and its plotting script reuse the v3.1 recorded data. `graph_dimension_sd.csv` and `memory_linear08_sd.csv` compute descriptive sample standard deviations directly from per-seed data. Mathematical positive-temperature propositions are supplied in the v3.2 manuscript; their scope is fixed positive temperature for algebraic controllability, and scaled temperature for Gaussian Gibbs statistics.
