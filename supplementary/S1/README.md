# Online Resource 1 (S1): historical experiments for Natural Computing manuscript v3.3.58

This package contains the recovered historical v3.2 supplementary code, scripts, recorded outputs, experimental configurations, and figures for the manuscript **From Functional Graphs to Dense Reservoirs: Connectivity Concentration and Accessible Memory**. The experimental data are preserved without modification. The accompanying original README is `README_v3_2_original.md`.

## Navigation
- `graph/`: 700 graph instances, exact dimensions and Fig. 1 / Table 3 source results.
- `dominance/`: full-row dominance, Fig. 2 / Table 5 data and Table 2 participation diagnostics, including N=50,000 sampled rows.
- `memory/`: 20-seed memory experiment, sensitivity plots and Table 6 / Figs. 3–4 source results.
- `matched/`: independently calibrated fresh-seed benchmark comparisons, NARMA10 and Lorenz, Tables 8–9 source results.
- `requirements.txt`: Python dependencies; `SHA256SUMS.txt` hashes every content file.

## Reproduction
From the directory containing this README (preferably in a COPY, because scripts may overwrite recorded data):

```bash
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
python graph/graph_exact_scaling.py --seeds 100 --out graph
python dominance/measure_dominance.py
python dominance/measure_participation.py dominance
python memory/run_memory_scale_sensitivity.py --out memory
python memory/regenerate_memory_figures.py
python matched/run_linear_narma_fresh.py
python matched/run_fresh_matched.py --stage evaluate --out matched
python matched/audit_fresh.py
```

## Scope and limitations
Historical data and source are supplied as archived in the v3.2 supplement. The local input-placement study (Table 4) and the noise-aware Lyapunov study (Table 7) are provided in **Online Resource 2**.
Table 2 is reproducible using `dominance/measure_participation.py` and its archived per-seed and summary CSV files. The N=50,000 values use 2,000 independently sampled rows, not a full N-by-N matrix.
Execution of every large historical experiment has not been repeated during this packaging audit. Presence and checksums of archived sources and data do not certify successful end-to-end regeneration.
