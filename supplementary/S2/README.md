# Online Resource 2 (S2): additional results for Natural Computing v3.3.58

Supports **Table 4 (local single-node placement)** and **Table 7 (noise-aware Lyapunov memory)**. This archive is supplementary to, not a substitute for, Online Resource 1.

## Reproduce tables from supplied source records
```bash
python local_input/report_table3.py local_input/results.csv
python noise_aware/report_table6.py noise_aware/tcr_noise_pilot_results.csv
```
Requires Python 3.11+, NumPy, SciPy, pandas. These reporting commands were executed successfully during packaging; outputs `table3.csv`, `table6.csv`, and `table6_contrasts.csv` are included. See `REPORTS_REPRODUCED.txt`. Historical output filenames (`table3.csv` and `table6.csv`) are retained unchanged.

## Full reruns
```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python local_input/tcr_local_rule_confirm.py
python noise_aware/tcr_noise_pilot.py
```
The scripts now write next to their source files, not to `/mnt/data`; original unmodified scripts are preserved alongside with suffix `_archived_original.py`. A full numerical rerun can be expensive (particularly the exhaustive N=500 placement experiment) and was **not** repeated as part of this packaging step.

`local_input/`: N=150 and N=500, both recurrent scales, 44 fresh networks, three noise levels, exact zero-temperature topology-only dimension rule and exhaustive oracle. Signal-history covariance is truncated with documented decay bounds; the noise covariance is obtained through Lyapunov solving. The numerical confirmation does not establish finite-noise optimality.

`noise_aware/`: N=60, 20 matched seeds, Gaussian and cycle controls. Both signal and state-noise covariances are calculated using discrete Lyapunov equations. This is a manuscript result, not a post-manuscript exploratory study.

Original 2017-era or v3.3.7 text in the archived README files may describe an earlier project status; this top-level README supersedes those historical notes.
