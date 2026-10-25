"""Reproduce manuscript Table 6 and the paired contrasts quoted in Section 6.5
from noise_aware/tcr_noise_pilot_results.csv.

Outputs: mean population memory per architecture at eps = 1e-6, 1e-4, 1e-2; stable-seed counts;
spectral-radius ranges; and paired seed-level differences (temperature 3.5 minus graph,
simple cycle minus graph) with unadjusted Student-t 95% intervals.

Usage: python report_table6.py [path/to/tcr_noise_pilot_results.csv]
Writes table6.csv and table6_contrasts.csv next to the input file.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("tcr_noise_pilot_results.csv")
d = pd.read_csv(src)
d["arm_label"] = np.where(d["arm"] == "temperature", "Temperature " + d["tau"].astype(str), d["arm"])
d.loc[(d["arm"] == "temperature") & (d["tau"].astype(str).isin(["0", "0.0"])), "arm_label"] = "Graph"

EPS = [1e-6, 1e-4, 1e-2]
stable = d[d["total"].notna()]
rows = []
for label, g in d.groupby("arm_label", sort=False):
    s = g[g["total"].notna()]
    row = {"architecture": label,
           "stable_seeds": f"{s['seed'].nunique()}/{g['seed'].nunique()}",
           "rho_min": s["rho"].min(), "rho_max": s["rho"].max()}
    for e in EPS:
        row[f"eps={e:g}"] = s.loc[np.isclose(s["eps"], e), "total"].mean()
    rows.append(row)
table = pd.DataFrame(rows)
print(table.round(3).to_string(index=False))
table.round(4).to_csv(src.with_name("table6.csv"), index=False)

wide = stable.pivot_table(index=["seed", "eps"], columns="arm_label", values="total")
contrasts = []
for e in EPS:
    x = wide[np.isclose(wide.index.get_level_values("eps"), e)]
    for arm in ["Temperature 3.5", "cycle"]:
        diff = (x[arm] - x["Graph"]).dropna()
        m, se = diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff))
        h = stats.t.ppf(0.975, len(diff) - 1) * se
        contrasts.append({"eps": f"{e:g}", "contrast": f"{arm} - Graph", "pairs": len(diff),
                          "mean": m, "ci_low": m - h, "ci_high": m + h})
contrasts = pd.DataFrame(contrasts)
print()
print(contrasts.round(2).to_string(index=False))
contrasts.round(4).to_csv(src.with_name("table6_contrasts.csv"), index=False)
