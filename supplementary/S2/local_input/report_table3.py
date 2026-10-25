"""Reproduce manuscript Table 3 from local_input/results.csv.

Table 3 reports, for the zero-temperature graphs only, the mean across networks of the
per-network ratio 100 * MC(rule node) / MC(exhaustive best single node).

Usage: python report_table3.py [path/to/results.csv]   (default: results.csv next to this script)
Writes table3.csv next to the input file.
"""
import sys
from pathlib import Path
import pandas as pd

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("results.csv")
d = pd.read_csv(src)
v = d[(d["status"] == "ok") & (d["tau"] == 0)].copy()
v["ratio_pct"] = 100 * v["rule"] / v["oracle"]

table = v.pivot_table(index=["n", "s0"], columns="eps", values="ratio_pct", aggfunc="mean")
graphs = v.groupby(["n", "s0"])["seed"].nunique().rename("graphs")
table = pd.concat([graphs, table], axis=1)
table.columns = ["graphs"] + [f"eps={e:g}" for e in table.columns[1:]]

print(table.round(1).to_string())
table.round(4).to_csv(src.with_name("table3.csv"))
