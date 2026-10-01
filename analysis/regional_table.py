# -*- coding: utf-8 -*-
"""PGR by anatomical division and cell class in the whole brain atlas.
Reads allen_fig.csv only, writes one table. Never writes to its input.
Run: python3 regional_table.py [--out .]"""
import os, argparse
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=HERE)
A = ap.parse_args()
SEED = 2026  # for parity; this script draws nothing

a = pd.read_csv(os.path.join(HERE, "allen_fig.csv"))
rows = []
def add(block, key, value, note=""):
    rows.append(dict(block=block, key=key, value=value, note=note))

add("meta", "n_cluster", len(a), "clusters with at least 50 nuclei")
add("meta", "n_division", a.division.nunique())
for cls in ("vascular", "neuronal"):
    s = a[a.cls == cls]
    g = s.groupby("division").PGR.agg(["count", "median", "max"]).sort_values("median", ascending=False)
    for div, r in g.iterrows():
        add(f"{cls}_by_division", div,
            f"clusters={int(r['count'])}; median={r['median']:.4f}; max={r['max']:.4f}")
    add(f"{cls}_by_division", "SUMMARY",
        f"top={g.index[0]} (median {g['median'].iloc[0]:.4f}, {int(g['count'].iloc[0])} clusters); "
        f"cortex median={g.loc['Cerebral cortex','median']:.4f} ({int(g.loc['Cerebral cortex','count'])} clusters)",
        "divisions with few clusters carry no weight; the count column is the caveat")
cb = a[a.division == "Cerebellum"]
add("cerebellum", "clusters", f"total={len(cb)}; " + "; ".join(f"{k}={v}" for k, v in cb.cls.value_counts().items()))
add("cerebellum", "PGR", f"median={cb.PGR.median():.4f}; max={cb.PGR.max():.4f}; above_0.5={int((cb.PGR>0.5).sum())}")
add("cerebellum", "vascular_clusters", len(cb[cb.cls == "vascular"]),
    "a single vascular cluster; no regional comparison follows from it")
out = os.path.join(A.out, "regional_numbers.tsv")
pd.DataFrame(rows)[["block", "key", "value", "note"]].to_csv(out, sep="\t", index=False)
print(f"written: {os.path.basename(out)} | {len(rows)} rows")
