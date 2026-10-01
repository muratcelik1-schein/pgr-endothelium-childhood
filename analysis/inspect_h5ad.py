#!/usr/bin/env python3
"""Discovery pass. Reports what is actually in an .h5ad: obs columns, their value
sets, var index, X layout and encoding. Guesses nothing and writes nothing but a
report. Every later script asserts against what this finds.

Run: python3 inspect_h5ad.py --h5ad data/devbrain.h5ad --out out/inspect_devbrain.txt
"""
import argparse, os, sys
import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--h5ad", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-levels", type=int, default=60)
A = ap.parse_args()

buf = []
def say(*a):
    line = " ".join(str(x) for x in a)
    buf.append(line); print(line, flush=True)

f = h5py.File(A.h5ad, "r")
say(f"file  {A.h5ad}  {os.path.getsize(A.h5ad)/1e9:.2f} GB")
say(f"top level keys: {sorted(f.keys())}")
say(f"root attrs: {dict(f.attrs)}")

# ---------------------------------------------------------------- X
for key in ("X", "raw/X"):
    if key in f:
        g = f[key]
        if isinstance(g, h5py.Group):
            say(f"\n{key}: sparse group, keys {sorted(g.keys())}, attrs {dict(g.attrs)}")
            say(f"  data dtype {g['data'].dtype}, len {g['data'].shape[0]:,}")
            say(f"  indptr len {g['indptr'].shape[0]:,}")
            d = g["data"][:100000]
            say(f"  first 100k values: min {d.min():.4g} max {d.max():.4g} "
                f"integral {bool(np.all(d == np.round(d)))}")
        else:
            say(f"\n{key}: dense {g.shape} {g.dtype}")

# ---------------------------------------------------------------- var
def read_index(grp):
    enc = grp.attrs.get("_index", "_index")
    enc = enc.decode() if isinstance(enc, bytes) else enc
    return grp[enc]

for vk in ("var", "raw/var"):
    if vk in f:
        v = f[vk]
        idx = read_index(v)
        say(f"\n{vk}: {idx.shape[0]:,} rows, columns {sorted(k for k in v.keys())}")
        vals = idx[:5]
        say(f"  index sample: {[x.decode() if isinstance(x, bytes) else x for x in vals]}")
        for c in v.keys():
            if c.startswith("_"): continue
            node = v[c]
            if isinstance(node, h5py.Group) and "categories" in node:
                cats = node["categories"][:8]
                say(f"  col {c!r}: categorical, e.g. "
                    f"{[x.decode() if isinstance(x, bytes) else x for x in cats]}")
            elif isinstance(node, h5py.Dataset):
                s = node[:3]
                say(f"  col {c!r}: {node.dtype}, e.g. "
                    f"{[x.decode() if isinstance(x, bytes) else x for x in s]}")

# ---------------------------------------------------------------- obs
o = f["obs"]
idx = read_index(o)
n = idx.shape[0]
say(f"\nobs: {n:,} rows")
say(f"obs columns: {sorted(k for k in o.keys() if not k.startswith('_'))}")
for c in sorted(o.keys()):
    if c.startswith("_"): continue
    node = o[c]
    if isinstance(node, h5py.Group) and "categories" in node:
        cats = node["categories"][:]
        cats = [x.decode() if isinstance(x, bytes) else x for x in cats]
        codes = node["codes"][:]
        cnt = np.bincount(codes[codes >= 0], minlength=len(cats))
        say(f"\n  [{c}] categorical, {len(cats)} levels")
        order = np.argsort(-cnt)
        for i in order[:A.max_levels]:
            say(f"      {cnt[i]:>9,}  {cats[i]!r}")
        if len(cats) > A.max_levels: say(f"      ... {len(cats)-A.max_levels} more")
    elif isinstance(node, h5py.Dataset):
        d = node[:]
        if d.dtype.kind in "fiu":
            say(f"\n  [{c}] numeric {d.dtype}: min {np.nanmin(d):.4g} "
                f"max {np.nanmax(d):.4g} median {np.nanmedian(d):.4g} "
                f"unique {len(np.unique(d)):,}")
            if len(np.unique(d)) <= 40:
                say(f"      values: {sorted(np.unique(d).tolist())}")
        else:
            u = np.unique(d)
            say(f"\n  [{c}] {d.dtype}, {len(u):,} unique")
            if len(u) <= A.max_levels:
                say(f"      {[x.decode() if isinstance(x, bytes) else x for x in u[:A.max_levels]]}")

# ---------------------------------------------------------------- obsm / uns
for k in ("obsm", "uns", "layers"):
    if k in f:
        say(f"\n{k}: {sorted(f[k].keys())}")

os.makedirs(os.path.dirname(A.out), exist_ok=True)
open(A.out, "w").write("\n".join(buf) + "\n")
say(f"\nwritten: {A.out}")
