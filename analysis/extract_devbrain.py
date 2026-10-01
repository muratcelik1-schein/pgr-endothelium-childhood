# -*- coding: utf-8 -*-
"""PREPROCESSING 1/2. Builds the derived matrices from the single-nucleus file of the developmental atlas.
Input : devbrain.h5ad  (downloaded from the CELLxGENE collection, ~10 GB)
Output: endo_pseudobulk.npy, endo_meta.csv, gene_sym.npy,
        mu_M.npy, mu_meta.tsv, mu_sym.npy, nu_M.npy, nu_meta.tsv,
        donor_pgr_counts.csv
Run: python3 extract_devbrain.py --h5ad <path> --out <directory>
"""
import argparse, os, collections
import numpy as np, pandas as pd, h5py

ap = argparse.ArgumentParser()
ap.add_argument("--h5ad", required=True)
ap.add_argument("--out", default=".")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
f = h5py.File(A.h5ad, "r")

def cat(n):
    g = f["obs"][n]
    if isinstance(g, h5py.Group) and "codes" in g:
        c = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
        return np.array(c)[g["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in g[:]])

cc, fa, dn, stu = cat("cellclass"), cat("final_annotation"), cat("donor_id"), cat("study")
sex, reg = cat("sex"), cat("region")
age = f["obs"]["age_years_postgestation"][:]
sym = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["var"]["gene_symbol"][:]])
G = len(sym)
idx, data, ptr = f["X"]["indices"], f["X"]["data"], f["X"]["indptr"][:]
np.save(f"{A.out}/gene_sym.npy", sym); np.save(f"{A.out}/mu_sym.npy", sym)

GROUPS = {"Endo":  (cc == "Vascular") & (fa == "Endo"),
          "VLMC":  (cc == "Vascular") & (fa == "VLMC"),
          "Astro": (cc == "Astro"),
          "Glut":  (cc == "Glutamatergic"),
          "GABA":  (cc == "GABAergic")}
donors = sorted(set(dn)); di = {d: i for i, d in enumerate(donors)}

def pseudobulk(mask):
    S = np.zeros((len(donors), G), dtype=np.float32)
    rows = np.where(mask)[0]
    for r in rows:
        s, e = ptr[r], ptr[r + 1]
        if e > s: np.add.at(S[di[dn[r]]], idx[s:e], data[s:e])
    n = np.zeros(len(donors), dtype=int)
    for r in rows: n[di[dn[r]]] += 1
    return S, n

PB, N = {}, {}
for k, mask in GROUPS.items():
    PB[k], N[k] = pseudobulk(mask)
    print(f"{k}: {int(mask.sum())} nuclei, {int((N[k]>0).sum())} donors", flush=True)

meta = pd.DataFrame(dict(
    key=[f"{stu[np.where(dn==d)[0][0]]}|{d}" for d in donors], donor=donors,
    study=[stu[np.where(dn == d)[0][0]] for d in donors],
    sex=[sex[np.where(dn == d)[0][0]] for d in donors],
    age=[float(age[np.where(dn == d)[0][0]]) for d in donors]))
meta["age_y"] = meta.age
for k in GROUPS: meta[f"n_{k}"] = N[k]
meta["n"] = N["Endo"]

np.save(f"{A.out}/endo_pseudobulk.npy", PB["Endo"])
meta[["key", "study", "donor", "age_y", "n"]].to_csv(f"{A.out}/endo_meta.csv", index=False)
# vascular panel: Endo, VLMC, VascOther (empty), Astro  -> 4 rows per donor
VO = np.zeros_like(PB["Endo"])
mu = np.empty((len(donors) * 4, G), dtype=np.float32)
for i in range(len(donors)):
    mu[i*4+0], mu[i*4+1], mu[i*4+2], mu[i*4+3] = PB["Endo"][i], PB["VLMC"][i], VO[i], PB["Astro"][i]
np.save(f"{A.out}/mu_M.npy", mu)
meta.assign(n_VascOther=0)[["donor", "study", "sex", "age", "n_Endo", "n_VLMC", "n_VascOther", "n_Astro"]] \
    .to_csv(f"{A.out}/mu_meta.tsv", sep="\t", index=False)
nu = np.empty((len(donors) * 4, G), dtype=np.float32)
Z = np.zeros_like(PB["Glut"])
for i in range(len(donors)):
    nu[i*4+0], nu[i*4+1], nu[i*4+2], nu[i*4+3] = PB["Glut"][i], PB["GABA"][i], Z[i], Z[i]
np.save(f"{A.out}/nu_M.npy", nu)
meta.assign(n_Oligo=0, n_OPC=0)[["donor", "study", "age", "n_Glut", "n_GABA", "n_Oligo", "n_OPC"]] \
    .to_csv(f"{A.out}/nu_meta.tsv", sep="\t", index=False)

# PGR+ endothelial nuclei per donor
col = int(np.where(sym == "PGR")[0][0])
sel = np.where((cc == "Vascular") & (fa == "Endo") & (age > 0.75))[0]
rec = collections.defaultdict(lambda: [0, 0])
for r in sel:
    s, e = ptr[r], ptr[r + 1]
    ii = idx[s:e]
    hit = bool(np.any(ii == col)) and data[s:e][ii == col][0] > 0
    rec[dn[r]][0] += int(hit); rec[dn[r]][1] += 1
dc = pd.DataFrame([dict(donor=d, study=stu[np.where(dn == d)[0][0]],
                        age=float(age[np.where(dn == d)[0][0]]), k=v[0], n=v[1])
                   for d, v in rec.items()])
dc[dc.n >= 10].to_csv(f"{A.out}/donor_pgr_counts.csv", index=False)
print("done. output directory:", A.out)
