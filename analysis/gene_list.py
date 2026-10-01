# -*- coding: utf-8 -*-
"""The 54 genes that meet all three developmental criteria, and the six above PGR.

The manuscript states that 54 of 18,794 genes rise at least five fold, go
undetected in half or more of the first band donors and increase without
reversal, and that PGR leads them on age association. It never lists them.
This script rebuilds that screen from the developmental atlas and writes the
list, so a reader can see which genes they are and what they look like.

The screen is the one `verify_all.py` implements and is reproduced step for
step: donors at or beyond the retention cut with at least ten endothelial
cells, counts per 10,000 per donor, genes detected in at least 20% of donors,
log transform, z score within contributing study, Spearman with donor age,
band means for the fold change and the zero fraction of the first band.

Production gate: PGR must return its four published values, association +0.680,
rank 7, fold 16.97 and 88% of first band donors without detection. If any of
them misses, the screen has not been reproduced and nothing is written.

Run: python3 gene_list.py --h5ad F --out gene_list.tsv
"""
import argparse, os, sys
import numpy as np
import h5py
from scipy import stats

SEED = 2026
AGE_MIN = 0.75
MIN_CELL = 10
DETECT = 0.20
BANDS = [(0.75, 2), (2, 6), (6, 12), (12, 20), (20, 99)]
FOLD_MIN, ZERO_MIN = 5.0, 50.0
GATE = dict(rho=0.680, rank=7, fold=16.97, zero=88)

ap = argparse.ArgumentParser()
ap.add_argument("--h5ad", required=True)
ap.add_argument("--out", required=True)
A = ap.parse_args()

def col(obs, name):
    g = obs[name]
    if isinstance(g, h5py.Group) and "categories" in g:
        c = np.array([x.decode() if isinstance(x, bytes) else x for x in g["categories"][:]], dtype=object)
        return c[g["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in g[:]], dtype=object)

f = h5py.File(A.h5ad, "r")
obs = f["obs"]
for c in ("cellclass", "final_annotation", "donor_id", "age_years_postgestation", "study"):
    assert c in obs, f"obs column absent: {c}"
cellclass = col(obs, "cellclass"); final_ann = col(obs, "final_annotation")
donor = col(obs, "donor_id"); study = col(obs, "study")
age = obs["age_years_postgestation"][:].astype(float)
sym = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var"]["gene_symbol"][:]], dtype=object)

X = f["X"]
enc = X.attrs.get("encoding-type", b""); enc = enc.decode() if isinstance(enc, bytes) else enc
assert enc == "csr_matrix", f"unexpected encoding {enc!r}"
indptr = X["indptr"][:]; data, indices = X["data"], X["indices"]
NG = len(sym)

keep = (cellclass == "Vascular") & (final_ann == "Endo") & (age > AGE_MIN)
sel = np.where(keep)[0]
print(f"endothelial nuclei above the cut: {len(sel)}", flush=True)

# donor x gene raw counts, one pass
dsel = donor[sel]
udon = sorted(set(dsel.tolist()))
di = {d: i for i, d in enumerate(udon)}
Praw = np.zeros((len(udon), NG), dtype=np.float64)
ncell = np.zeros(len(udon), dtype=int)
for j, r in enumerate(sel):
    i = di[dsel[j]]
    ncell[i] += 1
    a, b = indptr[r], indptr[r + 1]
    if b <= a: continue
    np.add.at(Praw[i], indices[a:b], data[a:b].astype(np.float64))
print(f"donors before the cell threshold: {len(udon)}", flush=True)

dage = np.array([age[sel[list(dsel).index(d)]] for d in udon]) if False else None
# donor age and study, taken from the first nucleus of that donor and asserted constant
dage = np.zeros(len(udon)); dstudy = np.empty(len(udon), dtype=object)
for d, i in di.items():
    m = dsel == d
    aa = np.unique(age[sel][m]); ss = np.unique(study[sel][m])
    assert len(aa) == 1 and len(ss) == 1, f"donor {d} carries {len(aa)} ages and {len(ss)} studies"
    dage[i] = aa[0]; dstudy[i] = ss[0]

k = ncell >= MIN_CELL
P = Praw[k]; ag = dage[k]; st = dstudy[k]
print(f"donors retained: {int(k.sum())}, nuclei {int(ncell[k].sum())}", flush=True)

CP = P / np.maximum(P.sum(1, keepdims=True), 1) * 1e4
det = (CP > 0).mean(0) >= DETECT
G = sym[det]; Xc = CP[:, det]
print(f"genes passing detection: {Xc.shape[1]}", flush=True)

Z = np.log1p(Xc)
for s in set(st):
    j = st == s
    Z[j] = (Z[j] - Z[j].mean(0)) / np.maximum(Z[j].std(0), 1e-9)
R = np.nan_to_num(np.array([stats.spearmanr(ag, Z[:, i])[0] for i in range(Z.shape[1])]))
rank = (-R).argsort().argsort() + 1
Mb = np.array([Xc[(ag >= lo) & (ag < hi)].mean(0) for lo, hi in BANDS])
fold = (Mb[-1] + 1e-4) / (Mb[0] + 1e-4)
mono = np.all(np.diff(Mb, axis=0) >= -1e-9, axis=0)
zero = (Xc[ag < 2] == 0).mean(0) * 100

pos = {}
for i, g in enumerate(G): pos.setdefault(g, i)
assert "PGR" in pos, "PGR did not pass detection"
i = pos["PGR"]
got = dict(rho=round(float(R[i]), 3), rank=int(rank[i]), fold=round(float(fold[i]), 2),
           zero=int(round(zero[i])))
print("production gate:", got, "expected", GATE, flush=True)
bad = [k2 for k2 in GATE if got[k2] != GATE[k2]]
if bad:
    print(f"GATE FAILED on {bad}: the screen is not the published one, nothing written")
    sys.exit(2)

passing = np.where((fold >= FOLD_MIN) & (zero >= ZERO_MIN) & mono)[0]
order = passing[np.argsort(-R[passing])]
print(f"genes passing all three criteria: {len(passing)}", flush=True)

above = np.where(rank < rank[i])[0]
above = above[np.argsort(rank[above])]

with open(A.out, "w", encoding="utf-8") as fh:
    fh.write("block\tgene\trho\trank\tfold\tzero_pct_band1\tmonotonic\tmean_cp10k\tband_means\n")
    fh.write(f"meta\tn_donor\t{int(k.sum())}\t\t\t\t\t\t\n")
    fh.write(f"meta\tn_gene_tested\t{Xc.shape[1]}\t\t\t\t\t\t\n")
    fh.write(f"meta\tn_passing\t{len(passing)}\t\t\t\t\t\t\n")
    for cls, idx in (("passing_all_three", order), ("above_PGR_on_association", above)):
        for j2 in idx:
            fh.write("\t".join([cls, str(G[j2]), f"{R[j2]:+.3f}", str(int(rank[j2])),
                                f"{fold[j2]:.2f}", f"{zero[j2]:.0f}", str(bool(mono[j2])),
                                f"{Xc[:, j2].mean():.4f}",
                                ";".join(f"{v:.4f}" for v in Mb[:, j2])]) + "\n")
print(f"written: {A.out}")
