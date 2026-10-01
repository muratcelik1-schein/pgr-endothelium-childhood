#!/usr/bin/env python3
"""Is the rise in the fraction of PGR positive nuclei an artefact of sequencing depth?

A nucleus sequenced deeper is more likely to show any transcript. If depth rises
with age, the positive fraction rises with it. This script removes that by
computing, for every nucleus, the probability that the gene would still be
detected if the nucleus had been sequenced to a fixed depth D.

For a nucleus with N total counts of which k belong to the gene, sampling D
counts without replacement gives

    P(detected at depth D) = 1 - prod_{i=0}^{D-1} (N-k-i)/(N-i)

which is exact and deterministic. No simulation, so nothing depends on a seed,
and the answer is identical on every run. Nuclei with N < D are dropped and the
number dropped is reported.

The donor level expected positive fraction at depth D is the mean of that
probability over the donor's retained nuclei. It is then correlated with age,
exactly as the raw fraction is, so the two are directly comparable.

Controls travel with the target: housekeeping genes, the related steroid
receptors and endothelial identity genes are all carried through the same
computation, and a gene whose raw rise is purely a depth effect will lose it.

Run: python3 depth_controlled_detection.py --h5ad F --celltype-col cell_type \
        --celltype "endothelial cell" --donor-col donor_id --age-col development_stage \
        --out out/depth_controlled.tsv
"""
import argparse, os, re, json, sys
import numpy as np
import h5py
from scipy import stats

TARGET = "PGR"
CONTROLS = {
    "other_receptor": ["NR3C1", "NR3C2", "ESR1", "AR", "GPER1"],
    "housekeeping": ["ACTB", "GAPDH", "RPL13A", "RPLP0", "TBP", "PGK1"],
    "identity": ["PECAM1", "CLDN5", "FLT1", "VWF"],
    "extra": ["PGR-AS1", "ZNF385B", "PAQR5", "ABCB1"],
}
DEPTHS = [1000, 1500, 2000, 3000]

ap = argparse.ArgumentParser()
ap.add_argument("--h5ad", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--celltype-col", required=True)
ap.add_argument("--celltype", required=True)
ap.add_argument("--donor-col", required=True)
ap.add_argument("--age-col", required=True)
ap.add_argument("--tissue-col", default=None)
ap.add_argument("--tissue-keep", default=None, help="comma separated labels to keep")
ap.add_argument("--min-nuclei", type=int, default=10)
ap.add_argument("--x-key", default="raw/X")
A = ap.parse_args()

rows = []
def add(block, key, value, note=""):
    rows.append((block, key, value, note)); print(f"{block:24s} {key:34s} {value}   {note}", flush=True)

f = h5py.File(A.h5ad, "r")

def cat(name):
    g = f["obs"][name]
    assert isinstance(g, h5py.Group) and "categories" in g, f"{name} is not categorical"
    c = [x.decode() if isinstance(x, bytes) else x for x in g["categories"][:]]
    return np.array(c, dtype=object), g["codes"][:]

ct_cats, ct_codes = cat(A.celltype_col)
dn_cats, dn_codes = cat(A.donor_col)
assert A.celltype in set(ct_cats), f"{A.celltype!r} not found. Available: {sorted(ct_cats)}"
is_ct = ct_codes == int(np.where(ct_cats == A.celltype)[0][0])

# --- age, either categorical stage labels or a numeric column
ao = f["obs"][A.age_col]
if isinstance(ao, h5py.Group) and "categories" in ao:
    ag_cats, ag_codes = cat(A.age_col)
    def parse(lbl):
        m = re.match(r"^(\d+(?:\.\d+)?)[- ]year", str(lbl))
        if m: return float(m.group(1))
        if "infant" in str(lbl).lower(): return 1.0
        if "month" in str(lbl).lower():
            m2 = re.match(r"^(\d+(?:\.\d+)?)[- ]month", str(lbl)); return float(m2.group(1)) / 12 if m2 else None
        return None
    amap = {s: parse(s) for s in ag_cats}
    bad = [k for k, v in amap.items() if v is None]
    assert not bad, f"unparsed age label: {bad}"
    age_per_cell = np.array([amap[ag_cats[c]] for c in ag_codes])
    add("meta", "age_labels", json.dumps({k: v for k, v in sorted(amap.items(), key=lambda kv: kv[1])}),
        "infant ordered at 1.0 where present; rank only")
else:
    age_per_cell = ao[:].astype(float)
    add("meta", "age_column", A.age_col, "numeric")

keep = is_ct.copy()
if A.tissue_col and A.tissue_keep:
    ti_cats, ti_codes = cat(A.tissue_col)
    want = [t.strip() for t in A.tissue_keep.split(",")]
    miss = [t for t in want if t not in set(ti_cats)]
    assert not miss, f"tissue label missing: {miss}. Available: {sorted(ti_cats)}"
    keep &= np.isin(ti_codes, [int(np.where(ti_cats == t)[0][0]) for t in want])
    add("meta", "tissue_filter", "; ".join(want))
add("meta", "cell_type", A.celltype)
add("meta", "n_nuclei_selected", int(keep.sum()))

# --- genes
vn = f["var"]["feature_name"]
vc = np.array([x.decode() if isinstance(x, bytes) else x for x in vn["categories"][:]], dtype=object)
sym = vc[vn["codes"][:]]
GENES = [TARGET] + [g for v in CONTROLS.values() for g in v]
gi = {}
for g in GENES:
    w = np.where(sym == g)[0]
    if len(w): gi[g] = int(w[0])
    else: add("gene_missing", g, "absent from var")
assert TARGET in gi

# --- per nucleus totals and gene counts, only for selected nuclei
X = f[A.x_key]
d0 = X["data"][:200000]
assert np.all(d0 == np.round(d0)), f"{A.x_key} is not integer; raw counts are required"
indptr = X["indptr"][:]; data, indices = X["data"], X["indices"]
sel = np.where(keep)[0]
tot = np.zeros(len(sel)); gc = {g: np.zeros(len(sel)) for g in gi}
want = np.array(sorted(gi.values())); back = {v: k for k, v in gi.items()}
for j, r in enumerate(sel):
    a, b = indptr[r], indptr[r + 1]
    if b <= a: continue
    dd = data[a:b]; ii = indices[a:b]
    tot[j] = dd.sum()
    hit = np.intersect1d(ii, want)
    for h in hit: gc[back[int(h)]][j] = dd[ii == h].sum()
donor = dn_codes[sel]; age = age_per_cell[sel]
add("meta", "median_depth", float(np.median(tot)))
add("meta", "depth_quartiles", f"{np.percentile(tot,25):.0f} / {np.percentile(tot,50):.0f} / {np.percentile(tot,75):.0f}")

# --- the confound, measured
pos = gc[TARGET] > 0
add("confound", "median_depth_target_pos_vs_neg",
    f"{np.median(tot[pos]) if pos.any() else float('nan'):.0f} vs {np.median(tot[~pos]):.0f}",
    "the reason this script exists")
dr, dp = stats.spearmanr(age, tot)
add("confound", "depth_vs_age_per_nucleus", f"rho={dr:+.3f}; p={dp:.3g}",
    "if positive, deeper nuclei in older donors would inflate the raw fraction")

# --- exact detection probability at fixed depth
def p_detect(N, k, D):
    """1 - prod_{i<D} (N-k-i)/(N-i), computed in logs, vectorised over nuclei."""
    N = np.asarray(N, dtype=np.float64); k = np.asarray(k, dtype=np.float64)
    ok = (N >= D) & (k > 0)
    out = np.zeros_like(N)
    if not ok.any(): return out, ok
    # log prod (N-k-i)/(N-i) = [lgamma(N-k+1)-lgamma(N-k-D+1)] - [lgamma(N+1)-lgamma(N-D+1)]
    from scipy.special import gammaln
    Nn, kk = N[ok], k[ok]
    logq = (gammaln(Nn - kk + 1) - gammaln(Nn - kk - D + 1)) - (gammaln(Nn + 1) - gammaln(Nn - D + 1))
    out[ok] = 1.0 - np.exp(np.clip(logq, -700, 0))
    out[(N >= D) & (k == 0)] = 0.0
    return out, (N >= D)

def donor_corr(vals, mask):
    """donor level mean of vals, then Spearman with age. Unit of analysis is the donor."""
    xs, ys, ns = [], [], []
    for d in np.unique(donor[mask]):
        m = mask & (donor == d)
        if m.sum() < A.min_nuclei: continue
        xs.append(age[m][0]); ys.append(vals[m].mean()); ns.append(int(m.sum()))
    if len(xs) < 4: return None
    rho, p = stats.spearmanr(xs, ys)
    return rho, p, len(xs), sum(ns), np.array(ys)

# raw, uncontrolled
allm = np.ones(len(sel), dtype=bool)
r = donor_corr(pos.astype(float), allm)
add("raw", TARGET, f"rho={r[0]:+.3f}; p={r[1]:.4f}; donors={r[2]}; nuclei={r[3]}",
    "uncontrolled positive fraction, for comparison")

for D in DEPTHS:
    pd_, retained = p_detect(tot, gc[TARGET], D)
    if retained.sum() < 50:
        add(f"depth{D}", "skipped", f"only {int(retained.sum())} nuclei reach depth {D}"); continue
    rr = donor_corr(pd_, retained)
    if rr is None:
        add(f"depth{D}", "skipped", "fewer than four donors retained"); continue
    add(f"depth{D}", "nuclei_retained", f"{int(retained.sum())} of {len(sel)} "
        f"({100*retained.mean():.0f}%)", f"nuclei with fewer than {D} counts dropped")
    add(f"depth{D}", TARGET, f"rho={rr[0]:+.3f}; p={rr[1]:.4f}; donors={rr[2]}; nuclei={rr[3]}",
        "expected positive fraction at fixed depth, target")
    for kind, gs in CONTROLS.items():
        for g in gs:
            if g not in gi: continue
            pg, ret_g = p_detect(tot, gc[g], D)
            rg = donor_corr(pg, ret_g)
            if rg is None: continue
            add(f"depth{D}", g, f"rho={rg[0]:+.3f}; p={rg[1]:.4f}", kind)

# --- how many controls move as much as the target, at the middle depth
Dm = DEPTHS[len(DEPTHS) // 2]
pd_, ret = p_detect(tot, gc[TARGET], Dm)
rt = donor_corr(pd_, ret)
if rt:
    beat = []
    for kind, gs in CONTROLS.items():
        for g in gs:
            if g not in gi: continue
            pg, rg_ = p_detect(tot, gc[g], Dm)
            rr = donor_corr(pg, rg_)
            if rr and rr[0] >= rt[0]: beat.append(f"{g}({kind})")
    add("summary", f"controls_matching_target_at_depth{Dm}", f"{len(beat)} of "
        f"{sum(len(v) for v in CONTROLS.values())}", "; ".join(beat) if beat else "none")
    add("summary", "verdict",
        "depth controlled rise holds" if rt[1] < 0.05 and rt[0] > 0 else
        "depth controlled rise does not reach significance",
        f"rho={rt[0]:+.3f}, p={rt[1]:.4f} at depth {Dm}")

add("limits", "method", "exact hypergeometric detection probability", "no simulation, no seed dependence")
add("limits", "dropped_nuclei", "nuclei below the fixed depth are dropped",
    "this removes the shallowest nuclei, which are also the least informative")

os.makedirs(os.path.dirname(A.out), exist_ok=True)
with open(A.out, "w") as fh:
    fh.write("block\tkey\tvalue\tnote\n")
    for r_ in rows: fh.write("\t".join(str(x) for x in r_) + "\n")
print(f"\nwritten: {A.out} ({len(rows)} rows)")
