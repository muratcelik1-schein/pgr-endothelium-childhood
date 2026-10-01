# -*- coding: utf-8 -*-
"""Is the endothelial signal contamination from the neighbouring vascular cells?

The transcript is second highest in vascular leptomeningeal cells, and in one
cohort the margin over them is only 1.22 fold. So the reading that has to be
excluded is not ambient RNA from neurons, which the manuscript already tests,
but ambient RNA and doublets from the mural and leptomeningeal cells next door.

Four things are measured, all inside the manuscript's own endothelial series.

  1  DOUBLETS. The atlas carries a doublet call of its own, in obs. The fraction
     of PGR positive endothelial nuclei that are called doublets is compared
     with the fraction among negative ones, and the age association is
     recomputed with every called doublet removed.

  2  MURAL SIGNAL. Every endothelial nucleus is scored for mural and
     leptomeningeal identity from markers of those cells, and PGR positive
     nuclei are compared with negative ones. The comparison means nothing on
     its own, so the identical comparison is run for control genes matched to
     the target on the fraction of nuclei in which they are detected. The
     target is then placed in that distribution.

  3  A SCORE THAT MOVES WITH AGE. If the mural score itself rises with age, part
     of the rise could be contamination. The donor level mural score is
     correlated with age, and the age association of the target is recomputed
     with the mural score removed, exactly as the manuscript does for depth.

  4  THE COMPANION GENE. ZNF385B is reported alongside the target and is far
     more abundant in neurons than in endothelium, which makes it the better
     candidate for an ambient signature. It is carried through every step.

Analysis unit is the donor for every association. Column names are discovered
and asserted. Seed fixed at 2026. Reads only, writes one table.

Run: python3 ambient_doublet.py --h5ad F --shipped source_numbers_v2.tsv --out ambient_doublet.tsv
"""
import argparse, os, sys
import numpy as np
import h5py
from scipy import stats

SEED = 2026
rng = np.random.default_rng(SEED)
TARGET = "PGR"
COMPANION = "ZNF385B"
# the cells next door, from the manuscript's own Supplementary Table S2
MURAL = ["DCN", "COL1A2", "LUM", "PDGFRB", "RGS5", "KCNJ8", "ACTA2", "MYH11"]
# the compartment the manuscript's own ambient score uses
NEURAL = ["SNAP25", "SYT1", "RBFOX3", "PLP1", "MBP", "MOBP", "AQP4", "GFAP", "SLC1A2", "CSF1R"]
IDENTITY = ["PECAM1", "CLDN5", "FLT1"]
AGE_MIN = 0.75
MIN_CELL = 10
N_MATCHED = 200          # control genes matched on detection fraction
MATCH_TOL = 0.25         # relative tolerance on the detection fraction

ap = argparse.ArgumentParser()
ap.add_argument("--h5ad", required=True)
ap.add_argument("--shipped", default=None, help="source_numbers_v2.tsv, for the reproduction gate")
ap.add_argument("--out", required=True)
A = ap.parse_args()

rows = []
def add(b, k, v, n=""):
    rows.append((b, k, v, n)); print(f"{b:20s} {k:36s} {v}   {n}", flush=True)
def flush():
    d = os.path.dirname(A.out)
    if d: os.makedirs(d, exist_ok=True)
    with open(A.out, "w") as fh:
        fh.write("block\tkey\tvalue\tnote\n")
        for r in rows: fh.write("\t".join(str(x) for x in r) + "\n")
def die(msg):
    add("VERDICT", "SCAN INVALID", msg); flush(); sys.exit(2)

f = h5py.File(A.h5ad, "r")
obs = f["obs"]

def col(name):
    g = obs[name]
    if isinstance(g, h5py.Group) and "categories" in g:
        c = np.array([x.decode() if isinstance(x, bytes) else x for x in g["categories"][:]], dtype=object)
        return c[g["codes"][:]]
    v = g[:]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in v], dtype=object)

NEED = ["cellclass", "final_annotation", "donor_id", "age_years_postgestation", "study"]
missing = [c for c in NEED if c not in obs]
assert not missing, f"obs columns absent: {missing}"
DBL = "scDblFinder.class"
has_dbl = DBL in obs
add("meta", "file", os.path.basename(A.h5ad))
add("meta", "doublet_column_present", has_dbl, DBL)

cellclass = col("cellclass"); final_ann = col("final_annotation")
donor = col("donor_id"); age = obs["age_years_postgestation"][:].astype(float)
dbl = col(DBL) if has_dbl else np.array(["unknown"] * len(age), dtype=object)
if has_dbl:
    add("meta", "doublet_levels", "; ".join(f"{s}={int((dbl==s).sum())}" for s in sorted(set(dbl))))

sym = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var"]["gene_symbol"][:]], dtype=object)
X = f["X"]
enc = X.attrs.get("encoding-type", b""); enc = enc.decode() if isinstance(enc, bytes) else enc
assert enc == "csr_matrix", f"unexpected matrix encoding {enc!r}"
indptr = X["indptr"][:]; data, indices = X["data"], X["indices"]

# ---------------------------------------------------------------- the series
keep = (cellclass == "Vascular") & (final_ann == "Endo") & (age > AGE_MIN)
sel = np.where(keep)[0]
add("series", "endothelial_nuclei_above_cut", len(sel))

# load the whole row for every selected nucleus, once
NG = len(sym)
gene_tot = np.zeros(NG)          # counts per gene, pooled
gene_det = np.zeros(NG)          # nuclei in which detected
tot = np.zeros(len(sel), dtype=np.float64)
mat = {}                          # gene index -> per nucleus counts, for the genes we need
want_names = [TARGET, COMPANION] + MURAL + NEURAL + IDENTITY
want_idx = {}
for g in dict.fromkeys(want_names):
    w = np.where(sym == g)[0]
    if len(w): want_idx[g] = int(w[0])
    else: add("gene_absent", g, "not in var")
assert TARGET in want_idx, "PGR absent from var"
for g in want_idx: mat[g] = np.zeros(len(sel))
back = {v: k for k, v in want_idx.items()}
want_set = set(want_idx.values())

pattern = [None] * len(sel)      # detected gene indices per nucleus, kept for the null
for j, r in enumerate(sel):
    a, b = indptr[r], indptr[r + 1]
    if b <= a:
        pattern[j] = np.empty(0, dtype=np.int32); continue
    dd = data[a:b].astype(np.float64); ii = indices[a:b]
    tot[j] = dd.sum()
    np.add.at(gene_tot, ii, dd)
    np.add.at(gene_det, ii, 1.0)
    pattern[j] = ii.astype(np.int32)
    for h_pos in range(len(ii)):
        h = int(ii[h_pos])
        if h in want_set: mat[back[h]][j] = dd[h_pos]

dn = donor[sel]; ag = age[sel]; db = dbl[sel]
ok_donor = np.array([np.sum(dn == d) >= MIN_CELL for d in dn])
sel2 = ok_donor
dn, ag, db, tot = dn[sel2], ag[sel2], db[sel2], tot[sel2]
pattern = [pattern[j] for j in range(len(sel2)) if sel2[j]]
for g in mat: mat[g] = mat[g][sel2]
n_nuc = len(dn); n_don = len(np.unique(dn))
add("series", "nuclei_after_donor_threshold", n_nuc)
add("series", "donors", n_don)

cp = lambda v: v / np.maximum(tot, 1) * 1e4
pos = mat[TARGET] > 0
add("series", "PGR_positive_nuclei", int(pos.sum()), f"{pos.mean()*100:.2f}% of nuclei")

def donor_frac(v_pos):
    xs, ys = [], []
    for d in np.unique(dn):
        m = dn == d
        xs.append(ag[m][0]); ys.append(float(v_pos[m].mean()))
    return np.array(xs), np.array(ys)

x0, y0 = donor_frac(pos)
r0 = stats.spearmanr(x0, y0)
add("gate", "donor_level_positive_fraction_vs_age", f"{r0[0]:+.3f}", f"p = {r0[1]:.3g}, n = {len(x0)}")
if not (n_don == 65 and n_nuc == 5388 and abs(r0[0] - 0.707) < 0.01):
    die(f"reproduction gate: donors {n_don} (65), nuclei {n_nuc} (5388), rho {r0[0]:+.3f} (+0.707)")
add("gate", "reproduction", "PASS", "65 donors, 5,388 nuclei, +0.707 as shipped")

# ---------------------------------------------------------------- 1 doublets
if has_dbl:
    is_d = np.array([s.lower().startswith("doublet") for s in db])
    add("doublet", "called_doublets_in_series", f"{int(is_d.sum())} of {n_nuc}", f"{is_d.mean()*100:.2f}%")
    fp, fn = is_d[pos].mean() * 100, is_d[~pos].mean() * 100
    tab = np.array([[int((is_d & pos).sum()), int((~is_d & pos).sum())],
                    [int((is_d & ~pos).sum()), int((~is_d & ~pos).sum())]])
    orr, pf = stats.fisher_exact(tab)
    add("doublet", "doublet_rate_positive_vs_negative", f"{fp:.2f}% against {fn:.2f}%",
        f"odds ratio {orr:.2f}, Fisher p = {pf:.3g}")
    m = ~is_d
    xs, ys = [], []
    for d in np.unique(dn[m]):
        mm = m & (dn == d)
        if mm.sum() < MIN_CELL: continue
        xs.append(ag[mm][0]); ys.append(float(pos[mm].mean()))
    rd = stats.spearmanr(xs, ys)
    add("doublet", "age_association_without_doublets", f"{rd[0]:+.3f}",
        f"p = {rd[1]:.3g}, n = {len(xs)} donors; against {r0[0]:+.3f} with them")

# ---------------------------------------------------------------- 2 mural score
def score(genes):
    have = [g for g in genes if g in mat]
    if not have: return None, []
    z = np.zeros(n_nuc)
    for g in have:
        v = np.log1p(cp(mat[g]))
        s = v.std()
        z += (v - v.mean()) / (s if s > 1e-9 else 1.0)
    return z / len(have), have

mural, mural_used = score(MURAL)
neural, neural_used = score(NEURAL)
ident, ident_used = score(IDENTITY)
add("mural", "markers_used", "; ".join(mural_used))
add("mural", "neural_markers_used", "; ".join(neural_used))

# positive control: does the score separate what it claims to?
vlmc = (cellclass == "Vascular") & (final_ann != "Endo") & (age > AGE_MIN)
vsel = np.where(vlmc)[0][:20000]
vt = np.zeros(len(vsel)); vz = {g: np.zeros(len(vsel)) for g in mural_used}
for j, r in enumerate(vsel):
    a, b = indptr[r], indptr[r + 1]
    if b <= a: continue
    dd = data[a:b].astype(np.float64); ii = indices[a:b]
    vt[j] = dd.sum()
    for h_pos in range(len(ii)):
        h = int(ii[h_pos])
        if h in want_set and back[h] in vz: vz[back[h]][j] = dd[h_pos]
vm = np.mean([np.log1p(vz[g] / np.maximum(vt, 1) * 1e4) for g in mural_used], axis=0)
em = np.mean([np.log1p(cp(mat[g])) for g in mural_used], axis=0)
u, pu = stats.mannwhitneyu(vm, em, alternative="greater")
add("mural", "positive_control_nonEndo_vs_Endo", f"{vm.mean():.4f} against {em.mean():.4f}",
    f"one sided p = {pu:.3g}, n = {len(vsel)} against {n_nuc}")
if not (vm.mean() > em.mean() and pu < 0.001):
    die("the mural score does not separate non endothelial vascular nuclei from endothelial ones")
add("mural", "positive_control", "PASS")

d_obs = float(mural[pos].mean() - mural[~pos].mean())
add("mural", "target_positive_minus_negative", f"{d_obs:+.4f}",
    "difference in mural score between PGR positive and PGR negative endothelial nuclei")

# matched control genes: same detection fraction across the same nuclei
det_frac_all = gene_det / max(len(sel), 1)
tgt_det = float(pos.mean())
cand = np.where((det_frac_all > tgt_det * (1 - MATCH_TOL)) &
                (det_frac_all < tgt_det * (1 + MATCH_TOL)))[0]
cand = np.array([c for c in cand if sym[c] not in (TARGET, COMPANION)])
add("matched", "candidate_genes_in_detection_window", len(cand),
    f"target detected in {tgt_det*100:.2f}% of nuclei, window +-{int(MATCH_TOL*100)}%")
if len(cand) < 30:
    die(f"only {len(cand)} matched control genes; the null cannot be built")
pick = rng.choice(cand, size=min(N_MATCHED, len(cand)), replace=False)

# membership is read from the stored sparse pattern, so the matrix is touched once
hit = {int(g): np.zeros(n_nuc, dtype=bool) for g in pick}
pickset = set(int(g) for g in pick)
for j in range(n_nuc):
    ii = pattern[j]
    if ii.size == 0: continue
    for h in ii:
        h = int(h)
        if h in pickset: hit[h][j] = True
null, null_genes = [], []
for gidx in pick:
    p2 = hit[int(gidx)]
    if p2.sum() < 20 or (~p2).sum() < 20: continue
    null.append(float(mural[p2].mean() - mural[~p2].mean()))
    null_genes.append(str(sym[int(gidx)]))
null = np.array(null)
pct = float((null <= d_obs).mean() * 100)
p_two = float(min(1.0, 2 * min((null >= d_obs).mean(), (null <= d_obs).mean())))
add("matched", "null_genes_used", len(null))
add("matched", "null_median_difference", f"{np.median(null):+.4f}",
    f"5th to 95th percentile {np.percentile(null,5):+.4f} to {np.percentile(null,95):+.4f}")
add("matched", "target_percentile_in_null", f"{pct:.1f}", f"two sided empirical p = {p_two:.4f}")

# the companion gene through the same machine
if COMPANION in mat:
    zp = mat[COMPANION] > 0
    dz = float(mural[zp].mean() - mural[~zp].mean())
    dn_neural = float(neural[zp].mean() - neural[~zp].mean())
    add("companion", "ZNF385B_mural_difference", f"{dz:+.4f}",
        f"percentile in the same null {float((null <= dz).mean()*100):.1f}")
    add("companion", "ZNF385B_neural_ambient_difference", f"{dn_neural:+.4f}")
add("target", "PGR_neural_ambient_difference",
    f"{float(neural[pos].mean() - neural[~pos].mean()):+.4f}",
    "the manuscript's own ambient compartment, for comparison")

# ---------------------------------------------------------------- 3 age
xs, ys_m, ys_n, ys_p = [], [], [], []
for d in np.unique(dn):
    m = dn == d
    xs.append(ag[m][0]); ys_m.append(float(mural[m].mean()))
    ys_n.append(float(neural[m].mean())); ys_p.append(float(pos[m].mean()))
xs = np.array(xs); ys_m = np.array(ys_m); ys_n = np.array(ys_n); ys_p = np.array(ys_p)
rm = stats.spearmanr(xs, ys_m); rn = stats.spearmanr(xs, ys_n)
add("age", "mural_score_vs_age", f"{rm[0]:+.3f}", f"p = {rm[1]:.3g}, n = {len(xs)} donors")
add("age", "neural_ambient_vs_age", f"{rn[0]:+.3f}", f"p = {rn[1]:.3g}")

def partial(y, x, c):
    R = np.vstack([stats.rankdata(v) for v in (y, x, c)])
    C = np.corrcoef(R)
    num = C[0, 1] - C[0, 2] * C[1, 2]
    den = np.sqrt((1 - C[0, 2] ** 2) * (1 - C[1, 2] ** 2))
    r = num / den if den > 1e-12 else np.nan
    n = len(y); dfree = n - 3
    t = r * np.sqrt(dfree / max(1 - r * r, 1e-12))
    return r, 2 * stats.t.sf(abs(t), dfree)

rp, pp = partial(ys_p, xs, ys_m)
add("age", "age_association_after_removing_mural_score", f"{rp:+.3f}",
    f"p = {pp:.3g}; against {r0[0]:+.3f} unadjusted")

# 4 drop the most mural nuclei and repeat
thr = np.percentile(mural, 90)
m = mural < thr
xs2, ys2 = [], []
for d in np.unique(dn[m]):
    mm = m & (dn == d)
    if mm.sum() < MIN_CELL: continue
    xs2.append(ag[mm][0]); ys2.append(float(pos[mm].mean()))
r2 = stats.spearmanr(xs2, ys2)
add("age", "age_association_without_top_decile_mural", f"{r2[0]:+.3f}",
    f"p = {r2[1]:.3g}, n = {len(xs2)} donors, {int(m.sum())} nuclei retained")

add("limits", "what_this_cannot_do",
    "A doublet call and a marker score bound contamination; they do not exclude it. "
    "Ambient RNA that is uniform across nuclei of one donor moves no within donor comparison.")
add("limits", "seed", SEED)
flush()
print(f"\nwritten: {A.out} | {len(rows)} rows")
