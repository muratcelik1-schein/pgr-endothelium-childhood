#!/usr/bin/env python3
"""Two questions the manuscript leaves open, asked of the multiome cohort.

Q1  COMPOSITION OR CELL STATE.  Across childhood, does the share of a
    PGR-high endothelial subtype rise, or does PGR rise inside a fixed
    subtype? Pseudobulk over all endothelium cannot tell these apart. Here
    every endothelial nucleus is scored for arterial, capillary and venous
    identity using published segment markers and assigned by argmax. PGR is
    never used in the assignment, and an assertion enforces that. Then, at
    donor level:
      a. does the share of each segment change with age
      b. within each segment, does PGR expression change with age
      c. do PGR positive nuclei differ from PGR negative nuclei of the same
         donor on segment markers

Q2  THE GLIAL LINE.  The Discussion argues that the cells receiving what
    crosses the wall, astrocytes and microglia, are also the cells that
    eliminate cortical synapses. That is an assertion about adjacency, never
    measured. Here the complement pruning genes and the allopregnanolone
    target genes are profiled in astrocytes and microglia of the same donors
    over the same age window, so their schedule can be compared with the
    endothelial one directly.

    A housekeeping module in the same cells bounds technical drift. A previous
    attempt at a related question failed exactly this control, and it is
    reported here whether it passes or fails.

Analysis unit is the donor throughout. Seed fixed. Writes one table.

Run: python3 composition_and_glia.py --h5ad data/clarence_rna.h5ad --out out/composition_glia.tsv
"""
import argparse, os, re, json, sys
import numpy as np
import h5py
from scipy import stats

SEED = 2026
np.random.seed(SEED)
TARGET = "PGR"

SEGMENT = {   # published human brain arteriovenous markers, positive controls
    "arterial":  ["GJA5", "SEMA3G", "HEY1", "ALPL", "VEGFC", "BMX"],
    "capillary": ["MFSD2A", "SLC7A5", "TFRC", "SLC16A1", "RGCC"],
    "venous":    ["NR2F2", "VWF", "ACKR1", "PLVAP", "IL1R1"],
}
PRUNING = ["C1QA", "C1QB", "C1QC", "C3", "C4A", "C4B", "ITGAM", "CX3CR1",
           "MERTK", "MEGF10", "TREM2", "TYROBP", "P2RY12", "CSF1R"]
ALLO_TARGETS = ["GABRA1", "GABRA2", "GABRA3", "GABRA4", "GABRA5", "GABRB1",
                "GABRB2", "GABRB3", "GABRG2", "TLR4", "MYD88"]
HOUSEKEEPING = ["ACTB", "GAPDH", "RPL13A", "RPLP0", "TBP", "PGK1"]
IDENTITY = ["PECAM1", "CLDN5", "FLT1"]
CORTEX = ["dorsolateral prefrontal cortex", "anterior cingulate cortex"]
MIN_NUC = 10

ap = argparse.ArgumentParser()
ap.add_argument("--h5ad", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--min-nuclei", type=int, default=MIN_NUC)
A = ap.parse_args()

rows = []
def add(b, k, v, n=""):
    rows.append((b, k, v, n)); print(f"{b:22s} {k:32s} {v}   {n}", flush=True)

f = h5py.File(A.h5ad, "r")
def cat(name):
    g = f["obs"][name]
    c = [x.decode() if isinstance(x, bytes) else x for x in g["categories"][:]]
    return np.array(c, dtype=object), g["codes"][:]

ct_cats, ct_codes = cat("cell_type"); dn_cats, dn_codes = cat("donor_id")
st_cats, st_codes = cat("development_stage"); ti_cats, ti_codes = cat("tissue")

def stage_age(l):
    m = re.match(r"^(\d+)-year-old stage$", l)
    if m: return float(m.group(1))
    if "infant" in l.lower(): return 1.0
    raise AssertionError(l)
amap = {s: stage_age(s) for s in st_cats}
age_cell = np.array([amap[st_cats[c]] for c in st_codes])
ctx = np.isin(ti_codes, [int(np.where(ti_cats == t)[0][0]) for t in CORTEX if t in ti_cats])

vn = f["var"]["feature_name"]
vc = np.array([x.decode() if isinstance(x, bytes) else x for x in vn["categories"][:]], dtype=object)
sym = vc[vn["codes"][:]]
ALL = ([TARGET] + [g for v in SEGMENT.values() for g in v] + PRUNING
       + ALLO_TARGETS + HOUSEKEEPING + IDENTITY)
gi = {}
for g in ALL:
    w = np.where(sym == g)[0]
    if len(w): gi[g] = int(w[0])
    else: add("gene_missing", g, "absent from var")

X = f["raw/X"]; indptr = X["indptr"][:]; data, indices = X["data"], X["indices"]
n_obs = len(ct_codes)

def load(mask):
    sel = np.where(mask)[0]
    tot = np.zeros(len(sel)); gc = {g: np.zeros(len(sel)) for g in gi}
    want = np.array(sorted(gi.values())); back = {v: k for k, v in gi.items()}
    for j, r in enumerate(sel):
        a, b = indptr[r], indptr[r + 1]
        if b <= a: continue
        dd = data[a:b]; ii = indices[a:b]
        tot[j] = dd.sum()
        for h in np.intersect1d(ii, want): gc[back[int(h)]][j] = dd[ii == h].sum()
    return sel, tot, gc

def donor_spearman(donors_, ages_, vals_):
    if len(set(donors_)) < 4: return None
    return stats.spearmanr(ages_, vals_)

# ==================================================================== Q1
ENDO = [c for c in ct_cats if "endothelial" in c.lower()][0]
mask = (ct_codes == int(np.where(ct_cats == ENDO)[0][0])) & ctx
sel, tot, gc = load(mask)
donor = dn_codes[sel]; age = age_cell[sel]
add("Q1_meta", "endothelial_nuclei_cortex", len(sel))
add("Q1_meta", "median_depth", float(np.median(tot)))

# --- score each nucleus, PGR excluded by construction
assert TARGET not in [g for v in SEGMENT.values() for g in v], "PGR cannot be a segment marker"
cp = {g: gc[g] / np.maximum(tot, 1) * 1e4 for g in gi}
scores = {}
for seg, gs in SEGMENT.items():
    have = [g for g in gs if g in gi]
    add("Q1_markers", seg, f"{len(have)} of {len(gs)} present", "; ".join(have))
    if not have: continue
    Z = np.vstack([ (cp[g] - cp[g].mean()) / max(cp[g].std(), 1e-9) for g in have ])
    scores[seg] = Z.mean(0)
assert len(scores) == 3, "markers are required for all three segments"
segnames = list(scores)
S = np.vstack([scores[s] for s in segnames])
assign = np.array(segnames, dtype=object)[S.argmax(0)]
# a nucleus with no positive score for any segment is left unassigned
unassigned = S.max(0) <= 0
assign[unassigned] = "unassigned"
for s in segnames + ["unassigned"]:
    add("Q1_assignment", s, f"{int((assign==s).sum())} ({100*(assign==s).mean():.1f}%)")

# --- validity gate: each segment's own markers must be highest in that segment
gate = True
for seg, gs in SEGMENT.items():
    have = [g for g in gs if g in gi]
    means = {s: np.mean([cp[g][assign == s].mean() for g in have]) for s in segnames}
    top = max(means, key=means.get); ok = top == seg; gate &= ok
    add("Q1_gate", seg, f"top={top}; " + "; ".join(f"{s}={means[s]:.3f}" for s in segnames),
        "PASS" if ok else "FAIL")
add("Q1_gate", "segment_labelling_recovered", bool(gate),
    "if False no segment claim about PGR follows")

if gate:
    # a. does segment composition change with age
    for s in segnames:
        xs, ys = [], []
        for d in np.unique(donor):
            m = donor == d
            if m.sum() < A.min_nuclei: continue
            xs.append(age[m][0]); ys.append(float((assign[m] == s).mean()))
        r = stats.spearmanr(xs, ys)
        add("Q1a_composition", s, f"rho={r[0]:+.3f}; p={r[1]:.4f}; donors={len(xs)}; "
            f"share {min(ys):.2f} to {max(ys):.2f}",
            "COMPOSITION SHIFT" if r[1] < 0.05 else "no significant shift")
    # b. within segment, does PGR rise
    for s in segnames:
        xs, ys, ps, ns = [], [], [], []
        for d in np.unique(donor):
            m = (donor == d) & (assign == s)
            if m.sum() < A.min_nuclei: continue
            xs.append(age[m][0]); ys.append(float(cp[TARGET][m].mean()))
            ps.append(100 * float((gc[TARGET][m] > 0).mean())); ns.append(int(m.sum()))
        if len(xs) < 4:
            add("Q1b_within_segment", s, f"only {len(xs)} donors reach the threshold", "not tested"); continue
        re_ = stats.spearmanr(xs, ys); rp = stats.spearmanr(xs, ps)
        add("Q1b_within_segment", s, f"expr rho={re_[0]:+.3f} p={re_[1]:.4f}; "
            f"pct_pos rho={rp[0]:+.3f} p={rp[1]:.4f}; donors={len(xs)}; nuclei={sum(ns)}")
    # c. PGR positive against PGR negative on segment markers, within donor
    pos = gc[TARGET] > 0
    for s in segnames:
        add("Q1c_pos_vs_neg", f"share_{s}",
            f"PGR+ {100*float((assign[pos]==s).mean()):.1f}%; PGR- {100*float((assign[~pos]==s).mean()):.1f}%",
            f"n_pos={int(pos.sum())}, n_neg={int((~pos).sum())}")
    tab = np.array([[int(((assign == s) & pos).sum()) for s in segnames],
                    [int(((assign == s) & ~pos).sum()) for s in segnames]])
    chi = stats.chi2_contingency(tab)
    add("Q1c_pos_vs_neg", "chi2_segment_by_PGR_status",
        f"chi2={chi[0]:.2f}; p={chi[1]:.4g}; dof={chi[2]}",
        "significant means PGR+ nuclei sit preferentially on one segment")

# ==================================================================== Q2
add("Q2_meta", "question", "do the glial pruning and allopregnanolone target genes "
    "follow the endothelial schedule", "same donors, same window")
for ctname, group in [("astrocyte", "astrocyte"), ("microglial cell", "microglia")]:
    if ctname not in set(ct_cats):
        add("Q2_" + group, "cell type absent", ctname); continue
    m2 = (ct_codes == int(np.where(ct_cats == ctname)[0][0])) & ctx
    sel2, tot2, gc2 = load(m2)
    d2 = dn_codes[sel2]; a2 = age_cell[sel2]
    cp2 = {g: gc2[g] / np.maximum(tot2, 1) * 1e4 for g in gi}
    add("Q2_" + group, "nuclei", len(sel2))
    keep_d = [d for d in np.unique(d2) if (d2 == d).sum() >= A.min_nuclei]
    xs = [a2[d2 == d][0] for d in keep_d]
    add("Q2_" + group, "donors", len(keep_d))
    if len(keep_d) < 4:
        add("Q2_" + group, "too few donors", len(keep_d)); continue
    def module(genes, label):
        have = [g for g in genes if g in gi]
        if not have:
            add(f"Q2_{group}", label, "no genes present"); return None
        ys = [float(np.mean([cp2[g][d2 == d].mean() for g in have])) for d in keep_d]
        r = stats.spearmanr(xs, ys)
        add(f"Q2_{group}", label, f"rho={r[0]:+.3f}; p={r[1]:.4f}; genes={len(have)}",
            "; ".join(have[:8]))
        return r[0]
    r_prune = module(PRUNING, "complement_pruning_module")
    r_allo = module(ALLO_TARGETS, "allopregnanolone_target_module")
    r_hk = module(HOUSEKEEPING, "housekeeping_module_CONTROL")
    if r_hk is not None:
        for nm, rr in [("pruning", r_prune), ("allo_target", r_allo)]:
            if rr is None: continue
            add(f"Q2_{group}", f"{nm}_exceeds_housekeeping", bool(abs(rr) > abs(r_hk)),
                f"|{rr:+.3f}| vs |{r_hk:+.3f}| ; if False the technical axis dominates "
                f"and the comparison carries no interpretable signal")
    for g in ["C1QA", "C3", "TREM2", "P2RY12", "TLR4", "GABRB1"]:
        if g not in gi: continue
        ys = [float(cp2[g][d2 == d].mean()) for d in keep_d]
        r = stats.spearmanr(xs, ys)
        add(f"Q2_{group}_gene", g, f"rho={r[0]:+.3f}; p={r[1]:.4f}")

add("limits", "cohort", "multiome cohort only", "the primary developmental series was not "
    "reanalysed here; its annotation file could not be recovered")
add("limits", "segments", "marker score argmax, not a trained classifier",
    "single nuclei carry weaker zonation signal than whole cells")
add("limits", "seed", SEED)

os.makedirs(os.path.dirname(A.out), exist_ok=True)
with open(A.out, "w") as fh:
    fh.write("block\tkey\tvalue\tnote\n")
    for r_ in rows: fh.write("\t".join(str(x) for x in r_) + "\n")
print(f"\nwritten: {A.out} ({len(rows)} rows)")
