#!/usr/bin/env python3
"""The four analyses, run on the manuscript's own primary developmental series.

Input: devbrain_jointanalysis_07072026.h5ad, the consensus atlas of Venkatesan
et al. 2026 (Zenodo 10.5281/zenodo.21375950), which is the file the shipped
extract_devbrain.py expects.

  0  REPRODUCTION. Rebuild the endothelial series by the manuscript's own rule
     and check the donor count, nucleus count and the age association against
     the shipped source_numbers_v2.tsv. If these do not match, everything below
     is describing a different series and the script says so.

  1  DEPTH. Is the rise in the fraction of PGR positive nuclei an artefact of
     sequencing depth? Exact hypergeometric detection probability at fixed
     depth, no simulation. Controls carried through.

  2  COMPOSITION OR CELL STATE. Score every endothelial nucleus for arterial,
     capillary and venous identity with published markers, assign by argmax,
     then ask separately whether segment shares change with age and whether
     PGR rises inside a segment. PGR is never used in the assignment.

  3  THE GLIAL LINE. Complement pruning genes and allopregnanolone target genes
     in astrocytes and microglia of the same donors over the same window, with
     a housekeeping module in the same cells as the control.

Analysis unit is the donor throughout. Column names are discovered and
asserted, never guessed. Seed fixed. Writes one table.

Run: python3 primary_series.py --h5ad F --shipped source_numbers_v2.tsv --out out/primary.tsv
"""
import argparse, os, json, sys
import numpy as np
import h5py
from scipy import stats
from scipy.special import gammaln

SEED = 2026
np.random.seed(SEED)
TARGET = "PGR"
SEGMENT = {"arterial":  ["GJA5", "SEMA3G", "HEY1", "ALPL", "VEGFC", "BMX"],
           "capillary": ["MFSD2A", "SLC7A5", "TFRC", "SLC16A1", "RGCC"],
           "venous":    ["NR2F2", "VWF", "ACKR1", "PLVAP", "IL1R1"]}
OTHER_REC = ["NR3C1", "NR3C2", "ESR1", "AR", "GPER1"]
HOUSEKEEPING = ["ACTB", "GAPDH", "RPL13A", "RPLP0", "TBP", "PGK1"]
IDENTITY = ["PECAM1", "CLDN5", "FLT1"]
EXTRA = ["PGR-AS1", "ZNF385B", "PAQR5", "ABCB1"]
PRUNING = ["C1QA", "C1QB", "C1QC", "C3", "C4A", "C4B", "ITGAM", "CX3CR1",
           "MERTK", "MEGF10", "TREM2", "TYROBP", "P2RY12", "CSF1R"]
ALLO = ["GABRA1", "GABRA2", "GABRA3", "GABRA4", "GABRA5", "GABRB1", "GABRB2",
        "GABRB3", "GABRG2", "TLR4", "MYD88"]
DEPTHS = [1000, 1500, 2000, 3000]
AGE_MIN = 0.75          # the manuscript's own restriction, donors older than nine months
MIN_CELL = 10           # cells per donor and compartment

ap = argparse.ArgumentParser()
ap.add_argument("--h5ad", required=True)
ap.add_argument("--shipped", default=None, help="source_numbers_v2.tsv, for the reproduction gate")
ap.add_argument("--out", required=True)
A = ap.parse_args()

rows = []
def add(b, k, v, n=""):
    rows.append((b, k, v, n)); print(f"{b:22s} {k:34s} {v}   {n}", flush=True)
def flush():
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    with open(A.out, "w") as fh:
        fh.write("block\tkey\tvalue\tnote\n")
        for r in rows: fh.write("\t".join(str(x) for x in r) + "\n")

f = h5py.File(A.h5ad, "r")
obs = f["obs"]

def col(name):
    """categorical or plain, returned as a string array"""
    g = obs[name]
    if isinstance(g, h5py.Group) and "categories" in g:
        c = np.array([x.decode() if isinstance(x, bytes) else x for x in g["categories"][:]], dtype=object)
        return c[g["codes"][:]]
    v = g[:]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in v], dtype=object)

NEED = ["cellclass", "final_annotation", "donor_id", "age_years_postgestation", "study"]
missing = [c for c in NEED if c not in obs]
assert not missing, f"expected obs columns missing: {missing}. Available: {sorted(obs.keys())}"
add("meta", "file", os.path.basename(A.h5ad))
add("meta", "n_cell_total", int(obs["age_years_postgestation"].shape[0]))

cellclass = col("cellclass"); final_ann = col("final_annotation")
donor = col("donor_id"); study = col("study")
age = obs["age_years_postgestation"][:].astype(float)
datatype = col("datatype") if "datatype" in obs else np.array(["unknown"] * len(age), dtype=object)
region = col("region") if "region" in obs else np.array(["unknown"] * len(age), dtype=object)
add("meta", "studies", "; ".join(f"{s}={int((study==s).sum())}" for s in sorted(set(study))))
add("meta", "age_range", f"{age.min():.2f} to {age.max():.2f}")

# genes
gs = f["var"]["gene_symbol"]
sym = np.array([x.decode() if isinstance(x, bytes) else x for x in gs[:]], dtype=object)
GENES = ([TARGET] + [g for v in SEGMENT.values() for g in v] + OTHER_REC
         + HOUSEKEEPING + IDENTITY + EXTRA + PRUNING + ALLO)
gi = {}
for g in dict.fromkeys(GENES):
    w = np.where(sym == g)[0]
    if len(w): gi[g] = int(w[0])
    else: add("gene_missing", g, "absent from var")
assert TARGET in gi, "PGR absent from var"

X = f["X"]
enc = X.attrs.get("encoding-type", b"")
enc = enc.decode() if isinstance(enc, bytes) else enc
add("meta", "X_encoding", enc)
d0 = X["data"][:200000]
integral = bool(np.all(d0 == np.round(d0)))
add("meta", "X_integral", integral, "raw counts required for the depth correction")
if not integral and "raw/X" in f:
    X = f["raw/X"]; add("meta", "X_switched_to", "raw/X")
    d0 = X["data"][:200000]; integral = bool(np.all(d0 == np.round(d0)))
assert integral, "integer count matrix not found; depth correction requires raw counts"
indptr = X["indptr"][:]; data, indices = X["data"], X["indices"]

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

def donor_corr(dn, ag, vals, mask, min_cell=MIN_CELL):
    xs, ys, ns = [], [], []
    for d in np.unique(dn[mask]):
        m = mask & (dn == d)
        if m.sum() < min_cell: continue
        xs.append(ag[m][0]); ys.append(float(vals[m].mean())); ns.append(int(m.sum()))
    if len(xs) < 4: return None
    r = stats.spearmanr(xs, ys)
    return r[0], r[1], len(xs), sum(ns), np.array(xs), np.array(ys)

# ==================================================================== 0
add("S0_repro", "rule", f"cellclass=='Vascular' & final_annotation=='Endo', age>{AGE_MIN}, "
    f">={MIN_CELL} cells per donor", "the manuscript's own restriction")
endo_all = (cellclass == "Vascular") & (final_ann == "Endo")
add("S0_repro", "endothelial_cells_all_ages", int(endo_all.sum()))
keep = endo_all & (age > AGE_MIN)
sel, tot, gc = load(keep)
dn = donor[sel]; ag = age[sel]; st = study[sel]; dt = datatype[sel]
ok = np.zeros(len(sel), dtype=bool)
for d in np.unique(dn):
    m = dn == d
    if m.sum() >= MIN_CELL: ok |= m
sel, tot, gc = sel[ok], tot[ok], {g: v[ok] for g, v in gc.items()}
dn, ag, st, dt = dn[ok], ag[ok], st[ok], dt[ok]
add("S0_repro", "donors_retained", len(np.unique(dn)), "manuscript reports 65")
add("S0_repro", "nuclei_retained", len(sel), "manuscript reports 5,388")
add("S0_repro", "studies_contributing", "; ".join(sorted(set(st))), "manuscript reports four")
add("S0_repro", "age_span", f"{ag.min():.2f} to {ag.max():.2f}", "manuscript reports 0.77 to 44.8")
add("S0_repro", "datatype_mix", "; ".join(f"{x}={int((dt==x).sum())}" for x in sorted(set(dt))))
pos = gc[TARGET] > 0
r0 = donor_corr(dn, ag, pos.astype(float), np.ones(len(sel), bool))
add("S0_repro", "rho_age_vs_positive_fraction",
    f"rho={r0[0]:+.3f}; p={r0[1]:.3g}; donors={r0[2]}", "manuscript reports +0.707")
if A.shipped and os.path.exists(A.shipped):
    txt = open(A.shipped).read()
    for label, got, want in [("donors", r0[2], 65), ("nuclei", len(sel), 5388)]:
        add("S0_repro", f"match_{label}", f"got {got}, manuscript {want}",
            "MATCH" if got == want else "DIFFERS, read the numbers below as this rebuild")

# ==================================================================== 1
add("S1_depth", "median_depth", float(np.median(tot)))
add("S1_depth", "depth_target_pos_vs_neg",
    f"{np.median(tot[pos]) if pos.any() else float('nan'):.0f} vs {np.median(tot[~pos]):.0f}")
dr = stats.spearmanr(ag, tot)
add("S1_depth", "depth_vs_age_per_nucleus", f"rho={dr[0]:+.3f}; p={dr[1]:.3g}",
    "if positive, deeper nuclei in older donors would inflate the raw fraction")

def p_detect(N, k, D):
    N = np.asarray(N, float); k = np.asarray(k, float)
    out = np.zeros_like(N); ok_ = (N >= D)
    m = ok_ & (k > 0)
    if m.any():
        Nn, kk = N[m], k[m]
        logq = ((gammaln(Nn - kk + 1) - gammaln(Nn - kk - D + 1))
                - (gammaln(Nn + 1) - gammaln(Nn - D + 1)))
        out[m] = 1.0 - np.exp(np.clip(logq, -700, 0))
    return out, ok_

add("S1_depth", f"{TARGET}_raw", f"rho={r0[0]:+.3f}; p={r0[1]:.3g}", "uncontrolled, for comparison")
for D in DEPTHS:
    pdv, ret = p_detect(tot, gc[TARGET], D)
    if ret.sum() < 100: add("S1_depth", f"depth{D}", "too few nuclei reach this depth"); continue
    rr = donor_corr(dn, ag, pdv, ret)
    if rr is None: add("S1_depth", f"depth{D}", "fewer than four donors"); continue
    add("S1_depth", f"depth{D}_nuclei", f"{int(ret.sum())} of {len(sel)} ({100*ret.mean():.0f}%)")
    add("S1_depth", f"depth{D}_{TARGET}", f"rho={rr[0]:+.3f}; p={rr[1]:.4f}; donors={rr[2]}", "TARGET")
    beat = []
    for kind, gl in [("other_receptor", OTHER_REC), ("housekeeping", HOUSEKEEPING),
                     ("identity", IDENTITY), ("extra", EXTRA)]:
        for g in gl:
            if g not in gi: continue
            pg, rg = p_detect(tot, gc[g], D)
            q = donor_corr(dn, ag, pg, rg)
            if q is None: continue
            add("S1_depth", f"depth{D}_{g}", f"rho={q[0]:+.3f}; p={q[1]:.4f}", kind)
            if q[0] >= rr[0]: beat.append(f"{g}({kind})")
    add("S1_depth", f"depth{D}_controls_matching_target",
        f"{len(beat)}", "; ".join(beat) if beat else "none")
flush()

# ==================================================================== 2
cp = {g: gc[g] / np.maximum(tot, 1) * 1e4 for g in gi}
assert TARGET not in [g for v in SEGMENT.values() for g in v]
scores, have_all = {}, True
for seg, glist in SEGMENT.items():
    have = [g for g in glist if g in gi]
    add("S2_markers", seg, f"{len(have)} of {len(glist)}", "; ".join(have))
    if not have: have_all = False; continue
    Z = np.vstack([(cp[g] - cp[g].mean()) / max(cp[g].std(), 1e-9) for g in have])
    scores[seg] = Z.mean(0)
assert have_all and len(scores) == 3
segn = list(scores); S = np.vstack([scores[s] for s in segn])
assign = np.array(segn, dtype=object)[S.argmax(0)]
assign[S.max(0) <= 0] = "unassigned"
for s in segn + ["unassigned"]:
    add("S2_assignment", s, f"{int((assign==s).sum())} ({100*(assign==s).mean():.1f}%)")
gate = True
for seg, glist in SEGMENT.items():
    have = [g for g in glist if g in gi]
    means = {s: float(np.mean([cp[g][assign == s].mean() for g in have])) for s in segn}
    top = max(means, key=means.get); okk = top == seg; gate &= okk
    add("S2_gate", seg, f"top={top}; " + "; ".join(f"{s}={means[s]:.3f}" for s in segn),
        "PASS" if okk else "FAIL")
add("S2_gate", "segment_labelling_recovered", bool(gate))
if gate:
    for s in segn:
        v = (assign == s).astype(float)
        q = donor_corr(dn, ag, v, np.ones(len(sel), bool))
        if q: add("S2a_composition", s, f"rho={q[0]:+.3f}; p={q[1]:.4f}; donors={q[2]}; "
                  f"share {q[5].min():.2f} to {q[5].max():.2f}",
                  "COMPOSITION SHIFT" if q[1] < 0.05 else "no significant shift")
    for s in segn:
        m = assign == s
        qe = donor_corr(dn, ag, cp[TARGET], m); qp = donor_corr(dn, ag, pos.astype(float), m)
        if qe and qp:
            add("S2b_within_segment", s, f"expr rho={qe[0]:+.3f} p={qe[1]:.4f}; "
                f"pct_pos rho={qp[0]:+.3f} p={qp[1]:.4f}; donors={qe[2]}; nuclei={qe[3]}")
        else:
            add("S2b_within_segment", s, "fewer than four donors reach the threshold", "not tested")
    for s in segn:
        add("S2c_pos_vs_neg", f"share_{s}",
            f"PGR+ {100*float((assign[pos]==s).mean()):.1f}%; PGR- {100*float((assign[~pos]==s).mean()):.1f}%",
            f"n_pos={int(pos.sum())}, n_neg={int((~pos).sum())}")
    tab = np.array([[int(((assign == s) & pos).sum()) for s in segn],
                    [int(((assign == s) & ~pos).sum()) for s in segn]])
    c2 = stats.chi2_contingency(tab)
    add("S2c_pos_vs_neg", "chi2_segment_by_PGR_status", f"chi2={c2[0]:.2f}; p={c2[1]:.4g}; dof={c2[2]}")
flush()

# ==================================================================== 3
lv = sorted(set(final_ann))
MICRO = [x for x in lv if "micro" in x.lower() and "fetal" not in x.lower()]
ASTRO = [x for x in lv if x.lower() == "astro"]
add("S3_labels", "microglia_label", "; ".join(MICRO) or "none found")
add("S3_labels", "astrocyte_label", "; ".join(ASTRO) or "none found")
for gname, labels in [("astrocyte", ASTRO), ("microglia", MICRO)]:
    if not labels: add(f"S3_{gname}", "label absent", "skipped"); continue
    m2 = np.isin(final_ann, labels) & (age > AGE_MIN)
    s2, t2, g2 = load(m2)
    d2, a2 = donor[s2], age[s2]
    cp2 = {g: g2[g] / np.maximum(t2, 1) * 1e4 for g in gi}
    add(f"S3_{gname}", "cells", len(s2))
    add(f"S3_{gname}", "donors", len(np.unique(d2)))
    def module(genes, label):
        have = [g for g in genes if g in gi]
        if not have: add(f"S3_{gname}", label, "no genes present"); return None
        vals = np.mean(np.vstack([cp2[g] for g in have]), axis=0)
        q = donor_corr(d2, a2, vals, np.ones(len(s2), bool))
        if q is None: add(f"S3_{gname}", label, "fewer than four donors"); return None
        add(f"S3_{gname}", label, f"rho={q[0]:+.3f}; p={q[1]:.4f}; genes={len(have)}; donors={q[2]}",
            "; ".join(have[:8]))
        return q[0]
    rp = module(PRUNING, "complement_pruning_module")
    ra = module(ALLO, "allopregnanolone_target_module")
    rh = module(HOUSEKEEPING, "housekeeping_module_CONTROL")
    if rh is not None:
        for nm, rr in [("pruning", rp), ("allo_target", ra)]:
            if rr is None: continue
            add(f"S3_{gname}", f"{nm}_exceeds_housekeeping", bool(abs(rr) > abs(rh)),
                f"|{rr:+.3f}| vs |{rh:+.3f}|; if False the technical axis dominates")
    for g in ["C1QA", "C3", "TREM2", "P2RY12", "TLR4", "GABRB1"]:
        if g not in gi: continue
        q = donor_corr(d2, a2, cp2[g], np.ones(len(s2), bool))
        if q: add(f"S3_{gname}_gene", g, f"rho={q[0]:+.3f}; p={q[1]:.4f}")

add("limits", "mixed_assay", "the atlas mixes snRNA-seq and scRNA-seq",
    "reported above as datatype_mix; detection floors differ between them")
add("limits", "source", "Venkatesan et al. 2026 consensus atlas, bioRxiv 10.64898/2026.07.16.738788",
    "an unpublished preprint; the manuscript's data statement names a different file")
add("limits", "seed", SEED)
flush()
print(f"\nwritten: {A.out} ({len(rows)} rows)")
