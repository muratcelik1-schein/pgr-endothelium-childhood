#!/usr/bin/env python3
"""Third cohort test: does endothelial PGR rise across postnatal age in the
multiome cohort of Clarence et al., Nat Genet 57 (2025), DOI 10.1038/s41588-025-02083-8?

This cohort is independent of the primary series in group, donors and assay
(10x multiome, single nuclei, four brain regions, ten donors spanning infancy
to 62 years). Only the RNA modality is analysed here; the chromatin modality
is not distributed in this file.

Design rules enforced, not assumed:
  * Every obs column name and label spelling is discovered at runtime and
    asserted. Nothing is guessed.
  * The cell type labelling is validated with endothelial identity markers
    BEFORE the target gene is examined. If endothelium does not separate, the
    run stops and reports "cell type labelling not recovered".
  * The analysis unit is the donor. No donor contributes more than one
    observation to any correlation.
  * Positive controls (endothelial identity), negative controls (housekeeping)
    and specificity controls (five related steroid receptors) are carried
    through every step and reported whether or not they behave.
  * The age variable is a rank. One stage is labelled "infant" with no exact
    age, so only its order is used and this is stated in the output.
  * Seed fixed. Nothing is written except the output table.

Run: python3 clarence_replication.py --h5ad data/clarence_rna.h5ad --out out/clarence_numbers.tsv
"""
import argparse, os, re, sys, json
import numpy as np
import h5py
from scipy import sparse, stats

SEED = 2026
np.random.seed(SEED)

TARGET = "PGR"
IDENTITY = ["PECAM1", "CLDN5", "FLT1", "VWF"]          # positive control, endothelium
OTHER_RECEPTORS = ["NR3C1", "NR3C2", "ESR1", "AR", "GPER1"]   # specificity
HOUSEKEEPING = ["ACTB", "GAPDH", "RPL13A", "RPLP0", "TBP", "PGK1"]  # technical drift
EXTRA = ["PGR-AS1", "PAQR5", "ZNF385B", "ABCB1"]       # reported alongside in the paper
MURAL = ["PDGFRB", "RGS5", "ACTA2", "DCN"]             # doublet check
CORTEX = ["dorsolateral prefrontal cortex", "anterior cingulate cortex"]
MIN_NUCLEI = 10                                        # same threshold as the paper

ap = argparse.ArgumentParser()
ap.add_argument("--h5ad", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--min-nuclei", type=int, default=MIN_NUCLEI)
A = ap.parse_args()

rows = []
def add(block, key, value, note=""):
    rows.append((block, key, value, note))
    print(f"{block:22s} {key:38s} {value}   {note}", flush=True)

f = h5py.File(A.h5ad, "r")

# ------------------------------------------------------------------ discovery
def cat(name):
    g = f["obs"][name]
    assert isinstance(g, h5py.Group) and "categories" in g, f"{name} is not categorical"
    cats = [x.decode() if isinstance(x, bytes) else x for x in g["categories"][:]]
    return np.array(cats, dtype=object), g["codes"][:]

need = ["cell_type", "donor_id", "development_stage", "tissue", "sex"]
missing = [c for c in need if c not in f["obs"]]
assert not missing, f"expected obs columns missing: {missing}"

ct_cats, ct_codes = cat("cell_type")
dn_cats, dn_codes = cat("donor_id")
st_cats, st_codes = cat("development_stage")
ti_cats, ti_codes = cat("tissue")
sx_cats, sx_codes = cat("sex")

ENDO = [c for c in ct_cats if "endothelial" in c.lower()]
assert len(ENDO) == 1, f"a single endothelial label was expected, found: {ENDO}"
ENDO = ENDO[0]
add("meta", "endothelial_label", ENDO, "discovered, not assumed")
add("meta", "n_cell_type", len(ct_cats), "; ".join(sorted(ct_cats)))
add("meta", "n_donor_total", len(dn_cats))
add("meta", "stages", "; ".join(sorted(st_cats)))
add("meta", "tissues", "; ".join(sorted(ti_cats)))

# age rank from the stage label. "infant stage" carries no exact age, so only
# its order is used, and that limit is written into the output.
def stage_age(lbl):
    m = re.match(r"^(\d+)-year-old stage$", lbl)
    if m: return float(m.group(1))
    if "infant" in lbl.lower(): return 1.0     # order only, below the 4-year stage
    raise AssertionError(f"unparsed stage label: {lbl!r}")
stage_age_map = {s: stage_age(s) for s in st_cats}
add("meta", "stage_to_age", json.dumps({k: v for k, v in sorted(stage_age_map.items(), key=lambda kv: kv[1])}),
    "infant placed at 1.0 for ordering only; no exact age is claimed")

# ------------------------------------------------------------------ genes
vn = f["var"]["feature_name"]
v_cats = np.array([x.decode() if isinstance(x, bytes) else x for x in vn["categories"][:]], dtype=object)
sym = v_cats[vn["codes"][:]]
GENES = [TARGET] + IDENTITY + OTHER_RECEPTORS + HOUSEKEEPING + EXTRA + MURAL
gi = {}
for g in GENES:
    w = np.where(sym == g)[0]
    if len(w) == 0:
        add("gene_missing", g, "absent from var", "reported, not silently dropped"); continue
    gi[g] = int(w[0])
    if len(w) > 1: add("gene_duplicate", g, len(w), "first index used")
assert TARGET in gi, "PGR absent from var"

# ------------------------------------------------------------------ counts
X = f["raw/X"]
d0 = X["data"][:200000]
assert np.all(d0 == np.round(d0)), "raw/X is not integer; raw counts were expected"
n_obs = f["obs"][f["obs"].attrs.get("_index", "_index")].shape[0]
indptr = X["indptr"][:]
data, indices = X["data"], X["indices"]

# per nucleus: total counts, and counts for each gene of interest
want = np.array(sorted(gi.values()))
want_set = {v: k for k, v in gi.items()}
tot = np.zeros(n_obs, dtype=np.float64)
gcount = {g: np.zeros(n_obs, dtype=np.float64) for g in gi}
CH = 20000
for s in range(0, n_obs, CH):
    e = min(s + CH, n_obs)
    lo, hi = indptr[s], indptr[e]
    dd = data[lo:hi]; ii = indices[lo:hi]
    ptr = indptr[s:e + 1] - lo
    for r in range(e - s):
        a, b = ptr[r], ptr[r + 1]
        if b <= a: continue
        tot[s + r] = dd[a:b].sum()
        idx = ii[a:b]
        hit = np.intersect1d(idx, want, assume_unique=False)
        for h in hit:
            gcount[want_set[int(h)]][s + r] = dd[a:b][idx == h].sum()
add("meta", "n_nucleus_total", int(n_obs))
add("meta", "median_counts_per_nucleus", float(np.median(tot)))

# ------------------------------------------------------------------ gate: identity
is_endo = ct_codes == int(np.where(ct_cats == ENDO)[0][0])
cp10k = lambda c, m: (c[m].sum() / max(tot[m].sum(), 1)) * 1e4
gate_ok = True
for g in IDENTITY:
    if g not in gi: continue
    per_ct = {c: cp10k(gcount[g], ct_codes == i) for i, c in enumerate(ct_cats)}
    top = max(per_ct, key=per_ct.get)
    ok = top == ENDO
    gate_ok &= ok
    add("gate_identity", g, f"top={top}; endo={per_ct[ENDO]:.4f}; "
        f"next={sorted(per_ct.values())[-2]:.4f}", "PASS" if ok else "FAIL")
add("gate_identity", "all_identity_markers_peak_in_endothelium", bool(gate_ok),
    "if False the cell type labelling is not recovered and no PGR result follows")
if not gate_ok:
    add("RESULT", "verdict", "cell type labelling not recovered", "stopping before the target gene")
    with open(A.out, "w") as fh:
        fh.write("block\tkey\tvalue\tnote\n")
        for r in rows: fh.write("\t".join(str(x) for x in r) + "\n")
    sys.exit(3)

# ------------------------------------------------------------------ per donor
def donor_table(mask_extra, label):
    out = []
    for di, dn in enumerate(dn_cats):
        m = is_endo & (dn_codes == di) & mask_extra
        n = int(m.sum())
        if n < A.min_nuclei: continue
        stages = {st_cats[c] for c in np.unique(st_codes[m])}
        assert len(stages) == 1, f"{dn} carries more than one stage: {stages}"
        st = stages.pop()
        rec = dict(donor=dn, stage=st, age=stage_age_map[st], n_nuclei=n,
                   total_counts=float(tot[m].sum()),
                   median_depth=float(np.median(tot[m])),
                   sex=sx_cats[np.unique(sx_codes[m])[0]])
        for g in gi:
            rec[g] = cp10k(gcount[g], m)
            rec[g + "_pct_pos"] = 100.0 * float((gcount[g][m] > 0).mean())
        out.append(rec)
    add(label, "n_donor_retained", len(out), f"threshold {A.min_nuclei} endothelial nuclei per donor")
    return out

all_tissue = np.ones(n_obs, dtype=bool)
ctx_codes = [int(np.where(ti_cats == t)[0][0]) for t in CORTEX if t in ti_cats]
assert ctx_codes, f"cortical tissue label not found, available: {list(ti_cats)}"
is_ctx = np.isin(ti_codes, ctx_codes)

for label, mask in [("cortex", is_ctx), ("all_regions", all_tissue)]:
    T = donor_table(mask, label)
    if len(T) < 4:
        add(label, "verdict", "too few donors retained", "no correlation computed"); continue
    age = np.array([r["age"] for r in T])
    add(label, "donors", "; ".join(f"{r['donor']}={r['stage']}(n={r['n_nuclei']})" for r in T))
    add(label, "age_range_used", f"{age.min()} to {age.max()}", "infant ordered at 1.0")
    add(label, "total_endothelial_nuclei", int(sum(r["n_nuclei"] for r in T)))

    for g in [TARGET] + OTHER_RECEPTORS + HOUSEKEEPING + IDENTITY + EXTRA:
        if g not in gi: continue
        y = np.array([r[g] for r in T]); p = np.array([r[g + "_pct_pos"] for r in T])
        rho, pv = stats.spearmanr(age, y)
        rho_p, pv_p = stats.spearmanr(age, p)
        kind = ("target" if g == TARGET else "housekeeping" if g in HOUSEKEEPING
                else "identity" if g in IDENTITY else "other_receptor" if g in OTHER_RECEPTORS else "extra")
        add(f"{label}_expr", g, f"rho={rho:+.3f}; p={pv:.4f}; "
            f"rho_pct_pos={rho_p:+.3f}; p={pv_p:.4f}", kind)

    # young against old, matching the paper's own contrast
    young = [r for r in T if r["age"] <= 6]; old = [r for r in T if r["age"] >= 20]
    if young and old:
        for g in [TARGET] + HOUSEKEEPING[:2] + OTHER_RECEPTORS[:2]:
            if g not in gi: continue
            ky = sum((gcount[g][is_endo & (dn_codes == int(np.where(dn_cats == r["donor"])[0][0])) & mask] > 0).sum() for r in young)
            ny = sum(r["n_nuclei"] for r in young)
            ko = sum((gcount[g][is_endo & (dn_codes == int(np.where(dn_cats == r["donor"])[0][0])) & mask] > 0).sum() for r in old)
            no = sum(r["n_nuclei"] for r in old)
            lo_y, hi_y = stats.beta.ppf([0.025, 0.975], [ky, ky + 1], [ny - ky + 1, ny - ky])
            lo_o, hi_o = stats.beta.ppf([0.025, 0.975], [ko, ko + 1], [no - ko + 1, no - ko])
            odds, pf = stats.fisher_exact([[ky, ny - ky], [ko, no - ko]])
            ey = np.mean([r[g] for r in young]); eo = np.mean([r[g] for r in old])
            add(f"{label}_contrast", g,
                f"young {ky}/{ny}={100*ky/ny:.2f}% [{100*np.nan_to_num(lo_y):.2f},{100*np.nan_to_num(hi_y):.2f}]; "
                f"old {ko}/{no}={100*ko/no:.2f}% [{100*np.nan_to_num(lo_o):.2f},{100*np.nan_to_num(hi_o):.2f}]; "
                f"fisher_p={pf:.3g}; expr_fold={eo/ey if ey else float('nan'):.2f}",
                "donors <=6y vs >=20y")

    # doublet check: do PGR positive endothelial nuclei carry mural markers
    mpos = is_endo & mask & (gcount[TARGET] > 0)
    mneg = is_endo & mask & (gcount[TARGET] == 0)
    for g in MURAL + IDENTITY:
        if g not in gi: continue
        add(f"{label}_doublet", g, f"PGR+ {cp10k(gcount[g], mpos):.4f}; PGR- {cp10k(gcount[g], mneg):.4f}",
            "mural signal higher in PGR+ would indicate doublets")
    add(f"{label}_doublet", "n_PGR_positive_endothelial_nuclei", int(mpos.sum()))
    add(f"{label}_doublet", "median_depth_PGR_pos_vs_neg",
        f"{np.median(tot[mpos]) if mpos.sum() else float('nan'):.0f} vs {np.median(tot[mneg]):.0f}",
        "a depth difference would confound the positive fraction")

    # is PGR endothelium specific here
    for i, c in enumerate(ct_cats):
        m = (ct_codes == i) & mask
        if m.sum() < 50: continue
        add(f"{label}_celltype", c, f"PGR={cp10k(gcount[TARGET], m):.4f}; "
            f"pct_pos={100*float((gcount[TARGET][m] > 0).mean()):.2f}; n={int(m.sum())}")

add("limits", "modality", "RNA only", "the chromatin modality is not in this file; accessibility is untested here")
add("limits", "infant_stage", "no exact age", "ordered below the 4-year stage; Spearman uses rank only")
add("limits", "donor_overlap", "not verified here",
    "donor identifiers differ from the primary series but overlap was not formally excluded")
add("limits", "seed", SEED)

os.makedirs(os.path.dirname(A.out), exist_ok=True)
with open(A.out, "w") as fh:
    fh.write("block\tkey\tvalue\tnote\n")
    for r in rows: fh.write("\t".join(str(x) for x in r) + "\n")
print(f"\nwritten: {A.out}  ({len(rows)} rows)")
