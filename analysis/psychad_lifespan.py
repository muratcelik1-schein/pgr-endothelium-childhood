#!/usr/bin/env python3
"""The childhood donors of the adult resource.

The manuscript uses PsychAD for adult cortex. The same release also carries
donors from one month of age upward, which were not used. This script asks the
developmental question of those donors.

What this is and is not. The donors are different from those of the primary
series, so this is an independent donor set. The annotation and processing
pipeline is the one the manuscript already relies on for its adult results, so
it is NOT an independent pipeline. The output says so, and any claim made from
it must say so too.

Two files of the release overlap in their young donors. Donors are therefore
pooled by identifier across files and each donor is counted once, with the
overlap reported.

Design rules enforced: column names discovered and asserted; endothelial
identity checked before the target gene is examined; analysis unit is the
donor; positive, negative and specificity controls carried through; the depth
correction computed exactly, with no simulation; seed fixed.

Run: python3 psychad_lifespan.py --dir /path/to/psychad --out out/psychad_lifespan.tsv
"""
import argparse, os, re, sys, json
import numpy as np
import h5py
from scipy import stats
from scipy.special import gammaln

SEED = 2026
np.random.seed(SEED)
TARGET = "PGR"
OTHER_REC = ["NR3C1", "NR3C2", "ESR1", "AR", "GPER1"]
HOUSEKEEPING = ["ACTB", "GAPDH", "RPL13A", "RPLP0", "TBP", "PGK1"]
IDENTITY = ["PECAM1", "CLDN5", "FLT1"]
EXTRA = ["PGR-AS1", "ZNF385B", "PAQR5", "ABCB1"]
COHORTS = ["Aging", "HBCC", "MSSM", "RADC"]
MIN_NUC = 10
DEPTHS = [500, 1000, 2000]

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--min-nuclei", type=int, default=MIN_NUC)
A = ap.parse_args()

rows = []
def add(b, k, v, n=""):
    rows.append((b, k, v, n)); print(f"{b:20s} {k:34s} {v}   {n}", flush=True)
def flush():
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    with open(A.out, "w") as fh:
        fh.write("block\tkey\tvalue\tnote\n")
        for r in rows: fh.write("\t".join(str(x) for x in r) + "\n")

def stage_years(s):
    m = re.match(r"^(\d+)-year-old stage$", s)
    if m: return float(m.group(1))
    m = re.match(r"^(\d+)-month-old stage$", s)
    if m: return float(m.group(1)) / 12
    if "newborn" in s.lower(): return 0.0
    if s.startswith("80 year-old and over"): return 80.0
    return None

GENES = [TARGET] + OTHER_REC + HOUSEKEEPING + IDENTITY + EXTRA
donors = {}          # donor -> dict(age, cohort, nuclei, tot list, gene counts)
seen_in = {}         # donor -> set of cohorts, to measure the overlap

for coh in COHORTS:
    p = os.path.join(A.dir, f"psychad_{coh}.h5ad")
    if not os.path.exists(p):
        add("meta", f"missing_{coh}", p); continue
    f = h5py.File(p, "r"); o = f["obs"]
    def cat(n):
        g = o[n]
        assert isinstance(g, h5py.Group) and "categories" in g, f"{n} is not categorical"
        c = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]], dtype=object)
        return c[g["codes"][:]]
    for need in ("development_stage", "donor_id", "cell_type"):
        assert need in o, f"{coh}: {need} missing"
    st = cat("development_stage"); dn = cat("donor_id"); ct = cat("cell_type")
    ENDO = sorted({x for x in set(ct) if "endothelial" in x.lower()})
    assert len(ENDO) == 1, f"{coh}: a single endothelial label was expected, found {ENDO}"
    add("meta", f"{coh}_endothelial_label", ENDO[0])
    is_endo = ct == ENDO[0]
    age = np.array([stage_years(s) if stage_years(s) is not None else np.nan for s in st])
    young = is_endo & (age < 20) & ~np.isnan(age)
    add("meta", f"{coh}_endothelial_nuclei_under20", int(young.sum()))
    if young.sum() == 0:
        f.close(); continue

    vn = f["var"]
    key = "feature_name" if "feature_name" in vn else "_index"
    g = vn[key]
    if isinstance(g, h5py.Group) and "categories" in g:
        vc = np.array([x.decode() if isinstance(x, bytes) else x for x in g["categories"][:]], dtype=object)
        sym = vc[g["codes"][:]]
    else:
        sym = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]], dtype=object)
    gi = {}
    for gg in GENES:
        w = np.where(sym == gg)[0]
        if len(w): gi[gg] = int(w[0])
    assert TARGET in gi, f"{coh}: PGR absent from var"

    X = f["raw/X"] if "raw/X" in f else f["X"]
    d0 = X["data"][:200000]
    assert np.all(d0 == np.round(d0)), f"{coh}: integer count matrix not found"
    indptr = X["indptr"][:]; data, indices = X["data"], X["indices"]
    want = np.array(sorted(gi.values())); back = {v: k for k, v in gi.items()}
    sel = np.where(young)[0]
    for r in sel:
        a_, b_ = indptr[r], indptr[r + 1]
        if b_ <= a_: continue
        dd = data[a_:b_]; ii = indices[a_:b_]
        d = dn[r]
        rec = donors.setdefault(d, dict(age=float(age[r]), cohorts=set(), tot=[], g={k: [] for k in GENES}))
        rec["cohorts"].add(coh)
        rec["tot"].append(float(dd.sum()))
        hit = {back[int(h)]: float(dd[ii == h].sum()) for h in np.intersect1d(ii, want)}
        for k in GENES: rec["g"][k].append(hit.get(k, 0.0))
    f.close()

add("meta", "unique_donors_under20", len(donors))
ov = [d for d, r in donors.items() if len(r["cohorts"]) > 1]
add("meta", "donors_present_in_more_than_one_file", len(ov),
    "counted once; the release ships overlapping subsets")
add("meta", "total_endothelial_nuclei", sum(len(r["tot"]) for r in donors.values()))

keep = {d: r for d, r in donors.items() if len(r["tot"]) >= A.min_nuclei}
add("meta", "donors_retained", len(keep), f"threshold {A.min_nuclei} endothelial nuclei")
if len(keep) < 8:
    add("RESULT", "verdict", "too few donors retained", "no correlation computed"); flush(); sys.exit(0)

ages = np.array([r["age"] for r in keep.values()])
for lo, hi, lab in [(0, 2, "<2y"), (2, 12, "2-11y"), (12, 20, "12-19y")]:
    m = (ages >= lo) & (ages < hi)
    add("bands", lab, f"donors={int(m.sum())}; nuclei="
        f"{sum(len(r['tot']) for r,k in zip(keep.values(), m) if k)}")
add("meta", "age_range", f"{ages.min():.2f} to {ages.max():.2f}")

tot = {d: np.array(r["tot"]) for d, r in keep.items()}
gc = {d: {k: np.array(v) for k, v in r["g"].items()} for d, r in keep.items()}

# --- gate: endothelial identity must be expressed
for g in IDENTITY:
    v = [float(gc[d][g].sum() / max(tot[d].sum(), 1) * 1e4) for d in keep]
    add("gate_identity", g, f"median {np.median(v):.3f} counts per 10,000 across donors",
        "PASS" if np.median(v) > 1 else "FAIL, endothelium not recovered")

def donor_corr(vals):
    xs = np.array([keep[d]["age"] for d in keep]); ys = np.array([vals[d] for d in keep])
    r = stats.spearmanr(xs, ys)
    return r[0], r[1]

# --- raw
for kind, gl in [("target", [TARGET]), ("other_receptor", OTHER_REC),
                 ("housekeeping", HOUSEKEEPING), ("identity", IDENTITY), ("extra", EXTRA)]:
    for g in gl:
        e = {d: float(gc[d][g].sum() / max(tot[d].sum(), 1) * 1e4) for d in keep}
        p = {d: float((gc[d][g] > 0).mean()) for d in keep}
        re_, pe = donor_corr(e); rp, pp = donor_corr(p)
        add("raw", g, f"expr rho={re_:+.3f} p={pe:.4f}; pct_pos rho={rp:+.3f} p={pp:.4f}", kind)

# --- depth correction, exact
def p_detect(N, k, D):
    N = np.asarray(N, float); k = np.asarray(k, float)
    out = np.zeros_like(N); ok = N >= D; m = ok & (k > 0)
    if m.any():
        Nn, kk = N[m], k[m]
        lq = ((gammaln(Nn - kk + 1) - gammaln(Nn - kk - D + 1))
              - (gammaln(Nn + 1) - gammaln(Nn - D + 1)))
        out[m] = 1 - np.exp(np.clip(lq, -700, 0))
    return out, ok

alltot = np.concatenate([tot[d] for d in keep])
add("depth", "median_nucleus_depth", float(np.median(alltot)))
posall = np.concatenate([gc[d][TARGET] for d in keep]) > 0
add("depth", "depth_target_pos_vs_neg",
    f"{np.median(alltot[posall]) if posall.any() else float('nan'):.0f} vs {np.median(alltot[~posall]):.0f}")
for D in DEPTHS:
    vals, nkeep = {}, 0
    for d in keep:
        pd_, ret = p_detect(tot[d], gc[d][TARGET], D)
        if ret.sum() < A.min_nuclei: continue
        vals[d] = float(pd_[ret].mean()); nkeep += int(ret.sum())
    if len(vals) < 8: add("depth", f"depth{D}", f"only {len(vals)} donors reach it"); continue
    xs = np.array([keep[d]["age"] for d in vals]); ys = np.array([vals[d] for d in vals])
    r = stats.spearmanr(xs, ys)
    add("depth", f"depth{D}_{TARGET}", f"rho={r[0]:+.3f}; p={r[1]:.4f}; donors={len(vals)}; nuclei={nkeep}", "TARGET")
    beat = []
    for kind, gl in [("other_receptor", OTHER_REC), ("housekeeping", HOUSEKEEPING), ("identity", IDENTITY)]:
        for g in gl:
            v2 = {}
            for d in keep:
                pg, rg = p_detect(tot[d], gc[d][g], D)
                if rg.sum() < A.min_nuclei: continue
                v2[d] = float(pg[rg].mean())
            if len(v2) < 8: continue
            rr = stats.spearmanr(np.array([keep[d]["age"] for d in v2]), np.array([v2[d] for d in v2]))
            add("depth", f"depth{D}_{g}", f"rho={rr[0]:+.3f}; p={rr[1]:.4f}", kind)
            if rr[0] >= r[0]: beat.append(g)
    add("depth", f"depth{D}_controls_matching_target", len(beat), "; ".join(beat) or "none")

add("limits", "independence", "independent donors, same resource and pipeline",
    "the manuscript already uses this release for its adult results, so this is not an independent pipeline")
add("limits", "young_donor_numbers", "the youngest bands are small", "see the bands block")
add("limits", "seed", SEED)
flush()
print(f"\nwritten: {A.out} ({len(rows)} rows)")
