# -*- coding: utf-8 -*-
"""SINGLE SOURCE OF NUMBERS (v2).
Regenerates every number that enters the manuscript from the source data and writes source_numbers_v2.tsv.
Run:  python3 verify_all.py [--data /tmp/dev] [--out .]
Writes nothing except its output table."""
import argparse, json, os, gzip, collections, statistics as stat
import numpy as np, pandas as pd, h5py
from scipy.stats import spearmanr, rankdata

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.environ.get("PGR_DATA", "/tmp/dev"))
ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
A = ap.parse_args()
D = A.data
BANDS = [(0.75, 2), (2, 6), (6, 12), (12, 20), (20, 99)]
BLAB = ["0.75-2y", "2-6y", "6-12y", "12-20y", ">20y"]
rows = []
def add(block, key, value, note=""):
    rows.append(dict(block=block, key=key, value=value, note=note))
    print(f"  {block:22} {key:26} {value}")

# ---------------------------------------------------------------- 1. endothelial pseudobulk
print("\n[1] developmental endothelial series")
P = np.load(f"{D}/endo_pseudobulk.npy")
sym = np.array([str(x) for x in np.load(f"{D}/gene_sym.npy", allow_pickle=True)])
m = pd.read_csv(f"{D}/endo_meta.csv")
k = (m.age_y.values > 0.75) & (m.n.values >= 10)
CP = P[k] / np.maximum(P[k].sum(1, keepdims=True), 1) * 1e4
ag, st = m.age_y.values[k], m.study.values[k]
keep = np.where((CP > 0).mean(0) >= 0.20)[0]
G, X = sym[keep], CP[:, keep]
pos = {}
for i, g in enumerate(G):
    pos.setdefault(g, i)
Z = np.log1p(X)
for s in set(st):
    j = st == s
    Z[j] = (Z[j] - Z[j].mean(0)) / np.maximum(Z[j].std(0), 1e-9)
R = np.nan_to_num(np.array([spearmanr(ag, Z[:, i])[0] for i in range(Z.shape[1])]))
rank = (-R).argsort().argsort() + 1
Mb = np.array([X[(ag >= lo) & (ag < hi)].mean(0) for lo, hi in BANDS])
fold = (Mb[-1] + 1e-4) / (Mb[0] + 1e-4)
mono = np.all(np.diff(Mb, axis=0) >= -1e-9, axis=0)
zero = (X[ag < 2] == 0).mean(0) * 100

add("series", "n_donor", int(k.sum()))
add("series", "n_study", len(set(st)))
add("series", "age_range", f"{ag.min():.2f}-{ag.max():.2f}")
add("series", "n_gene_tested", len(keep))
for (lo, hi), lab in zip(BANDS, BLAB):
    add("series", f"n_donor_{lab}", int(((ag >= lo) & (ag < hi)).sum()))

def prof(g):
    i = pos[g]
    return dict(rho=round(float(R[i]), 3), rank=int(rank[i]), fold=round(float(fold[i]), 2),
                zero_pct=int(round(zero[i])), monotonic=bool(mono[i]),
                bands=[round(float(x), 4) for x in Mb[:, i]])
for g in ["PGR", "PGR-AS1", "ZNF385B", "NR3C1", "NR3C2", "ESR1", "AR", "GPER1",
          "ABCB1", "AKR1C1", "AKR1C2", "AKR1C3", "SRD5A1"]:
    if g in pos:
        add("endo_dev", g, json.dumps(prof(g)))

# ---------------------------------------------------------------- 1b. full series from birth
print("\n[1b] full postnatal series, two age thresholds")
FB = [(0, 0.5), (0.5, 0.75), (0.75, 2), (2, 6), (6, 12), (12, 20), (20, 99)]
FLAB = ["0-0.5y", "0.5-0.75y", "0.75-2y", "2-6y", "6-12y", "12-20y", ">20y"]
kf = m.n.values >= 10
CPf = P[kf] / np.maximum(P[kf].sum(1, keepdims=True), 1) * 1e4
agf = m.age_y.values[kf]
symi = {}
for i, g in enumerate(sym):
    symi.setdefault(g, i)
add("full_series", "n_donor", int(kf.sum()))
for (lo, hi), lab in zip(FB, FLAB):
    add("full_series", f"n_donor_{lab}", int(((agf >= lo) & (agf < hi)).sum()))
for g in ["PGR", "ABCB1"]:
    i = symi[g]
    v = [float(CPf[(agf >= lo) & (agf < hi), i].mean()) for lo, hi in FB]
    # early share: fraction of the total increase completed by two years
    early = (v[2] - v[0]) / (max(v) - v[0])
    add("full_series", g, json.dumps(dict(bands=[round(x, 4) for x in v],
        early_fraction_by_2y=round(float(early), 3), peak_band=FLAB[int(np.argmax(v))])))

# ---------------------------------------------------------------- 2. three-criteria screen
print("\n[2] three-criteria screen, genome wide")
for ft, zt in [(5, 50), (5, 75), (5, 80), (10, 75), (10, 80), (20, 80)]:
    sel = np.where((fold >= ft) & (zero >= zt) & mono)[0]
    order = sel[np.argsort(-R[sel])]
    names = [G[i] for i in order]
    add("gate", f"fold>={ft}_zero>={zt}", f"n={len(sel)}; PGR_rho_rank={names.index('PGR')+1 if 'PGR' in names else 'NA'}; top5={','.join(names[:5])}")

# ---------------------------------------------------------------- 3. robustness
print("\n[3] robustness")
los = []
for s in sorted(set(st)):
    j = st != s
    Rj = np.nan_to_num(np.array([spearmanr(ag[j], Z[j, q])[0] for q in range(Z.shape[1])]))
    los.append(int((-Rj).argsort().argsort()[pos["PGR"]] + 1))
add("robust", "LOSO_rank", json.dumps(dict(zip(sorted(set(st)), los))))
ws = {}
for s in sorted(set(st)):
    j = st == s
    r, p = spearmanr(ag[j], np.log1p(X[j, pos["PGR"]]))
    ws[s] = [int(j.sum()), round(float(r), 3), round(float(p), 4)]
add("robust", "within_study", json.dumps(ws))

z = Z[:, pos["PGR"]]
dep = P[k].sum(1)
add("robust", "rho_depth_PGR", round(float(spearmanr(dep, z)[0]), 3))
add("robust", "rho_depth_age", round(float(spearmanr(dep, ag)[0]), 3))
Ad = np.column_stack([np.ones(k.sum()), np.log(dep) - np.log(dep).mean(), m.n.values[k] - m.n.values[k].mean()])
rd = lambda v: v - Ad @ np.linalg.lstsq(Ad, v, rcond=None)[0]
add("robust", "rho_after_depth", round(float(spearmanr(rd(ag.astype(float)), rd(z))[0]), 3))

mu = pd.read_csv(f"{D}/mu_meta.tsv", sep="\t")
sxmap = pd.Series(mu.sex.values, index=mu.donor.values)
sv = np.array([sxmap.get(d, "?") for d in m.donor.values[k]])
f_ = (sv == "female").astype(float)
As = np.column_stack([np.ones(len(f_)), f_ - f_.mean()])
rs = lambda v: v - As @ np.linalg.lstsq(As, v, rcond=None)[0]
add("robust", "sex_counts", json.dumps(dict(collections.Counter(sv))))
add("robust", "rho_after_sex", round(float(spearmanr(rs(ag.astype(float)), rs(z))[0]), 3))
add("robust", "rho_female", round(float(spearmanr(ag[f_ == 1], z[f_ == 1])[0]), 3))
add("robust", "rho_male", round(float(spearmanr(ag[f_ == 0], z[f_ == 0])[0]), 3))

def partial(a, b, c):
    ra, rb, rc = map(rankdata, (a, b, c))
    Ap = np.column_stack([np.ones(len(rc)), rc - rc.mean()])
    rp = lambda v: v - Ap @ np.linalg.lstsq(Ap, v, rcond=None)[0]
    return float(spearmanr(rp(ra), rp(rb))[0])
add("robust", "PGR_ZNF385B_partial_age", round(partial(Z[:, pos["PGR"]], Z[:, pos["ZNF385B"]], ag), 3))

# ---------------------------------------------------------------- 4. VLMC and neuron panels
print("\n[4] VLMC / neuron panels")
M = np.load(f"{D}/mu_M.npy"); Mn = np.load(f"{D}/nu_M.npy")
sym2 = np.array([str(x) for x in np.load(f"{D}/mu_sym.npy", allow_pickle=True)])
men = pd.read_csv(f"{D}/nu_meta.tsv", sep="\t")
gi = {}
for i, g in enumerate(sym2):
    gi.setdefault(g, i)
def panel(ci, src, col, minc):
    Aa = M[np.arange(ci, len(mu) * 4, 4)] if src == "v" else Mn[np.arange(ci, len(men) * 4, 4)]
    mm = mu if src == "v" else men
    kk = (mm.age.values > 0.75) & (mm[col].values >= minc)
    return Aa[kk] / np.maximum(Aa[kk].sum(1, keepdims=True), 1) * 1e4, mm.age.values[kk], int(kk.sum())
for nm, (ci, src, col, genes) in {
        "VLMC": (1, "v", "n_VLMC", ["AKR1C1", "AKR1C2", "AKR1C3", "CYP1B1", "PGR"]),
        "Glut": (0, "n", "n_Glut", ["ZNF385B", "PGR"]),
        "GABA": (1, "n", "n_GABA", ["ZNF385B", "PGR"])}.items():
    CPx, agx, nx = panel(ci, src, col, 10 if src == "v" else 30)
    add("panel", f"{nm}_n_donor", nx)
    for g in genes:
        if g not in gi: continue
        v = [CPx[(agx >= lo) & (agx < hi), gi[g]].mean() for lo, hi in BANDS]
        add("panel", f"{nm}_{g}", json.dumps(dict(fold=round(float((v[-1]+1e-4)/(v[0]+1e-4)), 2),
            monotonic=bool(all(v[j] <= v[j+1] + 1e-9 for j in range(4))),
            bands=[round(float(x), 4) for x in v])))

# ---------------------------------------------------------------- 5. single cell + region
print("\n[5] single cell and region")
with h5py.File(f"{D}/devbrain.h5ad", "r") as f:
    def cat(n):
        g = f["obs"][n]
        if isinstance(g, h5py.Group):
            c = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
            return np.array(c)[g["codes"][:]]
        return np.array([x.decode() if isinstance(x, bytes) else x for x in g[:]])
    cc, fa, reg, dn = cat("cellclass"), cat("final_annotation"), cat("region"), cat("donor_id")
    age = f["obs"]["age_years_postgestation"][:]
    sel = np.where((cc == "Vascular") & (fa == "Endo") & (age > 0.75))[0]
    gs = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["var"]["gene_symbol"][:]])
    GI = {}
    for i, g in enumerate(gs):
        GI.setdefault(g, i)
    AMB = ["SNAP25", "SYT1", "RBFOX3", "MEG3", "PLP1", "MBP", "MOBP", "GFAP", "AQP4", "SLC1A2"]
    gidx = {g: GI[g] for g in ["PGR"] + AMB if g in GI}
    cols = np.array(sorted(gidx.values())); c2g = {v: kk for kk, v in gidx.items()}
    idxs, data, ptr = f["X"]["indices"], f["X"]["data"], f["X"]["indptr"][:]
    out = collections.defaultdict(lambda: np.zeros(len(sel))); tot = np.zeros(len(sel))
    for j, r in enumerate(sel):
        s, e = ptr[r], ptr[r + 1]
        ii, dd = idxs[s:e], data[s:e]
        tot[j] = dd.sum()
        hit = np.isin(ii, cols)
        for c, v in zip(ii[hit], dd[hit]):
            out[c2g[c]][j] = v
pgr = out["PGR"]; ages = age[sel]; regs = reg[sel]; dons = dn[sel]
amb = sum(out[g] for g in AMB if g in out) / np.maximum(tot, 1)
add("singlecell", "n_cell", len(sel))
frac = []
for (lo, hi), lab in zip(BANDS, BLAB):
    j = (ages >= lo) & (ages < hi)
    frac.append(round(float(100 * (pgr[j] > 0).mean()), 2))
    add("singlecell", f"pct_pos_{lab}", f"{frac[-1]} (n_cell={int(j.sum())}, median_counts={int(np.median(tot[j]))})")
med = np.median(amb)
for lab, j0 in [("low_ambient", amb <= med), ("high_ambient", amb > med)]:
    v = [round(float(100 * (pgr[(ages >= lo) & (ages < hi) & j0] > 0).mean()), 2) for lo, hi in BANDS]
    add("singlecell", f"pct_pos_{lab}", json.dumps(v))
rr = []
for d in set(dons):
    j = dons == d
    rr.append(dict(donor=d, age=float(ages[j][0]), region=collections.Counter(regs[j]).most_common(1)[0][0],
                   n=int(j.sum()), pgr=float(pgr[j].sum() / max(tot[j].sum(), 1) * 1e4)))
DD = pd.DataFrame(rr); DD = DD[DD.n >= 10]
add("region", "n_donor", len(DD))
add("region", "rho_raw", round(float(spearmanr(DD.age, np.log1p(DD.pgr))[0]), 3))
for r_ in sorted(set(DD.region)):
    s = DD[DD.region == r_]
    if len(s) >= 6:
        add("region", f"rho_{r_}", f"{spearmanr(s.age, np.log1p(s.pgr))[0]:+.3f} (n={len(s)})")
Xr = pd.get_dummies(DD.region, drop_first=True).values.astype(float)
Ar = np.column_stack([np.ones(len(DD)), Xr - Xr.mean(0)])
rr_ = lambda v: v - Ar @ np.linalg.lstsq(Ar, v, rcond=None)[0]
add("region", "rho_after_region", round(float(spearmanr(rr_(DD.age.values.astype(float)), rr_(np.log1p(DD.pgr.values)))[0]), 3))

# ---------------------------------------------------------------- 6. GTEx
print("\n[6] GTEx v8 median TPM")
gp = f"{D}/gtex_med.gct.gz"
if not os.path.exists(gp):
    raise FileNotFoundError(
        f"GTEx median table missing: {gp}\n"
        "download: curl -o gtex_med.gct.gz https://storage.googleapis.com/adult-gtex/bulk-gex/"
        "v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz")
if True:
    with gzip.open(gp, "rt") as f:
        f.readline(); f.readline()
        hdr = f.readline().rstrip("\n").split("\t")
        bi = [i - 2 for i, h in enumerate(hdr) if i >= 2 and h.startswith("Brain")]
        ai = [i - 2 for i, h in enumerate(hdr) if i >= 2 and "Adrenal" in h]
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[1] in {"HSD3B1", "HSD3B2", "HSD3B7", "CYP11A1", "STAR", "CYP19A1"}:
                v = [float(x) for x in p[2:]]
                add("gtex", p[1], f"adrenal={v[ai[0]]:.2f}; brain_median={stat.median([v[i] for i in bi]):.3f}; max={hdr[2+max(range(len(v)), key=lambda i: v[i])]}")

# ---------------------------------------------------------------- 7. Allen cluster level
print("\n[7] Allen whole brain atlas, cluster level")
HERE = os.path.dirname(os.path.abspath(__file__))
al = pd.read_csv(os.path.join(HERE, "allen_fig.csv"))
add("allen", "n_cluster_ge50", len(al))
add("allen", "n_neuronal_cluster", f"{int((al.cls=='neuronal').sum())}; above_0.5={int(((al.cls=='neuronal')&(al.PGR>0.5)).sum())}")
add("allen", "n_vascular_cluster", int((al.cls == "vascular").sum()))
ctx = al[al.division == "Cerebral cortex"]
for cls in ["neuronal", "vascular"]:
    v = ctx[ctx.cls == cls].PGR.values
    add("allen", f"cortex_{cls}", f"n={len(v)}; max={v.max():.4f}; median={np.median(v):.4f}; above_0.5={int((v>0.5).sum())}")
neu = al[(al.cls == "neuronal") & (al.PGR > 0.5)]
add("allen", "neuronal_above_0.5", f"n={len(neu)}; " + "; ".join(f"{k}={v}" for k, v in collections.Counter(neu.division).most_common()))
for g in ["HSD3B1", "HSD3B2", "HSD3B7", "CYP11A1", "STAR", "CYP19A1"]:
    add("allen", g, f"detected={int((al[g]>0.1).sum())}; above_0.5={int((al[g]>0.5).sum())}; max={al[g].max():.4f}")

# ---------------------------------------------------------------- 8. PsychAD four cell types
print("\n[8] PsychAD adult cortex, four vascular cell types")
ps = pd.read_csv(os.path.join(HERE, "psychad_fig.csv")).set_index("gene")
add("psychad", "n_donor", "Endo=934; PC=833; SMC=134; VLMC=675")
for g in ps.index:
    r = ps.loc[g]
    add("psychad", g, "; ".join(f"{c}={r[c]:.4f}" for c in ["Endo", "PC", "SMC", "VLMC"]) + f"; peak={r.idxmax()}")

# ---------------------------------------------------------------- 9. partial correlation of PGR and ZNF385B in adults
print("\n[9] partial correlation in adult cohorts, age removed")
pdn = pd.read_csv(os.path.join(HERE, "psychad_donor.csv"))
for ct in ["Endo", "PC", "SMC"]:
    per = {}
    for coh, sub in pdn[pdn.celltype == ct].groupby("cohort"):
        per[coh] = round(partial(np.log1p(sub.PGR.values), np.log1p(sub.ZNF385B.values), sub.age.values), 3)
    if per:
        add("psychad_partial", ct, f"mean={np.mean(list(per.values())):+.3f}; " +
            "; ".join(f"{k}={v:+.3f}" for k, v in sorted(per.items())) +
            f"; all_positive={all(v > 0 for v in per.values())}")

# ---------------------------------------------------------------- 10. barrier panel
print("\n[10] barrier gene panel, from birth")
PANEL = {"tight_junction": ["CLDN5", "OCLN", "TJP1", "TJP2", "MARVELD2"],
         "transport": ["ABCB1", "ABCG2", "SLC2A1", "MFSD2A", "SLCO1A2"],
         "receptor": ["PGR"]}
early = {}
for grp, gg in PANEL.items():
    for g in gg:
        if g not in symi: continue
        v = [float(CPf[(agf >= lo) & (agf < hi), symi[g]].mean()) for lo, hi in FB]
        e = (v[2] - v[0]) / (max(v) - v[0]) if max(v) > v[0] else float("nan")
        early[g] = e
        add("barrier_panel", g, json.dumps(dict(group=grp, bands=[round(x, 3) for x in v],
            early_fraction_by_2y=round(float(e), 3), peak_band=FLAB[int(np.argmax(v))])))
bb = [v for k, v in early.items() if k != "PGR" and np.isfinite(v)]
add("barrier_panel", "SUMMARY", f"n={len(bb)}; median_early={np.median(bb):.2f}; "
    f"range={min(bb):.2f} to {max(bb):.2f}; PGR={early['PGR']:.2f}")

# ---------------------------------------------------------------- 11. PGR+ fraction at donor level
print("\n[11] PGR+ nucleus fraction, donor level")
from scipy.stats import beta as _beta
dc = pd.read_csv(os.path.join(HERE, "donor_pgr_counts.csv"))
dc["p"] = dc.k / dc.n
add("donor_sc", "n_donor", f"{len(dc)}; nuclei={int(dc.n.sum())}; positive={int(dc.k.sum())}")
r_, p_ = spearmanr(dc.age, dc.p)
add("donor_sc", "rho_age_proportion", f"{r_:+.3f} (p={p_:.2e}, donor level)")
for s_, g_ in dc.groupby("study"):
    if len(g_) >= 6:
        rr, pp = spearmanr(g_.age, g_.p)
        add("donor_sc", f"rho_{s_}", f"{rr:+.3f} (n={len(g_)}, p={pp:.3f})")
for (lo, hi), lab in zip(BANDS, BLAB):
    g_ = dc[(dc.age >= lo) & (dc.age < hi)]
    kk, nn_ = int(g_.k.sum()), int(g_.n.sum())
    lo_ci = _beta.ppf(.025, kk, nn_ - kk + 1) if kk > 0 else 0.0
    hi_ci = _beta.ppf(.975, kk + 1, nn_ - kk) if kk < nn_ else 1.0
    add("donor_sc", f"band_{lab}", f"donors={len(g_)}; donors_with_any_positive={int((g_.p>0).sum())}; "
        f"nuclei={nn_}; positive={kk}; pct={100*kk/nn_:.2f}; CI95=[{100*lo_ci:.2f},{100*hi_ci:.2f}]")

# ---------------------------------------------------------------- 12. permutation distribution
print("\n[12] permutation test for the three-criteria screen")
from scipy.stats import rankdata as _rd
Zr = np.apply_along_axis(_rd, 0, Z); Zr = (Zr - Zr.mean(0)) / np.maximum(Zr.std(0), 1e-9)
def _gate(a):
    ar = _rd(a); ar = (ar - ar.mean()) / ar.std()
    Rr = (ar @ Zr) / len(a)
    Mm = np.array([X[(a >= lo) & (a < hi)].mean(0) for lo, hi in BANDS])
    fo = (Mm[-1] + 1e-4) / (Mm[0] + 1e-4)
    mo = np.all(np.diff(Mm, axis=0) >= -1e-9, axis=0)
    ze = (X[a < 2] == 0).mean(0) * 100
    gsel = (fo >= 5) & (ze >= 50) & mo
    return int(gsel.sum()), (float(Rr[gsel].max()) if gsel.sum() else -1.0)
n0, r0 = _gate(ag)
NP = 2000; rngp = np.random.default_rng(2026); cg = cr = 0; mx = []
for _ in range(NP):
    aa = ag.copy()
    for s_ in sorted(set(st)):
        j_ = np.where(st == s_)[0]; aa[j_] = ag[rngp.permutation(j_)]
    ng, rg = _gate(aa); mx.append(rg); cg += ng >= n0; cr += rg >= r0
mx = np.array(mx)
add("permutation", "design", f"age shuffled within study, n={NP}, seed=2026")
add("permutation", "n_gate_genes", f"observed={n0}; p={(cg+1)/(NP+1):.4f}")
add("permutation", "max_rho_among_gate", f"observed={r0:.3f}; p={(cr+1)/(NP+1):.4f}; "
    f"null_median={np.median(mx):.3f}; null_p95={np.quantile(mx,.95):.3f}; null_max={mx.max():.3f}")

# ---------------------------------------------------------------- 13. PsychAD, unaffected donors only
print("\n[13] PsychAD four cell types, unaffected donors only")
pc = pd.read_csv(os.path.join(HERE, "psychad_control_fig.csv"), comment="#").set_index("gene")
add("psychad_control", "n_donor", "Endo=433; PC=383; SMC=44; VLMC=249 (disease_ontology_term_id = PATO:0000461)")
agree = 0
for g in pc.index:
    r_ = pc.loc[g]
    same = r_.idxmax() == ps.loc[g].idxmax()
    agree += same
    add("psychad_control", g, "; ".join(f"{c}={r_[c]:.4f}" for c in ["Endo", "PC", "SMC", "VLMC"])
        + f"; peak={r_.idxmax()}; same_peak_as_full_set={same}")
add("psychad_control", "SUMMARY", f"{agree}/{len(pc.index)} genes keep the same peak cell type")

# ---------------------------------------------------------------- 14. diagnosis test
print("\n[14] schizophrenia and bipolar disorder, endothelial PGR")
from scipy.stats import rankdata as _rk
dx = pd.read_csv(os.path.join(HERE, "psychad_diagnosis.csv"))
def _res(y, C):
    Am = np.column_stack([np.ones(len(y))] + [c - np.mean(c) for c in C])
    return y - Am @ np.linalg.lstsq(Am, y, rcond=None)[0]
def _ci(r, n_):
    z = np.arctanh(r); s = 1 / np.sqrt(n_ - 3)
    return np.tanh(z - 1.96 * s), np.tanh(z + 1.96 * s)
HK = ["ACTB", "GAPDH", "RPL13A", "RPLP0", "TBP", "PGK1"]
add("diagnosis", "n_donor", f"{len(dx)}; cohorts=" + "; ".join(f"{k}={v}" for k, v in dx.cohort.value_counts().items()))
# positive control: sex
sub = dx
cs_ = sorted(set(sub.cohort)); Dm = [np.array(sub.cohort == c, float) for c in cs_[1:]]
Cx = [_rk(sub.age.values)] + Dm
gsex = _res(np.array(sub.sex == "female", float), Cx)
for g in ["XIST", "UTY", "PGR", "ACTB"]:
    r_, p_ = spearmanr(gsex, _res(_rk(np.log1p(sub[g].values)), Cx))
    add("diagnosis_poscontrol", f"sex_{g}", f"rho={r_:+.3f}; p={p_:.2e}")
for lab, pos_c in [("SCZ", "scz"), ("BD", "bip")]:
    pos = (sub[pos_c] == "Yes").values
    neg = ((sub.scz == "No") & (sub.bip == "No")).values
    k_ = pos | neg; s2 = sub[k_]
    cs2 = sorted(set(s2.cohort)); D2 = [np.array(s2.cohort == c, float) for c in cs2[1:]]
    C2 = [_rk(s2.age.values), np.array(s2.sex == "female", float)] + D2
    gv = _res(pos[k_].astype(float), C2); n_ = int(k_.sum())
    add("diagnosis", f"{lab}_n", f"cases={int(pos.sum())}; controls={int(neg.sum())}; total={n_}; "
        f"case_cohorts=" + "; ".join(f"{k}={v}" for k, v in s2[pos[k_]].cohort.value_counts().items()))
    for g in ["PGR", "PGR-AS1", "PAQR5", "ZNF385B"] + HK:
        r_, p_ = spearmanr(gv, _res(_rk(np.log1p(s2[g].values)), C2))
        lo_, hi_ = _ci(r_, n_)
        add("diagnosis", f"{lab}_{g}", f"rho={r_:+.3f}; CI95=[{lo_:+.3f},{hi_:+.3f}]; p={p_:.3f}"
            + ("; housekeeping" if g in HK else ""))
    ypgr = _res(_rk(np.log1p(s2["PGR"].values)), C2)
    for iv, nm_ in [(C2[0], "age"), (C2[1], "sex")]:
        it = _res((pos[k_].astype(float) - pos[k_].mean()) * (iv - iv.mean()), C2 + [pos[k_].astype(float)])
        r_, p_ = spearmanr(it, ypgr)
        add("diagnosis", f"{lab}_interaction_{nm_}", f"rho={r_:+.3f}; p={p_:.3f}")
    for c_ in cs2:
        j_ = (s2.cohort == c_).values
        if (pos[k_] & j_).sum() >= 10:
            r_ = spearmanr(pos[k_][j_].astype(float), np.log1p(s2["PGR"].values[j_]))[0]
            add("diagnosis", f"{lab}_within_{c_}", f"rho={r_:+.3f}; n={int(j_.sum())}")
    for r0 in [0.05, 0.10, 0.15, 0.20]:
        z = np.arctanh(r0) * np.sqrt(n_ - 3)
        from scipy.stats import norm as _nm
        add("diagnosis_power", f"{lab}_rho{r0}", f"power={_nm.cdf(z-1.96)+_nm.cdf(-z-1.96):.2f}")

# ---------------------------------------------------------------- 15. independent replication
print("\n[15] Steyn temporal cortex cohort, independent replication")
from scipy.stats import fisher_exact as _fe, beta as _bta
ct = pd.read_csv(os.path.join(HERE, "steyn_celltype_pgr.csv")).sort_values("PGR", ascending=False)
add("steyn", "n_celltype", f"{len(ct)}; nuclei={int(ct.n_nuclei.sum())}")
for _, r in ct.iterrows():
    add("steyn_celltype", r.cell_type, f"PGR={r.PGR:.4f}; pct_positive={r.pct_PGR_positive:.2f}; "
        f"CLDN5={r.CLDN5:.4f}; n={int(r.n_nuclei)}")
sd = pd.read_csv(os.path.join(HERE, "steyn_endo_pseudobulk.csv")).sort_values("age")
sd["cp"] = sd.PGR / sd.total_counts * 1e4
sd["frac"] = 100 * sd.n_pgr_pos / sd.n_nuclei
sd["depth"] = sd.total_counts / sd.n_nuclei
add("steyn", "n_donor", f"{len(sd)}; age={sd.age.min()}-{sd.age.max()}; "
    f"nuclei={int(sd.n_nuclei.sum())}; PGR_positive={int(sd.n_pgr_pos.sum())}")
SHK = ["ACTB", "GAPDH", "RPL13A", "RPLP0", "TBP", "PGK1"]
Cs = [rankdata(sd.depth.values), rankdata(sd.n_nuclei.values)]
def _rs(y, C):
    Am = np.column_stack([np.ones(len(y))] + [c - np.mean(c) for c in C])
    return y - Am @ np.linalg.lstsq(Am, y, rcond=None)[0]
for lab, y in [("expression", sd.cp.values), ("positive_fraction", sd.frac.values)]:
    r0, p0 = spearmanr(sd.age.values, y)
    r1, p1 = spearmanr(_rs(rankdata(sd.age.values), Cs), _rs(rankdata(y), Cs))
    add("steyn", f"age_{lab}", f"rho={r0:+.3f}; p={p0:.4f}; adjusted_rho={r1:+.3f}; adjusted_p={p1:.4f}")
for g in SHK:
    r0, p0 = spearmanr(sd.age.values, sd[g] / sd.total_counts * 1e4)
    rr, pr = spearmanr(sd.age.values, sd.PGR / np.maximum(sd[g], 1))
    add("steyn_control", g, f"age_rho={r0:+.3f}; p={p0:.3f}; PGR_over_{g}_rho={rr:+.3f}; p={pr:.4f}")
ped = sd[sd.age <= 9]; adu = sd[sd.age >= 20]
k1, n1 = int(ped.n_pgr_pos.sum()), int(ped.n_nuclei.sum())
k2, n2 = int(adu.n_pgr_pos.sum()), int(adu.n_nuclei.sum())
_ci = lambda k, n: (100 * (_bta.ppf(.025, k, n - k + 1) if k else 0), 100 * _bta.ppf(.975, k + 1, n - k))
add("steyn", "paediatric_4to9", f"donors={len(ped)}; {k1}/{n1}={100*k1/n1:.2f}%; "
    f"CI95=[{_ci(k1,n1)[0]:.2f},{_ci(k1,n1)[1]:.2f}]")
add("steyn", "adult_20plus", f"donors={len(adu)}; {k2}/{n2}={100*k2/n2:.2f}%; "
    f"CI95=[{_ci(k2,n2)[0]:.2f},{_ci(k2,n2)[1]:.2f}]; fisher_p={_fe([[k1,n1-k1],[k2,n2-k2]])[1]:.2e}")
add("steyn", "depth_ratio", f"paediatric={ped.depth.median():.0f}; adult={adu.depth.median():.0f}; "
    f"ratio={adu.depth.median()/ped.depth.median():.2f}")
for g in ["PGR"] + SHK + ["NR3C1", "NR3C2", "ESR1", "AR", "GPER1", "CLDN5", "PECAM1"]:
    if g not in sd.columns: continue
    a_ = (ped[g] / ped.total_counts * 1e4).mean(); b_ = (adu[g] / adu.total_counts * 1e4).mean()
    add("steyn_fold", g, f"adult_over_paediatric={b_/max(a_,1e-9):.2f}")

pd.DataFrame(rows).to_csv(os.path.join(A.out, "source_numbers_v2.tsv"), sep="\t", index=False)
print(f"\nwritten: {os.path.join(A.out, 'source_numbers_v2.tsv')}  ({len(rows)} rows)")
