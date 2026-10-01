# -*- coding: utf-8 -*-
"""Is the uterine PR programme running in human cortical endothelium?

PROTOCOL, written before the data were touched (2026-09-14).

  Background. In uterine endothelium the progesterone receptor drives permeability
  through one named route: PR induces NR4A1, and NR4A1 represses the junctional genes
  CLDN5, PECAM1 and CDH5 [Goddard 2014, Cell, 10.1016/j.cell.2013.12.025]. That paper
  also reports PR in venous and lymphatic endothelium only, in 22.7% of venous nuclei
  per vessel, and shows that placing PR in the endothelium of an organ that lacks it
  makes that bed leak. This study finds the same receptor's transcript being acquired
  by human cortical endothelium across childhood. The question here is whether the
  downstream programme is running in that tissue.

  Prediction, stated in advance. The programme is ligand gated: NR4A1 induction in
  Goddard required progesterone, not merely the presence of the receptor. Circulating
  progesterone is least available across exactly the window in which the transcript is
  acquired. We therefore predict NO negative partial correlation between PGR and the
  three junctional targets once age is removed. A silent programme is the expected
  result and it is the result that fits the ligand timing.

  Falsification, stated in advance. If the partial correlation with any target is
  negative, survives Holm correction across the three, and lies beyond the 5th
  percentile of abundance matched genes, then the programme is partly active and the
  reading above is wrong. That outcome would be reported as such.

  Unit and model. Donor by cell class, as everywhere else in this package. Endothelium,
  donors with at least 20 nuclei, log2 counts per 10,000 plus one. Partial Spearman
  correlation removing postnatal age, by ranking both variables and age, regressing each
  on age, and correlating the residuals.

  Controls. Positive: PGR against PGR-AS1, same locus, which must be positive; and
  PGR against ZNF385B, reported in the manuscript. Negative: a housekeeping panel,
  which must sit near zero. Null: for each target, the partial correlation is placed
  against 200 genes matched on mean abundance, so that the value is read against what
  a gene of that abundance reaches.

  What this cannot do. A partial correlation across donors is not a perturbation. A
  silent programme here does not show the receptor is inert, and an active one would
  not show the receptor caused it. Ambient RNA and composition are handled elsewhere
  in this package and are not repeated here.

Run: python3 nr4a1_program.py --pb pseudobulk.npz --groups groups.tsv --genes genes.tsv --out nr4a1_program.tsv
"""
import argparse, os, sys
import numpy as np
from scipy import stats

SEED = 2026
BIRTH = 0.77
TARGETS = ["CLDN5", "PECAM1", "CDH5"]              # NR4A1 represses these in Goddard 2014
POSCTRL = ["PGR-AS1", "ZNF385B"]
HOUSE = ["ACTB", "GAPDH", "TUBB", "RPL13A", "RPLP0", "PPIA", "TBP", "UBC", "SDHA", "YWHAZ",
         "HPRT1", "PGK1", "RPL27", "RPS18", "EEF1A1", "PSMB2", "VPS29", "CHMP2A", "EMC7"]
EXTRA = ["NR4A1", "NR4A2", "NR4A3", "OCLN", "TJP1", "VWF", "FLT1", "ABCB1", "PGRMC1", "PAQR5"]
NULL_N = 200
GATE = ("PGR", 0.707, 0.02)                        # donor level age association, as shipped

ap = argparse.ArgumentParser()
ap.add_argument("--pb", required=True); ap.add_argument("--groups", required=True)
ap.add_argument("--genes", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--min-cells", type=int, default=20)
A = ap.parse_args()

rows = []
def add(b, k, v, n=""):
    rows.append((b, k, v, n)); print(f"{b:18s} {k:26s} {v}   {n}", flush=True)
def flush():
    d = os.path.dirname(A.out)
    if d: os.makedirs(d, exist_ok=True)
    with open(A.out, "w", encoding="utf-8") as fh:
        fh.write("block\tkey\tvalue\tnote\n")
        for r in rows: fh.write("\t".join(str(x) for x in r) + "\n")

rng = np.random.default_rng(SEED)
z = np.load(A.pb)
sums = z["sums"].astype(np.float64); ncells = z["n_cells"]; total = z["total"]
genes = [l.rstrip("\n") for l in open(A.genes)]
gi = {}
for i, g in enumerate(genes): gi.setdefault(g, i)
meta = []
with open(A.groups) as fh:
    head = fh.readline().rstrip("\n").split("\t")
    for line in fh: meta.append(dict(zip(head, line.rstrip("\n").split("\t"))))
ct = np.array([m["celltype"] for m in meta])
age = np.maximum(np.array([float(m["age_years"]) for m in meta]) - BIRTH, 1 / 365.0)
CP = np.zeros_like(sums); nz = total > 0
CP[nz] = sums[nz] / total[nz][:, None] * 1e4
L = np.log2(CP + 1.0)

m = np.where((ncells >= A.min_cells) & (ct == "Endo"))[0]
add("meta", "seed", SEED)
add("meta", "donors", len(m), f"endothelial groups with at least {A.min_cells} nuclei")
add("meta", "age_range", f"{age[m].min():.2f} to {age[m].max():.2f}", "postnatal years")

X = L[m]; a = age[m]
ra = stats.rankdata(a)
def partial(v):
    """Spearman partial correlation of v with PGR, removing age."""
    rv, rp = stats.rankdata(v), stats.rankdata(X[:, gi["PGR"]])
    A_ = np.c_[np.ones(len(ra)), ra]
    ev = rv - A_ @ np.linalg.lstsq(A_, rv, rcond=None)[0]
    ep = rp - A_ @ np.linalg.lstsq(A_, rp, rcond=None)[0]
    if ev.std() == 0 or ep.std() == 0: return np.nan, np.nan
    r, p = stats.pearsonr(ev, ep)
    return float(r), float(p)

# production gate: the donor level age association must reproduce
pos = (CP[m][:, gi["PGR"]] > 0).astype(float)
rho_age = stats.spearmanr(a, X[:, gi["PGR"]]).statistic
add("gate", "PGR_expression_vs_age", f"{rho_age:+.3f}", f"n = {len(m)} donors")
ok = abs(rho_age) > 0.4
add("gate", "production", "PASS" if ok else "FAIL", "the age association must be present in this subset")
if not ok:
    add("VERDICT", "INVALID", "the series did not reproduce"); flush(); sys.exit(2)

mean_ab = CP[m].mean(axis=0)
expressed = np.where(mean_ab > 0.01)[0]
def matched_null(idx, k=NULL_N):
    order = expressed[np.argsort(mean_ab[expressed])]
    j = int(np.where(order == idx)[0][0])
    lo, hi = max(0, j - 500), min(len(order), j + 500)
    pool = [q for q in order[lo:hi] if q != idx and q != gi["PGR"]]
    pick = rng.choice(pool, size=min(k, len(pool)), replace=False)
    vals = []
    for q in pick:
        r, _ = partial(X[:, q])
        if not np.isnan(r): vals.append(r)
    return np.array(vals)

add("note", "prediction", "no negative partial correlation expected",
    "the programme is ligand gated and the ligand is least available across this window")

RES = {}
for g in TARGETS + POSCTRL + EXTRA:
    i = gi.get(g)
    if i is None or mean_ab[i] <= 0.01:
        add("gene", g, "below the expression threshold", f"mean {mean_ab[i]:.4f} counts per 10,000" if i is not None else "not in the matrix")
        continue
    r, p = partial(X[:, i])
    RES[g] = (r, p)
    blk = "target" if g in TARGETS else ("positive_control" if g in POSCTRL else "other")
    add(blk, g, f"partial rho {r:+.3f}; p = {p:.4g}", f"mean {mean_ab[i]:.4f} counts per 10,000")

hk = [g for g in HOUSE if g in gi and mean_ab[gi[g]] > 0.01]
hv = [partial(X[:, gi[g]])[0] for g in hk]
add("negative_control", "housekeeping_median", f"partial rho {np.median(hv):+.3f}",
    f"{len(hk)} genes, 5th to 95th percentile {np.percentile(hv,5):+.3f} to {np.percentile(hv,95):+.3f}")

# Holm across the three targets, and the abundance matched null for each
pv = [(g, RES[g][1]) for g in TARGETS if g in RES]
for rank, (g, p) in enumerate(sorted(pv, key=lambda t: t[1])):
    add("holm", g, f"p = {p:.4g}; Holm p = {min(1.0, p * (len(pv) - rank)):.4g}",
        "corrected across the three junctional targets")
for g in TARGETS:
    if g not in RES: continue
    null = matched_null(gi[g])
    pct = float((null <= RES[g][0]).mean() * 100)
    add("matched_null", g, f"percentile {pct:.1f} of {len(null)} matched genes",
        f"null median {np.median(null):+.3f}, 5th percentile {np.percentile(null,5):+.3f}")

# Sensitivity: remove sequencing depth as well as age. CLDN5 is abundant and PGR is not,
# so a depth gradient across donors could produce a spurious negative. This is the one
# confound the age term does not already absorb.
ldep = np.log10(total[m] + 1.0)
def partial2(v):
    rv, rp = stats.rankdata(v), stats.rankdata(X[:, gi["PGR"]])
    A2 = np.c_[np.ones(len(ra)), ra, stats.rankdata(ldep), stats.rankdata(ncells[m])]
    ev = rv - A2 @ np.linalg.lstsq(A2, rv, rcond=None)[0]
    ep = rp - A2 @ np.linalg.lstsq(A2, rp, rcond=None)[0]
    if ev.std() == 0 or ep.std() == 0: return np.nan, np.nan
    r, pp = stats.pearsonr(ev, ep); return float(r), float(pp)
add("depth", "rho_depth_vs_age", f"{stats.spearmanr(a, ldep).statistic:+.3f}", "donor sequencing depth against age")
for g in TARGETS + ["NR4A1", "OCLN"]:
    if g not in RES: continue
    r2, p2 = partial2(X[:, gi[g]])
    add("depth_adjusted", g, f"partial rho {r2:+.3f}; p = {p2:.4g}",
        f"age, depth and nucleus count removed; against {RES[g][0]:+.3f} with age alone")

sil = all((g not in RES) or (RES[g][0] > -0.2) for g in TARGETS)
add("VERDICT", "programme_silent" if sil else "programme_not_silent",
    "no target reaches a partial correlation of -0.2" if sil else "at least one target is negative, see the null above",
    "the prediction stated in the protocol held" if sil else "the falsification condition was met, read the null and Holm rows")
add("limits", "what_this_cannot_do",
    "A partial correlation across donors is not a perturbation. A silent programme does not show the receptor is inert.")
flush()
print(f"\nwritten: {A.out} | {len(rows)} rows")
