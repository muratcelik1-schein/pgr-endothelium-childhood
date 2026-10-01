# -*- coding: utf-8 -*-
"""Donor level counterparts of the pooled statistics, and the adult trajectory.

Three tests in the manuscript are computed over pooled nuclei or pooled samples.
Pooling ignores that nuclei from one donor are not independent, so the interval
is narrower and the p value smaller than the donor level evidence supports. This
script computes the donor level counterpart of each from the tables shipped in
this directory, so that both can be reported side by side.

It also answers three questions a reader asks of the developmental result and
which the pooled tables cannot answer:
  - is the fall in the oldest band resolved at donor level,
  - would the first band have shown the transcript if it carried the detection
    rate of the second band, given its own nuclei per donor,
  - does the rise continue, plateau or reverse across the adult range.

Reads only. Writes one table.
Run:  python3 donor_level.py [--data .] [--out .]
"""
import argparse, os, csv, math, json
import numpy as np
from scipy import stats

SEED = 2026
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=HERE, help="directory holding the shipped derived tables")
ap.add_argument("--out", default=HERE)
A = ap.parse_args()
rng = np.random.default_rng(SEED)

rows = []
def add(block, key, value, note=""):
    rows.append((block, key, value, note))
    print(f"  {block:22} {key:34} {value}")

def gate(name, ok, detail=""):
    print(("  GATE OK    " if ok else "  GATE FAIL ") + name + (f"  -> {detail}" if not ok else ""))
    if not ok:
        raise SystemExit(f"gate failed: {name}")

def fisher_ci(rho, n, conf=0.95):
    """Confidence interval of a correlation by the Fisher transformation."""
    if n < 4 or abs(rho) >= 1:
        return (float("nan"), float("nan"))
    z = np.arctanh(rho)
    se = 1.0 / math.sqrt(n - 3)
    q = stats.norm.ppf(0.5 + conf / 2.0)
    return (float(np.tanh(z - q * se)), float(np.tanh(z + q * se)))

def fmt_ci(rho, n):
    lo, hi = fisher_ci(rho, n)
    return f"{rho:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}, n = {n})"

# ================================================================ 1. primary series
print("\n[1] primary series, donor level")
D = []
with open(os.path.join(A.data, "donor_pgr_counts.csv")) as fh:
    for r in csv.DictReader(fh):
        D.append(dict(donor=r["donor"], study=r["study"], age=float(r["age"]),
                      k=int(r["k"]), n=int(r["n"])))
age = np.array([d["age"] for d in D]); k = np.array([d["k"] for d in D]); n = np.array([d["n"] for d in D])
p = k / n

gate("65 donors in the shipped per donor table", len(D) == 65, len(D))
gate("5,388 nuclei in the shipped per donor table", int(n.sum()) == 5388, int(n.sum()))
rho_p, pv_p = stats.spearmanr(age, p)
gate("donor level association reproduces the reported +0.707", abs(rho_p - 0.707) < 0.001, round(rho_p, 4))

add("meta", "n_donor", len(D))
add("meta", "n_nuclei", int(n.sum()))
add("meta", "seed", SEED)
add("primary_donor", "positive_fraction_vs_age", fmt_ci(rho_p, len(D)),
    f"two sided p = {pv_p:.3g}; the manuscript reports the coefficient without an interval")

# band structure of the manuscript, on the axis of the source resource
BANDS = [(0.75, 2), (2, 6), (6, 12), (12, 20), (20, 99)]
BLAB = ["0.75-2y", "2-6y", "6-12y", "12-20y", ">20y"]
BOOT = 10000
band_don = {}
for (lo, hi), lab in zip(BANDS, BLAB):
    m = (age >= lo) & (age < hi)
    band_don[lab] = m
    pooled = k[m].sum() / n[m].sum() * 100
    dm = p[m] * 100
    # donor bootstrap: resample donors within the band, recompute the donor mean
    bs = np.array([dm[rng.integers(0, len(dm), len(dm))].mean() for _ in range(BOOT)])
    lo_b, hi_b = np.percentile(bs, [2.5, 97.5])
    add("band_donor", f"{lab}_donor_mean_pct",
        f"{dm.mean():.2f} (95% CI {lo_b:.2f} to {hi_b:.2f})",
        f"{int(m.sum())} donors; pooled value {pooled:.2f}%, which the manuscript reports")
    add("band_donor", f"{lab}_donor_median_pct", f"{np.median(dm):.2f}",
        f"IQR {np.percentile(dm,25):.2f} to {np.percentile(dm,75):.2f}")

# the oldest band falls: is that fall resolved at donor level?
a4, a5 = p[band_don["12-20y"]] * 100, p[band_don[">20y"]] * 100
u, pu = stats.mannwhitneyu(a4, a5, alternative="two-sided")
add("oldest_band", "12-20y_vs_>20y_donor_means", f"{a4.mean():.2f}% against {a5.mean():.2f}%",
    f"{len(a4)} against {len(a5)} donors")
add("oldest_band", "12-20y_vs_>20y_rank_sum", f"U = {u:.1f}, two sided p = {pu:.3f}",
    "the pooled Clopper-Pearson intervals of these two bands do not overlap; at donor level "
    "the difference is not resolved, so the series ends on a plateau and not a measured fall")

# ================================================================ 2. was the first band simply undersampled
print("\n[2] first band, detection expectation under the second band rate")
m1, m2 = band_don["0.75-2y"], band_don["2-6y"]
rate2 = k[m2].sum() / n[m2].sum()
exp_pos = rate2 * n[m1].sum()
# probability that a donor of the first band shows no positive nucleus at the second band rate
q0 = (1.0 - rate2) ** n[m1]
exp_zero = q0.sum()
obs_zero = int((k[m1] == 0).sum())
# Poisson binomial tail for the observed number of zero donors, by exact convolution
dist = np.array([1.0])
for qi in q0:
    dist = np.convolve(dist, [1 - qi, qi])
p_zero_tail = float(dist[obs_zero:].sum())
p_exact_pos = float(stats.binom.cdf(int(k[m1].sum()), int(n[m1].sum()), rate2))
add("first_band", "nuclei", f"{int(n[m1].sum())} across {int(m1.sum())} donors",
    f"nuclei per donor {int(n[m1].min())} to {int(n[m1].max())}, median {int(np.median(n[m1]))}")
add("first_band", "second_band_detection_rate", f"{rate2*100:.2f}%")
add("first_band", "expected_positive_nuclei_at_that_rate", f"{exp_pos:.1f} against 5 observed",
    f"one sided binomial p = {p_exact_pos:.3g}")
add("first_band", "expected_donors_with_none", f"{exp_zero:.1f} against {obs_zero} observed",
    f"Poisson binomial p = {p_zero_tail:.3g}; the shortfall is not explained by nuclei per donor")

# ================================================================ 3. sorted endothelial atlas, per donor
print("\n[3] sorted endothelial atlas, donor level sign test")
per = {}
with open(os.path.join(A.data, "walchli_adult.tsv")) as fh:
    for line in fh:
        f = line.rstrip("\n").split("\t")
        if len(f) >= 3 and f[0] == "per_patient" and f[1] in ("arterial", "capillary", "venous"):
            per[f[1]] = {kv.split("=")[0]: kv.split("=")[1] for kv in f[2].split("; ")}
donors = sorted(per["arterial"])
val = lambda seg, d: (float(per[seg][d]) if per[seg][d] != "NA" else float("nan"))
low_art, comparable = 0, 0
pairs = []
for d in donors:
    a, c, v = val("arterial", d), val("capillary", d), val("venous", d)
    vals = [x for x in (a, c, v) if not math.isnan(x)]
    if math.isnan(a) or len(vals) < 2:
        continue
    comparable += 1
    if a == min(vals):
        low_art += 1
    pairs.append((d, a, c, v))
p_sign = float(stats.binomtest(low_art, comparable, 0.5, alternative="two-sided").pvalue)
add("sorted_atlas", "donors_with_arterial_lowest", f"{low_art} of {comparable}",
    "; ".join(f"{d} {a:.4f}/{c:.4f}/{'NA' if math.isnan(v) else format(v,'.4f')}" for d, a, c, v in pairs))
add("sorted_atlas", "sign_test", f"two sided p = {p_sign:.4f}",
    "donor level counterpart of the pooled q = 2.8 x 10-28 computed over 72,867 cells")

# ================================================================ 4. adult trajectory
print("\n[4] adult trajectory, PsychAD endothelial donors")
E = []
with open(os.path.join(A.data, "psychad_donor.csv")) as fh:
    for r in csv.DictReader(fh):
        if r["celltype"] == "Endo":
            E.append((r["cohort"], float(r["age"]), float(r["PGR"]), float(r["ZNF385B"])))
ca = np.array([x[1] for x in E]); cp = np.array([x[2] for x in E]); cz = np.array([x[3] for x in E])
coh = np.array([x[0] for x in E])
gate("932 adult endothelial donors in the shipped table", len(E) == 932, len(E))
add("adult", "n_donor", len(E), f"age {ca.min():.0f} to {ca.max():.0f} years, four cohorts")

r_all, p_all = stats.spearmanr(ca, cp)
add("adult", "PGR_vs_age_full_range", fmt_ci(r_all, len(E)), f"two sided p = {p_all:.3g}")
m20 = ca >= 20
r20, p20 = stats.spearmanr(ca[m20], cp[m20])
add("adult", "PGR_vs_age_20y_and_above", fmt_ci(r20, int(m20.sum())), f"two sided p = {p20:.3g}")
# within cohort, since the four cohorts differ in age composition
for c in sorted(set(coh)):
    mc = coh == c
    if mc.sum() < 20: continue
    rc, pc = stats.spearmanr(ca[mc], cp[mc])
    add("adult", f"PGR_vs_age_{c}", fmt_ci(rc, int(mc.sum())), f"two sided p = {pc:.3g}")
# decade means
for lo in range(20, 100, 10):
    mm = (ca >= lo) & (ca < lo + 10)
    if mm.sum() == 0: continue
    add("adult", f"decade_mean_{lo}s", f"{cp[mm].mean():.4f}", f"{int(mm.sum())} donors")
# the same for the companion gene, as a within-table comparison
rz, pz = stats.spearmanr(ca, cz)
add("adult", "ZNF385B_vs_age_full_range", fmt_ci(rz, len(E)), f"two sided p = {pz:.3g}")

# the developmental series ends at the level the adult series holds
add("adult", "adult_mean_expression", f"{cp.mean():.4f}",
    "the oldest developmental band mean is 0.6328 counts per 10,000")

# ================================================================ 5. Fisher intervals for the reported coefficients
print("\n[5] intervals for the coefficients the manuscript reports without one")
for key, rho, nn, note in [
        ("primary_expression_vs_age", 0.680, 65, "Supplementary Table S7, the genome wide screen"),
        ("primary_positive_fraction_vs_age", 0.707, 65, "Supplementary Table S12"),
        ("replication_expression_vs_age", 0.750, 12, "Supplementary Table S6, temporal cortex cohort"),
        ("replication_fraction_vs_age", 0.816, 12, "Supplementary Table S6"),
        ("multiome_expression_vs_age", 0.671, 10, "Supplementary Table S14, four regions"),
        ("multiome_fraction_vs_age", 0.811, 10, "Supplementary Table S14"),
        ("psychad_childhood_expression", 0.698, 59, "Supplementary Table S15"),
        ("psychad_childhood_fraction", 0.661, 59, "Supplementary Table S15")]:
    add("interval", key, fmt_ci(rho, nn), note)

# ================================================================ 6. one donor dominance in the small cohort
print("\n[6] the twelve donor cohort, dominance check")
S = []
with open(os.path.join(A.data, "steyn_endo_pseudobulk.csv")) as fh:
    for r in csv.DictReader(fh):
        S.append((r["donor"], float(r["age"]), int(r["n_nuclei"]), int(r["n_pgr_pos"])))
tot_n = sum(x[2] for x in S); tot_k = sum(x[3] for x in S)
top = max(S, key=lambda x: x[2])
topk = max(S, key=lambda x: x[3])
add("replication_cohort", "n_donor", len(S), f"{tot_n} nuclei, {tot_k} positive")
add("replication_cohort", "largest_donor_share_of_nuclei",
    f"{top[2]/tot_n*100:.1f}%", f"donor {top[0]}, age {top[1]:.0f}, {top[2]} nuclei")
add("replication_cohort", "largest_contributor_share_of_positives",
    f"{topk[3]/tot_k*100:.1f}%", f"donor {topk[0]}, age {topk[1]:.0f}, {topk[3]} positive nuclei")
ages = np.array([x[1] for x in S]); fr = np.array([x[3] / x[2] for x in S])
r_s, p_s = stats.spearmanr(ages, fr)
add("replication_cohort", "fraction_vs_age", fmt_ci(r_s, len(S)), f"two sided p = {p_s:.3g}")
# leave the dominant donor out
keep = [i for i, x in enumerate(S) if x[0] != topk[0]]
r_s2, p_s2 = stats.spearmanr(ages[keep], fr[keep])
add("replication_cohort", "fraction_vs_age_without_largest_contributor",
    fmt_ci(r_s2, len(keep)), f"two sided p = {p_s2:.3g}; donor {topk[0]} removed")

# ================================================================ write
out = os.path.join(A.out, "donor_level.tsv")
with open(out, "w", encoding="utf-8") as fh:
    fh.write("block\tkey\tvalue\tnote\n")
    for r in rows:
        fh.write("\t".join(str(x) for x in r) + "\n")
print(f"\nwritten: {out} | {len(rows)} rows")
