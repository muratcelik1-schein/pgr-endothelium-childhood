# -*- coding: utf-8 -*-
"""Where the target sits among genes of its own abundance, and what the top of
that ranking is made of.

A correlation with age says a gene moves. It does not say whether moving is
unusual for a gene of that abundance, because low abundance genes have compressed
fold changes and high abundance genes have room to move. This script asks the
question the correlation cannot: inside its own abundance class, how far does the
target move, and which genes sit beside it.

Three quantities, one table.

  1  MATCHED PERCENTILE. Every gene is placed in one of twenty abundance bins
     inside the cell class, and its postnatal amplitude is ranked against the
     genes of that bin alone. A housekeeping panel is carried through as the
     negative control and must land near the middle, or the measure is broken and
     nothing is reported.

  2  WHAT THE TOP IS MADE OF. Each of the twenty bins has one gene at its top.
     Those twenty are listed with the direction and size of their amplitude, so
     that the company the target keeps is on record rather than asserted.

  3  LEAVE ONE STUDY OUT. The atlas pools four contributing studies for this cell
     class and one of them is dominant. The percentile is recomputed with each
     removed, and the negative control is re-checked in every run.

Analysis unit is the donor by cell class. Amplitude is the difference between the
last two and the first two of eight log spaced age bins, on log counts per 10,000,
so a change in how many cells were sampled does not enter it.

Run: python3 amplitude_rank.py --pb pseudobulk.npz --groups groups.tsv --genes genes.tsv --out amplitude_rank.tsv
"""
import argparse, os, sys
import numpy as np

SEED = 2026
BIRTH = 0.77
MERGE = {"Chandelier": "Pvalb"}
HOUSE = ["ACTB", "GAPDH", "TUBB", "RPL13A", "RPLP0", "PPIA", "TBP", "UBC", "SDHA", "YWHAZ",
         "HPRT1", "PGK1", "RPL27", "RPS18", "EEF1A1", "PSMB2", "VPS29", "CHMP2A", "EMC7",
         "GPI", "REEP5", "SNRPD3", "VCP"]
REPORT = ["PGR", "PGR-AS1", "AKR1C1", "AKR1C3", "PGRMC1", "PAQR5", "VWF", "FLT1",
          "PECAM1", "CLDN5", "ABCB1", "NR3C1", "NR3C2", "ESR1", "AR", "GPER1"]
GATE = ("PGR", 0.7317, 100.0)          # production gate, tolerance below
CLASSES = ["Endo", "VLMC", "Astro", "Micro/PVM", "L2/3 IT", "Oligo", "OPC", "Pvalb"]

ap = argparse.ArgumentParser()
ap.add_argument("--pb", required=True)
ap.add_argument("--groups", required=True)
ap.add_argument("--genes", required=True)
ap.add_argument("--min-cells", type=int, default=20)
ap.add_argument("--bins", type=int, default=8)
ap.add_argument("--abins", type=int, default=20)
ap.add_argument("--out", required=True)
A = ap.parse_args()

rows = []
def add(b, k, v, n=""):
    rows.append((b, k, v, n)); print(f"{b:16s} {k:26s} {v}   {n}", flush=True)
def flush():
    d = os.path.dirname(A.out)
    if d: os.makedirs(d, exist_ok=True)
    with open(A.out, "w", encoding="utf-8") as fh:
        fh.write("block\tkey\tvalue\tnote\n")
        for r in rows: fh.write("\t".join(str(x) for x in r) + "\n")

z = np.load(A.pb)
sums = z["sums"].astype(np.float64); ncells = z["n_cells"]; total = z["total"]
genes = [l.rstrip("\n") for l in open(A.genes)]
gi = {}
for i, g in enumerate(genes): gi.setdefault(g, i)
meta = []
with open(A.groups) as fh:
    head = fh.readline().rstrip("\n").split("\t")
    for line in fh: meta.append(dict(zip(head, line.rstrip("\n").split("\t"))))
ct = np.array([MERGE.get(m["celltype"], m["celltype"]) for m in meta])
age_all = np.maximum(np.array([float(m["age_years"]) for m in meta]) - BIRTH, 1 / 365.0)
study = np.array([m["study"] for m in meta])
ok = ncells >= A.min_cells
CP = np.zeros_like(sums); nz = total > 0
CP[nz] = sums[nz] / total[nz][:, None] * 1e4
L = np.log2(CP + 1.0)
add("meta", "seed", SEED)
add("meta", "groups_kept", int(ok.sum()), f"of {len(meta)}, at least {A.min_cells} cells")
add("meta", "amplitude", f"{A.bins} log spaced age bins, last two minus first two",
    "on log2 counts per 10,000, postnatal age")

def amplitude(idx):
    a = age_all[idx]
    la = np.log10(a); ed = np.quantile(la, np.linspace(0, 1, A.bins + 1)); ed[-1] += 1e-9
    b = np.clip(np.digitize(la, ed) - 1, 0, A.bins - 1)
    if len(set(b.tolist())) < A.bins: return None, None
    Lm = L[idx].T
    C = np.empty((Lm.shape[0], A.bins))
    for k in range(A.bins): C[:, k] = np.median(Lm[:, b == k], axis=1)
    return C[:, -2:].mean(axis=1) - C[:, :2].mean(axis=1), CP[idx].mean(axis=0)

def matched(fold, mean_ab):
    expressed = np.where(mean_ab > 0.01)[0]
    order = expressed[np.argsort(mean_ab[expressed])]
    abins = np.array_split(order, A.abins)
    bin_of = {g: k for k, bb in enumerate(abins) for g in bb}
    af = np.abs(fold)
    def pct(i):
        k = bin_of.get(i)
        if k is None: return None, 0
        pool = abins[k]
        return float((af[pool] <= af[i]).mean() * 100), len(pool)
    return pct, abins, bin_of, len(expressed)

# ---------------------------------------------------------------- 1 and 2
top20 = None
for cls in CLASSES:
    m = np.where(ok & (ct == cls))[0]
    if len(m) < 20:
        add("class", cls, "skipped", f"only {len(m)} groups"); continue
    fold, mean_ab = amplitude(m)
    if fold is None:
        add("class", cls, "skipped", "an age bin was empty"); continue
    pct, abins, bin_of, nexp = matched(fold, mean_ab)
    hk = [gi[g] for g in HOUSE if g in gi and gi[g] in bin_of]
    hk_med = float(np.median([pct(i)[0] for i in hk]))
    good = 30 <= hk_med <= 70
    add("negative_control", cls, f"housekeeping median percentile {hk_med:.1f}",
        "must lie between 30 and 70, or the class is not interpreted"
        + ("" if good else "  BROKEN, NOT INTERPRETED"))
    if not good: continue
    for g in REPORT:
        i = gi.get(g)
        if i is None or i not in bin_of:
            add(cls, g, "below the expression threshold"); continue
        p, n = pct(i)
        add(cls, g, f"log2 {fold[i]:+.4f}; percentile {p:.2f}", f"abundance bin holds {n} genes")
    if cls == "Endo":
        i = gi[GATE[0]]; p, _ = pct(i)
        okg = abs(fold[i] - GATE[1]) <= 0.01 and abs(p - GATE[2]) <= 0.5
        add("gate", "production", "PASS" if okg else "FAIL",
            f"PGR log2 {fold[i]:.4f} (expected {GATE[1]}), percentile {p:.1f} (expected {GATE[2]})")
        if not okg:
            add("VERDICT", "SCAN INVALID", "the production gate did not reproduce"); flush(); sys.exit(2)
        # what the top of the ranking is made of
        af = np.abs(fold)
        top20 = []
        for k, bb in enumerate(abins):
            j = bb[int(np.argmax(af[bb]))]
            top20.append((genes[j], float(fold[j]), float(mean_ab[j]), len(bb)))
        up = [t for t in top20 if t[1] > 0]; dn = [t for t in top20 if t[1] <= 0]
        add("top_of_each_bin", "n_bins", len(top20), "one gene tops each abundance bin")
        add("top_of_each_bin", "rising", len(up))
        add("top_of_each_bin", "falling", len(dn))
        for g, f_, ab, nb in sorted(top20, key=lambda x: -x[1]):
            add("top_of_each_bin", g, f"log2 {f_:+.4f}", f"mean {ab:.4f} counts per 10,000, bin of {nb}")

# ---------------------------------------------------------------- 3 leave one study out
add("loso", "studies", "; ".join(f"{s}={int(((ct=='Endo') & ok & (study==s)).sum())}"
                                 for s in sorted(set(study))), "endothelial donor groups per study")
for drop in [None] + sorted(set(study)):
    m = np.where(ok & (ct == "Endo") & ((study != drop) if drop else True))[0]
    fold, mean_ab = amplitude(m)
    if fold is None:
        add("loso", f"without_{drop}", "not run", "an age bin emptied"); continue
    pct, _, bin_of, _ = matched(fold, mean_ab)
    hk = [gi[g] for g in HOUSE if g in gi and gi[g] in bin_of]
    hk_med = float(np.median([pct(i)[0] for i in hk]))
    i = gi["PGR"]; p, _ = pct(i)
    add("loso", "all studies" if drop is None else f"without {drop}",
        f"PGR log2 {fold[i]:+.4f}; percentile {p:.2f}",
        f"{len(m)} groups; housekeeping median {hk_med:.1f}"
        + ("" if 30 <= hk_med <= 70 else "  CONTROL BROKEN, read with caution"))

add("limits", "what_this_is_not",
    "A percentile inside an abundance bin is a statement about amplitude, not about timing, "
    "and not about function. Twenty genes top the twenty bins, so leading one bin is not "
    "leading the transcriptome.")
flush()
print(f"\nwritten: {A.out} | {len(rows)} rows")
