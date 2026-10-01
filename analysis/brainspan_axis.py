#!/usr/bin/env python3
"""BrainSpan: a regional and developmental axis the manuscript does not use.

BrainSpan is bulk RNA-seq of 524 samples from 42 donors across 26 structures,
spanning 8 post-conceptional weeks to 40 years. It is a different assay from
the single nucleus atlases the manuscript rests on, it carries the childhood
window the manuscript is about, and it includes cerebellum.

Bulk tissue is usually a weakness for a cell type claim. Here it is the test.
If the transcript rises in bulk while the endothelial identity genes stay
flat, the rise cannot be more vessel: it has to be more transcript per
vessel. That is the manuscript's claim, measured a different way.

  Gate      A gene with a textbook postnatal trajectory must show it. MBP
            rises steeply after birth with myelination. If it does not, the
            age parsing or the matrix join is wrong and nothing is reported.

  Negative  Housekeeping genes must stay flat. If they move with age, the
            age association is technical.

  Test      PGR against age, postnatal samples only, per structure and
            pooled, with the endothelial identity genes measured alongside
            in the same samples.

  Cerebellum  CB and CBC are reported separately from the cerebrum, since
            the question was asked about both.

Writes analysis/brainspan_numbers.tsv.
"""

import csv
import os
import re
import sys

from scipy import stats

SEED = 2026  # declared for parity; this pass draws nothing

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "brainspan")
OUT = os.path.join(ROOT, "analysis", "brainspan_numbers.tsv")

TARGET = "PGR"
# Endothelial identity. If the rise were more vessel, these would rise too.
ENDOTHELIAL = ["CLDN5", "FLT1", "VWF"]
# A textbook postnatal riser. This is the gate.
POSITIVE_CONTROL = "MBP"
# Must stay flat. If these move, the association is technical.
HOUSEKEEPING = ["ACTB", "GAPDH", "TBP"]
# Measured for context, not gates.
CONTEXT = ["NR3C1", "NR3C2", "ESR1", "AR", "ABCB1", "SNAP25", "GFAP",
           "HSD3B2", "AKR1C1"]

CEREBELLAR = {"CB", "CBC"}
MIN_SAMPLES = 8          # a structure below this is not scored alone


def age_to_years(text):
    """BrainSpan age strings -> years after birth. Prenatal returns None."""
    t = text.strip().strip('"')
    m = re.match(r"^(\d+)\s*pcw$", t)
    if m:
        return None                      # prenatal, excluded by design
    m = re.match(r"^(\d+)\s*mos$", t)
    if m:
        return int(m.group(1)) / 12.0
    m = re.match(r"^(\d+)\s*yrs$", t)
    if m:
        return float(m.group(1))
    raise AssertionError(f"unparsed age string: {text!r}")


def load_rows_wanted():
    """gene symbol -> row number, for the panel only."""
    panel = set([TARGET, POSITIVE_CONTROL] + ENDOTHELIAL + HOUSEKEEPING
                + CONTEXT)
    wanted, seen = {}, {}
    with open(os.path.join(DATA, "rows_metadata.csv")) as fh:
        for r in csv.DictReader(fh):
            sym = r.get("gene_symbol")
            if sym in panel:
                seen.setdefault(sym, []).append(int(r["row_num"]))
    for sym, rn in seen.items():
        assert len(rn) == 1, f"{sym}: {len(rn)} rows, ambiguous"
        wanted[sym] = rn[0]
    missing = sorted(panel - set(wanted))
    return wanted, missing


def load_columns():
    cols = []
    with open(os.path.join(DATA, "columns_metadata.csv")) as fh:
        for r in csv.DictReader(fh):
            cols.append({
                "n": int(r["column_num"]),
                "donor": r["donor_id"],
                "age_text": r["age"],
                "age": age_to_years(r["age"]),
                "structure": r["structure_acronym"].strip('"'),
            })
    return cols


def load_expression(row_numbers):
    """Pull only the wanted rows out of the 183 MB matrix, in one pass."""
    want = set(row_numbers)
    got = {}
    path = os.path.join(DATA, "expression_matrix.csv")
    with open(path) as fh:
        for line in fh:
            head, _sep, rest = line.partition(",")
            try:
                rn = int(head)
            except ValueError:
                continue
            if rn in want:
                got[rn] = [float(x) for x in rest.split(",")]
                if len(got) == len(want):
                    break
    assert len(got) == len(want), (
        f"only {len(got)} of {len(want)} rows found in the matrix")
    return got


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [None] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def main():
    rows_out = []

    def add(block, key, value, note=""):
        rows_out.append((block, key, str(value), note))
        print(f"  {block}/{key:<32} {value}" + (f"   [{note}]" if note else ""))

    wanted, missing = load_rows_wanted()
    cols = load_columns()
    expr = load_expression(wanted.values())

    add("meta", "source", "BrainSpan RNA-Seq gene RPKM, Gencode v10")
    add("meta", "samples_total", len(cols))
    add("meta", "donors_total", len({c['donor'] for c in cols}))
    add("meta", "panel_missing", ";".join(missing) or "none",
        "absent from this build; dropped with a note, not silently")

    post = [c for c in cols if c["age"] is not None]
    add("meta", "samples_postnatal", len(post))
    add("meta", "donors_postnatal", len({c['donor'] for c in post}))
    ages = sorted({c["age"] for c in post})
    add("meta", "age_range_years", f"{min(ages):.2f} to {max(ages):.0f}")
    add("meta", "n_distinct_ages", len(ages))

    idx = [c["n"] - 1 for c in post]          # column_num is 1-based
    age = [c["age"] for c in post]

    def series(sym):
        return [expr[wanted[sym]][i] for i in idx]

    def rho_of(sym, sel=None):
        xs = age if sel is None else [age[i] for i in sel]
        ys = series(sym) if sel is None else [series(sym)[i] for i in sel]
        if len(xs) < 4 or len(set(ys)) < 3:
            return None, None, len(xs)
        r, p = stats.spearmanr(xs, ys)
        return float(r), float(p), len(xs)

    # -------------------------------------------------------------- the gate
    print("\n[gate] a textbook postnatal riser must rise")
    r, p, n = rho_of(POSITIVE_CONTROL)
    add("gate", f"{POSITIVE_CONTROL}_vs_age", f"rho={r:+.3f} p={p:.3g} n={n}")
    if r is None or r <= 0.3 or p >= 0.01:
        add("gate", "verdict", "SCAN INVALID",
            "the positive control did not rise; the age parsing or the "
            "matrix join is wrong and no statement is made about the target")
        write(rows_out)
        return 1
    add("gate", "verdict", "scan works")

    print("\n[negative control] housekeeping must stay flat")
    hk_ok = True
    for g in HOUSEKEEPING:
        r, p, n = rho_of(g)
        flat = abs(r) < 0.3
        hk_ok &= flat
        add("negative_control", f"{g}_vs_age",
            f"rho={r:+.3f} p={p:.3g}", "flat" if flat else "MOVES WITH AGE")
    add("negative_control", "all_flat", hk_ok,
        "if false, read every association below as partly technical")

    # ------------------------------------------------- the test, pooled
    print("\n[pooled] every panel gene against age, postnatal samples")
    panel = [TARGET] + ENDOTHELIAL + CONTEXT
    res, ps = {}, []
    for g in panel:
        r, p, n = rho_of(g)
        res[g] = (r, p, n)
        ps.append(p)
    adj = holm(ps)
    for g, a in zip(panel, adj):
        r, p, n = res[g]
        add("pooled", g, f"rho={r:+.3f} p={p:.3g} holm={a:.3g} n={n}")

    # --------------------------------- the discriminating comparison
    print("\n[vessel or transcript] target against the identity genes")
    tr = res[TARGET][0]
    endo_rhos = {g: res[g][0] for g in ENDOTHELIAL}
    add("vessel_or_transcript", "target_rho", f"{tr:+.3f}")
    add("vessel_or_transcript", "endothelial_rhos",
        "; ".join(f"{g}={v:+.3f}" for g, v in endo_rhos.items()))
    ceiling = max(abs(v) for v in endo_rhos.values())
    add("vessel_or_transcript", "identity_gene_ceiling", f"{ceiling:+.3f}",
        "largest absolute age association among the identity genes")
    add("vessel_or_transcript", "target_exceeds_ceiling", abs(tr) > ceiling,
        "if true, the rise is not explained by more vessel in the sample")

    # Ratio to each identity gene: a per-sample normalisation.
    for g in ENDOTHELIAL:
        num, den = series(TARGET), series(g)
        pairs = [(a, n / d) for a, n, d in zip(age, num, den) if d > 0]
        if len(pairs) < 10:
            add("ratio", f"{TARGET}_over_{g}", "too few usable samples")
            continue
        r, p = stats.spearmanr([a for a, _ in pairs], [v for _, v in pairs])
        add("ratio", f"{TARGET}_over_{g}",
            f"rho={r:+.3f} p={p:.3g} n={len(pairs)}",
            "target normalised per sample to an identity gene")

    # ------------------------------------------------------ by structure
    print("\n[structure] cerebrum and cerebellum")
    by_struct = {}
    for i, c in enumerate(post):
        by_struct.setdefault(c["structure"], []).append(i)
    scored = {s: v for s, v in by_struct.items() if len(v) >= MIN_SAMPLES}
    add("structure", "structures_scored", f"{len(scored)} of {len(by_struct)}",
        f"a structure needs at least {MIN_SAMPLES} postnatal samples")

    struct_rows, sps = [], []
    for s, sel in sorted(scored.items()):
        r, p, n = rho_of(TARGET, sel)
        if r is None:
            continue
        struct_rows.append((s, r, p, n))
        sps.append(p)
    sadj = holm(sps)
    for (s, r, p, n), a in zip(struct_rows, sadj):
        add("structure", s, f"rho={r:+.3f} p={p:.3g} holm={a:.3g} n={n}",
            "cerebellum" if s in CEREBELLAR else "")

    cb = [x for x in struct_rows if x[0] in CEREBELLAR]
    cx = [x for x in struct_rows if x[0] not in CEREBELLAR]
    if cb and cx:
        add("cerebellum", "cerebellar_structures",
            "; ".join(f"{s}={r:+.3f}" for s, r, _p, _n in cb))
        add("cerebellum", "non_cerebellar_median",
            f"{sorted(x[1] for x in cx)[len(cx) // 2]:+.3f}",
            f"across {len(cx)} structures")
        add("cerebellum", "reading",
            "a cerebellar association close to the cerebral one extends the "
            "finding to the cerebellum; a flat cerebellum bounds it to the "
            "cerebrum and is equally reportable")

    add("limit", "bulk_tissue",
        "BrainSpan is bulk tissue and carries no cell type resolution. It "
        "cannot show that the rise is endothelial; it can only show that it "
        "is not explained by the identity genes moving together with it")
    add("limit", "donor_counts",
        "postnatal donors are few per age band and each donor contributes "
        "many structures, so structures are not independent of one another")

    write(rows_out)
    return 0


def write(rows_out):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("block\tkey\tvalue\tnote\n")
        for r in rows_out:
            fh.write("\t".join(r) + "\n")
    print(f"\nwrote {len(rows_out)} rows -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
