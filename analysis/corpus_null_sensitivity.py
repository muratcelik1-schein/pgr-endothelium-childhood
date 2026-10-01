#!/usr/bin/env python3
"""How much of the corpus result depends on the size of the null?

The specificity metric in `dimension_scanner.py` divides a gene's detected
fraction in a cell type by that cell type's mean across every gene queried.
The queried set is the panel plus the null, so the size of the null sets the
stability of the denominator, and it also sets the resolution of the
permutation p, which is 1/(usable null + 1).

This script runs the same scanner unchanged at several null sizes and records
the target's endothelial rank at each, so that the configuration reported in
the manuscript is read against the ones that were not. It imports the scanner
rather than copying any of it, and it overrides only the null size and the
output path.

Run: python3 corpus_null_sensitivity.py --sizes 40 100 250 --out <tsv>
"""
import argparse, importlib.util, io, os, sys, contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--sizes", type=int, nargs="+", default=[40, 100, 250])
ap.add_argument("--out", default=os.path.join(HERE, "corpus_null_sensitivity.tsv"))
A = ap.parse_args()

def load():
    spec = importlib.util.spec_from_file_location(
        "dimension_scanner", os.path.join(HERE, "dimension_scanner.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["dimension_scanner"] = m
    spec.loader.exec_module(m)
    return m

rows = [("null_size", "key", "value", "note")]
for n in A.sizes:
    m = load()
    m.NULL_SIZE = n
    tmp = os.path.join(HERE, f".null{n}.tsv")
    m.OUT = tmp
    with contextlib.redirect_stdout(io.StringIO()):
        m.main()
    got = {}
    for ln in open(tmp).read().splitlines()[1:]:
        p = ln.split("\t")
        if len(p) >= 3: got[(p[0], p[1])] = p[2]
    os.remove(tmp)
    for key in [("corpus_profile", "PGR::best_endothelial_rank"),
                ("corpus_profile", "PGR::n_cell_types"),
                ("corpus_profile", "PGR::top3"),
                ("corpus_profile", "positive_control"),
                ("programme", "PGR::endothelial_rank_fraction"),
                ("programme", "null_size"),
                ("programme", "null_rank_median"),
                ("meta", "brain_cell_types_scored")]:
        rows.append((str(n), f"{key[0]}/{key[1]}", got.get(key, "absent"), ""))
    print(f"null {n:4d} -> endothelial rank {got.get(('corpus_profile','PGR::best_endothelial_rank'))}"
          f" of {got.get(('corpus_profile','PGR::n_cell_types'))}"
          f" | usable null {got.get(('programme','null_size'))}"
          f" | top3 {got.get(('corpus_profile','PGR::top3'))}", flush=True)

with open(A.out, "w") as fh:
    for r in rows: fh.write("\t".join(r) + "\n")
print("wrote", A.out)
