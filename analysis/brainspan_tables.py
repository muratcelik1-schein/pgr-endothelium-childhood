#!/usr/bin/env python3
"""Renders Supplementary Tables S20 and 21 from the output of brainspan_axis.py.

Every value printed in those two tables is produced here and written to
brainspan_tables.tsv, so that the submission checks can compare the manuscript
against a table rather than against a transcription. Nothing is rounded: a
p value that the scan wrote in exponent form is printed in exponent form and
one it wrote as a decimal is printed as a decimal, so each printed string
also appears in the scan's own output.

Run: python3 brainspan_tables.py [--in brainspan_numbers.tsv] [--out brainspan_tables.tsv]
"""
import argparse, csv, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="src", default=os.path.join(HERE, "brainspan_numbers.tsv"))
ap.add_argument("--out", default=os.path.join(HERE, "brainspan_tables.tsv"))
A = ap.parse_args()

rows = list(csv.DictReader(open(A.src), delimiter="\t"))
def val(block, key):
    for r in rows:
        if r["block"] == block and r["key"] == key: return r["value"]
    raise KeyError((block, key))

FIELD = re.compile(r"rho=([+-][\d.]+)|p=([\w.+-]+)|holm=([\w.+-]+)|n=(\d+)")
def parse(v):
    out = {}
    for rho, p, holm, n in FIELD.findall(v):
        if rho: out["rho"] = rho
        if p: out["p"] = p
        if holm: out["holm"] = holm
        if n: out["n"] = n
    return out

def show(s):
    """Exponent form becomes 'm x 10-e'; a decimal is left as the scan wrote it."""
    if s is None: return ""
    if "e" in s:
        m, e = s.split("e")
        return f"{m} x 10{int(e)}" if int(e) >= 0 else f"{m} x 10-{abs(int(e))}"
    return s

PANEL = [("MBP", ("gate", "MBP_vs_age"), "positive control, postnatal myelination"),
         ("ACTB", ("negative_control", "ACTB_vs_age"), "negative control"),
         ("GAPDH", ("negative_control", "GAPDH_vs_age"), "negative control"),
         ("TBP", ("negative_control", "TBP_vs_age"), "negative control"),
         ("PGR", ("pooled", "PGR"), "target"),
         ("CLDN5", ("pooled", "CLDN5"), "endothelial identity"),
         ("FLT1", ("pooled", "FLT1"), "endothelial identity"),
         ("VWF", ("pooled", "VWF"), "endothelial identity"),
         ("NR3C1", ("pooled", "NR3C1"), "related receptor"),
         ("NR3C2", ("pooled", "NR3C2"), "related receptor"),
         ("ESR1", ("pooled", "ESR1"), "related receptor"),
         ("AR", ("pooled", "AR"), "related receptor"),
         ("ABCB1", ("pooled", "ABCB1"), "barrier transporter"),
         ("SNAP25", ("pooled", "SNAP25"), "neuronal context"),
         ("GFAP", ("pooled", "GFAP"), "astrocytic context"),
         ("AKR1C1", ("pooled", "AKR1C1"), "converting enzyme"),
         ("HSD3B2", ("pooled", "HSD3B2"), "see legend"),
         ("PGR / CLDN5", ("ratio", "PGR_over_CLDN5"), "ratio within each sample"),
         ("PGR / FLT1", ("ratio", "PGR_over_FLT1"), "ratio within each sample"),
         ("PGR / VWF", ("ratio", "PGR_over_VWF"), "ratio within each sample")]

printed, t18 = [], ["| Gene | Association with age | p | Holm | Role |", "|---|---|---|---|---|"]
for lab, (b, k), role in PANEL:
    f = parse(val(b, k))
    p, h = show(f.get("p")), show(f.get("holm"))
    t18.append(f"| {lab} | {f['rho']} | {p} | {h} | {role} |")
    printed += [("18", lab, f["rho"]), ("18", lab + " p", p)] + ([("18", lab + " holm", h)] if h else [])

st = [(r["key"], parse(r["value"]), r["note"]) for r in rows
      if r["block"] == "structure" and r["key"] != "structures_scored"]
t19 = ["| Structure | Association with age | p | Holm | Samples |", "|---|---|---|---|---|"]
for k, f, note in sorted(st, key=lambda x: -float(x[1]["rho"])):
    lab = k + (" (cerebellum)" if note == "cerebellum" else "")
    p, h = show(f.get("p")), show(f.get("holm"))
    t19.append(f"| {lab} | {f['rho']} | {p} | {h} | {f['n']} |")
    printed += [("19", k, f["rho"]), ("19", k + " p", p), ("19", k + " holm", h)]

with open(A.out, "w") as fh:
    fh.write("table\tlabel\tprinted_value\n")
    for tb, lab, v in printed: fh.write(f"{tb}\t{lab}\t{v}\n")
print("\n".join(t18)); print(); print("\n".join(t19))
print(f"\nwrote {A.out} ({len(printed)} printed values)", flush=True)
