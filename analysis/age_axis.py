# -*- coding: utf-8 -*-
"""Verification of the unit of the age axis and derivation of the postnatal equivalents.

The age variable of the developmental series counts from conception. This script
verifies, from the donor labels, that for donors named `NY_...`
the difference between axis value and label is constant across donors, and that constant
is the position of birth on the axis. Band edges and series limits are then
converted to postnatal years.

The check guards an error that would otherwise pass unnoticed: the Methods'
"donors older than nine months" wording described a cut that in fact falls at birth.
The asserts below make a return of that wording impossible.

Input : donor_pgr_counts.csv (derived table shipped with the code)
Output: age_axis.tsv
Seed  : none, no randomness (2026 version)
"""
import csv, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "donor_pgr_counts.csv"), encoding="utf-8")))
age = sorted(float(r["age"]) for r in rows)

# --- 1. offset, from the labelled donors ---
lab = [(float(r["age"]), int(m.group(1)))
       for r in rows for m in [re.match(r"^(\d+)Y", r["donor"])] if m]
assert len(lab) >= 10, f"too few labelled donors: {len(lab)}"
off = sorted({round(a - y, 2) for a, y in lab})
assert len(off) == 1, f"offset is not constant: {off}"
BIRTH = off[0]

# --- 2. is the cut at birth ---
AGE_MIN = 0.75                       # the cut used by verify_all.py and extract_devbrain.py
assert abs(BIRTH - AGE_MIN) < 0.05, (
    f"cut {AGE_MIN} and birth {BIRTH} are at different positions; "
    "the Methods wording 'older than nine months' is wrong on this axis")
assert min(age) == BIRTH, f"youngest donor is not at birth: {min(age)}"

# --- 3. band edges and series limits ---
EDGES = [0.75, 2, 6, 12, 20]
FULL = [0, 0.5, 0.75]                # two further bands of the full series
pn = lambda x: round(x - BIRTH, 2)

out = [("quantity", "conception_axis", "postnatal", "note")]
out.append(("birth_on_axis", f"{BIRTH}", "0.00", "constant offset, from the labelled donors"))
out.append(("n_labelled_donors", str(len(lab)), "", "donors named NY_..."))
out.append(("series_min", f"{min(age)}", f"{pn(min(age)):.2f}", "youngest donor"))
out.append(("series_max", f"{max(age)}", f"{pn(max(age)):.2f}", "oldest donor"))
for e in EDGES:
    out.append((f"band_edge_{e}", f"{e}", f"{max(pn(e), 0.0):.2f}",
                "birth" if abs(e - BIRTH) < 0.05 else "postnatal equivalent"))
for e in FULL[:2]:
    out.append((f"full_series_edge_{e}", f"{e}", "prenatal",
                f"{round(BIRTH - e, 2)} years before birth, prenatal"))
out.append(("gestation_weeks_at_birth", f"{BIRTH}", "",
            f"{round(BIRTH * 52.1786, 1)} weeks"))

with open(os.path.join(HERE, "age_axis.tsv"), "w", encoding="utf-8", newline="") as f:
    csv.writer(f, delimiter="\t").writerows(out)

w = max(len(r[0]) for r in out)
for r in out:
    print(f"{r[0]:{w}}  {r[1]:>8}  {r[2]:>9}  {r[3]}")
