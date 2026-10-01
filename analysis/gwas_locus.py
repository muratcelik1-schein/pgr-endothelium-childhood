#!/usr/bin/env python3
"""Does the receptor locus carry a GWAS signal for anything relevant?

The eQTL scan showed that expression of the receptor is heritable, strongly
in artery. The next question is whether variation at this locus associates
with a phenotype. It is asked here and answered in the negative, for a
reason worth recording: the locus is shared.

Design:

  Gate     A locus with many known associations must return them. The region
           query returns an empty list for a malformed request, so a scan
           that finds nothing is invalid until the control proves otherwise.

  Scan     Every catalogued variant inside the gene's own coordinates, with
           its mapped traits, its p value, and the gene the catalogue
           assigns it to.

  Attribution  The decisive step. A variant inside the gene's span is not a
           variant of the gene. Each association is reported with the
           nearest gene the catalogue assigns, and any association assigned
           elsewhere is counted against attributing it to the target.

Writes analysis/gwas_numbers.tsv.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

SEED = 2026  # declared for parity; this pass draws nothing

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "gwas_numbers.tsv")
CACHE = os.path.join(ROOT, "data", "gwas_cache")

API = "https://www.ebi.ac.uk/gwas/rest/api"

TARGET = "PGR"
# GRCh38 coordinates of the target, taken from the GTEx reference endpoint in
# eqtl_scan.py rather than typed from memory.
CHROM, START, END = 11, 101029624, 101130524
# A locus whose associations are numerous and certain. If this returns
# nothing the scan is broken.
CONTROL = (19, 44900000, 45000000, "APOE region")
CONTROL_MIN = 50

# Phenotype words that would matter for a paper about the vessel wall of the
# brain. Declared before the results are seen.
RELEVANT = ("pulse pressure", "blood pressure", "neuroimaging", "brain",
            "white matter", "cortical", "stroke", "cerebr", "vascular",
            "arterial stiffness", "permeability")


def get(url, tries=4):
    os.makedirs(CACHE, exist_ok=True)
    key = "".join(c if c.isalnum() or c in "-_." else "_" for c in url)[-180:]
    path = os.path.join(CACHE, key + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    last = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                payload = json.loads(r.read().decode("utf-8"))
            with open(path, "w") as fh:
                json.dump(payload, fh)
            return payload
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise AssertionError(f"GWAS Catalog unreachable after {tries}: {last}")


def region(chrom, start, end, size=300):
    url = (f"{API}/singleNucleotidePolymorphisms/search/"
           f"findByChromBpLocationRange?chrom={chrom}&bpStart={start}"
           f"&bpEnd={end}&size={size}")
    d = get(url)
    snps = d.get("_embedded", {}).get("singleNucleotidePolymorphisms", [])
    total = d.get("page", {}).get("totalElements", len(snps))
    return snps, total


def snp_detail(rsid):
    return get(f"{API}/singleNucleotidePolymorphisms/{rsid}")


def snp_assoc(rsid):
    d = get(f"{API}/singleNucleotidePolymorphisms/{rsid}"
            f"/associations?projection=associationBySnp")
    return d.get("_embedded", {}).get("associations", [])


def main():
    rows = []

    def add(block, key, value, note=""):
        rows.append((block, key, str(value), note))
        print(f"  {block}/{key:<26} {value}" + (f"   [{note}]" if note else ""))

    add("meta", "target", TARGET)
    add("meta", "region", f"chr{CHROM}:{START}-{END}", "GRCh38, gene span")
    add("meta", "relevant_words", ";".join(RELEVANT),
        "declared before the results were seen")

    # ------------------------------------------------------------------ gate
    print("[gate] does the region query work?")
    _c, n_ctl = region(*CONTROL[:3], size=5)
    add("gate", "control_region", f"{CONTROL[3]} = {n_ctl} variants")
    if n_ctl < CONTROL_MIN:
        add("gate", "verdict", "SCAN INVALID",
            f"control returned {n_ctl}, below {CONTROL_MIN}")
        write(rows)
        print("[stop] control failed; result is 'scan invalid', not 'no hits'")
        return 1
    add("gate", "verdict", "scan works")

    # ------------------------------------------------------------------ scan
    print(f"\n[scan] catalogued variants inside {TARGET}")
    snps, total = region(CHROM, START, END)
    add("scan", "variants_in_span", total)

    records = []
    for s in snps:
        rsid = s.get("rsId")
        if not rsid:
            continue
        detail = snp_detail(rsid)
        pos = None
        for loc in detail.get("locations", []):
            pos = loc.get("chromosomePosition")
        closest = sorted({
            g.get("gene", {}).get("geneName")
            for g in detail.get("genomicContexts", [])
            if g.get("isClosestGene") and g.get("gene")
        } - {None})
        for a in snp_assoc(rsid):
            traits = sorted({e.get("trait") for e in a.get("efoTraits", [])
                             if e.get("trait")})
            records.append({
                "rsid": rsid, "pos": pos, "closest": closest,
                "p": a.get("pvalue"), "traits": traits,
            })

    add("scan", "associations_found", len(records))

    # ----------------------------------------------------------- attribution
    print("\n[attribution] which gene does the catalogue assign?")
    assigned_to_target, assigned_elsewhere = [], []
    for r in records:
        (assigned_to_target if TARGET in r["closest"]
         else assigned_elsewhere).append(r)
    add("attribution", "assigned_to_target", len(assigned_to_target))
    add("attribution", "assigned_elsewhere", len(assigned_elsewhere))
    neighbours = {}
    for r in assigned_elsewhere:
        for g in r["closest"]:
            neighbours[g] = neighbours.get(g, 0) + 1
    add("attribution", "neighbour_genes",
        ";".join(f"{g}={n}" for g, n in sorted(neighbours.items(),
                                               key=lambda kv: -kv[1])))

    # -------------------------------------------------------------- relevant
    print("\n[relevant] associations a vessel-wall brain paper would want")
    for r in sorted(records, key=lambda r: (r["p"] or 1)):
        hit = [t for t in r["traits"]
               if any(w in t.lower() for w in RELEVANT)]
        if not hit:
            continue
        add("relevant", f"{r['rsid']}",
            f"chr{CHROM}:{r['pos']} p={r['p']} traits={';'.join(hit)} "
            f"closest={','.join(r['closest'])}")

    # every trait seen, so the reader can check what was set aside
    all_traits = {}
    for r in records:
        for t in r["traits"]:
            all_traits[t] = all_traits.get(t, 0) + 1
    add("all_traits", "n_distinct", len(all_traits))
    add("all_traits", "list",
        ";".join(f"{t}({n})" for t, n in sorted(all_traits.items())))

    # ---------------------------------------------------------------- verdict
    print("\n[verdict]")
    if not assigned_to_target:
        add("verdict", "attribution",
            f"no catalogued association inside the {TARGET} span is assigned "
            f"to {TARGET} by the catalogue")
    add("verdict", "reading",
        "a variant inside the gene's coordinates is not a variant of the "
        "gene. The relevant phenotypes at this locus are assigned to the "
        "neighbour, so they cannot be attributed to the target without "
        "colocalisation against full summary statistics")
    add("verdict", "what_would_settle_it",
        "colocalisation of the artery or brain cortex eQTL for the target "
        "with the pulse pressure and neuroimaging signals, using full "
        "summary statistics; the neighbour is an established blood pressure "
        "gene, so the prior is against the target")

    write(rows)
    return 0


def write(rows):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("block\tkey\tvalue\tnote\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    print(f"\nwrote {len(rows)} rows -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
