#!/usr/bin/env python3
"""Is the receptor's expression under genetic control, and is that control
biased toward the vessel wall?

None of the genetic routes tried so far reached this question. The targetome
test asked whether the receptor's binding repertoire is co-regulated with it;
the isoform test asked whether short reads separate two transcripts; the
chromatin modality of the multiome cohort is not distributed. Cis-eQTL is a
fourth route and it is free.

Design:

  Gate    A known strong eQTL gene must return hits in brain cortex. The same
          query without datasetId returns zero for every gene, silently, so
          the gate is not optional: a scan that finds nothing is invalid
          until the control proves the scan works.

  Scan    All 54 GTEx tissues, target gene, hits and sample size recorded
          together. A zero in a small tissue is a statement about power, not
          about biology, and the output keeps both numbers side by side.

  Confound  eQTL discovery scales with sample size. The association between
          hits and sample size is computed and reported, so any tissue
          pattern is read against it rather than instead of it.

Limit written into the output: GTEx is bulk tissue. A cis-eQTL in brain
cortex cannot be attributed to a cell type, so this can support a statement
about the gene in the tissue and never about endothelium specifically.

Writes analysis/eqtl_numbers.tsv and data/eqtl_per_tissue.tsv.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

SEED = 2026  # declared for parity; this pass draws nothing

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "eqtl_numbers.tsv")
PER_TISSUE = os.path.join(ROOT, "data", "eqtl_per_tissue.tsv")
CACHE = os.path.join(ROOT, "data", "gtex_cache")

API = "https://gtexportal.org/api/v2"
DATASET = "gtex_v8"          # omitting this returns zero hits for every gene

TARGET = "PGR"
# A gene whose cis-eQTL in brain is established and strong. If this returns
# nothing, the scan is broken and no statement about the target is made.
POSITIVE_CONTROL = "ERAP2"
CONTROL_TISSUE = "Brain_Cortex"

# Tissue groups, by whether the sampled tissue is a vessel wall, brain, or
# neither. Assigned from the tissue identifier, not from expectation.
VASCULAR_PREFIX = ("Artery_",)
BRAIN_PREFIX = ("Brain_",)


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
    raise AssertionError(f"GTEx unreachable after {tries} tries: {last}\n{url}")


def gencode_id(symbol):
    d = get(f"{API}/reference/gene?geneId={symbol}")
    rows = [r for r in d.get("data", []) if r.get("geneSymbol") == symbol]
    assert rows, f"{symbol}: no exact symbol match at GTEx"
    assert len(rows) == 1, f"{symbol}: {len(rows)} matches, ambiguous"
    return rows[0]["gencodeId"]


def eqtl_count(gencode, tissue, page_items=1):
    url = (f"{API}/association/singleTissueEqtl?gencodeId={gencode}"
           f"&tissueSiteDetailId={tissue}&datasetId={DATASET}"
           f"&itemsPerPage={page_items}")
    d = get(url)
    n = d.get("paging_info", {}).get("totalNumberOfItems")
    assert n is not None, f"no paging_info for {tissue}"
    return n, d.get("data", [])


def tissues():
    d = get(f"{API}/dataset/tissueSiteDetail?itemsPerPage=100")
    out = {}
    for r in d.get("data", []):
        tid = r["tissueSiteDetailId"]
        n = (r.get("eqtlSampleSummary") or {}).get("totalCount")
        out[tid] = n
    assert len(out) >= 40, f"only {len(out)} tissues returned"
    return out


def spearman(xs, ys):
    """Rank correlation without a scipy dependency."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main():
    rows = []

    def add(block, key, value, note=""):
        rows.append((block, key, str(value), note))
        print(f"  {block}/{key:<38} {value}" + (f"   [{note}]" if note else ""))

    print("Resolving identifiers")
    tgt = gencode_id(TARGET)
    ctl = gencode_id(POSITIVE_CONTROL)
    add("meta", "target", f"{TARGET} = {tgt}")
    add("meta", "positive_control", f"{POSITIVE_CONTROL} = {ctl}")
    add("meta", "dataset", DATASET,
        "omitting datasetId returns zero for every gene, silently")

    # ------------------------------------------------------------------ gate
    print("\n[gate] does the scan work at all?")
    n_ctl, ex = eqtl_count(ctl, CONTROL_TISSUE)
    add("gate", f"{POSITIVE_CONTROL}::{CONTROL_TISSUE}", n_ctl)
    if n_ctl == 0:
        add("gate", "verdict", "SCAN INVALID",
            "the positive control returned nothing; no statement is made "
            "about the target")
        write(rows)
        print("\n[stop] positive control failed. Result is 'scan invalid', "
              "not 'no eQTL'.")
        return 1
    add("gate", "verdict", "scan works",
        f"example variant {ex[0].get('snpId')} p={ex[0].get('pValue')}"
        if ex else "")

    # ------------------------------------------------------------------ scan
    print(f"\n[scan] {TARGET} across all GTEx tissues")
    tis = tissues()
    per = {}
    for i, (tid, n_samp) in enumerate(sorted(tis.items()), 1):
        n_eqtl, _ = eqtl_count(tgt, tid)
        per[tid] = (n_eqtl, n_samp)
        if n_eqtl:
            print(f"    {i:>2}/{len(tis)}  {tid:<34} {n_eqtl:>5} eQTL  "
                  f"(n={n_samp})")

    hits = {t: v for t, v in per.items() if v[0] > 0}
    add("scan", "n_tissues_scanned", len(per))
    add("scan", "n_tissues_with_eqtl", len(hits))
    add("scan", "tissues_with_eqtl",
        ";".join(f"{t}={v[0]}" for t, v in sorted(hits.items(),
                                                  key=lambda kv: -kv[1][0])))

    # --------------------------------------------------------------- confound
    print("\n[confound] discovery scales with sample size")
    ts = [t for t in per if per[t][1]]
    rho = spearman([per[t][1] for t in ts], [per[t][0] for t in ts])
    add("confound", "spearman_nEQTL_vs_sampleSize", f"{rho:+.3f}",
        f"across {len(ts)} tissues; read any tissue pattern against this")
    # Sample sizes of the tissues with and without hits.
    with_n = sorted(per[t][1] for t in hits)
    without_n = sorted(per[t][1] for t in per if per[t][0] == 0 and per[t][1])
    if with_n and without_n:
        add("confound", "sampleSize_median_with_hits",
            with_n[len(with_n) // 2])
        add("confound", "sampleSize_median_without_hits",
            without_n[len(without_n) // 2])

    # ----------------------------------------------------------- tissue class
    print("\n[class] vessel wall, brain, or neither")
    for label, pref in (("vascular", VASCULAR_PREFIX), ("brain", BRAIN_PREFIX)):
        grp = [t for t in per if t.startswith(pref)]
        hit = [t for t in grp if per[t][0] > 0]
        add("class", f"{label}_tissues_with_eqtl",
            f"{len(hit)} of {len(grp)}",
            ";".join(f"{t}={per[t][0]}(n={per[t][1]})" for t in sorted(hit))
            or "none")
    other = [t for t in per
             if not t.startswith(VASCULAR_PREFIX + BRAIN_PREFIX)]
    add("class", "other_tissues_with_eqtl",
        f"{len([t for t in other if per[t][0] > 0])} of {len(other)}")

    # ------------------------------------------------------- top variants
    print("\n[top] strongest association in each tissue with hits")
    for tid in sorted(hits, key=lambda t: -hits[t][0])[:6]:
        _n, data = eqtl_count(tgt, tid, page_items=250)
        if not data:
            continue
        best = min(data, key=lambda r: float(r.get("pValue", 1)))
        add("top", tid,
            f"{best.get('snpId')} {best.get('variantId')} "
            f"p={best.get('pValue'):.3g} nes={best.get('nes')}",
            f"{hits[tid][0]} eQTLs, n={hits[tid][1]}")

    add("limit", "bulk_tissue",
        "GTEx measures bulk tissue, so a cis-eQTL in brain cortex cannot be "
        "attributed to endothelium; this supports a statement about the gene "
        "in the tissue and never about a cell type")
    add("limit", "no_colocalisation_yet",
        "presence of an eQTL says the gene's expression is heritable in that "
        "tissue; it says nothing about a phenotype until a GWAS signal at "
        "the same locus is tested for colocalisation")

    write(rows)
    with open(PER_TISSUE, "w") as fh:
        fh.write("tissue\tn_eqtl\teqtl_sample_size\n")
        for t in sorted(per):
            fh.write(f"{t}\t{per[t][0]}\t{per[t][1]}\n")
    print(f"wrote per-tissue table -> {os.path.relpath(PER_TISSUE, ROOT)}")
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
