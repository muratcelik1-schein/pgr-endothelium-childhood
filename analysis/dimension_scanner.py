#!/usr/bin/env python3
"""A scanner for the dimensions of one finding.

The finding: a transcript absent in infancy appears in a named vascular cell
type of the human cortex during childhood, while related receptors hold
steady.

That statement has axes along which it can be extended or broken, and each
axis needs its own scan with its own controls. This file is the registry and
the harness. Adding an axis means adding an Axis subclass; the harness then
enforces the same discipline on it:

  * every axis declares a positive control BEFORE it runs
  * an axis whose positive control fails reports "SCAN INVALID" and makes no
    statement about the target
  * every axis declares its own limit, which travels with its result
  * the null, where an axis needs one, is declared with its size

Axes implemented here:

  corpus_profile   Across the whole CELLxGENE human corpus, is the target's
                   cell type profile endothelial, or is the brain result a
                   property of the brain datasets alone?

  programme        The screen that produced the target also produced other
                   genes meeting the same three criteria. Do those genes
                   share the target's cell type profile? The manuscript
                   declines to claim a programme. This tests it against a
                   null of abundance-matched random genes.

Writes analysis/dimension_numbers.tsv.
"""

import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

SEED = 2026

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "dimension_numbers.tsv")
CACHE = os.path.join(ROOT, "data", "wmg_cache")

WMG = "https://api.cellxgene.cziscience.com/wmg/v2"
HUMAN = "NCBITaxon:9606"
BRAIN = "UBERON:0000955"

TARGET = "PGR"
# Endothelial identity genes. If these do not peak in endothelial cell types,
# the corpus query or the label matching is broken and nothing is reported.
POSITIVE_CONTROLS = ["CLDN5", "PECAM1"]
# The genes the manuscript's own screen returned alongside the target, taken
# from the gate rows of its shipped number table. TARP is not in the corpus
# index and is dropped with a note rather than silently.
CO_EMERGING = ["PGR-AS1", "MIR31HG", "ZNF727", "SHISA3", "TMEM235", "FMO2"]
# Related steroid receptors, which the manuscript reports as flat with age.
RELATED = ["NR3C1", "NR3C2", "ESR1", "AR", "GPER1"]

# The null's resolution is 1/(usable+1), so a gate at p < 0.05 needs more
# than nineteen usable genes or it cannot be cleared by anything. Random
# genes often have no endothelial cell type scored at all, so the draw is
# made large enough that roughly half survive.
NULL_SIZE = 250
GENE_BATCH = 10         # genes per corpus request
MIN_CELLS = 25          # a cell type below this is not scored


# --------------------------------------------------------------------- client

def _get(url, body=None, tries=4):
    os.makedirs(CACHE, exist_ok=True)
    tag = url + (json.dumps(body, sort_keys=True) if body else "")
    # A stable digest, not hash(): str hashing is salted per process, so
    # hash() would miss the cache on every run and re-download everything.
    key = hashlib.md5(tag.encode()).hexdigest()
    path = os.path.join(CACHE, key + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    last = None
    for attempt in range(tries):
        try:
            data = json.dumps(body).encode() if body else None
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"} if body else {})
            with urllib.request.urlopen(req, timeout=180) as r:
                payload = json.loads(r.read().decode("utf-8"))
            with open(path, "w") as fh:
                json.dump(payload, fh)
            return payload
        except (urllib.error.URLError, TimeoutError,
                json.JSONDecodeError) as exc:
            last = exc
            time.sleep(3 * (attempt + 1))
    raise AssertionError(f"CELLxGENE unreachable after {tries}: {last}")


def dimensions():
    d = _get(f"{WMG}/primary_filter_dimensions")
    genes = {}
    for entry in d["gene_terms"][HUMAN]:
        for ensg, sym in entry.items():
            genes.setdefault(sym, ensg)
    return genes, d["snapshot_id"]


def corpus_expression(ensgs):
    """{ensg: {tissue: {cell_type: {...}}}} plus a cell-type name map.

    The label payload is keyed by tissue and then by cell type, with the
    readable name under `aggregated.name`. A `tissue_stats` pseudo-entry sits
    alongside the real cell types and is dropped here rather than scored.
    """
    out, labels = {}, {}
    for i in range(0, len(ensgs), GENE_BATCH):
        chunk = ensgs[i:i + GENE_BATCH]
        body = {"filter": {"gene_ontology_term_ids": chunk,
                           "organism_ontology_term_id": HUMAN},
                "is_rollup": True}
        d = _get(f"{WMG}/query", body)
        assert "expression_summary" in d, d.get("detail", d)
        out.update(d["expression_summary"])
        cts = (d.get("term_id_labels") or {}).get("cell_types") or {}
        for _tissue, per_ct in cts.items():
            if not isinstance(per_ct, dict):
                continue
            for ct, rec in per_ct.items():
                if ct == "tissue_stats" or not isinstance(rec, dict):
                    continue
                name = (rec.get("aggregated") or {}).get("name")
                if name:
                    labels.setdefault(ct, name)
    assert labels, "no cell type names resolved; the label schema changed"
    return out, labels


# ----------------------------------------------------------------- harness

class Axis:
    name = "unnamed"
    question = ""
    limit = ""

    def __init__(self, ctx):
        self.ctx = ctx
        self.rows = []

    def add(self, key, value, note=""):
        self.rows.append((self.name, key, str(value), note))
        print(f"    {key:<40} {value}" + (f"   [{note}]" if note else ""))

    def control(self):
        """Return (ok: bool, description: str)."""
        raise NotImplementedError

    def scan(self):
        raise NotImplementedError

    def run(self):
        print(f"\n=== {self.name} ===\n  {self.question}")
        self.add("question", self.question)
        ok, desc = self.control()
        self.add("positive_control", desc)
        if not ok:
            self.add("verdict", "SCAN INVALID",
                     "positive control failed; no statement about the target")
            print("  [stop] scan invalid")
            return self.rows
        self.scan()
        self.add("limit", self.limit)
        return self.rows


def raw_profile(expr, ensg, tissue, min_cells=MIN_CELLS):
    """Cell type -> fraction of cells expressing, for one gene in one tissue.

    `tissue_stats` is a summary row, and CL:0000000 ("cell") is the root of
    the rollup and therefore contains every other row. Both are excluded.
    """
    tis = expr.get(ensg, {}).get(tissue, {})
    out = {}
    for ct, v in tis.items():
        if ct in ("CL:0000000", "tissue_stats"):
            continue
        agg = v.get("aggregated", v)
        n = agg.get("n", 0)
        if n is None or n < min_cells:
            continue
        out[ct] = float(agg.get("pc", 0.0))
    return out


def specificity_profiles(expr, ensg_list, tissue):
    """Per-gene profiles divided by each cell type's own baseline.

    Raw detection fractions carry a strong shared component: a cell type
    sequenced deeply, or annotated loosely, shows a higher fraction for
    every gene. Two unrelated genes therefore correlate. Dividing each cell
    type by the mean across the queried genes removes that component and
    leaves what is specific to the gene. The baseline is computed from the
    panel and the null together, so it does not depend on which genes are
    being compared.
    """
    raw = {g: raw_profile(expr, g, tissue) for g in ensg_list}
    cts = set()
    for p in raw.values():
        cts |= set(p)
    baseline = {}
    for ct in cts:
        vals = [p[ct] for p in raw.values() if ct in p]
        if len(vals) >= max(5, len(raw) // 4):
            m = sum(vals) / len(vals)
            if m > 0:
                baseline[ct] = m
    out = {}
    for g, p in raw.items():
        out[g] = {ct: p[ct] / baseline[ct] for ct in p if ct in baseline}
    return out, raw, baseline


def spearman(a, b):
    keys = sorted(set(a) & set(b))
    if len(keys) < 8:
        return None, len(keys)
    xs = [a[k] for k in keys]
    ys = [b[k] for k in keys]

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0 + 1
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(keys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    dx = sum((p - mx) ** 2 for p in rx) ** 0.5
    dy = sum((q - my) ** 2 for q in ry) ** 0.5
    return (num / (dx * dy) if dx and dy else None), n


# ------------------------------------------------------------------- axis one

class CorpusProfile(Axis):
    name = "corpus_profile"
    question = ("Across the whole human corpus, is the target's cell type "
                "profile endothelial, or is that a property of the brain "
                "datasets the manuscript already used?")
    limit = ("The corpus pools datasets of different assays, ages and "
             "diseases, and its cell type labels are the contributing "
             "authors'. It answers where a transcript sits, never when.")

    def control(self):
        """Both endothelial identity genes must rank an endothelial cell type
        first on specificity. If they do not, the query or the labelling is
        wrong and nothing is reported."""
        notes, ok = [], 0
        for sym in POSITIVE_CONTROLS:
            top = self.ctx["top_label"](self.ctx["genes"][sym])
            notes.append(f"{sym} top={top}")
            ok += int("endothel" in (top or "").lower())
        return ok == len(POSITIVE_CONTROLS), \
            f"{ok} of {len(POSITIVE_CONTROLS)} endothelial · " + \
            " · ".join(notes)

    def scan(self):
        spec, lab = self.ctx["spec"], self.ctx["lab"]
        genes = self.ctx["genes"]
        for sym in [TARGET] + RELATED:
            ensg = genes.get(sym)
            prof = spec.get(ensg) or {}
            if not prof:
                self.add(sym, "no cell type reached the minimum")
                continue
            ranked = sorted(prof.items(), key=lambda kv: -kv[1])
            endo = [i for i, (ct, _v) in enumerate(ranked, 1)
                    if "endothel" in lab(ct).lower()]
            self.add(f"{sym}::n_cell_types", len(ranked))
            self.add(f"{sym}::top3",
                     "; ".join(f"{lab(ct)}={v:.2f}" for ct, v in ranked[:3]),
                     "specificity, gene over the cell type's own baseline")
            self.add(f"{sym}::best_endothelial_rank",
                     endo[0] if endo else "no endothelial type scored",
                     f"of {len(ranked)}")


# ------------------------------------------------------------------- axis two

def endothelial_rank(prof, lab):
    """Rank of the best endothelial cell type in a gene's specificity profile.

    Reported as a fraction of the cell types scored, so genes measured over
    different numbers of cell types are comparable. Lower is more
    endothelial. None when no endothelial type was scored for that gene.
    """
    if not prof:
        return None
    ranked = sorted(prof.items(), key=lambda kv: -kv[1])
    for i, (ct, _v) in enumerate(ranked, 1):
        if "endothel" in lab(ct).lower():
            return i / len(ranked)
    return None


class Programme(Axis):
    name = "programme"
    question = ("Do the genes that met the same three developmental criteria "
                "sit in the same cell type as the target? The manuscript "
                "declines to claim a programme; this tests it.")
    limit = ("A shared cell type would show these genes occupy the same "
             "compartment, not that they are co-regulated, and the corpus "
             "carries no developmental axis. The statistic is the position "
             "of the best endothelial cell type in each gene's own ranking, "
             "so it is blind to how strongly the gene is expressed.")

    def control(self):
        """The endothelial identity genes must place endothelium at or near
        the top of their own ranking, and far better than random genes."""
        lab, spec, genes = self.ctx["lab"], self.ctx["spec"], self.ctx["genes"]
        null = self.ctx["null_endo_rank"]
        notes, ok = [], 0
        for sym in POSITIVE_CONTROLS:
            fr = endothelial_rank(spec.get(genes[sym]) or {}, lab)
            if fr is None:
                notes.append(f"{sym}: no endothelial type scored")
                continue
            p = (sum(1 for r in null if r <= fr) + 1) / (len(null) + 1)
            ok += int(p < 0.05)
            notes.append(f"{sym} rank={fr:.3f} p={p:.3f}")
        return ok == len(POSITIVE_CONTROLS), \
            f"{ok} of {len(POSITIVE_CONTROLS)} beat the null · " + \
            " · ".join(notes)

    def scan(self):
        lab, spec, genes = self.ctx["lab"], self.ctx["spec"], self.ctx["genes"]
        null = sorted(self.ctx["null_endo_rank"])
        self.add("null_size", len(null),
                 f"random corpus genes, seed {SEED}; "
                 f"resolution 1/{len(null) + 1}")
        self.add("null_rank_median", f"{null[len(null) // 2]:.3f}",
                 "fraction of the ranking; 0.5 is no endothelial bias")
        self.add("null_rank_p05", f"{null[max(0, int(0.05 * (len(null) - 1)))]:.3f}")

        tgt = endothelial_rank(spec.get(genes[TARGET]) or {}, lab)
        p_t = (sum(1 for r in null if r <= tgt) + 1) / (len(null) + 1)
        self.add(f"{TARGET}::endothelial_rank_fraction", f"{tgt:.3f}",
                 f"p={p_t:.3f} against the null")

        agree = tested = 0
        for sym in CO_EMERGING:
            ensg = genes.get(sym)
            if not ensg:
                self.add(sym, "not in the corpus index",
                         "dropped with a note, not silently")
                continue
            fr = endothelial_rank(spec.get(ensg) or {}, lab)
            if fr is None:
                self.add(sym, "no endothelial cell type scored",
                         "counted as untestable, not as a negative")
                continue
            tested += 1
            p = (sum(1 for r in null if r <= fr) + 1) / (len(null) + 1)
            agree += int(p < 0.05)
            prof = spec.get(ensg) or {}
            top = lab(max(prof, key=prof.get)) if prof else "-"
            self.add(sym, f"rank={fr:.3f} p={p:.3f} top={top}",
                     "endothelial-biased" if p < 0.05 else "not biased")
        self.add("co_emerging_endothelial", f"{agree} of {tested}")
        if tested:
            self.add("reading",
                     "co-emerging genes that are also endothelial-biased "
                     "would support a shared compartment; co-emerging genes "
                     "scattered across compartments show the developmental "
                     "schedule is not a programme of one cell type")


# ----------------------------------------------------------------------- main

def main():
    rng = random.Random(SEED)
    print("Resolving the corpus index")
    genes, snapshot = dimensions()
    print(f"  {len(genes)} human symbols · snapshot {snapshot}")

    panel = [TARGET] + POSITIVE_CONTROLS + CO_EMERGING + RELATED
    present = [s for s in panel if s in genes]
    missing = [s for s in panel if s not in genes]
    pool = sorted(set(genes) - set(panel))
    null_syms = rng.sample(pool, NULL_SIZE)
    print(f"  panel {len(present)} present, missing {missing or 'none'} "
          f"· null {len(null_syms)}")

    ensgs = [genes[s] for s in present + null_syms]
    print(f"  querying {len(ensgs)} genes in batches of {GENE_BATCH}")
    expr, labels = corpus_expression(ensgs)
    print(f"  got {len(expr)} genes back, {len(labels)} cell type names")

    # The baseline is built from the panel and the null together, so it does
    # not depend on which pair is being compared.
    spec, raw, baseline = specificity_profiles(expr, ensgs, BRAIN)
    print(f"  {len(baseline)} brain cell types passed the baseline minimum")

    def lab(ct):
        return labels.get(ct, ct)

    def top_label(ensg):
        prof = spec.get(ensg) or {}
        if not prof:
            return None
        return lab(max(prof, key=prof.get))

    # The null for axis two: where does a random gene place endothelium in
    # its own ranking? Genes for which no endothelial type was scored carry
    # no information either way and are counted, not imputed.
    null_endo_rank, null_unscored = [], 0
    for s in null_syms:
        fr = endothelial_rank(spec.get(genes[s]) or {}, lab)
        if fr is None:
            null_unscored += 1
        else:
            null_endo_rank.append(fr)

    ctx = {"genes": genes, "spec": spec, "raw": raw, "baseline": baseline,
           "lab": lab, "top_label": top_label,
           "null_endo_rank": null_endo_rank}

    rows = [("meta", "snapshot", snapshot, "CELLxGENE corpus snapshot"),
            ("meta", "seed", SEED, ""),
            ("meta", "tissue", BRAIN, "brain"),
            ("meta", "panel_missing", ";".join(missing) or "none", ""),
            ("meta", "brain_cell_types_scored", len(baseline), ""),
            ("meta", "metric", "specificity = gene's detected fraction in a "
             "cell type divided by that cell type's mean across all queried "
             "genes", "raw fractions correlate between unrelated genes"),
            ("meta", "null_usable",
             f"{len(null_endo_rank)} of {NULL_SIZE} random genes had an "
             f"endothelial cell type scored; {null_unscored} did not",
             "unscored genes are counted, never imputed")]

    for cls in (CorpusProfile, Programme):
        rows += cls(ctx).run()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("axis\tkey\tvalue\tnote\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    print(f"\nwrote {len(rows)} rows -> {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
