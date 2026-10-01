#!/usr/bin/env python3
"""Three more axes on the same harness: tissue, species, disease.

Imports the registry, the client and the controls from dimension_scanner.py
so the discipline is identical: every axis declares a positive control
before it runs, an axis whose control fails says SCAN INVALID and makes no
statement, and every axis carries its own limit.

  tissue    Is the target's endothelial bias a property of brain, or does it
            hold wherever the corpus has endothelium? The manuscript's claim
            is about the brain wall; a bias present in every tissue would
            mean the brain result is not about brain.

  species   The manuscript separates human from rodent on four axes, one of
            them cell type, and cites rodent reports for it. The corpus
            carries mouse. This measures that axis instead of citing it.

  disease   Does the endothelial bias survive in diseased tissue? The
            control genes bound how much of any shift is technical.

Writes analysis/dimension2_numbers.tsv.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dimension_scanner import (  # noqa: E402
    BRAIN, GENE_BATCH, HUMAN, POSITIVE_CONTROLS, SEED, TARGET, WMG,
    Axis, _get, corpus_expression, dimensions, endothelial_rank,
    specificity_profiles,
)

MOUSE = "NCBITaxon:10090"
MOUSE_TARGET = "Pgr"
MOUSE_CONTROLS = ["Cldn5", "Pecam1"]
MOUSE_RELATED = ["Gper1", "Nr3c1", "Nr3c2", "Esr1", "Ar"]

# Diseases the corpus annotates that a vessel-wall paper would ask about.
# MONDO terms, resolved from the corpus rather than typed from memory where
# possible; each is checked for a non-empty return before it is scored.
DISEASES = {
    "MONDO:0005090": "schizophrenia",
    "MONDO:0004975": "Alzheimer disease",
    "MONDO:0005147": "type 1 diabetes mellitus",
    "MONDO:0007254": "breast cancer",
}

NULL_SIZE_2 = 250
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "analysis", "dimension2_numbers.tsv")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mouse_dimensions():
    d = _get(f"{WMG}/primary_filter_dimensions")
    genes = {}
    for entry in d["gene_terms"][MOUSE]:
        for ensm, sym in entry.items():
            genes.setdefault(sym, ensm)
    return genes


def corpus_expression_filtered(ensgs, organism=HUMAN, disease=None):
    """Same call as the harness, with an optional organism and disease."""
    out, labels = {}, {}
    for i in range(0, len(ensgs), GENE_BATCH):
        chunk = ensgs[i:i + GENE_BATCH]
        filt = {"gene_ontology_term_ids": chunk,
                "organism_ontology_term_id": organism}
        if disease:
            filt["disease_ontology_term_ids"] = [disease]
        d = _get(f"{WMG}/query", {"filter": filt, "is_rollup": True})
        if "expression_summary" not in d:
            return {}, {}
        out.update(d["expression_summary"])
        cts = (d.get("term_id_labels") or {}).get("cell_types") or {}
        for _t, per_ct in cts.items():
            if not isinstance(per_ct, dict):
                continue
            for ct, rec in per_ct.items():
                if ct == "tissue_stats" or not isinstance(rec, dict):
                    continue
                name = (rec.get("aggregated") or {}).get("name")
                if name:
                    labels.setdefault(ct, name)
    return out, labels


def ranks_for(expr, ensg_list, tissue, lab, target_ensg, control_ensgs,
              null_ensgs):
    """Endothelial rank fraction for target, controls and the null."""
    spec, _raw, baseline = specificity_profiles(expr, ensg_list, tissue)
    tgt = endothelial_rank(spec.get(target_ensg) or {}, lab)
    ctl = [endothelial_rank(spec.get(g) or {}, lab) for g in control_ensgs]
    null = [r for r in (endothelial_rank(spec.get(g) or {}, lab)
                        for g in null_ensgs) if r is not None]
    return tgt, [c for c in ctl if c is not None], null, len(baseline)


def p_of(rank, null):
    if rank is None or not null:
        return None
    return (sum(1 for r in null if r <= rank) + 1) / (len(null) + 1)


# ----------------------------------------------------------------- axis three

class TissueAxis(Axis):
    name = "tissue"
    question = ("Is the endothelial bias a property of brain, or does it hold "
                "wherever the corpus has endothelium?")
    limit = ("Each tissue has its own baseline and its own contributing "
             "datasets, so tissues are compared on the rank statistic and "
             "never on the specificity value. A tissue where no endothelial "
             "cell type was scored is reported as untestable.")

    def control(self):
        ok = sum(1 for c in self.ctx["brain_ctl"] if c is not None and c < 0.1)
        return ok == len(POSITIVE_CONTROLS), (
            f"{ok} of {len(POSITIVE_CONTROLS)} identity genes place "
            f"endothelium in the top tenth of their own ranking in brain: "
            + ", ".join(f"{s}={c:.3f}" for s, c in
                        zip(POSITIVE_CONTROLS, self.ctx["brain_ctl"])))

    def scan(self):
        rows = self.ctx["per_tissue"]
        self.add("tissues_scanned", len(rows))
        testable = {t: v for t, v in rows.items() if v["target"] is not None}
        self.add("tissues_testable", len(testable),
                 "an endothelial cell type was scored for the target")
        biased = {t: v for t, v in testable.items()
                  if v["p"] is not None and v["p"] < 0.05}
        self.add("tissues_endothelial_biased", f"{len(biased)} of {len(testable)}")
        for t, v in sorted(testable.items(), key=lambda kv: kv[1]["target"]):
            self.add(f"{v['label']}",
                     f"rank={v['target']:.3f} p={v['p']:.3f} "
                     f"null={len(v['null'])} cell_types={v['n_ct']}",
                     "biased" if v["p"] < 0.05 else "")
        self.add("reading",
                 "a bias confined to a few tissues supports a claim about "
                 "the brain wall; a bias in every tissue would mean the "
                 "brain result is about the gene, not about brain")


# ------------------------------------------------------------------ axis four

class SpeciesAxis(Axis):
    name = "species"
    question = ("The manuscript separates human from rodent on cell type and "
                "cites rodent reports. Does the corpus mouse data agree?")
    limit = ("Mouse coverage in the corpus is smaller and its brain datasets "
             "differ in age and assay from the human ones. This compares "
             "where the transcript sits in each species, not when it "
             "appears, and an absence in mouse bounded by low coverage is "
             "reported as such.")

    def control(self):
        ctl = self.ctx["mouse_ctl"]
        ok = sum(1 for c in ctl if c is not None and c < 0.1)
        if not self.ctx["mouse_null"]:
            return False, "mouse null empty; coverage too low to score"
        return ok == len(MOUSE_CONTROLS), (
            f"{ok} of {len(MOUSE_CONTROLS)} mouse identity genes in the top "
            f"tenth: " + ", ".join(
                f"{s}=" + (f"{c:.3f}" if c is not None else "unscored")
                for s, c in zip(MOUSE_CONTROLS, ctl)))

    def scan(self):
        null = sorted(self.ctx["mouse_null"])
        self.add("mouse_null_size", len(null), f"resolution 1/{len(null) + 1}")
        self.add("mouse_null_median", f"{null[len(null) // 2]:.3f}")
        self.add("mouse_cell_types", self.ctx["mouse_n_ct"])
        t = self.ctx["mouse_target"]
        if t is None:
            self.add(f"{MOUSE_TARGET}::endothelial_rank",
                     "no endothelial cell type scored",
                     "untestable in mouse at current coverage, not negative")
        else:
            p = p_of(t, null)
            self.add(f"{MOUSE_TARGET}::endothelial_rank_fraction",
                     f"{t:.3f}", f"p={p:.3f}")
            self.add(f"{MOUSE_TARGET}::top_cell_type",
                     self.ctx["mouse_top"] or "-")
        h = self.ctx["human_target"]
        self.add("human_for_comparison", f"{h:.3f}" if h is not None else "-",
                 "same statistic, human brain")
        for sym, r in self.ctx["mouse_related"].items():
            self.add(f"{sym}::endothelial_rank",
                     f"{r:.3f}" if r is not None else "unscored")
        self.add("reading",
                 "a mouse rank far from the human one measures the species "
                 "difference the manuscript currently cites; an unscored "
                 "mouse target measures the corpus, not the mouse")


# ------------------------------------------------------------------ axis five

class DiseaseAxis(Axis):
    name = "disease"
    question = ("Does the endothelial bias survive in diseased tissue, and "
                "how much of any shift is technical?")
    limit = ("Each disease subset is a different set of datasets, so a shift "
             "confounds disease with cohort. The identity genes are measured "
             "in the same subsets and bound that confound: a shift they "
             "share is not about the target.")

    def control(self):
        got = [d for d, v in self.ctx["per_disease"].items()
               if v.get("ctl") and all(c is not None for c in v["ctl"])]
        return len(got) >= 2, (
            f"identity genes scored in {len(got)} of "
            f"{len(self.ctx['per_disease'])} disease subsets")

    def scan(self):
        base = self.ctx["human_target"]
        self.add("unfiltered_target_rank",
                 f"{base:.3f}" if base is not None else "-")
        for dis, v in self.ctx["per_disease"].items():
            label = DISEASES[dis]
            if v["target"] is None:
                self.add(label, "target unscored in this subset",
                         "untestable, not negative")
                continue
            ctl = ", ".join(f"{s}={c:.3f}" for s, c in
                            zip(POSITIVE_CONTROLS, v["ctl"])
                            if c is not None)
            self.add(label,
                     f"target={v['target']:.3f} p={v['p']:.3f} "
                     f"controls[{ctl}] null={len(v['null'])}")
        self.add("reading",
                 "the target and the identity genes moving together is a "
                 "cohort effect; the target moving alone is a candidate "
                 "disease effect and would need its own cohort to confirm")


# ----------------------------------------------------------------------- main

def main():
    rng = random.Random(SEED)
    print("Human corpus index")
    hgenes, snapshot = dimensions()
    hpanel = [TARGET] + POSITIVE_CONTROLS
    hpool = sorted(set(hgenes) - set(hpanel))
    hnull = rng.sample(hpool, NULL_SIZE_2)
    hensgs = [hgenes[s] for s in hpanel + hnull]

    print(f"  querying {len(hensgs)} human genes")
    hexpr, hlabels = corpus_expression(hensgs)

    def lab(ct):
        return hlabels.get(ct, ct)

    # ---- brain reference, and every tissue the response carries
    tissues = set()
    for g in hexpr.values():
        tissues |= set(g)
    print(f"  response carries {len(tissues)} tissues")

    tgt_ensg = hgenes[TARGET]
    ctl_ensgs = [hgenes[s] for s in POSITIVE_CONTROLS]
    null_ensgs = [hgenes[s] for s in hnull]

    per_tissue = {}
    for t in sorted(tissues):
        tg, ctl, null, n_ct = ranks_for(hexpr, hensgs, t, lab, tgt_ensg,
                                        ctl_ensgs, null_ensgs)
        if len(null) < 20:
            continue
        per_tissue[t] = {"target": tg, "ctl": ctl, "null": null,
                         "n_ct": n_ct, "p": p_of(tg, null),
                         "label": lab(t) if lab(t) != t else t}
    # tissue labels come from the tissue map, not the cell type map
    tmap = _get(f"{WMG}/primary_filter_dimensions")["tissue_terms"][HUMAN]
    tnames = {}
    for e in tmap:
        tnames.update(e)
    for t in per_tissue:
        per_tissue[t]["label"] = tnames.get(t, t)

    brain_tgt, brain_ctl, brain_null, _n = ranks_for(
        hexpr, hensgs, BRAIN, lab, tgt_ensg, ctl_ensgs, null_ensgs)
    brain_ctl_full = [endothelial_rank(
        specificity_profiles(hexpr, hensgs, BRAIN)[0].get(g) or {}, lab)
        for g in ctl_ensgs]

    # ---- mouse
    print("Mouse corpus index")
    mgenes = mouse_dimensions()
    mpanel = [MOUSE_TARGET] + MOUSE_CONTROLS + MOUSE_RELATED
    mpresent = [s for s in mpanel if s in mgenes]
    mpool = sorted(set(mgenes) - set(mpanel))
    mnull = rng.sample(mpool, NULL_SIZE_2)
    mensgs = [mgenes[s] for s in mpresent + mnull]
    print(f"  querying {len(mensgs)} mouse genes")
    mexpr, mlabels = corpus_expression_filtered(mensgs, organism=MOUSE)

    def mlab(ct):
        return mlabels.get(ct, ct)

    mspec, _r, mbase = specificity_profiles(mexpr, mensgs, BRAIN)
    m_tgt = endothelial_rank(mspec.get(mgenes[MOUSE_TARGET]) or {}, mlab)
    m_ctl = [endothelial_rank(mspec.get(mgenes[s]) or {}, mlab)
             for s in MOUSE_CONTROLS if s in mgenes]
    m_null = [r for r in (endothelial_rank(mspec.get(mgenes[s]) or {}, mlab)
                          for s in mnull) if r is not None]
    m_prof = mspec.get(mgenes[MOUSE_TARGET]) or {}
    m_top = mlab(max(m_prof, key=m_prof.get)) if m_prof else None
    m_related = {s: endothelial_rank(mspec.get(mgenes[s]) or {}, mlab)
                 for s in MOUSE_RELATED if s in mgenes}

    # ---- disease
    print("Disease subsets")
    per_disease = {}
    for dis, name in DISEASES.items():
        dexpr, dlabels = corpus_expression_filtered(hensgs, disease=dis)
        if not dexpr:
            per_disease[dis] = {"target": None, "ctl": [], "null": [],
                                "p": None}
            print(f"  {name}: empty return")
            continue

        def dlab(ct, _m=dlabels):
            return _m.get(ct, hlabels.get(ct, ct))
        tg, ctl, null, _n = ranks_for(dexpr, hensgs, BRAIN, dlab, tgt_ensg,
                                      ctl_ensgs, null_ensgs)
        per_disease[dis] = {"target": tg, "ctl": ctl, "null": null,
                            "p": p_of(tg, null)}
        print(f"  {name}: target={tg} null={len(null)}")

    ctx = {"per_tissue": per_tissue, "brain_ctl": brain_ctl_full,
           "human_target": brain_tgt,
           "mouse_target": m_tgt, "mouse_ctl": m_ctl, "mouse_null": m_null,
           "mouse_top": m_top, "mouse_related": m_related,
           "mouse_n_ct": len(mbase),
           "per_disease": per_disease}

    rows = [("meta", "snapshot", snapshot, ""),
            ("meta", "seed", SEED, ""),
            ("meta", "null_size_drawn", NULL_SIZE_2, "per organism"),
            ("meta", "tissues_with_usable_null", len(per_tissue),
             "a tissue needs at least twenty null genes scored")]

    for cls in (TissueAxis, SpeciesAxis, DiseaseAxis):
        rows += cls(ctx).run()

    with open(OUT, "w") as fh:
        fh.write("axis\tkey\tvalue\tnote\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    print(f"\nwrote {len(rows)} rows -> {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
