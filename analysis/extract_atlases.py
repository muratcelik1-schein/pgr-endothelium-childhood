# -*- coding: utf-8 -*-
"""PREPROCESSING 2/2. Builds the shipped tables from the Allen and PsychAD downloads.
Output: allen_fig.csv, psychad_fig.csv, psychad_control_fig.csv, psychad_donor.csv
Run:
  python3 extract_atlases.py --abc <abc_atlas directory> --clusters <steroidogenesis_cluster.tsv> --out .
  python3 extract_atlases.py --psychad <psychad h5ad directory> --vasc <pseudobulk directory> --out .
This script needs the raw downloads. Its outputs are shipped with the package, so
verify_all.py and make_figures.py run without it."""
import argparse, os, csv, collections, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--abc"); ap.add_argument("--clusters")
ap.add_argument("--psychad"); ap.add_argument("--vasc")
ap.add_argument("--out", default=".")
A = ap.parse_args()

# ---------- Allen: cluster-level expression and anatomical division
if A.abc and A.clusters:
    div = collections.defaultdict(collections.Counter)
    with open(f"{A.abc}/WHB-10Xv3/20241115/cell_metadata.csv") as fh:
        for r in csv.DictReader(fh):
            div[r["cluster_alias"]][r["anatomical_division_label"]] += 1
    GEN = ["PGR", "HSD3B1", "HSD3B2", "HSD3B7", "CYP19A1", "CYP11A1", "STAR"]
    with open(f"{A.out}/allen_fig.csv", "w", newline="") as o:
        w = csv.writer(o); w.writerow(["alias", "n_cell", "division", "cls"] + GEN)
        for r in csv.DictReader(open(A.clusters), delimiter="\t"):
            if int(r["n_cell"]) < 50: continue          # cluster-level comparison threshold
            a = r["cluster_alias"]
            dv = div[a].most_common(1)[0][0] if a in div else "?"
            snap, cldn, dcn = float(r["SNAP25"]), float(r["CLDN5"]), float(r["DCN"])
            cls = "neuronal" if snap > 2.0 else ("vascular" if (cldn > 0.5 or dcn > 1.0) and snap < 1.0 else "other")
            w.writerow([a, r["n_cell"], dv, cls] + [r[g] for g in GEN])
    print("allen_fig.csv written")

# ---------- PsychAD: four vascular cell types
if A.psychad and A.vasc:
    import h5py
    def cat(f, p):
        g = f["obs"][p]
        if isinstance(g, h5py.Group) and "codes" in g:
            c = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]])
            return c[g["codes"][:]]
        return np.array([x.decode() if isinstance(x, bytes) else x for x in g[:]])
    DX = {}
    for coh in ["Aging", "HBCC", "MSSM", "RADC"]:
        p = f"{A.psychad}/psychad_{coh}.h5ad"
        if not os.path.exists(p): continue
        f = h5py.File(p, "r"); d = cat(f, "donor_id"); s = cat(f, "disease_ontology_term_id")
        for i in range(0, len(d), 53): DX.setdefault(d[i], s[i])
        f.close()
    sym = np.array([str(s) for s in np.load(f"{A.vasc}/genes.npy", allow_pickle=True)])
    gi = {}
    for i, g in enumerate(sym): gi.setdefault(g, i)
    GEN = ["PGR", "PAQR5", "GPER1", "ESR1", "AR", "NR3C1", "NR3C2", "AKR1C1", "AKR1C2", "AKR1C3",
           "CYP1B1", "SRD5A1", "ABCB1", "PECAM1", "CLDN5", "PDGFRB", "RGS5", "ACTA2", "MYH11",
           "DCN", "COL1A2", "LUM"]
    CT, COH = ["Endo", "PC", "SMC", "VLMC"], ["Aging", "HBCC", "MSSM", "RADC"]
    def load(coh, ct):
        p = f"{A.vasc}/{coh}_{ct}_pb.npy"
        if not os.path.exists(p): return None, None
        M = np.load(p)
        rows = [l.rstrip("\n").split("\t") for l in open(f"{A.vasc}/{coh}_{ct}_meta.tsv")][1:]
        return M, rows
    for tag, ctrl in [("psychad_fig.csv", False), ("psychad_control_fig.csv", True)]:
        res, nn = {}, {}
        for ct in CT:
            vs = {g: [] for g in GEN}; tot = 0
            for coh in COH:
                M, rows = load(coh, ct)
                if M is None: continue
                n = np.array([int(r[3]) for r in rows]); dnm = [r[0] for r in rows]
                k = n >= 20
                if ctrl: k = k & np.array([DX.get(d, "?") == "PATO:0000461" for d in dnm])
                if k.sum() < (15 if ctrl else 20): continue
                CP = M[k] / np.maximum(M[k].sum(1, keepdims=True), 1) * 1e4; tot += int(k.sum())
                for g in GEN: vs[g].append(CP[:, gi[g]].mean())
            res[ct] = {g: (np.mean(v) if v else float("nan")) for g, v in vs.items()}; nn[ct] = tot
        with open(f"{A.out}/{tag}", "w", newline="") as o:
            w = csv.writer(o); w.writerow(["gene"] + CT)
            for g in GEN: w.writerow([g] + [f"{res[c][g]:.4f}" for c in CT])
            if ctrl: o.write("#donor," + ",".join(f"{c}={nn[c]}" for c in CT) + "\n")
        print(tag, "written", nn)
    with open(f"{A.out}/psychad_donor.csv", "w", newline="") as o:
        w = csv.writer(o); w.writerow(["cohort", "celltype", "age", "PGR", "ZNF385B"])
        for ct in ["Endo", "PC", "SMC"]:
            for coh in COH:
                M, rows = load(coh, ct)
                if M is None: continue
                n = np.array([int(r[3]) for r in rows])
                age = np.array([float(m.group(1)) if (m := re.search(r"(\d+)", r[2])) else np.nan for r in rows])
                k = (n >= 20) & np.isfinite(age)
                if k.sum() < 20: continue
                CP = M[k] / np.maximum(M[k].sum(1, keepdims=True), 1) * 1e4
                for a_, x, y in zip(age[k], CP[:, gi["PGR"]], CP[:, gi["ZNF385B"]]):
                    w.writerow([coh, ct, a_, round(float(x), 5), round(float(y), 5)])
    print("psychad_donor.csv written")
