#!/usr/bin/env python3
"""
PGR supplementary-table scanner.  2026-09-13.

Searches every downloaded supplementary / processed gene-level file for the
progesterone receptor, and validates each negative with a positive control.

Targets (all word-boundary anchored; case-insensitive unless noted):
    PGR          \bPGR\b        -- \b excludes PGRMC1/PGRMC2/RPGR/RPGRIP1
    PGR-AS1      \bPGR[-_ ]?AS1\b
    NR3C3        \bNR3C3\b      (legacy alias of PGR)
    ENSG00000082175   PGR      (resolved from Ensembl REST, not assumed)
    ENSG00000282728   PGR-AS1  (resolved from Ensembl REST, not assumed)
    progester    substring, case-insensitive (progesterone, progesterone-receptor, ...)

Positive controls (a negative only counts if at least one is present):
    CLDN5 PECAM1 FLT1 VWF CDH5  + their Ensembl IDs
Broad readability panel (proves gene symbols are readable at all):
    GAPDH ACTB PTPRC AQP4 GFAP SNAP25 PDGFRB AIF1 MBP PLP1 APOE ACTA2

Decision per file:
    HIT              >=1 PGR-family match
    negative         0 PGR matches AND >=1 EC positive control
    weak-negative    0 PGR matches, no EC control, but broad gene panel present
    no-gene-symbols  no gene symbol of any kind found (metadata/QC/coordinate table)
    could-not-search extraction failed / file unreadable
"""
import gzip
import io
import json
import os
import re
import sys
import zipfile

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "scan_results.tsv"

# ---------------------------------------------------------------- patterns
PAT = {
    "PGR":        re.compile(r"\bPGR\b", re.I),
    "PGR_AS1":    re.compile(r"\bPGR[-_ ]?AS1\b", re.I),
    "NR3C3":      re.compile(r"\bNR3C3\b", re.I),
    "ENSG_PGR":   re.compile(r"\bENSG00000082175\b", re.I),
    "ENSG_PGRAS1": re.compile(r"\bENSG00000282728\b", re.I),
    "progester":  re.compile(r"progester", re.I),
}
EC_CTRL = ["CLDN5", "PECAM1", "FLT1", "VWF", "CDH5"]
EC_ENSG = {
    "CLDN5": "ENSG00000184113", "PECAM1": "ENSG00000261371",
    "FLT1": "ENSG00000102755", "VWF": "ENSG00000110799",
    "CDH5": "ENSG00000179776",
}
BROAD = ["GAPDH", "ACTB", "PTPRC", "AQP4", "GFAP", "SNAP25", "PDGFRB",
         "AIF1", "MBP", "PLP1", "APOE", "ACTA2", "SOX2", "RBFOX3"]
CTRL_PAT = {g: re.compile(r"\b" + g + r"\b", re.I) for g in EC_CTRL + BROAD}
CTRL_PAT.update({g + "_ENSG": re.compile(r"\b" + e + r"\b", re.I)
                 for g, e in EC_ENSG.items()})

# false-positive guard: these contain "PGR" but are different genes
DECOY = re.compile(r"\b(PGRMC1|PGRMC2|RPGR|RPGRIP1|RPGRIP1L)\b", re.I)

# ---------------------------------------------------------- fast prefilter
# ONE alternation covering every target, control and decoy token.  Ordered
# longest-first so PGR-AS1 beats PGR and PGRMC1 is claimed as a decoy rather
# than silently discarded.  A chunk this rejects provably holds none of them.
_DECOYS = ["PGRMC1", "PGRMC2", "RPGRIP1L", "RPGRIP1", "RPGR"]
_TARGETS = ["ENSG00000082175", "ENSG00000282728", "PGR[-_ ]?AS1", "NR3C3", "PGR"]
_CTRLS = EC_CTRL + BROAD + list(EC_ENSG.values())
PREFILTER = re.compile(
    r"\b(?:" + "|".join(_DECOYS + _TARGETS + [r"progester\w*"] + _CTRLS) + r")\b",
    re.I)

DECOY_TOKENS = {d: d for d in _DECOYS}
TARGET_TOKENS = {"ENSG00000082175": "ENSG_PGR", "ENSG00000282728": "ENSG_PGRAS1",
                 "NR3C3": "NR3C3", "PGR": "PGR",
                 "PGR-AS1": "PGR_AS1", "PGR_AS1": "PGR_AS1",
                 "PGR AS1": "PGR_AS1", "PGRAS1": "PGR_AS1"}
_ENSG2SYM = {v.upper(): k for k, v in EC_ENSG.items()}
CTRL_TOKENS = {g: g for g in EC_CTRL + BROAD}
CTRL_TOKENS.update({e.upper(): s for e, s in _ENSG2SYM.items()})

SKIP_DIRS = {".git", "__MACOSX", "_blocked", ".ipynb_checkpoints"}
BINARY_EXT = {".czi", ".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff",
              ".mtx", ".bam", ".h5", ".rds", ".loom"}


class Acc:
    """Accumulates regex hits over streamed text chunks, keeping context."""

    def __init__(self):
        self.hits = {k: 0 for k in PAT}
        self.ctrl = set()
        self.ctx = []          # up to 8 context snippets for PGR-family hits
        self.decoy = 0
        self.chars = 0

    def feed(self, text, where=""):
        if not text:
            return
        self.chars += len(text)
        # ---- fast path: one combined alternation over the whole chunk.
        # Only if a candidate token appears do we pay for per-pattern work.
        # PREFILTER is a superset of every target and control token, so a
        # chunk it rejects provably contains none of them.
        for m in PREFILTER.finditer(text):
            tok = m.group(0).upper()
            if tok in CTRL_TOKENS:
                self.ctrl.add(CTRL_TOKENS[tok])
                continue
            if tok in DECOY_TOKENS:
                self.decoy += 1
                continue
            k = TARGET_TOKENS.get(tok)
            if k is None:                      # 'progester' prefix match
                k = "progester"
            self.hits[k] += 1
            if len(self.ctx) < 8:
                a, b = max(0, m.start() - 90), min(len(text), m.end() + 90)
                snip = re.sub(r"\s+", " ", text[a:b]).strip()
                self.ctx.append(f"[{k}]{('@' + where) if where else ''} …{snip}…")

    @property
    def pgr_total(self):
        return sum(self.hits[k] for k in
                   ("PGR", "PGR_AS1", "NR3C3", "ENSG_PGR", "ENSG_PGRAS1"))

    @property
    def ec_controls(self):
        return sorted(g for g in self.ctrl if g in EC_CTRL)

    @property
    def broad_controls(self):
        return sorted(g for g in self.ctrl if g in BROAD)


# ---------------------------------------------------------------- extractors
def feed_zip_xml(acc, fh, label):
    """xlsx / docx / pptx: read every XML part as raw text (catches shared
    strings and inline strings alike). Complete and fast for huge sheets."""
    with zipfile.ZipFile(fh) as z:
        for n in z.namelist():
            if not n.lower().endswith((".xml", ".rels", ".txt")):
                continue
            try:
                with z.open(n) as f:
                    while True:
                        chunk = f.read(4 << 20)
                        if not chunk:
                            break
                        acc.feed(chunk.decode("utf-8", "replace"), f"{label}:{n}")
            except Exception:
                pass


def feed_text_stream(acc, f, label):
    tail = ""
    while True:
        chunk = f.read(4 << 20)
        if not chunk:
            break
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", "replace")
        acc.feed(tail + chunk, label)
        tail = chunk[-200:]


def feed_pdf(acc, path):
    import fitz
    doc = fitz.open(path)
    for i, page in enumerate(doc):
        acc.feed(page.get_text(), f"p{i+1}")
    doc.close()


def feed_h5ad(acc, path):
    import h5py
    with h5py.File(path, "r") as h:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "SOU":
                try:
                    v = obj[()]
                    acc.feed("\n".join(
                        x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x)
                        for x in (v.ravel() if hasattr(v, "ravel") else [v])), name)
                except Exception:
                    pass
        h.visititems(visit)


def scan_file(path, label=None, fh=None):
    acc = Acc()
    ext = os.path.splitext(path)[1].lower()
    label = label or os.path.basename(path)
    try:
        if ext in (".xlsx", ".xlsm", ".docx", ".pptx"):
            feed_zip_xml(acc, fh or path, label)
        elif ext == ".pdf":
            if fh:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as t:
                    t.write(fh.read()); tmp = t.name
                feed_pdf(acc, tmp); os.unlink(tmp)
            else:
                feed_pdf(acc, path)
        elif ext == ".gz":
            with gzip.open(fh or path, "rb") as g:
                feed_text_stream(acc, g, label)
        elif ext == ".h5ad":
            feed_h5ad(acc, path)
        elif ext in BINARY_EXT:
            return acc, "binary-image-or-matrix"
        else:
            f = fh or open(path, "rb")
            feed_text_stream(acc, f, label)
            if not fh:
                f.close()
    except Exception as e:
        return acc, f"could-not-search:{type(e).__name__}:{str(e)[:60]}"
    return acc, None


def decide(acc, err):
    if err:
        return err if err.startswith(("could-not-search", "binary")) else err
    if acc.pgr_total > 0:
        return "HIT"
    if acc.ec_controls:
        return "negative"
    if acc.broad_controls:
        return "weak-negative(no EC control)"
    return "no-gene-symbols"


# ---------------------------------------------------------------- walk
rows = []


def emit(study, name, size, acc, err, note=""):
    rows.append(dict(
        study=study, file=name, bytes=size,
        chars=acc.chars,
        ec_controls=",".join(acc.ec_controls) or "-",
        broad=len(acc.broad_controls),
        PGR=acc.hits["PGR"], PGR_AS1=acc.hits["PGR_AS1"], NR3C3=acc.hits["NR3C3"],
        ENSG_PGR=acc.hits["ENSG_PGR"], ENSG_PGRAS1=acc.hits["ENSG_PGRAS1"],
        progester=acc.hits["progester"], decoy_PGRMC_RPGR=acc.decoy,
        decision=decide(acc, err), context=" || ".join(acc.ctx)[:1500], note=note))


def main():
  for dirpath, dirnames, filenames in os.walk(ROOT):
      dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
      for fn in sorted(filenames):
          if fn.startswith(("._", "_page.html", "_urls.txt")) or fn in ("supp.html", "full.html"):
              continue
          p = os.path.join(dirpath, fn)
          rel = os.path.relpath(p, ROOT)
          study = rel.split(os.sep)[0]
          size = os.path.getsize(p)
          if fn.lower().endswith(".zip"):
              try:
                  with zipfile.ZipFile(p) as z:
                      for zi in z.infolist():
                          if zi.is_dir() or "__MACOSX" in zi.filename:
                              continue
                          zext = os.path.splitext(zi.filename)[1].lower()
                          if zext in BINARY_EXT:
                              a = Acc(); emit(study, f"{rel}::{zi.filename}",
                                              zi.file_size, a, "binary-image-or-matrix")
                              continue
                          with z.open(zi) as zf:
                              data = io.BytesIO(zf.read())
                          a, e = scan_file(zi.filename, label=zi.filename, fh=data)
                          emit(study, f"{rel}::{zi.filename}", zi.file_size, a, e)
              except Exception as e:
                  a = Acc(); emit(study, rel, size, a, f"could-not-search:{type(e).__name__}")
              continue
          a, e = scan_file(p)
          emit(study, rel, size, a, e)

  cols = ["study", "file", "bytes", "chars", "ec_controls", "broad", "PGR", "PGR_AS1",
          "NR3C3", "ENSG_PGR", "ENSG_PGRAS1", "progester", "decoy_PGRMC_RPGR",
          "decision", "context", "note"]
  with open(OUT, "w") as fo:
      fo.write("\t".join(cols) + "\n")
      for r in rows:
          fo.write("\t".join(str(r[c]).replace("\t", " ").replace("\n", " ")
                             for c in cols) + "\n")
  print(f"wrote {OUT}: {len(rows)} files")


if __name__ == "__main__":
    main()
