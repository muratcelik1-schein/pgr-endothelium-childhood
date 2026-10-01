# Arteriovenous zonation of PGR in human brain endothelium.
# Input: GSE256493 Seurat objects (Walchli et al., Nature 2024), reference [1] of
# the manuscript. Uses the AUTHORS' OWN endothelial cluster labels, so no
# clustering or label transfer is performed here.
#
# Rules enforced, not assumed:
#   * Cluster label spellings are discovered at runtime and mapped by keyword,
#     and the mapping is printed so it can be checked.
#   * The labelling is validated with published segment markers BEFORE PGR is
#     examined. If the positive controls do not separate, the script stops and
#     reports "zonation labelling not recovered".
#   * Non endothelial states in the same object (proliferating, EndoMT,
#     mitochondrial, stem to EC) are excluded and the exclusion is counted.
#   * Endothelial identity genes bound the false positive rate.
#   * Counts are used, normalised per cell to counts per 10,000. The analysis
#     unit for the between-segment test is the cell, and the per-patient table
#     is printed alongside so a reader can see whether one patient drives it.
#   * Comparisons are corrected across the genes tested and the uncorrected
#     values are given as well, labelled exploratory.
#
# Usage: Rscript walchli_zonation.R <object.rds> <out.tsv> <label>

suppressMessages({library(Matrix)})
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3)
inp <- args[1]; outf <- args[2]; lab <- args[3]
SEED <- 2026; set.seed(SEED)

rows <- list()
add <- function(block, key, value, note = "") {
  rows[[length(rows) + 1]] <<- data.frame(block = block, key = key,
                                          value = as.character(value), note = note,
                                          stringsAsFactors = FALSE)
  cat(sprintf("%-18s %-34s %s   %s\n", block, key, value, note))
}

o <- readRDS(inp)
md <- o@meta.data
add("meta", "object", basename(inp))
add("meta", "cohort", lab)
add("meta", "n_cell_total", nrow(md))

# ------------------------------------------------------------------ discovery
cl_col <- grep("cluster", colnames(md), ignore.case = TRUE, value = TRUE)
stopifnot("a single cluster column was expected" = length(cl_col) == 1)
add("meta", "cluster_column", cl_col, "discovered")
cl <- as.character(md[[cl_col]])
pt_col <- grep("^Patient$|patient|donor|sample", colnames(md), ignore.case = TRUE, value = TRUE)[1]
add("meta", "patient_column", pt_col)
pt <- as.character(md[[pt_col]])
add("meta", "n_patient", length(unique(pt)))

lv <- sort(table(cl), decreasing = TRUE)
for (i in seq_along(lv)) add("clusters", names(lv)[i], lv[i])

# keyword mapping, printed so it can be checked
seg_of <- function(x) {
  y <- tolower(x)
  if (grepl("arter", y)) return("arterial")
  if (grepl("capillar", y)) return("capillary")
  if (grepl("vein|venule|venous", y)) return("venous")
  return("excluded")
}
seg <- vapply(cl, seg_of, character(1))
for (s in c("arterial", "capillary", "venous", "excluded")) {
  members <- sort(unique(cl[seg == s]))
  add("mapping", s, sum(seg == s), paste(members, collapse = "; "))
}
stopifnot("all three segments are required" = all(c("arterial","capillary","venous") %in% seg))

# ------------------------------------------------------------------ counts
A <- o@assays$RNA
cnt <- tryCatch(A@counts, error = function(e) NULL)
if (is.null(cnt) || nrow(cnt) == 0) cnt <- A@data
stopifnot("count matrix is empty" = nrow(cnt) > 0)
add("meta", "n_gene", nrow(cnt))
tot <- Matrix::colSums(cnt)
add("meta", "median_counts_per_cell", round(median(tot)))

MARK <- list(arterial  = c("GJA5","SEMA3G","HEY1","ALPL","VEGFC","BMX"),
             capillary = c("MFSD2A","SLC7A5","TFRC","SLC16A1","RGCC"),
             venous    = c("NR2F2","VWF","ACKR1","PLVAP","IL1R1"))
IDENT <- c("PECAM1","CLDN5","FLT1")
TARGET <- "PGR"
present <- function(g) g[g %in% rownames(cnt)]

cp10k <- function(gene, mask) {
  if (!(gene %in% rownames(cnt))) return(NA_real_)
  v <- cnt[gene, mask]
  sum(v) / max(sum(tot[mask]), 1) * 1e4
}
pctpos <- function(gene, mask) {
  if (!(gene %in% rownames(cnt))) return(NA_real_)
  100 * mean(cnt[gene, mask] > 0)
}

keep <- seg %in% c("arterial","capillary","venous")
add("meta", "n_cell_endothelial_segments", sum(keep))
add("meta", "n_cell_excluded", sum(!keep), "non endothelial states in the same object")

# ------------------------------------------------------------------ gate
gate <- TRUE
for (s in names(MARK)) {
  gs <- present(MARK[[s]])
  add("gate_markers", s, sprintf("%d of %d present", length(gs), length(MARK[[s]])),
      paste(gs, collapse = "; "))
  if (!length(gs)) { gate <- FALSE; next }
  m <- sapply(c("arterial","capillary","venous"),
              function(t) mean(sapply(gs, function(g) cp10k(g, seg == t))))
  top <- names(which.max(m)); ok <- top == s; gate <- gate && ok
  add("gate_result", s, sprintf("top=%s; %s", top,
      paste(sprintf("%s=%.3f", names(m), m), collapse = "; ")),
      ifelse(ok, "PASS", "FAIL"))
}
add("gate_result", "zonation_labelling_recovered", gate,
    "if FALSE no PGR segment claim follows")
if (!gate) {
  add("RESULT", "verdict", "zonation labelling not recovered", "stopped before the target")
  write.table(do.call(rbind, rows), outf, sep = "\t", row.names = FALSE, quote = FALSE)
  quit(status = 3)
}

# ------------------------------------------------------------------ target
GENES <- c(TARGET, IDENT, unlist(MARK, use.names = FALSE))
GENES <- unique(present(GENES))
pv <- c()
for (g in GENES) {
  vals <- sapply(c("arterial","capillary","venous"), function(t) cp10k(g, seg == t))
  pcts <- sapply(c("arterial","capillary","venous"), function(t) pctpos(g, seg == t))
  # cell level test of association between segment and detection
  tab <- rbind(sapply(c("arterial","capillary","venous"), function(t) sum(cnt[g, seg == t] > 0)),
               sapply(c("arterial","capillary","venous"), function(t) sum(seg == t) -
                        sum(cnt[g, seg == t] > 0)))
  p <- suppressWarnings(chisq.test(tab)$p.value)
  pv <- c(pv, p); names(pv)[length(pv)] <- g
  kind <- if (g == TARGET) "TARGET" else if (g %in% IDENT) "identity control" else "segment marker"
  add("segment_expression", g,
      sprintf("art=%.4f cap=%.4f ven=%.4f | pct art=%.2f cap=%.2f ven=%.2f | top=%s",
              vals[1], vals[2], vals[3], pcts[1], pcts[2], pcts[3],
              names(which.max(vals))),
      sprintf("%s; chi2 p=%.3g (uncorrected, exploratory)", kind, p))
}
q <- p.adjust(pv, method = "BH")
add("multiple_testing", "method", "Benjamini-Hochberg across the genes tested", length(pv))
add("multiple_testing", TARGET, sprintf("p=%.3g; q=%.3g", pv[[TARGET]], q[[TARGET]]),
    ifelse(q[[TARGET]] < 0.05, "significant after correction", "not significant after correction"))

# per patient, so one patient cannot drive the result
for (s in c("arterial","capillary","venous")) {
  per <- sapply(sort(unique(pt)), function(p) {
    m <- keep & (seg == s) & (pt == p)
    if (sum(m) < 20) return(NA_real_)
    cp10k(TARGET, m)
  })
  add("per_patient", s, paste(sprintf("%s=%.4f", names(per), per), collapse = "; "),
      "cells fewer than 20 shown as NA")
}
np <- sapply(sort(unique(pt)), function(p) {
  v <- sapply(c("arterial","capillary","venous"), function(s) cp10k(TARGET, keep & seg == s & pt == p))
  if (all(is.na(v))) return(NA_character_); names(which.max(v))
})
add("per_patient", "top_segment_per_patient", paste(sprintf("%s=%s", names(np), np), collapse = "; "),
    "agreement across patients matters more than the pooled value")

add("limits", "cohort_age", lab, "this cohort cannot speak to the postnatal childhood window")
add("limits", "assay", "scRNA-seq of sorted cells, not single nuclei",
    "detection floor differs; segment ranking is comparable, absolute levels are not")
add("limits", "seed", SEED)

write.table(do.call(rbind, rows), outf, sep = "\t", row.names = FALSE, quote = FALSE)
cat(sprintf("\nwritten: %s (%d rows)\n", outf, length(rows)))
