# GSE280569 (Steyn 2024) -> donor-level endothelial pseudobulk, RNA assay
suppressMessages({library(Seurat); library(Matrix)})
a <- commandArgs(trailingOnly=TRUE)
o <- readRDS(a[1]); md <- o@meta.data
M <- GetAssayData(o, assay="RNA", layer="counts")
sel <- which(md$broad_cell_type == "Endothelial Cells")
don <- as.character(md$biological_replicate)
age <- as.numeric(sub("_yr_old.*","",don))
sex <- as.character(md$sex); bat <- as.character(md$batch)
GEN <- c("PGR","PGR-AS1","ZNF385B","ABCB1","CLDN5","PECAM1","FLT1","MFSD2A","SLC2A1",
         "NR3C1","NR3C2","ESR1","AR","GPER1","ACTB","GAPDH","RPL13A","RPLP0","TBP","PGK1")
GEN <- GEN[GEN %in% rownames(M)]
cat("genes:", paste(GEN, collapse=" "), "\n")
out <- do.call(rbind, lapply(unique(don[sel]), function(d) {
  j <- sel[don[sel] == d]
  s <- M[, j, drop=FALSE]
  r <- data.frame(donor=d, age=age[j][1], sex=sex[j][1], batch=bat[j][1],
                  n_nuclei=length(j), total_counts=sum(s),
                  n_pgr_pos=sum(s["PGR", ] > 0))
  for (g in GEN) r[[g]] <- sum(s[g, ])
  r
}))
out <- out[order(out$age), ]
write.csv(out, file.path(dirname(a[1]), "steyn_endo_pseudobulk.csv"), row.names=FALSE)
print(out[, c("donor","age","sex","n_nuclei","total_counts","n_pgr_pos","PGR","CLDN5","ACTB")])
# PGR across all cell types: localisation replication
cat("\n--- PGR by cell type (all donors pooled, counts per 10,000) ---\n")
for (ct in unique(md$broad_cell_type)) {
  k <- which(md$broad_cell_type == ct)
  s <- M[, k, drop=FALSE]
  cat(sprintf("  %-20s n=%6d  PGR=%.4f  positive=%.2f%%  CLDN5=%.3f\n",
      ct, length(k), sum(s["PGR",])/sum(s)*1e4, 100*mean(s["PGR",]>0), sum(s["CLDN5",])/sum(s)*1e4))
}
cat("yazildi\n")
