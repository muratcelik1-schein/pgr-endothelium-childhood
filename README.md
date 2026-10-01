# Progesterone receptor expression in human cortical endothelium increases across childhood

Analysis code and derived tables for the article of the same title (Çelik, Çiçek, Karip, Şakiroğlu, Velioğlu, Kadak; manuscript submitted for publication). Released under the MIT licence.

The code regenerates every number in the article and its Supplementary Information and draws the two main figures and the five supplementary figures. The primary data are public resources named in the Data availability statement of the article; this repository holds the code that extracts them and the derived tables that the analyses and figures read.

## Contents

| Path | Contents |
|---|---|
| `analysis/` | Python and R scripts together with the derived tables they read and write |
| `requirements.txt` | Pinned Python package versions |
| `LICENSE` | MIT licence |
| `CITATION.cff` | Citation metadata |

All code is released under the MIT licence. Python 3.9.6 and R 4.3; Python package versions are pinned in `requirements.txt`. The scripts regenerate every number reported in the manuscript and the Supplementary Information and draw the two main figures and the five supplementary figures.

## Run order

Run every command from the `analysis/` directory.

```
pip install -r ../requirements.txt
```

### 1. External inputs

The single-nucleus file of the developmental atlas is `devbrain_jointanalysis_07072026.h5ad`, Zenodo https://doi.org/10.5281/zenodo.21375950, md5 d945ddfed95bf0c9ae8a27bb58afdc78; save it as `data/devbrain.h5ad`. The bulk expression table is the GTEx v8 gene median TPM release:

```
curl -o data/gtex_med.gct.gz https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz
```

### 2. Derived matrices

```
python3 extract_devbrain.py --h5ad data/devbrain.h5ad --out data
```

Writes the endothelial pseudobulk matrix, the donor table, the gene symbols and the cell type matrices into `data/`, together with `donor_pgr_counts.csv`; a copy of that file is shipped in this directory and read by the later steps. The cluster-level tables in this directory were built by `extract_atlases.py` (Allen and PsychAD downloads) and the replication tables by `extract_steyn.R` (GSE280569, R and Seurat). Both sets are included in this directory, so steps 4 and 5 run without those downloads.

### 3. Every reported number

```
python3 verify_all.py --data data --out .
```

Writes `source_numbers_v2.tsv`.

### 4. Secondary analyses

These need the developmental atlas and the GEO downloads named in the Data availability statement. Each script asserts the column names of its input and writes one table.

```
python3 inspect_h5ad.py --h5ad <file> --out <report>
python3 primary_series.py --h5ad devbrain_jointanalysis_07072026.h5ad --out primary_series.tsv
python3 clarence_replication.py --h5ad clarence_rna.h5ad --out clarence_numbers.tsv
python3 composition_and_glia.py --h5ad clarence_rna.h5ad --out composition_glia.tsv
python3 depth_controlled_detection.py --h5ad clarence_rna.h5ad --celltype-col cell_type --celltype "endothelial cell" --donor-col donor_id --age-col development_stage --out clarence_depth.tsv
Rscript walchli_zonation.R <GSE256493 object>.rds walchli_adult.tsv adult
python3 psychad_lifespan.py --dir <psychad h5ad directory> --out psychad_lifespan.tsv
python3 pgr_targetome.py --h5ad devbrain_jointanalysis_07072026.h5ad --peaks GSM1071297_PR_PR_infected+Prog_peaks.xlsx --out pgr_targetome.tsv
python3 regional_table.py --out .
python3 brainspan_axis.py
python3 brainspan_tables.py
python3 amplitude_rank.py --pb pseudobulk.npz --groups groups.tsv --genes genes.tsv --out amplitude_rank.tsv
python3 genome_scan.py --pb pseudobulk.npz --groups groups.tsv --genes genes.tsv --out genome_scan.tsv
python3 nr4a1_program.py --pb pseudobulk.npz --groups groups.tsv --genes genes.tsv --out nr4a1_program.tsv
python3 gene_list.py --h5ad <developmental atlas> --out gene_list.tsv
python3 prior_scan.py <downloaded supplementary files> prior_scan.tsv
python3 ambient_doublet.py --h5ad <developmental atlas> --out ambient_doublet.tsv
python3 donor_level.py
python3 segment_null.py --h5ad <developmental atlas> --peaks <PR ChIP peak table> --perm-map <permeability map> --out segment_null.tsv
python3 age_axis.py
```

`brainspan_axis.py` and `brainspan_tables.py` read the BrainSpan RNA sequencing release (gene-level RPKM, `columns_metadata.csv`, `rows_metadata.csv`) from `data/brainspan/` and render Supplementary Tables S20 and S21. The additional analyses of the Supplementary Results are run by `dimension_scanner.py`, `dimension_scanner_2.py`, `corpus_null_sensitivity.py`, `eqtl_scan.py` and `gwas_locus.py`; each queries a public interface, caches the response and stops if its positive control fails.

### 5. Figures

```
python3 make_figures.py --out ../figures
python3 make_figures_2.py --out ../figures
```

`make_figures.py` draws Figure 1 and Supplementary Figure S1 from the derived tables in this directory (`donor_pgr_counts.csv`, `allen_fig.csv`, `psychad_fig.csv`). `make_figures_2.py` draws Figure 2 and Supplementary Figures S2 to S5 from the derived tables in this directory alone, so these figures are reproduced without any download. Both scripts check every value they draw against the tables before drawing and write the drawn values to `figure_source_data.tsv`.

## Expected state

A clean run leaves every table and every PNG byte identical. The permutation is seeded at 2026 and returns identical values across runs and across values of PYTHONHASHSEED.

## Derived tables

The tables the analysis code reads, and the table of every number reported in the manuscript and the Supplementary Information. Expression is in counts per 10,000 unless a column name says otherwise.

| File | Contents |
|---|---|
| `source_numbers_v2.tsv` | Numbers of the principal analyses, written by `verify_all.py` |
| `primary_series.tsv` | Reproduction check, depth correction, segment analysis and glial comparison on the postnatal series, written by `primary_series.py` |
| `clarence_numbers.tsv` | The multiome cohort, written by `clarence_replication.py` |
| `clarence_depth.tsv` | Depth correction in that cohort, written by `depth_controlled_detection.py` |
| `composition_glia.tsv` | Segment and glial analyses in that cohort, written by `composition_and_glia.py` |
| `walchli_adult.tsv`, `walchli_fetal.tsv` | Arteriovenous segment in the sorted endothelial atlas, written by `walchli_zonation.R` |
| `allen_fig.csv` | Cluster-level expression and anatomical division, whole brain atlas, 3,168 clusters |
| `psychad_fig.csv` | Mean expression of 22 genes in four adult vascular cell types, all donors |
| `psychad_control_fig.csv` | The same for donors labelled as unaffected |
| `psychad_donor.csv` | Donor-level values behind the adult partial correlations |
| `psychad_diagnosis.csv` | Donor-level values behind the comparison across diagnosis |
| `donor_pgr_counts.csv` | Per-donor nucleus counts behind the single-nucleus result, 65 donors |
| `steyn_endo_pseudobulk.csv` | Donor-level endothelial values in the replication cohort, 12 donors |
| `steyn_celltype_pgr.csv` | Cell type values in the replication cohort |
| `psychad_lifespan.tsv` | The childhood donors of the adult resource, written by `psychad_lifespan.py` |
| `pgr_targetome.tsv` | The binding repertoire test, written by `pgr_targetome.py` |
| `regional_numbers.tsv` | PGR by anatomical division and cell class, written by `regional_table.py` |
| `brainspan_numbers.tsv` | The bulk developmental series, written by `brainspan_axis.py` |
| `brainspan_tables.tsv` | Every value printed in Supplementary Tables S20 and S21, written by `brainspan_tables.py` |
| `dimension_numbers.tsv`, `dimension2_numbers.tsv` | Corpus specificity scan, written by `dimension_scanner.py` and `dimension_scanner_2.py`, reported in the additional analyses |
| `corpus_null_sensitivity.tsv` | The same scan at three reference set sizes, written by `corpus_null_sensitivity.py`, reported in the additional analyses |
| `eqtl_numbers.tsv` | cis-eQTL counts by tissue, written by `eqtl_scan.py`, reported in the additional analyses |
| `gwas_numbers.tsv` | Catalogued associations inside the gene, written by `gwas_locus.py`, reported in the additional analyses |
| `donor_level.tsv` | Donor-level counterparts of the pooled statistics, the detection expectation of the first band and the adult trajectory, written by `donor_level.py` |
| `segment_null.tsv` | The arteriovenous preference against abundance-matched genes, and a donor axis control for the effector modules of the additional analyses, written by `segment_null.py` |
| `ambient_doublet.tsv` | Doublet calls and the mural contamination test against matched control genes, written by `ambient_doublet.py` |
| `prior_scan.tsv` | Which supplementary file of which earlier study contains the gene, one row per file, written by `prior_scan.py` |
| `prior_scan_steyn.tsv` | The endothelial rows of the temporal cortex cohort's own supplementary table |
| `genome_scan.tsv` | Per-gene age association, fold change, first band zero fraction and the three-criteria verdict for all 18,794 genes of the endothelial series, written by `genome_scan.py` |
| `nr4a1_program.tsv` | Partial correlations between PGR and the junctional pathway of reference 16, with controls and an abundance-matched reference distribution, written by `nr4a1_program.py` |
| `literature_anchors.tsv` | The published values drawn on Figure 2, each with its DOI |
| `figure_source_data.tsv` | Every value drawn in Figure 2 and in Supplementary Figures S2 to S5, written by `make_figures_2.py` as it draws them |
| `gene_list.tsv` | The 54 genes meeting all three developmental criteria and the six ranking above PGR, written by `gene_list.py` |
| `amplitude_rank.tsv` | Postnatal amplitude of each gene against genes of the same abundance, in eight cell classes, with the top gene of every abundance bin and the leave-one-study-out check, written by `amplitude_rank.py` |
| `age_axis.tsv` | The age axis of the postnatal series and its postnatal equivalents, written by `age_axis.py` |
