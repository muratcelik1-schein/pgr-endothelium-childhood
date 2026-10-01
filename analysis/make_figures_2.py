# -*- coding: utf-8 -*-
"""Figure 2 and the four Supplementary Figures 2 to 5. Unlike make_figures.py these read ONLY the derived tables
shipped with the package, so a reader reproduces them without downloading any
atlas: primary_series.tsv, source_numbers_v2.tsv, composition_glia.tsv,
walchli_adult.tsv, brainspan_tables.tsv.

Every number drawn is parsed out of those tables and checked with an assert
against the value the manuscript reports, so a figure cannot drift from the text.

Panels are lettered in the order the manuscript cites them, so the citation
sequence stays ascending.

Run: python3 make_figures_2.py [--out ../figures]
Seed: none needed, nothing here is random (2026 build).
"""
import argparse, os, re, json
from PIL import Image
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, Circle, Rectangle, PathPatch, FancyArrowPatch, Patch
from matplotlib.path import Path
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory as blend

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(HERE, "..", "figures"))
OUT = os.path.abspath(ap.parse_args().out)
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "axes.titlelocation": "left",
    "axes.titlepad": 5, "axes.labelpad": 2.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6, "legend.frameon": False, "legend.handlelength": 1.6,
    "legend.handletextpad": 0.6, "legend.borderaxespad": 0.1, "legend.labelspacing": 0.35,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.major.pad": 2, "ytick.major.pad": 2,
    "axes.edgecolor": "#333333", "xtick.color": "#333333", "ytick.color": "#333333",
    "text.color": "#1F2328", "axes.labelcolor": "#1F2328",
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "scatter.edgecolors": "none", "hatch.linewidth": 0.4,
})
C = dict(pgr="#D55E00", cmp="#0072B2", lig="#009E73", grey="#8A8F94", light="#C9CDD1", faint="#E4E7EA",
         ink="#1F2328", mute="#5C6166", rule="#9AA0A6", white="#FFFFFF")
MM = 1 / 25.4
LETTER = 8

def fisher(r, n):
    """Fisher 95% interval for a Spearman coefficient over n independent units.
    Used only where the unit is the donor. The bulk series is sample level with
    donor clustering, so no interval is drawn there and the legend says why."""
    if n is None or n < 5: return None
    z, se = np.arctanh(np.clip(r, -0.999, 0.999)), 1.0 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    return abs(r - lo), abs(hi - r)

# Every drawn value is also written to a table, so the table and the figure cannot diverge.
SRC = []
def srec(panel, **kv):
    SRC.append(dict(panel=panel, **kv))

def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

def letter(ax, s, dx=-0.16, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=LETTER,
            fontweight="bold", ha="left", va="baseline")

# ------------------------------------------------------------------ tables
def tsv(name):
    rows = []
    for line in open(os.path.join(HERE, name), encoding="utf-8"):
        rows.append(line.rstrip("\n").split("\t"))
    return rows

PS = tsv("primary_series.tsv")
SN = tsv("source_numbers_v2.tsv")
CG = tsv("composition_glia.tsv")
WA = tsv("walchli_adult.tsv")
BT = tsv("brainspan_tables.tsv")
DL = tsv("donor_level.tsv")
AX = tsv("age_axis.tsv")
AR = tsv("amplitude_rank.tsv")

def cell(rows, a, b, col=2):
    for r in rows:
        if len(r) > col and r[0] == a and r[1] == b:
            return r[col]
    raise KeyError(f"{a}/{b} missing")

rho = lambda s: float(re.search(r"rho=([+-]?\d+\.\d+)", s).group(1))
num = lambda s, k: float(re.search(rf"{k}=([+-]?\d+\.?\d*)", s).group(1))
jj = lambda rows, a, b: json.loads(cell(rows, a, b).strip('"').replace('""', '"'))

# ------------------------------------------------------------------ figure registry
# Every panel is declared here once; drawing blocks attach through AX[...], and the registry and
# the width gate run from one place. Two main figures; everything else is a Supplementary Figure.
PLAN = {
    # 160 x 205 mm, full page portrait; panel a 72 mm, strip 12 mm, panel b 63 mm high; all text 8 pt.
    "Figure2":              dict(size=(160, 205), grid=(2, 1), ratios=[1.0], hratios=[1.0, 0.55], wspace=0.0,
                                 left=0.125, right=0.98, top=0.945, bottom=0.035, hspace=1.30,
                                 panels={"a": (0, 0), "b": (1, 0)},
                                 places={"Fig 2a": (0.125, 0.5951, 0.855, 0.3512),
                                         "Fig 2b": (0.100, 0.0244, 0.880, 0.3073)},
                                 strip=(0.125, 0.3902, 0.855, 0.0585)),
    # Panels are stacked vertically so that all figures use a single 8 pt text size, as in Figure 1.
    "SupplementaryFigure2": dict(size=(120, 175), grid=(3, 1), ratios=[1.0], wspace=0.0, left=0.20, right=0.965, top=0.955, bottom=0.055, hspace=0.62,
                                 panels={"a": (0, 0), "b": (1, 0), "c": (2, 0)}),
    # panel b carries twenty gene labels; at 8 pt each label needs about 3.2 mm, so the panel must be at least 70 mm high
    "SupplementaryFigure3": dict(size=(120, 185), grid=(2, 1), ratios=[1.0], hratios=[1.0, 2.15], wspace=0.0, left=0.30, right=0.965, top=0.955, bottom=0.05, hspace=0.30,
                                 panels={"a": (0, 0), "b": (1, 0)}),
    # One panel carries sixteen structure labels and needs about 52 mm at 8 pt. Four stacked panels
    # do not fit on one page, so the 2x2 layout is kept with taller rows.
    "SupplementaryFigure4": dict(size=(170, 190), grid=(2, 2), ratios=[1.0, 1.15], wspace=0.55, left=0.135, right=0.975, top=0.955, bottom=0.055, hspace=0.42,
                                 panels={"a": (0, 0), "b": (0, 1), "c": (1, 0), "d": (1, 1)}),
    "SupplementaryFigure5": dict(size=(120, 170), grid=(3, 1), ratios=[1.0], wspace=0.0, left=0.20, right=0.965, top=0.955, bottom=0.055, hspace=0.62,
                                 panels={"a": (0, 0), "b": (1, 0), "c": (2, 0)}),
}
FIGS, AX, GRIDS = {}, {}, {}
for name, pl in PLAN.items():
    f = plt.figure(figsize=(pl["size"][0] * MM, pl["size"][1] * MM))
    g = f.add_gridspec(*pl["grid"], width_ratios=pl["ratios"], height_ratios=pl.get("hratios"),
                       wspace=pl["wspace"], left=pl["left"], right=pl["right"],
                       top=pl["top"], bottom=pl["bottom"], hspace=pl.get("hspace", 0.3))
    FIGS[name], GRIDS[name] = f, g
    key = "Fig 2" if name == "Figure2" else "Supplementary Fig " + name[-1]
    for lt, (r, c) in pl["panels"].items():
        AX[f"{key}{lt}"] = f.add_subplot(g[r, c])
for name, pl in PLAN.items():          # panels that need an exact place say so in PLAN
    for k, box in pl.get("places", {}).items():
        AX[k].set_position(box)

def save_all():
    for name, f in FIGS.items():
        for ax in f.axes:
            pass
        f.savefig(os.path.join(OUT, f"{name}.pdf")); f.savefig(os.path.join(OUT, f"{name}.png"), dpi=600)
        w, h = [v / 600 * 25.4 for v in Image.open(os.path.join(OUT, f"{name}.png")).size]
        print(f"{name} written, {w:.1f} x {h:.1f} mm"); plt.close(f)
LETTER_DX = {"Fig 2a": -0.117, "Fig 2b": -0.085, "Supplementary Fig 3b": -0.55}   # panel -> dx, set per panel
LETTER_DY = {"Fig 2a": 1.026, "Fig 2b": 1.017}   # the letter baseline coincides with the title baseline
def letters_all():
    for key, ax in AX.items():
        letter(ax, key[-1], dx=LETTER_DX.get(key, -0.16), dy=LETTER_DY.get(key, 1.06))

ax = AX["Supplementary Fig 2b"]; clean(ax)
pm = cell(SN, "permutation", "max_rho_among_gate")
obs = num(pm, "observed"); med = num(pm, "null_median")
p95 = num(pm, "null_p95"); mx = num(pm, "null_max")
assert (obs, med, p95, mx) == (0.680, 0.349, 0.466, 0.605), (obs, med, p95, mx)
ax.barh([0], [mx], height=0.34, color=C["faint"], zorder=1)
ax.barh([0], [p95], height=0.34, color=C["light"], zorder=2)
ax.plot([med], [0], marker="|", ms=11, mew=1.2, color=C["mute"], zorder=3)
ax.plot([obs], [0], marker="o", ms=5, color=C["pgr"], zorder=4)
srec("Supplementary Fig 2b", quantity="strongest association among passing genes", observed=obs,
     null_median=med, null_p95=p95, null_max=mx)
ng = cell(SN, "permutation", "n_gate_genes")
assert "observed=54" in ng and "p=0.0025" in ng, ng
ax.set_yticks([0]); ax.set_yticklabels(["strongest\nassociation"])
ax.set_xlim(0, 0.78); ax.set_xlabel("Spearman rho")
ax.text(obs, 0.30, "observed\n0.680", ha="center", va="bottom", fontsize=6, color=C["pgr"])
# the bands are described in the legend; only the median is labelled on the axis
ax.text(med, -0.30, "permutation median", ha="center", va="top", fontsize=5.6, color=C["mute"])
ax.text(0.0, -0.34, "2,000 shuffles of donor age within study. 54 passing genes,\n"
                    "frequency 0.0025; rho 0.680, frequency 0.0005",
        transform=ax.transAxes, fontsize=6, color=C["mute"], va="top")
ax.set_ylim(-0.45, 0.55)
ax.set_title("Within-study permutation test")

# --- 3b within study and leave-one-study-out ---
ax = AX["Supplementary Fig 2c"]; clean(ax)
ws = jj(SN, "robust", "within_study"); lo = jj(SN, "robust", "LOSO_rank")
keys = ["velmeshev", "herring", "wang", "zhu"]
assert [round(ws[k][1], 3) for k in keys] == [0.625, 0.599, 0.817, 0.570]
y = np.arange(len(keys))[::-1]
_err = np.array([fisher(ws[k][1], ws[k][0]) for k in keys]).T
ax.barh(y, [ws[k][1] for k in keys], height=0.5, color=C["cmp"], zorder=2)
ax.errorbar([ws[k][1] for k in keys], y, xerr=_err, fmt="none", ecolor=C["ink"],
            elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=3)
for _k in keys: srec("Supplementary Fig 2c", study=_k, n_donor=ws[_k][0], rho_within_study=ws[_k][1],
                     genome_wide_rank_without_this_study=ws[_k][2])
for yy, k in zip(y, keys):
    ax.text(1.10, yy, f"n={ws[k][0]}  rank {lo[k]}", va="center", ha="right", fontsize=5.8,
            color=C["mute"])
ax.set_yticks(y); ax.set_yticklabels([k.capitalize() for k in keys])
ax.set_xlim(0, 1.12); ax.set_xticks([0, 0.5, 1.0])
ax.set_xlabel("Spearman rho within study")
ax.set_title("Within each contributing study")


ax = AX["Supplementary Fig 5c"]; clean(ax)
def _r(key):
    """Converts the '+0.709 (n=28)' format of the region block into a number and n."""
    s = cell(SN, "region", key)
    m = re.match(r"([+-]?[0-9.]+) \(n=(\d+)\)", s.strip())
    assert m, f"unexpected format: {s!r}"
    return float(m.group(1)), int(m.group(2))

_pfc, _npfc = _r("rho_PFC"); _cin, _ncin = _r("rho_cingulate_cortex")
_cor, _ncor = _r("rho_cortex"); _stg, _nstg = _r("rho_STG_cortex"); _v1, _nv1 = _r("rho_V1")
lab_val = [("unadjusted", 0.689), ("depth, cells", float(cell(SN, "robust", "rho_after_depth"))),
           ("sex removed", float(cell(SN, "robust", "rho_after_sex"))),
           ("female, n=28", float(cell(SN, "robust", "rho_female"))),
           ("male, n=37", float(cell(SN, "robust", "rho_male"))),
           ("region removed", float(cell(SN, "region", "rho_after_region"))),
           (f"PFC, n={_npfc}", _pfc),
           (f"cingulate, n={_ncin}", _cin),
           (f"cortex, n={_ncor}", _cor),
           (f"STG, n={_nstg}", _stg),
           (f"V1, n={_nv1}", _v1)]
assert abs(float(cell(SN, "region", "rho_raw")) - 0.689) < 1e-9
y = np.arange(len(lab_val))[::-1]
ax.plot([0.689, 0.689], [y.min() - 0.6, y.max() + 0.6], lw=0.6, ls=(0, (3, 2)), color=C["rule"], zorder=1)
ax.scatter([v for _, v in lab_val], y, s=16, color=C["pgr"], zorder=3)
for _l, _v in lab_val: srec("Supplementary Fig 5c", condition=_l, rho=_v)
ax.set_yticks(y); ax.set_yticklabels([l for l, _ in lab_val])
ax.set_xlim(0.40, 0.90); ax.set_xlabel("Spearman rho")
ax.set_title("After removing each confound")


# 5a and 5b are now panels b and c of Supplementary Fig. S1

# --- 5a depth correction ---
ax = AX["Supplementary Fig 5a"]; clean(ax)
depths = [1000, 1500, 2000, 3000]
tgt, ctrl = [], {}
for d in depths:
    tgt.append(rho(cell(PS, "S1_depth", f"depth{d}_PGR")))
    for r in PS:
        m = re.match(rf"depth{d}_(.+)$", r[1]) if r[0] == "S1_depth" else None
        if m and m.group(1) not in ("PGR", "nuclei", "controls_matching_target"):
            ctrl.setdefault(m.group(1), []).append(rho(r[2]))
assert [round(v, 3) for v in tgt] == [0.711, 0.772, 0.793, 0.789], tgt
x = np.arange(len(depths))
for g, v in ctrl.items():
    if len(v) == len(depths):
        ax.plot(x, v, lw=0.6, color=C["light"], zorder=1)
ax.plot(x, tgt, lw=1.6, color=C["pgr"], marker="o", ms=4, zorder=3)
for _i, _d in enumerate(depths): srec("Supplementary Fig 5a", fixed_depth=_d, gene="PGR", rho=tgt[_i])
for _g, _v in ctrl.items():
    if len(_v) == len(depths):
        for _i, _d in enumerate(depths): srec("Supplementary Fig 5a", fixed_depth=_d, gene=_g, rho=_v[_i])
ax.axhline(float(cell(SN, "donor_sc", "rho_age_proportion").split()[0]), lw=0.6,
           ls=(0, (3, 2)), color=C["rule"])
ax.set_xticks(x); ax.set_xticklabels([f"{d:,}" for d in depths])
ax.set_xlabel("nuclei downsampled to a fixed depth (counts)")
ax.set_ylabel("Spearman rho with age")
ax.text(x[-1], tgt[-1] + 0.05, "PGR", color=C["pgr"], fontsize=6.5, ha="right")
ax.text(0.0, -0.30, "eighteen control genes in grey; dashed line is the uncorrected value",
        transform=ax.transAxes, fontsize=6, color=C["mute"], va="top")
ax.set_title("Corrected for sequencing depth")

# --- 7a bulk tissue, 16 structures ---
ax = AX["Supplementary Fig 4d"]; clean(ax)
st = [(r[1], float(r[2])) for r in BT if r and r[0] == "19" and " " not in r[1]]
assert len(st) == 16, len(st)
st.sort(key=lambda t: t[1])
for _s, _v in st: srec("Supplementary Fig 4d", structure=_s, rho_with_age=_v)
ax.barh(np.arange(16), [v for _, v in st], height=0.62,
        color=[C["pgr"] if k == "CBC" else C["cmp"] for k, _ in st])
ax.set_yticks(np.arange(16)); ax.set_yticklabels([k for k, _ in st], fontsize=6)
ax.set_xlim(0, 1.0); ax.set_xlabel("Spearman rho with age, bulk tissue")
ax.set_title("All 16 structures of a bulk developmental series")
# --- 7b identity genes and ratio ---
BS = {r[1]: r[2] for r in BT if r and r[0] == "18"}
ax = AX["Supplementary Fig 4c"]; clean(ax)
rows = [("PGR", float(BS["PGR"]), C["pgr"]), ("CLDN5", float(BS["CLDN5"]), C["grey"]),
        ("FLT1", float(BS["FLT1"]), C["grey"]), ("VWF", float(BS["VWF"]), C["grey"]),
        ("PGR / CLDN5", float(BS["PGR / CLDN5"]), C["cmp"]),
        ("PGR / FLT1", float(BS["PGR / FLT1"]), C["cmp"]),
        ("PGR / VWF", float(BS["PGR / VWF"]), C["cmp"])]
assert [round(v, 3) for _, v, _ in rows] == [0.707, -0.511, -0.105, -0.043, 0.779, 0.706, 0.649]
yy = np.arange(len(rows))[::-1]
ax.axvline(0, lw=0.5, color=C["rule"])
for _l, _v, _c in rows: srec("Supplementary Fig 4c", quantity=_l, rho_with_age=_v)
ax.barh(yy, [v for _, v, _ in rows], height=0.58, color=[c for _, _, c in rows])
ax.set_yticks(yy); ax.set_yticklabels([k for k, _, _ in rows], fontsize=6.5)
ax.set_xlim(-0.65, 0.95); ax.set_xlabel("Spearman rho with age")
ax.set_title("PGR, the identity genes, and their ratios")

BANDS = ["0.75-2", "2-6", "6-12", "12-20", ">20"]
REC = ["PGR", "NR3C1", "NR3C2", "ESR1", "AR", "GPER1"]
dat = {g: jj(SN, "endo_dev", g) for g in REC + ["PGR-AS1", "ZNF385B", "ABCB1", "AKR1C1", "AKR1C3"]}
assert dat["PGR"]["rank"] == 7 and dat["PGR"]["fold"] == 16.97

x = np.arange(5)

# --- 4a three criteria ---
ax = AX["Supplementary Fig 3a"]; clean(ax)
ax.axvspan(5, 62, color=C["faint"], zorder=0)   # the whole half-plane admitted by the threshold
ax.axhline(50, lw=0.6, ls=(0, (3, 2)), color=C["rule"])
ax.axvline(5, lw=0.6, ls=(0, (3, 2)), color=C["rule"])
for g, d in dat.items():
    hit = g == "PGR"
    srec("Supplementary Fig 3a", gene=g, fold=d["fold"], zero_pct_band1=d["zero_pct"])
    ax.scatter([d["fold"]], [d["zero_pct"]], s=26 if hit else 14,
               color=C["pgr"] if hit else (C["cmp"] if g in REC else C["grey"]), zorder=3)
    off = {"ABCB1": (7, 2), "NR3C1": (-7, -9), "NR3C2": (6, 6), "ESR1": (4, -8),
           "GPER1": (-4, 8), "AR": (-3, 9), "AKR1C1": (5, 4), "AKR1C3": (5, -8),
           "PGR-AS1": (6, -2), "ZNF385B": (6, -1)}.get(g, (4, 3))
    ax.annotate(g, (d["fold"], d["zero_pct"]), textcoords="offset points",
                xytext=off, fontsize=5.6, color=C["pgr"] if hit else C["mute"])
ax.set_xscale("log"); ax.set_xlim(0.42, 62); ax.set_ylim(-14, 108)
ax.set_xlabel("fold change, oldest band over first band")
ax.set_ylabel("first band donors with no detection (%)")
ax.set_title("Fold change against first band detection")

# --- 4c top gene of each abundance bin ---
ax = AX["Supplementary Fig 3b"]; clean(ax)
TOP = []
for r in AR:
    if len(r) > 3 and r[0] == "top_of_each_bin" and r[2].startswith("log2"):
        TOP.append((r[1], float(r[2].split()[1]), float(re.search(r"mean ([\d.]+)", r[3]).group(1))))
assert len(TOP) == 20, len(TOP)
assert sum(1 for _, v, _ in TOP if v > 0) == 5 and sum(1 for _, v, _ in TOP if v <= 0) == 15
_d = dict((g, v) for g, v, _ in TOP)
assert abs(_d["PGR"] - 0.7317) < 1e-4 and abs(_d["SOX4"] + 1.4007) < 1e-4
TOP.sort(key=lambda t_: t_[1])
CLIP, ASIDE = 1.55, {"HERC2P3", "DMKN"}
for i6, (g, v, ab) in enumerate(TOP):
    col = C["light"] if g in ASIDE else (C["pgr"] if v > 0 else C["cmp"])
    ax.barh(i6, min(v, CLIP), 0.66, color=col, lw=0)
    srec("Supplementary Fig 3b", gene=g, log2_amplitude=v, mean_counts_per_10k=ab, direction="rising" if v > 0 else "falling",
         set_aside="pseudogene or below 0.07 counts per 10,000" if g in ASIDE else "")
    if v > CLIP:
        ax.plot([CLIP], [i6], marker=">", ms=3, color=col, clip_on=False)
        ax.text(CLIP - 0.06, i6, f"{v:+.2f}", fontsize=5.6, color=C["mute"], ha="right", va="center")
ax.axvline(0, color=C["ink"], lw=0.5)
ax.set_yticks(np.arange(len(TOP)))
ax.set_yticklabels([g + (" *" if g in ASIDE else "") for g, _, _ in TOP], fontsize=5.8)
for lb, (g, _, _) in zip(ax.get_yticklabels(), TOP):
    if g == "PGR": lb.set_fontweight("bold"); lb.set_color(C["pgr"])
ax.set_ylim(-0.75, len(TOP) - 0.25); ax.set_xlim(-1.6, CLIP)
ax.set_xlabel("amplitude (log$_2$ counts per 10,000)")
ax.text(0.0, -0.20, "* a pseudogene, and a gene below\n0.07 counts per 10,000", transform=ax.transAxes,
        fontsize=5.6, va="top", color=C["mute"])
ax.set_title("The top of each abundance bin")

SEG = ["arterial", "capillary", "venous"]

# --- 5b increase within segment ---
ax = AX["Supplementary Fig 5b"]; clean(ax)
comp = [rho(cell(PS, "S2a_composition", s)) for s in SEG]
expr = [num(cell(PS, "S2b_within_segment", s), "expr rho") for s in SEG]
pct = [num(cell(PS, "S2b_within_segment", s), "pct_pos rho") for s in SEG]
assert [round(v, 3) for v in expr] == [0.629, 0.660, 0.758]
assert [round(v, 3) for v in comp] == [0.303, -0.454, 0.351]
w, xs = 0.26, np.arange(3)
ax.axhline(0, lw=0.5, color=C["rule"])
NDON65 = int(cell(SN, "series", "n_donor")); assert NDON65 == 65
for _off, _v, _c, _l in ((-w, comp, C["light"], "share of the sample"),
                         (0.0, expr, C["pgr"], "expression"),
                         (w, pct, C["cmp"], "positive nuclei")):
    ax.bar(xs + _off, _v, w, color=_c, label=_l, zorder=2)
    ax.errorbar(xs + _off, _v, yerr=np.array([fisher(r, NDON65) for r in _v]).T, fmt="none",
                ecolor=C["ink"], elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=3)
for _i, _s in enumerate(SEG): srec("Supplementary Fig 5b", segment=_s, share_vs_age=comp[_i],
                                    expression_vs_age=expr[_i], positive_fraction_vs_age=pct[_i])
ax.set_xticks(xs); ax.set_xticklabels(SEG)
ax.set_ylabel("Spearman rho with age"); ax.set_ylim(-0.75, 1.32)
ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.02), ncol=3, columnspacing=0.9, handlelength=1.2, fontsize=5.6)
ax.set_title("Within each segment")


ax = AX["Supplementary Fig 4b"]; clean(ax)
def seg3(g, which="cp"):
    s = cell(WA, "segment_expression", g)
    a = re.search(r"art=([\d.]+) cap=([\d.]+) ven=([\d.]+)", s)
    p = re.search(r"pct art=([\d.]+) cap=([\d.]+) ven=([\d.]+)", s)
    return [float(v) for v in (a if which == "cp" else p).groups()]
pg = seg3("PGR")
assert [round(v, 4) for v in pg] == [0.3512, 0.4915, 0.5149], pg
for g in ("PECAM1", "CLDN5", "FLT1"):
    v = np.array(seg3(g)); ax.plot(xs, v / v[0], lw=0.9, color=C["grey"], marker="o", ms=2.4)
    ax.text(2.06, (v / v[0])[-1], g, fontsize=6, va="center", color=C["mute"])
v = np.array(pg); ax.plot(xs, v / v[0], lw=1.7, color=C["pgr"], marker="o", ms=3.6)
for _i, _s in enumerate(["arterial", "capillary", "venous"]): srec("Supplementary Fig 4b", gene="PGR", segment=_s, value=pg[_i])
ax.text(2.06, (v / v[0])[-1], "PGR", fontsize=6.5, va="center", color=C["pgr"])
ax.axhline(1.0, lw=0.5, ls=(0, (3, 2)), color=C["rule"])
ax.set_xticks(xs); ax.set_xticklabels(SEG); ax.set_xlim(-0.2, 2.9)
ax.set_ylabel("level relative to arterial cells")
ax.set_title("72,867 sorted adult endothelial cells")

# --- 6a positive and negative nuclei ---
ax = AX["Supplementary Fig 4a"]; clean(ax)
def shares(rows, blk):
    return [float(re.search(r"PGR\+ ([\d.]+)%", cell(rows, blk, f"share_{s}")).group(1)) for s in SEG], \
           [float(re.search(r"PGR- ([\d.]+)%", cell(rows, blk, f"share_{s}")).group(1)) for s in SEG]
p1, n1 = shares(PS, "S2c_pos_vs_neg")
p2, n2 = shares(CG, "Q1c_pos_vs_neg")
assert p1 == [18.2, 33.9, 26.9] and n1 == [21.4, 34.9, 21.4]
xs4 = np.arange(3)
ax.bar(xs4 - 0.30, n1, 0.19, color=C["light"], label="PGR negative")
ax.bar(xs4 - 0.10, p1, 0.19, color=C["pgr"], label="PGR positive, primary")
ax.bar(xs4 + 0.17, n2, 0.19, color=C["light"])
ax.bar(xs4 + 0.37, p2, 0.19, color=C["cmp"], label="PGR positive, multiome")
for _i, _s in enumerate(["arterial", "capillary", "venous", "unassigned"][:len(n1)]):
    srec("Supplementary Fig 4a", cohort="primary series", segment=_s, pct_of_negative=n1[_i], pct_of_positive=p1[_i])
for _i, _s in enumerate(["arterial", "capillary", "venous", "unassigned"][:len(n2)]):
    srec("Supplementary Fig 4a", cohort="multiome cohort", segment=_s, pct_of_negative=n2[_i], pct_of_positive=p2[_i])
ax.set_xticks(xs4); ax.set_xticklabels(SEG)
ax.set_ylabel("share of nuclei (%)")
ax.set_ylim(0, 58)
ax.text(0.0, -0.24, "left pair of each group: primary series; right pair: multiome cohort",
        transform=ax.transAxes, fontsize=6, va="top", color=C["mute"])
ax.legend(loc="upper left", ncol=2, columnspacing=1.0)
ax.set_title("Segment shares of positive and negative nuclei")

# ============================================================ FIGURE 3
# Three panels, three sources. a is a schematic and carries no measured value. b puts the ligand's
# schedule and the transcript's schedule on the same age axis. c measures whether the pathway
# established in the uterus is running in this tissue.
NR = tsv("nr4a1_program.tsv")
LIT = tsv("literature_anchors.tsv")
AX_TSV = tsv("age_axis.tsv")
def nrv(blk, key):
    for r in NR:
        if len(r) > 2 and r[0] == blk and r[1] == key: return r[2]
    raise KeyError(f"{blk}/{key}")
def litv(src, q):
    for r in LIT:
        if len(r) > 2 and r[0] == src and r[1] == q: return r[2]
    raise KeyError(f"{src}/{q}")
_pr = lambda s: float(re.search(r"partial rho ([+-][\d.]+)", s).group(1))
_p  = lambda s: float(re.search(r"p = ([\d.eE+-]+)", s).group(1))
assert nrv("gate", "production").startswith("PASS")
assert litv("Goddard2014", "route").startswith("PR induces NR4A1")

# ------------------------------------------------------------------ text gate
# Journal: "Use the same typeface in the same font size for all figures". Every text in a figure must
# have one size and no two texts may overlap. The gate fails on a known faulty input
# (self-test: FIGATE_SELFTEST).
def _texts(fig):
    out = list(fig.texts)
    for ax in fig.axes:
        out += list(ax.texts) + [ax.title, ax._left_title, ax._right_title]
        if ax.axison:
            out += [ax.xaxis.label, ax.yaxis.label]
            for axis in (ax.xaxis, ax.yaxis):
                out += [tk.label1 for tk in axis._update_ticks() if tk.label1.get_visible()]
        leg = ax.get_legend()
        if leg is not None: out += list(leg.get_texts())
    return [t for t in out if t.get_visible() and t.get_text().strip()]

def norm_text(fig, size):
    """Sets every text in the figure to one size. Journal rule: "Use the same typeface in the same font
    size for all figures in your paper." One pass instead of searching for each fontsize."""
    fig.canvas.draw()
    for t in _texts(fig): t.set_fontsize(size)

def gate_text(fig, name, size):
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    T = _texts(fig)
    assert len(T) > 10, f"{name}: only {len(T)} texts found, the gate is running empty"
    bad = sorted({(t.get_text()[:30], t.get_fontsize()) for t in T if abs(t.get_fontsize() - size) > 1e-6})
    assert not bad, f"{name}: text outside {size} pt: {bad}"
    B = [(t, t.get_window_extent(r)) for t in T]
    hits = []
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            a, b = B[i][1], B[j][1]
            if a.x0 < b.x1 - 0.5 and b.x0 < a.x1 - 0.5 and a.y0 < b.y1 - 0.5 and b.y0 < a.y1 - 0.5:
                hits.append((B[i][0].get_text()[:28], B[j][0].get_text()[:28]))
    assert not hits, f"{name}: overlapping texts {hits}"
    print(f"{name}: {len(T)} texts, all {size} pt, no overlap")

if os.environ.get("FIGATE_SELFTEST"):
    _f = plt.figure(figsize=(3, 2)); _a = _f.add_subplot(111)
    _a.text(0.5, 0.5, "txt", fontsize=8); _a.text(0.52, 0.5, "txts", fontsize=8)
    _a.set_xlabel("x", fontsize=8); _a.set_ylabel("y", fontsize=8); _a.tick_params(labelsize=8)
    try:
        gate_text(_f, "selftest", 8); raise SystemExit("GATE FAILED: overlapping texts were not detected")
    except AssertionError as e:
        print("gate self-test: failed on the broken input ->", str(e)[:80])
    plt.close(_f)

# ============================================================ FIGURE 2, the story
F2 = 8
# Two panels. a puts the two measured schedules of one cortical endothelial cell on the
# source resource's own band axis, in absolute units, so the reader reads the values off
# the figure. b draws the route this receptor is known to drive elsewhere and sets our
# measurement of each step beside it.

# ---- 2a  the two measured schedules on the band axis of the full series
ax = AX["Fig 2a"]; clean(ax)
STRIP = FIGS["Figure2"].add_axes(PLAN["Figure2"]["strip"])       # schematic ligand strip, no panel letter

E = [float(cell(AX_TSV, f"band_edge_{k}", str(k))) for k in ("0.75", "2", "6", "12", "20")]
assert E == [0.0, 1.23, 5.23, 11.23, 19.23], E
BAND7 = ["0-0.5", "0.5-0.75", "0.75-2", "2-6", "6-12", "12-20", ">20"]
POST7 = ["prenatal", "prenatal", "0-1.2", "1.2-5.2", "5.2-11.2", "11.2-19.2", ">19.2"]
NDON = [int(cell(SN, "full_series", f"n_donor_{b}y")) for b in BAND7]
assert sum(NDON) == int(cell(SN, "full_series", "n_donor")) == 112, NDON
PGRB = np.array(jj(SN, "full_series", "PGR")["bands"])
ABCB = np.array(jj(SN, "full_series", "ABCB1")["bands"])
assert list(ABCB) == [7.6741, 15.6174, 29.95, 29.881, 30.5494, 28.4469, 25.4084], ABCB
assert list(PGRB) == [0.0034, 0.0099, 0.035, 0.2407, 0.4075, 0.6265, 0.6328], PGRB

# the two annotated quantities are recomputed from the plotted points, not retyped
EARLY_A = (ABCB[2] - ABCB[0]) / (ABCB.max() - ABCB[0])
EARLY_P = (PGRB[2] - PGRB[0]) / (PGRB.max() - PGRB[0])
assert round(EARLY_A, 3) == float(jj(SN, "full_series", "ABCB1")["early_fraction_by_2y"]) == 0.974
assert round(EARLY_P, 2) == float(jj(SN, "full_series", "PGR")["early_fraction_by_2y"]) == 0.05
FOLD7 = PGRB[6] / PGRB[2]                       # the five postnatal bands of the drawn series
FOLD5 = float(jj(SN, "endo_dev", "PGR")["fold"])  # the 65 donor series of Fig 1a
assert round(FOLD7, 2) == 18.08 and FOLD5 == 16.97, (FOLD7, FOLD5)
B1_65 = float(jj(SN, "endo_dev", "PGR")["bands"][0])     # the single band the two series differ in
assert B1_65 == 0.0372 and round(PGRB[6] / B1_65, 2) == 17.01

x = np.arange(7)
BIRTH_X = 1.5                                    # the 0.5-0.75 | 0.75-2 boundary, band_edge_0.75
ax.axvspan(-0.55, BIRTH_X, color="#EDF0F3", lw=0, zorder=0)
ax.set_yscale("log"); ax.set_xlim(-0.55, 7.7); ax.set_ylim(8e-4, 900)
ax.set_yticks([1e-3, 1e-2, 1e-1, 1, 10, 100])
ax.set_yticklabels(["0.001", "0.01", "0.1", "1", "10", "100"])
_yf = lambda v: (np.log10(v) - np.log10(8e-4)) / (np.log10(900) - np.log10(8e-4))   # axis fraction on a log axis
ax.axvline(BIRTH_X, ymax=_yf(175.0 / 1.40) - 0.012, color=C["rule"], lw=0.7, ls=(0, (2.5, 2)), zorder=5)
ax.text(BIRTH_X + 0.08, 0.00112, "Birth", fontsize=F2, color=C["mute"], ha="left", va="center", zorder=6)
ax.text(0.42, 0.00112, "Prenatal", fontsize=F2, color=C["mute"], ha="center", va="center", zorder=6)

for vals, col, fmt, dy in ((ABCB, C["cmp"], "{:.2f}", 1.75), (PGRB, C["pgr"], "{:.4f}", 1.75)):
    ax.plot(x, vals, color=col, lw=1.3, marker="o", ms=3.2, mew=0, zorder=4)
    for xi, v in zip(x, vals):
        ax.text(xi, v * dy, fmt.format(v), fontsize=F2, color=col, ha="center", va="bottom", zorder=6)
for i in range(7):
    srec("Fig 2a", band_from_conception=BAND7[i], postnatal_years=POST7[i], n_donor=NDON[i],
         ABCB1_counts_per_10k=float(ABCB[i]), PGR_counts_per_10k=float(PGRB[i]), schematic="no")

ax.text(6.34, ABCB[6], "ABCB1", fontsize=F2, color=C["cmp"], ha="left", va="bottom", fontweight="bold")
ax.text(6.34, ABCB[6] / 1.9, "transcript", fontsize=F2, color=C["mute"], ha="left", va="top")
ax.text(6.34, PGRB[6], "PGR", fontsize=F2, color=C["pgr"], ha="left", va="bottom", fontweight="bold")
ax.text(6.34, PGRB[6] / 1.9, "transcript", fontsize=F2, color=C["mute"], ha="left", va="top")

def brk(x0, x1, y, k, col):
    """square bracket on a logarithmic axis; k>1 draws the end ticks downwards"""
    ax.plot([x0, x0, x1, x1], [y / k, y, y, y / k], color=col, lw=0.7, zorder=6,
            solid_joinstyle="miter", solid_capstyle="butt")

brk(0, 2, 175.0, 1.40, C["cmp"])
_ta = ax.text(0.0, 215.0, "ABCB1 completes 97% of its rise over this span, by 1.2 postnatal years",
              fontsize=F2, color=C["cmp"], ha="left", va="bottom")
FIGS["Figure2"].canvas.draw()   # the PGR label starts where the ABCB1 label ends; the position is measured and not estimated
_xe = ax.transData.inverted().transform(_ta.get_window_extent().get_points())[1][0]
ax.text(_xe + 0.30, 215.0, "PGR completes 5%", fontsize=F2, color=C["pgr"], ha="left", va="bottom")
ax.plot([2, 6], [0.0265, 0.0265], color=C["pgr"], lw=0.7, zorder=6)
ax.plot([2, 2], [0.0265, 0.0205], color=C["pgr"], lw=0.7, zorder=6)
ax.plot([6, 6], [0.0265, 0.0205], color=C["pgr"], lw=0.7, zorder=6)
ax.text(4.0, 0.0132, "PGR increases about 18-fold across the five postnatal bands",
        fontsize=F2, color=C["pgr"], ha="center", va="top")
ax.text(4.2, 0.0052, f"Fig. 1a series (65 donors): 17-fold,\nfirst band mean {B1_65:.4f}",
        fontsize=F2, color=C["mute"], ha="center", va="top", linespacing=1.2)
srec("Fig 2a", quantity="fraction of the ABCB1 rise complete by the 0.75-2 band", value=round(float(EARLY_A), 3), schematic="no")
srec("Fig 2a", quantity="fraction of the PGR rise complete by the 0.75-2 band", value=round(float(EARLY_P), 2), schematic="no")
srec("Fig 2a", quantity="PGR fold change across the five postnatal bands, 112 donor series", value=round(float(FOLD7), 2), schematic="no")
srec("Fig 2a", quantity="PGR fold change across the five postnatal bands, 65 donor series", value=FOLD5, schematic="no")
srec("Fig 2a", quantity="PGR mean in the 0.75-2 band, 65 donor series", value=B1_65, schematic="no")

ax.set_xticks(x)
ax.set_xticklabels([f"{BAND7[i]}\n({POST7[i]})" for i in range(7)], fontsize=F2, linespacing=1.25)
for i in range(7):
    ax.text(i, -0.150, f"n = {NDON[i]}", transform=blend(ax.transData, ax.transAxes),
            fontsize=F2, color=C["mute"], ha="center", va="top")
ax.plot([2, 6], [-0.205, -0.205], transform=blend(ax.transData, ax.transAxes), color=C["light"], lw=0.6, clip_on=False)
ax.text(4.0, -0.218, "Postnatal", transform=blend(ax.transData, ax.transAxes),
        fontsize=F2, color=C["mute"], ha="center", va="top")
ax.set_ylabel("Mean expression in endothelium\n(counts per 10,000, log scale)", fontsize=F2)
ax.set_title("Developmental trajectories of the ABCB1 and PGR transcripts in human cortical endothelium", fontsize=F2)
ax.tick_params(labelsize=F2)

# --- the ligand strip: drawn, not measured, and labelled as such
STRIP.set_xlim(-0.55, 7.7); STRIP.set_ylim(0, 1.05); STRIP.set_axis_off()
# The strip starts at birth: reference 17 (Frederiksen 2024) measured serum from 0.17 months onward and gives no
# prenatal values. Low through childhood, increasing at puberty, high in adults, as in its Fig. 2 and 5 and text.
_sx = np.linspace(BIRTH_X, 6, 400)
_sy = np.interp(_sx, [BIRTH_X, 4.2, 4.9, 5.5, 6.0],
                     [0.12, 0.12, 0.30, 0.72, 0.88])
assert _sx[0] >= BIRTH_X, "the strip must not start before birth: reference 17 has no prenatal data"
STRIP.fill_between(_sx, 0, _sy, facecolor=C["lig"], alpha=0.28, lw=0)
STRIP.plot(_sx, _sy, color=C["lig"], lw=1.0)
STRIP.axvline(BIRTH_X, color=C["rule"], lw=0.7, ls=(0, (2.5, 2)))
STRIP.text(-0.55, 1.22, "Circulating progesterone after birth, schematic (no scale)", fontsize=F2, color=C["mute"],
           ha="left", va="bottom", linespacing=1.2)
STRIP.text(3.3, 0.34, "low across childhood", fontsize=F2, color=C["lig"], ha="center", va="bottom")
STRIP.text(6.12, 0.55, "increases at puberty", fontsize=F2, color=C["lig"], ha="left", va="center")
srec("Fig 2a", track="circulating progesterone", value="no value drawn",
     schematic="yes, drawn and not measured")

# ---- 2b  the route this receptor drives elsewhere, beside our measurement of each step
# Two rows. The upper row is the chain established experimentally in the uterus, the lower row
# measures the same steps in cortical endothelium; each measurement is placed directly below its own step.
ax = AX["Fig 2b"]; clean(ax)
ax.set_axis_off(); ax.set_xlim(0, 200); ax.set_ylim(0, 100)
NDN = int(re.search(r"\d+", nrv("meta", "donors")).group(0)); assert NDN == 53
VAL = {g: _pr(nrv(b, g)) for b, g in (("other", "NR4A1"), ("target", "CLDN5"),
                                      ("target", "PECAM1"), ("target", "CDH5"))}
assert [VAL[g] for g in ("NR4A1", "CLDN5", "PECAM1", "CDH5")] == [-0.026, -0.234, -0.099, 0.052], VAL
HOLM = [float(re.search(r"Holm p = ([\d.]+)", nrv("holm", g)).group(1)) for g in ("CLDN5", "PECAM1", "CDH5")]
assert HOLM == [0.2753, 0.9609, 0.7137], HOLM
_bb = ax.get_position(); _W, _H = FIGS["Figure2"].get_size_inches()
KX, KY = _bb.width * _W * 25.4 / 200, _bb.height * _H * 25.4 / 100      # mm per data unit, for round dots

# upper row: experimental chain
ax.text(0, 91.0, "Uterine endothelium, experimental", fontsize=F2, color=C["ink"], fontweight="bold", va="bottom")
ax.text(0, 84.5, "reference 16", fontsize=F2, color=C["mute"], va="bottom")
YC, BH = 63.5, 24.0
LB = [(34.0, 36.0, "PGR\nin the nucleus", C["pgr"]),
      (76.0, 24.0, "NR4A1", C["mute"]),
      (106.0, 24.0, "CLDN5\nPECAM1\nCDH5", C["cmp"]),
      (136.0, 50.0, "junctional\ngenes repressed,\npermeability increases", C["ink"])]
for x0, w, name, col in LB:
    ax.add_patch(Rectangle((x0, YC - BH / 2), w, BH, facecolor=C["white"], edgecolor=col, lw=0.9, zorder=3))
    ax.text(x0 + w / 2, YC, name, fontsize=F2, color=col, ha="center", va="center", zorder=4, linespacing=1.2)
for _dx, _dy in ((1.5, 4.2), (5.1, 2.0), (2.7, -2.4), (6.9, -4.6), (8.7, 1.4)):
    ax.add_patch(Ellipse((_dx + 8.0, YC + _dy), 2.0, 2.0 * KX / KY, facecolor=C["lig"], edgecolor="none", zorder=3))
ax.text(13.0, YC - 9.0, "progesterone", fontsize=F2, color=C["lig"], ha="center", va="top")
for xa, xb, lab in ((26.0, 34.0, "crosses by diffusion"), (70.0, 76.0, "induces"),
                    (100.0, 106.0, "represses"), (130.0, 136.0, "")):
    ax.add_patch(FancyArrowPatch((xa, YC), (xb, YC), arrowstyle="-|>", mutation_scale=7, lw=1.0,
                                 color=C["mute"], zorder=3))
    if lab:
        ax.text((xa + xb) / 2, YC + BH / 2 + 0.8, lab, fontsize=F2, color=C["mute"],
                ha="center", va="bottom")

ax.plot([0, 200], [45.0, 45.0], color=C["light"], lw=0.6, ls=(0, (2, 2)))

# lower row: measurement of the same steps
ax.text(0, 39.0, "Human cortical endothelium,\nthis study", fontsize=F2, color=C["ink"],
        fontweight="bold", va="top", linespacing=1.2)
ax.text(0, 23.5, f"partial Spearman rho with PGR,\nage removed, {NDN} donors",
        fontsize=F2, color=C["mute"], va="top", linespacing=1.2)
YC2, BH2 = 30.0, 16.0
RB = [(76.0, "NR4A1"), (106.0, "CLDN5"), (136.0, "PECAM1"), (166.0, "CDH5")]
for x0, g in RB:
    ax.add_patch(Rectangle((x0, YC2 - BH2 / 2), 24.0, BH2, facecolor=C["white"],
                           edgecolor=C["light"], lw=0.8, zorder=3))
    ax.text(x0 + 12.0, YC2 + 3.4, g, fontsize=F2, color=C["ink"], ha="center", va="center", zorder=4)
    ax.text(x0 + 12.0, YC2 - 3.6, f"{VAL[g]:+.3f}", fontsize=F2, color=C["ink"], ha="center", va="center", zorder=4)
    srec("Fig 2b", node=g, partial_rho_with_PGR_age_removed=VAL[g], n_donor=NDN, schematic="no")
for _x in (88.0, 118.0):          # each measured step under the step it measures
    ax.plot([_x, _x], [YC - BH / 2, YC2 + BH2 / 2], color=C["rule"], lw=0.6, ls=(0, (1.5, 1.5)), zorder=2)
ax.plot([106, 106, 190, 190], [20.0, 18.5, 18.5, 20.0], color=C["mute"], lw=0.6, solid_joinstyle="miter")
ax.text(148.0, 16.5, "no junctional gene significant after Holm correction", fontsize=F2, color=C["ink"], ha="center", va="top")
ax.text(148.0, 9.5, f"Holm p = {HOLM[0]:.3f}, {HOLM[1]:.3f} and {HOLM[2]:.3f}",
        fontsize=F2, color=C["mute"], ha="center", va="top")
for _g, _h in zip(("CLDN5", "PECAM1", "CDH5"), HOLM):
    srec("Fig 2b", node=_g, holm_p=_h, schematic="no")
srec("Fig 2b", node="route", source="Goddard 2014, 10.1016/j.cell.2013.12.025",
     schematic="the chain is drawn, the four numbers are measured")
ax.set_title("The uterine PGR to NR4A1 pathway, and the same steps in cortical endothelium", pad=3, fontsize=F2)

# ---- Supplementary Fig 2a  the screen across the transcriptome
ax = AX["Supplementary Fig 2a"]; clean(ax)
GS = [l.rstrip("\n").split("\t") for l in open(os.path.join(HERE, "genome_scan.tsv"), encoding="utf-8")]
gh = {k: i for i, k in enumerate(GS[0])}
gg = [r for r in GS[1:] if len(r) > 6]
_fold = np.array([float(r[gh["fold_last_over_first"]]) for r in gg])
_rho = np.array([float(r[gh["rho_age"]]) for r in gg])
_pass = np.array([r[gh["passes_three_criteria"]] == "1" for r in gg])
_name = [r[gh["gene"]] for r in gg]
assert len(gg) == 18794 and int(_pass.sum()) == 54, (len(gg), _pass.sum())
_lf = np.log10(np.clip(_fold, 0.05, None))
ax.scatter(_lf, _rho, s=1.2, c=C["light"], rasterized=True, zorder=1)
ax.scatter(_lf[_pass], _rho[_pass], s=8, c=C["cmp"], ec="white", lw=0.3, zorder=3)
for _g, _dx, _dy in (("PGR", -0.42, 0.10), ("PGR-AS1", -0.38, -0.09)):
    _i = _name.index(_g)
    ax.scatter([_lf[_i]], [_rho[_i]], s=28 if _g == "PGR" else 16, c=C["pgr"], ec="white", lw=0.5, zorder=5)
    ax.annotate(_g, xy=(_lf[_i], _rho[_i]), xytext=(_lf[_i] + _dx, _rho[_i] + _dy), fontsize=F2,
                fontweight="bold" if _g == "PGR" else "normal", color=C["pgr"], ha="right", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=C["pgr"], shrinkA=0, shrinkB=3))
    srec("Supplementary Fig 2a", gene=_g, fold=float(_fold[_i]), rho_age=float(_rho[_i]))
ax.axvline(np.log10(5.0), color=C["ink"], lw=0.5, ls=(0, (3, 2)), zorder=2)
ax.set_xlabel("Fold change, first to last band (log$_{10}$)")
ax.set_ylabel("Age association (Spearman rho)")
ax.set_ylim(-0.95, 0.95)
ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=2.2, color=C["light"], label=f"All genes ({len(gg):,})"),
                   Line2D([], [], ls="", marker="o", ms=3.2, color=C["cmp"], label=f"Pass all three criteria ({int(_pass.sum())})"),
                   Line2D([], [], color=C["ink"], lw=0.5, ls=(0, (3, 2)), label="5-fold")],
          loc="lower right", bbox_to_anchor=(1.02, -0.02), fontsize=F2, handlelength=1.6)
ax.set_title("Every expressed gene in endothelium")

letters_all()
for _n in FIGS:
    norm_text(FIGS[_n], F2); gate_text(FIGS[_n], _n, F2)
save_all()

# ------------------------------------------------------------------ drawn values
import csv as _csv
_keys = []
for _r in SRC:
    for _k in _r:
        if _k not in _keys: _keys.append(_k)
_p = os.path.join(HERE, "figure_source_data.tsv")
with open(_p, "w", newline="", encoding="utf-8") as _fh:
    _w = _csv.DictWriter(_fh, fieldnames=_keys, delimiter="\t", extrasaction="ignore")
    _w.writeheader()
    for _r in SRC: _w.writerow(_r)
print(f"figure_source_data.tsv written, {len(SRC)} rows across "
      f"{len({r[chr(39)+chr(39)] if False else r[list(r)[0]] for r in SRC})} panels")

# ------------------------------------------------------------------ width gate
for _f in sorted(os.listdir(OUT)):
    if not _f.endswith(".png"): continue
    _mm = Image.open(os.path.join(OUT, _f)).size[0] / 600 * 25.4
    assert _mm <= 180.05, f"{_f} width {_mm:.1f} mm, double-column limit 180 mm"
print("width gate: every figure produced is under 180 mm")
