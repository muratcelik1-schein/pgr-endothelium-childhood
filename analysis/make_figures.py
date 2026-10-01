# -*- coding: utf-8 -*-
"""Two main figures. Every number drawn is read from the CSV/TSV files (first in
--data, then in this directory); analysis constants (0.5 and 0.1 thresholds, the
gate criteria) are named once below and every count derived from them is checked
against source_numbers_v2.tsv with an assert before drawing.

Run: python3 make_figures.py [--data /tmp/dev] [--out ../figures]"""
import argparse, os, json, collections
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory as blend, offset_copy
from scipy.stats import spearmanr, rankdata, beta as beta_dist

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"))
A = ap.parse_args(); OUT = os.path.abspath(A.out)
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

def path(name):
    for d in (HERE,):
        p = os.path.join(d, name)
        if os.path.exists(p): return p
    raise FileNotFoundError(name)

# ------------------------------------------------------------------ style
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
    "scatter.edgecolors": "none",
})
# one semantic across both figures: vermillion = the progesterone axis (PGR, or the
# step that forms progesterone); blue = the named comparator; greys = everything else
C = dict(pgr="#D55E00", cmp="#0072B2", grey="#8A8F94", light="#C9CDD1", faint="#E4E7EA",
         ink="#1F2328", mute="#5C6166", rule="#9AA0A6")
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#F1F4F8", "#C6D8EA", "#84B2D6", "#3B84BA", "#0B4E85"])
MM = 1 / 25.4
SMALL, NOTE, LETTER = 6, 6, 8
DASH = (0, (3, 2))
rng = np.random.default_rng(2026)

def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)

# The text gate has one source, make_figures_2.py, so that both scripts apply the same rule;
# the gate is read from there and not copied.
_g = open(os.path.join(HERE, "make_figures_2.py"), encoding="utf-8").read()
exec(_g[_g.index("def _texts(fig):"):_g.index("if os.environ.get(\"FIGATE_SELFTEST\")")])

def title(ax, s, **kw): ax.set_title(s, **kw)

def letters(fig, groups):
    """Panel letters share the baseline of each panel title and sit 1 mm left of the
    panel's full extent (labels included); panels listed in one group (a column)
    share the same x so the letters line up."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    W = fig.get_size_inches()[0]
    for group in groups:
        x0 = min(ax.get_tightbbox(r).transformed(inv).x0 for ax, _ in group)
        for ax, s in group:
            t = ax.title; x_, y_ = t.get_transform().transform(t.get_position())
            yb = inv.transform((x_, y_))[1]
            fig.text(x0 - 1.0 * MM / W, yb, s, fontsize=LETTER, fontweight="bold", ha="left", va="baseline")

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", dpi=600); fig.savefig(f"{OUT}/{name}.png", dpi=600); plt.close(fig)   # dpi governs the rasterised clouds
    from PIL import Image
    w, h = Image.open(f"{OUT}/{name}.png").size
    assert w <= 180 / 25.4 * 600 + 1, f"{name}: {w / 600 * 25.4:.1f} mm wide, limit 180"
    print(f"{name} written, {w / 600 * 25.4:.1f} x {h / 600 * 25.4:.1f} mm")

# ------------------------------------------------------------------ source table
sn = pd.read_csv(path("source_numbers_v2.tsv"), sep="\t", dtype=str, keep_default_na=False)
def SN(block, key):
    v = sn[(sn.block == block) & (sn.key == key)].value
    assert len(v) == 1, (block, key); return v.iloc[0]
def SNJ(block, key): return json.loads(SN(block, key))
def kv(s): return {p.split("=")[0].strip(): p.split("=")[1].strip() for p in s.split(";") if "=" in p}

THR = 0.5     # cluster-level threshold used throughout the text (counts per 10,000)
DET = 0.1     # "detectable signal" threshold for the synthesis genes
al = pd.read_csv(path("allen_fig.csv"))
ps = pd.read_csv(path("psychad_fig.csv")).set_index("gene")

# ================================================================ FIGURE 1
# ---- text gate
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

# The finding of this paper is timing; location was already known and moved to Supplementary Fig. S1.
# Three panels, three questions: which receptor, which cell type, how many donors.
# Panels are stacked vertically so that the figure is legible on an A4 page and all text is 8 pt.
# Data, asserts and drawn values are unchanged.
BANDS5 = ["0.75-2", "2-6", "6-12", "12-20", ">20"]
BIRTH = 0.77
F1 = 8
fig = plt.figure(figsize=(130 * MM, 195 * MM))
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.15], left=0.16, right=0.965,
                      top=0.955, bottom=0.075, hspace=0.62)

# ---- 1a  only one of six steroid receptors moves
ax = fig.add_subplot(gs[0, 0]); clean(ax)
OTH = ["NR3C1", "NR3C2", "ESR1", "AR", "GPER1"]
ED = {g: SNJ("endo_dev", g) for g in ["PGR"] + OTH}
ax.plot([-0.3, 4.3], [1, 1], color=C["faint"], lw=0.6, zorder=0)
for g in OTH:
    v = np.array(ED[g]["bands"]); ax.plot(range(5), v / v[0], color=C["grey"], lw=0.9, zorder=2)
v = np.array(ED["PGR"]["bands"]); fp = v / v[0]
ax.plot(range(5), fp, "-o", color=C["pgr"], lw=2.0, ms=4.2, mec="white", mew=0.6, zorder=4)
ax.annotate(f"PGR\n{fp[-1]:.0f}-fold", xy=(4, fp[-1]), xytext=(4.45, fp[-1]), fontsize=F1, fontweight="bold",
            color=C["pgr"], ha="left", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=C["pgr"], shrinkA=2, shrinkB=3))
ax.text(4.24, np.median([np.array(ED[g]["bands"])[-1] / np.array(ED[g]["bands"])[0] for g in OTH]),
        "five related\nreceptors", fontsize=F1, color=C["mute"], va="center", ha="left", linespacing=1.15)
ax.set_yscale("log"); ax.set_yticks([0.25, 0.5, 1, 2, 5, 10, 20])
ax.set_yticklabels(["0.25", "0.5", "1", "2", "5", "10", "20"]); ax.set_ylim(0.16, 40)
ax.set_xticks(range(5)); ax.set_xticklabels(BANDS5, fontsize=F1); ax.set_xlim(-0.3, 5.5)
ax.set_xlabel("Age band (years)"); ax.set_ylabel("Expression relative\nto first band")
title(ax, "Six steroid receptors, endothelium", fontsize=F1)

# ---- 1b  and only one cell of the wall
ax = fig.add_subplot(gs[1, 0]); clean(ax)
SER = [("Endothelium", ED["PGR"]["bands"], C["pgr"], 2.0, 4.0),
       ("Leptomeningeal", SNJ("panel", "VLMC_PGR")["bands"], C["cmp"], 1.1, 2.8),
       ("Excitatory neurons", SNJ("panel", "Glut_PGR")["bands"], C["grey"], 0.9, 2.2),
       ("Inhibitory neurons", SNJ("panel", "GABA_PGR")["bands"], C["light"], 0.9, 2.2)]
assert max(SER[2][1]) <= 0.0101 and max(SER[3][1]) <= 0.0101
for nm, vv, col, lw, ms in SER:
    ax.plot(range(5), vv, "-o", color=col, lw=lw, ms=ms, mec="white", mew=0.4,
            zorder=4 if col == C["pgr"] else 3)
for nm, vv, col, _, _ in SER[:2]:
    ax.text(4.14, vv[-1], nm, fontsize=F1, va="center", ha="left", color=col)
ax.annotate("neurons", xy=(4.05, 0.004), xytext=(4.14, 0.085), fontsize=F1, color=C["mute"],
            va="center", ha="left", arrowprops=dict(arrowstyle="-", lw=0.5, color=C["rule"], shrinkA=1, shrinkB=1))
ax.set_xticks(range(5)); ax.set_xticklabels(BANDS5, fontsize=F1); ax.set_xlim(-0.25, 5.7)
ax.set_ylim(-0.045, 0.70)
ax.set_xlabel("Age band (years)"); ax.set_ylabel("PGR\n(counts per 10,000)")
title(ax, "PGR by cell type", fontsize=F1)

# ---- 1c  every donor, on a continuous postnatal axis
ax = fig.add_subplot(gs[2, 0]); clean(ax)
dc = pd.read_csv(path("donor_pgr_counts.csv"))
dc["post"] = dc.age - BIRTH
dc["pct"] = 100 * dc.k / dc.n
assert len(dc) == int(SN("series", "n_donor")) and int(dc.n.sum()) == 5388
wx = lambda a: np.log10(np.asarray(a, float) + 1.0)
sz = 6 + 34 * (dc.n.values / dc.n.values.max()) ** 0.6
ax.scatter(wx(dc.post.values), dc.pct.values, s=sz, facecolor=C["pgr"], edgecolor="white",
           lw=0.35, alpha=0.85, zorder=3)
EDG = [0.0, 1.23, 5.23, 11.23, 19.23]
for i, (lo, hi) in enumerate(zip(EDG, EDG[1:] + [dc.post.max() + 1])):
    g = dc[(dc.post >= lo) & (dc.post < hi)]
    k_, n_ = int(g.k.sum()), int(g.n.sum())
    ax.plot([wx(lo), wx(min(hi, dc.post.max()))], [100 * k_ / n_] * 2, color=C["ink"], lw=1.1, zorder=4)
    ax.text((wx(lo) + wx(min(hi, dc.post.max()))) / 2, 100 * k_ / n_ + 1.4,
            f"{int((g.k > 0).sum())}/{len(g)}", fontsize=F1, color=C["ink"], ha="center", va="bottom", zorder=5)
for e in EDG[1:]:
    ax.axvline(wx(e), color=C["faint"], lw=0.5, zorder=0)
TK = [0, 1, 2, 5, 10, 20, 40]
ax.set_xticks(wx(TK)); ax.set_xticklabels([str(v) for v in TK], fontsize=F1)
ax.set_xlim(wx(-0.03), wx(46)); ax.set_ylim(-1.6, 34)
ax.set_xlabel("Postnatal age (years), log scale")
ax.set_ylabel("Endothelial nuclei\nwith PGR (%)")
ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=3.4, color=C["pgr"], label="One donor, area is nuclei"),
                   Line2D([], [], color=C["ink"], lw=1.1, label="Pooled per band")],
          loc="upper left", bbox_to_anchor=(0.01, 1.0), fontsize=F1, handlelength=1.5,
          frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)   # band lines stay behind the text
title(ax, f"{len(dc)} donors, {int(dc.n.sum()):,} nuclei", fontsize=F1)
ax1a, ax1b, ax1c = fig.axes[0], fig.axes[1], fig.axes[2]
for _ax in (ax1a, ax1b, ax1c):
    _ax.tick_params(labelsize=F1); _ax.xaxis.label.set_size(F1); _ax.yaxis.label.set_size(F1)
letters(fig, [[(ax1a, "a"), (ax1b, "b"), (ax1c, "c")]])
norm_text(fig, F1); gate_text(fig, "Figure1", F1)
save(fig, "Figure1")

# ================================================================ SUPPLEMENTARY FIGURE 1
# Location of the transcript: context for the developmental result, shown here and not in the main figures.
# Three panels stacked vertically for a single 8 pt text size, as in Figure 1.
fig = plt.figure(figsize=(120 * MM, 170 * MM))
gsS = fig.add_gridspec(3, 1, height_ratios=[0.82, 1.0, 1.0], left=0.26, right=0.965, top=0.955, bottom=0.055, hspace=0.62)
# ---- 1a  cortical clusters, neuronal vs vascular
ax_a = fig.add_subplot(gsS[0, 0]); clean(ax_a)
ctx = al[al.division == "Cerebral cortex"]
gN = ctx[ctx.cls == "neuronal"].PGR.values; gV = ctx[ctx.cls == "vascular"].PGR.values
tN, tV = kv(SN("allen", "cortex_neuronal")), kv(SN("allen", "cortex_vascular"))
assert (len(gN), int((gN > THR).sum())) == (int(tN["n"]), int(tN["above_0.5"]))
assert (len(gV), int((gV > THR).sum())) == (int(tV["n"]), int(tV["above_0.5"]))
ax_a.scatter(np.clip(rng.normal(0, .09, len(gN)), -.3, .3), gN, s=3, c=C["grey"], alpha=.55, rasterized=True, zorder=2)
ax_a.scatter(1 + np.clip(rng.normal(0, .07, len(gV)), -.3, .3), gV, s=14, c=C["pgr"], ec="white", lw=0.4, zorder=3)
ax_a.axhline(THR, color=C["ink"], lw=0.5, ls=DASH, zorder=1)
for x, g, col, w in [(0, gN, C["mute"], "normal"), (1, gV, C["pgr"], "bold")]:
    ax_a.text(x, 2.42, f"{int((g > THR).sum())}\nabove {THR}", ha="center", va="center", fontsize=NOTE, color=col, fontweight=w, linespacing=1.1)
ax_a.set_xticks([0, 1]); ax_a.set_xticklabels([f"Neuronal\nn = {len(gN)}", f"Vascular\nn = {len(gV)}"])
ax_a.set_xlim(-0.55, 1.55); ax_a.set_ylim(-0.1, 2.6); ax_a.set_yticks([0, 0.5, 1, 1.5, 2])
ax_a.set_ylabel("PGR (counts per 10,000)")
title(ax_a, "Cerebral cortex")

# ---- 1b  four vascular cell types x thirteen genes, colour scaled within row
ax_c = fig.add_subplot(gsS[1, 0])
CT = ["Endo", "PC", "SMC", "VLMC"]; CTL = ["Endo-\nthelium", "Peri-\ncyte", "Smooth\nmuscle", "VLMC"]
nd = kv(SN("psychad", "n_donor"))
BLOCKS = [("Receptor", ["PGR", "PAQR5", "GPER1", "ESR1", "AR"]),
          ("Conversion", ["AKR1C1", "AKR1C2", "AKR1C3", "CYP1B1"]),
          ("Identity", ["CLDN5", "RGS5", "MYH11", "COL1A2"])]
GAP = 0.45; y = 0.0; rows, spans = [], []
for name, gg in BLOCKS:
    M = ps.loc[gg, CT].values; Mn = M / M.max(1, keepdims=True)
    for g in gg:
        t = kv(SN("psychad", g)); assert all(abs(ps.loc[g, c] - float(t[c])) < 1e-4 for c in CT), g
        assert CT[int(np.argmax(ps.loc[g, CT].values))] == t["peak"], g
    ax_c.pcolormesh(np.arange(5), y + np.arange(len(gg) + 1), Mn, cmap=SEQ, vmin=0, vmax=1,
                    edgecolors="white", linewidth=1.0)
    for j, g in enumerate(gg):
        rows.append((y + j + 0.5, g, M[j].max()))
    spans.append((name, y, y + len(gg))); y += len(gg) + GAP
ax_c.set_xlim(0, 4); ax_c.set_ylim(y - GAP, 0)
ax_c.set_yticks([r[0] for r in rows]); ax_c.set_yticklabels([r[1] for r in rows])
for lab, r in zip(ax_c.get_yticklabels(), rows):
    if r[1] == "PGR": lab.set_color(C["pgr"]); lab.set_fontweight("bold")
ax_c.set_xticks(np.arange(4) + 0.5)
ax_c.set_xticklabels([f"{l}\n{int(nd[c]):,}" for l, c in zip(CTL, CT)], linespacing=1.1, fontsize=SMALL)
ax_c.text(0.5, -0.34, "donors per cell type", transform=ax_c.transAxes, ha="center", va="top", fontsize=SMALL, color=C["mute"])
ax_c.tick_params(length=0, pad=3)
for s in ax_c.spines.values(): s.set_visible(False)
trc = blend(ax_c.transAxes, ax_c.transData)
for name, y0, y1 in spans:
    ax_c.annotate(name, xy=(0, (y0 + y1) / 2), xycoords=trc, xytext=(-40, 0), textcoords="offset points",
                  rotation=90, ha="center", va="center", fontsize=SMALL, color=C["mute"])
    ax_c.plot([0, 0], [y0 + 0.1, y1 - 0.1], color=C["rule"], lw=0.6, clip_on=False,
              transform=offset_copy(trc, fig=fig, x=-34 / 72, y=0, units="inches"))
for yc, g, mx in rows:
    ax_c.annotate(f"{mx:.2f}", xy=(1, yc), xycoords=trc, xytext=(6, 0), textcoords="offset points",
                  ha="left", va="center", fontsize=SMALL, color=C["mute"])
ax_c.annotate("Row\nmax", xy=(1, y - GAP), xycoords=trc, xytext=(6, -3), textcoords="offset points",
              ha="left", va="top", fontsize=SMALL, color=C["mute"], linespacing=1.15)
cax = ax_c.inset_axes([0.64, 1.10, 0.34, 0.035])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, 1)), cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
cb.outline.set_linewidth(0.4); cax.xaxis.tick_top()
cb.ax.set_xticklabels(["0", "", "row max"]); cb.ax.tick_params(length=1.5, width=0.4, labelsize=SMALL, pad=1.5)
title(ax_c, "Adult cortex vasculature", y=1.13)

# ---- 1c  steroid synthesis genes
ax_d = fig.add_subplot(gsS[2, 0]); clean(ax_d)
SG = ["HSD3B1", "HSD3B2", "HSD3B7", "CYP11A1", "STAR", "CYP19A1"]
PROG = {"HSD3B1", "HSD3B2"}                     # the 3-beta-HSD step that forms progesterone
det = {g: int((al[g] > DET).sum()) for g in SG}
assert all(det[g] == int(kv(SN("allen", g))["detected"]) for g in SG), det
for i, g in enumerate(SG):
    col = C["pgr"] if g in PROG else C["cmp"]
    ax_d.plot([0, det[g]], [i, i], color=col, lw=1.2, solid_capstyle="butt", zorder=2)
    ax_d.scatter([det[g]], [i], s=16, c="white" if det[g] == 0 else col, ec=col, lw=1.0, zorder=3)
    ax_d.text(det[g] + 6, i, f"{det[g]}", va="center", ha="left", fontsize=SMALL, color=C["ink"])
ax_d.set_yticks(range(len(SG))); ax_d.set_yticklabels(SG)
for lab, g in zip(ax_d.get_yticklabels(), SG):
    if g in PROG: lab.set_color(C["pgr"]); lab.set_fontweight("bold")
ax_d.invert_yaxis(); ax_d.set_ylim(len(SG) - 0.5, -0.6)
ax_d.set_xlim(0, 195); ax_d.set_xticks([0, 50, 100, 150])
ax_d.set_xlabel(f"Clusters with signal, of {len(al):,}")
ax_d.legend(handles=[Line2D([], [], color=C["pgr"], marker="o", ms=3.5, lw=1.2, label="Step that forms progesterone")],
            loc="upper left", bbox_to_anchor=(0.0, -0.22), borderaxespad=0.0)   # below the axis, away from the data labels
title(ax_d, "Steroid synthesis, whole brain")

letters(fig, [[(ax_a, "a")], [(ax_c, "b")], [(ax_d, "c")]])
norm_text(fig, F1); gate_text(fig, "SupplementaryFigure1", F1)
save(fig, "SupplementaryFigure1")
