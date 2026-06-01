"""
Figure 2: Physicochemical basis of elution order in reversed-phase lipidomics.
Native (vector) generator -- replaces the prior raster composite of two
pre-rendered PNGs, and restores a committed, auditable generator for the figure.

Panel A: lipid-class RT distributions on RP-LC (violin), ordered by median RT,
         colored by a shared polar->hydrophobic RT gradient (lipid_class_rt.py logic).
Panel B: head-group RT residual from the ECN model (box) per head-group class,
         ordered by mean offset, colored by the SAME RT gradient so each class
         keeps its color across panels (feature_analysis.py logic: residual =
         RT minus the global linear RT~ECN fit). Reproduces the published
         offsets exactly (LPC = -169 s ... CE = +389 s; ECN model fit n = 1,752).

Data: src/preprocessing.load_all_datasets() -- the 4 RP-LC cohorts. The raw
feature tables are controlled-access (DUA), like the rest of the manuscript's
clinical-data analyses; this script (with lipid_class_rt.py + feature_analysis.py)
is the committed, auditable record and regenerates the figure with data access.
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT = FIGURES_DIR / "fig2_elution_physics.pdf"
sys.path.insert(0, str(BASE / "src"))

from preprocessing import load_all_datasets  # noqa: E402

BLUE = "#0072B2"; ORANGE = "#E69F00"; GREEN = "#009E73"; VERMILLION = "#D55E00"
PURPLE = "#CC79A7"; SKY = "#56B4E9"; GREY = "#7A7A7A"; DARK = "#1A1A1A"
FS_TITLE = 10.5; FS_BODY = 9.0; FS_SMALL = 8.0; FS_TICK = 8.0; FS_LEGEND = 7.5

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
    "figure.facecolor": "white", "savefig.facecolor": "white", "axes.facecolor": "white",
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
})
GRAD = plt.cm.RdYlBu_r   # blue = polar/early, red = hydrophobic/late

# Panel-B head-group set (the published 12 classes; LPC is the floor, CE the ceiling)
HEADGROUPS = ["LPC", "FA", "LPE", "PE", "PC", "SM", "PI", "DG", "GlcCer", "Ceramide", "TG", "CE"]


def ptitle(ax, t):
    ax.set_title(t, fontsize=FS_TITLE, fontweight="bold", loc="left")


def extract_class(a):
    if pd.isna(a) or a == "":
        return None
    m = re.match(r"^([A-Za-z]+)", str(a).strip())
    return m.group(1) if m else None


df = load_all_datasets()

# ---------- Panel A data (lipid_class_rt.py logic) ----------
annA = df[df["annotation"].notna() & (df["annotation"] != "")].copy()
annA["lc"] = annA["annotation"].apply(extract_class)
fA = annA.groupby("feature_id").first().reset_index()
fA["lc"] = fA["lc"].replace({"Cer": "Ceramide", "GlcCer": "Ceramide",
                             "Gal": "Ceramide", "Lactosylceramide": "Ceramide"})
fA = fA[~fA["lc"].isin(["iSTD", "CSH", "Cholesterol"])]
vc = fA["lc"].value_counts()
fA = fA[fA["lc"].isin(vc[vc >= 10].index)]
orderA = fA.groupby("lc")["rt"].median().sort_values().index.tolist()
dataA = [fA[fA["lc"] == c]["rt"].values for c in orderA]
medA = [float(np.median(d)) for d in dataA]

# ---------- Panel B data (feature_analysis.py logic) ----------
feat = df.drop_duplicates(subset=["study", "feature_id"]).copy()
ann = feat[feat.total_carbons.notna() & (~feat.is_istd) & feat.lipid_class.notna()].copy()
ann["ecn"] = ann.total_carbons - 2 * ann.total_unsat
coeffs = np.polyfit(ann.ecn, ann.rt, 1)                       # global linear RT ~ ECN fit
ann["resid"] = ann.rt - np.polyval(coeffs, ann.ecn)           # residual after removing ECN (min)
ann["lc"] = ann.lipid_class.replace({"Cer": "Ceramide"})
n_model = len(ann)                                            # ECN-model fit population (= 1,752)
annB = ann[ann["lc"].isin(HEADGROUPS)].copy()
orderB = annB.groupby("lc")["resid"].mean().sort_values().index.tolist()
dataB = [annB[annB["lc"] == c]["resid"].values for c in orderB]
medRT_B = [float(np.median(annB[annB["lc"] == c]["rt"].values)) for c in orderB]

# ---------- shared color scale: color == median RT, identical across panels ----------
GLO = min(min(medA), min(medRT_B))
GHI = max(max(medA), max(medRT_B))
def cnorm(v):
    return GRAD((v - GLO) / (GHI - GLO))

# ====================================================================
fig = plt.figure(figsize=(14, 5.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.20)

# ---------- Panel A: violins ----------
ax_a = fig.add_subplot(gs[0])
pos = list(range(len(orderA)))
parts = ax_a.violinplot(dataA, positions=pos, showmedians=True, showextrema=False, widths=0.85)
for i, body in enumerate(parts["bodies"]):
    body.set_facecolor(cnorm(medA[i]))
    body.set_alpha(0.75); body.set_edgecolor(DARK); body.set_linewidth(0.5)
parts["cmedians"].set_color(DARK); parts["cmedians"].set_linewidth(1.6)
rng = np.random.default_rng(42)
for i, d in enumerate(dataA):
    ax_a.scatter(np.full_like(d, i) + rng.uniform(-0.18, 0.18, size=len(d)), d,
                 s=5, alpha=0.35, color=cnorm(medA[i]), edgecolors="none", rasterized=True)
ax_a.set_xticks(pos)
ax_a.set_xticklabels([f"{c} (n={len(dataA[i])})" for i, c in enumerate(orderA)],
                     fontsize=FS_TICK, rotation=30, ha="right", rotation_mode="anchor")
ax_a.set_ylabel("Retention time (min)", fontsize=FS_BODY)
top_a = max(np.max(d) for d in dataA)
ax_a.set_ylim(0, top_a * 1.20)   # headroom for the polar->hydrophobic annotation band
ax_a.annotate("Polar / hydrophilic", xy=(0.02, 0.985), xycoords="axes fraction",
              fontsize=FS_SMALL, color="#2166AC", fontstyle="italic", ha="left", va="top")
ax_a.annotate("Hydrophobic", xy=(0.98, 0.985), xycoords="axes fraction",
              fontsize=FS_SMALL, color="#B2182B", fontstyle="italic", ha="right", va="top")
ax_a.annotate("", xy=(0.985, 0.915), xytext=(0.015, 0.915), xycoords="axes fraction",
              arrowprops=dict(arrowstyle="->", color=GREY, lw=1.3))
ax_a.text(0.98, 0.04,
          f"{len(fA):,} annotated features | {len(orderA)} lipid classes\n4 cohorts combined | RP-LC (CSH column)",
          transform=ax_a.transAxes, ha="right", va="bottom", fontsize=FS_SMALL - 0.5,
          bbox=dict(boxstyle="round,pad=0.3", facecolor="#F2F2F2", edgecolor="none"))
ptitle(ax_a, "A   Lipid-class retention time on RP-LC")
ax_a.spines["top"].set_visible(False); ax_a.spines["right"].set_visible(False)

# ---------- Panel B: ECN-residual boxes ----------
ax_b = fig.add_subplot(gs[1])
posB = list(range(len(orderB)))
ax_b.axhline(0, color=DARK, linewidth=1.1, linestyle=":", zorder=1)
bp = ax_b.boxplot(dataB, positions=posB, widths=0.62, patch_artist=True,
                  showfliers=True, flierprops=dict(marker="o", markersize=2.2,
                  markerfacecolor=GREY, markeredgecolor="none", alpha=0.4),
                  medianprops=dict(color=DARK, linewidth=1.4),
                  whiskerprops=dict(color=DARK, linewidth=0.8),
                  capprops=dict(color=DARK, linewidth=0.8),
                  boxprops=dict(linewidth=0.6, edgecolor=DARK))
for i, box in enumerate(bp["boxes"]):
    box.set_facecolor(cnorm(medRT_B[i])); box.set_alpha(0.85)
ax_b.set_xticks(posB)
ax_b.set_xticklabels([f"{c} (n={len(dataB[i])})" for i, c in enumerate(orderB)],
                     fontsize=FS_TICK, rotation=30, ha="right", rotation_mode="anchor")
ax_b.set_ylabel("RT residual from ECN model (min)", fontsize=FS_BODY)
ax_b.text(0.02, 0.96, "ECN explains $R^2$ = 0.27 alone;\nhead group adds the offsets shown",
          transform=ax_b.transAxes, ha="left", va="top", fontsize=FS_SMALL,
          bbox=dict(boxstyle="round,pad=0.3", facecolor="#F2F2F2", edgecolor="none"))
ptitle(ax_b, "B   Head-group offset from the ECN trend")
ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False)

fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Panel A: {len(fA)} features, {len(orderA)} classes")
print(f"Panel B: ECN-model fit n = {n_model}; plotted classes = {len(orderB)}")
print(f"Panel B offsets (s): LPC={np.mean(annB[annB.lc=='LPC'].resid)*60:+.0f}  CE={np.mean(annB[annB.lc=='CE'].resid)*60:+.0f}")
print(f"Saved: {OUT}")
print(f"Saved: {OUT.with_suffix('.png')}")
