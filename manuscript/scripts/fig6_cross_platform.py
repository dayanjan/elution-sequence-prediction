"""
Figure 6: Cross-platform validation — models are chromatographic-method-specific.
A: ST000983 RT scatter (same chromatography, different MS) — identity reference.
B: Within-training cross-cohort RT RESIDUALS (per-cohort RT - cross-cohort mean),
   colored by cohort — shows tightness + per-cohort spread (not a redundant diagonal).
C: Different chromatography failure bars, broken y-axis so the failures are comparable.

Data sources: data/sequences/tokenized_features.parquet (cross-cohort, Panel B),
data/st000983_rt_comparison.csv (Panel A). Panel C values from the external
validation runs (ST003514 different-column; LSTM training reference).
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

BASE = Path(__file__).resolve().parents[2]
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT = FIGURES_DIR / "fig6_cross_platform.pdf"

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
STATBOX = dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5)


def ptitle(ax, t):
    ax.set_title(t, fontsize=FS_TITLE, fontweight="bold", loc="left")


def cross_cohort_points():
    df = pd.read_parquet(BASE / "data/sequences/tokenized_features.parquet")
    feat = df.drop_duplicates(["study", "feature_id"]).copy()
    s = feat["annotation"].astype(str).str.strip()
    feat = feat[feat["annotation"].notna() & s.ne("") & s.str.lower().ne("nan")].copy()
    feat["annotation"] = feat["annotation"].astype(str).str.strip()
    piv = feat.groupby(["annotation", "study"])["rt"].median().unstack("study")
    multi = piv.dropna(thresh=2)
    rows = []
    for _, r in multi.iterrows():
        vals = r.dropna(); ref = vals.mean()
        for st, v in vals.items():
            rows.append((ref, (v - ref) * 60.0, st))   # residual in seconds
    return pd.DataFrame(rows, columns=["ref", "resid_s", "study"]), len(multi)


fig = plt.figure(figsize=(15, 4.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.85], wspace=0.34)

# --- Panel A: ST000983 RT scatter (identity reference only) ---
ax_a = fig.add_subplot(gs[0])
rt_df = pd.read_csv(BASE / "data/st000983_rt_comparison.csv")
x = rt_df["rt_train"].values; y = rt_df["rt_st983"].values
lim = [min(x.min(), y.min()) - 0.3, max(x.max(), y.max()) + 0.3]
ax_a.plot(lim, lim, ":", color=GREY, linewidth=0.8, zorder=1, label="identity (y = x)")
ax_a.scatter(x, y, s=10, alpha=0.45, color=BLUE, edgecolors="none", rasterized=True, zorder=2)
r = stats.pearsonr(x, y)[0]; mae_s = rt_df["rt_diff_s"].abs().mean()
ax_a.text(0.05, 0.95, f"r = {r:.4f}\nMAE = {mae_s:.1f} s\nn = {len(x)} lipids",
          transform=ax_a.transAxes, fontsize=FS_SMALL, va="top", bbox=STATBOX)
ax_a.set_xlim(lim); ax_a.set_ylim(lim); ax_a.set_aspect("equal", adjustable="box")
ax_a.set_xlabel("Training-cohort RT (min)", fontsize=FS_BODY)
ax_a.set_ylabel("ST000983 RT (min)", fontsize=FS_BODY)
ax_a.legend(loc="lower right", fontsize=FS_LEGEND, frameon=False)
ptitle(ax_a, "A   Same chromatography, different MS")
ax_a.spines["top"].set_visible(False); ax_a.spines["right"].set_visible(False)

# --- Panel B: cross-cohort RT residuals (distinct palette, vertical spread) ---
ax_b = fig.add_subplot(gs[1])
P, n_lip = cross_cohort_points()
P = P.sample(frac=1.0, random_state=0)  # shuffle draw order so no cohort sits on top
cohort_color = {"cardiac_arrest": SKY, "gvhd": GREEN, "pcos": ORANGE, "redhart2": PURPLE}
cohort_label = {"cardiac_arrest": "Cardiac arrest", "gvhd": "GVHD", "pcos": "PCOS", "redhart2": "REDHART 1"}
ax_b.axhline(0, color=GREY, linewidth=0.8, linestyle=":", zorder=1)
for st in ["cardiac_arrest", "gvhd", "pcos", "redhart2"]:
    g = P[P["study"] == st]
    ax_b.scatter(g["ref"], g["resid_s"], s=10, alpha=0.5, color=cohort_color[st],
                 edgecolors="none", rasterized=True, label=cohort_label[st], zorder=2)
mae_b = P["resid_s"].abs().mean(); sd_b = P["resid_s"].std()
ax_b.text(0.05, 0.95, f"MAE = {mae_b:.1f} s\nSD = {sd_b:.1f} s\nn = {n_lip} lipids",
          transform=ax_b.transAxes, fontsize=FS_SMALL, va="top", bbox=STATBOX)
span = max(20, np.percentile(P["resid_s"].abs(), 99) * 1.3)
ax_b.set_ylim(-span, span)
ax_b.set_xlabel("Cross-cohort mean RT (min)", fontsize=FS_BODY)
ax_b.set_ylabel("Per-cohort RT residual (s)", fontsize=FS_BODY)
leg = ax_b.legend(loc="lower right", fontsize=FS_LEGEND, frameon=False, handletextpad=0.2, ncol=2)
for h in leg.legend_handles:
    h.set_alpha(1.0)
ptitle(ax_b, "B   Within-training cross-cohort RT")
ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False)

# --- Panel C: failure bars with a broken y-axis ---
gsC = gs[2].subgridspec(2, 1, height_ratios=[1, 2.4], hspace=0.10)
ax_hi = fig.add_subplot(gsC[0]); ax_lo = fig.add_subplot(gsC[1])
labels = ["Random", "LSTM", "Transformer", "LSTM"]
sub = ["", "ST003514", "ST003514", "training"]
accs = [1.5, 5.1, 3.1, 98.4]
colors = [GREY, VERMILLION, VERMILLION, BLUE]
for ax in (ax_hi, ax_lo):
    ax.bar(range(4), accs, color=colors, edgecolor="white", width=0.64)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax_hi.set_ylim(94, 101); ax_lo.set_ylim(0, 7.2)
ax_hi.set_yticks([95, 100]); ax_lo.set_yticks([0, 2, 4, 6])
ax_hi.spines["bottom"].set_visible(False)
ax_hi.tick_params(bottom=False, labelbottom=False)
ax_lo.set_xticks(range(4))
ax_lo.set_xticklabels([f"{l}\n{s}" if s else l for l, s in zip(labels, sub)], fontsize=FS_TICK - 0.5)
d = 0.012
for ax in (ax_hi, ax_lo):
    y0 = 0 if ax is ax_hi else 1
    ax.plot([-0.04, 0.04], [y0 - d * 2.4, y0 + d * 2.4], transform=ax.transAxes,
            color=DARK, clip_on=False, lw=0.9)
for i, v in enumerate(accs):
    tgt = ax_lo if v < 90 else ax_hi
    tgt.text(i, v + 0.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=FS_SMALL, fontweight="bold")
ax_lo.set_ylabel("Top-1 accuracy (%)", fontsize=FS_BODY)
ax_lo.yaxis.set_label_coords(-0.16, 0.72)
ptitle(ax_hi, "C   Different chromatography")

fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")
print(f"Saved: {OUT.with_suffix('.png')}")
