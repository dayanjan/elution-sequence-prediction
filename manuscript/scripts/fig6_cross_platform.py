"""
Figure 6: Cross-platform validation — models are chromatographic-method-specific.
Panel A: ST000983 RT scatter (r=0.9993, same chromatography)
Panel B: Cross-cohort RT scatter (existing asset)
Panel C: ST003514 failure bar chart (5.1% top-1)
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path
from scipy import stats

# Paths
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
ASSETS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figures"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUTPUT = FIGURES_DIR / "fig6_cross_platform.pdf"

# --- Load ST000983 RT comparison data ---
rt_csv = DATA_DIR / "st000983_rt_comparison.csv"
if rt_csv.exists():
    rt_df = pd.read_csv(rt_csv)
    has_rt_data = True
else:
    has_rt_data = False

# Load existing cross-cohort figure
cross_cohort_img = mpimg.imread(ASSETS_DIR / "cross_cohort_rt_consistency.png")

fig = plt.figure(figsize=(16, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8], wspace=0.3)

# --- Panel A: ST000983 RT scatter ---
ax_a = fig.add_subplot(gs[0])
if has_rt_data:
    # Columns: lipid, rt_st983, class, rt_train, rt_diff_s
    # RT values are in minutes; rt_diff_s is precomputed difference in seconds
    x = rt_df["rt_train"].values   # training cohort RT (min)
    y = rt_df["rt_st983"].values   # ST000983 RT (min)

    ax_a.scatter(x, y, s=15, alpha=0.6, color="#1f77b4", edgecolors="none")

    # Regression line
    slope, intercept, r, p, se = stats.linregress(x, y)
    x_line = np.array([x.min(), x.max()])
    ax_a.plot(x_line, slope * x_line + intercept, "r--", linewidth=1)

    # Identity line
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax_a.plot(lim, lim, "k:", linewidth=0.5, alpha=0.5)

    # MAE from precomputed rt_diff_s column (in seconds)
    mae_s = rt_df["rt_diff_s"].abs().mean()
    ax_a.text(0.05, 0.92, f"r = {r:.4f}\nMAE = {mae_s:.1f} s\nn = {len(x)}",
              transform=ax_a.transAxes, fontsize=9, verticalalignment="top",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax_a.set_xlabel("Training cohort RT (min)", fontsize=10)
    ax_a.set_ylabel("ST000983 RT (min)", fontsize=10)
else:
    # Use existing cross-platform figure if available
    cp_img_path = ASSETS_DIR / "cross_platform_rt_correlation.png"
    if cp_img_path.exists():
        cp_img = mpimg.imread(cp_img_path)
        ax_a.imshow(cp_img)
        ax_a.axis("off")
    else:
        ax_a.text(0.5, 0.5, "ST000983 RT data\nnot found", ha="center", va="center",
                  transform=ax_a.transAxes, fontsize=12)

ax_a.set_title("(A) Same chromatography, different MS", fontsize=10, fontweight="bold", loc="left")
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)

# --- Panel B: Cross-cohort RT (existing image) ---
ax_b = fig.add_subplot(gs[1])
ax_b.imshow(cross_cohort_img)
ax_b.axis("off")
ax_b.set_title("(B) Within-training cross-cohort RT", fontsize=10, fontweight="bold", loc="left")

# --- Panel C: ST003514 failure ---
ax_c = fig.add_subplot(gs[2])

models = ["Random\nbaseline", "LSTM\n(ST003514)", "Transf.\n(ST003514)", "LSTM\n(training)"]
accs = [1.1, 5.1, 5.1, 98.4]
colors = ["#bdbdbd", "#e15759", "#e15759", "#1f77b4"]

bars = ax_c.bar(range(4), accs, color=colors, edgecolor="white", width=0.6)
ax_c.set_xticks(range(4))
ax_c.set_xticklabels(models, fontsize=8)
ax_c.set_ylabel("Top-1 Accuracy (%)", fontsize=10)
ax_c.set_ylim(0, 105)

for bar, val in zip(bars, accs):
    ax_c.text(bar.get_x() + bar.get_width() / 2, val + 2,
              f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax_c.set_title("(C) Different chromatography", fontsize=10, fontweight="bold", loc="left")
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
print(f"Saved: {OUTPUT}")
print(f"Saved: {OUTPUT.with_suffix('.png')}")
plt.close()
