"""
Figure 7: Dark lipidome annotation via dual mass and retention time filtering.
Panel A: Filtering funnel (bar/waterfall chart)
Panel B: m/z vs RT scatter of recovered annotations overlaid on unannotated features
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Paths
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
GDRIVE = Path("H:/My Drive/0000 Fun with coding/088 Lights-Out R01 Grant/Specific Aim 1/poc3_elution_sequence")
ANNOT_DIR = GDRIVE / "04_dark_lipidome_annotation" / "outputs"
OUTPUT = FIGURES_DIR / "fig7_dark_lipidome.pdf"

# Load annotation summary (threshold sensitivity)
summary = pd.read_csv(ANNOT_DIR / "annotation_summary.csv")

# Try to load the full candidate/annotation data for scatter plot
try:
    unique_annot = pd.read_parquet(ANNOT_DIR / "unique_annotations.parquet")
    has_scatter_data = True
except Exception:
    has_scatter_data = False

try:
    all_candidates = pd.read_parquet(ANNOT_DIR / "all_candidates.parquet")
    has_candidate_data = True
except Exception:
    has_candidate_data = False

fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.3]})

# --- Panel A: Filtering funnel ---
ax = axes[0]

stages = ["Total\nfeatures", "Unannotated", "In m/z\nrange", "Annotated\n(dual filter)"]
values = [15242, 13266, 4700, 168]
colors = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759"]

bars = ax.barh(range(len(stages) - 1, -1, -1), values, color=colors, height=0.6, edgecolor="white")
ax.set_yticks(range(len(stages) - 1, -1, -1))
ax.set_yticklabels(stages, fontsize=10)
ax.set_xlabel("Number of features", fontsize=11)
ax.set_xscale("log")
ax.set_xlim(50, 30000)

for bar, val in zip(bars, values):
    ax.text(val * 1.15, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", ha="left", va="center", fontsize=10, fontweight="bold")

# Add percentage annotations
ax.annotate("87%", xy=(13266, 2.3), fontsize=9, color="#666666", ha="center")
ax.annotate("35%", xy=(4700, 1.3), fontsize=9, color="#666666", ha="center")
ax.annotate("3.6%", xy=(168, 0.3), fontsize=9, color="#666666", ha="center")

ax.set_title("A   Filtering funnel", fontsize=12, fontweight="bold", loc="left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Panel B: m/z vs RT scatter ---
ax = axes[1]

if has_candidate_data and has_scatter_data:
    # Plot all unannotated as gray background
    if "mz" in all_candidates.columns and "rt" in all_candidates.columns:
        ax.scatter(all_candidates["rt"], all_candidates["mz"],
                   s=1, alpha=0.05, color="#cccccc", rasterized=True, label="Unannotated")

    # Plot recovered annotations
    if "mz" in unique_annot.columns and "rt" in unique_annot.columns:
        mz_col, rt_col = "mz", "rt"
    elif "query_mz" in unique_annot.columns and "query_rt" in unique_annot.columns:
        mz_col, rt_col = "query_mz", "query_rt"
    else:
        mz_col = unique_annot.columns[unique_annot.columns.str.contains("mz", case=False)][0]
        rt_col = unique_annot.columns[unique_annot.columns.str.contains("rt", case=False)][0]

    ax.scatter(unique_annot[rt_col], unique_annot[mz_col],
               s=15, alpha=0.7, color="#e15759", edgecolors="black", linewidths=0.3,
               zorder=5, label=f"Recovered (n={len(unique_annot)})")
    ax.legend(fontsize=9, loc="upper left")
elif has_scatter_data:
    if "query_mz" in unique_annot.columns:
        mz_col, rt_col = "query_mz", "query_rt"
    else:
        cols = unique_annot.columns.tolist()
        mz_col = [c for c in cols if "mz" in c.lower()][0]
        rt_col = [c for c in cols if "rt" in c.lower()][0]

    ax.scatter(unique_annot[rt_col], unique_annot[mz_col],
               s=20, alpha=0.7, color="#e15759", edgecolors="black", linewidths=0.3,
               zorder=5, label=f"Recovered (n={len(unique_annot)})")
    ax.legend(fontsize=9, loc="upper left")
else:
    # Fallback: RT threshold sensitivity from summary
    ax.plot(summary["rt_threshold_s"], summary["unique_matches"], "o-",
            color="#e15759", linewidth=2, markersize=8)
    ax.set_xlabel("RT tolerance (s)", fontsize=11)
    ax.set_ylabel("Unique annotations", fontsize=11)
    ax.set_title("B   Threshold sensitivity", fontsize=12, fontweight="bold", loc="left")

if has_scatter_data or has_candidate_data:
    ax.set_xlabel("Retention time (min)", fontsize=11)
    ax.set_ylabel("m/z", fontsize=11)
    ax.set_title("B   Recovered annotations in m/z-RT space", fontsize=12,
                 fontweight="bold", loc="left")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
print(f"Saved: {OUTPUT}")
print(f"Saved: {OUTPUT.with_suffix('.png')}")
plt.close()
