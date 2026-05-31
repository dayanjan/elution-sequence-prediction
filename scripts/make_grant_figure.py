"""Create grant-quality preliminary data figure for SA1.

Two-panel figure:
  (A) Leave-one-cohort-out RT prediction scatter (4 sub-panels)
  (B) Feature importance bar chart (top features only)

Uses existing Colab-generated PNGs, composited with proper labels.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

GDRIVE_OUTPUTS = os.path.join(
    os.path.dirname(__file__), "..", "gdrive", "poc2_rt_prediction", "outputs"
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
PROPOSAL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "..",
    "proposal", "04_research_strategy", "figures", "proposal-figures"
)

def make_figure():
    scatter_path = os.path.join(GDRIVE_OUTPUTS, "predicted_vs_actual_rt.png")
    importance_path = os.path.join(GDRIVE_OUTPUTS, "feature_importance.png")

    if not os.path.exists(scatter_path):
        print(f"ERROR: {scatter_path} not found")
        return
    if not os.path.exists(importance_path):
        print(f"ERROR: {importance_path} not found")
        return

    scatter_img = mpimg.imread(scatter_path)
    importance_img = mpimg.imread(importance_path)

    # Create figure: scatter on top (wide), importance below-right
    fig = plt.figure(figsize=(7.5, 5.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.1, 1], width_ratios=[1.2, 1],
                           hspace=0.25, wspace=0.2)

    # Panel A: scatter (spans full top row)
    ax_scatter = fig.add_subplot(gs[0, :])
    ax_scatter.imshow(scatter_img)
    ax_scatter.axis("off")
    ax_scatter.text(-0.02, 1.02, "A", transform=ax_scatter.transAxes,
                    fontsize=9, fontweight="bold", va="top", ha="right")

    # Panel B: feature importance (bottom left, larger)
    ax_importance = fig.add_subplot(gs[1, 0])
    ax_importance.imshow(importance_img)
    ax_importance.axis("off")
    ax_importance.text(-0.02, 1.05, "B", transform=ax_importance.transAxes,
                       fontsize=9, fontweight="bold", va="top", ha="right")

    # Panel C: summary table (bottom right)
    ax_table = fig.add_subplot(gs[1, 1])
    ax_table.axis("off")

    table_data = [
        ["GBM + RDKit", "0.102", "0.996", "99.7%"],
        ["GBM + class", "0.112", "0.997", "99.9%"],
        ["RF + RDKit", "0.111", "0.996", "99.7%"],
        ["RF + class", "0.117", "0.996", "99.9%"],
        ["GBM (basic)", "0.139", "0.993", "99.6%"],
    ]
    col_labels = ["Model", "MAE\n(min)", "R\u00b2", "\u00b11 min"]

    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.38, 0.18, 0.18, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")

    # Highlight best row
    for j in range(len(col_labels)):
        cell = table[1, j]
        cell.set_facecolor("#D6E4F0")

    # Alternate row colors
    for i in range(2, len(table_data) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            cell.set_facecolor("#F2F2F2" if i % 2 == 0 else "white")

    ax_table.set_title("Cross-Cohort CV Summary", fontsize=9, fontweight="bold", pad=8)
    ax_table.text(-0.05, 1.05, "C", transform=ax_table.transAxes,
                  fontsize=9, fontweight="bold", va="top", ha="right")

    # Save to both locations
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PROPOSAL_DIR, exist_ok=True)

    out1 = os.path.join(OUTPUT_DIR, "fig_prelim_rt_prediction.png")
    out2 = os.path.join(PROPOSAL_DIR, "fig4_prelim_rt_prediction.png")

    fig.savefig(out1, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out2, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    make_figure()
