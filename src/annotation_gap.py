"""Annotation gap visualization across 4 clinical lipidomics cohorts."""
import sys
sys.path.insert(0, "src")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from preprocessing import load_all_datasets

def make_annotation_gap_figure(output_path="outputs/figures/annotation_gap.png"):
    df = load_all_datasets()

    # Compute per-study stats
    studies = []
    for study, g in df.groupby("study"):
        total = g["feature_id"].nunique()
        annotated = g[g["annotation"].notna() & (g["annotation"] != "")]["feature_id"].nunique()
        unannotated = total - annotated
        nice_name = {
            "cardiac_arrest": "Cardiac\nArrest",
            "gvhd": "GVHD",
            "pcos": "PCOS",
            "redhart2": "REDHART 2",
        }.get(study, study)
        studies.append({
            "name": nice_name, "total": total,
            "annotated": annotated, "unannotated": unannotated,
            "pct_annotated": annotated / total * 100,
            "pct_unannotated": unannotated / total * 100,
        })

    # Sort by total features
    studies.sort(key=lambda s: s["total"])

    names = [s["name"] for s in studies]
    ann = [s["annotated"] for s in studies]
    unann = [s["unannotated"] for s in studies]
    pct_unann = [s["pct_unannotated"] for s in studies]

    # Figure
    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(names))
    width = 0.55

    bars_unann = ax.bar(x, unann, width, label="Unannotated", color="#d62728", alpha=0.85)
    bars_ann = ax.bar(x, ann, width, bottom=unann, label="Annotated", color="#2ca02c", alpha=0.85)

    # Labels on each bar
    for i, s in enumerate(studies):
        # Percentage label in the unannotated portion
        ax.text(i, s["unannotated"] / 2, f"{s['pct_unannotated']:.0f}%",
                ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        # Total count on top
        ax.text(i, s["total"] + 80, f"n={s['total']:,}",
                ha="center", va="bottom", fontsize=8, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Number of detected features", fontsize=11)
    ax.set_title("The annotation gap in clinical lipidomics", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_ylim(0, max(s["total"] for s in studies) * 1.12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Overall summary text
    total_all = sum(s["total"] for s in studies)
    ann_all = sum(s["annotated"] for s in studies)
    ax.text(0.98, 0.02,
            f"Overall: {ann_all:,}/{total_all:,} annotated ({ann_all/total_all*100:.0f}%)\n"
            f"4 cohorts · 342 samples · RP-LC/MS",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    make_annotation_gap_figure()
