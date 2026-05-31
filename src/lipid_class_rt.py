"""Lipid class RT structure visualization across 4 clinical lipidomics cohorts."""
import sys
sys.path.insert(0, "src")

import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from preprocessing import load_all_datasets


def extract_class(annotation):
    if pd.isna(annotation) or annotation == "":
        return None
    m = re.match(r"^([A-Za-z]+)", str(annotation).strip())
    return m.group(1) if m else None


def make_lipid_class_rt_figure(output_path="outputs/figures/lipid_class_rt_structure.png"):
    df = load_all_datasets()
    ann = df[df["annotation"].notna() & (df["annotation"] != "")].copy()
    ann["lipid_class"] = ann["annotation"].apply(extract_class)

    # Get unique feature-level data
    feat = ann.groupby("feature_id").first().reset_index()

    # Merge ceramide variants, drop non-lipid classes
    class_map = {"Cer": "Ceramide", "GlcCer": "Ceramide", "Gal": "Ceramide",
                 "Lactosylceramide": "Ceramide"}
    feat["lipid_class"] = feat["lipid_class"].replace(class_map)
    feat = feat[~feat["lipid_class"].isin(["iSTD", "CSH", "Cholesterol"])]

    # Keep classes with >= 10 features
    class_counts = feat["lipid_class"].value_counts()
    keep = class_counts[class_counts >= 10].index
    feat = feat[feat["lipid_class"].isin(keep)]

    # Order by median RT
    class_order = feat.groupby("lipid_class")["rt"].median().sort_values().index.tolist()

    print("Classes ordered by median RT:")
    for c in class_order:
        subset = feat[feat["lipid_class"] == c]
        print(f"  {c:20s}: median RT={subset['rt'].median():.2f} min, n={len(subset)}")

    # Prepare data
    data_by_class = [feat[feat["lipid_class"] == c]["rt"].values for c in class_order]
    positions = list(range(len(class_order)))
    medians = [np.median(d) for d in data_by_class]
    rt_min, rt_max = min(medians), max(medians)

    # Figure
    fig, ax = plt.subplots(figsize=(12, 5))

    # Violin plot
    cmap = plt.cm.RdYlBu_r
    parts = ax.violinplot(data_by_class, positions=positions, showmedians=True, showextrema=False)

    for i, body in enumerate(parts["bodies"]):
        color = cmap((medians[i] - rt_min) / (rt_max - rt_min))
        body.set_facecolor(color)
        body.set_alpha(0.7)
        body.set_edgecolor("black")
        body.set_linewidth(0.5)

    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    # Overlay jittered points
    rng = np.random.default_rng(42)
    for i, d in enumerate(data_by_class):
        jitter = rng.uniform(-0.2, 0.2, size=len(d))
        color = cmap((medians[i] - rt_min) / (rt_max - rt_min))
        ax.scatter(np.full_like(d, i) + jitter, d, s=8, alpha=0.4,
                   color=color, edgecolors="none")

    # Labels with counts
    labels = [f"{c}\n(n={len(data_by_class[i])})" for i, c in enumerate(class_order)]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Retention time (min)", fontsize=11)
    ax.set_title("Lipid class retention time distributions on RP-LC",
                 fontsize=13, fontweight="bold")

    # Annotate chromatographic logic
    ax.annotate("Polar / hydrophilic", xy=(0.02, 0.95), xycoords="axes fraction",
                fontsize=9, color="#2166ac", fontstyle="italic", ha="left", va="top")
    ax.annotate("Hydrophobic", xy=(0.98, 0.95), xycoords="axes fraction",
                fontsize=9, color="#b2182b", fontstyle="italic", ha="right", va="top")
    ax.annotate("", xy=(0.98, 0.90), xytext=(0.02, 0.90), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

    # Summary box
    n_classes = len(class_order)
    n_features = len(feat)
    ax.text(0.98, 0.05,
            f"{n_features:,} annotated features | {n_classes} lipid classes\n"
            f"4 cohorts combined | RP-LC (CSH column)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    make_lipid_class_rt_figure()
