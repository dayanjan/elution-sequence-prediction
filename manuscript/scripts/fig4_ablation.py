"""
Figure 4: Ablation study — feature importance and data scaling.
Panel A: Horizontal bar chart showing accuracy drop when each feature is removed.
Panel B: Data scaling curve (25/50/75/100% training data vs top-1 accuracy).
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
GDRIVE = Path("H:/My Drive/0000 Fun with coding/088 Lights-Out R01 Grant/Specific Aim 1/poc3_elution_sequence")
RESULTS_JSON = GDRIVE / "07_ablation_study" / "outputs" / "ablation_results.json"
OUTPUT_PDF = FIGURES_DIR / "fig4_ablation.pdf"
OUTPUT_PNG = FIGURES_DIR / "fig4_ablation.png"

# --- Load data ---
with open(RESULTS_JSON) as f:
    data = json.load(f)

full_top1 = data["full"]["test_top1"] * 100

# --- Panel A: Feature ablation ---
# Order: biggest drop first, no_sequence at bottom (separated)
feature_conditions = [
    ("−m/z bin", data["no_mz"]["test_top1"] * 100),
    ("−mass defect", data["no_md"]["test_top1"] * 100),
    ("−polarity", data["no_polarity"]["test_top1"] * 100),
    ("−intensity rank", data["no_intensity"]["test_top1"] * 100),
    ("−RT gap", data["no_rt_gap"]["test_top1"] * 100),
]
# Sort by drop magnitude (largest drop first)
feature_conditions.sort(key=lambda x: x[1])

feature_names = [c[0] for c in feature_conditions]
feature_drops = [full_top1 - c[1] for c in feature_conditions]

no_seq_drop = full_top1 - data["no_sequence"]["test_top1"] * 100

# Data scaling
scale_fracs = [25, 50, 75, 100]
scale_top1 = [
    data["data_25pct"]["test_top1"] * 100,
    data["data_50pct"]["test_top1"] * 100,
    data["data_75pct"]["test_top1"] * 100,
    full_top1,
]

# --- Plot ---
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.3, 1]})

# Panel A: Feature importance as accuracy drop
# Individual features
y_pos = np.arange(len(feature_names))
bars = ax_a.barh(y_pos + 1, feature_drops, height=0.6, color="#4292c6", edgecolor="white")
ax_a.set_yticks(np.concatenate([[0], y_pos + 1]))
ax_a.set_yticklabels(["No sequence\n(ctx = 1)"] + feature_names, fontsize=10)

# No-sequence bar (visually separated, different color)
ax_a.barh(0, no_seq_drop, height=0.6, color="#d62728", edgecolor="white")

# Add value labels
for i, v in enumerate(feature_drops):
    ax_a.text(v + 0.3, i + 1, f"−{v:.2f} pp", va="center", fontsize=9)
ax_a.text(no_seq_drop + 0.3, 0, f"−{no_seq_drop:.1f} pp", va="center", fontsize=9, fontweight="bold")

ax_a.set_xlabel("Accuracy drop from full model (pp)", fontsize=11)
ax_a.set_title("A  Feature ablation", fontsize=12, fontweight="bold", loc="left")
ax_a.set_xlim(0, no_seq_drop * 1.15)

# Add break indicator on x-axis
ax_a.axvline(x=1.0, color="#cccccc", linestyle=":", linewidth=0.8)

# Horizontal separator between no-sequence and features
ax_a.axhline(y=0.55, color="#999999", linestyle="--", linewidth=0.8)

ax_a.invert_yaxis()
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)

# Panel B: Data scaling curve
ax_b.plot(scale_fracs, scale_top1, "o-", color="#1f77b4", linewidth=2, markersize=8, zorder=3)
ax_b.fill_between(scale_fracs, min(scale_top1) - 0.05, scale_top1, alpha=0.1, color="#1f77b4")

# Annotate points
for frac, acc in zip(scale_fracs, scale_top1):
    ax_b.annotate(f"{acc:.2f}%", (frac, acc), textcoords="offset points",
                  xytext=(0, 12), ha="center", fontsize=9)

ax_b.set_xlabel("Training data (%)", fontsize=11)
ax_b.set_ylabel("Top-1 accuracy (%)", fontsize=11)
ax_b.set_title("B  Data scaling", fontsize=12, fontweight="bold", loc="left")
ax_b.set_xticks(scale_fracs)
ax_b.set_ylim(98.1, 98.5)
ax_b.set_xlim(15, 110)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)
ax_b.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(OUTPUT_PDF, bbox_inches="tight", dpi=300)
fig.savefig(OUTPUT_PNG, bbox_inches="tight", dpi=300)
print(f"Saved: {OUTPUT_PDF}")
print(f"Saved: {OUTPUT_PNG}")
plt.close()
