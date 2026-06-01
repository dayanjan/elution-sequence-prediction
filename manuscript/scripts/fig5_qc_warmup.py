"""
Figure 5: QC warm-up experiment — a null result.
Panel A: Dose-response (accuracy vs N QC injections)
Panel B: 4-condition comparison
Panel C: Control conditions
Panel D: Delta accuracy (carry-hidden minus cold start)
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
GDRIVE = Path("H:/My Drive/0000 Fun with coding/088 Lights-Out R01 Grant/Specific Aim 1/poc3_elution_sequence")
QC_JSON = GDRIVE / "03_qc_warmup" / "outputs" / "phase4_results.json"
OUTPUT = FIGURES_DIR / "fig5_qc_warmup.pdf"

with open(QC_JSON) as f:
    qc = json.load(f)

# Extract dose-response data
prime_doses = sorted(qc["prime_only_dose_response"].keys(), key=int)
carry_doses = sorted(qc["carry_hidden_dose_response"].keys(), key=int)

prime_top1 = [qc["prime_only_dose_response"][d]["top1"] * 100 for d in prime_doses]
carry_top1 = [qc["carry_hidden_dose_response"][d]["top1"] * 100 for d in carry_doses]
doses = [int(d) for d in prime_doses]

# Controls
ctrl_analytical = qc["controls"]["analytical_warmup"]
ctrl_cross = qc["controls"]["cross_cohort"]
ctrl_shuffled = qc["controls"]["shuffled"]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# --- Panel A: Dose-response ---
ax = axes[0]
ax.plot(doses, prime_top1, "o-", color="#1f77b4", linewidth=1.5, markersize=5, label="Prime-only")
ax.plot(doses, carry_top1, "s-", color="#ff7f0e", linewidth=1.5, markersize=5, label="Carry-hidden")
ax.set_xlabel("N QC injections", fontsize=10)
ax.set_ylabel("Top-1 Accuracy (%)", fontsize=10)
ax.set_ylim(98.39, 98.43)
ax.legend(fontsize=8)
ax.set_title("A   Dose-response", fontsize=11, fontweight="bold", loc="left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Panel B: 4-condition comparison ---
ax = axes[1]
cold_start = qc["prime_only_dose_response"]["0"]["top1"] * 100
prime_10 = qc["prime_only_dose_response"]["10"]["top1"] * 100
carry_10 = qc["carry_hidden_dose_response"]["10"]["top1"] * 100
# "Both" = carry with 10 priming (same as carry_10 since carry subsumes prime)
both_10 = carry_10  # They're essentially the same in this data

conditions = ["Cold\nstart", "Prime\nonly", "Carry\nhidden", "Both"]
values = [cold_start, prime_10, carry_10, both_10]
colors = ["#bdbdbd", "#1f77b4", "#ff7f0e", "#2ca02c"]

bars = ax.bar(range(4), values, color=colors, edgecolor="white", width=0.6)
ax.set_xticks(range(4))
ax.set_xticklabels(conditions, fontsize=8)
ax.set_ylabel("Top-1 Accuracy (%)", fontsize=10)
ax.set_ylim(98.39, 98.43)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
            f"{val:.3f}%", ha="center", va="bottom", fontsize=8)

ax.set_title("B   Conditioning modes", fontsize=11, fontweight="bold", loc="left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Panel C: Control conditions ---
ax = axes[2]
ctrl_names = ["Same-\ncohort QC", "Analytical\nsample", "Cross-\ncohort QC", "Shuffled\nQC"]
ctrl_vals = [
    prime_10,  # same-cohort QC
    ctrl_analytical.get("4", ctrl_analytical.get("8", 0)) * 100,
    ctrl_cross.get("4", ctrl_cross.get("8", 0)) * 100,
    ctrl_shuffled.get("4", ctrl_shuffled.get("8", 0)) * 100,
]
ctrl_colors = ["#1f77b4", "#9467bd", "#d62728", "#8c564b"]

bars = ax.bar(range(4), ctrl_vals, color=ctrl_colors, edgecolor="white", width=0.6)
ax.set_xticks(range(4))
ax.set_xticklabels(ctrl_names, fontsize=8)
ax.set_ylabel("Top-1 Accuracy (%)", fontsize=10)
ax.set_ylim(98.39, 98.43)

for bar, val in zip(bars, ctrl_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
            f"{val:.3f}%", ha="center", va="bottom", fontsize=8)

ax.set_title("C   Control conditions", fontsize=11, fontweight="bold", loc="left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Panel D: Delta accuracy ---
ax = axes[3]
deltas = [c - p for c, p in zip(carry_top1, prime_top1)]
ax.bar(doses, deltas, color="#2ca02c", edgecolor="white", width=1.5, alpha=0.7)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("N QC injections", fontsize=10)
ax.set_ylabel("Delta (carry - cold) (%)", fontsize=10)
ax.set_title("D   Carry-hidden benefit", fontsize=11, fontweight="bold", loc="left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Format y-axis in scientific notation range
max_delta = max(abs(d) for d in deltas) if any(d != 0 for d in deltas) else 0.01
ax.set_ylim(-max_delta * 2, max_delta * 2)

plt.tight_layout()
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
print(f"Saved: {OUTPUT}")
print(f"Saved: {OUTPUT.with_suffix('.png')}")
plt.close()
