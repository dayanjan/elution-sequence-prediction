"""
Figure 9: Transfer learning recovers cross-polarity performance on ST000990.
Single panel: held-out top-1 accuracy vs number of QC calibration injections N,
for full fine-tune / head-only fine-tune / Markov-only (all evaluated on the
strictly held-out ANALYTICAL samples), with the 2.6% zero-shot floor (dashed) and
98.4% within-method ceiling (dotted). A separate vermillion star marks the 99.6%
held-out-QC accuracy at N=15 (a DIFFERENT evaluation set), showing the
representation-diversity gap between QC and analytical recovery.

Data source: outputs/phase9_transfer/phase9_gpu_results.json (the GPU run that
reproduces manuscript Table 6).
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
DATA_JSON = BASE / "outputs/phase9_transfer/phase9_gpu_results.json"
OUT = FIGURES_DIR / "fig9_transfer_recovery.pdf"

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

# --- load ---
with open(DATA_JSON) as f:
    R = json.load(f)

N_VALUES = [1, 2, 5, 10, 15]
floor = R["zero_shot_all"]["top1"] * 100                       # 2.6%
ceiling = R["reference"]["within_method_ceiling_top1"] * 100   # 98.4%


def series(method):
    """Pull held-out analytical top-1 (%) at each N."""
    out = []
    for N in N_VALUES:
        if method == "full_finetune":
            rec = next(r for r in R["full_finetune"] if r["N"] == N)
            out.append(rec["eval_top1"] * 100)
        else:
            rec = next(r for r in R["recovery_table"]
                       if r["method"] == method and r["N"] == N)
            out.append(rec["top1"] * 100)
    return out


full = series("full_finetune")
head = series("fine_tune_head")
markov = series("markov_only")
qc_n15 = next(r for r in R["full_finetune"] if r["N"] == 15)["heldout_qc_top1"] * 100  # 99.6%

x = np.arange(len(N_VALUES))   # categorical: equal spacing for 1,2,5,10,15

fig, ax = plt.subplots(figsize=(7.2, 4.6))

# reference lines
ax.axhline(floor, color=GREY, linewidth=1.0, linestyle="--", zorder=1)
ax.axhline(ceiling, color=DARK, linewidth=1.0, linestyle=":", zorder=1)
ax.text(x[-1], floor + 1.5, f"Zero-shot floor ({floor:.1f}%)",
        ha="right", va="bottom", fontsize=FS_SMALL, color=GREY)
ax.text(x[0], ceiling - 2.0, f"Within-method ceiling ({ceiling:.1f}%)",
        ha="left", va="top", fontsize=FS_SMALL, color=DARK)

# recovery curves (held-out analytical)
ax.plot(x, full, "-o", color=BLUE, linewidth=2.0, markersize=7,
        markeredgecolor="white", markeredgewidth=0.8, label="Full fine-tune", zorder=4)
ax.plot(x, head, "-s", color=ORANGE, linewidth=1.8, markersize=6,
        markeredgecolor="white", markeredgewidth=0.8, label="Head-only fine-tune", zorder=3)
ax.plot(x, markov, "-^", color=GREEN, linewidth=1.8, markersize=6,
        markeredgecolor="white", markeredgewidth=0.8, label="Markov-only", zorder=3)

# value labels on the full fine-tune line
for xi, v in zip(x, full):
    ax.annotate(f"{v:.0f}%", (xi, v), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=FS_SMALL, fontweight="bold", color=BLUE)

# 99.6% held-out QC callout at N=15 (vermillion: a DIFFERENT eval set from the
# blue analytical curve — distinct color so it cannot read as the curve terminus)
ax.plot(x[-1], qc_n15, marker="*", color=VERMILLION, markersize=16,
        markeredgecolor="white", markeredgewidth=0.8, zorder=5)
ax.annotate(f"Held-out QC: {qc_n15:.1f}%", (x[-1], qc_n15),
            textcoords="offset points", xytext=(-12, 8), ha="right", va="bottom",
            fontsize=FS_SMALL, color=VERMILLION, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([str(n) for n in N_VALUES])
ax.set_xlim(-0.4, len(N_VALUES) - 0.4)
ax.set_ylim(0, 106)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_xlabel("QC calibration injections ($N$)", fontsize=FS_BODY)
ax.set_ylabel("Held-out top-1 accuracy (%)", fontsize=FS_BODY)
ax.legend(loc="upper left", bbox_to_anchor=(0.30, 0.82), fontsize=FS_LEGEND,
          frameon=False, handletextpad=0.5, labelspacing=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")
print(f"Saved: {OUT.with_suffix('.png')}")
