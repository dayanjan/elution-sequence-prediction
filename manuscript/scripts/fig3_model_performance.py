"""
Figure 3: Model performance and dataset context.
Panel A: Bar chart — top-1 accuracy for baselines vs LSTM vs Transformer
Panel B: Learning curves — val top-1 and val loss vs epoch
Panel C: Annotation gap — stacked bars from existing annotation_gap.png
"""
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path

# Paths
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
ASSETS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figures"
GDRIVE = Path("H:/My Drive/0000 Fun with coding/088 Lights-Out R01 Grant/Specific Aim 1/poc3_elution_sequence")
RESULTS_JSON = GDRIVE / "01_train_models" / "outputs" / "neural_model_results.json"
OUTPUT = FIGURES_DIR / "fig3_model_performance.pdf"

# --- Load data ---
with open(RESULTS_JSON) as f:
    results = json.load(f)

lstm_history = results["lstm"]["history"]
transformer_history = results["transformer"]["history"]
lstm_test = results["lstm"]["test_metrics"]
transformer_test = results["transformer"]["test_metrics"]

# Baseline results (from manuscript Table 1)
baselines = {
    "Random": 1.1,
    "Global\nfrequency": 3.4,
    "Same-as-\nprevious": 23.1,
    "Markov\norder-1": 25.1,
    "Joint\n(RT,m/z)": 56.8,
}

# --- Panel A: Bar chart ---
fig = plt.figure(figsize=(16, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 1], wspace=0.35)

ax_a = fig.add_subplot(gs[0])
names = list(baselines.keys()) + ["Transformer", "LSTM"]
values = list(baselines.values()) + [transformer_test["top1"] * 100, lstm_test["top1"] * 100]
colors = ["#bdbdbd"] * len(baselines) + ["#ff7f0e", "#1f77b4"]

bars = ax_a.barh(range(len(names)), values, color=colors, edgecolor="white", height=0.7)
ax_a.set_yticks(range(len(names)))
ax_a.set_yticklabels(names, fontsize=9)
ax_a.set_xlabel("Top-1 Accuracy (%)", fontsize=11)
ax_a.set_xlim(0, 105)
ax_a.axvline(x=56.8, color="#999999", linestyle="--", linewidth=0.8, alpha=0.5)

# Add value labels
for bar, val in zip(bars, values):
    if val > 15:
        ax_a.text(val - 1.5, bar.get_y() + bar.get_height() / 2,
                  f"{val:.1f}%", ha="right", va="center", fontsize=9,
                  fontweight="bold", color="white")
    else:
        ax_a.text(val + 1, bar.get_y() + bar.get_height() / 2,
                  f"{val:.1f}%", ha="left", va="center", fontsize=9)

ax_a.set_title("(A)", fontsize=13, fontweight="bold", loc="left")
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)

# --- Panel B: Learning curves ---
ax_b = fig.add_subplot(gs[1])
epochs_lstm = [h["epoch"] for h in lstm_history]
val_top1_lstm = [h["val_top1"] * 100 for h in lstm_history]
val_top1_trans = [h["val_top1"] * 100 for h in transformer_history]
val_loss_lstm = [h["val_loss"] for h in lstm_history]
val_loss_trans = [h["val_loss"] for h in transformer_history]

ax_b.plot(epochs_lstm, val_top1_lstm, color="#1f77b4", linewidth=1.5, label="LSTM")
ax_b.plot(epochs_lstm, val_top1_trans, color="#ff7f0e", linewidth=1.5, label="Transformer")
ax_b.set_xlabel("Epoch", fontsize=11)
ax_b.set_ylabel("Validation Top-1 Accuracy (%)", fontsize=11)
ax_b.set_ylim(90, 100)
ax_b.legend(fontsize=9, loc="lower right")
ax_b.set_title("(B)", fontsize=13, fontweight="bold", loc="left")
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)

# Inset: validation loss
ax_inset = ax_b.inset_axes([0.45, 0.15, 0.5, 0.4])
ax_inset.plot(epochs_lstm, val_loss_lstm, color="#1f77b4", linewidth=1, linestyle="--")
ax_inset.plot(epochs_lstm, val_loss_trans, color="#ff7f0e", linewidth=1, linestyle="--")
ax_inset.set_xlabel("Epoch", fontsize=7)
ax_inset.set_ylabel("Val Loss", fontsize=7)
ax_inset.tick_params(labelsize=7)
ax_inset.spines["top"].set_visible(False)
ax_inset.spines["right"].set_visible(False)

# --- Panel C: Annotation gap (from existing image) ---
ax_c = fig.add_subplot(gs[2])
annotation_gap = mpimg.imread(ASSETS_DIR / "annotation_gap.png")
ax_c.imshow(annotation_gap)
ax_c.axis("off")
ax_c.set_title("(C)", fontsize=13, fontweight="bold", loc="left")

fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
print(f"Saved: {OUTPUT}")
print(f"Saved: {OUTPUT.with_suffix('.png')}")
plt.close()
