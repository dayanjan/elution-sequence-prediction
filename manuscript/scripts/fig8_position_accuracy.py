"""
Figure 8: Position-dependent prediction accuracy.
Top-1 accuracy vs sequence position for LSTM (from per-position CSV).
Smoothed with moving average, with IQR shading across test samples.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
GDRIVE = Path("H:/My Drive/0000 Fun with coding/088 Lights-Out R01 Grant/Specific Aim 1/poc3_elution_sequence")
POS_CSV = GDRIVE / "06_acquisition_simulation" / "outputs" / "lstm_position_accuracy.csv"
OUTPUT = FIGURES_DIR / "fig8_position_accuracy.pdf"

# Load position accuracy data (large file — sample_idx, pos, rt_seconds, rt_fraction, pred_mz, actual_mz, correct)
print("Loading position accuracy data...")
df = pd.read_csv(POS_CSV)
print(f"Loaded {len(df):,} rows, {df['sample_idx'].nunique()} samples")

# Compute per-position accuracy across all samples
# Group by position, compute mean accuracy
pos_acc = df.groupby("pos")["correct"].agg(["mean", "count"]).reset_index()
pos_acc.columns = ["pos", "accuracy", "n"]

# Also compute per-sample per-position accuracy for IQR
# Use coarser position bins to get meaningful per-sample statistics
BIN_SIZE = 50
df["pos_bin"] = (df["pos"] // BIN_SIZE) * BIN_SIZE
sample_pos_acc = df.groupby(["sample_idx", "pos_bin"])["correct"].mean().reset_index()
sample_pos_acc.columns = ["sample_idx", "pos_bin", "accuracy"]

# Compute quartiles across samples for each position bin
iqr = sample_pos_acc.groupby("pos_bin")["accuracy"].agg(
    median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75),
    mean="mean"
).reset_index()

# Smooth the overall accuracy with moving average
WINDOW = 50
pos_acc_sorted = pos_acc.sort_values("pos")
pos_acc_sorted["accuracy_smooth"] = pos_acc_sorted["accuracy"].rolling(
    window=WINDOW, center=True, min_periods=10
).mean()

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 4.5))

# IQR shading
ax.fill_between(iqr["pos_bin"], iqr["q25"] * 100, iqr["q75"] * 100,
                alpha=0.2, color="#1f77b4", label="IQR across samples")

# Smoothed accuracy line
ax.plot(pos_acc_sorted["pos"], pos_acc_sorted["accuracy_smooth"] * 100,
        color="#1f77b4", linewidth=1.5, label="LSTM (50-token moving avg)")

# Reference line at overall mean
overall_acc = df["correct"].mean() * 100
ax.axhline(overall_acc, color="#999999", linestyle="--", linewidth=0.8, alpha=0.7)
ax.text(pos_acc_sorted["pos"].max() * 0.98, overall_acc + 0.5,
        f"Overall: {overall_acc:.1f}%", ha="right", fontsize=9, color="#666666")

# RT axis on top (approximate mapping from position to RT)
# Use the rt_fraction column to map positions to approximate RT in minutes
pos_rt = df.groupby("pos")["rt_seconds"].median().reset_index()
ax2 = ax.twiny()
# Map a few reference points
rt_ticks_min = [1, 2, 4, 6, 8, 10, 12]
for rt_min in rt_ticks_min:
    rt_sec = rt_min * 60
    closest = pos_rt.iloc[(pos_rt["rt_seconds"] - rt_sec).abs().argsort().iloc[0]]

rt_positions = []
rt_labels = []
for rt_min in rt_ticks_min:
    rt_sec = rt_min * 60
    mask = pos_rt["rt_seconds"] <= rt_sec
    if mask.any():
        closest_pos = pos_rt.loc[mask, "pos"].max()
        rt_positions.append(closest_pos)
        rt_labels.append(f"{rt_min}")

ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(rt_positions)
ax2.set_xticklabels(rt_labels, fontsize=9)
ax2.set_xlabel("Approximate retention time (min)", fontsize=10)

ax.set_xlabel("Sequence position (token index)", fontsize=11)
ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
ax.set_ylim(85, 100)
ax.legend(fontsize=9, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
print(f"Saved: {OUTPUT}")
print(f"Saved: {OUTPUT.with_suffix('.png')}")
plt.close()
