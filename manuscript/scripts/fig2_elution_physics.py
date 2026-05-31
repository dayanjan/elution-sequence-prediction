"""
Figure 2: Physicochemical basis of elution order in reversed-phase lipidomics.
Panel A: lipid_class_rt_structure.png (existing)
Panel B: headgroup_rt_residuals.png (existing)
Assembled side-by-side as a 2-panel composite.
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

# Paths
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
ASSETS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figures"
OUTPUT = FIGURES_DIR / "fig2_elution_physics.pdf"

# Load existing panels
panel_a = mpimg.imread(ASSETS_DIR / "lipid_class_rt_structure.png")
panel_b = mpimg.imread(ASSETS_DIR / "headgroup_rt_residuals.png")

# Create composite
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, img, label in zip(axes, [panel_a, panel_b], ["A", "B"]):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"({label})", fontsize=14, fontweight="bold", loc="left", pad=8)

plt.tight_layout(w_pad=2)
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
print(f"Saved: {OUTPUT}")
print(f"Saved: {OUTPUT.with_suffix('.png')}")
plt.close()
