from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


BASE_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = BASE_DIR / "figures"
PDF_PATH = FIG_DIR / "fig1_study_overview.pdf"
PNG_PATH = FIG_DIR / "fig1_study_overview.png"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#999999"
DARK = "#222222"
LIGHT = "#F7F7F7"


def add_box(ax, xy, w, h, text, fc="white", ec=DARK, color=DARK, fontsize=8.5,
            weight="normal", radius=0.025, lw=1.0, ha="center"):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha=ha,
        va="center",
        fontsize=fontsize,
        color=color,
        fontweight=weight,
        linespacing=1.2,
    )
    return box


def arrow(ax, start, end, color=DARK, lw=1.2, style="-|>", mutation_scale=10,
          ls="-", connectionstyle="arc3,rad=0"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        color=color,
        linewidth=lw,
        linestyle=ls,
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(patch)
    return patch


def draw_peak(ax, center, base_y, height, width, color=BLUE, alpha=0.26):
    x = np.linspace(center - 2.8 * width, center + 2.8 * width, 140)
    y = base_y + height * np.exp(-0.5 * ((x - center) / width) ** 2)
    ax.fill_between(x, base_y, y, color=color, alpha=alpha, linewidth=0)
    ax.plot(x, y, color=color, linewidth=1.4)
    ax.plot([center, center], [base_y, base_y + height], color=color, alpha=0.45, lw=0.8)


def panel_label(ax, label):
    ax.text(0.01, 0.98, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=14, fontweight="bold", color=DARK)


def setup_panel(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_panel_a(ax):
    setup_panel(ax)
    panel_label(ax, "A")
    ax.text(0.09, 0.965, "Reactive vs. Predictive acquisition",
            fontsize=11.5, fontweight="bold", ha="left", va="top")

    left, right = 0.13, 0.95
    top_y, bot_y = 0.64, 0.20
    current_x = 0.56
    observed_xs = [0.25, 0.38, 0.50]
    future_xs = [0.69, 0.82]
    peak_xs = observed_xs + future_xs
    heights = [0.14, 0.18, 0.15, 0.17, 0.14]
    widths = [0.022, 0.026, 0.021, 0.025, 0.021]

    for y, label in [(top_y, "Reactive acquisition (DDA / DIA)"),
                     (bot_y, "Predictive acquisition (this work)")]:
        ax.plot([left, right], [y, y], color="#555555", lw=1.0)
        arrow(ax, (right - 0.02, y), (right, y), color="#555555", lw=1.0)
        ax.text(left - 0.02, y + 0.19, label, ha="right", va="center",
                fontsize=9.4, fontweight="bold")
        ax.text(right, y - 0.055, "RT", ha="right", va="top", fontsize=8.5, color="#555555")

    ax.plot([current_x, current_x], [bot_y - 0.15, top_y + 0.22],
            color=GREY, lw=1.1, ls=(0, (3, 3)))
    ax.text(current_x, top_y + 0.225, "current RT (now)", ha="center", va="bottom",
            fontsize=8.3, color="#666666")

    for x, h, w in zip(observed_xs, heights[:3], widths[:3]):
        draw_peak(ax, x, top_y, h, w, BLUE)
        draw_peak(ax, x, bot_y, h * 0.95, w, BLUE)

    for x, h, w in zip(future_xs, heights[3:], widths[3:]):
        for y, scale in [(top_y, 1.0), (bot_y, 0.95)]:
            px = np.linspace(x - 2.8 * w, x + 2.8 * w, 140)
            py = y + h * scale * np.exp(-0.5 * ((px - x) / w) ** 2)
            ax.fill_between(px, y, py, color=BLUE, alpha=0.045, linewidth=0)
            ax.plot(px, py, color=BLUE, linewidth=1.2, alpha=0.45, ls=(0, (3, 2)))

    ax.text(left - 0.02, top_y + 0.05, "features eluting now",
            ha="right", va="center", fontsize=8.2, color="#555555")
    ax.text((left + current_x) / 2, bot_y - 0.055, "features seen so far",
            ha="center", va="top", fontsize=8.2, color="#555555")

    for x in observed_xs:
        add_box(ax, (x - 0.043, top_y - 0.122), 0.086, 0.054, "MS/MS", fc="#FFF7E0",
                ec=ORANGE, color=ORANGE, fontsize=8.5, weight="bold", radius=0.012)
        arrow(ax, (x, top_y + 0.018), (x, top_y - 0.068),
              color=ORANGE, lw=1.0, mutation_scale=8)
    ax.text((left + current_x) / 2, top_y - 0.205,
            "detect -> select -> fragment\n(only after the peak appears)",
            ha="center", va="center", fontsize=7.9, color=ORANGE, fontweight="bold",
            linespacing=1.1)

    add_box(ax, (0.585, bot_y - 0.118), 0.135, 0.066, "MS/MS\npre-configured",
            fc="#EAF6F1", ec=GREEN, color=GREEN, fontsize=7.6, weight="bold", radius=0.012)
    arrow(ax, (current_x + 0.01, bot_y + 0.12), (future_xs[0] - 0.01, bot_y + 0.19),
          color=GREEN, lw=1.5, ls="--", mutation_scale=12,
          connectionstyle="arc3,rad=-0.15")
    ax.text(0.62, bot_y + 0.20, "forecast", ha="center", va="bottom",
            fontsize=8.4, color=GREEN, fontweight="bold")
    arrow(ax, (0.685, bot_y - 0.052), (future_xs[0], bot_y + 0.035),
          color=GREEN, lw=1.0, mutation_scale=8)
    ax.text((left + right) / 2, bot_y - 0.175,
            "forecast -> pre-configure -> fragment  (before the peak arrives)",
            ha="center", va="center", fontsize=8.0, color=GREEN, fontweight="bold")


def draw_token(ax, x, y, w, h, fill="#FFFFFF", edge=BLUE, qmark=False, labels=True):
    if qmark:
        add_box(ax, (x, y), w, h, "?\nnext m/z bin\n110 classes", fc="#F3F3F3",
                ec=GREY, color="#444444", fontsize=7.6, weight="bold", radius=0.012)
        return
    fields = ["m/z bin", "mass defect", "RT gap", "polarity", "intensity rank"]
    add_box(ax, (x, y), w, h, "", fc=fill, ec=edge, radius=0.012, lw=1.1)
    field_w = w / len(fields)
    for i, field in enumerate(fields):
        if i:
            ax.plot([x + i * field_w, x + i * field_w], [y, y + h],
                    color=edge, lw=0.65, alpha=0.75)
        if labels:
            ax.text(x + (i + 0.5) * field_w, y + h / 2, field,
                    ha="center", va="center", fontsize=7.0, color=DARK, rotation=90)
        else:
            ax.add_patch(Rectangle(
                (x + i * field_w + 0.006, y + 0.035),
                field_w - 0.012,
                h - 0.07,
                facecolor=edge,
                edgecolor="none",
                alpha=0.12 + i * 0.03,
            ))


def draw_panel_b(ax):
    setup_panel(ax)
    panel_label(ax, "B")
    ax.text(0.09, 0.96, "Tokenization: each feature -> a 5-field token (no structure required)",
            fontsize=11.2, fontweight="bold", ha="left", va="top")

    draw_peak(ax, 0.17, 0.63, 0.17, 0.027, PURPLE, alpha=0.28)
    ax.text(0.17, 0.56, "eluting feature", ha="center", va="top", fontsize=8.5)
    arrow(ax, (0.25, 0.68), (0.34, 0.68), color=PURPLE, lw=1.2)
    draw_token(ax, 0.35, 0.58, 0.31, 0.18, fill="#FBF4F8", edge=PURPLE)

    ax.text(0.72, 0.67,
            "5 fields, all available for every detected ion\nno structural annotation needed",
            ha="left", va="center", fontsize=8.8, color="#444444")

    seq_y, seq_h, seq_w = 0.19, 0.16, 0.092
    xs = [0.10, 0.22, 0.34, 0.46, 0.58, 0.70]
    for i, x in enumerate(xs):
        draw_token(ax, x, seq_y, seq_w, seq_h, fill="#EFF6FA", edge=BLUE, labels=False)
        ax.text(x + seq_w / 2, seq_y - 0.04, f"t{i + 1}", ha="center", va="top",
                fontsize=7.8, color="#555555")
        if i < len(xs) - 1:
            arrow(ax, (x + seq_w + 0.01, seq_y + seq_h / 2),
                  (xs[i + 1] - 0.01, seq_y + seq_h / 2), color=GREY, lw=0.9,
                  mutation_scale=8)
    arrow(ax, (0.80, seq_y + seq_h / 2), (0.86, seq_y + seq_h / 2),
          color=VERMILLION, lw=1.2, mutation_scale=10)
    ax.text(0.83, seq_y + seq_h / 2 + 0.07, "predict next ->",
            ha="center", va="bottom", fontsize=8.5, color=VERMILLION, fontweight="bold")
    draw_token(ax, 0.87, seq_y, 0.10, seq_h, qmark=True)
    ax.text(0.40, 0.075, "Ordered left-to-right by retention time",
            ha="center", va="center", fontsize=8.4, color="#555555")


def draw_panel_c(ax):
    setup_panel(ax)
    panel_label(ax, "C")
    ax.text(0.09, 0.96, "Study design",
            fontsize=11.5, fontweight="bold", ha="left", va="top")

    y, h = 0.23, 0.62
    x1, w1 = 0.07, 0.30
    x2, w2 = 0.43, 0.20
    x3, w3 = 0.69, 0.28

    add_box(ax, (x1, y), w1, h, "", fc="#EFF6FA", ec=BLUE,
            color=DARK, fontsize=7.5, weight="normal", radius=0.018)
    ax.text(x1 + w1 / 2, y + h - 0.055, "Training data",
            ha="center", va="center", fontsize=9.1, color=BLUE, fontweight="bold")
    ax.text(x1 + w1 / 2, y + h - 0.125,
            "4 clinical cohorts, 342 plasma samples",
            ha="center", va="center", fontsize=7.3, color=DARK)
    train_lines = [
        "REDHART 2 (n=102)",
        "Cardiac Arrest (n=97)",
        "GVHD (n=68)",
        "PCOS (n=75)",
        "SCIEX TripleTOF 6600+",
        "Waters CSH C18, dual-polarity",
        "15,242 consensus features",
        "13% annotated",
    ]
    for i, line in enumerate(train_lines):
        ax.text(x1 + w1 / 2, y + h - 0.190 - i * 0.050, line,
                ha="center", va="center", fontsize=7.2, color=DARK)

    add_box(ax, (x2, y), w2, h, "", fc="#FFF7E0", ec=ORANGE,
            color=DARK, fontsize=8.0, radius=0.018)
    ax.text(x2 + w2 / 2, y + h - 0.055, "Models",
            ha="center", va="center", fontsize=9.1, color=ORANGE, fontweight="bold")
    ax.text(x2 + w2 / 2, y + h - 0.175, "LSTM\n98.4% top-1",
            ha="center", va="center", fontsize=8.2, color=DARK, linespacing=1.35)
    ax.text(x2 + w2 / 2, y + h - 0.340, "Transformer\n98.1% top-1",
            ha="center", va="center", fontsize=8.2, color=DARK, linespacing=1.35)
    ax.text(x2 + w2 / 2, y + 0.065, "next-m/z-bin\nprediction",
            ha="center", va="center", fontsize=7.8, color=DARK, linespacing=1.25)

    add_box(ax, (x3, y), w3, h, "", fc="#F6FBF8", ec=GREEN,
            color=DARK, fontsize=7.6, radius=0.018)
    ax.text(x3 + w3 / 2, y + h - 0.055, "3 validation experiments",
            ha="center", va="center", fontsize=9.1, color=GREEN, fontweight="bold")

    validation = [
        (["Same method, different MS",
          "ST000983, Agilent 6530",
          "transfers - RT r = 0.999"], "✓", GREEN),
        (["Different column chemistry",
          "ST003514, Agilent 6545",
          "fails - 5.1% top-1"], "✗", VERMILLION),
        (["Same instrument, positive-only",
          "polarity",
          "ST000990, SCIEX 6600",
          "fails - 2.8% top-1"], "✗", VERMILLION),
    ]
    for i, (lines, mark, color) in enumerate(validation):
        yy = y + h - [0.150, 0.325, 0.445][i]
        for j, line in enumerate(lines):
            ax.text(x3 + 0.025, yy - j * 0.044, line, ha="left", va="top",
                    fontsize=5.9, color=DARK)
        ax.text(x3 + w3 - 0.03, yy - 0.050, mark, ha="center", va="center",
                fontsize=14, color=color, fontweight="bold")

    arrow(ax, (x1 + w1 + 0.025, y + h / 2), (x2 - 0.025, y + h / 2),
          color="#555555", lw=1.2, mutation_scale=12)
    arrow(ax, (x2 + w2 + 0.025, y + h / 2), (x3 - 0.025, y + h / 2),
          color="#555555", lw=1.2, mutation_scale=12)

    add_box(ax, (0.09, 0.045), 0.82, 0.09,
            "Take-home: models are chromatographic-method- and polarity-mode-specific",
            fc=LIGHT, ec=GREY, color=DARK, fontsize=7.5, weight="bold",
            radius=0.018, lw=0.9)


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    })

    fig = plt.figure(figsize=(7.0, 8.5), constrained_layout=False)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.75, 1.15, 1.25], hspace=0.13)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]

    draw_panel_a(axes[0])
    draw_panel_b(axes[1])
    draw_panel_c(axes[2])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"matplotlib {matplotlib.__version__}")
    print(f"{PDF_PATH}: {PDF_PATH.stat().st_size} bytes")
    print(f"{PNG_PATH}: {PNG_PATH.stat().st_size} bytes")


if __name__ == "__main__":
    main()
