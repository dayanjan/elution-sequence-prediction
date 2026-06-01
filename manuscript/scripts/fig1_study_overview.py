from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


BASE_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = BASE_DIR / "figures"
PDF_PATH = FIG_DIR / "fig1_study_overview.pdf"
PNG_PATH = FIG_DIR / "fig1_study_overview.png"

# Okabe-Ito colorblind-safe palette
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#7A7A7A"
DARK = "#1A1A1A"
LIGHT = "#F4F4F4"

# Typographic scale (pt at final size) — one system, no local arithmetic
FS_PANEL = 13.0     # A / B / C labels
FS_TITLE = 10.5     # panel titles
FS_HEAD = 9.0       # box headings
FS_BODY = 8.4       # body text
FS_SMALL = 7.6      # secondary text
FS_TINY = 6.9       # captions / cell labels
FS_MICRO = 6.4      # dense list items

ARR = "→"      # rightwards arrow (renders via fallback; arrows also drawn graphically)


def add_box(ax, xy, w, h, text="", fc="white", ec=DARK, color=DARK, fontsize=FS_SMALL,
            weight="normal", radius=0.025, lw=1.0, ha="center", va="center", pad=0.012):
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle=f"round,pad={pad},rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
    )
    ax.add_patch(box)
    if text:
        tx = xy[0] + w / 2 if ha == "center" else xy[0] + 0.02
        ax.text(tx, xy[1] + h / 2, text, ha=ha, va=va, fontsize=fontsize,
                color=color, fontweight=weight, linespacing=1.3)
    return box


def arrow(ax, start, end, color=DARK, lw=1.2, style="-|>", mutation_scale=11,
          ls="-", connectionstyle="arc3,rad=0"):
    patch = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=mutation_scale,
        color=color, linewidth=lw, linestyle=ls, connectionstyle=connectionstyle,
        shrinkA=0, shrinkB=0, joinstyle="miter", capstyle="round",
    )
    ax.add_patch(patch)
    return patch


def draw_check(ax, x, y, color=GREEN, fontsize=13.0):
    # mathtext glyph: aspect-immune and font-robust (avoids Arial tofu)
    ax.text(x, y, r"$\checkmark$", color=color, fontsize=fontsize,
            ha="center", va="center")


def draw_cross(ax, x, y, color=VERMILLION, fontsize=14.0):
    ax.text(x, y, r"$\times$", color=color, fontsize=fontsize,
            ha="center", va="center")


def draw_peak(ax, center, base_y, height, width, color=BLUE, alpha=0.26, lw=1.4):
    x = np.linspace(center - 2.8 * width, center + 2.8 * width, 160)
    y = base_y + height * np.exp(-0.5 * ((x - center) / width) ** 2)
    ax.fill_between(x, base_y, y, color=color, alpha=alpha, linewidth=0)
    ax.plot(x, y, color=color, linewidth=lw)


def panel_label(ax, label):
    ax.text(-0.012, 1.02, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=FS_PANEL, fontweight="bold", color=DARK)


def panel_title(ax, text):
    ax.text(0.05, 1.02, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=FS_TITLE, fontweight="bold", color=DARK)


def setup_panel(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ---------------------------------------------------------------------------
# Panel A : Reactive vs. predictive acquisition
# ---------------------------------------------------------------------------
def draw_panel_a(ax):
    setup_panel(ax)
    panel_label(ax, "A")
    panel_title(ax, "Reactive vs. predictive acquisition")

    left, right = 0.17, 0.96
    top_y, bot_y = 0.64, 0.21
    current_x = 0.56
    observed_xs = [0.27, 0.40, 0.51]
    future_xs = [0.70, 0.83]
    heights = [0.13, 0.16, 0.14, 0.15, 0.13]
    widths = [0.022, 0.026, 0.021, 0.025, 0.021]

    for y, label in [(top_y, "Reactive\n(DDA / DIA)"),
                     (bot_y, "Predictive\n(this work)")]:
        ax.plot([left, right], [y, y], color=GREY, lw=1.0)
        arrow(ax, (right - 0.015, y), (right, y), color=GREY, lw=1.0, mutation_scale=9)
        ax.text(left - 0.035, y + 0.10, label, ha="right", va="center",
                fontsize=FS_BODY, fontweight="bold", color=DARK, linespacing=1.2)
        ax.text(right, y - 0.045, "RT", ha="right", va="top", fontsize=FS_SMALL, color=GREY)

    # current-RT marker
    ax.plot([current_x, current_x], [bot_y - 0.075, top_y + 0.195],
            color=GREY, lw=1.1, ls=(0, (3, 3)))
    ax.text(current_x, top_y + 0.205, "current RT (now)", ha="center", va="bottom",
            fontsize=FS_SMALL, color=GREY)

    for x, h, w in zip(observed_xs, heights[:3], widths[:3]):
        draw_peak(ax, x, top_y, h, w, BLUE)
        draw_peak(ax, x, bot_y, h * 0.95, w, BLUE)

    for x, h, w in zip(future_xs, heights[3:], widths[3:]):
        for y, scale in [(top_y, 1.0), (bot_y, 0.95)]:
            px = np.linspace(x - 2.8 * w, x + 2.8 * w, 160)
            py = y + h * scale * np.exp(-0.5 * ((px - x) / w) ** 2)
            ax.fill_between(px, y, py, color=BLUE, alpha=0.05, linewidth=0)
            ax.plot(px, py, color=BLUE, linewidth=1.1, alpha=0.5, ls=(0, (3, 2)))

    # observed-vs-future is encoded by solid/dashed peaks + the current-RT divider;
    # one subtle cue under the observed peaks of the bottom (predictive) row.
    ax.text((left + observed_xs[-1]) / 2, bot_y - 0.05, "features seen so far",
            ha="center", va="top", fontsize=FS_TINY, color=GREY, style="italic")

    # reactive MS/MS on observed peaks
    for x in observed_xs:
        add_box(ax, (x - 0.044, top_y - 0.120), 0.088, 0.050, "MS/MS", fc="#FFF3DB",
                ec=ORANGE, color=ORANGE, fontsize=FS_SMALL, weight="bold", radius=0.01)
        arrow(ax, (x, top_y + 0.012), (x, top_y - 0.066),
              color=ORANGE, lw=1.0, mutation_scale=8)
    ax.text((left + current_x) / 2, top_y - 0.200,
            f"detect {ARR} select {ARR} fragment\n(only after the peak appears)",
            ha="center", va="center", fontsize=FS_SMALL, color=ORANGE, fontweight="bold",
            linespacing=1.2)

    # predictive: forecast arc + pre-configured MS/MS on future peak
    arrow(ax, (current_x + 0.005, bot_y + 0.12), (future_xs[0], bot_y + 0.165),
          color=GREEN, lw=1.6, ls="--", mutation_scale=13,
          connectionstyle="arc3,rad=-0.20")
    ax.text(0.62, bot_y + 0.205, "forecast", ha="center", va="bottom",
            fontsize=FS_BODY, color=GREEN, fontweight="bold")
    add_box(ax, (future_xs[0] - 0.066, bot_y - 0.122), 0.132, 0.060,
            "MS/MS\npre-configured", fc="#E4F4EE", ec=GREEN, color=GREEN,
            fontsize=FS_TINY, weight="bold", radius=0.01)
    arrow(ax, (future_xs[0], bot_y - 0.058), (future_xs[0], bot_y + 0.028),
          color=GREEN, lw=1.0, mutation_scale=8)
    ax.text((left + right) / 2, 0.045,
            f"forecast {ARR} pre-configure {ARR} fragment   (before the peak arrives)",
            ha="center", va="center", fontsize=FS_SMALL, color=GREEN, fontweight="bold")


# ---------------------------------------------------------------------------
# Panel B : Tokenization
# ---------------------------------------------------------------------------
def draw_demo_token(ax, x, y, w, h):
    fields = ["m/z bin", "mass defect", "RT gap", "polarity", "intensity rank"]
    add_box(ax, (x, y), w, h, "", fc="#FBF1F7", ec=PURPLE, radius=0.012, lw=1.2)
    field_w = w / len(fields)
    for i, field in enumerate(fields):
        if i:
            ax.plot([x + i * field_w, x + i * field_w], [y + 0.012, y + h - 0.012],
                    color=PURPLE, lw=0.7, alpha=0.7)
        ax.text(x + (i + 0.5) * field_w, y + h / 2, field, ha="center", va="center",
                fontsize=FS_TINY, color=DARK, rotation=90)


def draw_seq_token(ax, x, y, w, h, edge=BLUE, fill="#EAF3F9"):
    add_box(ax, (x, y), w, h, "", fc=fill, ec=edge, radius=0.012, lw=1.1)
    field_w = w / 5
    for i in range(5):
        if i:
            ax.plot([x + i * field_w, x + i * field_w], [y, y + h],
                    color=edge, lw=0.55, alpha=0.6)
        ax.add_patch(Rectangle(
            (x + i * field_w + 0.005, y + 0.030),
            field_w - 0.010, h - 0.060,
            facecolor=edge, edgecolor="none", alpha=0.12 + i * 0.04))


def draw_panel_b(ax):
    setup_panel(ax)
    panel_label(ax, "B")
    panel_title(ax, f"Tokenization: each feature {ARR} a 5-field token")

    # upper row: eluting feature -> demo token (lifted to sit just under the title)
    draw_peak(ax, 0.135, 0.58, 0.21, 0.024, PURPLE, alpha=0.30)
    ax.text(0.135, 0.525, "eluting feature", ha="center", va="top", fontsize=FS_SMALL,
            color=DARK)
    arrow(ax, (0.215, 0.69), (0.305, 0.69), color=PURPLE, lw=1.3, mutation_scale=11)
    draw_demo_token(ax, 0.315, 0.50, 0.30, 0.38)

    ax.text(0.665, 0.69,
            "Five fields, available for\nevery detected ion —\nno structural\nannotation needed",
            ha="left", va="center", fontsize=FS_BODY, color=DARK, linespacing=1.4)

    # lower row: token sequence ordered by retention time
    seq_y, seq_h, seq_w = 0.165, 0.155, 0.092
    xs = [0.085, 0.205, 0.325, 0.445, 0.565, 0.685]
    for i, x in enumerate(xs):
        draw_seq_token(ax, x, seq_y, seq_w, seq_h)
        ax.text(x + seq_w / 2, seq_y - 0.028, f"t$_{{{i + 1}}}$", ha="center", va="top",
                fontsize=FS_SMALL, color=GREY)
        if i < len(xs) - 1:
            arrow(ax, (x + seq_w + 0.006, seq_y + seq_h / 2),
                  (xs[i + 1] - 0.006, seq_y + seq_h / 2), color=GREY, lw=0.9,
                  mutation_scale=8)
    arrow(ax, (xs[-1] + seq_w + 0.010, seq_y + seq_h / 2),
          (0.862, seq_y + seq_h / 2), color=VERMILLION, lw=1.4, mutation_scale=12)
    ax.text(0.895, seq_y + seq_h + 0.052, "predict", ha="center", va="bottom",
            fontsize=FS_SMALL, color=VERMILLION, fontweight="bold")
    add_box(ax, (0.872, seq_y - 0.014), 0.108, seq_h + 0.028, "?\nnext m/z bin\n(110 classes)",
            fc="#F0F0F0", ec=VERMILLION, color=DARK, fontsize=FS_MICRO, weight="bold",
            radius=0.012, lw=1.1)
    ax.text(0.435, 0.045, "ordered left-to-right by retention time",
            ha="center", va="center", fontsize=FS_SMALL, color=GREY, style="italic")


# ---------------------------------------------------------------------------
# Panel C : Study design
# ---------------------------------------------------------------------------
def draw_panel_c(ax):
    setup_panel(ax)
    panel_label(ax, "C")
    panel_title(ax, "Study design")

    y, h = 0.25, 0.57
    x1, w1 = 0.04, 0.30
    x2, w2 = 0.39, 0.21
    x3, w3 = 0.65, 0.31
    mid = y + h / 2

    # --- Training data ---
    add_box(ax, (x1, y), w1, h, "", fc="#EAF3F9", ec=BLUE, radius=0.02, lw=1.2)
    ax.text(x1 + w1 / 2, y + h - 0.05, "Training data", ha="center", va="center",
            fontsize=FS_HEAD, color=BLUE, fontweight="bold")
    ax.text(x1 + w1 / 2, y + h - 0.115, "4 clinical cohorts · 342 plasma samples",
            ha="center", va="center", fontsize=FS_TINY, color=DARK)
    train_lines = [
        "REDHART 1  (n = 102)",
        "Cardiac arrest  (n = 97)",
        "GVHD  (n = 68)",
        "PCOS  (n = 75)",
        "",
        "SCIEX TripleTOF 6600+",
        "Waters CSH C18, dual-polarity",
        "15,242 consensus features",
        "13% annotated",
    ]
    lx = x1 + 0.035
    for i, line in enumerate(train_lines):
        ax.text(lx, y + h - 0.170 - i * 0.0445, line, ha="left", va="center",
                fontsize=FS_MICRO, color=DARK)

    # --- Models ---
    add_box(ax, (x2, y), w2, h, "", fc="#FFF3DB", ec=ORANGE, radius=0.02, lw=1.2)
    ax.text(x2 + w2 / 2, y + h - 0.05, "Models", ha="center", va="center",
            fontsize=FS_HEAD, color=ORANGE, fontweight="bold")
    ax.text(x2 + w2 / 2, y + h - 0.175, "LSTM\n98.4% top-1", ha="center", va="center",
            fontsize=FS_BODY, color=DARK, linespacing=1.4)
    ax.text(x2 + w2 / 2, y + h - 0.330, "Transformer\n98.0% top-1", ha="center",
            va="center", fontsize=FS_BODY, color=DARK, linespacing=1.4)
    ax.text(x2 + w2 / 2, y + 0.070, "next m/z bin\nprediction", ha="center", va="center",
            fontsize=FS_TINY, color=GREY, linespacing=1.3, style="italic")

    # --- Validation ---
    add_box(ax, (x3, y), w3, h, "", fc="#EDF8F3", ec=GREEN, radius=0.02, lw=1.2)
    ax.text(x3 + w3 / 2, y + h - 0.05, "Three validation experiments", ha="center",
            va="center", fontsize=FS_HEAD, color=GREEN, fontweight="bold")
    validation = [
        ("check", ["Same method, different MS", "ST000983, Agilent 6530",
                   "transfers (RT r = 0.999)"]),
        ("cross", ["Different column chemistry", "ST003514, Agilent 6545",
                   "fails (5.1% top-1)"]),
        ("cross", ["Same instrument, opposite polarity", "ST000990, SCIEX 6600",
                   "fails (2.6% top-1)"]),
    ]
    block_top = y + h - 0.145
    block_h = 0.135
    mark_x = x3 + 0.035
    text_x = x3 + 0.065
    for i, (mark, lines) in enumerate(validation):
        top = block_top - i * block_h
        if mark == "check":
            draw_check(ax, mark_x, top - 0.012)
        else:
            draw_cross(ax, mark_x, top - 0.012)
        ax.text(text_x, top, lines[0], ha="left", va="top",
                fontsize=FS_MICRO, color=DARK, fontweight="bold")
        for j, line in enumerate(lines[1:], start=1):
            ax.text(text_x, top - j * 0.036, line, ha="left", va="top",
                    fontsize=FS_MICRO - 0.5, color=GREY)

    # flow arrows (with edge clearance)
    arrow(ax, (x1 + w1 + 0.014, mid), (x2 - 0.014, mid), color=GREY, lw=1.3,
          mutation_scale=12)
    arrow(ax, (x2 + w2 + 0.014, mid), (x3 - 0.014, mid), color=GREY, lw=1.3,
          mutation_scale=12)

    # take-home banner
    add_box(ax, (0.05, 0.075), 0.90, 0.105,
            "Take-home: models are chromatographic-method- and polarity-mode-specific",
            fc="#EFEFEF", ec=DARK, color=DARK, fontsize=FS_SMALL, weight="bold",
            radius=0.02, lw=1.0)


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    })

    fig = plt.figure(figsize=(7.0, 8.5), constrained_layout=False)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.70, 1.05, 1.25],
                          hspace=0.16, left=0.02, right=0.98, top=0.965, bottom=0.02)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]

    draw_panel_a(axes[0])
    draw_panel_b(axes[1])
    draw_panel_c(axes[2])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"matplotlib {matplotlib.__version__}")
    print(f"{PNG_PATH.name}: {PNG_PATH.stat().st_size} bytes")


if __name__ == "__main__":
    main()
