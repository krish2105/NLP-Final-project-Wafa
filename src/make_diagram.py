"""Render a clean architecture/pipeline diagram PNG for the README and ADD.

Run:  python -m src.make_diagram   ->  outputs/figures/architecture.png
"""
from __future__ import annotations

import sys

try:
    from . import config
except ImportError:  # pragma: no cover
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src import config


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    NAVY, BLUE, LIGHT = "#0f4c81", "#1a6fb0", "#eef4fb"
    GREEN, AMBER, RED, PURPLE = "#2fa36b", "#e0a02c", "#e5484d", "#7c5cff"

    def box(x, y, w, h, title, sub, edge, fill=LIGHT):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    linewidth=2, edgecolor=edge, facecolor=fill))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=10.5,
                fontweight="bold", color=NAVY)
        ax.text(x + w / 2, y + h - 0.62, sub, ha="center", va="top", fontsize=7.6, color="#40566e")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                     linewidth=1.8, color=BLUE))

    # Band labels
    for bx, label, col in [(0.2, "LISTEN", BLUE), (5.0, "UNDERSTAND", GREEN), (9.4, "ACT", PURPLE)]:
        ax.text(bx, 4.95, label, fontsize=11, fontweight="bold", color=col)

    # Row of stages
    box(0.2, 3.0, 2.1, 1.5, "NLP Pipeline", "translate (ar/tl) +\nDistilBERT / TF-IDF\nissue + churn", BLUE)
    box(2.55, 3.0, 2.1, 1.5, "Entity Extract", "spaCy + regex +\nnative-script\nleaver / closure", BLUE)
    box(5.0, 3.0, 2.1, 1.5, "Churn Model", "LogReg / RF\n(nationality\nEXCLUDED)", GREEN)
    box(7.35, 3.0, 1.9, 1.5, "Fusion", "text + behaviour\n-> risk band\n+ reasons", GREEN)
    box(9.6, 3.0, 1.5, 1.5, "Decision", "transparent\nif/elif rules", PURPLE)
    box(11.25, 3.0, 1.55, 1.5, "Outreach", "Qwen2.5-0.5B\n+ guardrails", PURPLE)

    # Bottom row: inputs & governance
    box(0.2, 0.5, 2.1, 1.3, "Raw message\n+ Customer ID", "4 languages\nEN AR HI TL", "#7a8aa0", "#ffffff")
    box(5.0, 0.5, 2.1, 1.3, "Portfolio View", "segment-level\nrisk summary", GREEN, "#ffffff")
    box(9.6, 0.5, 3.2, 1.3, "Human Review Dashboard", "Approve / Edit / Reject / Override\n-> Audit Log (nothing auto-sends)", PURPLE, "#ffffff")

    # Arrows across the top row
    xs = [(2.3, 2.55), (4.65, 5.0), (7.1, 7.35), (9.25, 9.6), (11.1, 11.25)]
    for x1, x2 in xs:
        arrow(x1, 3.75, x2, 3.75)
    # input -> nlp
    arrow(1.25, 1.8, 1.25, 3.0)
    # churn -> portfolio
    arrow(6.05, 3.0, 6.05, 1.8)
    # outreach -> dashboard
    arrow(12.0, 3.0, 11.6, 1.8)

    ax.text(6.5, 0.08, "Project Wafa · Falcon Bank UAE · Listen → Understand → Act",
            ha="center", fontsize=9, style="italic", color="#40566e")

    fig.tight_layout()
    out = config.FIGURES_DIR / "architecture.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
