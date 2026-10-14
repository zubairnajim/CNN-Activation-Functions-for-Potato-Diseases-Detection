"""
tools/make_method_figures.py
============================
Figures for the Methods section that do not depend on training results:

  fig_duplicates.png  examples of repeated photographs found by the audit (label conflicts first)
  fig_framework.png   the evaluation protocol (curation -> folds -> nested LR selection -> test)

Usage
-----
  python tools/make_method_figures.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from potato_rev import config as C  # noqa: E402

OUT = os.path.join(C.OUT_DIR, "analysis", "figures")


def fig_duplicates() -> None:
    with open(C.MANIFEST_PATH) as f:
        groups = json.load(f)["duplicate_groups"]
    groups = sorted(groups, key=lambda g: not g["label_conflict"])[:6]
    fig, axes = plt.subplots(2, len(groups), figsize=(2.0 * len(groups), 5.6))
    for j, g in enumerate(groups):
        for i, path in enumerate(g["members"][:2]):
            ax = axes[i, j]
            ax.imshow(Image.open(os.path.join(C.DATA_DIR, path)).convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            cls, name = path.split("/")
            ax.set_title(f"{cls}\n{name}", fontsize=7, color="#c0392b" if g["label_conflict"] else "black")
        axes[1, j].set_xlabel("label conflict: excluded" if g["label_conflict"] else "same label: one kept",
                              fontsize=7, color="#c0392b" if g["label_conflict"] else "#2c7a3f")
    fig.suptitle("Repeated photographs found by the audit (top and bottom: the two copies)", fontsize=9)
    fig.tight_layout(h_pad=2.5)
    fig.savefig(os.path.join(OUT, "fig_duplicates.png"), dpi=300)
    plt.close(fig)


def fig_framework() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=color, ec="#333333", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.6)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.0, color="#333333"))

    box(1, 30, 16, 11, "Public tuber image set\n451 images, 7 classes", "#e8eef7")
    box(21, 30, 18, 11, "Integrity audit\nMD5 + perceptual hash\n+ embedding review", "#e8eef7")
    box(43, 30, 18, 11, "Curated set: 431 images\n12 copies removed,\n8 label conflicts excluded", "#e8eef7")
    box(65, 30, 33, 11, "5 stratified folds (86-87 images)\nphotos of one specimen kept together\nrotation k: test = fold k, val = fold k+1,\ntrain = other 3 folds", "#e8eef7")
    arrow(17.5, 35.5, 20.5, 35.5)
    arrow(39.5, 35.5, 42.5, 35.5)
    arrow(61.5, 35.5, 64.5, 35.5)

    box(1, 13, 23, 12, "49 configurations\n8 ImageNet backbones\nx 6 activations\n(+ MobileNetV2 native ReLU6)\nall hidden sites replaced", "#fdf2e3")
    box(26.5, 13, 23, 12, "Fine-tuning per LR candidate\nAdamW, warm-up,\nlabel smoothing, plateau LR,\n<= 100 epochs, early\nstopping on validation loss", "#fdf2e3")
    box(52, 13, 22, 12, "Nested selection\nkeep the LR with the\nlowest validation loss\n(4 LRs per architecture)", "#fdf2e3")
    box(76.5, 13, 21.5, 12, "Single evaluation on\nthe test fold: accuracy,\nprecision, recall,\nspecificity, F1, error, AUC", "#fdf2e3")
    arrow(81, 29.5, 81, 25.8)
    arrow(24.4, 19, 26.1, 19)
    arrow(49.9, 19, 51.6, 19)
    arrow(74.4, 19, 76.1, 19)

    box(1, 1, 97, 8, "Across the 5 rotations: mean ± SD, pooled out-of-fold predictions (n = 431) with bootstrap CIs, "
                     "Friedman + Wilcoxon-Holm tests\nAblations: no augmentation, no pre-training  |  "
                     "100-epoch curves, inference cost, Grad-CAM", "#e9f5ec")
    arrow(87.5, 12.5, 87.5, 9.6)
    fig.savefig(os.path.join(OUT, "fig_framework.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_duplicates()
    fig_framework()
    print(f"[saved] {OUT}")
