from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import numpy as np  

from gestures import LABELS  
from synth import HAND_EDGES, generate_clip, sample_subject  

FINGER_COLORS = {0: "#ffb347", 1: "#ff4fd8", 2: "#3df5ff", 3: "#b6ff3b", 4: "#9b7bff"}


def edge_color(a: int, b: int) -> str:
    finger = 0 if max(a, b) <= 4 else (max(a, b) - 1) // 4
    return FINGER_COLORS[finger]


def draw_hand(ax, frames: np.ndarray, aspect: float, trail: bool) -> None:
    pts = frames.copy()
    pts[..., 0] = (1 - pts[..., 0]) * aspect  
    mid = pts[len(pts) // 2] if not trail else pts[-1]
    if trail:
        for lid, color in ((0, "#888"), (8, "#ff4fd8")):
            ax.plot(pts[:, lid, 0], pts[:, lid, 1], "-", color=color, lw=1, alpha=0.8)
    for a, b in HAND_EDGES:
        ax.plot([mid[a, 0], mid[b, 0]], [mid[a, 1], mid[b, 1]], "-", color=edge_color(a, b), lw=2)
    ax.scatter(mid[:, 0], mid[:, 1], s=6, c="white", zorder=3)
    cx, cy = mid[:, 0].mean(), mid[:, 1].mean()
    span = max(np.ptp(pts[..., 0]), np.ptp(pts[..., 1])) * 0.6 + 0.02
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy + span, cy - span)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#0b0b12")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/synthetic_gallery.png")
    parser.add_argument("--per-class", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    fig, axes = plt.subplots(len(LABELS), args.per_class, figsize=(2.0 * args.per_class, 2.0 * len(LABELS)))
    fig.patch.set_facecolor("#05050a")
    for row, label in enumerate(LABELS):
        for col in range(args.per_class):
            clip = generate_clip(label, rng, sample_subject(rng))
            draw_hand(axes[row, col], clip["frames"], clip["aspect"], trail=label in ("swipe_undo", "circle_clear"))
        axes[row, 0].set_ylabel(label, color="white", fontsize=10)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, facecolor=fig.get_facecolor())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
