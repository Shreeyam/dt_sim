"""Lookahead-window figure for method.tex.

The main text now frames the decision rule in terms of a local break-even
window bounded by two scheduled anchors. This figure intentionally avoids the
renewal-process construction and shows the scheduling object directly.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  # registers the "science" style

REPO_ROOT = Path(__file__).resolve().parents[2]
MPL_STYLE = REPO_ROOT / "shreeyam.mplstyle"
OUT_DIR = Path(__file__).resolve().parents[1] / "figures"

plt.style.use(["science", "grid", str(MPL_STYLE)])

INDIGO = "#4a5de9"
MAGENTA = "#ea1a69"


def plot_chain_abstraction(out_path: Path) -> None:
    rng = np.random.default_rng(12)
    field_of_regard = 45
    t_min, t_max = 0, 140
    left_anchor = np.array([32.0, -22.0])
    right_anchor = np.array([108.0, 26.0])
    window_start, window_end = 38.0, 102.0

    inside_t = np.sort(rng.uniform(window_start + 3, window_end - 3, 18))
    inside_theta = rng.uniform(-field_of_regard + 7, field_of_regard - 7, inside_t.size)
    candidates = np.column_stack([inside_t, inside_theta])

    chosen_idx = [1, 5, 9, 14]
    chain = np.vstack([left_anchor, candidates[chosen_idx], right_anchor])
    schedule = np.array([
        [t_min, 34.0],
        [4.0, 31.0],
        [16.0, -34.0],
        [25.0, -8.0],
        left_anchor,
        [44.0, -18.0],
        [56.0, 11.0],
        [68.0, -7.0],
        [81.0, 18.0],
        [95.0, 2.0],
        right_anchor,
        [120.0, -16.0],
        [132.0, 21.0],
        [138.0, -28.0],
        [t_max, -31.0],
    ])

    outside_left = np.column_stack([
        rng.uniform(t_min + 4, window_start - 4, 10),
        rng.uniform(-55, 55, 10),
    ])
    outside_right = np.column_stack([
        rng.uniform(window_end + 4, t_max - 4, 10),
        rng.uniform(-55, 55, 10),
    ])
    outside_cross = np.column_stack([
        rng.uniform(window_start, window_end, 8),
        rng.choice([-1, 1], 8) * rng.uniform(field_of_regard + 4, 57, 8),
    ])
    outside = np.vstack([outside_left, outside_right, outside_cross])

    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    ax.grid(False)

    # Gray the non-actionable region in both along-track time and cross-track angle.
    ax.axvspan(t_min, window_start, color="0.88", zorder=0.3)
    ax.axvspan(window_end, t_max, color="0.88", zorder=0.3)
    ax.axhspan(field_of_regard, 60, color="0.88", zorder=0.3)
    ax.axhspan(-60, -field_of_regard, color="0.88", zorder=0.3)
    ax.axvspan(window_start, window_end, ymin=0.125, ymax=0.875,
               color="C0", alpha=0.08, zorder=0.35)

    outside_schedule = (schedule[:, 0] < window_start) | (schedule[:, 0] > window_end)
    inside_schedule = ~outside_schedule
    ax.plot(schedule[:, 0], schedule[:, 1], "-", color=MAGENTA, linewidth=1.1,
            alpha=0.85, zorder=1.3)
    ax.plot(schedule[outside_schedule, 0], schedule[outside_schedule, 1],
            "o", color=MAGENTA, markersize=3.8, alpha=0.75, zorder=1.35)
    ax.plot(schedule[inside_schedule, 0], schedule[inside_schedule, 1],
            "o", color=MAGENTA, markersize=4.0, alpha=0.9, zorder=1.4,
            label="Existing schedule")

    ax.plot(outside[:, 0], outside[:, 1], "o", color="0.68", markersize=3,
            zorder=1.0, label="Outside window")
    ax.plot(candidates[:, 0], candidates[:, 1],
            "o", markerfacecolor="white", markeredgecolor="0.05",
            markersize=4, zorder=2.0, label="Candidate accesses")

    ax.plot(chain[:, 0], chain[:, 1], "-", color=INDIGO, linewidth=1.4,
            zorder=2.5, label="Insertable chain")
    ax.plot([left_anchor[0], right_anchor[0]], [left_anchor[1], right_anchor[1]],
            "o", markerfacecolor=MAGENTA, markeredgecolor="0.05",
            markeredgewidth=0.8, markersize=7.5, zorder=3.0,
            label="Schedule anchors")

    ax.axvline(window_start, color="0.35", linestyle="--", linewidth=0.8)
    ax.axvline(window_end, color="0.35", linestyle="--", linewidth=0.8)
    ax.axhline(field_of_regard, color="0.35", linestyle=":", linewidth=0.8)
    ax.axhline(-field_of_regard, color="0.35", linestyle=":", linewidth=0.8)

    ax.annotate(
        "lookahead window",
        xy=((window_start + window_end) / 2, 55),
        ha="center",
        va="top",
        fontsize=8,
    )
    ax.text(left_anchor[0] - 3, left_anchor[1] - 7, r"$a_L$", ha="right", fontsize=9)
    ax.text(right_anchor[0] + 3, right_anchor[1] + 6, r"$a_R$", ha="left", fontsize=9)

    ax.set_xlim(t_min, t_max)
    ax.set_ylim(-60, 60)
    ax.set_xlabel("Along-track time [s]")
    ax.set_ylabel(r"Cross-track angle $\theta$ [deg]")
    ax.set_xticks([left_anchor[0], window_start, window_end, right_anchor[0]])
    ax.set_xticklabels([r"$a_L$", r"$t_1$", r"$t_2$", r"$a_R$"])
    ax.set_yticks([-field_of_regard, 0, field_of_regard])
    ax.set_yticklabels([r"$-\theta_{\mathrm{FoR}}$", "0", r"$+\theta_{\mathrm{FoR}}$"])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              fontsize=7, ncol=3, frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "chain_abstraction.pdf"
    plot_chain_abstraction(out)
    print(f"Wrote {out}")
