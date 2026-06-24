"""Plot the final spring network from an optimized geometry file.

Reads an ``experiment.json`` produced by the global-stiffness optimization
(e.g. run_taichi_global_stiffness_reduced_v3.py) and draws the surviving spring
network on the 4x4 main-node grid, annotating each spring with the final
stiffness value it converged to.

The geometry file stores ``k_mat_opt``: a symmetric (ROWS*COLS, ROWS*COLS)
matrix where entry [i, j] is the optimized stiffness (N/m) of the spring between
main nodes i and j. Main node ``idx = r * COLS + c`` sits at grid position
``[c * SPACING, -r * SPACING]`` (matching generate_subspring_config).
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Grid geometry (mirrors run_taichi_global_stiffness_reduced_v3.py)
SPACING = 0.053  # meters between adjacent main nodes

# --- Path to the final geometry file ---
EXPERIMENT_JSON = (
    Path(__file__).parent.parent
    / "experiments" / "Taichi_run+heatmap" / "M4_linear_memory_1" / "experiment.json"
)


def main():
    with open(EXPERIMENT_JSON, "r") as f:
        data = json.load(f)

    k_mat = np.array(data["k_mat_opt"])
    n_nodes = k_mat.shape[0]

    # Infer the (square) grid dimensions from the node count.
    side = int(round(np.sqrt(n_nodes)))
    assert side * side == n_nodes, f"{n_nodes} nodes is not a square grid"
    rows = cols = side

    # Reconstruct main-node positions: idx = r * cols + c -> [c*SPACING, -r*SPACING]
    positions = np.zeros((n_nodes, 2))
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            positions[idx] = [c * SPACING, -r * SPACING]

    # Collect surviving springs (upper triangle, non-zero stiffness).
    springs = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            k = k_mat[i, j]
            if k > 0.0:
                springs.append((i, j, k))

    k_values = np.array([k for _, _, k in springs])
    print(f"Loaded {EXPERIMENT_JSON}")
    print(f"Grid: {rows}x{cols} ({n_nodes} nodes) | Active springs: {len(springs)}")
    print(f"Stiffness range: {k_values.min():.2f} - {k_values.max():.2f} N/m")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=k_values.min(), vmax=k_values.max())

    # Edge width scales with stiffness for quick visual ranking.
    w_min, w_max = 1.5, 7.0
    k_lo, k_hi = k_values.min(), k_values.max()
    k_span = max(k_hi - k_lo, 1e-9)

    for i, j, k in springs:
        xa, ya = positions[i]
        xb, yb = positions[j]
        color = cmap(norm(k))
        width = w_min + (w_max - w_min) * (k - k_lo) / k_span
        ax.plot([xa, xb], [ya, yb], color=color, linewidth=width, zorder=1)

        # Label each spring with its final stiffness at the edge midpoint.
        xm, ym = (xa + xb) / 2.0, (ya + yb) / 2.0
        ax.text(
            xm, ym, f"{k:.1f}", fontsize=8, ha="center", va="center",
            color="black", zorder=3,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
        )

    # Main nodes, labeled with their index.
    ax.scatter(positions[:, 0], positions[:, 1], c="firebrick", s=120,
               zorder=2, edgecolors="black")
    for idx in range(n_nodes):
        ax.text(positions[idx, 0], positions[idx, 1], str(idx),
                fontsize=8, ha="center", va="center", color="white", zorder=4)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Final spring stiffness (N/m)")

    ax.set_title(f"Final Spring Network — {EXPERIMENT_JSON.parent.name}")
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()

    out_path = EXPERIMENT_JSON.parent / "final_spring_network.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
