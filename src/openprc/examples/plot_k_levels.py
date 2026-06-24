"""Plot evolution of K_level values across epochs from a loss_log.csv file."""

import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DEFAULT_CSV = (
    "OpenPRC-dev/src/openprc/experiments/Taichi_run+heatmap/M4_linear_memory_1/loss_log.csv"
)


def plot_k_levels(csv_path: Path, save_png: bool = True, show: bool = True):
    df = pd.read_csv(csv_path)

    k_cols = [c for c in df.columns if c.startswith("K_level_")]
    if not k_cols:
        raise ValueError(f"No K_level_* columns found in {csv_path}")

    epochs = df["Epoch"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for col in k_cols:
        label = col.replace("K_level_", "Level ")
        ax.plot(epochs, df[col], marker="o", markersize=3, linewidth=1.5, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Stiffness k (N/m)")
    ax.set_title(f"K-level evolution — {csv_path.parent.name}")
    ax.legend(title="Material level")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_png:
        out_path = csv_path.parent / "k_levels_evolution.png"
        fig.savefig(out_path, dpi=150)
        print(f"Saved → {out_path}")

    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot K-level evolution from loss_log.csv")
    parser.add_argument("csv", nargs="?", default=str(DEFAULT_CSV),
                        help="Path to loss_log.csv (default: M4_run0)")
    parser.add_argument("--no-save", action="store_true", help="Skip saving PNG")
    parser.add_argument("--no-show", action="store_true", help="Skip interactive display")
    args = parser.parse_args()

    plot_k_levels(
        Path(args.csv),
        save_png=not args.no_save,
        show=not args.no_show,
    )
