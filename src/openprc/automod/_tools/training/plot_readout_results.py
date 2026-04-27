"""
Plot results from a trained PRC readout run.

Reads the artifacts written by train_readout.py and produces:
  - <target>_timeseries.png  : true vs predicted, first 5 output dims
  - <target>_scatter.png     : per-output pred-vs-true scatter
  - <target>_residuals.png   : residual histogram + per-output bars
  - rollout_decay.png         : MSE vs horizon, all targets on one plot

Plus an optional --pdf flag that saves paper-ready PDF copies alongside
the PNGs.

Usage:
    python plot_readout_results.py --run-dir <bundle>/<robot>/training/<run_id>
    python plot_readout_results.py --run-dir ... --pdf
    python plot_readout_results.py --run-dir ... --max-output-dims 12
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List

import numpy as np


def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        ) from e


def _load_predictions(run_dir: str) -> Dict[str, dict]:
    """Load all <target>.npz files in the predictions/ subdir.

    Files starting with underscore are treated as sidecar metadata
    (e.g. _split_meta.npz) and skipped. Any .npz that doesn't carry both
    Y_true and Y_pred keys is also skipped with a warning rather than
    crashing the whole plot pass.
    """
    pred_dir = os.path.join(run_dir, "predictions")
    if not os.path.exists(pred_dir):
        raise FileNotFoundError(f"No predictions dir at {pred_dir}")
    out = {}
    for fn in sorted(os.listdir(pred_dir)):
        if not fn.endswith(".npz"):
            continue
        if fn.startswith("_"):
            continue   # sidecar metadata
        name = fn[:-4]
        path = os.path.join(pred_dir, fn)
        with np.load(path) as z:
            keys = set(z.files)
            if "Y_true" not in keys or "Y_pred" not in keys:
                print(f"  skip {fn}: missing Y_true/Y_pred (has {sorted(keys)})")
                continue
            out[name] = {"Y_true": z["Y_true"], "Y_pred": z["Y_pred"]}
    return out


def _load_summary(run_dir: str) -> dict:
    p = os.path.join(run_dir, "summary.json")
    with open(p) as f:
        return json.load(f)


def plot_timeseries(plt, target: str, Y_true: np.ndarray, Y_pred: np.ndarray,
                     out_path: str, max_dims: int = 5,
                     max_frames: int = 600, save_pdf: bool = False):
    """Plot true vs predicted time series for the first max_dims output channels."""
    n_dims = min(max_dims, Y_true.shape[1])
    n_frames = min(max_frames, Y_true.shape[0])
    fig, axes = plt.subplots(n_dims, 1, figsize=(11, 1.5 * n_dims),
                              sharex=True, squeeze=False)
    for i in range(n_dims):
        ax = axes[i, 0]
        ax.plot(Y_true[:n_frames, i], color="black", lw=1.0,
                label="true" if i == 0 else None)
        ax.plot(Y_pred[:n_frames, i], color="tab:red", lw=1.0, alpha=0.85,
                label="pred" if i == 0 else None)
        ax.set_ylabel(f"dim {i}", rotation=0, labelpad=20, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(loc="upper right", fontsize=9)
    axes[-1, 0].set_xlabel(f"frame (test, first {n_frames})")
    fig.suptitle(f"{target}: true vs predicted (first {n_dims} dims)",
                  fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140)
    if save_pdf:
        fig.savefig(out_path.replace(".png", ".pdf"))
    plt.close(fig)


def plot_scatter(plt, target: str, Y_true: np.ndarray, Y_pred: np.ndarray,
                  out_path: str, max_dims: int = 12, save_pdf: bool = False):
    """Per-output pred-vs-true scatter, arranged in a grid."""
    n_dims = min(max_dims, Y_true.shape[1])
    cols = min(4, n_dims)
    rows = int(np.ceil(n_dims / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows),
                              squeeze=False)

    for i in range(n_dims):
        ax = axes[i // cols, i % cols]
        ax.scatter(Y_true[:, i], Y_pred[:, i], s=2, alpha=0.4, c="tab:blue")
        # Identity line
        lo = float(min(Y_true[:, i].min(), Y_pred[:, i].min()))
        hi = float(max(Y_true[:, i].max(), Y_pred[:, i].max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        # R² annotation
        ss_res = float(((Y_pred[:, i] - Y_true[:, i]) ** 2).sum())
        ss_tot = float(((Y_true[:, i] - Y_true[:, i].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        ax.set_title(f"dim {i}  R²={r2:.3f}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    # Hide unused subplots
    for i in range(n_dims, rows * cols):
        axes[i // cols, i % cols].axis("off")
    fig.suptitle(f"{target}: per-output pred-vs-true scatter", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140)
    if save_pdf:
        fig.savefig(out_path.replace(".png", ".pdf"))
    plt.close(fig)


def plot_residuals(plt, target: str, Y_true: np.ndarray, Y_pred: np.ndarray,
                    out_path: str, save_pdf: bool = False):
    """Two panels: residual histogram (all dims), and per-output RMSE bars."""
    residuals = (Y_pred - Y_true).ravel()
    per_dim_rmse = np.sqrt(np.mean((Y_pred - Y_true) ** 2, axis=0))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

    # Histogram
    ax = axes[0]
    ax.hist(residuals, bins=80, color="tab:blue", alpha=0.7,
            edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("residual (pred − true)")
    ax.set_ylabel("count")
    ax.set_title(f"{target}: residual distribution", fontsize=10)
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=8)
    # Annotate stats
    ax.text(0.03, 0.95,
            f"mean = {residuals.mean():.4g}\nstd  = {residuals.std():.4g}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    # Per-output RMSE bars
    ax = axes[1]
    ax.bar(np.arange(len(per_dim_rmse)), per_dim_rmse, color="tab:red", alpha=0.75)
    ax.set_xlabel("output dim index")
    ax.set_ylabel("RMSE (test)")
    ax.set_title(f"{target}: per-dimension RMSE", fontsize=10)
    ax.grid(alpha=0.2, axis="y")
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    if save_pdf:
        fig.savefig(out_path.replace(".png", ".pdf"))
    plt.close(fig)


def plot_rollout_decay(plt, summary: dict, out_path: str, save_pdf: bool = False):
    """One curve per target: MSE vs rollout horizon."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for target, info in summary["targets"].items():
        rollout = info.get("rollout_mse", {})
        if not rollout:
            continue
        horizons = sorted(int(h) for h in rollout.keys())
        mses = [rollout[str(h)] for h in horizons]
        ax.plot(horizons, mses, marker="o", lw=1.5, label=target)

    ax.set_xlabel("rollout horizon (frames)")
    ax.set_ylabel("MSE on test split")
    ax.set_title("Rollout-horizon MSE decay", fontsize=11)
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    if save_pdf:
        fig.savefig(out_path.replace(".png", ".pdf"))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plot PRC readout training results")
    p.add_argument("--run-dir", required=True,
                   help="Path to a training run directory "
                        "(e.g. go1/training/20260425_120000)")
    p.add_argument("--max-output-dims", type=int, default=12,
                   help="Cap number of output dims shown in scatter plots")
    p.add_argument("--max-timeseries-dims", type=int, default=5)
    p.add_argument("--max-timeseries-frames", type=int, default=600)
    p.add_argument("--pdf", action="store_true",
                   help="Also save PDF versions alongside PNGs")
    args = p.parse_args()

    plt = _import_matplotlib()

    run_dir = os.path.abspath(args.run_dir)
    summary = _load_summary(run_dir)
    preds = _load_predictions(run_dir)

    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"=== Plotting run: {summary.get('run_id')} (robot: {summary.get('robot')}) ===")
    print(f"  targets: {list(preds.keys())}")
    print(f"  output dir: {plot_dir}")

    for target, arrs in preds.items():
        print(f"\n  plotting target: {target}")
        plot_timeseries(plt, target, arrs["Y_true"], arrs["Y_pred"],
                          os.path.join(plot_dir, f"{target}_timeseries.png"),
                          max_dims=args.max_timeseries_dims,
                          max_frames=args.max_timeseries_frames,
                          save_pdf=args.pdf)
        plot_scatter(plt, target, arrs["Y_true"], arrs["Y_pred"],
                       os.path.join(plot_dir, f"{target}_scatter.png"),
                       max_dims=args.max_output_dims, save_pdf=args.pdf)
        plot_residuals(plt, target, arrs["Y_true"], arrs["Y_pred"],
                         os.path.join(plot_dir, f"{target}_residuals.png"),
                         save_pdf=args.pdf)

    print(f"\n  plotting rollout decay")
    plot_rollout_decay(plt, summary,
                        os.path.join(plot_dir, "rollout_decay.png"),
                        save_pdf=args.pdf)

    print(f"\nDone. Plots in {plot_dir}/")


if __name__ == "__main__":
    main()
