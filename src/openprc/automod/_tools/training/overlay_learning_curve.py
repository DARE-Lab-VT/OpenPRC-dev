"""
Overlay learning-curve comparison: data-efficient vs. sequential training.

Reads the pre-computed CSV outputs from:
  - sequential_clip_training  -> sequential_learning_curve.csv
  - data_efficient_training   -> cumulative_learning_curve.csv

For each (feature, target) pair, produces one figure with both curves
plotted against the number of included trajectories (k), so the reader can
directly compare how quickly best-first vs. natural-order accumulates R2.

The x-axis is trajectory count k, not trajectory identity, because the two
strategies use different orderings -- the comparison is about *data efficiency
as a function of dataset size*, not about specific trajectories.

Run from the repo's src directory:

    python3 -m openprc.automod._tools.training.overlay_learning_curves \\
      --bundle-dir openprc/automod/robot_bundle

Or point directly at the two CSV files if they live elsewhere:

    python3 -m openprc.automod._tools.training.overlay_learning_curves \\
      --sequential-csv  path/to/sequential_learning_curve.csv \\
      --de-csv          path/to/cumulative_learning_curve.csv \\
      --output-dir      path/to/output
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Color palette -- consistent with the source scripts.
#
# Two-curve overlay:
#   VT Maroon  -> data-efficient (best-first)  -- primary / best-case result
#   VT Stone   -> sequential in-clip           -- natural-order baseline
#
# Distinct markers (circle / diamond) keep the figure readable in greyscale
# and for colorblind viewers.
# ---------------------------------------------------------------------------
SERIES = [
    {
        "key":    "de",
        "label":  "Data-efficient (best-first)",
        "color":  "#861F41",   # VT Maroon
        "marker": "o",
        "ms":     5,
        "lw":     1.6,
        "zorder": 4,
    },
    {
        "key":    "seq",
        "label":  "Sequential (natural order)",
        "color":  "#75787B",   # VT Stone
        "marker": "D",
        "ms":     4.5,
        "lw":     1.3,
        "zorder": 3,
    },
]

# Shared neutrals
INK       = "#2E3440"
GRID      = "#D8DEE9"
ZERO      = "#888888"
MARKER_EC = "#FFFFFF"


# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def _csv_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _safe_token(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in s)


def _load_curve_csv(path: Path) -> List[Dict[str, str]]:
    """Load a learning-curve CSV; return list of raw string-value dicts."""
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _extract_pairs(rows: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    """Sorted unique (feature, target) pairs present in a row set."""
    pairs = {(r["feature"], r["target"]) for r in rows}
    return sorted(pairs)


def _get_curve(
    rows: List[Dict[str, str]],
    feature: str,
    target: str,
) -> List[Tuple[int, float]]:
    """
    Return sorted (k, r2) pairs for a given feature/target slice.
    Non-finite r2 values are kept as NaN so gaps render correctly.
    """
    subset = [
        r for r in rows
        if r["feature"] == feature and r["target"] == target
    ]
    subset.sort(key=lambda r: int(r["k"]))
    out = []
    for r in subset:
        try:
            r2 = float(r["r2"])
        except (KeyError, ValueError):
            r2 = float("nan")
        out.append((int(r["k"]), r2))
    return out


# ---------------------------------------------------------------------------
# Matplotlib setup -- identical rcParams system as the source scripts
# ---------------------------------------------------------------------------

def _setup_matplotlib():
    """
    Publication-quality rcParams matching NeurIPS / ICLR / ICML standards.
    See individual training scripts for full design rationale.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required. Install with: pip install matplotlib"
        ) from e

    plt.rcParams.update({
        # --- Resolution ---
        "figure.dpi":            150,
        "savefig.dpi":           300,

        # --- Typography ---
        "font.family":           "serif",
        "font.serif":            ["DejaVu Serif", "Times New Roman", "serif"],
        "mathtext.fontset":      "dejavuserif",
        "font.size":             9,
        "axes.titlesize":        10,
        "axes.labelsize":        9,
        "xtick.labelsize":       8,
        "ytick.labelsize":       8,
        "legend.fontsize":       8,
        "legend.title_fontsize": 8,

        # --- Colour / text ---
        "text.color":            INK,
        "axes.labelcolor":       INK,
        "xtick.color":           INK,
        "ytick.color":           INK,

        # --- Axes spines ---
        "axes.edgecolor":        INK,
        "axes.linewidth":        0.8,
        "axes.spines.top":       False,
        "axes.spines.right":     False,

        # --- Ticks ---
        "xtick.direction":       "in",
        "ytick.direction":       "in",
        "xtick.major.size":      3.5,
        "ytick.major.size":      3.5,
        "xtick.major.width":     0.8,
        "ytick.major.width":     0.8,
        "xtick.minor.visible":   False,
        "ytick.minor.visible":   False,
        "xtick.major.pad":       3,
        "ytick.major.pad":       3,

        # --- Grid ---
        "axes.grid":             True,
        "axes.grid.axis":        "y",
        "grid.color":            GRID,
        "grid.linewidth":        0.6,
        "grid.linestyle":        "--",
        "grid.alpha":            0.7,

        # --- Lines & markers ---
        "lines.linewidth":       1.5,
        "lines.markersize":      5,

        # --- Legend ---
        "legend.frameon":        False,
        "legend.borderpad":      0.4,
        "legend.labelspacing":   0.3,

        # --- Layout ---
        "axes.titlepad":         6,
        "axes.labelpad":         4,
        "figure.constrained_layout.use": True,

        # --- Background ---
        "axes.facecolor":        "white",
        "figure.facecolor":      "white",

        # --- PDF/SVG font embedding ---
        "pdf.fonttype":          42,
        "ps.fonttype":           42,
    })
    return plt


def _savefig(fig, out_path: Path, also_pdf: bool) -> None:
    fig.savefig(out_path, bbox_inches="tight")
    if also_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")


# ---------------------------------------------------------------------------
# Core overlay plot
# ---------------------------------------------------------------------------

def _plot_overlay(
    curves: Dict[str, List[Tuple[int, float]]],
    feature: str,
    target: str,
    plot_dir: Path,
    also_pdf: bool,
    plt,
) -> None:
    """
    Draw one overlay figure for a single (feature, target) pair.
    `curves` maps series key ("seq", "de") to sorted (k, r2) lists.
    Series with no data for this pair are silently skipped.
    """
    all_ks = [k for pts in curves.values() for k, r2 in pts if math.isfinite(r2)]
    if not all_ks:
        return
    k_max = max(all_ks)

    width = max(5.0, 0.38 * k_max + 2.4)
    fig, ax = plt.subplots(figsize=(width, 3.5))

    # --- Draw each series ---
    for s in SERIES:
        pts = curves.get(s["key"], [])
        if not pts:
            continue
        ks  = [k  for k,  _  in pts]
        r2s = [r2 for _,  r2 in pts]
        ax.plot(
            ks,
            r2s,
            color=s["color"],
            marker=s["marker"],
            linewidth=s["lw"],
            markersize=s["ms"],
            markeredgecolor=MARKER_EC,
            markeredgewidth=0.7,
            label=s["label"],
            zorder=s["zorder"],
        )

    # --- y = 0 reference ---
    ax.axhline(0.0, color=ZERO, linewidth=0.8, linestyle="--", zorder=1)

    # --- Axes labels and title ---
    ax.set_xlabel("Number of included trajectories ($k$)")
    ax.set_ylabel("Test $R^2$")
    ax.set_title(f"Learning Curve Comparison: {target} / {feature}")
    ax.set_xticks(list(range(1, k_max + 1)))

    # --- Legend ---
    ax.legend(
        loc="lower right",
        handlelength=2.0,
        handletextpad=0.5,
        borderaxespad=0.5,
    )

    _savefig(
        fig,
        plot_dir / f"overlay_{_safe_token(target)}_{_safe_token(feature)}.png",
        also_pdf,
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def _resolve_csv_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    """
    Return (seq_csv, de_csv).
    Explicit CLI paths take priority; otherwise derive from --bundle-dir.
    """
    if args.sequential_csv and args.de_csv:
        return Path(args.sequential_csv), Path(args.de_csv)

    if not args.bundle_dir:
        raise ValueError(
            "Either --bundle-dir or both --sequential-csv and --de-csv "
            "must be supplied."
        )
    base = Path(args.bundle_dir).resolve() / args.robot / "training"
    return (
        base / "sequential_clip_training" / "sequential_learning_curve.csv",
        base / "data_efficient_training"  / "cumulative_learning_curve.csv",
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Overlay learning curves from sequential and data-efficient "
            "training studies."
        )
    )
    # Option A: derive paths from bundle dir
    p.add_argument("--bundle-dir", type=Path, default=None,
                   help="Root bundle directory (same as used in training scripts).")
    p.add_argument("--robot", default="go1")

    # Option B: supply CSV paths explicitly
    p.add_argument("--sequential-csv", default=None,
                   help="Path to sequential_learning_curve.csv")
    p.add_argument("--de-csv", default=None,
                   help="Path to cumulative_learning_curve.csv (data-efficient)")

    p.add_argument("--features", default=None,
                   help="Comma-separated feature names to plot. "
                        "Default: all found in CSVs.")
    p.add_argument("--targets", default=None,
                   help="Comma-separated target names to plot. "
                        "Default: all found in CSVs.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to write overlay plots. "
                        "Default: <bundle-dir>/<robot>/training/overlay_plots")
    p.add_argument("--pdf", action="store_true",
                   help="Also save vector PDF versions of each plot.")
    args = p.parse_args()

    seq_csv, de_csv = _resolve_csv_paths(args)

    print("=== Overlay learning curve comparison ===")
    print(f"sequential csv:     {seq_csv}")
    print(f"data-efficient csv: {de_csv}")

    seq_rows = _load_curve_csv(seq_csv)
    de_rows  = _load_curve_csv(de_csv)

    # Union of (feature, target) pairs across both sources
    all_pairs: set = set()
    for rows in (seq_rows, de_rows):
        all_pairs.update(_extract_pairs(rows))

    # Apply CLI filters if given
    feature_filter = set(_csv_list(args.features)) if args.features else None
    target_filter  = set(_csv_list(args.targets))  if args.targets  else None
    pairs = sorted(
        (f, t) for f, t in all_pairs
        if (feature_filter is None or f in feature_filter)
        and (target_filter  is None or t in target_filter)
    )

    if not pairs:
        raise RuntimeError("No (feature, target) pairs found after filtering.")

    # Output directory
    if args.output_dir:
        out_dir = args.output_dir
    elif args.bundle_dir:
        out_dir = (
            Path(args.bundle_dir).resolve()
            / args.robot / "training" / "overlay_plots"
        )
    else:
        out_dir = seq_csv.parent.parent / "overlay_plots"

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir:         {out_dir}")
    print(f"pairs to plot:      {pairs}\n")

    plt = _setup_matplotlib()

    for feature, target in pairs:
        curves = {
            "de":  _get_curve(de_rows,  feature, target),
            "seq": _get_curve(seq_rows, feature, target),
        }
        if not any(curves.values()):
            print(f"  skip (no data): {target} / {feature}")
            continue

        print(f"  plotting: {target} / {feature}")
        _plot_overlay(curves, feature, target, out_dir, args.pdf, plt)

    print(f"\n=== Overlay plots written to: {out_dir} ===")


if __name__ == "__main__":
    main()