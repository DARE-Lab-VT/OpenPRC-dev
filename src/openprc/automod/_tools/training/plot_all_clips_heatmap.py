"""
All-clips R2 heatmap.

Reads metrics.csv from a single training run that used all available
trajectories, and plots a heatmap of Test R2 with targets on rows and
features on columns.

This gives an at-a-glance view of which (feature, target) combinations
are informative when the full dataset is used.

The colormap is a custom diverging scale:
  VT Stone  (#75787B) -> white -> VT Maroon (#861F41)
centered at R2 = 0 (chance level), so productive pairs stand out in maroon
and below-chance pairs recede into grey.

Data source
-----------
The input CSV must have at minimum the columns:
  feature, target, r2

This is satisfied by:
  - The final row of sequential_learning_curve.csv        (all clips added)
  - The final row of sequential_concat_learning_curve.csv (all clips added)
  - The final row of cumulative_learning_curve.csv        (all clips added)
  - Any custom CSV with those three columns

Run from the repo's src directory:

    python3 -m openprc.automod._tools.training.all_clips_heatmap \\
      --bundle-dir openprc/automod/robot_bundle

Or supply the CSV directly:

    python3 -m openprc.automod._tools.training.all_clips_heatmap \\
      --csv   path/to/sequential_learning_curve.csv \\
      --output-dir path/to/output
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Color palette
#
# Diverging scale centered at R2 = 0:
#   below chance  -> VT Stone  #75787B  (neutral grey, recedes)
#   chance (R2=0) -> #F7F7F7           (near-white pivot)
#   perfect (R2=1)-> VT Maroon #861F41  (saturated, advances)
#
# Cell text is white on dark cells, ink on light cells (auto-contrast).
# ---------------------------------------------------------------------------
TARGETS_DEFAULT  = "base_lin_vel,base_ang_vel,qvel"
FEATURES_DEFAULT = "strain,strain_rate,node_vel"

COLORS = {
    "ink":          "#2E3440",
    "grid":         "#D8DEE9",
    "maroon":       "#861F41",   # VT Maroon — high R2 end
    "stone":        "#75787B",   # VT Stone  — negative R2 end
    "pivot":        "#F7F7F7",   # near-white — R2 = 0 pivot
    "text_light":   "#F7F7F7",   # cell annotation on dark fill
    "text_dark":    "#2E3440",   # cell annotation on light fill
    "zero_line":    "#888888",
}

# ---------------------------------------------------------------------------
# Colormap builder
# ---------------------------------------------------------------------------

def _make_diverging_cmap():
    """
    Build a three-point diverging colormap:
      stone -> pivot (at vmin..0) and pivot -> maroon (at 0..vmax)
    using matplotlib's LinearSegmentedColormap.
    """
    from matplotlib.colors import LinearSegmentedColormap

    def _hex_to_rgb(h: str) -> Tuple[float, float, float]:
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    stone  = _hex_to_rgb(COLORS["stone"])
    pivot  = _hex_to_rgb(COLORS["pivot"])
    maroon = _hex_to_rgb(COLORS["maroon"])

    cdict = {
        "red":   [(0.0, stone[0],  stone[0]),
                  (0.5, pivot[0],  pivot[0]),
                  (1.0, maroon[0], maroon[0])],
        "green": [(0.0, stone[1],  stone[1]),
                  (0.5, pivot[1],  pivot[1]),
                  (1.0, maroon[1], maroon[1])],
        "blue":  [(0.0, stone[2],  stone[2]),
                  (0.5, pivot[2],  pivot[2]),
                  (1.0, maroon[2], maroon[2])],
    }
    return LinearSegmentedColormap("vt_diverging", cdict, N=256)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _csv_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _safe_token(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in s)


def _load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _extract_all_clips_rows(
    rows: List[Dict[str, str]],
    feature_filter: Optional[List[str]],
    target_filter: Optional[List[str]],
) -> List[Dict[str, str]]:
    """
    Keep only the row with the largest k per (feature, target) pair —
    that is the all-clips result. If the CSV has no k column (e.g. a
    hand-built CSV), keep all rows.
    """
    has_k = "k" in (rows[0].keys() if rows else [])

    if has_k:
        # Group by (feature, target), keep row with max k
        best: Dict[Tuple[str, str], Dict[str, str]] = {}
        for r in rows:
            key = (r["feature"], r["target"])
            if key not in best or int(r["k"]) > int(best[key]["k"]):
                best[key] = r
        rows = list(best.values())

    if feature_filter:
        rows = [r for r in rows if r["feature"] in feature_filter]
    if target_filter:
        rows = [r for r in rows if r["target"] in target_filter]

    return rows


def _build_grid(
    rows: List[Dict[str, str]],
    expected_features: Optional[List[str]] = None,
    expected_targets: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[List[float]]]:
    """
    Return (targets, features, grid) where grid[i][j] is R2 for
    targets[i] x features[j]. Targets on rows, features on columns.
    Missing cells are NaN.

    If expected_features / expected_targets are given, the grid always
    includes those rows/columns (as NaN) even if absent from the CSV.
    A warning is printed for any expected entry not found in the data.
    This handles e.g. qvel being stored under a different name in
    metrics.csv (qvel_0, qvel_1 ...) so the cell shows n/a rather
    than silently disappearing from the heatmap.
    """
    found_features = sorted({r["feature"] for r in rows})
    found_targets  = sorted({r["target"]  for r in rows})

    if expected_features:
        missing_f = sorted(set(expected_features) - set(found_features))
        if missing_f:
            print(f"  WARNING: expected features not found in CSV: {missing_f}")
        features = sorted(set(found_features) | set(expected_features))
    else:
        features = found_features

    if expected_targets:
        missing_t = sorted(set(expected_targets) - set(found_targets))
        if missing_t:
            print(f"  WARNING: expected targets not found in CSV: {missing_t}")
            print(f"           These will appear as n/a in the heatmap.")
            print(f"           Likely cause: multi-dim target stored under a")
            print(f"           different name in metrics.csv (e.g. qvel_0...).")
        targets = sorted(set(found_targets) | set(expected_targets))
    else:
        targets = found_targets

    lookup = {(r["feature"], r["target"]): r for r in rows}

    grid = []
    for target in targets:
        row_vals = []
        for feature in features:
            cell = lookup.get((feature, target))
            if cell is None:
                row_vals.append(float("nan"))
            else:
                try:
                    row_vals.append(float(cell["r2"]))
                except (KeyError, ValueError):
                    row_vals.append(float("nan"))
        grid.append(row_vals)

    return targets, features, grid


# ---------------------------------------------------------------------------
# Matplotlib setup
# ---------------------------------------------------------------------------

def _setup_matplotlib():
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
        "font.serif":            ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "mathtext.fontset":      "stix",
        "svg.fonttype":          "none",
        "font.size":             9,
        "axes.titlesize":        10,
        "axes.labelsize":        9,
        "xtick.labelsize":       8,
        "ytick.labelsize":       8,

        # --- Colour / text ---
        "text.color":            COLORS["ink"],
        "axes.labelcolor":       COLORS["ink"],
        "xtick.color":           COLORS["ink"],
        "ytick.color":           COLORS["ink"],

        # --- Axes spines (all off for heatmap — cell borders replace them) ---
        "axes.edgecolor":        COLORS["ink"],
        "axes.linewidth":        0.8,
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "axes.spines.left":      False,
        "axes.spines.bottom":    False,

        # --- Ticks (none for heatmap) ---
        "xtick.bottom":          False,
        "ytick.left":            False,
        "xtick.major.pad":       4,
        "ytick.major.pad":       4,

        # --- Grid (off — cell borders do this job) ---
        "axes.grid":             False,

        # --- Layout ---
        "axes.titlepad":         8,
        "axes.labelpad":         6,
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
# Core plot
# ---------------------------------------------------------------------------

def _luminance(hex_color: str) -> float:
    """Perceived luminance of a hex color (0=dark, 1=light)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    # sRGB linearisation then ITU-R BT.709 coefficients
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _cell_text_color(fill_rgba) -> str:
    """Return ink or light text depending on fill luminance."""
    r, g, b = fill_rgba[0], fill_rgba[1], fill_rgba[2]
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    return COLORS["text_light"] if lum < 0.35 else COLORS["text_dark"]


def _plot_heatmap(
    targets: List[str],
    features: List[str],
    grid: List[List[float]],
    title: str,
    out_path: Path,
    also_pdf: bool,
    plt,
    vmin: float,
    vmax: float,
) -> None:
    import numpy as np

    cmap = _make_diverging_cmap()

    n_rows = len(targets)
    n_cols = len(features)

    # Cell size: 1.5 in wide, 1.1 in tall — comfortable for text annotations
    fig_w = max(4.0, 1.5 * n_cols + 1.2)   # +1.2 for colorbar
    fig_h = max(2.5, 1.1 * n_rows + 0.8)   # +0.8 for title + x labels

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    data = np.array(grid, dtype=float)

    # --- Draw heatmap via imshow ---
    # vcenter at 0 so the colormap pivot aligns with chance level
    from matplotlib.colors import TwoSlopeNorm
    # Guard: if all values are on one side of zero, fall back to plain Normalize
    finite_vals = [v for row in grid for v in row if math.isfinite(v)]
    if not finite_vals:
        raise RuntimeError("No finite R2 values found in grid.")

    data_min = min(finite_vals)
    data_max = max(finite_vals)

    if data_min < 0 < data_max:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        # All values on one side — use simple linear norm, pivot loses meaning
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    # --- Cell border lines ---
    # Draw thin ink lines between cells for visual separation
    for x in range(n_cols + 1):
        ax.axvline(x - 0.5, color=COLORS["ink"], linewidth=0.5, zorder=2)
    for y in range(n_rows + 1):
        ax.axhline(y - 0.5, color=COLORS["ink"], linewidth=0.5, zorder=2)

    # --- R2 value annotations inside each cell ---
    for i in range(n_rows):
        for j in range(n_cols):
            val = grid[i][j]
            if not math.isfinite(val):
                label = "n/a"
            else:
                label = f"{val:.3f}"
            fill_rgba = cmap(norm(val) if math.isfinite(val) else 0.5)
            txt_color = _cell_text_color(fill_rgba)
            ax.text(
                j, i, label,
                ha="center", va="center",
                fontsize=12,
                fontweight="bold" if math.isfinite(val) and val > 0 else "normal",
                color=txt_color,
                zorder=3,
            )

    # --- Axis ticks and labels ---
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(features, rotation=0, fontsize=12)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(targets, fontsize=12)
    ax.set_xlabel("Feature", fontsize=16)
    ax.set_ylabel("Target", fontsize=16)
    ax.set_title(title)

    # --- Colorbar ---
    # Compact, right-hand side, labelled at key values
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Test $R^2$", fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    # No zero-line on the colorbar: with mostly positive data it renders
    # as a distracting artifact rather than a useful reference.

    _savefig(fig, out_path, also_pdf)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _resolve_csv_path(args: argparse.Namespace) -> Path:
    if args.csv:
        return Path(args.csv)
    if not args.bundle_dir:
        raise ValueError(
            "Either --bundle-dir or --csv must be supplied."
        )
    # Default: use the data-efficient cumulative curve, whose _read_r2_metrics
    # initialises all requested targets (including multi-dim qvel) and does a
    # dict-key lookup rather than an exact string match, so qvel rows are
    # always written even if the underlying metrics.csv uses sub-dim names.
    return (
        Path(args.bundle_dir).resolve()
        / args.robot / "training"
        / "data_efficient_training"
        / "cumulative_learning_curve.csv"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Plot a heatmap of Test R2 for all (feature, target) pairs "
            "using the all-clips training result."
        )
    )
    p.add_argument("--bundle-dir", type=Path, default=None,
                   help="Root bundle directory.")
    p.add_argument("--robot", default="go1")
    p.add_argument("--csv", default=None,
                   help="Path to a learning-curve CSV. The row with the "
                        "largest k (all clips) is used per (feature, target). "
                        "Default: cumulative_learning_curve.csv from "
                        "data_efficient_training, which correctly handles "
                        "multi-dimensional targets such as qvel.")
    p.add_argument("--features", default=FEATURES_DEFAULT,
                   help="Comma-separated features to include. Default: all three.")
    p.add_argument("--targets", default=TARGETS_DEFAULT,
                   help="Comma-separated targets to include. Default: all three.")
    p.add_argument("--vmin", type=float, default=None,
                   help="Colormap lower bound. Default: min R2 in data.")
    p.add_argument("--vmax", type=float, default=None,
                   help="Colormap upper bound. Default: max(1.0, max R2).")
    p.add_argument("--title", default=None,
                   help="Figure title. Default: auto-generated.")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--pdf", action="store_true",
                   help="Also save a vector PDF.")
    args = p.parse_args()

    csv_path = _resolve_csv_path(args)
    print(f"=== All-clips R2 heatmap ===")
    print(f"source csv: {csv_path}")

    all_rows = _load_csv(csv_path)
    feature_filter = _csv_list(args.features) or None
    target_filter  = _csv_list(args.targets)  or None
    rows = _extract_all_clips_rows(all_rows, feature_filter, target_filter)
    # Note: expected_features / expected_targets are passed to _build_grid below
    # so that any targets absent from the CSV (e.g. qvel stored as qvel_0 ...)
    # still appear as n/a cells rather than silently disappearing.

    if not rows:
        raise RuntimeError("No rows remain after filtering.")

    expected_features = _csv_list(args.features) or None
    expected_targets  = _csv_list(args.targets)  or None
    targets, features, grid = _build_grid(
        rows,
        expected_features=expected_features,
        expected_targets=expected_targets,
    )
    print(f"features:   {features}")
    print(f"targets:    {targets}")

    # Determine colormap bounds
    finite_vals = [v for row in grid for v in row if math.isfinite(v)]
    data_min = min(finite_vals)
    data_max = max(finite_vals)
    vmin = args.vmin if args.vmin is not None else min(data_min, -0.1)
    vmax = args.vmax if args.vmax is not None else max(data_max, 1.0)

    title = args.title or f"All-Clips Test $R^2$ — {args.robot}"

    # Output directory
    if args.output_dir:
        out_dir = args.output_dir
    elif args.bundle_dir:
        out_dir = (
            Path(args.bundle_dir).resolve()
            / args.robot / "training" / "all_clips_heatmap"
        )
    else:
        out_dir = csv_path.parent.parent / "all_clips_heatmap"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "all_clips_r2_heatmap.pdf"
    print(f"output:     {out_path}")

    plt = _setup_matplotlib()
    _plot_heatmap(
        targets, features, grid,
        title=title,
        out_path=out_path,
        also_pdf=args.pdf,
        plt=plt,
        vmin=vmin,
        vmax=vmax,
    )

    print("=== Done ===")


if __name__ == "__main__":
    main()