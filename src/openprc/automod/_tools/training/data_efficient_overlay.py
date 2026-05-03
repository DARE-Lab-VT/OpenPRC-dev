"""
Data-efficient training overlay figures.

Produces two complementary summary figures from the CSVs written by
data_efficient_training.py:

Figure 1 — Learning-curve overlay (one figure per feature)
  The cumulative R2 curves for all targets are overlaid on the same axes,
  with one figure per feature.  This lets the reader compare how quickly
  each target saturates as training data grows, for a given feature type.

Figure 2 — Single-trajectory ranking heatmap
  Instead of 9 separate bar charts, all single-trajectory R2 values are
  collapsed into one heatmap:
    rows    = trajectory clips, sorted by mean R2 across all columns
    columns = (feature, target) pairs, grouped by feature
    color   = R2, diverging VT Stone -> white -> VT Maroon centred at 0

  This gives an immediate gestalt of which clips are universally useful
  and which are feature- or target-specific.

Run from the repo's src directory:

    python3 -m openprc.automod._tools.training.data_efficient_overlay \\
      --bundle-dir openprc/automod/robot_bundle

Or supply CSV paths directly:

    python3 -m openprc.automod._tools.training.data_efficient_overlay \\
      --curve-csv   path/to/cumulative_learning_curve.csv \\
      --ranking-csv path/to/single_trajectory_ranking.csv \\
      --output-dir  path/to/output
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Color system
#
# Line overlay — two independent visual channels for 9 lines:
#   Color      -> feature  (which sensor modality)
#   Linestyle  -> target   (which quantity to predict)
#
#   feature colors:
#     strain       VT Maroon  #861F41
#     strain_rate  VT Orange  #E5751F
#     node_vel     VT Stone   #75787B
#
#   target linestyles:
#     base_lin_vel  solid   ——
#     base_ang_vel  dashed  - -
#     qvel          dotted  ...
#
# Legend is split into two compact blocks (colors for features, styles for
# targets) so 9 lines only need 6 legend entries total.
#
# Heatmap — same diverging scale as all_clips_heatmap.py:
#   VT Stone -> near-white -> VT Maroon, centred at R2 = 0
# ---------------------------------------------------------------------------

FEATURE_COLORS: Dict[str, str] = {
    "strain":      "#861F41",   # VT Maroon — warm dark red
    "strain_rate": "#E5751F",   # VT Orange — warm mid orange
    "node_vel":    "#75787B",   # VT Stone  — neutral grey
}

TARGET_LINESTYLES: Dict[str, Dict] = {
    "base_lin_vel": {"ls": "-",        "lw": 3.0},   # solid,  heaviest
    "base_ang_vel": {"ls": "--",       "lw": 3.0},   # dashed, medium
    "qvel":         {"ls": (0,(1,2)),   "lw": 3.0},   # dotted, loose spacing
}

# Kept for heatmap fallback only
TARGET_STYLES: List[Dict] = [
    {"target": "base_lin_vel", "color": "#861F41", "marker": "o",
     "ms": 5,   "lw": 1.6, "zorder": 5},
    {"target": "base_ang_vel", "color": "#E5751F", "marker": "s",
     "ms": 4.5, "lw": 1.4, "zorder": 4},
    {"target": "qvel",         "color": "#75787B", "marker": "D",
     "ms": 4,   "lw": 1.3, "zorder": 3},
]

_FALLBACK_COLORS  = ["#4477AA", "#228833", "#AA3377"]
_FALLBACK_MARKERS = ["^", "v", "P"]
_FALLBACK_STYLES  = ["-", "--", "-.", ":"]

INK       = "#2E3440"
GRID      = "#D8DEE9"
ZERO      = "#888888"
MARKER_EC = "#FFFFFF"

HMAP_LOW  = "#75787B"
HMAP_MID  = "#F7F7F7"
HMAP_HIGH = "#861F41"


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


def _r2(row: Dict[str, str]) -> float:
    try:
        v = float(row["r2"])
        return v if math.isfinite(v) else float("nan")
    except (KeyError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Colormap
# ---------------------------------------------------------------------------

def _make_diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    def _hex_rgb(h: str) -> Tuple[float, float, float]:
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    lo, mid, hi = _hex_rgb(HMAP_LOW), _hex_rgb(HMAP_MID), _hex_rgb(HMAP_HIGH)
    cdict = {
        ch: [(0.0, lo[i], lo[i]), (0.5, mid[i], mid[i]), (1.0, hi[i], hi[i])]
        for i, ch in enumerate(("red", "green", "blue"))
    }
    return LinearSegmentedColormap("vt_div", cdict, N=256)


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
        "figure.dpi":            150,
        "savefig.dpi":           300,
        "font.family":           "serif",
        "font.serif":            ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "mathtext.fontset":      "stix",
        "svg.fonttype":          "none",
        "font.size":             9,
        "axes.titlesize":        10,
        "axes.labelsize":        9,
        "xtick.labelsize":       8,
        "ytick.labelsize":       8,
        "legend.fontsize":       8,
        "legend.title_fontsize": 8,
        "text.color":            INK,
        "axes.labelcolor":       INK,
        "xtick.color":           INK,
        "ytick.color":           INK,
        "axes.edgecolor":        INK,
        "axes.linewidth":        0.8,
        "axes.spines.top":       False,
        "axes.spines.right":     False,
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
        "axes.grid":             True,
        "axes.grid.axis":        "y",
        "grid.color":            GRID,
        "grid.linewidth":        0.6,
        "grid.linestyle":        "--",
        "grid.alpha":            0.7,
        "lines.linewidth":       1.5,
        "lines.markersize":      5,
        "legend.frameon":        False,
        "legend.borderpad":      0.4,
        "legend.labelspacing":   0.3,
        "axes.titlepad":         6,
        "axes.labelpad":         4,
        "figure.constrained_layout.use": True,
        "axes.facecolor":        "white",
        "figure.facecolor":      "white",
        "pdf.fonttype":          42,
        "ps.fonttype":           42,
    })
    return plt


def _savefig(fig, out_path: Path, also_pdf: bool) -> None:
    fig.savefig(out_path, bbox_inches="tight")
    if also_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")


# ---------------------------------------------------------------------------
# Figure 1 — 3-row subplot overlay (one row per target, features overlaid)
# ---------------------------------------------------------------------------

def _plot_curve_overlays(
    curve_rows: List[Dict[str, str]],
    features: List[str],
    targets: List[str],
    out_dir: Path,
    also_pdf: bool,
    plt,
) -> None:
    """
    Three vertically stacked subplots — one per target.
    Within each subplot, the three feature curves are overlaid.

    Encoding (consistent across all subplots):
      Color      = feature   VT Maroon / VT Orange / VT Stone
      Linestyle  = feature   solid / dashed / dotted  (redundant channel
                             for greyscale / colorblind safety)
      Markers    = feature   circle / square / diamond

    Each subplot has its own y-axis so targets with very different R2
    ranges (e.g. base_lin_vel ~0.96 vs base_ang_vel ~0.65) are not
    compressed onto a shared scale.  A shared x-axis with ticks only on
    the bottom panel keeps the figure compact.

    The legend appears once, to the right of the top subplot.
    """
    import matplotlib.lines as mlines

    all_ks = [int(r["k"]) for r in curve_rows if math.isfinite(_r2(r))]
    if not all_ks:
        raise RuntimeError("No finite R2 values found in curve CSV.")
    k_max = max(all_ks)

    width = max(4.0, 0.24 * k_max + 2.0)

    # Per-feature visual spec — color + linestyle + marker, all three channels
    feature_specs = []
    for fi, feature in enumerate(features):
        color  = FEATURE_COLORS.get(feature, _FALLBACK_COLORS[fi % len(_FALLBACK_COLORS)])
        ts     = list(TARGET_LINESTYLES.values())
        ls     = ts[fi % len(ts)]["ls"] if fi < len(ts) else _FALLBACK_STYLES[fi % 4]
        lw     = ts[fi % len(ts)]["lw"] if fi < len(ts) else 1.4
        marker = ["o", "s", "D", "^"][fi % 4]
        feature_specs.append({
            "feature": feature,
            "color":   color,
            "ls":      ls,
            "lw":      lw,
            "marker":  marker,
        })

    legend_handles = [
        mlines.Line2D(
            [], [],
            color=s["color"],
            linestyle=s["ls"],
            linewidth=s["lw"],
            marker=s["marker"],
            markersize=8,
            markeredgecolor=MARKER_EC,
            markeredgewidth=0.6,
            label=s["feature"],
        )
        for s in feature_specs
    ]

    for target in targets:
        fig, ax = plt.subplots(figsize=(width, 3.5))

        for spec in feature_specs:
            rows = sorted(
                [r for r in curve_rows
                 if r["feature"] == spec["feature"] and r["target"] == target],
                key=lambda r: int(r["k"]),
            )
            if not rows:
                continue

            ax.plot(
                [int(r["k"]) for r in rows],
                [_r2(r) for r in rows],
                color=spec["color"],
                linestyle=spec["ls"],
                linewidth=spec["lw"],
                marker=spec["marker"],
                markersize=8,
                markeredgecolor=MARKER_EC,
                markeredgewidth=0.6,
                zorder=3,
            )

        ax.axhline(0.0, color=ZERO, linewidth=0.8, linestyle="--", zorder=1)
        ax.set_xlabel("Number of included trajectories ($k$)", fontsize=18)
        ax.set_ylabel("Test $R^2$", fontsize=18)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Sequential Learning Curves — {target}")
        ax.set_xticks(list(range(1, k_max + 1)))
        ax.tick_params(axis="both", labelsize=16)
        ax.legend(
            handles=legend_handles,
            title="Feature",
            loc="upper right",
            handlelength=2.2,
            handletextpad=0.5,
            labelspacing=0.4,
            fontsize=18,
            title_fontsize=18
        )

        _savefig(fig, out_dir / f"curve_overlay_{_safe_token(target)}.pdf", also_pdf)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — single-trajectory ranking heatmap
# ---------------------------------------------------------------------------

def _luminance(rgba) -> float:
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * lin(rgba[0]) + 0.7152 * lin(rgba[1]) + 0.0722 * lin(rgba[2])


def _plot_ranking_heatmap(
    ranking_rows: List[Dict[str, str]],
    features: List[str],
    targets: List[str],
    out_dir: Path,
    also_pdf: bool,
    plt,
) -> None:
    """
    Rows    = trajectory clips, sorted descending by mean R2 across all columns
    Columns = (feature, target) pairs grouped by feature
    Color   = R2, diverging VT Stone -> white -> VT Maroon centred at 0

    Figure sizing rationale
    -----------------------
    n_cols = 9 (3 features x 3 targets).
    Column width: 1.1 in — wide enough for rotated target labels without
      clipping, and for 4-character R2 values (e.g. "0.963") at 7pt.
    Row height: 0.40 in — compact for up to ~14 trajectory rows while still
      keeping 7pt annotations readable.
    Extra margins: +1.5 in right for colorbar, +1.2 in bottom for two-tier
      x-axis labels (target name + feature group label).
    """
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm, Normalize

    cmap = _make_diverging_cmap()

    # Ordered columns: features grouped, targets within each group
    cols: List[Tuple[str, str]] = [
        (feature, target)
        for feature in features
        for target in targets
    ]
    n_cols = len(cols)
    n_targets = len(targets)

    # Trajectories sorted by mean R2 descending
    all_trajectories = sorted({r["trajectory"] for r in ranking_rows})
    lookup: Dict[Tuple[str, str, str], float] = {
        (r["feature"], r["target"], r["trajectory"]): _r2(r)
        for r in ranking_rows
    }

    # Sort trajectories by natural numeric order (corridor_000, corridor_001, ...)
    import re
    def _natural_key(s: str) -> tuple:
        match = re.search(r"(\d+)$", s)
        return (s[:match.start()], int(match.group(1))) if match else (s, -1)

    trajectories = sorted(all_trajectories, key=_natural_key)
    n_rows = len(trajectories)

    # Data matrix
    data = np.full((n_rows, n_cols), float("nan"))
    for i, traj in enumerate(trajectories):
        for j, (feature, target) in enumerate(cols):
            data[i, j] = lookup.get((feature, target, traj), float("nan"))

    # Normalisation
    finite_vals = data[np.isfinite(data)]
    if len(finite_vals) == 0:
        raise RuntimeError("No finite R2 values in ranking data.")
    vmin = min(float(finite_vals.min()), -0.05)
    vmax = max(float(finite_vals.max()),  1.00)
    norm = (
        TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        if vmin < 0 < vmax
        else Normalize(vmin=vmin, vmax=vmax)
    )

    # ---------------------------------------------------------------------------
    # Figure sizing
    # col_w  = 1.1 in per column  (fits rotated labels + 5-char values at 7pt)
    # row_h  = 0.40 in per row    (compact; 7pt text fits comfortably)
    # pad_r  = 1.5 in             (colorbar + right margin)
    # pad_b  = 1.4 in             (target tick labels rotated 35° + feature labels)
    # pad_t  = 0.5 in             (title)
    # pad_l  = 1.6 in             (trajectory y-tick labels, longest ~15 chars)
    # ---------------------------------------------------------------------------
    col_w, row_h       = 1.1, 0.40
    pad_l, pad_r       = 1.6, 1.5
    pad_t, pad_b       = 0.5, 1.4
    fig_w = col_w * n_cols + pad_l + pad_r
    fig_h = row_h * n_rows + pad_t + pad_b

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Suppress heatmap-irrelevant rcParam effects
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(bottom=False, left=False)

    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    # Thin cell borders
    for x in range(n_cols + 1):
        ax.axvline(x - 0.5, color=INK, linewidth=0.4, zorder=2)
    for y in range(n_rows + 1):
        ax.axhline(y - 0.5, color=INK, linewidth=0.4, zorder=2)

    # Thick feature-group separators
    for g in range(1, len(features)):
        ax.axvline(g * n_targets - 0.5, color=INK, linewidth=1.8, zorder=3)

    # Cell value annotations
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            label = f"{val:.2f}" if math.isfinite(val) else "n/a"
            fill  = cmap(norm(val) if math.isfinite(val) else 0.5)
            txt   = "#F7F7F7" if _luminance(fill) < 0.35 else INK
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=11, color=txt, zorder=4)

    # Y-axis: trajectory labels
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(trajectories, fontsize=11)
    ax.set_ylabel("Trajectory", labelpad=4, fontsize=14)

    # X-axis bottom tier: target names
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [t for _, t in cols],
        rotation=0, fontsize=11,
    )

    # X-axis second tier: feature group labels centred over each group
    # Drawn via ax.text in axis-fraction x, data y using a blended transform
    import matplotlib.transforms as mtransforms
    blended = mtransforms.blended_transform_factory(
        ax.transData, ax.transAxes
    )
    for g, feature in enumerate(features):
        centre_x = g * n_targets + (n_targets - 1) / 2.0
        ax.text(
            centre_x, -0.18,
            feature,
            ha="center", va="top",
            fontsize=14, fontweight="bold",
            color=INK,
            transform=blended,
        )

    ax.set_title(
        "Single-Trajectory $R^2$ — All Features & Targets",
        pad=8,
    )

    # Colorbar — narrow, right-hand side
    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.02)
    cbar.set_label("Test $R^2$", fontsize=14)
    cbar.ax.tick_params(labelsize=11)

    _savefig(fig, out_dir / "ranking_heatmap.pdf", also_pdf)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    if args.curve_csv and args.ranking_csv:
        return Path(args.curve_csv), Path(args.ranking_csv)
    if not args.bundle_dir:
        raise ValueError(
            "Either --bundle-dir or both --curve-csv and --ranking-csv must be given."
        )
    de_base  = (
        Path(args.bundle_dir).resolve()
        / args.robot / "training" / "data_efficient_training"
    )
    seq_base = (
        Path(args.bundle_dir).resolve()
        / args.robot / "training" / "sequential_clip_training"
    )
    return (
        seq_base / "sequential_learning_curve.csv",
        de_base  / "cumulative_learning_curve.csv",
        de_base  / "single_trajectory_ranking.csv",
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Overlay data-efficient learning curves and produce a "
            "single-trajectory ranking heatmap."
        )
    )
    p.add_argument("--bundle-dir",   type=Path, default=None)
    p.add_argument("--robot",        default="go1")
    p.add_argument("--curve-csv",    default=None,
                   help="Path to sequential_learning_curve.csv (natural clip order)")
    p.add_argument("--fallback-curve-csv", default=None,
                   help="Path to cumulative_learning_curve.csv used as fallback "
                        "for targets missing from --curve-csv (e.g. qvel).")
    p.add_argument("--ranking-csv",  default=None,
                   help="Path to single_trajectory_ranking.csv")
    p.add_argument("--features",     default="strain,strain_rate,node_vel",
                   help="Comma-separated features. Default: all three.")
    p.add_argument("--targets",      default="base_lin_vel,base_ang_vel,qvel",
                   help="Comma-separated targets. Default: all three.")
    p.add_argument("--output-dir",   type=Path, default=None)
    p.add_argument("--pdf",          action="store_true",
                   help="Also save vector PDF versions.")
    args = p.parse_args()

    curve_csv, fallback_curve_csv, ranking_csv = _resolve_paths(args)
    features = _csv_list(args.features)
    targets  = _csv_list(args.targets)

    print("=== Data-efficient overlay figures ===")
    print(f"seq csv:          {curve_csv}")
    print(f"fallback curve csv: {fallback_curve_csv}")
    print(f"ranking csv:        {ranking_csv}")
    print(f"features:    {features}")
    print(f"targets:     {targets}")

    seq_rows      = _load_csv(curve_csv)
    fallback_rows = _load_csv(fallback_curve_csv)
    ranking_rows  = _load_csv(ranking_csv)

    # Merge: for each (feature, target) use seq_rows if any rows exist,
    # otherwise fall back to fallback_rows (handles qvel and other
    # multi-dim targets absent from the sequential CSV).
    seq_pairs = {(r["feature"], r["target"]) for r in seq_rows}
    curve_rows = seq_rows + [
        r for r in fallback_rows
        if (r["feature"], r["target"]) not in seq_pairs
    ]
    missing = {(r["feature"], r["target"]) for r in fallback_rows} - seq_pairs
    if missing:
        print(f"  NOTE: using fallback CSV for {sorted(missing)}")

    if args.output_dir:
        out_dir = args.output_dir
    elif args.bundle_dir:
        out_dir = (
            Path(args.bundle_dir).resolve()
            / args.robot / "training"
            / "data_efficient_training" / "overlay_plots"
        )
    else:
        out_dir = curve_csv.parent / "overlay_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir:  {out_dir}\n")

    plt = _setup_matplotlib()

    print("Plotting 9-line curve overlay...")
    _plot_curve_overlays(curve_rows, features, targets, out_dir, args.pdf, plt)

    print("Plotting ranking heatmap...")
    _plot_ranking_heatmap(ranking_rows, features, targets, out_dir, args.pdf, plt)

    print(f"\n=== Done — plots written to: {out_dir} ===")


if __name__ == "__main__":
    main()