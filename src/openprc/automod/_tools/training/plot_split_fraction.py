"""
Overlay split-method comparison plots for the split-fraction sweep.

Reuses _build_specs and _collect_rows from split_fraction_sweep.py to read
from the existing splitfrac_* run directories, then overlays all split
methods on the same figure.

Run from the repo's src directory:

    python3 -m openprc.automod._tools.training.plot_split_method_overlay \\
      --bundle-dir openprc/automod/robot_bundle
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from split_fraction_sweep import (  # noqa: E402
    FRACTIONS_DEFAULT,
    FEATURES_DEFAULT,
    TARGETS_DEFAULT,
    _build_specs,
    _collect_rows,
    _csv_list,
    _float_list,
    _resolve_trajectories,
    _safe_token,
    _finite_values,
    _setup_matplotlib,
    _savefig,
)


# ---------------------------------------------------------------------------
# One colour + marker per split method
# ---------------------------------------------------------------------------
METHOD_STYLES = {
    "fixed-test": {"color": "#861F41", "marker": "o"},   # VT Maroon
    "concat":     {"color": "#E5751F", "marker": "s"},   # VT Orange
    "in-clip":          {"color": "#75787B", "marker": "^"},   # muted blue
    "concat-fixed-test": {"color": "#E5751F", "marker": "s"},   # VT Orange
}
FALLBACK_STYLES = [
    {"color": "#2CA02C", "marker": "D"},
]

COLORS_ZERO = "#888888"
COLORS_EC   = "#FFFFFF"


def _style_for(method: str, idx: int) -> Dict:
    if method in METHOD_STYLES:
        return METHOD_STYLES[method]
    return FALLBACK_STYLES[idx % len(FALLBACK_STYLES)]


# ---------------------------------------------------------------------------
# Plot — same structure as _plot_curves() but with multiple lines
# ---------------------------------------------------------------------------

def _plot_overlay_curves(
    sweeps: List[tuple],          # [(label, split_method, rows), ...]
    features: Sequence[str],
    targets: Sequence[str],
    plot_dir: Path,
    also_pdf: bool,
) -> None:
    plt = _setup_matplotlib()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    for feature in features:
        for target in targets:
            all_fractions = sorted({
                float(r["train_fraction"])
                for _label, _method, rows in sweeps
                for r in rows
                if r["feature"] == feature and r["target"] == target
            })
            if not all_fractions:
                print(f"  no data for target={target}, feature={feature} — skipping")
                continue

            fig, ax = plt.subplots(figsize=(5.5, 3.2))
            plotted_any = False

            for idx, (label, split_method, rows) in enumerate(sweeps):
                subset = sorted(
                    (r for r in rows if r["feature"] == feature and r["target"] == target),
                    key=lambda r: float(r["train_fraction"]),
                )
                if not subset:
                    continue

                fractions = [100.0 * float(r["train_fraction"]) for r in subset]
                r2s      = _finite_values([r["r2"]     for r in subset])
                lambdas  = _finite_values([r["lambda"] for r in subset])
                style    = _style_for(split_method, idx)

                ax.plot(
                    fractions, r2s,
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=1.5,
                    markersize=6,
                    markeredgecolor=COLORS_EC,
                    markeredgewidth=0.8,
                    label=label,
                    zorder=3 + idx,
                )

                for x, y, lam in zip(fractions, r2s, lambdas):
                    if math.isfinite(y):
                        ann = f"{y:.3f}"
                        # if math.isfinite(lam):
                        #     ann += f"\n$\\lambda$={lam:g}"
                        va = "bottom" if y >= 0 else "top"
                        ax.annotate(
                            ann,
                            xy=(x, y),
                            xytext=(0, 5 if y >= 0 else -5),
                            textcoords="offset points",
                            ha="center", va=va,
                            fontsize=12,
                            color=style["color"],
                        )
                plotted_any = True

            if not plotted_any:
                plt.close(fig)
                continue

            ax.axhline(0.0, color=COLORS_ZERO, linewidth=0.8, linestyle="--", zorder=1)
            x_ticks = [100.0 * f for f in all_fractions]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f"{int(x)}%" for x in x_ticks])
            ax.set_xlabel("Training fraction", fontsize=14)
            ax.set_ylabel("Test $R^2$", fontsize=14)
            ax.set_title(f"Split-method comparison: {target} / {feature}")
            ax.set_ylim(0.55, 1.05)
            ax.tick_params(axis="both", labelsize=12)
            ax.legend(loc="lower right", fontsize=12, title_fontsize=14)

            fname = f"split_fraction_overlay_{_safe_token(target)}_{_safe_token(feature)}.pdf"
            _savefig(fig, plot_dir / fname, also_pdf)
            plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Overlay split-method plots by reusing split_fraction_sweep data."
    )
    p.add_argument("--bundle-dir", required=True, type=Path)
    p.add_argument("--robot", default="go1")
    p.add_argument("--features", default=FEATURES_DEFAULT)
    p.add_argument("--targets", default=TARGETS_DEFAULT)
    p.add_argument("--train-fractions", default=FRACTIONS_DEFAULT)
    p.add_argument(
        "--split-methods", default="fixed-test,concat",
        help="Comma-separated split methods to overlay. Default: fixed-test,concat",
    )
    p.add_argument(
        "--labels", default=None,
        help="Comma-separated legend labels, in the same order as --split-methods. "
             "Default: the split method name itself.",
    )
    p.add_argument("--trajectories", default=None)
    p.add_argument("--target-fps", type=float, default=50.0)
    p.add_argument("--skip-seconds", type=float, default=2.0)
    p.add_argument("--fixed-test-fraction", type=float, default=0.3)
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Default: <bundle-dir>/<robot>/training/split_fraction_sweep/plots/",
    )
    p.add_argument("--pdf", action="store_true")
    args = p.parse_args()

    args.bundle_dir = args.bundle_dir.resolve()
    features      = _csv_list(args.features)
    targets       = _csv_list(args.targets)
    fractions     = _float_list(args.train_fractions)
    split_methods = _csv_list(args.split_methods)
    labels        = _csv_list(args.labels) if args.labels else split_methods
    if len(labels) != len(split_methods):
        p.error(f"--labels count ({len(labels)}) must match --split-methods count ({len(split_methods)}).")
    trajectories  = _resolve_trajectories(args)

    # For each split method, rebuild the same specs _build_specs would have
    # used, then read metrics from the existing run dirs via _collect_rows.
    sweeps = []
    for label, split_method in zip(labels, split_methods):
        specs = _build_specs(features, targets, fractions, trajectories, split_method)
        rows  = _collect_rows(args, specs)
        sweeps.append((label, split_method, rows))
        print(f"  loaded {len(rows)} rows for split_method={split_method} (label={label!r})")

    plot_dir = args.output_dir or (
        args.bundle_dir / args.robot / "training" / "split_fraction_sweep" / "plots"
    )
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"features:      {features}")
    print(f"targets:       {targets}")
    print(f"split methods: {split_methods}")
    print(f"output:        {plot_dir}")

    _plot_overlay_curves(sweeps, features, targets, plot_dir, args.pdf)

    print(f"\n=== Overlay plots complete: {plot_dir} ===")


if __name__ == "__main__":
    main()