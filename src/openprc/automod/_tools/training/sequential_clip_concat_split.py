"""
Sequential cumulative PRC readout training with concatenate-then-split.

This is the same cumulative clip experiment as sequential_clip_training.py,
except each training run uses a global split after concatenating selected
clips instead of splitting inside each clip.

It does not modify existing scripts. It launches train_readout_concat_split.py
as a subprocess.

Run from the repo's src directory:

    python3 -m openprc.automod._tools.training.sequential_clip_concat_split \\
      --bundle-dir openprc/automod/robot_bundle
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


TARGETS_DEFAULT = "base_lin_vel,base_ang_vel,qvel"
FEATURES_DEFAULT = "strain,strain_rate,node_vel"

# ---------------------------------------------------------------------------
# Publication-quality color palette (Paul Tol's colorblind-safe muted scheme,
# widely used in NeurIPS / ICLR / ICML figures).
# ---------------------------------------------------------------------------
COLORS = {
    "ink":       "#2E3440",   # near-black for text / spines / ticks
    "grid":      "#D8DEE9",   # light cool grey for gridlines
    "line":      "#861F41",   # VT Maroon — main data line
    "marker_ec": "#FFFFFF",   # white marker edge for visual pop
    "zero":      "#888888",   # neutral mid-grey for the y=0 reference line
    "annot":     "#2E3440",   # annotation text colour (same as ink)
}


@dataclass(frozen=True)
class TrainRunSpec:
    run_id: str
    feature: str
    target: str
    trajectories: List[str]


def _csv_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _safe_token(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in s)


def _natural_key(s: str) -> tuple:
    match = re.search(r"(\d+)$", s)
    if match:
        return (s[:match.start()], int(match.group(1)))
    return (s, -1)


def _run_dir(bundle_dir: Path, robot: str, run_id: str) -> Path:
    return bundle_dir / robot / "training" / run_id


def _load_manifest(bundle_dir: Path) -> dict:
    with (bundle_dir / "manifest.json").open() as f:
        return json.load(f)


def _manifest_trajectories(bundle_dir: Path, robot: str) -> List[dict]:
    manifest = _load_manifest(bundle_dir)
    if robot not in manifest["robots"]:
        raise KeyError(f"robot {robot!r} not in manifest")
    return list(manifest["robots"][robot]["trajectories"])


def _has_simulation(bundle_dir: Path, robot: str, clip_id: str) -> bool:
    return (
        bundle_dir / robot / "reservoir_sims" / clip_id / "output" / "simulation.h5"
    ).exists()


def _resolve_trajectories(args: argparse.Namespace) -> List[str]:
    trajs = _manifest_trajectories(args.bundle_dir, args.robot)
    by_id = {t["id"]: t for t in trajs}
    requested = _csv_list(args.trajectories)

    if requested:
        missing = [tid for tid in requested if tid not in by_id]
        if missing:
            raise KeyError(
                f"Requested trajectory id(s) not in manifest for {args.robot}: "
                f"{missing}. Available examples: {list(by_id)[:10]}"
            )
        no_sim = [
            tid for tid in requested
            if not _has_simulation(args.bundle_dir, args.robot, tid)
        ]
        if no_sim:
            raise FileNotFoundError(
                "These requested trajectories do not have reservoir simulations: "
                f"{no_sim}"
            )
        return sorted(list(dict.fromkeys(requested)), key=_natural_key)

    usable = [
        t["id"] for t in trajs
        if _has_simulation(args.bundle_dir, args.robot, t["id"])
    ]
    if not usable:
        raise RuntimeError(
            f"No usable trajectories for {args.robot}; no simulation.h5 files found."
        )
    return sorted(usable, key=_natural_key)


def _metrics_target_present(metrics_path: Path, target: str) -> bool:
    if not metrics_path.exists():
        return False
    with metrics_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if (
                row.get("target") == target
                and row.get("scope") == "overall"
                and row.get("horizon") == "1"
            ):
                return True
    return False


def _run_complete(run_dir: Path, target: str) -> bool:
    return (
        (run_dir / "summary.json").exists()
        and (run_dir / "concat_split_config.json").exists()
        and _metrics_target_present(run_dir / "metrics.csv", target)
    )


def _call_concat_split(spec: TrainRunSpec, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "-m",
        "openprc.automod._tools.training.train_readout_concat_split",
        "--bundle-dir",
        str(args.bundle_dir),
        "--robot",
        args.robot,
        "--features",
        spec.feature,
        "--targets",
        spec.target,
        "--trajectories",
        ",".join(spec.trajectories),
        "--target-fps",
        str(args.target_fps),
        "--skip-seconds",
        str(args.skip_seconds),
        "--concat-train-fraction",
        str(args.concat_train_fraction),
        "--cv-mode",
        args.cv_mode,
        "--run-id",
        spec.run_id,
    ]

    print(f"\n=== Training: {spec.run_id} ===")
    print(f"target:       {spec.target}")
    print(f"trajectories: {spec.trajectories}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def _train_needed_runs(args: argparse.Namespace, specs: Iterable[TrainRunSpec]) -> None:
    if args.no_train:
        print("\nTraining disabled by --no-train; reading existing outputs.")
        return

    for spec in specs:
        out = _run_dir(args.bundle_dir, args.robot, spec.run_id)
        if _run_complete(out, spec.target) and not args.force_train:
            print(f"skip existing training run: {spec.run_id}")
            continue
        _call_concat_split(spec, args)


def _read_target_metrics(run_dir: Path, target: str) -> Dict[str, float]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics.csv for run: {run_dir}")

    out = {"r2": float("nan"), "mse": float("nan"), "lambda": float("nan")}
    with metrics_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if (
                row.get("target") == target
                and row.get("scope") == "overall"
                and row.get("horizon") == "1"
            ):
                for key in out:
                    try:
                        out[key] = float(row[key])
                    except (KeyError, ValueError):
                        out[key] = float("nan")
                return out
    return out


def _build_specs(
    features: Sequence[str],
    targets: Sequence[str],
    trajectories: Sequence[str],
) -> List[TrainRunSpec]:
    specs = []
    for feature in features:
        for target in targets:
            for k in range(1, len(trajectories) + 1):
                specs.append(
                    TrainRunSpec(
                        run_id=(
                            f"seq_concat_{_safe_token(target)}_"
                            f"{_safe_token(feature)}_top_{k}"
                        ),
                        feature=feature,
                        target=target,
                        trajectories=list(trajectories[:k]),
                    )
                )
    return specs


def _collect_rows(args: argparse.Namespace, specs: Sequence[TrainRunSpec]) -> List[Dict[str, object]]:
    rows = []
    for spec in specs:
        run_dir = _run_dir(args.bundle_dir, args.robot, spec.run_id)
        metrics = _read_target_metrics(run_dir, spec.target)
        rows.append({
            "feature": spec.feature,
            "target": spec.target,
            "k": len(spec.trajectories),
            "trajectory_added": spec.trajectories[-1],
            "included_trajectories": ",".join(spec.trajectories),
            "run_id": spec.run_id,
            "lambda": metrics["lambda"],
            "mse": metrics["mse"],
            "r2": metrics["r2"],
        })
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "feature",
        "target",
        "k",
        "trajectory_added",
        "included_trajectories",
        "run_id",
        "lambda",
        "mse",
        "r2",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _setup_matplotlib():
    """
    Configure matplotlib for publication-quality figures matching the visual
    standards of top ML venues (NeurIPS, ICLR, ICML).

    Design choices:
    - Serif font (DejaVu Serif) — matches LaTeX Computer Modern feel without
      requiring a full TeX installation.
    - 9 pt base font, 8 pt ticks/legend — correct for single-column figures
      that will be ~3.5 in wide in a two-column paper.
    - Top/right spines removed — the single most impactful cleanliness change.
    - Inward ticks, no minor ticks — standard in NeurIPS/ICLR submissions.
    - 300 DPI save — meets most camera-ready requirements.
    - No background patch — pure white, nothing to fight with LaTeX.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from e

    plt.rcParams.update({
        # --- Resolution ---
        "figure.dpi":           150,
        "savefig.dpi":          300,

        # --- Typography ---
        "font.family":          "serif",
        "font.serif":           ["DejaVu Serif", "Times New Roman", "serif"],
        "mathtext.fontset":     "dejavuserif",
        "font.size":            9,
        "axes.titlesize":       10,
        "axes.labelsize":       9,
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "legend.fontsize":      8,
        "legend.title_fontsize": 8,

        # --- Colour / text ---
        "text.color":           COLORS["ink"],
        "axes.labelcolor":      COLORS["ink"],
        "xtick.color":          COLORS["ink"],
        "ytick.color":          COLORS["ink"],

        # --- Axes spines ---
        "axes.edgecolor":       COLORS["ink"],
        "axes.linewidth":       0.8,
        "axes.spines.top":      False,   # remove top spine
        "axes.spines.right":    False,   # remove right spine

        # --- Ticks ---
        "xtick.direction":      "in",
        "ytick.direction":      "in",
        "xtick.major.size":     3.5,
        "ytick.major.size":     3.5,
        "xtick.major.width":    0.8,
        "ytick.major.width":    0.8,
        "xtick.minor.visible":  False,
        "ytick.minor.visible":  False,
        "xtick.major.pad":      3,
        "ytick.major.pad":      3,

        # --- Grid ---
        "axes.grid":            True,
        "axes.grid.axis":       "y",           # horizontal rules only
        "grid.color":           COLORS["grid"],
        "grid.linewidth":       0.6,
        "grid.linestyle":       "--",
        "grid.alpha":           0.7,

        # --- Lines & markers ---
        "lines.linewidth":      1.5,
        "lines.markersize":     5,

        # --- Legend ---
        "legend.frameon":       False,
        "legend.borderpad":     0.4,
        "legend.labelspacing":  0.3,

        # --- Layout ---
        "axes.titlepad":        6,
        "axes.labelpad":        4,
        "figure.constrained_layout.use": True,

        # --- Background ---
        "axes.facecolor":       "white",
        "figure.facecolor":     "white",

        # --- PDF/SVG font embedding (avoids Type 3 fonts in camera-ready) ---
        "pdf.fonttype":         42,
        "ps.fonttype":          42,
    })
    return plt


def _savefig(fig, out_path: Path, also_pdf: bool) -> None:
    fig.savefig(out_path, bbox_inches="tight")
    if also_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")


def _finite_values(values: Sequence[object]) -> List[float]:
    out = []
    for value in values:
        v = float(value)
        out.append(v if math.isfinite(v) else float("nan"))
    return out


def _plot_curves(
    rows: Sequence[Dict[str, object]],
    features: Sequence[str],
    targets: Sequence[str],
    plot_dir: Path,
    also_pdf: bool,
) -> None:
    plt = _setup_matplotlib()
    for feature in features:
        for target in targets:
            subset = [
                r for r in rows
                if r["feature"] == feature and r["target"] == target
            ]
            subset = sorted(subset, key=lambda r: int(r["k"]))
            if not subset:
                continue

            ks = [int(r["k"]) for r in subset]
            r2s = _finite_values([r["r2"] for r in subset])
            added = [str(r["trajectory_added"]) for r in subset]

            # ----------------------------------------------------------------
            # Figure sizing: ~5 in wide (single column + a little breathing
            # room), 3.2 in tall — a comfortable 16:10 ratio for a line plot.
            # Grows slightly for many clips so tick labels don't crowd.
            # ----------------------------------------------------------------
            width = max(5.0, 0.38 * len(subset) + 2.4)
            fig, ax = plt.subplots(figsize=(width, 3.2))

            # Main curve — Tol muted blue, filled circle markers with white edge
            ax.plot(
                ks,
                r2s,
                color=COLORS["line"],
                marker="o",
                linewidth=1.5,
                markersize=5,
                markeredgecolor=COLORS["marker_ec"],
                markeredgewidth=0.8,
                zorder=3,
            )

            # y = 0 reference line — thin, neutral grey, behind the data
            ax.axhline(
                0.0,
                color=COLORS["zero"],
                linewidth=0.8,
                linestyle="--",
                zorder=1,
            )

            ax.set_xticks(ks)
            ax.set_xticklabels(
                [f"{k}\n{clip}" for k, clip in zip(ks, added)],
                rotation=45,
                ha="right",
            )
            ax.set_xlabel("Number of included sequential clips")
            ax.set_ylabel("Test $R^2$")
            ax.set_title(f"Sequential Concat-Split: {target} / {feature}")

            # Value annotations — offset slightly above/below the marker
            for k, value in zip(ks, r2s):
                if math.isfinite(value):
                    va = "bottom" if value >= 0 else "top"
                    y_offset = 0.012 * (ax.get_ylim()[1] - ax.get_ylim()[0]) if ax.get_ylim()[1] != ax.get_ylim()[0] else 0.01
                    ax.annotate(
                        f"{value:.3f}",
                        xy=(k, value),
                        xytext=(0, 5 if value >= 0 else -5),
                        textcoords="offset points",
                        ha="center",
                        va=va,
                        fontsize=7,
                        color=COLORS["annot"],
                    )

            _savefig(
                fig,
                plot_dir / f"sequential_concat_curve_{_safe_token(target)}_{_safe_token(feature)}.png",
                also_pdf,
            )
            plt.close(fig)


def _write_summary(
    path: Path,
    args: argparse.Namespace,
    features: Sequence[str],
    targets: Sequence[str],
    trajectories: Sequence[str],
    rows: Sequence[Dict[str, object]],
) -> None:
    final_by_pair = {}
    for feature in features:
        for target in targets:
            key = f"{target}/{feature}"
            subset = [
                r for r in rows
                if r["feature"] == feature and r["target"] == target
            ]
            subset = sorted(subset, key=lambda r: int(r["k"]))
            final_by_pair[key] = subset[-1] if subset else None

    payload = {
        "robot": args.robot,
        "features": list(features),
        "targets": list(targets),
        "trajectory_order": list(trajectories),
        "split_mode": "concat",
        "concat_train_fraction": args.concat_train_fraction,
        "target_fps": args.target_fps,
        "skip_seconds": args.skip_seconds,
        "final_by_target_feature": final_by_pair,
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sequential cumulative training using concat-then-split runs."
    )
    p.add_argument("--bundle-dir", required=True, type=Path)
    p.add_argument("--robot", default="go1")
    p.add_argument("--features", default=FEATURES_DEFAULT)
    p.add_argument("--targets", default=TARGETS_DEFAULT)
    p.add_argument("--trajectories", default=None,
                   help="Comma-separated trajectory IDs. Default: all usable "
                        "manifest trajectories sorted naturally.")
    p.add_argument("--target-fps", type=float, default=50.0)
    p.add_argument("--concat-train-fraction", type=float, default=0.7)
    p.add_argument("--cv-mode", choices=["grouped", "temporal"], default="temporal")
    p.add_argument("--skip-seconds", type=float, default=2.0)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--no-train", action="store_true",
                   help="Do not launch training; read existing runs only.")
    p.add_argument("--force-train", action="store_true",
                   help="Re-run training even if metrics.csv already exists.")
    p.add_argument("--pdf", action="store_true",
                   help="Also save vector PDF versions of each plot.")
    args = p.parse_args()

    args.bundle_dir = args.bundle_dir.resolve()
    features = _csv_list(args.features)
    targets = _csv_list(args.targets)
    trajectories = _resolve_trajectories(args)

    out_dir = args.output_dir or (
        args.bundle_dir / args.robot / "training" / "sequential_clip_concat_split"
    )
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("=== Sequential concat-split training study ===")
    print(f"robot:                 {args.robot}")
    print(f"features:              {features}")
    print(f"targets:               {targets}")
    print(f"trajectory order:      {trajectories}")
    print(f"concat_train_fraction: {args.concat_train_fraction}")
    print(f"output:                {out_dir}")

    specs = _build_specs(features, targets, trajectories)
    _train_needed_runs(args, specs)
    rows = _collect_rows(args, specs)

    csv_path = out_dir / "sequential_concat_learning_curve.csv"
    summary_path = out_dir / "sequential_concat_summary.json"
    _write_csv(csv_path, rows)
    _write_summary(summary_path, args, features, targets, trajectories, rows)
    _plot_curves(rows, features, targets, plot_dir, args.pdf)

    print("\n=== Sequential concat-split training report complete ===")
    print(f"curve csv: {csv_path}")
    print(f"summary:   {summary_path}")
    print(f"plots:     {plot_dir}")


if __name__ == "__main__":
    main()