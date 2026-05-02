"""
Data-efficient PRC readout training study.

This script asks:

  1. Which single trajectory gives the best readout R2?
  2. If trajectories are added from best to worst, does performance improve
     and saturate as training data grows?

It does not modify train_readout.py. Instead, it creates temporary filtered
bundle manifests and launches train_readout.py as a subprocess.

Run from the repo's src directory:

    python3 -m openprc.automod._tools.training.data_efficient_training \\
      --bundle-dir openprc/automod/robot_bundle
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


TARGETS_DEFAULT = "base_lin_vel,base_ang_vel,qvel"
FEATURES_DEFAULT = "strain,strain_rate,node_vel"

# ---------------------------------------------------------------------------
# Publication-quality color palette.
# Line chart: VT Maroon (#861F41) — consistent with the other training scripts.
# Bar chart:  VT Orange (#E5751F) — perceptually distinct, warm complement.
# Supporting neutrals follow the same Nord-inspired system used across the
# other training scripts.
# ---------------------------------------------------------------------------
COLORS = {
    "ink":       "#2E3440",   # near-black for text / spines / ticks
    "muted":     "#5B677A",   # secondary text / de-emphasised elements
    "grid":      "#D8DEE9",   # light cool grey for gridlines
    "bar":       "#E5751F",   # VT Orange — single-trajectory bar chart
    "line":      "#861F41",   # VT Maroon — cumulative learning curve
    "marker_ec": "#FFFFFF",   # white marker edge for visual pop
    "zero":      "#888888",   # neutral mid-grey for y=0 reference line
    "annot":     "#2E3440",   # annotation text (same as ink)
}


@dataclass(frozen=True)
class TrainRunSpec:
    run_id: str
    feature: str
    trajectories: List[str]
    targets: List[str]


def _csv_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _safe_token(s: str) -> str:
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


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
        return list(dict.fromkeys(requested))

    usable = [
        t["id"] for t in trajs
        if _has_simulation(args.bundle_dir, args.robot, t["id"])
    ]
    if not usable:
        raise RuntimeError(
            f"No usable trajectories for {args.robot}; no simulation.h5 files found."
        )
    return usable


def _metrics_targets_present(metrics_path: Path, targets: Sequence[str]) -> bool:
    if not metrics_path.exists():
        return False
    found = set()
    with metrics_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("scope") == "overall" and row.get("horizon") == "1":
                found.add(row.get("target", ""))
    return set(targets).issubset(found)


def _run_complete(run_dir: Path, targets: Sequence[str]) -> bool:
    return (
        (run_dir / "summary.json").exists()
        and _metrics_targets_present(run_dir / "metrics.csv", targets)
    )


def _filtered_bundle_for_run(spec: TrainRunSpec, args: argparse.Namespace) -> Path:
    manifest = _load_manifest(args.bundle_dir)
    robot_entry = manifest["robots"][args.robot]
    by_id = {t["id"]: t for t in robot_entry["trajectories"]}
    missing = [tid for tid in spec.trajectories if tid not in by_id]
    if missing:
        raise KeyError(
            f"Requested trajectory id(s) not in manifest for {args.robot}: {missing}"
        )

    robot_entry["trajectories"] = [by_id[tid] for tid in spec.trajectories]

    bundle_dir = args.filtered_bundle_root / spec.run_id
    if bundle_dir.exists() and args.force_train:
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    robot_link = bundle_dir / args.robot
    if not robot_link.exists():
        robot_link.symlink_to(args.bundle_dir / args.robot, target_is_directory=True)

    with (bundle_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    return bundle_dir


def _call_train_readout(spec: TrainRunSpec, args: argparse.Namespace) -> None:
    filtered_bundle = _filtered_bundle_for_run(spec, args)
    cmd = [
        sys.executable,
        "-m",
        "openprc.automod._tools.training.train_readout",
        "--bundle-dir",
        str(filtered_bundle),
        "--robot",
        args.robot,
        "--features",
        spec.feature,
        "--targets",
        ",".join(spec.targets),
        "--split-mode",
        args.split_mode,
        "--target-fps",
        str(args.target_fps),
        "--skip-seconds",
        str(args.skip_seconds),
        "--run-id",
        spec.run_id,
    ]

    print(f"\n=== Training: {spec.run_id} ===")
    print(f"targets:      {spec.targets}")
    print(f"trajectories: {spec.trajectories}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def _train_needed_runs(
    args: argparse.Namespace,
    specs: Iterable[TrainRunSpec],
    targets: Sequence[str],
) -> None:
    if args.no_train:
        print("\nTraining disabled by --no-train; reading existing outputs.")
        return

    for spec in specs:
        out = _run_dir(args.bundle_dir, args.robot, spec.run_id)
        if _run_complete(out, spec.targets) and not args.force_train:
            print(f"skip existing training run: {spec.run_id}")
            continue
        _call_train_readout(spec, args)


def _read_r2_metrics(run_dir: Path, targets: Sequence[str]) -> Dict[str, float]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics.csv for run: {run_dir}")

    out = {target: float("nan") for target in targets}
    with metrics_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("scope") != "overall" or row.get("horizon") != "1":
                continue
            target = row.get("target", "")
            if target in out:
                try:
                    out[target] = float(row["r2"])
                except ValueError:
                    out[target] = float("nan")
    return out


def _single_specs(
    features: Sequence[str],
    trajectories: Sequence[str],
    targets: Sequence[str],
) -> List[TrainRunSpec]:
    specs = []
    for feature in features:
        for clip_id in trajectories:
            specs.append(
                TrainRunSpec(
                    run_id=f"de_single_{_safe_token(clip_id)}_{_safe_token(feature)}",
                    feature=feature,
                    trajectories=[clip_id],
                    targets=list(targets),
                )
            )
    return specs


def _collect_single_rows(
    args: argparse.Namespace,
    specs: Sequence[TrainRunSpec],
    targets: Sequence[str],
) -> List[Dict[str, object]]:
    rows = []
    for spec in specs:
        run_dir = _run_dir(args.bundle_dir, args.robot, spec.run_id)
        r2_by_target = _read_r2_metrics(run_dir, targets)
        clip_id = spec.trajectories[0]
        for target, r2 in r2_by_target.items():
            rows.append({
                "feature": spec.feature,
                "target": target,
                "trajectory": clip_id,
                "run_id": spec.run_id,
                "r2": r2,
            })

    for feature in sorted({str(r["feature"]) for r in rows}):
        for target in sorted({str(r["target"]) for r in rows}):
            subset = [
                r for r in rows
                if r["feature"] == feature and r["target"] == target
            ]
            subset.sort(
                key=lambda r: (
                    -float(r["r2"]) if math.isfinite(float(r["r2"])) else float("inf"),
                    str(r["trajectory"]),
                )
            )
            for rank, row in enumerate(subset, start=1):
                row["rank"] = rank

    return rows


def _ranking_for(
    single_rows: Sequence[Dict[str, object]],
    feature: str,
    target: str,
) -> List[Dict[str, object]]:
    rows = [
        r for r in single_rows
        if r["feature"] == feature and r["target"] == target
    ]
    return sorted(
        rows,
        key=lambda r: (
            int(r.get("rank", 10**9)),
            str(r["trajectory"]),
        ),
    )


def _cumulative_specs(
    single_rows: Sequence[Dict[str, object]],
    features: Sequence[str],
    targets: Sequence[str],
) -> List[TrainRunSpec]:
    specs = []
    for feature in features:
        for target in targets:
            ranked = _ranking_for(single_rows, feature, target)
            ordered = [str(r["trajectory"]) for r in ranked]
            for k in range(1, len(ordered) + 1):
                specs.append(
                    TrainRunSpec(
                        run_id=(
                            f"de_cum_{_safe_token(target)}_"
                            f"{_safe_token(feature)}_top_{k}"
                        ),
                        feature=feature,
                        trajectories=ordered[:k],
                        targets=[target],
                    )
                )
    return specs


def _collect_cumulative_rows(
    args: argparse.Namespace,
    specs: Sequence[TrainRunSpec],
    targets: Sequence[str],
) -> List[Dict[str, object]]:
    rows = []
    for spec in specs:
        parts = spec.run_id.split("_top_")
        k = int(parts[-1])
        prefix = parts[0]
        target_for_order = None
        for target in targets:
            token = f"de_cum_{_safe_token(target)}_"
            if prefix.startswith(token):
                target_for_order = target
                break
        if target_for_order is None:
            raise RuntimeError(f"could not infer target from run id {spec.run_id}")

        run_dir = _run_dir(args.bundle_dir, args.robot, spec.run_id)
        r2_by_target = _read_r2_metrics(run_dir, targets)
        rows.append({
            "feature": spec.feature,
            "target": target_for_order,
            "k": k,
            "trajectory_added": spec.trajectories[-1],
            "included_trajectories": ",".join(spec.trajectories),
            "run_id": spec.run_id,
            "r2": r2_by_target[target_for_order],
        })
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
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


def _finite_r2(values: Sequence[object]) -> List[float]:
    out = []
    for value in values:
        v = float(value)
        out.append(v if math.isfinite(v) else float("nan"))
    return out


def _plot_rankings(
    single_rows: Sequence[Dict[str, object]],
    features: Sequence[str],
    targets: Sequence[str],
    plot_dir: Path,
    also_pdf: bool,
) -> None:
    plt = _setup_matplotlib()
    for feature in features:
        for target in targets:
            rows = _ranking_for(single_rows, feature, target)
            if not rows:
                continue
            labels = [str(r["trajectory"]) for r in rows]
            values = _finite_r2([r["r2"] for r in rows])
            x = list(range(len(rows)))

            # ----------------------------------------------------------------
            # Figure sizing: ~5 in wide, 3.2 in tall; grows for many clips.
            # ----------------------------------------------------------------
            width = max(5.0, 0.38 * len(rows) + 2.4)
            fig, ax = plt.subplots(figsize=(width, 3.2))

            # Bars — VT Orange, slight transparency so grid shows through
            ax.bar(
                x,
                values,
                color=COLORS["bar"],
                alpha=0.85,
                width=0.6,
                zorder=3,
            )

            # y = 0 reference line
            ax.axhline(
                0.0,
                color=COLORS["zero"],
                linewidth=0.8,
                linestyle="--",
                zorder=1,
            )

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("Test $R^2$")
            ax.set_xlabel("Trajectory ranked by single-trajectory $R^2$")
            ax.set_title(f"Single-Trajectory Ranking: {target} / {feature}")

            # Value annotations
            for xi, value in zip(x, values):
                if math.isfinite(value):
                    va = "bottom" if value >= 0 else "top"
                    ax.annotate(
                        f"{value:.3f}",
                        xy=(xi, value),
                        xytext=(0, 4 if value >= 0 else -4),
                        textcoords="offset points",
                        ha="center",
                        va=va,
                        fontsize=7,
                        color=COLORS["annot"],
                    )

            _savefig(
                fig,
                plot_dir / f"ranking_{_safe_token(target)}_{_safe_token(feature)}.png",
                also_pdf,
            )
            plt.close(fig)


def _plot_learning_curves(
    cumulative_rows: Sequence[Dict[str, object]],
    features: Sequence[str],
    targets: Sequence[str],
    plot_dir: Path,
    also_pdf: bool,
) -> None:
    plt = _setup_matplotlib()
    for feature in features:
        for target in targets:
            rows = [
                r for r in cumulative_rows
                if r["feature"] == feature and r["target"] == target
            ]
            rows = sorted(rows, key=lambda r: int(r["k"]))
            if not rows:
                continue
            ks = [int(r["k"]) for r in rows]
            values = _finite_r2([r["r2"] for r in rows])
            added = [str(r["trajectory_added"]) for r in rows]

            # ----------------------------------------------------------------
            # Figure sizing: ~5 in wide, 3.2 in tall; grows for many clips.
            # ----------------------------------------------------------------
            width = max(5.0, 0.38 * len(rows) + 2.4)
            fig, ax = plt.subplots(figsize=(width, 3.2))

            # Main curve — VT Maroon, filled circle markers with white edge
            ax.plot(
                ks,
                values,
                color=COLORS["line"],
                marker="o",
                linewidth=1.5,
                markersize=5,
                markeredgecolor=COLORS["marker_ec"],
                markeredgewidth=0.8,
                zorder=3,
            )

            # y = 0 reference line
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
            ax.set_xlabel("Number of included trajectories")
            ax.set_ylabel("Test $R^2$")
            ax.set_title(f"Data-Efficiency Curve: {target} / {feature}")

            # Value annotations
            for k, value in zip(ks, values):
                if math.isfinite(value):
                    va = "bottom" if value >= 0 else "top"
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
                plot_dir / f"learning_curve_{_safe_token(target)}_{_safe_token(feature)}.png",
                also_pdf,
            )
            plt.close(fig)


def _write_summary(
    path: Path,
    args: argparse.Namespace,
    features: Sequence[str],
    targets: Sequence[str],
    trajectories: Sequence[str],
    single_rows: Sequence[Dict[str, object]],
    cumulative_rows: Sequence[Dict[str, object]],
) -> None:
    best_single = {}
    final_cumulative = {}
    for feature in features:
        for target in targets:
            key = f"{target}/{feature}"
            ranked = _ranking_for(single_rows, feature, target)
            best_single[key] = ranked[0] if ranked else None
            curve = [
                r for r in cumulative_rows
                if r["feature"] == feature and r["target"] == target
            ]
            curve = sorted(curve, key=lambda r: int(r["k"]))
            final_cumulative[key] = curve[-1] if curve else None

    payload = {
        "robot": args.robot,
        "features": list(features),
        "targets": list(targets),
        "trajectories": list(trajectories),
        "target_fps": args.target_fps,
        "split_mode": args.split_mode,
        "skip_seconds": args.skip_seconds,
        "best_single_trajectory_by_target_feature": best_single,
        "final_cumulative_by_target_feature": final_cumulative,
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run data-efficient PRC readout training experiments."
    )
    p.add_argument("--bundle-dir", required=True, type=Path)
    p.add_argument("--robot", default="go1")
    p.add_argument("--features", default=FEATURES_DEFAULT)
    p.add_argument("--targets", default=TARGETS_DEFAULT)
    p.add_argument("--trajectories", default=None,
                   help="Comma-separated trajectory IDs. Default: all usable "
                        "manifest trajectories with simulation.h5.")
    p.add_argument("--target-fps", type=float, default=50.0)
    p.add_argument("--split-mode", choices=["in-clip", "multi-clip"], default="in-clip")
    p.add_argument("--skip-seconds", type=float, default=2.0)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--no-train", action="store_true",
                   help="Do not launch train_readout; read existing runs only.")
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
        args.bundle_dir / args.robot / "training" / "data_efficient_training"
    )
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    args.filtered_bundle_root = out_dir / "_filtered_bundles"

    print("=== Data-efficient training study ===")
    print(f"robot:        {args.robot}")
    print(f"features:     {features}")
    print(f"targets:      {targets}")
    print(f"trajectories: {trajectories}")
    print(f"output:       {out_dir}")

    single_specs = _single_specs(features, trajectories, targets)
    _train_needed_runs(args, single_specs, targets)
    single_rows = _collect_single_rows(args, single_specs, targets)

    cumulative_specs = _cumulative_specs(single_rows, features, targets)
    _train_needed_runs(args, cumulative_specs, targets)
    cumulative_rows = _collect_cumulative_rows(args, cumulative_specs, targets)

    ranking_csv = out_dir / "single_trajectory_ranking.csv"
    curve_csv = out_dir / "cumulative_learning_curve.csv"
    summary_json = out_dir / "data_efficient_summary.json"
    _write_csv(
        ranking_csv,
        single_rows,
        ["feature", "target", "trajectory", "rank", "run_id", "r2"],
    )
    _write_csv(
        curve_csv,
        cumulative_rows,
        [
            "feature",
            "target",
            "k",
            "trajectory_added",
            "included_trajectories",
            "run_id",
            "r2",
        ],
    )
    _write_summary(
        summary_json,
        args,
        features,
        targets,
        trajectories,
        single_rows,
        cumulative_rows,
    )

    _plot_rankings(single_rows, features, targets, plot_dir, args.pdf)
    _plot_learning_curves(cumulative_rows, features, targets, plot_dir, args.pdf)

    print("\n=== Data-efficient training report complete ===")
    print(f"ranking csv: {ranking_csv}")
    print(f"curve csv:   {curve_csv}")
    print(f"summary:     {summary_json}")
    print(f"plots:       {plot_dir}")


if __name__ == "__main__":
    main()