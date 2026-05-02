"""
Autoprobing feasibility report for physical reservoir automodeling.

This script automates the first non-optimization autoprobing experiment:

  1. Feature ablation:
       strain vs strain_rate vs node_vel
  2. Probe clip comparison:
       selected trajectory clips crossed with every selected feature
  3. Reservoir diagnostics:
       effective rank, condition number, and linear correlation score
  4. Target-specific heatmaps:
       one heatmap per target-feature pair, without averaging target R2 scores

Run from the repo's src directory, for example:

    python3 -m openprc.automod._tools.training.autoprobe_feasibility \\
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

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_readout import FeatureSpec, _load_simulation_features  # noqa: E402


TARGETS_DEFAULT = "base_lin_vel,base_ang_vel,qvel"
FEATURES_DEFAULT = "strain,strain_rate,node_vel"
TRAJECTORIES_DEFAULT = "corridor_000,corridor_006,corridor_012"

COLORS = {
    "ink": "#17202A",
    "muted": "#5B677A",
    "grid": "#D8DEE9",
    "strain": "#3366CC",
    "strain_rate": "#D95F02",
    "node_vel": "#009E73",
    "base_lin_vel": "#4477AA",
    "base_ang_vel": "#CC6677",
    "qvel": "#228833",
    "effective_rank": "#AA3377",
    "condition_number": "#EE7733",
    "linear_correlation_score": "#0077BB",
}


@dataclass
class RunSpec:
    experiment: str
    run_id: str
    feature: str
    clip_id: str
    include_trajectories: Optional[List[str]]


def _csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _run_dir(bundle_dir: Path, robot: str, run_id: str) -> Path:
    return bundle_dir / robot / "training" / run_id


def _run_complete(run_dir: Path) -> bool:
    return (run_dir / "metrics.csv").exists() and (run_dir / "summary.json").exists()


def _filtered_bundle_for_run(spec: RunSpec, args: argparse.Namespace) -> Path:
    if not spec.include_trajectories:
        return args.bundle_dir

    original_manifest_path = args.bundle_dir / "manifest.json"
    with original_manifest_path.open() as f:
        manifest = json.load(f)

    robot_entry = manifest["robots"][args.robot]
    by_id = {t["id"]: t for t in robot_entry["trajectories"]}
    missing = [tid for tid in spec.include_trajectories if tid not in by_id]
    if missing:
        raise KeyError(
            f"Requested trajectory id(s) not in manifest for {args.robot}: "
            f"{missing}. Available examples: {list(by_id)[:10]}"
        )
    robot_entry["trajectories"] = [by_id[tid] for tid in spec.include_trajectories]

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


def _call_train_readout(
    spec: RunSpec,
    args: argparse.Namespace,
) -> None:
    train_bundle_dir = _filtered_bundle_for_run(spec, args)
    cmd = [
        sys.executable,
        "-m",
        "openprc.automod._tools.training.train_readout",
        "--bundle-dir",
        str(train_bundle_dir),
        "--robot",
        args.robot,
        "--features",
        spec.feature,
        "--targets",
        args.targets,
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
    if spec.include_trajectories:
        print(f"filtered trajectories: {spec.include_trajectories}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def _train_needed_runs(args: argparse.Namespace, specs: Sequence[RunSpec]) -> None:
    if args.no_train:
        print("\nTraining disabled by --no-train; reading existing run outputs.")
        return

    for spec in specs:
        out = _run_dir(args.bundle_dir, args.robot, spec.run_id)
        if _run_complete(out) and not args.force_train:
            print(f"skip existing training run: {spec.run_id}")
            continue
        _call_train_readout(spec, args)


def _read_r2_metrics(run_dir: Path, targets: Sequence[str]) -> Dict[str, float]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics.csv for run: {run_dir}")

    r2 = {f"{target}_R2": float("nan") for target in targets}
    with metrics_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("scope") != "overall" or row.get("horizon") != "1":
                continue
            target = row.get("target", "")
            if target in targets:
                try:
                    r2[f"{target}_R2"] = float(row["r2"])
                except ValueError:
                    r2[f"{target}_R2"] = float("nan")

    return r2


def _simulation_paths(bundle_dir: Path, robot: str, clip_id: str) -> tuple[Path, Path]:
    sim_root = bundle_dir / robot / "reservoir_sims" / clip_id
    return sim_root / "output" / "simulation.h5", sim_root / "input" / "geometry.h5"


def _load_feature_matrix(
    bundle_dir: Path,
    robot: str,
    clip_id: str,
    feature: str,
    target_fps: float,
    skip_seconds: float,
) -> np.ndarray:
    import h5py

    sim_path, geom_path = _simulation_paths(bundle_dir, robot, clip_id)
    if not sim_path.exists():
        raise FileNotFoundError(f"missing simulation.h5 for {clip_id}: {sim_path}")
    with h5py.File(sim_path, "r") as f:
        frame_rate = float(f.attrs.get("frame_rate", 50.0))

    dt = 1.0 / max(frame_rate, 1e-9)
    X = _load_simulation_features(
        str(sim_path),
        str(geom_path),
        FeatureSpec(level=feature),
        dt=dt,
    )
    effective_fps = frame_rate
    if target_fps and target_fps < frame_rate:
        stride = max(1, int(round(frame_rate / target_fps)))
        X = X[::stride]
        effective_fps = frame_rate / stride

    skip_frames = int(round(skip_seconds * effective_fps))
    if skip_frames >= X.shape[0] - 20:
        raise RuntimeError(
            f"{clip_id}: skip_seconds={skip_seconds} leaves too few frames "
            f"for diagnostics (T={X.shape[0]}, skip={skip_frames})"
        )
    return np.asarray(X[skip_frames:], dtype=np.float64)


def _choose_columns(X: np.ndarray, max_channels: int, seed: int) -> np.ndarray:
    norms = np.linalg.norm(X, axis=0)
    usable = np.flatnonzero(norms > 1e-12)
    if usable.size == 0:
        raise RuntimeError("all reservoir feature channels are numerically zero")
    if usable.size <= max_channels:
        return usable
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(usable, size=max_channels, replace=False))


def _rank_condition_all_channels(X: np.ndarray, chunk_size: int = 2048) -> Dict[str, float]:
    norms = np.linalg.norm(X, axis=0)
    usable = np.flatnonzero(norms > 1e-12)
    if usable.size == 0:
        raise RuntimeError("all reservoir feature channels are numerically zero")

    gram = np.zeros((X.shape[0], X.shape[0]), dtype=np.float64)
    for start in range(0, usable.size, chunk_size):
        cols = usable[start:start + chunk_size]
        block = X[:, cols] / norms[cols]
        gram += block @ block.T

    eigvals = np.linalg.eigvalsh(gram)
    max_eig = float(np.max(eigvals)) if eigvals.size else 0.0
    eigvals = eigvals[eigvals > max(max_eig * 1e-12, 1e-14)]
    if eigvals.size == 0:
        return {
            "effective_rank": 0.0,
            "condition_number": float("inf"),
        }

    probs = eigvals / eigvals.sum()
    return {
        "effective_rank": float(np.exp(-np.sum(probs * np.log(probs)))),
        "condition_number": float(eigvals.max() / eigvals.min()),
    }


def _diagnostics_for_matrix(
    X: np.ndarray,
    max_lcs_channels: int,
    seed: int,
) -> Dict[str, float]:
    """
    Compute the requested autoprobing diagnostics.

    Each reservoir channel is treated as one sensor time-series. Following the
    attached method, channels are normalized by their own time-series norm and
    stacked as rows of Delta X. The linear correlation score is the mean of
    column-wise sums of abs(Delta X Delta X^T), including the diagonal term.
    """
    rank_cond = _rank_condition_all_channels(X)

    cols = _choose_columns(X, max_lcs_channels, seed)
    Xs = X[:, cols]
    norms = np.linalg.norm(Xs, axis=0)
    keep = norms > 1e-12
    Xs = Xs[:, keep]
    norms = norms[keep]
    if Xs.shape[1] == 0:
        raise RuntimeError("no nonzero channels left after normalization")

    delta = (Xs / norms).T
    corr = delta @ delta.T
    linear_correlation_score = float(np.mean(np.sum(np.abs(corr), axis=0)))

    return {
        "effective_rank": rank_cond["effective_rank"],
        "condition_number": rank_cond["condition_number"],
        "linear_correlation_score": linear_correlation_score,
    }


def _mean_dict(dicts: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(dicts)
    if not rows:
        return {
            "effective_rank": float("nan"),
            "condition_number": float("nan"),
            "linear_correlation_score": float("nan"),
        }
    keys = rows[0].keys()
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}


def _manifest_clip_ids(bundle_dir: Path, robot: str) -> List[str]:
    with (bundle_dir / "manifest.json").open() as f:
        manifest = json.load(f)
    return [t["id"] for t in manifest["robots"][robot]["trajectories"]]


def _compute_diagnostics(
    bundle_dir: Path,
    robot: str,
    clips: Sequence[str],
    feature: str,
    target_fps: float,
    skip_seconds: float,
    max_lcs_channels: int,
    seed: int,
) -> Dict[str, float]:
    vals = []
    for i, clip_id in enumerate(clips):
        print(f"diagnostics: {feature} / {clip_id}")
        X = _load_feature_matrix(
            bundle_dir,
            robot,
            clip_id,
            feature,
            target_fps=target_fps,
            skip_seconds=skip_seconds,
        )
        vals.append(
            _diagnostics_for_matrix(
                X,
                max_lcs_channels=max_lcs_channels,
                seed=seed + i,
            )
        )
    return _mean_dict(vals)


def _build_run_specs(features: Sequence[str], clips: Sequence[str]) -> List[RunSpec]:
    specs: List[RunSpec] = []
    for feature in features:
        specs.append(
            RunSpec(
                experiment="feature_ablation",
                run_id=f"{feature}_ablation",
                feature=feature,
                clip_id="all",
                include_trajectories=None,
            )
        )
    for clip_id in clips:
        for feature in features:
            specs.append(
                RunSpec(
                    experiment="probe_clip",
                    run_id=f"probe_{clip_id}_{feature}",
                    feature=feature,
                    clip_id=clip_id,
                    include_trajectories=[clip_id],
                )
            )
    return specs


def _collect_report_rows(
    args: argparse.Namespace,
    specs: Sequence[RunSpec],
    targets: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    all_manifest_clips = _manifest_clip_ids(args.bundle_dir, args.robot)

    diagnostic_cache: Dict[tuple[str, str], Dict[str, float]] = {}

    for spec in specs:
        run_dir = _run_dir(args.bundle_dir, args.robot, spec.run_id)
        r2 = _read_r2_metrics(run_dir, targets)

        if spec.experiment == "feature_ablation":
            key = (spec.feature, "all")
            if key not in diagnostic_cache:
                diagnostic_cache[key] = _compute_diagnostics(
                    args.bundle_dir,
                    args.robot,
                    all_manifest_clips,
                    spec.feature,
                    args.target_fps,
                    args.skip_seconds,
                    args.max_lcs_channels,
                    args.random_seed,
                )
            diag = diagnostic_cache[key]
        else:
            key = (spec.feature, spec.clip_id)
            if key not in diagnostic_cache:
                diagnostic_cache[key] = _compute_diagnostics(
                    args.bundle_dir,
                    args.robot,
                    [spec.clip_id],
                    spec.feature,
                    args.target_fps,
                    args.skip_seconds,
                    args.max_lcs_channels,
                    args.random_seed,
                )
            diag = diagnostic_cache[key]

        row: Dict[str, object] = {
            "experiment": spec.experiment,
            "run_id": spec.run_id,
            "robot": args.robot,
            "clip_id": spec.clip_id,
            "feature": spec.feature,
            "effective_rank": diag["effective_rank"],
            "condition_number": diag["condition_number"],
            "linear_correlation_score": diag["linear_correlation_score"],
        }
        row.update(r2)
        rows.append(row)

    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], targets: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "run_id",
        "robot",
        "clip_id",
        "feature",
        "effective_rank",
        "condition_number",
        "linear_correlation_score",
        *[f"{target}_R2" for target in targets],
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float_column(rows: Sequence[Dict[str, object]], key: str) -> np.ndarray:
    return np.array([float(r.get(key, float("nan"))) for r in rows], dtype=float)


def _finite_fill(values: np.ndarray, fill: Optional[float] = None) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(out)
    if fill is None:
        fill = float(np.nanmax(out[finite])) if finite.any() else 0.0
    out[~finite] = fill
    return out


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 240,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "axes.edgecolor": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.8,
        "legend.frameon": False,
    })
    return plt


def _savefig(fig, out_path: Path, also_pdf: bool) -> None:
    fig.savefig(out_path, bbox_inches="tight")
    if also_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")


def _row_label(row: Dict[str, object]) -> str:
    if row["experiment"] == "feature_ablation":
        return "all clips"
    return str(row["clip_id"])


def _comfortable_blue_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "comfortable_blues",
        ["#F7FBFF", "#DEEBF7", "#C6DBEF", "#9ECAE1", "#6BAED6", "#4292C6"],
    )


def _plot_target_feature_heatmaps(
    rows: Sequence[Dict[str, object]],
    targets: Sequence[str],
    out_dir: Path,
    also_pdf: bool,
) -> None:
    plt = _setup_matplotlib()
    if not rows:
        return
    features = sorted({str(r["feature"]) for r in rows})
    metrics = [
        "effective_rank",
        "condition_number",
        "linear_correlation_score",
    ]
    cmap = _comfortable_blue_cmap()

    for target in targets:
        safe_target = target.replace("/", "_").replace(" ", "_")
        target_r2 = f"{target}_R2"
        cols = metrics + [target_r2]
        for feature in features:
            feature_rows = [r for r in rows if str(r["feature"]) == feature]
            if not feature_rows:
                continue
            sorted_rows = sorted(
                feature_rows,
                key=lambda r: (
                    0 if r["experiment"] == "feature_ablation" else 1,
                    str(r["clip_id"]),
                ),
            )
            data = np.array([[float(r[c]) for c in cols] for r in sorted_rows], dtype=float)
            normed = data.copy()
            for j in range(normed.shape[1]):
                col = _finite_fill(normed[:, j])
                finite = np.isfinite(col)
                if finite.any():
                    lo, hi = np.nanmin(col), np.nanmax(col)
                    normed[:, j] = (col - lo) / max(hi - lo, 1e-12)
                else:
                    normed[:, j] = 0.0

            height = max(4.2, 0.5 * len(sorted_rows) + 1.8)
            fig, ax = plt.subplots(figsize=(9.2, height), constrained_layout=True)
            im = ax.imshow(normed, cmap=cmap, aspect="auto", vmin=0.08, vmax=1.0)
            ax.set_yticks(np.arange(len(sorted_rows)))
            ax.set_yticklabels([_row_label(r) for r in sorted_rows])
            ax.set_xticks(np.arange(len(cols)))
            ax.set_xticklabels(
                ["effective rank", "condition number", "linear correlation", f"{target} R2"],
                rotation=24,
                ha="right",
            )
            ax.set_title(f"Autoprobing Heatmap: {target} / {feature}")
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    value = data[i, j]
                    label = f"{value:.2g}" if abs(value) >= 1000 else f"{value:.3f}"
                    ax.text(
                        j,
                        i,
                        label,
                        ha="center",
                        va="center",
                        color=COLORS["ink"],
                        fontsize=9,
                    )
            fig.colorbar(im, ax=ax, label="column-normalized value")
            safe_feature = feature.replace("/", "_").replace(" ", "_")
            _savefig(fig, out_dir / f"heatmap_{safe_target}_{safe_feature}.png", also_pdf)
            plt.close(fig)


def _write_summary(path: Path, args: argparse.Namespace, rows: Sequence[Dict[str, object]]) -> None:
    targets = _csv_list(args.targets)
    best_by_target = {}
    for target in targets:
        key = f"{target}_R2"
        candidates = [r for r in rows if math.isfinite(float(r.get(key, float("nan"))))]
        best_by_target[target] = (
            max(candidates, key=lambda r: float(r[key])) if candidates else None
        )
    payload = {
        "robot": args.robot,
        "targets": targets,
        "target_fps": args.target_fps,
        "split_mode": args.split_mode,
        "skip_seconds": args.skip_seconds,
        "max_lcs_channels": args.max_lcs_channels,
        "best_condition_by_target_R2": best_by_target,
        "rows": list(rows),
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run the autoprobing feasibility experiment and make report plots."
    )
    p.add_argument("--bundle-dir", required=True, type=Path)
    p.add_argument("--robot", default="go1")
    p.add_argument("--features", default=FEATURES_DEFAULT)
    p.add_argument("--diagnostic-feature", default=None, help=argparse.SUPPRESS)
    p.add_argument("--targets", default=TARGETS_DEFAULT)
    p.add_argument("--trajectories", default=TRAJECTORIES_DEFAULT)
    p.add_argument("--target-fps", type=float, default=50.0)
    p.add_argument("--split-mode", choices=["in-clip", "multi-clip"], default="in-clip")
    p.add_argument("--skip-seconds", type=float, default=2.0)
    p.add_argument("--max-lcs-channels", type=int, default=2048,
                   help="Maximum reservoir channels used for the O(N^2) "
                        "linear-correlation score. Effective rank and "
                        "conditioning use all nonzero channels.")
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--no-train", action="store_true",
                   help="Do not launch train_readout; only summarize existing runs.")
    p.add_argument("--force-train", action="store_true",
                   help="Re-run training even if metrics.csv already exists.")
    p.add_argument("--pdf", action="store_true",
                   help="Also save vector PDF versions of each plot.")
    args = p.parse_args()

    features = _csv_list(args.features)
    targets = _csv_list(args.targets)
    clips = _csv_list(args.trajectories)
    args.bundle_dir = args.bundle_dir.resolve()
    out_dir = args.output_dir or (
        args.bundle_dir / args.robot / "training" / "autoprobe_feasibility"
    )
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    args.filtered_bundle_root = out_dir / "_filtered_bundles"

    specs = _build_run_specs(features, clips)
    _train_needed_runs(args, specs)

    rows = _collect_report_rows(args, specs, targets)
    csv_path = out_dir / "autoprobe_metrics.csv"
    summary_path = out_dir / "autoprobe_summary.json"
    _write_csv(csv_path, rows, targets)
    _write_summary(summary_path, args, rows)

    _plot_target_feature_heatmaps(rows, targets, plot_dir, args.pdf)

    print("\n=== Autoprobing feasibility report complete ===")
    print(f"metrics: {csv_path}")
    print(f"summary: {summary_path}")
    print(f"plots:   {plot_dir}")


if __name__ == "__main__":
    main()
