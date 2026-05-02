"""
Sweep train/test split fractions for PRC readout training.

This script does not modify existing training scripts. By default, it uses a
fixed-test in-clip design: train on the first X% of each selected clip and
always test on the last 30% of each selected clip. Lower train fractions leave
the middle of each clip unused, which keeps the test set identical across the
sweep.

Use --split-method concat-fixed-test to apply the same fixed-test idea after
concatenating selected clips into one global state matrix.

Run from the repo's src directory:

    python3 -m openprc.automod._tools.training.split_fraction_sweep \\
      --bundle-dir openprc/automod/robot_bundle
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from train_readout import (  # noqa: E402
    DEFAULT_HORIZONS,
    DEFAULT_LAMBDAS,
    _list_target_specs,
    _parse_features,
    collect_data,
    save_predictions,
    save_readout,
    train_one_target,
    write_metrics_csv,
    write_summary_json,
)


TARGETS_DEFAULT = "base_lin_vel,base_ang_vel,qvel"
FEATURES_DEFAULT = "strain,strain_rate,node_vel"
FRACTIONS_DEFAULT = "0.1,0.2,0.3,0.4,0.5,0.6,0.7"
FIXED_TEST_FRACTION_DEFAULT = 0.3

# ---------------------------------------------------------------------------
# Publication-quality color palette — consistent with the other training
# scripts. VT Maroon is the primary data line across all scripts; the
# supporting neutrals follow the same Nord-inspired system.
# ---------------------------------------------------------------------------
COLORS = {
    "ink":       "#2E3440",   # near-black for text / spines / ticks
    "grid":      "#D8DEE9",   # light cool grey for gridlines
    "line":      "#861F41",   # VT Maroon — main data line
    "marker_ec": "#FFFFFF",   # white marker edge for visual pop
    "zero":      "#888888",   # neutral mid-grey for y=0 reference line
    "annot":     "#2E3440",   # annotation text (same as ink)
}


@dataclass(frozen=True)
class TrainRunSpec:
    run_id: str
    feature: str
    target: str
    train_fraction: float
    trajectories: List[str]
    split_method: str


def _csv_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _float_list(s: str) -> List[float]:
    values = [float(x.strip()) for x in s.split(",") if x.strip()]
    for value in values:
        if not 0.0 < value < 1.0:
            raise ValueError(f"train fractions must be in (0, 1), got {value}")
    return values


def _safe_token(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in s)


def _fraction_token(value: float) -> str:
    return f"{int(round(100.0 * value)):02d}pct"


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
        and _metrics_target_present(run_dir / "metrics.csv", target)
    )


def _parse_horizons(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_lambdas(s: str) -> List[float]:
    if s == "auto":
        return list(DEFAULT_LAMBDAS)
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _filtered_bundle(
    bundle_dir: Path,
    robot: str,
    trajectories: Sequence[str],
    out_dir: Path,
) -> Path:
    """Create a manifest-filtered bundle while symlinking robot data in place."""
    with (bundle_dir / "manifest.json").open() as f:
        manifest = json.load(f)

    robot_entry = manifest["robots"][robot]
    by_id = {t["id"]: t for t in robot_entry["trajectories"]}
    missing = [tid for tid in trajectories if tid not in by_id]
    if missing:
        raise KeyError(
            f"Requested trajectory id(s) not in manifest for {robot}: {missing}"
        )
    robot_entry["trajectories"] = [by_id[tid] for tid in trajectories]

    filtered_root = out_dir / "_filtered_bundle"
    if filtered_root.exists():
        shutil.rmtree(filtered_root)
    filtered_root.mkdir(parents=True, exist_ok=True)

    robot_link = filtered_root / robot
    robot_link.symlink_to(bundle_dir / robot, target_is_directory=True)

    with (filtered_root / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    return filtered_root


def _build_fixed_test_masks(
    group_ids: np.ndarray,
    train_fraction: float,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    if train_fraction + test_fraction > 1.0 + 1e-12:
        raise ValueError(
            "For fixed-test splitting, train_fraction + test_fraction must be "
            f"<= 1.0, got {train_fraction} + {test_fraction}"
        )

    train_mask = np.zeros(len(group_ids), dtype=bool)
    test_mask = np.zeros(len(group_ids), dtype=bool)

    for gid in np.unique(group_ids):
        idx = np.flatnonzero(group_ids == gid)
        n = len(idx)
        n_train = int(round(train_fraction * n))
        n_test = int(round(test_fraction * n))
        n_train = max(1, min(n - 1, n_train))
        n_test = max(1, min(n - n_train, n_test))
        train_mask[idx[:n_train]] = True
        test_mask[idx[-n_test:]] = True

    if np.any(train_mask & test_mask):
        raise RuntimeError("fixed-test split produced overlapping train/test rows")
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise RuntimeError(
            f"fixed-test split produced train={train_mask.sum()}, "
            f"test={test_mask.sum()} rows"
        )
    return train_mask, test_mask


def _build_concat_fixed_test_masks(
    n_rows: int,
    train_fraction: float,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    if train_fraction + test_fraction > 1.0 + 1e-12:
        raise ValueError(
            "For concat-fixed-test splitting, train_fraction + test_fraction "
            f"must be <= 1.0, got {train_fraction} + {test_fraction}"
        )
    if n_rows < 4:
        raise RuntimeError(f"need at least 4 rows for concat fixed-test, got {n_rows}")

    n_train = int(round(train_fraction * n_rows))
    n_test = int(round(test_fraction * n_rows))
    n_train = max(1, min(n_rows - 1, n_train))
    n_test = max(1, min(n_rows - n_train, n_test))

    train_mask = np.zeros(n_rows, dtype=bool)
    test_mask = np.zeros(n_rows, dtype=bool)
    train_mask[:n_train] = True
    test_mask[-n_test:] = True

    if np.any(train_mask & test_mask):
        raise RuntimeError("concat fixed-test split produced overlapping rows")
    return train_mask, test_mask


def _run_direct_training(
    spec: TrainRunSpec,
    args: argparse.Namespace,
    in_clip_bundle_dir: Path,
    split_kind: str,
) -> None:
    feat = _parse_features(spec.feature)
    target_specs = _list_target_specs([spec.target])
    lambdas = _parse_lambdas(args.lambdas)
    horizons = _parse_horizons(args.horizons)

    out_root = _run_dir(args.bundle_dir, args.robot, spec.run_id)
    pred_dir = out_root / "predictions"
    out_root.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    if split_kind == "fixed-test":
        split_label = "fixed-test"
        test_label = "last part of each clip"
    elif split_kind == "concat-fixed-test":
        split_label = "concat-fixed-test"
        test_label = "last part of concatenated matrix"
    else:
        raise ValueError(f"unknown direct split kind: {split_kind}")

    print(f"\n=== Training: {spec.run_id} ===")
    print(f"split method:   {split_label}")
    print(f"feature:        {spec.feature}")
    print(f"target:         {spec.target}")
    print(f"train fraction: {spec.train_fraction:.2f}")
    print(f"test fraction:  {args.fixed_test_fraction:.2f} ({test_label})")
    print(f"trajectories:   {spec.trajectories}")

    X, Ys, group_ids, _train_unused, _test_unused, train_clip_ids, test_clip_ids = collect_data(
        str(in_clip_bundle_dir),
        args.robot,
        feat,
        target_specs,
        split_mode="in-clip",
        in_clip_train_fraction=1.0 - args.fixed_test_fraction,
        skip_seconds=args.skip_seconds,
        target_fps=args.target_fps,
    )

    if split_kind == "fixed-test":
        train_mask, test_mask = _build_fixed_test_masks(
            group_ids,
            spec.train_fraction,
            args.fixed_test_fraction,
        )
    else:
        train_mask, test_mask = _build_concat_fixed_test_masks(
            len(X),
            spec.train_fraction,
            args.fixed_test_fraction,
        )

    X_tr = X[train_mask]
    X_te = X[test_mask]
    g_tr = group_ids[train_mask]
    g_te = group_ids[test_mask]
    Y_tr_dict = {k: v[train_mask] for k, v in Ys.items()}
    Y_te_dict = {k: v[test_mask] for k, v in Ys.items()}

    np.savez(
        pred_dir / "_split_meta.npz",
        test_group_ids=g_te.astype(np.int32),
        train_group_ids=g_tr.astype(np.int32),
        split_mode=np.array(split_label),
        train_fraction=np.float32(spec.train_fraction),
        fixed_test_fraction=np.float32(args.fixed_test_fraction),
    )

    results = []
    for ts in target_specs:
        result, W, b, Y_pred = train_one_target(
            ts.name,
            X_tr,
            Y_tr_dict[ts.name],
            g_tr,
            X_te,
            Y_te_dict[ts.name],
            lambdas=lambdas,
            n_folds=args.n_folds,
            horizons=horizons,
            cv_mode=args.cv_mode,
        )
        results.append(result)
        save_readout(str(out_root), ts.name, W, b, result, feat)
        save_predictions(str(pred_dir), ts.name, Y_te_dict[ts.name], Y_pred)

    write_metrics_csv(str(out_root / "metrics.csv"), results, horizons)
    write_summary_json(
        str(out_root / "summary.json"),
        args.robot,
        spec.run_id,
        feat,
        results,
        train_clip_ids,
        test_clip_ids,
    )

    with (out_root / "split_fraction_config.json").open("w") as f:
        json.dump({
            "split_method": split_label,
            "train_fraction": spec.train_fraction,
            "fixed_test_fraction": args.fixed_test_fraction,
            "trajectories": spec.trajectories,
            "source_bundle_dir": str(args.bundle_dir),
        }, f, indent=2)

    print(f"  results saved to {out_root}")


def _run_fixed_test_training(
    spec: TrainRunSpec,
    args: argparse.Namespace,
    in_clip_bundle_dir: Path,
) -> None:
    _run_direct_training(spec, args, in_clip_bundle_dir, "fixed-test")


def _call_training(
    spec: TrainRunSpec,
    args: argparse.Namespace,
    in_clip_bundle_dir: Path,
) -> None:
    if spec.split_method == "fixed-test":
        _run_fixed_test_training(spec, args, in_clip_bundle_dir)
        return
    if spec.split_method == "concat-fixed-test":
        _run_direct_training(spec, args, in_clip_bundle_dir, "concat-fixed-test")
        return
    if spec.split_method == "in-clip":
        cmd = [
            sys.executable,
            "-m",
            "openprc.automod._tools.training.train_readout",
            "--bundle-dir",
            str(in_clip_bundle_dir),
            "--robot",
            args.robot,
            "--features",
            spec.feature,
            "--targets",
            spec.target,
            "--target-fps",
            str(args.target_fps),
            "--skip-seconds",
            str(args.skip_seconds),
            "--split-mode",
            "in-clip",
            "--in-clip-train-fraction",
            str(spec.train_fraction),
            "--cv-mode",
            args.cv_mode,
            "--run-id",
            spec.run_id,
        ]
    elif spec.split_method == "concat":
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
            str(spec.train_fraction),
            "--cv-mode",
            args.cv_mode,
            "--run-id",
            spec.run_id,
        ]
    else:
        raise ValueError(f"unknown split method: {spec.split_method}")

    print(f"\n=== Training: {spec.run_id} ===")
    print(f"split method:   {spec.split_method}")
    print(f"feature:        {spec.feature}")
    print(f"target:         {spec.target}")
    print(f"train fraction: {spec.train_fraction:.2f}")
    print(f"trajectories:   {spec.trajectories}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def _train_needed_runs(
    args: argparse.Namespace,
    specs: Iterable[TrainRunSpec],
    in_clip_bundle_dir: Path,
) -> None:
    if args.no_train:
        print("\nTraining disabled by --no-train; reading existing outputs.")
        return

    for spec in specs:
        out = _run_dir(args.bundle_dir, args.robot, spec.run_id)
        if _run_complete(out, spec.target) and not args.force_train:
            print(f"skip existing training run: {spec.run_id}")
            continue
        _call_training(spec, args, in_clip_bundle_dir)


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
    fractions: Sequence[float],
    trajectories: Sequence[str],
    split_method: str,
) -> List[TrainRunSpec]:
    specs = []
    for feature in features:
        for target in targets:
            for fraction in fractions:
                specs.append(
                    TrainRunSpec(
                        run_id=(
                            f"splitfrac_{_safe_token(split_method)}_"
                            f"{_safe_token(target)}_"
                            f"{_safe_token(feature)}_{_fraction_token(fraction)}"
                        ),
                        feature=feature,
                        target=target,
                        train_fraction=fraction,
                        trajectories=list(trajectories),
                        split_method=split_method,
                    )
                )
    return specs


def _collect_rows(args: argparse.Namespace, specs: Sequence[TrainRunSpec]) -> List[Dict[str, object]]:
    rows = []
    for spec in specs:
        run_dir = _run_dir(args.bundle_dir, args.robot, spec.run_id)
        metrics = _read_target_metrics(run_dir, spec.target)
        rows.append({
            "split_method": spec.split_method,
            "feature": spec.feature,
            "target": spec.target,
            "train_fraction": spec.train_fraction,
            "test_fraction": (
                args.fixed_test_fraction
                if spec.split_method in ("fixed-test", "concat-fixed-test")
                else 1.0 - spec.train_fraction
            ),
            "unused_fraction": (
                max(0.0, 1.0 - args.fixed_test_fraction - spec.train_fraction)
                if spec.split_method in ("fixed-test", "concat-fixed-test")
                else 0.0
            ),
            "n_trajectories": len(spec.trajectories),
            "included_trajectories": ",".join(spec.trajectories),
            "run_id": spec.run_id,
            "lambda": metrics["lambda"],
            "mse": metrics["mse"],
            "r2": metrics["r2"],
        })
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "split_method",
        "feature",
        "target",
        "train_fraction",
        "test_fraction",
        "unused_fraction",
        "n_trajectories",
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
    Publication-quality rcParams matching NeurIPS / ICLR / ICML standards.
    See individual training scripts for full design rationale.
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
        "text.color":            COLORS["ink"],
        "axes.labelcolor":       COLORS["ink"],
        "xtick.color":           COLORS["ink"],
        "ytick.color":           COLORS["ink"],

        # --- Axes spines ---
        "axes.edgecolor":        COLORS["ink"],
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
        "grid.color":            COLORS["grid"],
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

        # --- PDF/SVG font embedding (avoids Type 3 fonts in camera-ready) ---
        "pdf.fonttype":          42,
        "ps.fonttype":           42,
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
    split_method: str,
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
            subset = sorted(subset, key=lambda r: float(r["train_fraction"]))
            if not subset:
                continue

            fractions = [100.0 * float(r["train_fraction"]) for r in subset]
            r2s = _finite_values([r["r2"] for r in subset])
            lambdas = _finite_values([r["lambda"] for r in subset])

            # ----------------------------------------------------------------
            # Fixed width: this plot has a continuous x-axis (percentages)
            # rather than discrete clip indices, so a compact fixed size works
            # better than the adaptive width used in the other scripts.
            # ----------------------------------------------------------------
            fig, ax = plt.subplots(figsize=(5.5, 3.2))

            # Main curve — VT Maroon, filled circle markers with white edge
            ax.plot(
                fractions,
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

            ax.set_xticks(fractions)
            ax.set_xticklabels([f"{int(x)}%" for x in fractions])

            if split_method == "fixed-test":
                xlabel = "Training fraction from the start of each clip"
                title_prefix = "Fixed-Test In-Clip Sweep"
            elif split_method == "in-clip":
                xlabel = "Training fraction within each clip"
                title_prefix = "Variable-Test In-Clip Sweep"
            elif split_method == "concat-fixed-test":
                xlabel = "Training fraction from the start of concatenated clips"
                title_prefix = "Fixed-Test Concat Sweep"
            else:
                xlabel = "Training fraction after concatenating clips"
                title_prefix = "Concat Split Fraction Sweep"

            ax.set_xlabel(xlabel)
            ax.set_ylabel("Test $R^2$")
            ax.set_title(f"{title_prefix}: {target} / {feature}")
            ax.set_ylim(0.55, 1.05)


            # Value annotations with optional lambda label on a second line
            for x, y, lam in zip(fractions, r2s, lambdas):
                if math.isfinite(y):
                    label = f"{y:.3f}"
                    if math.isfinite(lam):
                        label += f"\n$\\lambda$={lam:g}"
                    va = "bottom" if y >= 0 else "top"
                    ax.annotate(
                        label,
                        xy=(x, y),
                        xytext=(0, 5 if y >= 0 else -5),
                        textcoords="offset points",
                        ha="center",
                        va=va,
                        fontsize=7,
                        color=COLORS["annot"],
                    )

            _savefig(
                fig,
                plot_dir / f"split_fraction_{_safe_token(target)}_{_safe_token(feature)}.svg",
                also_pdf,
            )
            plt.close(fig)


def _write_summary(
    path: Path,
    args: argparse.Namespace,
    features: Sequence[str],
    targets: Sequence[str],
    fractions: Sequence[float],
    trajectories: Sequence[str],
    rows: Sequence[Dict[str, object]],
) -> None:
    best_by_pair = {}
    for feature in features:
        for target in targets:
            subset = [
                r for r in rows
                if r["feature"] == feature and r["target"] == target
            ]
            finite = [r for r in subset if math.isfinite(float(r["r2"]))]
            best_by_pair[f"{target}/{feature}"] = (
                max(finite, key=lambda r: float(r["r2"])) if finite else None
            )

    payload = {
        "robot": args.robot,
        "features": list(features),
        "targets": list(targets),
        "train_fractions": list(fractions),
        "trajectory_order": list(trajectories),
        "split_mode": args.split_method,
        "fixed_test_fraction": (
            args.fixed_test_fraction
            if args.split_method in ("fixed-test", "concat-fixed-test")
            else None
        ),
        "target_fps": args.target_fps,
        "skip_seconds": args.skip_seconds,
        "best_by_target_feature": best_by_pair,
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train readouts across multiple train/test fractions."
    )
    p.add_argument("--bundle-dir", required=True, type=Path)
    p.add_argument("--robot", default="go1")
    p.add_argument("--features", default=FEATURES_DEFAULT)
    p.add_argument("--targets", default=TARGETS_DEFAULT)
    p.add_argument("--train-fractions", default=FRACTIONS_DEFAULT,
                   help="Comma-separated fractions in (0,1). Default: 0.1..0.7.")
    p.add_argument("--split-method",
                   choices=["fixed-test", "in-clip", "concat", "concat-fixed-test"],
                   default="fixed-test",
                   help="Default fixed-test: train on the first X%% of each "
                        "clip and always test on the last fixed fraction. "
                        "in-clip: split each clip X/(1-X). concat: "
                        "concatenate selected clips, then split once. "
                        "concat-fixed-test: concatenate first, train on the "
                        "first X%%, and always test on the last fixed fraction.")
    p.add_argument("--fixed-test-fraction", type=float,
                   default=FIXED_TEST_FRACTION_DEFAULT,
                   help="For --split-method fixed-test, keep this final "
                        "fraction of every clip as test set. Default: 0.3.")
    p.add_argument("--lambdas", default="auto")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    p.add_argument("--trajectories", default=None,
                   help="Comma-separated trajectory IDs. Default: all usable "
                        "manifest trajectories sorted naturally.")
    p.add_argument("--target-fps", type=float, default=50.0)
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
    fractions = _float_list(args.train_fractions)
    if args.split_method in ("fixed-test", "concat-fixed-test"):
        if not 0.0 < args.fixed_test_fraction < 1.0:
            raise ValueError(
                "--fixed-test-fraction must be in (0, 1), "
                f"got {args.fixed_test_fraction}"
            )
        max_train = 1.0 - args.fixed_test_fraction
        too_large = [x for x in fractions if x > max_train + 1e-12]
        if too_large:
            raise ValueError(
                f"For {args.split_method} splitting, train fractions must be <= "
                f"{max_train:g}; got {too_large}"
            )
    trajectories = _resolve_trajectories(args)

    out_dir = args.output_dir or (
        args.bundle_dir / args.robot / "training" / "split_fraction_sweep"
    )
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("=== Split fraction sweep ===")
    print(f"robot:           {args.robot}")
    print(f"split method:    {args.split_method}")
    print(f"features:        {features}")
    print(f"targets:         {targets}")
    print(f"train fractions: {fractions}")
    if args.split_method in ("fixed-test", "concat-fixed-test"):
        print(f"fixed test frac: {args.fixed_test_fraction}")
    print(f"trajectories:    {trajectories}")
    print(f"output:          {out_dir}")

    in_clip_bundle_dir = _filtered_bundle(
        args.bundle_dir,
        args.robot,
        trajectories,
        out_dir,
    )

    specs = _build_specs(
        features,
        targets,
        fractions,
        trajectories,
        args.split_method,
    )
    _train_needed_runs(args, specs, in_clip_bundle_dir)
    rows = _collect_rows(args, specs)

    csv_path = out_dir / "split_fraction_sweep.csv"
    summary_path = out_dir / "split_fraction_summary.json"
    _write_csv(csv_path, rows)
    _write_summary(summary_path, args, features, targets, fractions, trajectories, rows)
    _plot_curves(rows, features, targets, args.split_method, plot_dir, args.pdf)

    print("\n=== Split fraction sweep report complete ===")
    print(f"curve csv: {csv_path}")
    print(f"summary:   {summary_path}")
    print(f"plots:     {plot_dir}")


if __name__ == "__main__":
    main()
