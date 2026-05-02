"""
PRC readout training with concatenate-then-split evaluation.

This script is intentionally separate from train_readout.py. It reuses the
existing loading, target construction, ridge training, and output writers, but
changes the train/test split:

  1. Load and concatenate all usable clips.
  2. Split the concatenated rows once, globally.
  3. Train on the first fraction and test on the remaining fraction.

Run from the repo's src directory:

    python3 -m openprc.automod._tools.training.train_readout_concat_split \\
      --bundle-dir openprc/automod/robot_bundle --robot go1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from train_readout import (  # noqa: E402
    DEFAULT_HORIZONS,
    FEATURE_LEVELS,
    FeatureSpec,
    _list_target_specs,
    collect_data,
    save_predictions,
    save_readout,
    train_one_target,
    write_metrics_csv,
    write_summary_json,
)


DEFAULT_LAMBDAS = [
    1e-4, 1e-3, 1e-2, 1e-1,
    1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0,
    1e4, 1e5, 1e6,
]


def _parse_features(s: str) -> FeatureSpec:
    s = s.strip().lower()
    if s not in FEATURE_LEVELS:
        raise ValueError(
            f"unknown feature level: {s!r}; valid: {sorted(FEATURE_LEVELS)}"
        )
    return FeatureSpec(level=s)


def _parse_lambdas(s: str) -> List[float]:
    if s == "auto":
        return list(DEFAULT_LAMBDAS)
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_horizons(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _csv_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _filtered_bundle(bundle_dir: Path, robot: str, trajectories: List[str], out_root: Path) -> Path:
    if not trajectories:
        return bundle_dir

    with (bundle_dir / "manifest.json").open() as f:
        manifest = json.load(f)

    robot_entry = manifest["robots"][robot]
    by_id = {t["id"]: t for t in robot_entry["trajectories"]}
    missing = [tid for tid in trajectories if tid not in by_id]
    if missing:
        raise KeyError(
            f"Requested trajectory id(s) not in manifest for {robot}: "
            f"{missing}. Available examples: {list(by_id)[:10]}"
        )
    robot_entry["trajectories"] = [by_id[tid] for tid in trajectories]

    filtered_root = out_root / "_filtered_bundle"
    if filtered_root.exists():
        shutil.rmtree(filtered_root)
    filtered_root.mkdir(parents=True, exist_ok=True)

    robot_link = filtered_root / robot
    robot_link.symlink_to(bundle_dir / robot, target_is_directory=True)

    with (filtered_root / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    return filtered_root


def _build_concat_masks(n_rows: int, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"concat_train_fraction must be in (0, 1), got {train_fraction}")
    if n_rows < 4:
        raise RuntimeError(f"need at least 4 rows for concat split, got {n_rows}")

    n_train = int(round(train_fraction * n_rows))
    n_train = max(2, min(n_rows - 2, n_train))
    train_mask = np.zeros(n_rows, dtype=bool)
    train_mask[:n_train] = True
    test_mask = ~train_mask
    return train_mask, test_mask


def main() -> None:
    p = argparse.ArgumentParser(
        description="PRC readout training with one global concat split."
    )
    p.add_argument("--bundle-dir", required=True, type=Path)
    p.add_argument("--robot", required=True)
    p.add_argument(
        "--features",
        default="node_vel",
        help="One feature level: strain, strain_rate, strain_accel, node_vel, node_acc.",
    )
    p.add_argument(
        "--targets",
        default="base_lin_vel,base_ang_vel,base_lin_acc,base_ang_acc,qvel,qacc",
    )
    p.add_argument("--trajectories", default=None,
                   help="Optional comma-separated trajectory IDs to include.")
    p.add_argument("--lambdas", default="auto")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    p.add_argument("--skip-seconds", type=float, default=2.0)
    p.add_argument("--target-fps", type=float, default=50.0)
    p.add_argument("--concat-train-fraction", type=float, default=0.7,
                   help="Fraction of concatenated rows used for training.")
    p.add_argument("--cv-mode", choices=["grouped", "temporal"], default="temporal",
                   help="Default temporal. Grouped CV still uses clip group ids "
                        "inside the globally selected train rows.")
    p.add_argument("--run-id", default=None)
    args = p.parse_args()

    bundle_dir = args.bundle_dir.resolve()
    run_id = args.run_id or f"concat_split_{time.strftime('%Y%m%d_%H%M%S')}"
    out_root = bundle_dir / args.robot / "training" / run_id
    pred_dir = out_root / "predictions"
    out_root.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    trajectories = _csv_list(args.trajectories)
    train_bundle_dir = _filtered_bundle(bundle_dir, args.robot, trajectories, out_root)

    feat = _parse_features(args.features)
    target_names = [t.strip() for t in args.targets.split(",") if t.strip()]
    target_specs = _list_target_specs(target_names)
    lambdas = _parse_lambdas(args.lambdas)
    horizons = _parse_horizons(args.horizons)

    feat_level = FEATURE_LEVELS[feat.level]
    print(f"=== PRC readout concat-split training: {args.robot} / {run_id} ===")
    print(f"  features:              {feat.label()} ({feat_level} level)")
    print(f"  targets:               {[t.name for t in target_specs]}")
    print(f"  trajectories:          {trajectories or 'all manifest clips'}")
    print(f"  target_fps:            {args.target_fps}")
    print(f"  skip_seconds:          {args.skip_seconds}")
    print(f"  concat_train_fraction: {args.concat_train_fraction}")
    print(f"  cv_mode:               {args.cv_mode}")
    print()

    print("Stage: collect data")
    X, Ys, group_ids, _train_unused, _test_unused, train_clip_ids, test_clip_ids = collect_data(
        str(train_bundle_dir),
        args.robot,
        feat,
        target_specs,
        split_mode="in-clip",
        in_clip_train_fraction=0.7,
        skip_seconds=args.skip_seconds,
        target_fps=args.target_fps,
    )

    train_mask, test_mask = _build_concat_masks(len(X), args.concat_train_fraction)
    X_tr = X[train_mask]
    X_te = X[test_mask]
    g_tr = group_ids[train_mask]
    g_te = group_ids[test_mask]
    Y_tr_dict = {k: v[train_mask] for k, v in Ys.items()}
    Y_te_dict = {k: v[test_mask] for k, v in Ys.items()}

    print(f"  concat split rows: train={train_mask.sum()}, test={test_mask.sum()}")

    np.savez(
        pred_dir / "_split_meta.npz",
        test_group_ids=g_te.astype(np.int32),
        train_group_ids=g_tr.astype(np.int32),
        split_mode=np.array("concat"),
        concat_train_fraction=np.float32(args.concat_train_fraction),
    )

    print("\nStage: train + evaluate")
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
        run_id,
        feat,
        results,
        train_clip_ids,
        test_clip_ids,
    )

    with (out_root / "concat_split_config.json").open("w") as f:
        json.dump({
            "split_mode": "concat",
            "concat_train_fraction": args.concat_train_fraction,
            "trajectories": trajectories or None,
            "source_bundle_dir": str(bundle_dir),
        }, f, indent=2)

    print("\n=== Done ===")
    print(f"  results saved to {out_root}")
    print("  split behavior: concatenated rows first, then global train/test split")


if __name__ == "__main__":
    main()
