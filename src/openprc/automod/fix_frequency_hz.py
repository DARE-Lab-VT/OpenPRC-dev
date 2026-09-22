"""
Patch the frequency_hz attribute on existing trajectory HDF5 files.

Background: an earlier version of write_trajectory computed frequency_hz
from `1/median(diff(time))`, which gave wildly wrong results (~29 kHz) when
the underlying ROS bag had clustered timestamps. The correct formula is
`(N - 1) / (t_end - t_start)`. This script reads each .h5, recomputes
frequency_hz, and writes it back — no re-ingest needed.

Usage:
    python fix_frequency_hz.py --bundle-dir .              # fix all robots
    python fix_frequency_hz.py --bundle-dir . --robot go1  # one robot
    python fix_frequency_hz.py --bundle-dir . --dry-run    # preview only

Idempotent: running twice does nothing the second time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import h5py
import numpy as np


def _expected_freq(time_arr: np.ndarray) -> float:
    if len(time_arr) < 2:
        return float("nan")
    dur = float(time_arr[-1] - time_arr[0])
    return (len(time_arr) - 1) / dur if dur > 0 else float("nan")


def patch_one_file(h5_path: str, dry_run: bool = False) -> dict:
    """Returns {'old', 'new', 'fixed': bool, 'duration_s'}."""
    with h5py.File(h5_path, "r" if dry_run else "r+") as f:
        time_arr = f["time"][:]
        old = float(f.attrs.get("frequency_hz", float("nan")))
        new = _expected_freq(time_arr)
        duration = float(time_arr[-1] - time_arr[0]) if len(time_arr) >= 2 else 0.0
        # Tolerance: within 1% means it's already correct
        already_correct = (
            np.isfinite(old) and np.isfinite(new)
            and abs(old - new) / max(new, 1e-6) < 0.01
        )
        fixed = False
        if not already_correct and not dry_run:
            f.attrs["frequency_hz"] = float(new)
            fixed = True
        return {
            "old": old, "new": new, "duration_s": duration, "fixed": fixed,
            "already_correct": already_correct,
        }


def collect_trajectory_files(bundle_dir: str, robot_filter: str = None) -> List[str]:
    """Walk the manifest and collect every trajectory HDF5 path."""
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        # Fallback: glob for trajectory.h5 files
        out = []
        for root, dirs, files in os.walk(bundle_dir):
            if "trajectories" not in root:
                continue
            if robot_filter and f"/{robot_filter}/" not in root:
                continue
            for fn in files:
                if fn.endswith(".h5"):
                    out.append(os.path.join(root, fn))
        return out

    with open(manifest_path) as f:
        manifest = json.load(f)

    out = []
    for robot_name, robot_entry in manifest["robots"].items():
        if robot_filter and robot_name != robot_filter:
            continue
        for t in robot_entry.get("trajectories", []):
            path = os.path.join(bundle_dir, t["path"])
            if os.path.exists(path):
                out.append(path)
    return out


def main():
    p = argparse.ArgumentParser(description="Fix frequency_hz on trajectory HDF5s")
    p.add_argument("--bundle-dir", required=True)
    p.add_argument("--robot", default=None,
                   help="Restrict to one robot (default: all robots in manifest)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    files = collect_trajectory_files(args.bundle_dir, args.robot)
    if not files:
        print(f"No trajectory HDF5 files found")
        sys.exit(0)

    print(f"Checking {len(files)} trajectory file(s)...")
    n_fixed = 0
    n_ok = 0
    for path in files:
        try:
            r = patch_one_file(path, dry_run=args.dry_run)
        except Exception as e:
            print(f"  ERROR {path}: {e}")
            continue

        rel = os.path.relpath(path, args.bundle_dir)
        if r["already_correct"]:
            print(f"  ok   {rel}: freq={r['new']:.2f} Hz "
                  f"({r['duration_s']:.1f}s)")
            n_ok += 1
        else:
            verb = "would fix" if args.dry_run else "FIXED"
            print(f"  {verb} {rel}: {r['old']:.2f} → {r['new']:.2f} Hz "
                  f"(over {r['duration_s']:.1f}s)")
            if r["fixed"]:
                n_fixed += 1

    print(f"\nDone. {n_fixed} fixed, {n_ok} already correct"
          + (" (dry run, nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
