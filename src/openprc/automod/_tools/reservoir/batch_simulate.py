"""
Batch-simulate every trajectory of a robot through DEMLAT.

Walks the manifest, finds every trajectory with split in {train, test},
checks which already have a simulation.h5, runs reservoir_to_demlat.py
on the rest. Skips already-completed sims unless --force is given.

This is what you run before training so that train_readout.py has
something to chew on.

Usage:
    python batch_simulate.py --bundle-dir . --robot go1
    python batch_simulate.py --bundle-dir . --robot g1 --max 8
    python batch_simulate.py --bundle-dir . --robot g1 --splits train
    python batch_simulate.py --bundle-dir . --robot go1 --force \\
        --physics-dt 0.005 --damping-scale 1.5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import List


HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(
        description="Batch-simulate trajectories for a robot"
    )
    p.add_argument("--bundle-dir", required=True)
    p.add_argument("--robot", required=True)
    p.add_argument(
        "--splits", default="train,test",
        help="Comma-separated splits to process (default: train,test)",
    )
    p.add_argument(
        "--max", type=int, default=None,
        help="Cap number of trajectories simulated this run (defaults to all)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-run sims even if simulation.h5 already exists",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="List what would be simulated without running anything",
    )
    # Pass-through to reservoir_to_demlat
    p.add_argument("--physics-dt", type=float, default=0.005)
    p.add_argument("--save-dt", type=float, default=0.02)
    p.add_argument("--damping-scale", type=float, default=1.0)
    p.add_argument("--no-gravity", action="store_true")
    p.add_argument("--gravity", type=float, default=-9.81)
    args = p.parse_args()

    splits = {s.strip() for s in args.splits.split(",") if s.strip()}

    manifest_path = os.path.join(args.bundle_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    if args.robot not in manifest["robots"]:
        raise KeyError(f"robot {args.robot!r} not in manifest")
    trajs = manifest["robots"][args.robot]["trajectories"]

    sim_root = os.path.join(args.bundle_dir, args.robot, "reservoir_sims")
    pending: List[dict] = []
    already: List[str] = []

    for t in trajs:
        if t.get("split", "train") not in splits:
            continue
        sim_h5 = os.path.join(sim_root, t["id"], "output", "simulation.h5")
        if os.path.exists(sim_h5) and not args.force:
            already.append(t["id"])
            continue
        pending.append(t)

    print(f"Robot: {args.robot}")
    print(f"  splits: {sorted(splits)}")
    print(f"  trajectories matching splits: "
          f"{len(pending) + len(already)}")
    print(f"  already simulated:            {len(already)}")
    print(f"  pending:                      {len(pending)}")

    if args.max is not None:
        pending = pending[: args.max]
        print(f"  capped this run to first {args.max}: "
              f"{[t['id'] for t in pending][:5]}"
              f"{'...' if len(pending) > 5 else ''}")

    if not pending:
        print("\nNothing to do. Use --force to re-run completed sims.")
        return

    if args.dry_run:
        print("\nDry-run, would simulate:")
        for t in pending:
            print(f"  {t['split']:5s}  {t['id']}")
        return

    # Build the per-trajectory command
    runner = os.path.join(
        os.path.dirname(HERE), "reservoir", "reservoir_to_demlat.py"
    )
    if not os.path.exists(runner):
        raise FileNotFoundError(
            f"Could not find reservoir_to_demlat.py at {runner}"
        )

    base_cmd = [
        sys.executable, runner,
        "--bundle-dir", args.bundle_dir,
        "--robot", args.robot,
        "--physics-dt", str(args.physics_dt),
        "--save-dt", str(args.save_dt),
        "--damping-scale", str(args.damping_scale),
    ]
    if args.no_gravity:
        base_cmd.append("--no-gravity")
    else:
        base_cmd.extend(["--gravity", str(args.gravity)])

    n_done = 0
    n_failed = 0
    t_start = time.time()

    for i, t in enumerate(pending):
        clip_id = t["id"]
        cmd = base_cmd + ["--trajectory", clip_id]
        print(f"\n[{i+1}/{len(pending)}] {clip_id}")
        elapsed = time.time() - t_start
        if i > 0:
            avg = elapsed / i
            remaining = avg * (len(pending) - i)
            print(f"  elapsed: {elapsed:.0f}s  "
                  f"avg/clip: {avg:.0f}s  "
                  f"est. remaining: {remaining:.0f}s")
        try:
            subprocess.run(cmd, check=True)
            n_done += 1
        except subprocess.CalledProcessError as e:
            print(f"  FAILED: exit code {e.returncode}")
            n_failed += 1

    total = time.time() - t_start
    print(f"\n=== Batch done in {total:.0f}s ===")
    print(f"  succeeded: {n_done}")
    print(f"  failed:    {n_failed}")
    if n_failed:
        print(f"  re-run with --force to retry failed ones, or fix the")
        print(f"  underlying issue and re-run without --force "
              f"(successful clips will be skipped).")


if __name__ == "__main__":
    main()
