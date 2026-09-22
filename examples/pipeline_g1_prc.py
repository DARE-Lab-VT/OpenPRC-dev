"""
G1 PRC Pipeline
===============
Full Physical Reservoir Computing pipeline for the Unitree G1 humanoid:

  Stage 1 — Reservoir preprocessing:
    Walk the G1's URDF and turn each link's visual mesh into a spring-mass
    reservoir spec (.npz) stored under g1/reservoir/.

  Stage 2 — Batch simulation:
    For every trajectory in the manifest, compose a DEMLAT simulation where
    FK-derived actuator signals drive the anchor nodes, then run it on GPU.
    Output lands in g1/reservoir_sims/<trajectory_id>/.

  Stage 3 — Readout training:
    Collect reservoir features (node velocities of interior nodes) and
    target signals (body-frame velocity, joint velocity) across all clips,
    run temporal k-fold CV to pick ridge regularization lambda, refit on
    the full train split, and evaluate on the held-out test split.

  Stage 4 — Plots:
    Time-series, scatter, residual histograms, and rollout-decay curves.

Usage:
    python examples/pipeline_g1_prc.py --bundle-dir /path/to/robot_bundle

Assumptions:
    - The bundle already has G1 trajectories ingested
      (run _tools/fetch/fetch_g1.py first if needed).
    - The manifest.json exists (run automod.build_manifest if not).
    - CUDA is available for Stage 2; fall back to cpu backend if not.

Notes on G1 vs Go1:
    - G1 is a 27-DOF whole-body humanoid (no dedicated contact-force
      sensors), so targets are body velocity and joint velocity only.
    - Trajectories come from LeRobot HuggingFace datasets at 30 fps.
    - The floating base (base_pose) is present, so body_vel works fine.
"""

import argparse
from pathlib import Path

import openprc.automod as automod


def parse_args():
    p = argparse.ArgumentParser(description="G1 PRC pipeline")
    p.add_argument("--bundle-dir", required=True,
                   help="Root of the robot bundle directory")
    p.add_argument("--preset", default="small",
                   choices=list(automod.PRESETS.keys()),
                   help="Reservoir size preset (default: small)")
    p.add_argument("--max-sims", type=int, default=None,
                   help="Cap number of new simulations this run")
    p.add_argument("--features", default="node_vel",
                   choices=list(automod.FEATURE_LEVELS.keys()),
                   help="Reservoir feature level for training")
    p.add_argument("--targets", default="body_vel,qvel,base_lin_vel,base_ang_vel",
                   help="Comma-separated training targets")
    p.add_argument("--target-fps", type=float, default=30.0,
                   help="Resample to this frame rate before training "
                        "(G1 LeRobot data is 30 fps; keep at 30 or lower)")
    p.add_argument("--skip-stages", default="",
                   help="Comma-separated stages to skip: preprocess,simulate,train,plot")
    p.add_argument("--force-preprocess", action="store_true",
                   help="Regenerate reservoir .npz even if cached")
    p.add_argument("--force-simulate", action="store_true",
                   help="Re-run simulations even if already complete")
    return p.parse_args()


def main():
    args = parse_args()
    skip = {s.strip() for s in args.skip_stages.split(",") if s.strip()}
    bundle_dir = str(Path(args.bundle_dir).resolve())
    robot = "g1"
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    # ------------------------------------------------------------------
    # Stage 1: Reservoir preprocessing
    # ------------------------------------------------------------------
    if "preprocess" not in skip:
        print("\n" + "=" * 60)
        print("STAGE 1 — Reservoir preprocessing")
        print("=" * 60)
        automod.batch_preprocess(
            bundle_dir=bundle_dir,
            robot_name=robot,
            params=automod.PRESETS[args.preset],
            force=args.force_preprocess,
        )
    else:
        print("Skipping Stage 1 (preprocess)")

    # ------------------------------------------------------------------
    # Stage 2: Batch simulation
    # ------------------------------------------------------------------
    if "simulate" not in skip:
        print("\n" + "=" * 60)
        print("STAGE 2 — Batch simulation")
        print("=" * 60)
        result = automod.batch_simulate(
            bundle_dir=bundle_dir,
            robot_name=robot,
            splits=("train", "test"),
            max_trajectories=args.max_sims,
            force=args.force_simulate,
            physics_dt=0.01,       # G1 trajectories are 30 fps; coarser dt fine
            save_dt=0.0333,        # match 30 fps save rate
            gravity=-9.81,
            damping_scale=1.0,
        )
        print(f"Simulation batch result: {result}")
    else:
        print("Skipping Stage 2 (simulate)")

    # ------------------------------------------------------------------
    # Stage 3: Readout training
    # ------------------------------------------------------------------
    run_dir = None
    if "train" not in skip:
        print("\n" + "=" * 60)
        print("STAGE 3 — Readout training")
        print("=" * 60)
        run_dir = automod.run_training(
            bundle_dir=bundle_dir,
            robot_name=robot,
            features=args.features,
            targets=targets,
            n_folds=5,
            skip_seconds=1.0,
            target_fps=args.target_fps,
            split_mode="in-clip",
            in_clip_train_fraction=0.7,
        )
    else:
        print("Skipping Stage 3 (train)")

    # ------------------------------------------------------------------
    # Stage 4: Plots
    # ------------------------------------------------------------------
    if "plot" not in skip and run_dir is not None:
        print("\n" + "=" * 60)
        print("STAGE 4 — Plots")
        print("=" * 60)
        plot_dir = automod.plot_run(run_dir, max_output_dims=12)
        print(f"Plots written to: {plot_dir}")
    else:
        if "plot" not in skip:
            print("Skipping Stage 4 (no run_dir — did you skip training?)")
        else:
            print("Skipping Stage 4 (plot)")

    print("\nDone.")


if __name__ == "__main__":
    main()
