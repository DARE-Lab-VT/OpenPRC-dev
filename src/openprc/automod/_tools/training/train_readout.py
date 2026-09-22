"""
PRC readout training.

For one robot:
  1. Walk every trajectory clip in the manifest with split=train.
     For each clip:
        a. Load reservoir simulation features (bar strains + optional positions/velocities).
        b. Load matching trajectory data.
        c. Compute targets via Savitzky-Golay smoothing (qvel, qacc, etc.).
  2. Stack features and targets into a big design matrix X (T_total × D_feat),
     keeping per-clip group IDs for grouped CV.
  3. For each target:
        a. Grouped-k-fold CV over a log-spaced lambda grid.
        b. Re-fit on all train data with the chosen lambda.
        c. Evaluate on test-split clips, compute MSE / R² overall and per-output.
        d. Run multi-step autoregressive rollouts at horizons {1, 5, 10, 50, 100}.
  4. Save:
        - <robot>/training/<run_id>/readout_<target>.npz  (W, b, lambda, feature spec)
        - <robot>/training/<run_id>/metrics.csv             (every (target, horizon, metric))
        - <robot>/training/<run_id>/summary.json            (high-level numbers)
        - <robot>/training/<run_id>/predictions/<target>.npz
              (y_true, y_pred for plotting)

Usage:
    python train_readout.py --bundle-dir . --robot go1 \\
        [--features strain | strain+pos | strain+pos+vel] \\
        [--targets qvel,qacc,base_vel,...] \\
        [--n-folds 5] [--lambdas auto | 1e-4,1e-2,1.0,...] \\
        [--run-id <auto-timestamp>]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from readout_math import (  # noqa: E402
    ridge_regression,
    ridge_kernel_fit,
    predict,
    grouped_kfold_indices,
    cv_ridge_select_lambda,
    cv_ridge_select_lambda_kernel,
    savitzky_golay_derivative,
    mse,
    r_squared,
    base_pose_to_body_velocity,
)


# Sentinel used in CSV when a metric is undefined for that horizon
NAN_SENTINEL = "nan"

# Default rollout horizons (in trajectory-frame steps)
DEFAULT_HORIZONS = [1, 5, 10, 50, 100]

# Default lambda grid (log-spaced, broad). Denser around the typical
# regime (1 to 1000) where physical reservoirs tend to land.
DEFAULT_LAMBDAS = [
    1e-4, 1e-3, 1e-2, 1e-1,
    1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0,
    1e4, 1e5, 1e6,
]


@dataclass
class FeatureSpec:
    """
    What goes into the reservoir state vector X. Pick ONE physical level
    per training run; ablations are run separately.

    Levels (dimensionally consistent with which targets):
        strain         — dimensionless deformation; pairs with pose/qpos
        strain_rate    — dε/dt; pairs with velocity targets
        strain_accel   — d²ε/dt²; pairs with acceleration targets
        node_vel       — absolute node velocities (m/s); pairs with velocity targets
        node_acc       — dv/dt of nodes (m/s²); pairs with acceleration targets

    The principle: predict velocities from velocity-level features, accelerations
    from acceleration-level features. This leverages the dimensional structure
    of the physical reservoir — strain rates ARE velocities at each spring,
    so mapping them to body velocity is dimensionally consistent.

    Position-level features (pos) are intentionally NOT supported: in a real
    deployment we cannot observe absolute node positions, only proprioceptive
    derivatives (velocities, accelerations). Exposing pos would weaken the
    auto-modeling claim.
    """
    level: str = "node_vel"  # one of the 5 levels above

    def label(self) -> str:
        return self.level


# Allowed feature levels and what target levels they're physically matched to
FEATURE_LEVELS = {
    "strain":       "kinematic",
    "strain_rate":  "velocity",
    "strain_accel": "acceleration",
    "node_vel":     "velocity",
    "node_acc":     "acceleration",
}


@dataclass
class TargetSpec:
    """A target signal we want the readout to predict."""
    name: str                 # 'qvel', 'qacc', 'base_vel', ...
    source: str               # 'derived' | 'direct'
    field: str                # underlying HDF5 dataset path or 'qpos'
    derivative_order: int = 0  # 0 = direct, 1 = SG vel, 2 = SG acc


@dataclass
class TrainResult:
    target_name: str
    n_features: int
    n_outputs: int
    n_train_samples: int
    n_test_samples: int
    chosen_lambda: float
    cv_mse_grid: np.ndarray   # (n_lambdas, n_folds)
    test_mse: float
    test_r2: float
    test_mse_per_output: np.ndarray
    test_r2_per_output: np.ndarray
    rollout_mse: Dict[int, float] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def _load_bundle_manifest(bundle_dir: str) -> dict:
    with open(os.path.join(bundle_dir, "manifest.json")) as f:
        return json.load(f)


def _load_trajectory(bundle_dir: str, traj_path: str) -> dict:
    """Read the bundle-format trajectory HDF5 and return a dict of arrays."""
    import h5py
    with h5py.File(os.path.join(bundle_dir, traj_path), "r") as f:
        out = {
            "time": f["time"][:].astype(np.float64),
            "qpos": f["qpos"][:].astype(np.float32),
            "joint_names": list(f.attrs["joint_names"]),
            "frequency_hz": float(f.attrs.get("frequency_hz", 50.0)),
        }
        if "qvel" in f:
            out["qvel"] = f["qvel"][:].astype(np.float32)
        if "tau" in f:
            out["tau"] = f["tau"][:].astype(np.float32)
        if "base_pose" in f:
            out["base_pose"] = f["base_pose"][:].astype(np.float32)
        if "base_vel" in f:
            out["base_vel"] = f["base_vel"][:].astype(np.float32)
        # Contact forces (Go1)
        if "contact/foot_force" in f:
            out["foot_force"] = f["contact/foot_force"][:].astype(np.float32)
    return out


def _load_simulation_features(
    sim_path: str, geom_path: str, feat: FeatureSpec, dt: float,
) -> np.ndarray:
    """
    Build the (T, D_feat) feature matrix from a DEMLAT simulation output,
    using a single physical level chosen by feat.level.

    For node-derived levels (node_vel, node_acc), we strip out anchor nodes
    by default. Anchors are position-actuated — DEMLAT teleports them to
    follow the FK trajectory each step, so their velocities are essentially
    discrete-differenced echoes of the input signal. Including them lets the
    readout "cheat" by leaking the target into the features rather than
    learning anything from the reservoir's transformation. Interior nodes
    are where the real reservoir dynamics live.

    Anchor identification comes from the DEMLAT geometry attribute bitmask:
    bit 1 (0x02) marks position-actuated nodes (anchors).

    For derived levels (strain_rate, strain_accel, node_acc), we compute the
    derivative on the fly using Savitzky-Golay smoothing — no re-simulation
    needed. The simulation HDF5 already contains positions, velocities, and
    bar strains.

    dt is the simulation save-rate timestep (1 / save_dt), used for derivatives.
    """
    import h5py
    level = feat.level
    if level not in FEATURE_LEVELS:
        raise ValueError(f"unknown feature level: {level!r}; "
                         f"valid: {list(FEATURE_LEVELS)}")

    # Load anchor mask from geometry. Used to filter node_vel / node_acc.
    interior_mask = None
    if level in ("node_vel", "node_acc"):
        if not os.path.exists(geom_path):
            raise RuntimeError(
                f"geometry.h5 not found at {geom_path}; needed to filter anchors"
            )
        with h5py.File(geom_path, "r") as gf:
            attrs = gf["nodes/attributes"][:].astype(np.uint8)
        # Bit 1 = position actuator; we want everything WITHOUT that bit set
        interior_mask = (attrs & 0x02) == 0
        n_total = len(attrs)
        n_interior = int(interior_mask.sum())
        n_anchor = n_total - n_interior

    with h5py.File(sim_path, "r") as f:
        if level == "strain":
            if "time_series/elements/bars/strain" not in f:
                raise RuntimeError(
                    f"strain dataset missing in {sim_path}; "
                    f"reservoir_to_demlat must have run with auto_process=True"
                )
            return f["time_series/elements/bars/strain"][:].astype(np.float32)

        if level == "strain_rate":
            if "time_series/elements/bars/strain" not in f:
                raise RuntimeError(f"strain dataset missing in {sim_path}")
            strain = f["time_series/elements/bars/strain"][:].astype(np.float32)
            return savitzky_golay_derivative(strain, dt=dt, order=1)

        if level == "strain_accel":
            if "time_series/elements/bars/strain" not in f:
                raise RuntimeError(f"strain dataset missing in {sim_path}")
            strain = f["time_series/elements/bars/strain"][:].astype(np.float32)
            return savitzky_golay_derivative(strain, dt=dt, order=2)

        if level == "node_vel":
            if "time_series/nodes/velocities" not in f:
                raise RuntimeError(f"velocities dataset missing in {sim_path}")
            vel = f["time_series/nodes/velocities"][:]   # (T, N, 3)
            # Anchors are position-actuated; their velocities trivially echo
            # the input. Filter to interior nodes only.
            vel = vel[:, interior_mask, :]
            print(f"      node_vel: keeping {vel.shape[1]} interior of {n_total} nodes "
                  f"({n_anchor} anchors filtered out)") if False else None
            return vel.reshape(vel.shape[0], -1).astype(np.float32)

        if level == "node_acc":
            if "time_series/nodes/velocities" not in f:
                raise RuntimeError(f"velocities dataset missing in {sim_path}")
            vel = f["time_series/nodes/velocities"][:]   # (T, N, 3)
            vel = vel[:, interior_mask, :]
            vel_flat = vel.reshape(vel.shape[0], -1).astype(np.float32)
            return savitzky_golay_derivative(vel_flat, dt=dt, order=1)

    raise RuntimeError(f"unhandled feature level: {level!r}")


def _compute_target(traj: dict, target: TargetSpec) -> Optional[np.ndarray]:
    """
    Compute or extract a target array (T, D_target) from a loaded trajectory.
    Returns None if the underlying data isn't present (caller should skip).
    """
    if target.source == "direct":
        return traj.get(target.field)

    if target.source == "derived":
        base = traj.get(target.field)
        if base is None:
            return None
        dt = 1.0 / traj["frequency_hz"]
        return savitzky_golay_derivative(
            base.astype(np.float32), dt=dt, order=target.derivative_order
        )

    if target.source == "body_vel":
        # 6D body-frame velocity: [vx, vy, vz, wx, wy, wz].
        # Linear velocity is rotated from world to body frame using the
        # orientation quaternion. This has 5-10x more variance than world-frame
        # linear velocity for a walking/turning robot and is the physically
        # meaningful target for legged locomotion.
        bp = traj.get("base_pose")
        if bp is None:
            return None
        fps = float(traj["frequency_hz"])
        return base_pose_to_body_velocity(bp, fps)

    if target.source in ("base_lin_vel", "base_ang_vel",
                          "base_lin_acc", "base_ang_acc"):
        bp = traj.get("base_pose")
        if bp is None:
            return None
        from readout_math import (
            base_pose_to_base_velocities, base_pose_to_base_accelerations,
        )
        fps = float(traj["frequency_hz"])
        if target.source == "base_lin_vel":
            lin, _ = base_pose_to_base_velocities(bp, fps)
            return lin
        if target.source == "base_ang_vel":
            _, ang = base_pose_to_base_velocities(bp, fps)
            return ang
        if target.source == "base_lin_acc":
            lin, _ = base_pose_to_base_accelerations(bp, fps)
            return lin
        if target.source == "base_ang_acc":
            _, ang = base_pose_to_base_accelerations(bp, fps)
            return ang

    return None


def _list_target_specs(target_names: List[str]) -> List[TargetSpec]:
    """Build TargetSpec objects from CLI names."""
    specs = []
    for name in target_names:
        n = name.lower().strip()
        if n == "body_vel":
            # 6D body-frame velocity: [vx,vy,vz,wx,wy,wz].
            # Linear velocity rotated to body frame — the correct target for
            # legged robots. body-frame vx has 5-10x more variance than
            # world-frame for a walking/turning robot.
            specs.append(TargetSpec("body_vel", "body_vel", "base_pose"))
        elif n == "qvel":
            specs.append(TargetSpec("qvel", "derived", "qpos", 1))
        elif n == "qacc":
            specs.append(TargetSpec("qacc", "derived", "qpos", 2))
        elif n == "base_vel":
            specs.append(TargetSpec("base_vel", "direct", "base_vel"))
        elif n == "base_acc":
            specs.append(TargetSpec("base_acc", "derived", "base_pose", 2))
        elif n == "base_lin_vel":
            specs.append(TargetSpec("base_lin_vel", "base_lin_vel", "base_pose"))
        elif n == "base_ang_vel":
            specs.append(TargetSpec("base_ang_vel", "base_ang_vel", "base_pose"))
        elif n == "base_lin_acc":
            specs.append(TargetSpec("base_lin_acc", "base_lin_acc", "base_pose"))
        elif n == "base_ang_acc":
            specs.append(TargetSpec("base_ang_acc", "base_ang_acc", "base_pose"))
        elif n == "tau":
            specs.append(TargetSpec("tau", "direct", "tau"))
        elif n == "foot_force":
            specs.append(TargetSpec("foot_force", "direct", "foot_force"))
        else:
            raise ValueError(
                f"Unknown target {name!r}. Supported: body_vel, qvel, qacc, "
                f"base_lin_vel, base_ang_vel, base_lin_acc, base_ang_acc, "
                f"base_vel, base_acc, tau, foot_force"
            )
    return specs


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def collect_data(
    bundle_dir: str,
    robot_name: str,
    feat: FeatureSpec,
    targets: List[TargetSpec],
    split_mode: str = "in-clip",
    in_clip_train_fraction: float = 0.7,
    skip_seconds: float = 0.0,
    target_fps: Optional[float] = None,
) -> Tuple[
    np.ndarray, Dict[str, np.ndarray], np.ndarray,
    np.ndarray, np.ndarray, List[str], List[str],
]:
    """
    Walk every trajectory of `robot_name`, load reservoir features + each
    target, stack across clips, and produce train/test row masks.

    split_mode:
      - "in-clip"   : each clip's first in_clip_train_fraction of frames are
                      train, the remainder is test. All clips contribute to both.
                      Avoids between-clip distribution shift.
      - "multi-clip": clips with manifest `split=='train'` go to train rows;
                      clips with `split=='test'` go to test rows. Original
                      behavior — tests the harder generalization claim.

    skip_seconds: drop the first N seconds of each clip (transient).

    Returns:
        X:            (T_total, D_feat)
        Ys:           {target_name: (T_total, D_target)}
        group_ids:    (T_total,) per-row clip index for grouped CV
        train_mask:   (T_total,) bool
        test_mask:    (T_total,) bool
        train_clip_ids, test_clip_ids: lists for reporting
    """
    if split_mode not in ("in-clip", "multi-clip"):
        raise ValueError(f"split_mode must be 'in-clip' or 'multi-clip', got {split_mode!r}")
    if not 0.0 < in_clip_train_fraction < 1.0:
        raise ValueError(
            f"in_clip_train_fraction must be in (0, 1), got {in_clip_train_fraction}"
        )

    manifest = _load_bundle_manifest(bundle_dir)
    if robot_name not in manifest["robots"]:
        raise KeyError(f"robot {robot_name!r} not in manifest")
    trajs = manifest["robots"][robot_name]["trajectories"]

    print(f"  collecting data ({split_mode} mode)...")

    X_chunks: List[np.ndarray] = []
    Y_chunks: Dict[str, List[np.ndarray]] = {t.name: [] for t in targets}
    group_chunks: List[np.ndarray] = []
    train_chunks: List[np.ndarray] = []
    test_chunks: List[np.ndarray] = []
    clip_ids_seen: List[str] = []
    train_clip_ids: List[str] = []
    test_clip_ids: List[str] = []
    skipped: List[str] = []

    sim_dir_root = os.path.join(bundle_dir, robot_name, "reservoir_sims")
    n_loaded = 0

    for clip_i, t_entry in enumerate(trajs):
        clip_id = t_entry["id"]
        clip_split = t_entry.get("split", "train")
        sim_dir = os.path.join(sim_dir_root, clip_id)
        sim_path = os.path.join(sim_dir, "output", "simulation.h5")
        geom_path = os.path.join(sim_dir, "input", "geometry.h5")

        if not os.path.exists(sim_path):
            skipped.append(f"{clip_id} (no sim)")
            continue

        # Load features. Need the simulation's dt for derivative computation
        # (Savitzky-Golay needs the timestep). DEMLAT records this as
        # frame_rate = 1/dt_save in the simulation HDF5 root attrs.
        try:
            import h5py
            with h5py.File(sim_path, "r") as f_attrs:
                fr = float(f_attrs.attrs.get("frame_rate", 50.0))
            sim_dt = 1.0 / max(fr, 1e-9)
            X_clip = _load_simulation_features(sim_path, geom_path, feat, dt=sim_dt)
        except Exception as e:
            print(f"    skip {clip_id}: features load failed ({e})")
            skipped.append(f"{clip_id} (feat err)")
            continue

        # Load trajectory
        traj = _load_trajectory(bundle_dir, t_entry["path"])

        # The simulation save rate may differ from the trajectory rate (e.g.
        # legkilo trajectories are 497 Hz but DEMLAT default save_dt=0.01s
        # gives 100 Hz strain). Both span the same wall-clock duration, so we
        # resample features onto the trajectory time grid via linear interp.
        sim_t_max = max(traj["time"][-1] - traj["time"][0], 1e-9)
        T_feat = X_clip.shape[0]
        sim_times = np.linspace(0.0, sim_t_max, T_feat)
        traj_times = traj["time"] - traj["time"][0]
        T_traj = len(traj_times)
        if T_traj < 20:
            skipped.append(f"{clip_id} (trajectory too short, T={T_traj})")
            continue
        if T_feat != T_traj:
            X_resampled = np.empty((T_traj, X_clip.shape[1]), dtype=np.float32)
            for d in range(X_clip.shape[1]):
                X_resampled[:, d] = np.interp(traj_times, sim_times, X_clip[:, d])
            X_clip = X_resampled
        T = T_traj

        if T < 20:
            skipped.append(f"{clip_id} (too short, T={T})")
            continue

        # Subsample features to target_fps BEFORE computing derivative targets.
        # High-rate trajectories (e.g. go1 at 497 Hz) require subsampling first:
        # SG derivatives computed at the original rate span only ~20 ms (11
        # frames at 497 Hz) and amplify sensor noise into apparent velocities
        # of 10+ m/s. At 50 Hz the same window covers 220 ms and gives clean
        # derivatives. Direct targets (qvel, tau) are unaffected by this order.
        fps = float(traj["frequency_hz"])
        effective_fps = fps
        stride = 1
        if target_fps is not None and target_fps < fps:
            stride = max(1, int(round(fps / target_fps)))
            X_clip = X_clip[::stride]
            T = X_clip.shape[0]
            effective_fps = fps / stride
            if T < 20:
                skipped.append(f"{clip_id} (too short after subsample, T={T})")
                continue

        # Build a subsampled view of the trajectory for derivative targets.
        # For direct targets (qvel, tau) the subsampling is just indexing;
        # for derivative targets (body_vel, base_lin_vel, ...) the derivative
        # must be computed from the subsampled signal using effective_fps.
        traj_sub = dict(traj)
        traj_sub["frequency_hz"] = effective_fps
        for raw_key in ("base_pose", "qpos", "qvel", "tau", "foot_force",
                        "base_vel", "base_pose"):
            if raw_key in traj and traj[raw_key] is not None:
                traj_sub[raw_key] = traj[raw_key][:T_traj:stride][:T]

        # Compute targets from the subsampled trajectory
        target_arrays: Dict[str, Optional[np.ndarray]] = {}
        any_missing = False
        for ts in targets:
            arr = _compute_target(traj_sub, ts)
            if arr is None:
                any_missing = True
                break
            target_arrays[ts.name] = arr[:T]

        if any_missing:
            skipped.append(f"{clip_id} (target missing)")
            continue

        # Drop transient. Sanity-check that frequency_hz is plausible — a
        # broken trajectory writer once wrote 29 kHz here, which collapsed
        # every clip to <30 frames. If skip_frames >= T, skip with a clear
        # error instead of silently using only a handful of frames.
        if not (1.0 < fps < 5000.0):
            skipped.append(f"{clip_id} (suspicious freq={fps:.1f} Hz; "
                            f"run _tools/fix_frequency_hz.py)")
            continue
        skip_frames = int(round(skip_seconds * effective_fps))
        if skip_frames >= T - 20:
            skipped.append(f"{clip_id} (skip_frames {skip_frames} too large "
                            f"vs T={T} at {effective_fps:.1f} Hz)")
            continue
        X_clip = X_clip[skip_frames:T]
        for name in target_arrays:
            target_arrays[name] = target_arrays[name][skip_frames:T]
        T_kept = T - skip_frames
        if n_loaded < 5:
            print(f"    {clip_id}: T_sim={T_feat}, T_traj={T_traj}, "
                  f"freq={fps:.1f} Hz, eff_fps={effective_fps:.1f}, "
                  f"skip={skip_frames}, kept={T_kept}")

        # Build per-clip train/test masks
        if split_mode == "in-clip":
            # First train_fraction frames → train, rest → test
            n_train_in = int(round(in_clip_train_fraction * T_kept))
            n_train_in = max(2, min(T_kept - 2, n_train_in))   # at least 2 in each
            train_in = np.zeros(T_kept, dtype=bool)
            train_in[:n_train_in] = True
            test_in = ~train_in
            # Both ids for reporting
            train_clip_ids.append(clip_id)
            test_clip_ids.append(clip_id)
        else:
            # multi-clip: entire clip goes to one side
            if clip_split == "train":
                train_in = np.ones(T_kept, dtype=bool)
                test_in = np.zeros(T_kept, dtype=bool)
                train_clip_ids.append(clip_id)
            elif clip_split == "test":
                train_in = np.zeros(T_kept, dtype=bool)
                test_in = np.ones(T_kept, dtype=bool)
                test_clip_ids.append(clip_id)
            else:
                # Validation or other — skip
                skipped.append(f"{clip_id} (split={clip_split})")
                continue

        X_chunks.append(X_clip)
        for name in Y_chunks:
            Y_chunks[name].append(target_arrays[name])
        group_chunks.append(np.full(T_kept, clip_i, dtype=np.int32))
        train_chunks.append(train_in)
        test_chunks.append(test_in)
        clip_ids_seen.append(clip_id)
        n_loaded += 1
        if n_loaded % 10 == 0:
            print(f"    loaded {n_loaded} clips")

    if not X_chunks:
        # Build a clear, actionable message based on why we got nothing
        no_sim = [s for s in skipped if "(no sim)" in s]
        feat_err = [s for s in skipped if "(feat err)" in s]
        target_missing = [s for s in skipped if "(target missing)" in s]

        msg = [
            f"No usable clips for {robot_name!r} in {split_mode} mode.",
            f"  manifest had {len(trajs)} clips total, all skipped:",
        ]
        if no_sim:
            msg.append(
                f"  • {len(no_sim)}/{len(trajs)} have no simulation.h5 yet."
            )
            msg.append(
                f"    Run reservoir_to_demlat.py on these first. Example:"
            )
            for s in no_sim[:3]:
                clip_id_s = s.split(" ")[0]
                msg.append(
                    f"      python3 _tools/reservoir/reservoir_to_demlat.py "
                    f"--bundle-dir . --robot {robot_name} --trajectory {clip_id_s}"
                )
            if len(no_sim) > 3:
                msg.append(f"    ... and {len(no_sim) - 3} more")
        if feat_err:
            msg.append(f"  • {len(feat_err)} had feature-load errors: "
                        f"{[s.split()[0] for s in feat_err[:3]]}")
        if target_missing:
            msg.append(f"  • {len(target_missing)} are missing required target "
                        f"signals (qpos/base_pose/etc).")
        raise RuntimeError("\n".join(msg))

    X = np.concatenate(X_chunks, axis=0)
    Ys = {name: np.concatenate(chunks, axis=0) for name, chunks in Y_chunks.items()}
    group_ids = np.concatenate(group_chunks, axis=0)
    train_mask = np.concatenate(train_chunks, axis=0)
    test_mask = np.concatenate(test_chunks, axis=0)

    print(f"  collected X: {X.shape}, "
          f"{n_loaded} clips, {len(skipped)} skipped")
    print(f"  train rows: {train_mask.sum()}, test rows: {test_mask.sum()}")
    if skipped:
        print(f"  skipped (first 5): {skipped[:5]}")

    return X, Ys, group_ids, train_mask, test_mask, train_clip_ids, test_clip_ids


def train_one_target(
    target_name: str,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    group_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    lambdas: List[float],
    n_folds: int,
    horizons: List[int],
    cv_mode: str = "grouped",
) -> Tuple[TrainResult, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train one readout: CV on train, refit, evaluate on test, compute rollouts.
    Returns (result_summary, W, b, Y_test_pred).

    cv_mode:
      "grouped"  — leave-one-clip-out (tests cross-clip generalization; requires
                   enough clips for meaningful CV, degrades with < 3 clips)
      "temporal" — k contiguous temporal folds within the concatenated train set
                   (matches the in-clip train/test split; valid even with 1 clip)
    """
    from readout_math import kfold_indices
    print(f"\n  Training target: {target_name}")
    print(f"    X_train: {X_train.shape},  Y_train: {Y_train.shape}")
    print(f"    X_test:  {X_test.shape},   Y_test:  {Y_test.shape}")

    # Choose CV strategy
    n_groups = len(np.unique(group_train))
    if cv_mode == "grouped" and n_groups >= 3:
        folds = grouped_kfold_indices(group_train, k=n_folds)
        cv_label = f"grouped ({n_groups} clips, {len(folds)} folds)"
    else:
        if cv_mode == "grouped" and n_groups < 3:
            print(f"    ⚠ only {n_groups} clip(s) in train set; "
                  f"switching CV to temporal to avoid degenerate leave-one-out")
        folds = kfold_indices(len(X_train), k=n_folds)
        cv_label = f"temporal ({n_folds} folds)"
    print(f"    CV: {cv_label}, {len(lambdas)} lambdas")

    t0 = time.time()
    # For wide matrices (n_samples << n_features) the kernel form is
    # dramatically faster — precompute X X^T once, reuse across all
    # (lambda, fold) pairs. Bar strain features are typically ~30000 per
    # robot while we have hundreds to a few thousand samples; the kernel
    # form turns multi-minute CV into a few seconds.
    n_train, n_feat = X_train.shape
    if n_feat > n_train:
        chosen_lambda, mse_grid = cv_ridge_select_lambda_kernel(
            X_train, Y_train, lambdas, folds, fit_intercept=True,
        )
    else:
        chosen_lambda, mse_grid = cv_ridge_select_lambda(
            X_train, Y_train, lambdas, folds, fit_intercept=True,
        )
    cv_time = time.time() - t0
    print(f"    chosen λ = {chosen_lambda:g}  (CV took {cv_time:.1f}s)")

    # Show the per-lambda CV MSE so we can tell if the chosen λ is at an
    # edge of the grid. If it is, the grid likely needs to extend further;
    # if it's mid-grid, we're properly converged.
    mean_mse_per_lambda = mse_grid.mean(axis=1)
    chosen_idx = lambdas.index(chosen_lambda)
    grid_str = "    CV grid: " + " ".join(
        f"[{lam:.0e}={val:.4g}]{'★' if i == chosen_idx else ''}"
        for i, (lam, val) in enumerate(zip(lambdas, mean_mse_per_lambda))
    )
    print(grid_str)
    if chosen_idx == 0:
        print(f"    ⚠ chosen λ is at the LOW end of the grid; "
               f"add smaller lambdas")
    elif chosen_idx == len(lambdas) - 1:
        print(f"    ⚠ chosen λ is at the HIGH end of the grid; "
               f"add larger lambdas (or expect over-regularization)")

    # Refit on full train set with chosen lambda
    if n_feat > n_train:
        # Wide matrix: use the memory-efficient kernel form
        W, b = ridge_kernel_fit(X_train, Y_train, chosen_lambda, fit_intercept=True)
    else:
        W, b = ridge_regression(X_train, Y_train, chosen_lambda, fit_intercept=True)

    # Test-set evaluation
    Y_pred = predict(X_test, W, b)
    test_mse_overall = float(mse(Y_test, Y_pred))
    test_r2_overall = float(r_squared(Y_test, Y_pred))
    test_mse_per_out = mse(Y_test, Y_pred, axis=0)
    test_r2_per_out = r_squared(Y_test, Y_pred, axis=0)

    print(f"    test MSE: {test_mse_overall:.6g}, R²: {test_r2_overall:.4f}")

    # Multi-step horizons: predict y(t+h) given X(t).
    # We approximate this by shifting the target h frames forward and recomputing
    # MSE against the unshifted prediction. This measures "how well does X(t)
    # predict targets h steps ahead given a one-step-trained W".
    rollout_mse: Dict[int, float] = {}
    for h in horizons:
        if h == 1:
            rollout_mse[h] = test_mse_overall
            continue
        if h >= len(Y_test):
            rollout_mse[h] = float("nan")
            continue
        # Y_test[t+h] vs Y_pred[t]
        err = mse(Y_test[h:], Y_pred[: -h] if h > 0 else Y_pred)
        rollout_mse[h] = float(err)
    print(f"    rollout MSE @ {horizons}: "
          f"{[f'{rollout_mse[h]:.4g}' for h in horizons]}")

    return TrainResult(
        target_name=target_name,
        n_features=X_train.shape[1],
        n_outputs=Y_train.shape[1],
        n_train_samples=X_train.shape[0],
        n_test_samples=X_test.shape[0],
        chosen_lambda=chosen_lambda,
        cv_mse_grid=mse_grid,
        test_mse=test_mse_overall,
        test_r2=test_r2_overall,
        test_mse_per_output=test_mse_per_out,
        test_r2_per_output=test_r2_per_out,
        rollout_mse=rollout_mse,
    ), W, b, Y_pred


# -----------------------------------------------------------------------------
# Output writers
# -----------------------------------------------------------------------------

def save_readout(out_dir: str, target_name: str, W: np.ndarray,
                  b: np.ndarray, result: TrainResult, feat: FeatureSpec):
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, f"readout_{target_name}.npz"),
        W=W,
        b=b,
        chosen_lambda=np.float64(result.chosen_lambda),
        n_features=np.int32(result.n_features),
        n_outputs=np.int32(result.n_outputs),
        feature_spec=json.dumps(asdict(feat)),
        target_name=target_name,
    )


def save_predictions(out_dir: str, target_name: str,
                      Y_true: np.ndarray, Y_pred: np.ndarray):
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, f"{target_name}.npz"),
        Y_true=Y_true.astype(np.float32),
        Y_pred=Y_pred.astype(np.float32),
    )


def write_metrics_csv(out_path: str, results: List[TrainResult],
                       horizons: List[int]):
    """
    One row per (target, horizon) with overall MSE/R² and per-horizon MSE.
    Plus one row per (target, output_dim) for per-joint breakdowns.
    """
    rows = []
    # Overall + horizon rows
    for r in results:
        for h in horizons:
            rows.append({
                "target": r.target_name,
                "scope": "overall",
                "horizon": h,
                "n_train": r.n_train_samples,
                "n_test": r.n_test_samples,
                "n_features": r.n_features,
                "n_outputs": r.n_outputs,
                "lambda": f"{r.chosen_lambda:g}",
                "mse": f"{r.rollout_mse.get(h, float('nan')):.6g}",
                "r2": f"{r.test_r2:.6f}" if h == 1 else NAN_SENTINEL,
            })
        # Per-output rows (horizon=1 only)
        for d in range(r.n_outputs):
            rows.append({
                "target": r.target_name,
                "scope": f"output_{d}",
                "horizon": 1,
                "n_train": r.n_train_samples,
                "n_test": r.n_test_samples,
                "n_features": r.n_features,
                "n_outputs": r.n_outputs,
                "lambda": f"{r.chosen_lambda:g}",
                "mse": f"{r.test_mse_per_output[d]:.6g}",
                "r2": f"{r.test_r2_per_output[d]:.6f}",
            })

    fieldnames = ["target", "scope", "horizon", "n_train", "n_test",
                   "n_features", "n_outputs", "lambda", "mse", "r2"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(out_path: str, robot_name: str, run_id: str,
                        feat: FeatureSpec, results: List[TrainResult],
                        train_clip_ids: List[str], test_clip_ids: List[str]):
    summary = {
        "robot": robot_name,
        "run_id": run_id,
        "feature_spec": asdict(feat),
        "feature_label": feat.label(),
        "train_clip_count": len(train_clip_ids),
        "test_clip_count": len(test_clip_ids),
        "targets": {
            r.target_name: {
                "chosen_lambda": r.chosen_lambda,
                "test_mse": r.test_mse,
                "test_r2": r.test_r2,
                "n_features": r.n_features,
                "n_outputs": r.n_outputs,
                "rollout_mse": {str(h): v for h, v in r.rollout_mse.items()},
            }
            for r in results
        },
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

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
    return [float(x.strip()) for x in s.split(",")]


def _parse_horizons(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",")]


def main():
    p = argparse.ArgumentParser(description="PRC readout training (per-robot)")
    p.add_argument("--bundle-dir", required=True)
    p.add_argument("--robot", required=True)
    p.add_argument(
        "--features", default="node_vel",
        help="Reservoir feature level (one of: strain, strain_rate, "
             "strain_accel, node_vel, node_acc). Pick one physical level. "
             "For velocity targets use node_vel or strain_rate; "
             "for acceleration targets use node_acc or strain_accel. "
             "Run separately for each level to build the ablation table. "
             "Default: node_vel.",
    )
    p.add_argument(
        "--targets",
        default="base_lin_vel,base_ang_vel,base_lin_acc,base_ang_acc,qvel,qacc",
        help="Comma-separated target names. Supported: "
             "base_lin_vel, base_ang_vel, base_lin_acc, base_ang_acc, "
             "qvel, qacc, base_vel, base_acc, tau, foot_force",
    )
    p.add_argument("--lambdas", default="auto",
                   help="Comma-separated lambda grid, or 'auto' for the default")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS),
                   help="Comma-separated rollout horizons in trajectory frames")
    p.add_argument("--skip-seconds", type=float, default=2.0,
                   help="Drop the first N seconds of each clip (transient).")
    p.add_argument("--target-fps", type=float, default=50.0,
                   help="Subsample features and targets to this frame rate "
                        "before training. Default 50 Hz — plenty of temporal "
                        "density for body-velocity prediction, and keeps "
                        "memory tractable on high-rate IMU data (default "
                        "trajectories at 500 Hz would otherwise need ~6 GB "
                        "of RAM for the kernel matrix).")
    p.add_argument("--split-mode", choices=["in-clip", "multi-clip"],
                   default="in-clip",
                   help="in-clip: 70/30 split within each clip (eliminates "
                        "between-clip distribution shift, easier task). "
                        "multi-clip: use manifest split=train/test labels "
                        "(harder generalization test).")
    p.add_argument("--in-clip-train-fraction", type=float, default=0.7,
                   help="For --split-mode in-clip: fraction of each clip's "
                        "frames used for training (default 0.7)")
    p.add_argument("--cv-mode", choices=["grouped", "temporal"], default=None,
                   help="CV fold strategy: 'grouped' = leave-one-clip-out "
                        "(tests cross-clip generalization; needs ≥3 clips), "
                        "'temporal' = contiguous temporal folds within the "
                        "concatenated train set (matches in-clip test split; "
                        "works with a single clip). "
                        "Default: 'temporal' for in-clip split mode, "
                        "'grouped' for multi-clip split mode.")
    p.add_argument("--run-id", default=None,
                   help="Override the auto-generated run id (default: timestamp)")
    args = p.parse_args()

    feat = _parse_features(args.features)
    target_names = [t.strip() for t in args.targets.split(",") if t.strip()]
    target_specs = _list_target_specs(target_names)
    lambdas = _parse_lambdas(args.lambdas)
    horizons = _parse_horizons(args.horizons)
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    # Default CV mode: temporal for in-clip (test is last 30% of same clip),
    # grouped for multi-clip (test is held-out clips).
    cv_mode = args.cv_mode or (
        "temporal" if args.split_mode == "in-clip" else "grouped"
    )

    # Physical-level sanity check: warn if features and targets are mismatched.
    # This is purely informational — the readout will still try to learn the
    # mapping. The match exists for the ablation story.
    feat_level = FEATURE_LEVELS[feat.level]
    velocity_targets = {"qvel", "base_lin_vel", "base_ang_vel", "base_vel", "tau"}
    accel_targets = {"qacc", "base_lin_acc", "base_ang_acc", "base_acc"}
    mismatched = []
    for ts in target_specs:
        if feat_level == "velocity" and ts.name in accel_targets:
            mismatched.append(f"{ts.name} (acc) ≠ {feat.level} (vel)")
        elif feat_level == "acceleration" and ts.name in velocity_targets:
            mismatched.append(f"{ts.name} (vel) ≠ {feat.level} (acc)")
        elif feat_level == "kinematic" and ts.name in (velocity_targets | accel_targets):
            mismatched.append(f"{ts.name} (deriv) ≠ {feat.level} (kinematic)")

    print(f"=== PRC readout training: {args.robot} / {run_id} ===")
    print(f"  features:        {feat.label()}  ({feat_level} level)")
    print(f"  targets:         {[t.name for t in target_specs]}")
    print(f"  lambdas:         {lambdas}")
    print(f"  horizons:        {horizons}")
    print(f"  skip_seconds:    {args.skip_seconds}")
    print(f"  target_fps:      {args.target_fps}")
    print(f"  split_mode:      {args.split_mode}"
          + (f" ({args.in_clip_train_fraction:.0%}/"
             f"{1-args.in_clip_train_fraction:.0%})"
             if args.split_mode == "in-clip" else ""))
    print(f"  cv_mode:         {cv_mode}")
    if mismatched:
        print(f"  ⚠ physical level mismatches (feature→target):")
        for m in mismatched:
            print(f"      {m}")
        print(f"    The readout will still train — this is just an FYI for ablations.")
    print()

    out_root = os.path.join(args.bundle_dir, args.robot, "training", run_id)
    os.makedirs(out_root, exist_ok=True)
    pred_dir = os.path.join(out_root, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    # ---------- collect data ----------
    print("Stage: collect data")
    X, Ys, group_ids, train_mask, test_mask, train_clip_ids, test_clip_ids = collect_data(
        args.bundle_dir, args.robot, feat, target_specs,
        split_mode=args.split_mode,
        in_clip_train_fraction=args.in_clip_train_fraction,
        skip_seconds=args.skip_seconds,
        target_fps=args.target_fps,
    )

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise RuntimeError(
            f"After splitting, train rows={train_mask.sum()}, "
            f"test rows={test_mask.sum()}. Need both > 0."
        )

    X_tr = X[train_mask]
    X_te = X[test_mask]
    g_tr = group_ids[train_mask]
    g_te = group_ids[test_mask]
    Y_tr_dict = {k: v[train_mask] for k, v in Ys.items()}
    Y_te_dict = {k: v[test_mask] for k, v in Ys.items()}

    # Save group ids and a metadata sidecar for downstream tools (integrator)
    np.savez(
        os.path.join(pred_dir, "_split_meta.npz"),
        test_group_ids=g_te.astype(np.int32),
        train_group_ids=g_tr.astype(np.int32),
    )

    # ---------- train each target ----------
    print("\nStage: train + evaluate")
    results: List[TrainResult] = []
    for ts in target_specs:
        Y_tr = Y_tr_dict[ts.name]
        Y_te = Y_te_dict[ts.name]
        result, W, b, Y_pred = train_one_target(
            ts.name, X_tr, Y_tr, g_tr, X_te, Y_te,
            lambdas=lambdas, n_folds=args.n_folds, horizons=horizons,
            cv_mode=cv_mode,
        )
        results.append(result)
        save_readout(out_root, ts.name, W, b, result, feat)
        save_predictions(pred_dir, ts.name, Y_te, Y_pred)

    # ---------- write summary outputs ----------
    metrics_path = os.path.join(out_root, "metrics.csv")
    write_metrics_csv(metrics_path, results, horizons)
    summary_path = os.path.join(out_root, "summary.json")
    write_summary_json(summary_path, args.robot, run_id, feat, results,
                        train_clip_ids, test_clip_ids)

    print(f"\n=== Done ===")
    print(f"  results saved to {out_root}")
    print(f"    readout_<target>.npz       — trained weights per target")
    print(f"    metrics.csv                — full metric table")
    print(f"    summary.json               — high-level summary")
    print(f"    predictions/<target>.npz   — y_true and y_pred for plotting")
    print(f"\nNext: plot results with plot_readout_results.py")


if __name__ == "__main__":
    main()
