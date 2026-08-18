"""
Runtime: turn per-link reservoirs + a trajectory into a DEMLAT simulation.

Inputs:
  - Bundle root with <robot>/reservoir/<link>.npz files (from batch_preprocess.py)
  - A trajectory HDF5 (already validated by Stage-1 tooling)

Output:
  - <bundle>/<robot>/reservoir_sims/<trajectory_id>/
      input/    config.json, geometry.h5, signals.h5
      output/   simulation.h5  (filled in by Engine.run)

Pipeline:
  1. Load robot metadata, reservoir _index.json, and trajectory.
  2. Run forward kinematics on the trajectory → per-link world transforms (T, 4, 4).
  3. For each link's reservoir:
        - Transform interior nodes to world frame at t=0 (their initial position).
        - Add nodes to DEMLAT setup with global indexing offset.
        - For each anchor, generate the (T_phys, 3) world-frame target trajectory
          via composed transform: T_world_link(t) @ mesh_origin @ p_local.
        - Add bars (springs).
        - Register one position actuator per anchor.
  4. Set physics + simulation params.
  5. Save inputs to disk and run Engine(BarHingeModel, backend='cuda').

CLI:
    python reservoir_to_demlat.py \\
        --bundle-dir . --robot go1 --trajectory corridor_000 \\
        [--physics-dt 0.005] [--save-dt 0.02] [--duration <full>] \\
        [--no-gravity] [--damping-scale 1.0]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # _tools/ on path
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "kinematics"))

from mesh_to_reservoir import load_reservoir  # noqa: E402


# -----------------------------------------------------------------------------
# Pure-function core (testable without DEMLAT or h5py)
# -----------------------------------------------------------------------------

def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 homogeneous transform to a (N, 3) point cloud.
    Returns (N, 3) transformed points.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected (N, 3) points, got {points.shape}")
    if T.shape != (4, 4):
        raise ValueError(f"expected (4, 4) transform, got {T.shape}")
    R = T[:3, :3]
    t = T[:3, 3]
    return (points @ R.T) + t


def compose_anchor_signal(
    anchor_local: np.ndarray,           # (K, 3) link-local rest positions
    mesh_origin_in_link: np.ndarray,    # (4, 4)
    T_world_link_seq: np.ndarray,       # (T_traj, 4, 4)
) -> np.ndarray:
    """
    Compute world-frame anchor positions over time.

    For each timestep t and each anchor k:
        p_world(t, k) = T_world_link(t) @ mesh_origin @ p_anchor_local(k)

    Returns (T_traj, K, 3) float32.
    """
    if anchor_local.ndim != 2 or anchor_local.shape[1] != 3:
        raise ValueError(f"expected (K, 3) anchors, got {anchor_local.shape}")
    if mesh_origin_in_link.shape != (4, 4):
        raise ValueError(f"expected (4, 4) mesh_origin, got {mesh_origin_in_link.shape}")
    if T_world_link_seq.ndim != 3 or T_world_link_seq.shape[1:] != (4, 4):
        raise ValueError(
            f"expected (T_traj, 4, 4) transform sequence, "
            f"got {T_world_link_seq.shape}"
        )

    K = anchor_local.shape[0]
    T_traj = T_world_link_seq.shape[0]

    # First: transform anchors from mesh frame to link frame (constant)
    anchors_in_link = transform_points(
        anchor_local.astype(np.float64), mesh_origin_in_link.astype(np.float64)
    )  # (K, 3)

    # Then: apply per-timestep T_world_link (vectorized)
    # T_world_link_seq is (T, 4, 4); we want world positions for K anchors at each T.
    # Reformulate as batched matmul: anchors_in_link (K, 3) → world via T_seq.
    # World position = T[:3, :3] @ p + T[:3, 3]
    R_seq = T_world_link_seq[:, :3, :3]   # (T, 3, 3)
    t_seq = T_world_link_seq[:, :3, 3]    # (T, 3)

    # einsum: (T, 3, 3) @ (K, 3).T → (T, K, 3)
    rotated = np.einsum("tij,kj->tki", R_seq, anchors_in_link)
    out = rotated + t_seq[:, None, :]   # broadcast (T, 1, 3) + (T, K, 3)
    return out.astype(np.float32)


def upsample_signal(
    signal: np.ndarray, time_in: np.ndarray, dt_out: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linearly interpolate a per-anchor (T_in, 3) signal from its native
    sampling times to a uniform output grid at step `dt_out`.

    DEMLAT's actuation lookup is `idx = round(t_curr / dt_base)`, so the
    output array's row `i` corresponds to time `i * dt_out`.

    Returns (signal_out, t_out).
    """
    if signal.ndim != 2 or signal.shape[1] != 3:
        raise ValueError(f"expected (T, 3) signal, got {signal.shape}")
    if len(time_in) != signal.shape[0]:
        raise ValueError(
            f"signal has {signal.shape[0]} steps but time_in has "
            f"{len(time_in)} entries"
        )

    t0 = float(time_in[0])
    tF = float(time_in[-1])
    # Add a tiny epsilon before floor to avoid truncating the endpoint when
    # (tF - t0) / dt_out is integer in real arithmetic but slightly less in
    # floating-point. e.g. 0.3 / 0.05 = 5.999... → floor → 5 → off-by-one.
    n_out = int(np.floor((tF - t0) / dt_out + 1e-9)) + 1
    if n_out < 2:
        raise ValueError(f"upsample produces only {n_out} samples; "
                         f"check duration vs dt_out")
    t_out = t0 + np.arange(n_out, dtype=np.float64) * dt_out

    # Per-axis linear interpolation
    out = np.empty((n_out, 3), dtype=np.float32)
    for axis in range(3):
        out[:, axis] = np.interp(t_out, time_in, signal[:, axis]).astype(np.float32)
    return out, t_out


def offset_edges(edges: np.ndarray, offset: int) -> np.ndarray:
    """Shift node indices in an edge list by a global offset."""
    if len(edges) == 0:
        return edges.copy()
    out = edges.copy().astype(np.int32)
    out += int(offset)
    return out


# -----------------------------------------------------------------------------
# I/O (DEMLAT and h5py dependent)
# -----------------------------------------------------------------------------

def _load_demlat():
    try:
        from openprc.demlat import SimulationSetup, Simulation, BarHingeModel, Engine
    except ImportError as e:
        raise ImportError(
            "openprc.demlat is required to run reservoir simulations. "
            "Install or activate the openprc package."
        ) from e
    return SimulationSetup, Simulation, BarHingeModel, Engine


def _load_trajectory(bundle_dir: str, robot_name: str, trajectory_id: str):
    """Read trajectory HDF5 and return (qpos, base_pose_or_None, time, joint_names)."""
    import h5py
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    robot_entry = manifest["robots"][robot_name]
    match = next(t for t in robot_entry["trajectories"] if t["id"] == trajectory_id)
    h5_path = os.path.join(bundle_dir, match["path"])
    with h5py.File(h5_path, "r") as f:
        joint_names = list(f.attrs["joint_names"])
        time = f["time"][:].astype(np.float64)
        qpos = f["qpos"][:].astype(np.float32)
        base_pose = f["base_pose"][:].astype(np.float32) if "base_pose" in f else None
    return qpos, base_pose, time, joint_names


def _load_reservoir_index(bundle_dir: str, robot_name: str) -> dict:
    """Read the reservoir index produced by batch_preprocess.py."""
    index_path = os.path.join(bundle_dir, robot_name, "reservoir", "_index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Reservoir index not found at {index_path}. Run "
            f"batch_preprocess.py first."
        )
    with open(index_path) as f:
        return json.load(f)


def _build_link_to_reservoir_mapping(
    bundle_dir: str, robot_name: str, fk_link_names: List[str],
) -> Dict[str, str]:
    """
    Map URDF link names (from FK) to reservoir .npz file paths.

    A URDF link may have several visuals; batch_preprocess.py disambiguates
    them as `<link>__1`, `<link>__2`, etc. Here we pick the primary
    (no suffix) one for the runtime, and warn on duplicates.
    """
    reservoir_dir = os.path.join(bundle_dir, robot_name, "reservoir")
    available = {
        Path(f).stem: os.path.join(reservoir_dir, f)
        for f in os.listdir(reservoir_dir)
        if f.endswith(".npz")
    }

    mapping: Dict[str, str] = {}
    skipped: List[str] = []
    duplicates: List[str] = []
    for link in fk_link_names:
        if link in available:
            mapping[link] = available[link]
            # Note any extra disambiguated versions
            extras = [n for n in available if n.startswith(link + "__")]
            if extras:
                duplicates.append(f"{link} (also has: {extras})")
        else:
            skipped.append(link)

    if skipped:
        print(f"  warning: {len(skipped)} FK links have no reservoir "
              f".npz; they'll be skipped: {skipped[:5]}"
              f"{'...' if len(skipped) > 5 else ''}")
    if duplicates:
        print(f"  note: {len(duplicates)} links have multiple visuals; "
              f"using primary only: {duplicates[:3]}"
              f"{'...' if len(duplicates) > 3 else ''}")

    return mapping


# -----------------------------------------------------------------------------
# Main builder
# -----------------------------------------------------------------------------

def build_simulation(
    bundle_dir: str,
    robot_name: str,
    trajectory_id: str,
    physics_dt: float = 0.005,
    save_dt: float = 0.02,
    duration: Optional[float] = None,
    gravity: float = -9.81,
    damping_scale: float = 1.0,
):
    """
    Compose a DEMLAT simulation from per-link reservoirs and a trajectory.
    Saves the experiment under <bundle>/<robot>/reservoir_sims/<traj_id>/
    and runs Engine.run(). Returns the experiment path.
    """
    SimulationSetup, Simulation, BarHingeModel, Engine = _load_demlat()

    # Robot + trajectory metadata
    robot_dir = os.path.join(bundle_dir, robot_name)
    metadata_path = os.path.join(robot_dir, "metadata.json")
    with open(metadata_path) as f:
        robot_meta = json.load(f)
    urdf_path = os.path.join(robot_dir, robot_meta["urdf_path"])

    print(f"=== Reservoir simulation: {robot_name} / {trajectory_id} ===")

    qpos, base_pose, time, joint_names = _load_trajectory(
        bundle_dir, robot_name, trajectory_id
    )
    T_traj = len(time)
    traj_duration = float(time[-1] - time[0])
    if duration is None:
        duration = traj_duration
    else:
        duration = min(duration, traj_duration)
    print(f"  trajectory: {T_traj} frames, {traj_duration:.2f}s "
          f"(simulating {duration:.2f}s)")

    # Truncate trajectory to requested duration
    keep_mask = (time - time[0]) <= duration + 1e-9
    qpos = qpos[keep_mask]
    if base_pose is not None:
        base_pose = base_pose[keep_mask]
    time = time[keep_mask]
    T_traj = len(time)
    time = time - time[0]   # rebase to zero

    # Forward kinematics
    print(f"  computing FK...")
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "kinematics"))
    from urdf_fk import RobotFK
    fk = RobotFK(
        urdf_path=urdf_path,
        joint_names=joint_names,
        floating_base=robot_meta["floating_base"],
    )
    fk_seq = fk.compute_sequence(qpos, base_pose)   # {link: (T, 4, 4)}
    print(f"  FK done for {len(fk.link_names)} links × {T_traj} frames")

    # Map FK link names to reservoir .npz paths
    link_to_npz = _build_link_to_reservoir_mapping(
        bundle_dir, robot_name, fk.link_names
    )
    if not link_to_npz:
        raise RuntimeError(
            f"No links have reservoirs. Check {os.path.join(robot_dir, 'reservoir')}"
        )
    print(f"  using reservoirs for {len(link_to_npz)} links")

    # Output experiment path
    exp_dir = os.path.join(robot_dir, "reservoir_sims", trajectory_id)
    setup = SimulationSetup(exp_dir, overwrite=True)

    # Sim params
    setup.set_simulation_params(duration=duration, dt=physics_dt, save_interval=save_dt)
    setup.set_physics(gravity=gravity, damping=0.1, enable_collision=False)

    # ---------------------------------------------------------------
    # Add links (nodes + bars + actuator signals) one at a time, with
    # a running global-index offset.
    # ---------------------------------------------------------------
    global_offset = 0
    actuator_count = 0
    total_nodes = 0
    total_bars = 0
    total_anchors = 0

    for link_name, npz_path in link_to_npz.items():
        reservoir = load_reservoir(npz_path)
        node_pos_local = reservoir["node_positions"].astype(np.float64)
        n_anchors = reservoir["n_anchors"]
        edges = reservoir["edges"]
        rest_lengths = reservoir["rest_lengths"]
        stiffnesses = reservoir["stiffnesses"]
        dampings = reservoir["damping_coefficients"] * damping_scale
        mesh_origin = reservoir["mesh_origin_in_link"].astype(np.float64)

        T_world_link_seq = fk_seq[link_name].astype(np.float64)
        if T_world_link_seq.shape[0] < T_traj:
            raise RuntimeError(
                f"FK produced {T_world_link_seq.shape[0]} frames for {link_name} "
                f"but trajectory has {T_traj}"
            )
        T_world_link_seq = T_world_link_seq[:T_traj]

        # Initial-frame world positions for all nodes (interior + anchors)
        # Use composed transform to get world coords at t=0.
        node_world_t0 = transform_points(
            transform_points(node_pos_local, mesh_origin),
            T_world_link_seq[0],
        )

        # Add nodes — anchors get fixed=False since they're position-actuated
        # (the actuator handles their motion; "fixed" means *no DoF at all*).
        node_mass = float(reservoir.get("base_node_mass", 0.01)) \
            if isinstance(reservoir.get("base_node_mass", None), (int, float)) \
            else 0.01
        for k in range(len(node_world_t0)):
            setup.add_node(node_world_t0[k].tolist(), mass=node_mass, fixed=False)

        # Add bars with offsets
        offset_edges_arr = offset_edges(edges, global_offset)
        for e_i, (a, b) in enumerate(offset_edges_arr.tolist()):
            setup.add_bar(
                node_a=int(a), node_b=int(b),
                stiffness=float(stiffnesses[e_i]),
                damping=float(dampings[e_i]),
                rest_length=float(rest_lengths[e_i]),
            )

        # Anchor signals: composed world-frame trajectory per anchor
        anchor_local = node_pos_local[:n_anchors]
        anchor_world_seq = compose_anchor_signal(
            anchor_local.astype(np.float32),
            mesh_origin.astype(np.float32),
            T_world_link_seq.astype(np.float32),
        )   # (T_traj, n_anchors, 3)

        # Upsample each anchor's signal from trajectory rate to physics rate
        for k in range(n_anchors):
            sig_traj = anchor_world_seq[:, k, :]
            sig_upsampled, _ = upsample_signal(sig_traj, time, physics_dt)
            global_anchor_idx = global_offset + k
            sig_name = f"anchor_{link_name}_{k}"
            setup.add_signal(sig_name, sig_upsampled, dt=physics_dt)
            setup.add_actuator(
                node_idx=global_anchor_idx, signal_name=sig_name,
                type="position", dof=[1, 1, 1],
            )
            actuator_count += 1

        total_nodes += len(node_world_t0)
        total_bars += len(edges)
        total_anchors += n_anchors
        global_offset += len(node_world_t0)

    print(f"  total: {total_nodes} nodes, {total_bars} bars, "
          f"{total_anchors} anchors driving {actuator_count} actuators")

    # Persist setup to disk (geometry.h5, signals.h5, config.json)
    setup.save()
    print(f"  experiment saved to {exp_dir}")

    # Run
    print(f"\n=== Running DEMLAT engine ===")
    sim = Simulation(exp_dir)
    eng = Engine(BarHingeModel, backend="cuda")
    result = eng.run(sim)
    print(f"\nDone. Frames: {result.get('frames', '?')}, "
          f"output at: {result.get('path', '?')}")
    return exp_dir


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compose & run a DEMLAT reservoir simulation from a trajectory"
    )
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--physics-dt", type=float, default=0.005,
                        help="Physics time step (default 5 ms)")
    parser.add_argument("--save-dt", type=float, default=0.02,
                        help="Save interval (default 20 ms = 50 Hz)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Simulate only this many seconds (default: full trajectory)")
    parser.add_argument("--gravity", type=float, default=-9.81)
    parser.add_argument("--no-gravity", action="store_true",
                        help="Set gravity to 0 (overrides --gravity)")
    parser.add_argument("--damping-scale", type=float, default=1.0,
                        help="Multiplier on stored per-spring damping (default 1.0)")
    args = parser.parse_args()

    g = 0.0 if args.no_gravity else args.gravity

    build_simulation(
        bundle_dir=args.bundle_dir,
        robot_name=args.robot,
        trajectory_id=args.trajectory,
        physics_dt=args.physics_dt,
        save_dt=args.save_dt,
        duration=args.duration,
        gravity=g,
        damping_scale=args.damping_scale,
    )


if __name__ == "__main__":
    main()
