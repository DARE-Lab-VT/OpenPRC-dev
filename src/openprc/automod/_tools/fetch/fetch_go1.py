"""
Go1 bundle ingestion.

Pipeline:
  1. Convert MuJoCo Menagerie's go1.xml (MJCF) → go1/urdf/go1.urdf + meshes.
  2. Write go1/metadata.json.
  3. Parse each legkilo .bag file: extract /high_state and /state_SDK,
     remap joints from Unitree-SDK order to URDF order, reorder quaternions
     from xyzw to wxyz, normalize foot forces per-sequence, slice into 30s
     clips, write each clip as a schema-compliant HDF5 via write_trajectory.
  4. Emit a per-robot manifest fragment that fetch_all.py will merge into
     the top-level manifest.json.

Usage:
    # Step 1: convert the Menagerie MJCF to URDF (run once)
    python fetch_go1.py --convert-urdf \\
        --menagerie-dir /path/to/mujoco_menagerie \\
        --bundle-dir /path/to/robot_bundle

    # Step 2: ingest bags (run per-bag or all at once)
    python fetch_go1.py --ingest-bag corridor.bag \\
        --bundle-dir /path/to/robot_bundle

Dependencies: rosbags (pip install rosbags). No ROS install required.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # so _tools.write_trajectory imports
sys.path.insert(0, HERE)

from write_trajectory import write_trajectory  # noqa: E402
from go1_transforms import (  # noqa: E402
    GO1_URDF_JOINT_NAMES,
    GO1_CONTACT_NAMES,
    remap_legkilo_joints_to_urdf,
    reorder_quaternion_xyzw_to_wxyz,
    remap_foot_forces_to_urdf,
    normalize_foot_force_per_sequence,
    compute_contact_flags,
    slice_into_clips,
    interpolate_base_pose,
)


# -----------------------------------------------------------------------------
# Constants specific to the Go1 + legkilo combo (non-transform)
# -----------------------------------------------------------------------------

# Legkilo bag sequence defaults
CLIP_DURATION_S = 30.0
TEST_CLIPS_PER_SEQUENCE = 2  # last N clips per sequence go to test split

# Source identity
SOURCE_NAME = "legkilo"
SOURCE_URL = "https://github.com/ouguangjun/legkilo-dataset"
SOURCE_CITATION = (
    "Ou, G., Li, D., Li, H. (2024). Leg-KILO: Robust Kinematic-Inertial-Lidar "
    "Odometry for Dynamic Legged Robots. IEEE RA-L 9(10):8194-8201."
)
SOURCE_LICENSE = "MIT"  # TODO verify from repo LICENSE before publication


# -----------------------------------------------------------------------------
# Pure-function transforms: see go1_transforms.py for implementations.
# This section intentionally left as a placeholder header.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Bag I/O (requires rosbags; isolated so the pure functions above are testable)
# -----------------------------------------------------------------------------

def _lazy_import_rosbags():
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    except ImportError as e:
        raise ImportError(
            "rosbags is required to parse legkilo .bag files. "
            "Install with: pip install rosbags"
        ) from e
    return AnyReader, Stores, get_types_from_msg, get_typestore


# Legkilo custom message definitions — these aren't in standard ROS, so we
# register them at runtime. The layout comes from the legkilo repo
# (unitree_legged_msgs/msg/*.msg). We inline them here so the parser doesn't
# need a ROS workspace.
LEGKILO_MSG_DEFINITIONS = {
    "unitree_legged_msgs/msg/Cartesian": """
float32 x
float32 y
float32 z
""",
    "unitree_legged_msgs/msg/IMU": """
float32[4] quaternion
float32[3] gyroscope
float32[3] accelerometer
float32[3] rpy
int8 temperature
""",
    "unitree_legged_msgs/msg/MotorState": """
uint8 mode
float32 q
float32 dq
float32 ddq
float32 tauEst
float32 q_raw
float32 dq_raw
float32 ddq_raw
int8 temperature
uint32[2] reserve
""",
    "unitree_legged_msgs/msg/BmsState": """
uint8 version_h
uint8 version_l
uint8 bms_status
uint8 SOC
int32 current
uint16 cycle
int8[2] BQ_NTC
int8[2] MCU_NTC
uint16[10] cell_vol
""",
    "unitree_legged_msgs/msg/HighState": """
std_msgs/Header header
uint8[2] head
uint8 levelFlag
uint8 frameReserve
uint32[2] SN
uint32[2] version
uint16 bandWidth
unitree_legged_msgs/IMU imu
unitree_legged_msgs/MotorState[20] motorState
unitree_legged_msgs/BmsState bms
int16[4] footForce
int16[4] footForceEst
uint8 mode
float32 progress
uint8 gaitType
float32 footRaiseHeight
float32[3] position
float32 bodyHeight
float32[3] velocity
float32 yawSpeed
float32[4] rangeObstacle
unitree_legged_msgs/Cartesian[4] footPosition2Body
unitree_legged_msgs/Cartesian[4] footSpeed2Body
uint8[40] wirelessRemote
uint32 reserve
uint32 crc
""",
}


def parse_legkilo_bag(bag_path: str) -> dict:
    """
    Read a legkilo .bag file and return a dict of numpy arrays:
        {
          "time":       (T,) float64 seconds from first /high_state message
          "qpos":       (T, 12) joint positions, URDF order
          "qvel":       (T, 12) joint velocities, URDF order
          "tau":        (T, 12) estimated joint torques, URDF order
          "base_pose":  (T, 7) from /state_SDK, interpolated to /high_state times
          "foot_force_raw":     (T, 4) int16 ADC, URDF leg order
          "foot_force_est_raw": (T, 4) int16 ADC, URDF leg order
          "base_imu_quat_wxyz": (T, 4) from /high_state IMU
        }
    """
    AnyReader, Stores, get_types_from_msg, get_typestore = _lazy_import_rosbags()

    typestore = get_typestore(Stores.ROS1_NOETIC)
    # Register legkilo's custom message types
    for msg_name, msg_def in LEGKILO_MSG_DEFINITIONS.items():
        typestore.register(get_types_from_msg(msg_def, msg_name))

    high_state_times = []
    motor_q = []       # (T, 20)
    motor_dq = []      # (T, 20)
    motor_tau = []     # (T, 20)
    foot_force = []    # (T, 4)
    foot_force_est = []  # (T, 4)
    imu_quat = []      # (T, 4) xyzw

    state_sdk_times = []
    state_sdk_pos = []   # (T', 3)
    state_sdk_quat = []  # (T', 4) xyzw

    with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:
        connections = {c.topic: c for c in reader.connections}
        needed = ["/high_state", "/state_SDK"]
        missing = [t for t in needed if t not in connections]
        if missing:
            raise ValueError(
                f"Bag is missing required topics: {missing}. "
                f"Found topics: {list(connections)}"
            )

        for conn, timestamp_ns, raw in reader.messages(
            connections=[connections[t] for t in needed]
        ):
            msg = reader.deserialize(raw, conn.msgtype)
            t_s = timestamp_ns * 1e-9
            if conn.topic == "/high_state":
                high_state_times.append(t_s)
                qs = np.array([m.q for m in msg.motorState], dtype=np.float32)
                dqs = np.array([m.dq for m in msg.motorState], dtype=np.float32)
                taus = np.array([m.tauEst for m in msg.motorState], dtype=np.float32)
                motor_q.append(qs)
                motor_dq.append(dqs)
                motor_tau.append(taus)
                foot_force.append(np.asarray(msg.footForce, dtype=np.float32))
                foot_force_est.append(
                    np.asarray(msg.footForceEst, dtype=np.float32)
                )
                imu_quat.append(np.asarray(msg.imu.quaternion, dtype=np.float32))
            elif conn.topic == "/state_SDK":
                state_sdk_times.append(t_s)
                p = msg.pose.pose.position
                state_sdk_pos.append(np.array([p.x, p.y, p.z], dtype=np.float32))
                q = msg.pose.pose.orientation
                state_sdk_quat.append(
                    np.array([q.x, q.y, q.z, q.w], dtype=np.float32)
                )

    if not high_state_times:
        raise ValueError(f"No /high_state messages found in {bag_path}")

    hs_t = np.asarray(high_state_times, dtype=np.float64)
    t0 = hs_t[0]
    hs_t = hs_t - t0  # seconds from start

    motor_q = np.stack(motor_q)              # (T, 20)
    motor_dq = np.stack(motor_dq)
    motor_tau = np.stack(motor_tau)
    foot_force = np.stack(foot_force)        # (T, 4) legkilo order
    foot_force_est = np.stack(foot_force_est)
    imu_quat_xyzw = np.stack(imu_quat)

    # Joint remap to URDF order (takes first 12 SDK slots and reorders)
    qpos = remap_legkilo_joints_to_urdf(motor_q)
    qvel = remap_legkilo_joints_to_urdf(motor_dq)
    tau = remap_legkilo_joints_to_urdf(motor_tau)

    # Contact remap to URDF leg order
    foot_force = remap_foot_forces_to_urdf(foot_force)
    foot_force_est = remap_foot_forces_to_urdf(foot_force_est)

    # IMU quat: xyzw → wxyz
    imu_quat_wxyz = reorder_quaternion_xyzw_to_wxyz(imu_quat_xyzw)

    # Interpolate /state_SDK onto /high_state timebase
    base_pose = None
    if state_sdk_times:
        sdk_t = np.asarray(state_sdk_times, dtype=np.float64) - t0
        sdk_pos = np.stack(state_sdk_pos)
        sdk_quat_xyzw = np.stack(state_sdk_quat)
        base_pose = interpolate_base_pose(sdk_t, sdk_pos, sdk_quat_xyzw, hs_t)

    return {
        "time": hs_t,
        "qpos": qpos,
        "qvel": qvel,
        "tau": tau,
        "base_pose": base_pose,
        "foot_force_raw": foot_force,
        "foot_force_est_raw": foot_force_est,
        "base_imu_quat_wxyz": imu_quat_wxyz,
    }


# -----------------------------------------------------------------------------
# High-level ingest: bag → clips → HDF5s + manifest fragment
# -----------------------------------------------------------------------------

def ingest_bag(
    bag_path: str,
    bundle_dir: str,
    sequence_name: Optional[str] = None,
    clip_duration_s: float = CLIP_DURATION_S,
    test_clips: int = TEST_CLIPS_PER_SEQUENCE,
) -> list[dict]:
    """
    Parse one bag and emit trajectory HDF5 clips. Returns a list of manifest
    trajectory entries (not yet merged into manifest.json).
    """
    # Expand ~ and normalize before handing off to rosbags, and fail with a
    # clear message if the path is wrong (rosbags' own error buries the cause).
    bag_path = os.path.abspath(os.path.expanduser(bag_path))
    if not os.path.exists(bag_path):
        parent = os.path.dirname(bag_path)
        hint = ""
        if os.path.isdir(parent):
            bags = sorted(f for f in os.listdir(parent) if f.lower().endswith(".bag"))
            if bags:
                hint = f"\n  .bag files found in {parent}:\n    " + "\n    ".join(bags)
            else:
                hint = f"\n  No .bag files in {parent}"
        raise FileNotFoundError(
            f"Bag not found: {bag_path}{hint}"
        )
    if not bag_path.lower().endswith(".bag"):
        print(f"  warning: {bag_path} does not end in .bag; proceeding anyway")

    if sequence_name is None:
        sequence_name = Path(bag_path).stem  # 'corridor', 'slope', etc.

    print(f"Parsing bag: {bag_path} (sequence={sequence_name})")
    parsed = parse_legkilo_bag(bag_path)
    T = parsed["time"].shape[0]
    print(f"  {T} samples, {parsed['time'][-1]:.1f}s total")

    # Per-sequence foot-force normalization (from the full sequence, not per-clip)
    ff_all = parsed["foot_force_raw"]
    ff_norm, ff_scale = normalize_foot_force_per_sequence(ff_all)
    ff_est_all = parsed["foot_force_est_raw"]
    ff_est_norm, ff_est_scale = normalize_foot_force_per_sequence(ff_est_all)
    contact_flags_all = compute_contact_flags(ff_norm)

    scale_desc = (
        f"foot_force divided by max abs value in sequence "
        f"(raw max={ff_scale:.0f} ADC, est max={ff_est_scale:.0f} ADC). "
        f"Negative raw readings clipped to 0 before scaling."
    )

    clips = slice_into_clips(parsed["time"], clip_duration_s)
    print(f"  {len(clips)} clips of {clip_duration_s}s each")

    out_dir = os.path.join(bundle_dir, "go1", "trajectories")
    os.makedirs(out_dir, exist_ok=True)

    manifest_entries = []
    n_clips = len(clips)
    for clip_idx, (start, end) in enumerate(clips):
        clip_id = f"{sequence_name}_{clip_idx:03d}"
        out_path = os.path.join(out_dir, f"{clip_id}.h5")

        t_clip = parsed["time"][start:end] - parsed["time"][start]

        split = "test" if clip_idx >= n_clips - test_clips else "train"

        write_trajectory(
            out_path=out_path,
            robot_name="go1",
            trajectory_id=clip_id,
            joint_names=GO1_URDF_JOINT_NAMES,
            contact_names=GO1_CONTACT_NAMES,
            time=t_clip,
            qpos=parsed["qpos"][start:end],
            qvel=parsed["qvel"][start:end],
            tau=parsed["tau"][start:end],
            base_pose=(parsed["base_pose"][start:end]
                       if parsed["base_pose"] is not None else None),
            contact_foot_force=ff_norm[start:end],
            contact_foot_force_est=ff_est_norm[start:end],
            contact_flags=contact_flags_all[start:end],
            contact_force_units="normalized_per_sequence",
            contact_force_scale=scale_desc,
            source=SOURCE_NAME,
            source_type="real_hardware",
            source_url=SOURCE_URL,
            source_citation=SOURCE_CITATION,
            source_license=SOURCE_LICENSE,
            notes=(
                f"Sequence '{sequence_name}', clip {clip_idx+1}/{n_clips}. "
                f"Split: {split}. Base pose from /state_SDK (Unitree built-in "
                f"Kalman filter; not MoCap ground truth)."
            ),
            overwrite=True,
        )

        # Compute manifest fields
        with_contact_flags = contact_flags_all is not None
        signals = ["qpos", "qvel", "tau", "base_pose",
                   "contact/foot_force", "contact/foot_force_est"]
        if with_contact_flags:
            signals.append("contact/contact_flags")

        manifest_entries.append({
            "id": clip_id,
            "path": f"go1/trajectories/{clip_id}.h5",
            "duration_s": float(t_clip[-1] - t_clip[0]),
            "frequency_hz": 1.0 / float(np.median(np.diff(t_clip))),
            "n_timesteps": int(end - start),
            "source": SOURCE_NAME,
            "source_type": "real_hardware",
            "source_license": SOURCE_LICENSE,
            "signals_present": signals,
            "task": f"{sequence_name} sequence",
            "split": split,
        })
        print(f"  wrote {clip_id}.h5 ({split})")

    return manifest_entries


# -----------------------------------------------------------------------------
# Sub-commands: URDF conversion + metadata
# -----------------------------------------------------------------------------

def cmd_convert_urdf(menagerie_dir: str, bundle_dir: str):
    """Convert Menagerie's unitree_go1/go1.xml to go1/urdf/go1.urdf."""
    from mjcf_to_urdf import convert_mjcf_to_urdf

    mjcf_path = os.path.join(menagerie_dir, "unitree_go1", "go1.xml")
    urdf_out = os.path.join(bundle_dir, "go1", "urdf", "go1.urdf")
    mesh_dir = os.path.join(bundle_dir, "go1", "urdf", "meshes")

    print(f"Converting {mjcf_path} → {urdf_out}")
    convert_mjcf_to_urdf(mjcf_path, urdf_out, mesh_dir=mesh_dir)

    # Write metadata.json
    metadata = {
        "schema_version": "0.1.0",
        "robot_name": "go1",
        "display_name": "Unitree Go1",
        "urdf_path": "urdf/go1.urdf",
        "mesh_dir": "urdf/meshes",
        "floating_base": True,
        "n_dof_actuated": 12,
        "n_dof_total": 18,
        "joint_names": GO1_URDF_JOINT_NAMES,
        "contact_names": GO1_CONTACT_NAMES,
        "urdf_source": {
            "origin": "mujoco_menagerie",
            "origin_url": (
                "https://github.com/google-deepmind/mujoco_menagerie/"
                "tree/main/unitree_go1"
            ),
            "conversion_tool": "mjcf_urdf_simple_converter",
            "conversion_notes": (
                "Kinematics and meshes preserved. Dynamics parameters "
                "(damping, armature, solver settings) are MuJoCo-specific "
                "and not represented in URDF."
            ),
            "license": "Apache-2.0",
        },
        "mass_kg": 12.0,
        "nominal_height_m": 0.30,
        "notes": (
            "Floating base with 6-DoF free joint at root. /qpos covers the "
            "12 leg joints only; /base_pose carries the floating base state. "
            "Trajectory qpos columns follow joint_names order above; this is "
            "the URDF order, which differs from the raw Unitree SDK "
            "motorState[] indexing (FR,FL,RL,RR → FR,FL,RR,RL)."
        ),
    }
    metadata_path = os.path.join(bundle_dir, "go1", "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {metadata_path}")


def cmd_ingest_bag(bag_path: str, bundle_dir: str, sequence_name: Optional[str]):
    entries = ingest_bag(bag_path, bundle_dir, sequence_name)
    # Write/update a per-robot manifest fragment
    fragment_path = os.path.join(bundle_dir, "go1", "_manifest_fragment.json")
    if os.path.exists(fragment_path):
        with open(fragment_path) as f:
            fragment = json.load(f)
    else:
        fragment = {"trajectories": []}
    # Replace any existing entries with the same ids
    existing_ids = {e["id"] for e in entries}
    fragment["trajectories"] = [
        e for e in fragment["trajectories"] if e["id"] not in existing_ids
    ]
    fragment["trajectories"].extend(entries)
    fragment["trajectories"].sort(key=lambda e: e["id"])
    with open(fragment_path, "w") as f:
        json.dump(fragment, f, indent=2)
    print(f"Appended {len(entries)} entries to {fragment_path}")
    print(f"Total trajectories now: {len(fragment['trajectories'])}")


def main():
    parser = argparse.ArgumentParser(description="Go1 bundle ingestion")
    parser.add_argument("--bundle-dir", required=True,
                        help="Path to the robot_bundle directory")

    sub = parser.add_subparsers(dest="command", required=True)

    p_convert = sub.add_parser("convert-urdf",
                                help="Convert Menagerie MJCF to URDF")
    p_convert.add_argument("--menagerie-dir", required=True,
                            help="Path to a clone of google-deepmind/mujoco_menagerie")

    p_ingest = sub.add_parser("ingest-bag",
                               help="Parse a legkilo .bag into trajectory clips")
    p_ingest.add_argument("bag_path", help="Path to a .bag file")
    p_ingest.add_argument("--sequence-name", default=None,
                           help="Override sequence name (default: bag filename)")

    args = parser.parse_args()

    if args.command == "convert-urdf":
        cmd_convert_urdf(args.menagerie_dir, args.bundle_dir)
    elif args.command == "ingest-bag":
        cmd_ingest_bag(args.bag_path, args.bundle_dir, args.sequence_name)


if __name__ == "__main__":
    main()
