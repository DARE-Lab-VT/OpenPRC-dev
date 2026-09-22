"""
Self-test: synthesize a fake Go1 trajectory, build a minimal bundle,
and run the validator end-to-end. No real data; no network.

Run from the bundle root:
    python _tools/selftest.py

Exit 0 on success, 1 on failure.

This test exercises:
  - write_trajectory() on a floating-base quadruped with all optional signals
  - robot metadata.json consistency checks
  - manifest.json schema and cross-file consistency
  - the validator's error-raising paths (by deliberately corrupting a file)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np

# Make the _tools directory importable
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from write_trajectory import write_trajectory
from validate_bundle import validate_bundle


GO1_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
GO1_CONTACT_NAMES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]


def synth_trajectory(T: int = 500, freq: float = 500.0, seed: int = 0):
    """Generate a plausible-shaped Go1 trotting trajectory — not physically accurate,
    just schema-shaped so the writer and validator have something to chew on."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / freq
    t = np.arange(T) * dt

    # Trot-ish: alternating diagonal legs at 2 Hz
    phase = 2 * np.pi * 2.0 * t
    qpos = np.zeros((T, 12), dtype=np.float32)
    # Hip abduction: small constant
    qpos[:, [0, 3, 6, 9]] = 0.0
    # Thigh: oscillating, with diagonal pairs in phase
    diag1 = np.sin(phase)        # FR, RL
    diag2 = np.sin(phase + np.pi)  # FL, RR
    qpos[:, 1] = 0.9 + 0.2 * diag1   # FR_thigh
    qpos[:, 10] = 0.9 + 0.2 * diag1  # RL_thigh
    qpos[:, 4] = 0.9 + 0.2 * diag2   # FL_thigh
    qpos[:, 7] = 0.9 + 0.2 * diag2   # RR_thigh
    # Calf
    qpos[:, 2] = -1.8 + 0.15 * diag1
    qpos[:, 11] = -1.8 + 0.15 * diag1
    qpos[:, 5] = -1.8 + 0.15 * diag2
    qpos[:, 8] = -1.8 + 0.15 * diag2
    qpos += 0.005 * rng.standard_normal(qpos.shape).astype(np.float32)

    qvel = np.gradient(qpos, dt, axis=0).astype(np.float32)
    tau = 5.0 * rng.standard_normal((T, 12)).astype(np.float32)

    # Floating base: moving forward at 0.4 m/s, bobbing in z
    base_pose = np.zeros((T, 7), dtype=np.float32)
    base_pose[:, 0] = 0.4 * t                         # x
    base_pose[:, 2] = 0.30 + 0.01 * np.sin(phase)     # z
    # Identity quaternion wxyz
    base_pose[:, 3] = 1.0

    base_vel = np.zeros((T, 6), dtype=np.float32)
    base_vel[:, 0] = 0.4

    # Contact forces: diagonal pairs in contact alternately
    foot_force = np.zeros((T, 4), dtype=np.float32)
    c1 = (diag1 < 0).astype(np.float32)  # FR, RL in stance when diag1 < 0
    c2 = (diag2 < 0).astype(np.float32)
    foot_force[:, 0] = 30.0 * c1  # FR
    foot_force[:, 3] = 30.0 * c1  # RL
    foot_force[:, 1] = 30.0 * c2  # FL
    foot_force[:, 2] = 30.0 * c2  # RR
    foot_force += 0.5 * rng.standard_normal(foot_force.shape).astype(np.float32)
    foot_force = np.clip(foot_force, 0, None)

    contact_flags = foot_force > 5.0

    return dict(
        time=t,
        qpos=qpos,
        qvel=qvel,
        tau=tau,
        base_pose=base_pose,
        base_vel=base_vel,
        contact_foot_force=foot_force,
        contact_flags=contact_flags,
    )


def build_minimal_bundle(bundle_dir: str, schema_dir: str):
    """Create a minimal bundle with a stub Go1 directory and one synthetic trajectory."""
    os.makedirs(bundle_dir, exist_ok=True)
    # Copy schemas
    dst_schema = os.path.join(bundle_dir, "_schema")
    os.makedirs(dst_schema, exist_ok=True)
    for name in ("manifest.schema.json",
                 "robot_metadata.schema.json",
                 "trajectory.schema.json"):
        shutil.copy(os.path.join(schema_dir, name), os.path.join(dst_schema, name))

    # Go1 skeleton
    go1 = os.path.join(bundle_dir, "go1")
    os.makedirs(os.path.join(go1, "urdf", "meshes"), exist_ok=True)
    os.makedirs(os.path.join(go1, "trajectories"), exist_ok=True)

    # Stub URDF (minimum to satisfy file-exists check; real URDF comes in Stage 2)
    stub_urdf = """<?xml version="1.0"?>
<robot name="go1_stub">
  <link name="trunk"/>
</robot>
"""
    with open(os.path.join(go1, "urdf", "go1.urdf"), "w") as f:
        f.write(stub_urdf)
    # Stub mesh placeholder
    with open(os.path.join(go1, "urdf", "meshes", "trunk.stl"), "w") as f:
        f.write("# stub")

    # robot metadata
    metadata = {
        "schema_version": "0.1.0",
        "robot_name": "go1",
        "display_name": "Unitree Go1",
        "urdf_path": "urdf/go1.urdf",
        "mesh_dir": "urdf/meshes",
        "floating_base": True,
        "n_dof_actuated": 12,
        "n_dof_total": 18,
        "joint_names": GO1_JOINT_NAMES,
        "contact_names": GO1_CONTACT_NAMES,
        "urdf_source": {
            "origin": "mujoco_menagerie",
            "origin_url": "https://github.com/google-deepmind/mujoco_menagerie",
            "conversion_tool": "stub",
            "conversion_notes": "Placeholder for self-test only.",
            "license": "Apache-2.0",
        },
        "mass_kg": 12.0,
        "nominal_height_m": 0.30,
        "notes": "Self-test stub. Replace with real URDF in Stage 2.",
    }
    with open(os.path.join(go1, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Synthetic trajectory
    traj = synth_trajectory(T=500, freq=500.0, seed=42)
    traj_path = os.path.join(go1, "trajectories", "synth_trot_001.h5")
    write_trajectory(
        out_path=traj_path,
        robot_name="go1",
        trajectory_id="synth_trot_001",
        joint_names=GO1_JOINT_NAMES,
        contact_names=GO1_CONTACT_NAMES,
        contact_force_units="N",
        contact_force_scale="synthetic Newton values, not real sensor data",
        source="selftest_synthetic",
        source_type="sim_rollout",
        source_url="https://example.invalid/selftest",
        source_citation="Self-test synthetic data, not for publication",
        source_license="MIT",
        notes="Synthetic trot for schema self-test",
        **traj,
    )

    # Manifest
    manifest = {
        "schema_version": "0.1.0",
        "version": "0.1.0",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "robots": {
            "go1": {
                "metadata_path": "go1/metadata.json",
                "urdf_path": "go1/urdf/go1.urdf",
                "n_dof_actuated": 12,
                "floating_base": True,
                "trajectory_count": 1,
                "trajectories": [
                    {
                        "id": "synth_trot_001",
                        "path": "go1/trajectories/synth_trot_001.h5",
                        "duration_s": 499 / 500.0,
                        "frequency_hz": 500.0,
                        "n_timesteps": 500,
                        "source": "selftest_synthetic",
                        "source_type": "sim_rollout",
                        "source_license": "MIT",
                        "signals_present": [
                            "qpos", "qvel", "tau",
                            "base_pose", "base_vel",
                            "contact/foot_force", "contact/contact_flags",
                        ],
                        "task": "synthetic trot",
                        "split": "train",
                    }
                ],
            }
        },
        "sources": {
            "selftest_synthetic": {
                "citation": "Self-test synthetic data, not for publication",
                "url": "https://example.invalid/selftest",
                "license": "MIT",
            }
        },
    }
    with open(os.path.join(bundle_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def test_valid_bundle_passes():
    """Happy path: a correctly-built bundle should validate cleanly."""
    with tempfile.TemporaryDirectory() as tmp:
        bundle = os.path.join(tmp, "bundle")
        build_minimal_bundle(bundle, os.path.join(os.path.dirname(HERE), "_schema"))
        report = validate_bundle(bundle, verbose=False)
        if not report.ok:
            print("FAIL: valid bundle reported errors:")
            for e in report.errors:
                print(f"  - {e}")
            return False
        if report.files_checked != 1:
            print(f"FAIL: expected 1 file checked, got {report.files_checked}")
            return False
        print(f"  PASS: valid bundle (checked {report.files_checked} file, "
              f"{len(report.warnings)} warnings)")
        return True


def test_writer_rejects_bad_shapes():
    """Writer should refuse mismatched joint_names vs qpos columns."""
    import h5py  # noqa: F401 (ensure import works)
    with tempfile.TemporaryDirectory() as tmp:
        t = np.arange(100) / 100.0
        q = np.zeros((100, 12), dtype=np.float32)
        try:
            write_trajectory(
                out_path=os.path.join(tmp, "bad.h5"),
                robot_name="go1",
                trajectory_id="bad",
                joint_names=["only_one"],   # mismatch: 1 name, 12 columns
                time=t,
                qpos=q,
                source="test",
                source_type="sim_rollout",
                source_url="https://example.invalid",
                source_citation="test",
                source_license="MIT",
            )
        except ValueError as e:
            if "joint_names has 1 entries but qpos has 12" in str(e):
                print("  PASS: writer rejects joint_names/qpos mismatch")
                return True
            print(f"FAIL: wrong error message: {e}")
            return False
        print("FAIL: writer accepted invalid joint_names/qpos")
        return False


def test_writer_rejects_non_unit_quat():
    """Writer should reject non-unit-norm quaternions in base_pose."""
    with tempfile.TemporaryDirectory() as tmp:
        t = np.arange(100) / 100.0
        q = np.zeros((100, 12), dtype=np.float32)
        bp = np.zeros((100, 7), dtype=np.float32)
        bp[:, 3] = 2.0  # qw = 2, not unit norm
        try:
            write_trajectory(
                out_path=os.path.join(tmp, "bad.h5"),
                robot_name="go1",
                trajectory_id="bad_quat",
                joint_names=GO1_JOINT_NAMES,
                time=t,
                qpos=q,
                base_pose=bp,
                source="test",
                source_type="sim_rollout",
                source_url="https://example.invalid",
                source_citation="test",
                source_license="MIT",
            )
        except ValueError as e:
            if "unit norm" in str(e):
                print("  PASS: writer rejects non-unit quaternion")
                return True
            print(f"FAIL: wrong error: {e}")
            return False
        print("FAIL: writer accepted bad quaternion")
        return False


def test_validator_catches_mismatched_joint_names():
    """Validator should catch when a trajectory's joint_names drift from metadata."""
    import h5py
    with tempfile.TemporaryDirectory() as tmp:
        bundle = os.path.join(tmp, "bundle")
        build_minimal_bundle(bundle, os.path.join(os.path.dirname(HERE), "_schema"))
        # Corrupt: change joint_names in the trajectory attrs
        h5path = os.path.join(bundle, "go1", "trajectories", "synth_trot_001.h5")
        with h5py.File(h5path, "a") as f:
            wrong = list(GO1_JOINT_NAMES)
            wrong[0] = "WRONG_NAME"
            f.attrs["joint_names"] = wrong
        report = validate_bundle(bundle, verbose=False)
        if report.ok:
            print("FAIL: validator missed joint_names mismatch")
            return False
        if any("joint_names does not match" in e for e in report.errors):
            print("  PASS: validator catches joint_names mismatch")
            return True
        print("FAIL: validator reported errors, but not the expected one:")
        for e in report.errors:
            print(f"    - {e}")
        return False


def test_validator_catches_missing_trajectory_file():
    """Validator should error if a manifest entry points to a non-existent file."""
    with tempfile.TemporaryDirectory() as tmp:
        bundle = os.path.join(tmp, "bundle")
        build_minimal_bundle(bundle, os.path.join(os.path.dirname(HERE), "_schema"))
        # Delete the trajectory file but leave the manifest
        os.remove(os.path.join(bundle, "go1", "trajectories", "synth_trot_001.h5"))
        report = validate_bundle(bundle, verbose=False)
        if report.ok:
            print("FAIL: validator missed deleted trajectory file")
            return False
        if any("file not found" in e for e in report.errors):
            print("  PASS: validator catches missing trajectory file")
            return True
        print("FAIL: unexpected errors:")
        for e in report.errors:
            print(f"    - {e}")
        return False


def test_writer_requires_contact_force_units_when_force_present():
    """
    Regression: the writer must require contact_force_units whenever any
    foot_force signal is supplied. Catches the class of bug where the
    selftest fixture or a fetch script forgets to declare units.
    """
    import numpy as np
    sys.path.insert(0, HERE)
    from write_trajectory import write_trajectory

    with tempfile.TemporaryDirectory() as tmp:
        T = 20
        try:
            write_trajectory(
                out_path=os.path.join(tmp, "bad.h5"),
                robot_name="go1",
                trajectory_id="bad",
                joint_names=GO1_JOINT_NAMES,
                contact_names=GO1_CONTACT_NAMES,
                time=np.arange(T) / 50.0,
                qpos=np.zeros((T, 12), dtype=np.float32),
                contact_foot_force=np.zeros((T, 4), dtype=np.float32),
                # deliberately omit contact_force_units
                source="test",
                source_type="sim_rollout",
                source_url="https://example.invalid",
                source_citation="test",
                source_license="MIT",
            )
        except ValueError as e:
            if "contact_force_units must be provided" in str(e):
                print("  PASS: writer requires contact_force_units when foot_force present")
                return True
            print(f"FAIL: wrong error message: {e}")
            return False
        print("FAIL: writer accepted foot_force without contact_force_units")
        return False


def main():
    print("Running self-tests...\n")
    tests = [
        test_valid_bundle_passes,
        test_writer_rejects_bad_shapes,
        test_writer_rejects_non_unit_quat,
        test_validator_catches_mismatched_joint_names,
        test_validator_catches_missing_trajectory_file,
        test_writer_requires_contact_force_units_when_force_present,
    ]
    results = []
    for test in tests:
        print(f"- {test.__name__}")
        try:
            results.append(test())
        except Exception as e:
            import traceback
            print(f"  EXCEPTION: {e}")
            traceback.print_exc()
            results.append(False)
        print()

    passed = sum(results)
    total = len(results)
    print(f"=== {passed}/{total} tests passed ===")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
