"""
Bundle validator.

Checks every file in the bundle against the schemas and enforces cross-file
consistency. Run this before any publication or release.

Usage:
    python validate_bundle.py /path/to/robot_bundle
    python validate_bundle.py .                          # current dir
    python validate_bundle.py . --robot go1              # just one robot
    python validate_bundle.py . --verbose                # list every file checked

Exit codes:
    0 — all checks passed
    1 — at least one check failed
    2 — validator error (bug in the validator itself, or unreadable bundle)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import h5py
import jsonschema
import numpy as np


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    files_checked: int = 0

    def error(self, msg: str):
        self.errors.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)

    @property
    def ok(self) -> bool:
        return not self.errors


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _major(version: str) -> int:
    return int(version.split(".")[0])


def validate_manifest(bundle_dir: str, report: ValidationReport) -> Optional[dict]:
    """Validate manifest.json against its schema. Returns the parsed manifest, or None on failure."""
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    schema_path = os.path.join(bundle_dir, "_schema", "manifest.schema.json")

    if not os.path.exists(manifest_path):
        report.error(f"manifest.json not found at {manifest_path}")
        return None

    if not os.path.exists(schema_path):
        report.error(f"_schema/manifest.schema.json not found")
        return None

    try:
        manifest = _load_json(manifest_path)
        schema = _load_json(schema_path)
    except json.JSONDecodeError as e:
        report.error(f"JSON parse error: {e}")
        return None

    try:
        jsonschema.validate(manifest, schema)
    except jsonschema.ValidationError as e:
        report.error(f"manifest.json: {e.message} (at {list(e.path)})")
        return None

    return manifest


def validate_robot_metadata(
    robot_dir: str, robot_name: str, bundle_dir: str, report: ValidationReport
) -> Optional[dict]:
    metadata_path = os.path.join(robot_dir, "metadata.json")
    schema_path = os.path.join(bundle_dir, "_schema", "robot_metadata.schema.json")

    if not os.path.exists(metadata_path):
        report.error(f"[{robot_name}] metadata.json not found")
        return None

    try:
        metadata = _load_json(metadata_path)
        schema = _load_json(schema_path)
    except json.JSONDecodeError as e:
        report.error(f"[{robot_name}] metadata JSON parse error: {e}")
        return None

    try:
        jsonschema.validate(metadata, schema)
    except jsonschema.ValidationError as e:
        report.error(
            f"[{robot_name}] metadata.json: {e.message} (at {list(e.path)})"
        )
        return None

    # Cross-check: robot_name in metadata matches directory
    if metadata["robot_name"] != robot_name:
        report.error(
            f"[{robot_name}] metadata robot_name is {metadata['robot_name']!r}, "
            f"directory name is {robot_name!r}"
        )

    # Cross-check: n_dof_total consistent with floating_base
    if metadata["floating_base"]:
        expected_total = metadata["n_dof_actuated"] + 6
        if metadata["n_dof_total"] != expected_total:
            report.error(
                f"[{robot_name}] floating_base=true but n_dof_total "
                f"({metadata['n_dof_total']}) != n_dof_actuated + 6 "
                f"({expected_total})"
            )
    else:
        if metadata["n_dof_total"] != metadata["n_dof_actuated"]:
            report.error(
                f"[{robot_name}] floating_base=false but n_dof_total "
                f"({metadata['n_dof_total']}) != n_dof_actuated "
                f"({metadata['n_dof_actuated']})"
            )

    # Check URDF file exists
    urdf_abspath = os.path.join(robot_dir, metadata["urdf_path"])
    if not os.path.exists(urdf_abspath):
        report.error(f"[{robot_name}] URDF not found at {metadata['urdf_path']}")

    # Check mesh dir exists
    mesh_abspath = os.path.join(robot_dir, metadata["mesh_dir"])
    if not os.path.isdir(mesh_abspath):
        report.error(f"[{robot_name}] mesh_dir not found at {metadata['mesh_dir']}")

    return metadata


def validate_trajectory(
    h5_path: str,
    robot_metadata: dict,
    manifest_entry: dict,
    bundle_dir: str,
    report: ValidationReport,
    verbose: bool = False,
):
    """Validate a single trajectory HDF5 file."""
    rel = os.path.relpath(h5_path, bundle_dir)
    schema_path = os.path.join(bundle_dir, "_schema", "trajectory.schema.json")
    schema = _load_json(schema_path)

    if verbose:
        print(f"  checking {rel}")

    try:
        with h5py.File(h5_path, "r") as f:
            # Convert attrs to plain dict of JSON-compatible values
            attrs = {}
            for k, v in f.attrs.items():
                if isinstance(v, np.ndarray):
                    attrs[k] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    attrs[k] = v.item()
                elif isinstance(v, bytes):
                    attrs[k] = v.decode("utf-8")
                else:
                    attrs[k] = v

            # Validate attrs against schema
            try:
                jsonschema.validate(attrs, schema)
            except jsonschema.ValidationError as e:
                report.error(
                    f"[{rel}] attrs schema: {e.message} (at {list(e.path)})"
                )
                return

            # Cross-check: robot_name matches
            if attrs["robot_name"] != robot_metadata["robot_name"]:
                report.error(
                    f"[{rel}] robot_name attr {attrs['robot_name']!r} "
                    f"!= metadata {robot_metadata['robot_name']!r}"
                )

            # Cross-check: joint_names matches robot metadata (order and length)
            if list(attrs["joint_names"]) != list(robot_metadata["joint_names"]):
                report.error(
                    f"[{rel}] joint_names does not match robot metadata"
                )

            # Cross-check: schema_version major compatibility with manifest
            manifest_version = manifest_entry.get("_bundle_schema_version", "0.1.0")
            if _major(attrs["schema_version"]) != _major(manifest_version):
                report.error(
                    f"[{rel}] schema_version {attrs['schema_version']} "
                    f"incompatible with bundle {manifest_version}"
                )

            # Cross-check: manifest entry matches file attrs
            if manifest_entry["id"] != attrs["trajectory_id"]:
                report.error(
                    f"[{rel}] manifest id {manifest_entry['id']!r} "
                    f"!= trajectory_id attr {attrs['trajectory_id']!r}"
                )
            if manifest_entry["n_timesteps"] != attrs["n_timesteps"]:
                report.error(
                    f"[{rel}] manifest n_timesteps ({manifest_entry['n_timesteps']}) "
                    f"!= attr ({attrs['n_timesteps']})"
                )
            if set(manifest_entry["signals_present"]) != set(attrs["signals_present"]):
                report.error(
                    f"[{rel}] manifest signals_present != attr signals_present"
                )

            # Required datasets exist
            if "qpos" not in f:
                report.error(f"[{rel}] /qpos missing")
                return
            if "time" not in f:
                report.error(f"[{rel}] /time missing")
                return

            # Dataset checks
            T = f["time"].shape[0]
            qpos = f["qpos"]

            if qpos.shape != (T, robot_metadata["n_dof_actuated"]):
                report.error(
                    f"[{rel}] qpos shape {qpos.shape} != "
                    f"(T={T}, n_dof={robot_metadata['n_dof_actuated']})"
                )

            # signals_present consistency with actual datasets
            def _dataset_exists(path):
                try:
                    return path in f
                except Exception:
                    return False

            for sig in attrs["signals_present"]:
                if not _dataset_exists(sig):
                    report.error(f"[{rel}] signals_present lists '{sig}' but dataset missing")

            # Find datasets not in signals_present
            actual_datasets = []
            def _collect(name, obj):
                if isinstance(obj, h5py.Dataset) and name not in ("time",):
                    actual_datasets.append(name)
            f.visititems(_collect)
            extras = set(actual_datasets) - set(attrs["signals_present"])
            if extras:
                report.error(
                    f"[{rel}] datasets present but not in signals_present: {sorted(extras)}"
                )

            # Floating-base must have base_pose
            if robot_metadata["floating_base"] and "base_pose" not in attrs["signals_present"]:
                report.warn(
                    f"[{rel}] floating_base robot but no base_pose signal"
                )

            # Quick sanity: no NaN in qpos
            qpos_data = qpos[:]
            if np.any(~np.isfinite(qpos_data)):
                report.error(f"[{rel}] qpos contains NaN/inf")

            # Time monotonic
            time_data = f["time"][:]
            if np.any(np.diff(time_data) < 0):
                report.error(f"[{rel}] /time is not monotonically non-decreasing")

            report.files_checked += 1

    except (OSError, KeyError) as e:
        report.error(f"[{rel}] could not open or read: {e}")


def validate_bundle(bundle_dir: str, robot_filter: Optional[str] = None, verbose: bool = False) -> ValidationReport:
    report = ValidationReport()

    manifest = validate_manifest(bundle_dir, report)
    if manifest is None:
        return report

    bundle_schema_version = manifest["schema_version"]

    for robot_name, robot_entry in manifest["robots"].items():
        if robot_filter and robot_name != robot_filter:
            continue

        if verbose:
            print(f"[{robot_name}]")

        robot_dir = os.path.join(bundle_dir, robot_name)
        if not os.path.isdir(robot_dir):
            report.error(f"[{robot_name}] directory not found")
            continue

        metadata = validate_robot_metadata(robot_dir, robot_name, bundle_dir, report)
        if metadata is None:
            continue

        # Cross-check manifest-declared urdf_path matches metadata
        if robot_entry["urdf_path"] != os.path.join(robot_name, metadata["urdf_path"]):
            report.error(
                f"[{robot_name}] manifest urdf_path {robot_entry['urdf_path']!r} "
                f"!= metadata urdf_path {metadata['urdf_path']!r}"
            )

        # Trajectory count consistency
        declared = robot_entry["trajectory_count"]
        actual = len(robot_entry["trajectories"])
        if declared != actual:
            report.error(
                f"[{robot_name}] trajectory_count {declared} != "
                f"len(trajectories) {actual}"
            )

        # Validate each trajectory file
        for traj_entry in robot_entry["trajectories"]:
            h5_rel = traj_entry["path"]
            h5_path = os.path.join(bundle_dir, h5_rel)
            if not os.path.exists(h5_path):
                report.error(f"[{robot_name}/{traj_entry['id']}] file not found: {h5_rel}")
                continue

            # Inject bundle schema version so the trajectory validator can cross-check
            entry = dict(traj_entry)
            entry["_bundle_schema_version"] = bundle_schema_version
            validate_trajectory(h5_path, metadata, entry, bundle_dir, report, verbose)

    return report


def main():
    parser = argparse.ArgumentParser(description="Validate a robot bundle.")
    parser.add_argument("bundle_dir", help="Path to the bundle directory")
    parser.add_argument("--robot", help="Validate only this robot", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.bundle_dir):
        print(f"error: {args.bundle_dir} is not a directory", file=sys.stderr)
        sys.exit(2)

    report = validate_bundle(args.bundle_dir, args.robot, args.verbose)

    print(f"\n=== Validation report ===")
    print(f"Files checked: {report.files_checked}")
    print(f"Errors:        {len(report.errors)}")
    print(f"Warnings:      {len(report.warnings)}")

    if report.warnings:
        print("\nWarnings:")
        for w in report.warnings:
            print(f"  - {w}")

    if report.errors:
        print("\nErrors:")
        for e in report.errors:
            print(f"  - {e}")
        sys.exit(1)

    print("\nAll checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
