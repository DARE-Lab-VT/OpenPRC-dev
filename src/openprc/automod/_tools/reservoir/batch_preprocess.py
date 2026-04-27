"""
Batch-generate reservoir .npz files for every link of a robot.

Walks the robot's URDF and creates one .npz per link with a visual mesh.
Preprocessing is deterministic (given the rng_seed), so this runs once and
the runtime pipeline just loads the cached .npz files.

Usage:
    python batch_preprocess.py --bundle-dir . --robot go1 --preset small
    python batch_preprocess.py --bundle-dir . --robot go1 --preset medium \\
        --node-density 2e6 --seed 42
    python batch_preprocess.py --bundle-dir . --robot go1 --force  # regenerate

Output lands at <bundle-dir>/<robot>/reservoir/<link_name>.npz.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mesh_to_reservoir import (  # noqa: E402
    ReservoirParams,
    PRESETS,
    mesh_to_reservoir,
    save_reservoir,
)


def extract_link_meshes_from_urdf(urdf_path: str) -> List[Tuple[str, str, np.ndarray]]:
    """
    Parse a URDF and return a list of (link_name, mesh_abspath, visual_origin_4x4)
    for every link that has at least one mesh visual.

    A link with multiple visuals gets multiple entries — they'll each produce
    a separate reservoir. This is rare for robot URDFs (usually one mesh
    per link) but we handle it for generality.
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))

    entries: List[Tuple[str, str, np.ndarray]] = []
    for link in root.findall("link"):
        link_name = link.get("name")
        if not link_name:
            continue
        for visual in link.findall("visual"):
            geom = visual.find("geometry")
            if geom is None:
                continue
            mesh_el = geom.find("mesh")
            if mesh_el is None:
                continue
            filename = mesh_el.get("filename")
            if not filename:
                continue

            # Resolve URDF-style path (strip package:// or file://, otherwise relative)
            if filename.startswith("package://"):
                tail = filename[len("package://"):]
                tail = tail.split("/", 1)[1] if "/" in tail else tail
                mesh_path = os.path.join(urdf_dir, tail)
            elif filename.startswith("file://"):
                mesh_path = filename[len("file://"):]
            elif os.path.isabs(filename):
                mesh_path = filename
            else:
                mesh_path = os.path.join(urdf_dir, filename)
            mesh_path = os.path.normpath(mesh_path)

            # Visual origin (xyz + rpy)
            origin_el = visual.find("origin")
            origin = np.eye(4, dtype=np.float64)
            if origin_el is not None:
                xyz = origin_el.get("xyz", "0 0 0").split()
                rpy = origin_el.get("rpy", "0 0 0").split()
                origin[:3, 3] = [float(c) for c in xyz]
                roll, pitch, yaw = (float(c) for c in rpy)
                Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)],
                               [0, np.sin(roll), np.cos(roll)]])
                Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0],
                               [-np.sin(pitch), 0, np.cos(pitch)]])
                Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                               [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
                origin[:3, :3] = Rz @ Ry @ Rx

            entries.append((link_name, mesh_path, origin))

    return entries


def batch_preprocess(
    bundle_dir: str,
    robot_name: str,
    params: ReservoirParams,
    force: bool = False,
) -> Dict[str, dict]:
    """
    Generate reservoir .npz for every link with a visual mesh.
    Returns a dict {link_name: summary} for manifest-style reporting.
    """
    robot_dir = os.path.join(bundle_dir, robot_name)
    metadata_path = os.path.join(robot_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"{metadata_path} not found")
    with open(metadata_path) as f:
        robot_meta = json.load(f)

    urdf_path = os.path.join(robot_dir, robot_meta["urdf_path"])
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    reservoir_dir = os.path.join(robot_dir, "reservoir")
    os.makedirs(reservoir_dir, exist_ok=True)

    entries = extract_link_meshes_from_urdf(urdf_path)
    if not entries:
        raise RuntimeError(f"No mesh visuals found in {urdf_path}")

    print(f"Processing {len(entries)} link visuals for robot '{robot_name}'")
    print(f"Output: {reservoir_dir}")
    print(f"Params: {asdict(params)}")
    print()

    # If a link has multiple visuals (rare), disambiguate by appending an index
    seen_names: Dict[str, int] = {}
    summary: Dict[str, dict] = {}
    n_generated = 0
    n_cached = 0
    n_failed = 0

    for link_name, mesh_path, origin in entries:
        # Disambiguate duplicate link-visual combos
        count = seen_names.get(link_name, 0)
        seen_names[link_name] = count + 1
        out_link_name = link_name if count == 0 else f"{link_name}__{count}"
        out_path = os.path.join(reservoir_dir, f"{out_link_name}.npz")

        if os.path.exists(out_path) and not force:
            print(f"[cache] {out_link_name}: {out_path} exists (use --force to regenerate)")
            n_cached += 1
            continue

        if not os.path.exists(mesh_path):
            print(f"[skip]  {out_link_name}: mesh not found at {mesh_path}")
            n_failed += 1
            continue

        try:
            reservoir = mesh_to_reservoir(
                mesh_path,
                params,
                link_name=out_link_name,
                mesh_origin_in_link=origin,
                verbose=True,
            )
            save_reservoir(reservoir, out_path)
            n_generated += 1
            summary[out_link_name] = {
                "path": out_path,
                "n_nodes": int(len(reservoir["node_positions"])),
                "n_anchors": int(reservoir["n_anchors"]),
                "n_springs": int(len(reservoir["edges"])),
            }
            print(f"[ok]    {out_link_name} → {out_path}")
        except Exception as e:
            n_failed += 1
            print(f"[fail]  {out_link_name}: {e}")
        print()

    print(f"\nSummary: {n_generated} generated, {n_cached} cached, {n_failed} failed")

    # Write an index file so the runtime knows what reservoirs exist and where
    index = {
        "robot_name": robot_name,
        "reservoir_dir": "reservoir",
        "params": asdict(params),
        "links": summary,
    }
    index_path = os.path.join(reservoir_dir, "_index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote index: {index_path}")

    return summary


def _build_params_from_args(args) -> ReservoirParams:
    base = PRESETS[args.preset]
    fields = asdict(base)
    for name in fields:
        cli_value = getattr(args, name, None)
        if cli_value is not None:
            fields[name] = cli_value
    if args.seed is not None:
        fields["rng_seed"] = args.seed
    return ReservoirParams(**fields)


def main():
    p = argparse.ArgumentParser(description="Batch-generate link reservoirs for a robot")
    p.add_argument("--bundle-dir", required=True)
    p.add_argument("--robot", required=True, help="Robot name, e.g. go1")
    p.add_argument("--preset", choices=list(PRESETS.keys()), default="small")
    p.add_argument("--force", action="store_true",
                   help="Regenerate .npz files even if cached")
    p.add_argument("--seed", type=int, default=None)
    for field_name, default in asdict(ReservoirParams()).items():
        if field_name == "rng_seed":
            continue
        p.add_argument(
            f"--{field_name.replace('_', '-')}",
            dest=field_name,
            type=type(default),
            default=None,
            help=f"(override preset default: {default})",
        )
    args = p.parse_args()

    params = _build_params_from_args(args)
    batch_preprocess(args.bundle_dir, args.robot, params, force=args.force)


if __name__ == "__main__":
    main()
