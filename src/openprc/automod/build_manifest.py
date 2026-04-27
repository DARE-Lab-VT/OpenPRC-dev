"""
Assemble per-robot _manifest_fragment.json files into the top-level
manifest.json that the validator reads.

Each fetch_*.py script writes a fragment at <robot>/_manifest_fragment.json
containing just that robot's trajectory list. This tool walks the bundle,
merges all fragments with their robot metadata, and writes manifest.json.

Run this after every ingest (or add --auto to have fetch scripts call it
themselves — not wired yet, but easy to add later).

Usage:
    python _tools/build_manifest.py /path/to/robot_bundle
    python _tools/build_manifest.py .                    # current dir
    python _tools/build_manifest.py . --dry-run          # print without writing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

SCHEMA_VERSION = "0.1.0"

# Source citations for each known dataset. New sources get added here as we
# ingest them. The validator doesn't require sources to appear in manifest,
# but having them centralized means the paper's data table is a one-liner.
KNOWN_SOURCES = {
    "legkilo": {
        "citation": (
            "Ou, G., Li, D., Li, H. (2024). Leg-KILO: Robust Kinematic-"
            "Inertial-Lidar Odometry for Dynamic Legged Robots. "
            "IEEE RA-L 9(10):8194-8201."
        ),
        "url": "https://github.com/ouguangjun/legkilo-dataset",
        "license": "MIT",  # TODO verify from repo LICENSE before publication
    },
    "droid": {
        "citation": (
            "Khazatsky, A., et al. (2024). DROID: A Large-Scale In-the-Wild "
            "Robot Manipulation Dataset. RSS 2024."
        ),
        "url": "https://droid-dataset.github.io/",
        "license": "MIT",
    },
    "robopianist": {
        "citation": (
            "Zakka, K., et al. (2023). RoboPianist: Dexterous Piano Playing "
            "with Deep Reinforcement Learning. CoRL 2023."
        ),
        "url": "https://kzakka.com/robopianist/",
        "license": "Apache-2.0",
    },
    "rp1m": {
        "citation": (
            "Zhao, Y., et al. (2024). RP1M: A Large-Scale Motion Dataset "
            "for Piano Playing with Bi-Manual Dexterous Robot Hands."
        ),
        "url": "https://rp1m.github.io/",
        "license": "Apache-2.0",
    },
    "unitree_g1_hf": {
        "citation": "Unitree Robotics, G1 Task Datasets on Hugging Face.",
        "url": "https://huggingface.co/unitreerobotics",
        "license": "Apache-2.0",
    },
    "eth_anymal": {
        "citation": "ETH RSL, per-paper ANYmal releases. See per-trajectory source_citation.",
        "url": "https://github.com/leggedrobotics",
        "license": "varies",
    },
    "selftest_synthetic": {
        "citation": "Self-test synthetic data, not for publication.",
        "url": "https://example.invalid/selftest",
        "license": "MIT",
    },
}


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_manifest(bundle_dir: str) -> dict:
    """Scan the bundle and produce the manifest dict."""
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "version": SCHEMA_VERSION,  # track content version separately later if needed
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "robots": {},
        "sources": {},
    }

    # Discover robots: any top-level directory that has metadata.json
    candidates = sorted(
        d for d in os.listdir(bundle_dir)
        if os.path.isdir(os.path.join(bundle_dir, d))
        and not d.startswith("_")
        and os.path.exists(os.path.join(bundle_dir, d, "metadata.json"))
    )

    if not candidates:
        raise ValueError(
            f"No robot directories found in {bundle_dir}. "
            "A robot directory must contain metadata.json."
        )

    sources_used = set()

    for robot_name in candidates:
        robot_dir = os.path.join(bundle_dir, robot_name)
        metadata = _load_json(os.path.join(robot_dir, "metadata.json"))

        fragment_path = os.path.join(robot_dir, "_manifest_fragment.json")
        if os.path.exists(fragment_path):
            fragment = _load_json(fragment_path)
            trajectories = fragment.get("trajectories", [])
        else:
            trajectories = []
            print(f"  warning: no _manifest_fragment.json in {robot_name}, "
                  "robot will have 0 trajectories")

        # Ensure trajectories are sorted deterministically
        trajectories = sorted(trajectories, key=lambda e: e["id"])

        # Collect source names referenced by this robot
        for t in trajectories:
            if "source" in t:
                sources_used.add(t["source"])

        urdf_rel = os.path.join(robot_name, metadata["urdf_path"])
        manifest["robots"][robot_name] = {
            "metadata_path": f"{robot_name}/metadata.json",
            "urdf_path": urdf_rel,
            "n_dof_actuated": metadata["n_dof_actuated"],
            "floating_base": metadata["floating_base"],
            "trajectory_count": len(trajectories),
            "trajectories": trajectories,
        }

    # Populate sources section with just the ones actually used
    for src in sorted(sources_used):
        if src in KNOWN_SOURCES:
            manifest["sources"][src] = KNOWN_SOURCES[src]
        else:
            manifest["sources"][src] = {
                "citation": f"TODO: add citation for source '{src}' to build_manifest.py",
                "url": "TODO",
                "license": "TODO",
            }
            print(f"  warning: source '{src}' not in KNOWN_SOURCES; "
                  "added placeholder citation")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build top-level manifest.json")
    parser.add_argument("bundle_dir", help="Path to the bundle directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print manifest to stdout without writing")
    args = parser.parse_args()

    if not os.path.isdir(args.bundle_dir):
        print(f"error: {args.bundle_dir} is not a directory", file=sys.stderr)
        sys.exit(2)

    manifest = build_manifest(args.bundle_dir)

    # Summary
    print(f"Discovered {len(manifest['robots'])} robot(s):")
    for name, entry in manifest["robots"].items():
        print(f"  {name}: {entry['trajectory_count']} trajectories, "
              f"{entry['n_dof_actuated']} DoF, "
              f"floating_base={entry['floating_base']}")
    print(f"Sources referenced: {sorted(manifest['sources'].keys())}")

    if args.dry_run:
        print("\n=== manifest.json (dry-run) ===")
        print(json.dumps(manifest, indent=2))
        return

    out_path = os.path.join(args.bundle_dir, "manifest.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
