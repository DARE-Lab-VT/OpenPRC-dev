"""
ANYmal C bundle ingestion — STUB.

Unlike Go1/G1/Panda, there is no single canonical ANYmal C dataset analogous
to legkilo or DROID. ETH RSL releases rollout data alongside individual papers
but does not maintain a consolidated, versioned benchmark dataset.

This file documents the current state of the search and provides a skeleton
for integrating data once a specific source is chosen. Don't run it as-is —
it will raise NotImplementedError until a source is wired in.

CANDIDATE SOURCES (ranked by reputability × availability):

1. Published paper GitHub releases. ETH RSL papers in recent years (2023-2026)
   often include rollout data for reproducibility. Example searches:
     - github.com/leggedrobotics → browse recent ANYmal papers
     - legged_gym repository (training) sometimes ships trained policy rollouts
     - Look for .npz or .pkl files in paper-associated repos with joint_states
   Challenge: each paper's data format differs; requires per-paper parsing.

2. LocoMuJoCo ANYmal motions (if still maintained). Al-Hafez et al. NeurIPS 2023
   D&B track. Check the current state of the repository — sim rollouts, not
   real hardware. Acceptable as a fallback but weaker story than real.

3. ANYmal-specific datasets on HuggingFace. Periodic searches of the
   unitreerobotics-style community datasets may surface ANYmal logs; none
   found as of April 2026.

4. Direct request to ETH RSL (Dr. Marco Hutter's group). For a NeurIPS
   submission this is legitimate — academic groups often share data under
   a simple attribution license. Not scalable but works for this project.

RECOMMENDED NEXT STEP:

Instead of blocking on ANYmal, proceed with the 4 confirmed robots
(Go1, Panda, G1 body, G1 Dex3) and document ANYmal as "future work" or
"under preparation" in the paper until a source is secured. A 4-robot
demonstration with real hardware across morphologies is stronger than a
5-robot demonstration where one has sim data or scattered trajectories.

If you want to proceed with a specific source, add a new function
`ingest_<source_name>` below and wire it into the CLI.

Usage:
    python fetch_anymal.py --help
    # Once a source is chosen:
    # python fetch_anymal.py --bundle-dir . ingest-github --repo-url ... --data-path ...
"""

from __future__ import annotations

import argparse
import sys


ANYMAL_C_URDF_JOINT_NAMES = [
    # Left front
    "LF_HAA", "LF_HFE", "LF_KFE",
    # Right front
    "RF_HAA", "RF_HFE", "RF_KFE",
    # Left hind
    "LH_HAA", "LH_HFE", "LH_KFE",
    # Right hind
    "RH_HAA", "RH_HFE", "RH_KFE",
]
ANYMAL_C_CONTACT_NAMES = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]


def not_yet_implemented(*args, **kwargs):
    raise NotImplementedError(
        "ANYmal C ingestion is not yet implemented. See the module docstring "
        "for candidate data sources. When you have one chosen:\n"
        "  1. Download the data\n"
        "  2. Implement a parser that produces {qpos, qvel?, tau?, base_pose?, ...}\n"
        "  3. Call write_trajectory() with source_type='real_hardware' or "
        "'sim_rollout' as appropriate\n"
        "  4. Append a manifest fragment entry.\n"
        "Look at fetch_go1.py as a template."
    )


def main():
    parser = argparse.ArgumentParser(
        description="ANYmal C bundle ingestion (stub — see module docstring)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--bundle-dir", required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("convert-urdf")
    sub.add_parser("ingest")

    args = parser.parse_args()
    not_yet_implemented()


if __name__ == "__main__":
    main()
